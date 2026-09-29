"""
Inference backends behind one interface: PyTorch (eager or torch.compile), ONNX Runtime and TensorRT,
plus an ensemble that averages several backends (the per-architecture model bundles from export_models.py).

Every backend maps a float32 (B, C, H, W) image batch to float32 (B, 1, H, W) probabilities on the batch's device.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from dataset import MRISegmentationDataset as SegDataset
from evaluation import PatientEvaluator
from network import ModelFactory

COMPILE_MODES: List[str] = ["none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
BUNDLE_MANIFEST = "manifest.json"


def compile_model(model: nn.Module, mode: str) -> nn.Module:
    """
    Wrap a model with torch.compile. The compiled module shares its parameters with `model`.

    :param model: Model to compile
    :param mode: One of COMPILE_MODES; 'none' returns the model unchanged
    :return: The compiled module, or `model` itself for mode 'none'
    """
    if mode not in COMPILE_MODES:
        raise ValueError(f"Unknown compile mode '{mode}'. Choose from {COMPILE_MODES}")
    if mode == "none":
        return model
    return cast(nn.Module, torch.compile(model, mode=mode))


def load_trained_model(arch: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Build the architecture and load a checkpoint saved by train.py (or a bundle's model.pt).

    :param arch: Architecture name registered in ModelFactory
    :param checkpoint_path: Path to a state_dict .pt file
    :param device: Device to load the model onto
    :return: The model in eval mode on `device`
    """
    model = ModelFactory.create(arch,
                                in_channels=SegDataset.num_input_channels,
                                out_channels=SegDataset.num_output_channels)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(PatientEvaluator.strip_compile_prefix(state))
    return model.to(device).eval()


class InferenceBackend(ABC):
    """Maps a (B, C, H, W) image batch to float32 (B, 1, H, W) probabilities on the batch's device."""

    @abstractmethod
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: Preprocessed image batch, shape (B, C, H, W)
        :return: Probabilities, shape (B, 1, H, W), float32, on images.device
        """


class PyTorchBackend(InferenceBackend):
    """A PyTorch model run eagerly or through torch.compile, optionally under fp16 autocast (CUDA only)."""

    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 compile_mode: str = "none",
                 use_amp: bool = False) -> None:
        """
        :param model: Trained model
        :param device: Device to run on
        :param compile_mode: One of COMPILE_MODES
        :param use_amp: Run under fp16 autocast; ignored on CPU
        """
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.model = compile_model(model.to(device).eval(), compile_mode)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            probabilities = self.model(images.to(self.device))
        return probabilities.float().to(images.device)


class OnnxRuntimeBackend(InferenceBackend):
    """An exported .onnx model run with ONNX Runtime (CUDA execution provider when available, else CPU)."""

    def __init__(self, onnx_path: str, device: torch.device) -> None:
        """
        :param onnx_path: Model exported by export_models.py (fp32 or fp16 input)
        :param device: Requested device; falls back to CPU if onnxruntime-gpu is not installed
        """
        import onnxruntime as ort  # optional dependency: uv sync --group onnx

        providers: List[Any] = ["CPUExecutionProvider"]
        if device.type == "cuda":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, ("CUDAExecutionProvider", {"device_id": device.index or 0}))
            else:
                print("[ONNX Runtime] CUDAExecutionProvider not available (needs onnxruntime-gpu); running on CPU")

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name: str = model_input.name
        self.input_dtype: npt.DTypeLike = np.float16 if model_input.type == "tensor(float16)" else np.float32

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.detach().cpu().numpy().astype(self.input_dtype)
        probabilities = self.session.run(None, {self.input_name: batch})[0]
        return torch.from_numpy(probabilities.astype(np.float32)).to(images.device)


class TensorRTBackend(InferenceBackend):
    """
    A serialized TensorRT engine (.engine / .plan) built by export_models.py.

    Engines only run with the TensorRT version and GPU model they were built with. Input and output buffers are
    PyTorch tensors; the engine runs on its own CUDA stream, ordered after and before the caller's current stream.
    """

    def __init__(self, engine_path: str, device: torch.device) -> None:
        """
        :param engine_path: Serialized engine file
        :param device: CUDA device to run on
        """
        import tensorrt as trt  # optional dependency: uv sync --group tensorrt

        if device.type != "cuda":
            raise ValueError(f"TensorRT engines run on CUDA devices only, got {device}")
        self.device = device

        dtype_map = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        with open(engine_path, "rb") as fp:
            engine_bytes = fp.read()

        with torch.cuda.device(device):
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
            if self.engine is None:
                raise RuntimeError(f"Could not load {engine_path}: engines only load with the TensorRT version "
                                   f"and GPU model they were built with; rebuild it with export_models.py")
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.Stream(device=device)

        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        input_names = [name for name in tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        output_names = [name for name in tensor_names if name not in input_names]
        if len(input_names) != 1 or len(output_names) != 1:
            raise ValueError(f"Expected one input and one output tensor, got {input_names} and {output_names}")

        self.input_name = input_names[0]
        self.output_name = output_names[0]
        self.input_dtype = dtype_map[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtype = dtype_map[self.engine.get_tensor_dtype(self.output_name)]
        _, _, max_shape = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self.max_batch = int(max_shape[0])

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        chunks = [self._run(chunk) for chunk in images.split(self.max_batch)]
        return torch.cat(chunks).to(images.device)

    def _run(self, images: torch.Tensor) -> torch.Tensor:
        """Run one batch that fits the engine's optimization profile."""
        with torch.cuda.device(self.device):
            batch = images.to(self.device, self.input_dtype).contiguous()
            if not self.context.set_input_shape(self.input_name, tuple(batch.shape)):
                raise ValueError(f"Input shape {tuple(batch.shape)} is outside the engine's optimization profile")

            output_shape = tuple(self.context.get_tensor_shape(self.output_name))
            output = torch.empty(output_shape, dtype=self.output_dtype, device=self.device)
            self.context.set_tensor_address(self.input_name, batch.data_ptr())
            self.context.set_tensor_address(self.output_name, output.data_ptr())

            caller_stream = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(caller_stream)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed (see the TensorRT log above)")
            caller_stream.wait_stream(self.stream)

        return output.float()


class EnsembleBackend(InferenceBackend):
    """Averages the probabilities of several backends (e.g. the K fold models of one architecture)."""

    def __init__(self, members: Sequence[InferenceBackend]) -> None:
        """
        :param members: At least one backend
        """
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = list(members)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        total = self.members[0].predict(images)
        for member in self.members[1:]:
            total = total + member.predict(images)
        return total / len(self.members)


def create_backend(model_path: str,
                   arch: str,
                   device: torch.device,
                   compile_mode: str = "none",
                   use_amp: bool = False) -> InferenceBackend:
    """
    Pick the backend from the file type: .pt checkpoint -> PyTorch, .onnx -> ONNX Runtime, .engine/.plan -> TensorRT.

    :param model_path: Model file
    :param arch: Architecture name (only used to rebuild the network for .pt checkpoints)
    :param device: Device to run on
    :param compile_mode: torch.compile mode; only valid for .pt checkpoints
    :param use_amp: fp16 autocast; only valid for .pt checkpoints (ONNX / TensorRT precision is fixed at export)
    :return: Backend ready for predict()
    """
    extension = os.path.splitext(model_path)[1].lower()
    if extension == ".pt":
        return PyTorchBackend(load_trained_model(arch, model_path, device), device, compile_mode, use_amp)

    if compile_mode != "none" or use_amp:
        raise ValueError("--compile and --amp only apply to .pt checkpoints; the precision of ONNX and "
                         "TensorRT models is fixed when they are exported")
    if extension == ".onnx":
        return OnnxRuntimeBackend(model_path, device)
    if extension in (".engine", ".plan"):
        return TensorRTBackend(model_path, device)
    raise ValueError(f"Unknown model file type '{extension}' ({model_path}); expected .pt, .onnx, .engine or .plan")


def read_bundle_manifest(bundle_dir: str) -> Dict[str, Any]:
    """
    :param bundle_dir: One architecture's bundle folder written by export_models.py
    :return: The parsed manifest.json
    """
    with open(os.path.join(bundle_dir, BUNDLE_MANIFEST)) as fp:
        manifest: Dict[str, Any] = json.load(fp)
    return manifest


def load_bundle(bundle_dir: str,
                file_name: str,
                device: torch.device,
                compile_mode: str = "none",
                use_amp: bool = False) -> Tuple[EnsembleBackend, Dict[str, Any]]:
    """
    Load every member of an architecture bundle as one ensemble.

    :param bundle_dir: One architecture's bundle folder written by export_models.py
    :param file_name: Model file inside each member folder, e.g. 'model.pt' or 'model.fp16.engine'
    :param device: Device to run on
    :param compile_mode: torch.compile mode for 'model.pt'
    :param use_amp: fp16 autocast for 'model.pt'
    :return: (ensemble backend, manifest)
    """
    manifest = read_bundle_manifest(bundle_dir)
    members: List[InferenceBackend] = []
    for member in manifest["members"]:
        path = os.path.join(bundle_dir, member["name"], file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist; export it with export_models.py --formats")
        members.append(create_backend(path, manifest["arch"], device, compile_mode, use_amp))
    return EnsembleBackend(members), manifest
