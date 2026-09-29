"""
Save the trained fold models of every architecture as self-contained bundles for later prediction,
optionally exported to ONNX and compiled to TensorRT engines.
Usage: python export_models.py --runs-dir ./runs --out-dir ./models \
           --formats fp32.onnx fp16.onnx fp32.engine fp16.engine

Per architecture this writes <out-dir>/<arch>/manifest.json and one folder per fold run (e.g. fold0_seed0/) holding
model.pt and the requested exports (model.fp32.onnx, model.fp16.engine, ...).
predict.py --bundle-dir <out-dir>/<arch> averages the probabilities of all members (a K-fold ensemble).

TensorRT 11 builds strongly typed engines: the precision comes from the ONNX file, so an fp16 engine is built
from the fp16 ONNX export. Engines only run with the TensorRT version and GPU model they were built with.
"""

import argparse
import copy
import glob
import importlib.util
import json
import os
import shutil
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from dataset import MRISegmentationDataset as SegDataset
from inference import BUNDLE_MANIFEST, create_backend, load_trained_model
from network import ModelFactory

EXPORT_FORMATS: List[str] = ["fp32.onnx", "fp16.onnx", "fp32.engine", "fp16.engine"]
CHECKPOINT_NAME = "model.pt"
ONNX_OPSET = 20  # TensorRT 11.3 parses opsets 9-24
INPUT_NAME = "image"
OUTPUT_NAME = "probability"


class ModelExporter:
    """Exports one trained model to ONNX and builds TensorRT engines from ONNX files."""

    @staticmethod
    def export_onnx(model: nn.Module, onnx_path: str, image_size: int, precision: str) -> None:
        """
        Export with the torch.export-based ONNX exporter and a dynamic batch dimension.

        :param model: Trained model (not modified; an fp16 copy is exported for precision 'fp16')
        :param onnx_path: Output .onnx file (weights embedded, single file)
        :param image_size: Fixed spatial size H = W of the exported graph
        :param precision: 'fp32' or 'fp16' (input, weights and output dtype)
        """
        dtype = torch.float16 if precision == "fp16" else torch.float32
        export_model = copy.deepcopy(model).eval().to(dtype)
        device = next(export_model.parameters()).device

        # Batch 2, not 1: torch.export specializes dimensions of size 1 to constants
        example = torch.randn(2, SegDataset.num_input_channels, image_size, image_size, dtype=dtype, device=device)
        batch = torch.export.Dim("batch", min=1)

        torch.onnx.export(export_model,
                          (example,),
                          onnx_path,
                          dynamo=True,
                          opset_version=ONNX_OPSET,
                          input_names=[INPUT_NAME],
                          output_names=[OUTPUT_NAME],
                          dynamic_shapes=({0: batch},),
                          external_data=False)

    @staticmethod
    def build_tensorrt_engine(onnx_path: str,
                              engine_path: str,
                              image_size: int,
                              max_batch: int,
                              opt_batch: int,
                              allow_tf32: bool,
                              workspace_gib: float) -> None:
        """
        Build and serialize a TensorRT engine from an ONNX file (TensorRT 11 API).

        :param onnx_path: ONNX model; its tensor dtypes decide the engine precision
        :param engine_path: Output engine file
        :param image_size: Spatial size H = W the ONNX model was exported with
        :param max_batch: Largest batch the engine accepts (predict.py splits larger batches)
        :param opt_batch: Batch size TensorRT tunes kernels for
        :param allow_tf32: Let fp32 convolutions / matmuls use TF32 tensor cores (TensorRT's default)
        :param workspace_gib: Scratch memory limit for kernel selection
        """
        import tensorrt as trt  # optional dependency: uv sync --group tensorrt

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 2**30))
        if not allow_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)

        channels = SegDataset.num_input_channels
        profile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name,
                          (1, channels, image_size, image_size),
                          (opt_batch, channels, image_size, image_size),
                          (max_batch, channels, image_size, image_size))
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f"TensorRT engine build failed for {onnx_path} (see the TensorRT log above)")
        with open(engine_path, "wb") as fp:
            fp.write(serialized_engine)

    @staticmethod
    def probability_difference(reference: nn.Module, model_path: str, image_size: int, device: torch.device) -> float:
        """
        Largest absolute probability difference between an exported model and the fp32 PyTorch model
        on a random batch. A quick sanity check of the export, not an accuracy measurement.

        :param reference: The fp32 PyTorch model that was exported
        :param model_path: Exported .onnx or .engine file
        :param image_size: Spatial size the model was exported with
        :param device: Device to run both models on
        :return: max |p_exported - p_pytorch| over all pixels
        """
        images = torch.randn(4, SegDataset.num_input_channels, image_size, image_size, device=device)
        with torch.no_grad():
            expected = reference(images).float()
        exported = create_backend(model_path, arch="", device=device)
        return float((exported.predict(images) - expected).abs().max())


class BundleExporter:
    """Collects the fold checkpoints of one architecture into a bundle and exports them."""

    def __init__(self, cfg: argparse.Namespace, device: torch.device) -> None:
        """
        :param cfg: Parsed command-line arguments
        :param device: Device used for export (TensorRT engines need CUDA)
        """
        self.cfg = cfg
        self.device = device
        self.opt_batch: int = cfg.opt_batch or cfg.max_batch
        self.formats = self._formats_to_write(cfg.formats)

    def run_dirs(self, arch: str) -> List[str]:
        """Finished runs of one architecture: <runs-dir>/<arch>/fold*_seed*/ with a best checkpoint."""
        pattern = os.path.join(self.cfg.runs_dir, arch, "fold*_seed*", "checkpoints", "best_model.pt")
        checkpoints = sorted(glob.glob(pattern))
        return [os.path.dirname(os.path.dirname(path)) for path in checkpoints]

    def export(self, arch: str) -> None:
        run_dirs = self.run_dirs(arch)
        if not run_dirs:
            print(f"[Export] skip {arch}: no checkpoints under {os.path.join(self.cfg.runs_dir, arch)}")
            return

        image_size = self._common_image_size(run_dirs)
        bundle_dir = os.path.join(self.cfg.out_dir, arch)
        if os.path.exists(bundle_dir):
            if not self.cfg.overwrite:
                raise SystemExit(f"{bundle_dir} exists; pass --overwrite to replace it")
            shutil.rmtree(bundle_dir)

        members = [self._export_member(arch, run_dir, bundle_dir, image_size) for run_dir in run_dirs]

        manifest = {"arch": arch,
                    "image_size": image_size,
                    "in_channels": SegDataset.num_input_channels,
                    "out_channels": SegDataset.num_output_channels,
                    "files": [CHECKPOINT_NAME] + [f"model.{fmt}" for fmt in self.formats],
                    "members": members,
                    "exported_with": self._versions()}
        with open(os.path.join(bundle_dir, BUNDLE_MANIFEST), "w") as fp:
            json.dump(manifest, fp, indent=2)
        print(f"[Export] {arch}: {len(members)} members -> {bundle_dir}")

    def _export_member(self, arch: str, run_dir: str, bundle_dir: str, image_size: int) -> Dict[str, Any]:
        """Copy one run's checkpoint into the bundle and write its exports. Returns its manifest entry."""
        name = os.path.basename(run_dir)
        member_dir = os.path.join(bundle_dir, name)
        os.makedirs(member_dir)

        model = load_trained_model(arch, os.path.join(run_dir, "checkpoints", "best_model.pt"), self.device)
        torch.save(model.state_dict(), os.path.join(member_dir, CHECKPOINT_NAME))

        for fmt in self.formats:
            path = os.path.join(member_dir, f"model.{fmt}")
            self._write_format(model, fmt, path, image_size)
            if self._can_run(fmt):
                difference = ModelExporter.probability_difference(model, path, image_size, self.device)
                print(f"[Export] {arch}/{name}/model.{fmt}: max |probability difference| vs PyTorch fp32 "
                      f"on a random batch = {difference:.2e}")

        return {"name": name, **self._run_info(run_dir)}

    def _write_format(self, model: nn.Module, fmt: str, path: str, image_size: int) -> None:
        """Write model.<fmt>; an engine is built from the ONNX file of the same precision next to it."""
        precision, kind = fmt.split(".")
        if kind == "onnx":
            ModelExporter.export_onnx(model, path, image_size, precision)
            return

        onnx_path = os.path.join(os.path.dirname(path), f"model.{precision}.onnx")
        ModelExporter.build_tensorrt_engine(onnx_path,
                                            path,
                                            image_size,
                                            max_batch=self.cfg.max_batch,
                                            opt_batch=self.opt_batch,
                                            allow_tf32=not self.cfg.no_tf32,
                                            workspace_gib=self.cfg.workspace_gib)

    @staticmethod
    def _formats_to_write(requested: List[str]) -> List[str]:
        """Requested formats plus the ONNX files the engines are built from, ONNX files first."""
        needed = set(requested)
        for fmt in requested:
            if fmt.endswith(".engine"):
                needed.add(fmt.replace(".engine", ".onnx"))
        return [fmt for fmt in EXPORT_FORMATS if fmt in needed]

    @staticmethod
    def _can_run(fmt: str) -> bool:
        """Whether the exported file can be executed here for the sanity check (engines are built on CUDA)."""
        if fmt.endswith(".engine"):
            return True
        return importlib.util.find_spec("onnxruntime") is not None

    def _common_image_size(self, run_dirs: List[str]) -> int:
        """Image size the runs were trained with (logs/config.json); all members of a bundle must agree."""
        sizes = set()
        for run_dir in run_dirs:
            config = self._read_json(os.path.join(run_dir, "logs", "config.json"))
            sizes.add(config["image_size"] if config else self.cfg.image_size)
        if len(sizes) != 1:
            raise SystemExit(f"Runs were trained with different image sizes {sorted(sizes)}; bundle them separately")
        return int(sizes.pop())

    def _run_info(self, run_dir: str) -> Dict[str, Any]:
        """Provenance of one member: its fold, seed, validation score and held-out test score."""
        summary = self._read_json(os.path.join(run_dir, "logs", "train_summary.json")) or {}
        test_results = self._read_json(os.path.join(run_dir, "test_results.json"))

        test_dice_mean: Optional[float] = None
        if test_results:
            dice_values = [values["dice"] for values in test_results["per_patient"].values()]
            test_dice_mean = sum(dice_values) / len(dice_values)

        return {"fold": summary.get("fold"),
                "seed": summary.get("seed"),
                "best_val_dice": summary.get("best_val_dice"),
                "best_epoch": summary.get("best_epoch"),
                "test_dice_mean": test_dice_mean,
                "source_run": os.path.abspath(run_dir)}

    def _versions(self) -> Dict[str, Optional[str]]:
        versions: Dict[str, Optional[str]] = {"torch": torch.__version__, "tensorrt": None, "gpu": None}
        if any(fmt.endswith(".engine") for fmt in self.formats):
            import tensorrt as trt

            versions["tensorrt"] = trt.__version__
            versions["gpu"] = torch.cuda.get_device_name(self.device)
        return versions

    @staticmethod
    def _read_json(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path) as fp:
            content: Dict[str, Any] = json.load(fp)
        return content


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle and export trained models per architecture")
    parser.add_argument("--runs-dir", type=str, default="./runs", help="Output of run_cv.sh (default: ./runs)")
    parser.add_argument("--out-dir", type=str, default="./models", help="Bundle root folder (default: ./models)")
    parser.add_argument("--archs", nargs="+", default=ModelFactory.available(), choices=ModelFactory.available(),
                        help="Architectures to export (default: all registered)")
    parser.add_argument("--formats", nargs="*", default=[], choices=EXPORT_FORMATS,
                        help="Exports besides model.pt; .engine formats also write the matching .onnx (default: none)")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Used only for runs without logs/config.json (default: 256)")
    parser.add_argument("--max-batch", type=int, default=32, help="Largest TensorRT batch (default: 32)")
    parser.add_argument("--opt-batch", type=int, default=None, help="Batch TensorRT tunes for (default: --max-batch)")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 in fp32 TensorRT engines")
    parser.add_argument("--workspace-gib", type=float, default=4.0, help="TensorRT builder workspace (default: 4)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Export device (default: cuda:0)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing bundles in --out-dir")
    return parser.parse_args()


def main() -> None:
    cfg = _parse_args()
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)
    if device.type != "cuda" and any(fmt.endswith(".engine") for fmt in cfg.formats):
        raise SystemExit("TensorRT engines can only be built on a CUDA device")
    if cfg.opt_batch is not None and not 1 <= cfg.opt_batch <= cfg.max_batch:
        raise SystemExit(f"--opt-batch must be between 1 and --max-batch ({cfg.max_batch}), got {cfg.opt_batch}")

    exporter = BundleExporter(cfg, device)
    for arch in cfg.archs:
        exporter.export(arch)


if __name__ == "__main__":
    main()
