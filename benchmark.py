"""
Compute benchmark: parameters, FLOPs, inference latency/throughput and peak inference memory, per backend.
Usage: python benchmark.py --archs unet resunet swinunetr --backends pytorch fp16.engine \
           --out ./runs/compute_benchmark.json
Run every architecture on the SAME GPU type in ONE invocation so numbers are comparable.

Backends: 'pytorch' (eager, or torch.compile with --compile; fp16 autocast unless --no-amp) and the export formats of
export_models.py ('fp32.onnx', 'fp16.onnx', 'fp32.engine', 'fp16.engine'). Exports are made from untrained weights
into --export-dir; latency does not depend on the weight values.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from dataset import MRISegmentationDataset as SegDataset
from export_models import EXPORT_FORMATS, ModelExporter
from inference import COMPILE_MODES, InferenceBackend, PyTorchBackend, create_backend
from network import ModelFactory

BACKENDS: List[str] = ["pytorch"] + EXPORT_FORMATS


class ModelBenchmark:
    """Measures cost metrics of one model under fixed settings (eval mode, no_grad)."""

    def __init__(self,
                 device: torch.device,
                 image_size: int,
                 warmup_iters: int,
                 timed_iters: int,
                 use_amp: bool,
                 compile_mode: str,
                 export_dir: str,
                 allow_tf32: bool) -> None:
        """
        :param device: Device to benchmark on
        :param image_size: Spatial size H = W of the random input
        :param warmup_iters: Untimed calls before timing (also triggers torch.compile / engine warm-up)
        :param timed_iters: Timed calls per batch size
        :param use_amp: fp16 autocast for the 'pytorch' backend (CUDA only)
        :param compile_mode: torch.compile mode for the 'pytorch' backend
        :param export_dir: Folder for the ONNX files and TensorRT engines built for the benchmark
        :param allow_tf32: Allow TF32 in fp32 TensorRT engines
        """
        self.device = device
        self.image_size = image_size
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters
        self.use_amp = use_amp and device.type == "cuda"
        self.compile_mode = compile_mode
        self.export_dir = export_dir
        self.allow_tf32 = allow_tf32

    def _input(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, SegDataset.num_input_channels,
                           self.image_size, self.image_size, device=self.device)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def count_flops(self, model: nn.Module) -> int:
        """FLOPs of one forward pass at batch size 1 (eager fp32; counts matmul/conv-type ops only)."""
        x = self._input(1)
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            model(x)
        return counter.get_total_flops()

    def measure_latency(self, backend: InferenceBackend, batch_size: int) -> Dict[str, float]:
        """Mean/median milliseconds per forward pass and images per second."""
        x = self._input(batch_size)

        for _ in range(self.warmup_iters):
            backend.predict(x)
        self._sync()

        times = torch.tensor([self._time_one_call(backend, x) for _ in range(self.timed_iters)])
        mean_ms = float(times.mean())
        return {"batch_size": batch_size,
                "latency_ms_mean": mean_ms,
                "latency_ms_median": float(times.median()),
                "latency_ms_std": float(times.std()),
                "throughput_img_per_s": batch_size * 1000.0 / mean_ms}

    def _time_one_call(self, backend: InferenceBackend, x: torch.Tensor) -> float:
        """Milliseconds for one predict() call: CUDA events on GPU, wall clock on CPU."""
        if self.device.type != "cuda":
            start = time.perf_counter()
            backend.predict(x)
            return (time.perf_counter() - start) * 1000.0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        backend.predict(x)
        end_event.record()
        torch.cuda.synchronize(self.device)
        return start_event.elapsed_time(end_event)

    def measure_peak_memory(self, backend: InferenceBackend, batch_size: int) -> Optional[float]:
        """Peak memory (MiB) of PyTorch's CUDA allocator during one forward pass; None on CPU."""
        if self.device.type != "cuda":
            return None
        x = self._input(batch_size)
        self._sync()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        backend.predict(x)
        self._sync()
        return torch.cuda.max_memory_allocated(self.device) / 2**20

    def run(self, arch: str, batch_sizes: List[int], backends: List[str]) -> Dict[str, Any]:
        model = ModelFactory.create(arch,
                                    in_channels=SegDataset.num_input_channels,
                                    out_channels=SegDataset.num_output_channels)
        model.to(self.device).eval()

        per_backend: Dict[str, Dict[str, Any]] = {}
        for name in backends:
            print(f"[Benchmark] {arch} / {name} ...")
            backend = self._create_backend(arch, model, name, max_batch=max(batch_sizes))
            measurements: Dict[str, Any] = {"latency": [self.measure_latency(backend, bs) for bs in batch_sizes]}
            if name == "pytorch":
                measurements["peak_inference_memory_mb"] = {str(bs): self.measure_peak_memory(backend, bs)
                                                            for bs in batch_sizes}
            per_backend[name] = measurements

        return {"arch": arch,
                "n_params": self.count_params(model),
                "gflops_per_image": self.count_flops(model) / 1e9,
                "backends": per_backend}

    def _create_backend(self, arch: str, model: nn.Module, name: str, max_batch: int) -> InferenceBackend:
        """The PyTorch model itself, or an ONNX / TensorRT export of it written to --export-dir."""
        if name == "pytorch":
            return PyTorchBackend(model, self.device, self.compile_mode, self.use_amp)

        arch_dir = os.path.join(self.export_dir, arch)
        os.makedirs(arch_dir, exist_ok=True)
        precision, kind = name.split(".")
        onnx_path = os.path.join(arch_dir, f"model.{precision}.onnx")
        ModelExporter.export_onnx(model, onnx_path, self.image_size, precision)
        if kind == "onnx":
            return create_backend(onnx_path, arch, self.device)

        engine_path = os.path.join(arch_dir, f"model.{precision}.engine")
        ModelExporter.build_tensorrt_engine(onnx_path,
                                            engine_path,
                                            self.image_size,
                                            max_batch=max_batch,
                                            opt_batch=max_batch,
                                            allow_tf32=self.allow_tf32,
                                            workspace_gib=4.0)
        return create_backend(engine_path, arch, self.device)

    @classmethod
    def cli(cls) -> None:
        parser = argparse.ArgumentParser(description="Compute benchmark for segmentation models")
        parser.add_argument("--archs", nargs="+", default=ModelFactory.available(), choices=ModelFactory.available())
        parser.add_argument("--backends", nargs="+", default=["pytorch"], choices=BACKENDS,
                            help="'pytorch' and/or export formats (default: pytorch)")
        parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16])
        parser.add_argument("--image-size", type=int, default=256)
        parser.add_argument("--warmup-iters", type=int, default=20)
        parser.add_argument("--timed-iters", type=int, default=100)
        parser.add_argument("--no-amp", action="store_true",
                            help="Disable fp16 autocast of the 'pytorch' backend (default: AMP on, as in training)")
        parser.add_argument("--compile", type=str, nargs="?", const="default", default="none", choices=COMPILE_MODES,
                            help="torch.compile mode of the 'pytorch' backend; '--compile' alone means 'default'")
        parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 in fp32 TensorRT engines")
        parser.add_argument("--export-dir", type=str, default="./benchmark_exports",
                            help="Where ONNX files and TensorRT engines for the benchmark are written")
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--out", type=str, default="./compute_benchmark.json")
        args = parser.parse_args()

        device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
        bench = cls(device, args.image_size, args.warmup_iters, args.timed_iters,
                    use_amp=not args.no_amp, compile_mode=args.compile, export_dir=args.export_dir,
                    allow_tf32=not args.no_tf32)

        env: Dict[str, Any] = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                               "torch": torch.__version__,
                               "amp": bench.use_amp,
                               "compile": args.compile,
                               "image_size": args.image_size,
                               "notes": ["'amp' and 'compile' apply to the 'pytorch' backend only",
                                         "peak_inference_memory_mb counts PyTorch's allocator only, so it is "
                                         "reported for the 'pytorch' backend only",
                                         "onnx latencies include host<->device copies (ONNX Runtime reads numpy "
                                         "arrays)"]}
        if any(name.endswith(".engine") for name in args.backends):
            import tensorrt as trt

            env["tensorrt"] = trt.__version__
            env["tensorrt_tf32"] = not args.no_tf32

        results = [bench.run(arch, args.batch_sizes, args.backends) for arch in args.archs]

        with open(args.out, "w") as fp:
            json.dump({"env": env, "results": results}, fp, indent=2)
        print(f"[Benchmark] wrote {args.out}")


if __name__ == "__main__":
    ModelBenchmark.cli()
