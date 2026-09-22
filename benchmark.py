"""
Compute benchmark: parameters, FLOPs, inference latency/throughput and peak inference memory.
Usage: python benchmark.py --archs unet resunet swinunetr --out ./runs/compute_benchmark.json
Run every architecture on the SAME GPU type in ONE invocation so numbers are comparable.
"""

import argparse
import json
import time
from typing import Dict, List

import torch
from torch.utils.flop_counter import FlopCounterMode

from dataset import MRISegmentationDataset as SegDataset
from network import ModelFactory


class ModelBenchmark:
    """Measures cost metrics of one model under fixed settings (eval mode, no_grad)."""

    def __init__(self,
                 device: torch.device,
                 image_size: int,
                 warmup_iters: int,
                 timed_iters: int,
                 use_amp: bool,
                 use_compile: bool) -> None:
        self.device = device
        self.image_size = image_size
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters
        self.use_amp = use_amp and device.type == "cuda"
        self.use_compile = use_compile

    def _input(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, SegDataset.num_input_channels,
                           self.image_size, self.image_size, device=self.device)

    def _forward(self, model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            return model(x)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def count_params(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def count_flops(self, model: torch.nn.Module) -> int:
        """FLOPs of one forward pass at batch size 1 (eager fp32; counts matmul/conv-type ops only)."""
        x = self._input(1)
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            model(x)
        return counter.get_total_flops()

    def measure_latency(self, model: torch.nn.Module, batch_size: int) -> Dict[str, float]:
        """Mean/median milliseconds per forward pass and images per second."""
        x = self._input(batch_size)

        for _ in range(self.warmup_iters):
            self._forward(model, x)
        self._sync()

        times_ms: List[float] = []
        for _ in range(self.timed_iters):
            if self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                self._forward(model, x)
                end.record()
                torch.cuda.synchronize(self.device)
                times_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                self._forward(model, x)
                times_ms.append((time.perf_counter() - t0) * 1000.0)

        times_t = torch.tensor(times_ms)
        mean_ms = float(times_t.mean())
        return {"batch_size": batch_size,
                "latency_ms_mean": mean_ms,
                "latency_ms_median": float(times_t.median()),
                "latency_ms_std": float(times_t.std()),
                "throughput_img_per_s": batch_size * 1000.0 / mean_ms}

    def measure_peak_memory(self, model: torch.nn.Module, batch_size: int) -> float:
        """Peak allocated GPU memory (MiB) during one forward pass; None on CPU."""
        if self.device.type != "cuda":
            return None
        x = self._input(batch_size)
        self._sync()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self._forward(model, x)
        self._sync()
        return torch.cuda.max_memory_allocated(self.device) / 2**20

    def run(self, arch: str, batch_sizes: List[int]) -> Dict:
        model = ModelFactory.create(arch,
                                    in_channels=SegDataset.num_input_channels,
                                    out_channels=SegDataset.num_output_channels)
        model.to(self.device).eval()

        result = {"arch": arch,
                  "n_params": self.count_params(model),
                  "gflops_per_image": self.count_flops(model) / 1e9}

        timed_model = torch.compile(model) if self.use_compile else model
        result["latency"] = [self.measure_latency(timed_model, bs) for bs in batch_sizes]
        result["peak_inference_memory_mb"] = {str(bs): self.measure_peak_memory(timed_model, bs)
                                              for bs in batch_sizes}
        return result

    @classmethod
    def cli(cls) -> None:
        parser = argparse.ArgumentParser(description="Compute benchmark for segmentation models")
        parser.add_argument("--archs", nargs="+", default=ModelFactory.available(), choices=ModelFactory.available())
        parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16])
        parser.add_argument("--image-size", type=int, default=256)
        parser.add_argument("--warmup-iters", type=int, default=20)
        parser.add_argument("--timed-iters", type=int, default=100)
        parser.add_argument("--no-amp", action="store_true", help="Disable fp16 autocast (default: AMP on, as in training)")
        parser.add_argument("--compile", action="store_true", help="Time the torch.compile'd model")
        parser.add_argument("--device", type=str, default="cuda:0")
        parser.add_argument("--out", type=str, default="./compute_benchmark.json")
        args = parser.parse_args()

        device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
        bench = cls(device, args.image_size, args.warmup_iters, args.timed_iters,
                    use_amp=not args.no_amp, use_compile=args.compile)

        env = {"device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
               "torch": torch.__version__,
               "amp": bench.use_amp,
               "compile": args.compile,
               "image_size": args.image_size}

        results = []
        for arch in args.archs:
            print(f"[Benchmark] {arch} ...")
            results.append(bench.run(arch, args.batch_sizes))

        with open(args.out, "w") as fp:
            json.dump({"env": env, "results": results}, fp, indent=2)
        print(f"[Benchmark] wrote {args.out}")


if __name__ == "__main__":
    ModelBenchmark.cli()
