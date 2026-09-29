"""Smoke tests for benchmark.ModelBenchmark on CPU (tiny image, few iterations)."""

import importlib.util
from pathlib import Path

import pytest
import torch

from benchmark import ModelBenchmark

HAS_ONNX = all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxscript", "onnxruntime"))


def _benchmark(export_dir: Path) -> ModelBenchmark:
    return ModelBenchmark(torch.device("cpu"), image_size=64, warmup_iters=1, timed_iters=2, use_amp=True,
                          compile_mode="none", export_dir=str(export_dir), allow_tf32=True)


def test_pytorch_backend_result_layout(tmp_path: Path) -> None:
    result = _benchmark(tmp_path).run("unet", batch_sizes=[1, 2], backends=["pytorch"])

    assert result["arch"] == "unet"
    assert result["n_params"] > 0 and result["gflops_per_image"] > 0
    pytorch = result["backends"]["pytorch"]
    assert [m["batch_size"] for m in pytorch["latency"]] == [1, 2]
    assert all(m["latency_ms_mean"] > 0 and m["throughput_img_per_s"] > 0 for m in pytorch["latency"])
    assert pytorch["peak_inference_memory_mb"] == {"1": None, "2": None}  # CPU: not measured


@pytest.mark.skipif(not HAS_ONNX, reason="needs onnx, onnxscript and onnxruntime (uv sync --group onnx)")
def test_onnx_backend_is_exported_and_timed(tmp_path: Path) -> None:
    result = _benchmark(tmp_path).run("unet", batch_sizes=[2], backends=["fp32.onnx"])

    assert (tmp_path / "unet" / "model.fp32.onnx").exists()
    onnx = result["backends"]["fp32.onnx"]
    assert onnx["latency"][0]["latency_ms_mean"] > 0
    assert "peak_inference_memory_mb" not in onnx  # PyTorch's allocator does not see ONNX Runtime memory
