"""
Tests for export_models.py: ONNX export parity, model bundles and ensemble loading.
ONNX tests need the 'onnx' dependency group; the TensorRT test needs a CUDA GPU and TensorRT and is skipped otherwise.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest
import torch
import torch.nn as nn

from mriseg.export_models import BundleExporter, build_tensorrt_engine, export_onnx
from mriseg.inference import create_backend, load_bundle, read_bundle_manifest, read_checkpoint, save_checkpoint
from mriseg.network import create_model

CPU = torch.device("cpu")
HAS_ONNX = all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxscript", "onnxruntime"))
HAS_TENSORRT = importlib.util.find_spec("tensorrt") is not None and torch.cuda.is_available()
needs_onnx = pytest.mark.skipif(not HAS_ONNX, reason="needs onnx, onnxscript and onnxruntime (uv sync --group onnx)")


def _model(arch: str, seed: int, in_channels: int = 3) -> nn.Module:
    torch.manual_seed(seed)
    return create_model(arch, in_channels, 1).eval()


def _record(arch: str, fold: int, seed: int, image_size: int, context_slices: int) -> Dict[str, Any]:
    return {"arch": arch, "image_size": image_size, "context_slices": context_slices, "fold": fold, "seed": seed,
            "train_patients": ["T"], "val_patients": ["V"], "test_patients": ["A", "B"]}


def _write_trained_run(runs_dir: Path, arch: str, fold: int, seed: int, model: nn.Module, image_size: int,
                       context_slices: int = 0, meta: Optional[Dict[str, Any]] = None) -> None:
    """A finished run in the layout run_cv.sh produces (checkpoint with its record, train summary, test results)."""
    run_dir = runs_dir / arch / f"fold{fold}_seed{seed}"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    save_checkpoint(str(run_dir / "checkpoints" / "best_model.pt"), model.state_dict(),
                    _record(arch, fold, seed, image_size, context_slices))
    (run_dir / "logs" / "train_summary.json").write_text(json.dumps(
        {"fold": fold, "seed": seed, "best_val_dice": 0.8 + fold / 100, "best_epoch": 10 + fold}))
    (run_dir / "test_results.json").write_text(json.dumps(
        {"meta": {"arch": arch, "fold": fold, "seed": seed, **(meta or {})},
         "per_patient": {"A": {"dice": 0.5}, "B": {"dice": 0.7}}}))


def _export_args(runs_dir: Path,
                 out_dir: Path,
                 formats: Sequence[str] = (),
                 overwrite: bool = False) -> argparse.Namespace:
    return argparse.Namespace(runs_dir=str(runs_dir), out_dir=str(out_dir), archs=["unet"], formats=list(formats),
                              max_batch=8, opt_batch=None, no_tf32=False, workspace_gib=1.0,
                              device="cpu", overwrite=overwrite)


# ------------------------------------------------------------------- ONNX

@needs_onnx
@pytest.mark.parametrize("arch, image_size", [("unet", 64), ("resunet", 64), ("swinunetr", 128)])
def test_fp32_onnx_matches_pytorch_for_any_batch_size(tmp_path: Path, arch: str, image_size: int) -> None:
    model = _model(arch, seed=0)
    path = str(tmp_path / "model.fp32.onnx")
    export_onnx(model, path, image_size, "fp32", 3)
    backend = create_backend(path, CPU)

    for batch_size in (1, 3):
        images = torch.randn(batch_size, 3, image_size, image_size)
        with torch.no_grad():
            expected = torch.sigmoid(model(images))  # the exported graph ends with the sigmoid
        assert torch.allclose(backend.predict(images), expected, atol=1e-5)


@needs_onnx
def test_fp16_onnx_is_close_and_leaves_the_model_in_fp32(tmp_path: Path) -> None:
    model = _model("unet", seed=0)
    path = str(tmp_path / "model.fp16.onnx")
    export_onnx(model, path, 64, "fp16", 3)

    assert all(p.dtype == torch.float32 for p in model.parameters())
    backend = create_backend(path, CPU)
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = torch.sigmoid(model(images))
    probabilities = backend.predict(images)
    assert probabilities.dtype == torch.float32
    assert float((probabilities - expected).abs().max()) < 5e-3


# ---------------------------------------------------------------- bundles

def test_bundle_manifest_and_ensemble_prediction(tmp_path: Path) -> None:
    models = [_model("unet", seed=fold) for fold in range(2)]
    for fold, model in enumerate(models):
        _write_trained_run(tmp_path / "runs", "unet", fold, 0, model, image_size=64)

    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")
    bundle_dir = str(tmp_path / "models" / "unet")
    manifest = read_bundle_manifest(bundle_dir)

    assert manifest["arch"] == "unet"
    assert manifest["image_size"] == 64
    assert manifest["files"] == ["model.pt"]
    assert [m["name"] for m in manifest["members"]] == ["fold0_seed0", "fold1_seed0"]
    assert [m["best_epoch"] for m in manifest["members"]] == [10, 11]
    assert manifest["members"][0]["test_dice_mean"] == pytest.approx(0.6)
    assert manifest["context_slices"] == 0 and manifest["in_channels"] == 3
    assert [(m["fold"], m["seed"]) for m in manifest["members"]] == [(0, 0), (1, 0)]  # from the checkpoint records
    assert manifest["members"][0]["postprocess"] is None

    ensemble, _ = load_bundle(bundle_dir, "model.pt", CPU)
    images = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        expected = (torch.sigmoid(models[0](images)) + torch.sigmoid(models[1](images))) / 2
    assert torch.allclose(ensemble.predict(images), expected, atol=1e-6)


def test_context_bundle_records_input_and_postprocess(tmp_path: Path) -> None:
    models = [_model("unet", seed=fold, in_channels=9) for fold in range(2)]
    for fold, (model, method) in enumerate(zip(models, ["lcc", "none"])):
        _write_trained_run(tmp_path / "runs", "unet", fold, 0, model, image_size=64,
                           context_slices=1, meta={"postprocess": method})

    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")
    bundle_dir = str(tmp_path / "models" / "unet")
    manifest = read_bundle_manifest(bundle_dir)

    assert manifest["context_slices"] == 1 and manifest["in_channels"] == 9
    assert [m["postprocess"] for m in manifest["members"]] == ["lcc", "none"]

    ensemble, _ = load_bundle(bundle_dir, "model.pt", CPU)
    images = torch.randn(2, 9, 64, 64)
    with torch.no_grad():
        expected = (torch.sigmoid(models[0](images)) + torch.sigmoid(models[1](images))) / 2
    assert torch.allclose(ensemble.predict(images), expected, atol=1e-6)


def test_runs_with_different_context_slices_are_not_bundled(tmp_path: Path) -> None:
    _write_trained_run(tmp_path / "runs", "unet", 0, 0, _model("unet", 0), image_size=64)
    _write_trained_run(tmp_path / "runs", "unet", 1, 0, _model("unet", 1, in_channels=9), image_size=64,
                       context_slices=1)
    with pytest.raises(SystemExit, match="different context_slices"):
        BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")


def test_existing_bundle_is_only_replaced_with_overwrite(tmp_path: Path) -> None:
    _write_trained_run(tmp_path / "runs", "unet", 0, 0, _model("unet", 0), image_size=64)
    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")

    with pytest.raises(SystemExit, match="--overwrite"):
        BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")
    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models", overwrite=True), CPU).export("unet")


def test_runs_with_different_image_sizes_are_not_bundled_and_keep_the_old_bundle(tmp_path: Path) -> None:
    _write_trained_run(tmp_path / "runs", "unet", 0, 0, _model("unet", 0), image_size=64)
    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")
    _write_trained_run(tmp_path / "runs", "unet", 1, 0, _model("unet", 1), image_size=128)

    with pytest.raises(SystemExit, match="different image_size"):
        BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models", overwrite=True), CPU).export("unet")

    members = read_bundle_manifest(str(tmp_path / "models" / "unet"))["members"]
    assert [m["name"] for m in members] == ["fold0_seed0"]


def test_failed_export_keeps_the_old_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_trained_run(tmp_path / "runs", "unet", 0, 0, _model("unet", 0), image_size=64)
    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")
    _write_trained_run(tmp_path / "runs", "unet", 1, 0, _model("unet", 1), image_size=64)

    def fail(*args: Any) -> None:
        raise RuntimeError("TensorRT build failed")

    monkeypatch.setattr(BundleExporter, "_export_member", fail)
    with pytest.raises(RuntimeError, match="TensorRT build failed"):
        BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models", overwrite=True), CPU).export("unet")

    members = read_bundle_manifest(str(tmp_path / "models" / "unet"))["members"]
    assert [m["name"] for m in members] == ["fold0_seed0"]  # the old bundle, untouched

    monkeypatch.undo()
    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models", overwrite=True), CPU).export("unet")
    members = read_bundle_manifest(str(tmp_path / "models" / "unet"))["members"]
    assert [m["name"] for m in members] == ["fold0_seed0", "fold1_seed0"]
    assert sorted(os.listdir(tmp_path / "models")) == ["unet"]  # no leftover build folder


def test_bundle_member_keeps_the_checkpoint_record(tmp_path: Path) -> None:
    _write_trained_run(tmp_path / "runs", "unet", 0, 0, _model("unet", 0), image_size=64)
    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")

    _, meta = read_checkpoint(str(tmp_path / "models" / "unet" / "fold0_seed0" / "model.pt"), CPU)
    assert meta == _record("unet", 0, 0, 64, 0)


def test_architecture_without_runs_is_skipped(tmp_path: Path) -> None:
    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")
    assert not os.path.exists(tmp_path / "models" / "unet")


@pytest.mark.parametrize("requested, expected", [(["fp16.engine"], ["fp16.onnx", "fp16.engine"]),
                                                 (["fp32.engine", "fp32.onnx"], ["fp32.onnx", "fp32.engine"]),
                                                 ([], [])])
def test_engines_bring_their_onnx_file(requested: List[str], expected: List[str]) -> None:
    assert BundleExporter._formats_to_write(requested) == expected


@needs_onnx
def test_onnx_bundle_matches_checkpoint_bundle(tmp_path: Path) -> None:
    for fold in range(2):
        _write_trained_run(tmp_path / "runs", "unet", fold, 0, _model("unet", fold), image_size=64)
    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models", formats=["fp32.onnx"]), CPU).export("unet")

    bundle_dir = str(tmp_path / "models" / "unet")
    assert read_bundle_manifest(bundle_dir)["files"] == ["model.pt", "model.fp32.onnx"]
    checkpoint_ensemble, _ = load_bundle(bundle_dir, "model.pt", CPU)
    onnx_ensemble, _ = load_bundle(bundle_dir, "model.fp32.onnx", CPU)

    images = torch.randn(3, 3, 64, 64)
    assert torch.allclose(onnx_ensemble.predict(images), checkpoint_ensemble.predict(images), atol=1e-5)


def test_missing_export_in_bundle_raises(tmp_path: Path) -> None:
    _write_trained_run(tmp_path / "runs", "unet", 0, 0, _model("unet", 0), image_size=64)
    BundleExporter(_export_args(tmp_path / "runs", tmp_path / "models"), CPU).export("unet")
    with pytest.raises(FileNotFoundError, match="export_models.py"):
        load_bundle(str(tmp_path / "models" / "unet"), "model.fp16.engine", CPU)


# --------------------------------------------------------------- TensorRT

@pytest.mark.skipif(not HAS_TENSORRT, reason="needs a CUDA GPU and TensorRT (uv sync --group tensorrt)")
@pytest.mark.parametrize("precision, tolerance", [("fp32", 1e-2), ("fp16", 5e-2)])
def test_tensorrt_engine_matches_pytorch(tmp_path: Path, precision: str, tolerance: float) -> None:
    """Catches a broken export / runtime path (tolerances allow TF32 in PyTorch's convolutions and fp16 rounding)."""
    device = torch.device("cuda:0")
    model = _model("unet", seed=0).to(device)
    onnx_path = str(tmp_path / f"model.{precision}.onnx")
    engine_path = str(tmp_path / f"model.{precision}.engine")
    export_onnx(model, onnx_path, 128, precision, 3)
    build_tensorrt_engine(onnx_path, engine_path, 128, max_batch=4, opt_batch=4,
                          allow_tf32=False, workspace_gib=1.0)
    backend = create_backend(engine_path, device)

    images = torch.randn(6, 3, 128, 128, device=device)  # larger than max_batch: predict() splits it
    with torch.no_grad():
        expected = torch.sigmoid(model(images))
    probabilities = backend.predict(images)

    assert probabilities.shape == expected.shape
    assert probabilities.dtype == torch.float32
    assert float((probabilities - expected).abs().max()) < tolerance
