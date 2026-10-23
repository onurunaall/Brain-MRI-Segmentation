"""Tests for inference.py: torch.compile modes, backends, ensembles and model-file dispatch (CPU only)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mriseg.inference import (COMPILE_MODES, EnsembleBackend, InferenceBackend, PyTorchBackend, compile_model,
                              create_backend, load_trained_model, model_record, read_checkpoint, save_checkpoint)
from mriseg.network import create_model

CPU = torch.device("cpu")


class ConstantBackend(InferenceBackend):
    """Predicts the same probability everywhere."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return torch.full((images.shape[0], 1, images.shape[2], images.shape[3]), self.value)


# --------------------------------------------------------------- compile

def test_compile_none_returns_the_model_itself() -> None:
    model = nn.Conv2d(3, 1, kernel_size=1)
    assert compile_model(model, "none") is model


def test_every_compile_mode_is_accepted_by_torch() -> None:
    # torch.compile validates `mode` when called (compilation itself happens on the first forward pass)
    for mode in COMPILE_MODES:
        compile_model(nn.Identity(), mode)


def test_unknown_compile_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown compile mode"):
        compile_model(nn.Identity(), "fast")


# -------------------------------------------------------------- backends

def test_pytorch_backend_returns_float32_probabilities() -> None:
    torch.manual_seed(0)
    backend = PyTorchBackend(create_model("unet", 3, 1), CPU)
    probabilities = backend.predict(torch.randn(2, 3, 32, 32))

    assert probabilities.shape == (2, 1, 32, 32)
    assert probabilities.dtype == torch.float32
    assert 0.0 <= float(probabilities.min()) and float(probabilities.max()) <= 1.0


def test_amp_is_ignored_on_cpu() -> None:
    assert not PyTorchBackend(nn.Identity(), CPU, use_amp=True).use_amp


def test_ensemble_averages_member_probabilities() -> None:
    ensemble = EnsembleBackend([ConstantBackend(0.2), ConstantBackend(0.6)])
    probabilities = ensemble.predict(torch.zeros(3, 3, 4, 4))
    assert torch.allclose(probabilities, torch.full((3, 1, 4, 4), 0.4))


def test_empty_ensemble_raises() -> None:
    with pytest.raises(ValueError):
        EnsembleBackend([])


# ------------------------------------------------------ create_backend

def _save_trained(path: Path, arch: str = "unet", context_slices: int = 0) -> nn.Module:
    """A model saved the way train.py saves it: weights plus the record that describes them."""
    torch.manual_seed(0)
    model = create_model(arch, 3 * (2 * context_slices + 1), 1).eval()
    save_checkpoint(str(path), model.state_dict(), {"arch": arch, "context_slices": context_slices, "image_size": 32})
    return model


@pytest.mark.parametrize("arch, context_slices", [("unet", 0), ("resunet", 1)])
def test_checkpoint_record_rebuilds_the_model(tmp_path: Path, arch: str, context_slices: int) -> None:
    path = tmp_path / "best_model.pt"
    trained = _save_trained(path, arch, context_slices)

    backend = create_backend(str(path), CPU)  # no architecture or input size passed: the record has them
    images = torch.randn(1, 3 * (2 * context_slices + 1), 32, 32)
    with torch.no_grad():
        expected = torch.sigmoid(trained(images))  # the model returns logits, the backend probabilities
    assert torch.allclose(backend.predict(images), expected)

    model, record = load_trained_model(str(path), CPU)
    assert type(model) is type(trained) and record["context_slices"] == context_slices


def test_weights_only_checkpoint_is_refused(tmp_path: Path) -> None:
    torch.save(create_model("unet", 3, 1).state_dict(), tmp_path / "old.pt")
    with pytest.raises(ValueError, match="older version"):
        read_checkpoint(str(tmp_path / "old.pt"), CPU)


def test_exported_files_take_the_record_of_the_model_pt_next_to_them(tmp_path: Path) -> None:
    _save_trained(tmp_path / "model.pt")
    assert model_record(str(tmp_path / "model.fp16.engine"))["arch"] == "unet"
    assert model_record(str(tmp_path / "model.pt"))["image_size"] == 32
    with pytest.raises(FileNotFoundError, match="no model.pt next to it"):
        model_record(str(tmp_path / "elsewhere" / "model.fp32.onnx"))


def test_unknown_model_file_type_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model file type"):
        create_backend(str(tmp_path / "model.bin"), CPU)


@pytest.mark.parametrize("compile_mode, use_amp", [("default", False), ("none", True)])
def test_compile_and_amp_are_rejected_for_exported_models(tmp_path: Path, compile_mode: str, use_amp: bool) -> None:
    with pytest.raises(ValueError, match="only apply to .pt checkpoints"):
        create_backend(str(tmp_path / "model.fp32.onnx"), CPU, compile_mode, use_amp)
