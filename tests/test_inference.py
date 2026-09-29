"""Tests for inference.py: torch.compile modes, backends, ensembles and model-file dispatch (CPU only)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from inference import (COMPILE_MODES, EnsembleBackend, InferenceBackend, PyTorchBackend, compile_model,
                       create_backend)
from network import ModelFactory

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
    backend = PyTorchBackend(ModelFactory.create("unet", 3, 1), CPU)
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

def test_checkpoint_saved_from_compiled_model_loads(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = ModelFactory.create("unet", 3, 1).eval()
    path = tmp_path / "best_model.pt"
    torch.save({f"_orig_mod.{key}": value for key, value in model.state_dict().items()}, path)

    backend = create_backend(str(path), "unet", CPU)
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = model(images)
    assert torch.allclose(backend.predict(images), expected)


def test_unknown_model_file_type_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model file type"):
        create_backend(str(tmp_path / "model.bin"), "unet", CPU)


@pytest.mark.parametrize("compile_mode, use_amp", [("default", False), ("none", True)])
def test_compile_and_amp_are_rejected_for_exported_models(tmp_path: Path, compile_mode: str, use_amp: bool) -> None:
    with pytest.raises(ValueError, match="only apply to .pt checkpoints"):
        create_backend(str(tmp_path / "model.fp32.onnx"), "unet", CPU, compile_mode, use_amp)
