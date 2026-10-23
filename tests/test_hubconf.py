"""Tests for the torch.hub entry point."""

from pathlib import Path

import pytest
import torch

import hubconf
from mriseg.inference import save_checkpoint
from mriseg.network import UNetModel, create_model


def test_pretrained_unet_is_built_as_its_checkpoint_describes(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trained = create_model("unet", 9, 1).eval()  # 2.5D model, as train.py --context-slices 1 makes
    save_checkpoint(str(tmp_path / "best_model.pt"), trained.state_dict(),
                    {"arch": "unet", "context_slices": 1, "image_size": 256})

    model = hubconf.unet_segmentation(pretrained=True, weights_path=str(tmp_path / "best_model.pt"))
    images = torch.randn(1, 9, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(images), trained(images))


def test_untrained_unet_takes_keyword_arguments() -> None:
    model = hubconf.unet_segmentation(in_channels=9, base_filters=8)
    assert isinstance(model, UNetModel) and model.enc_block1[0].in_channels == 9


def test_other_architectures_are_refused(tmp_path: Path) -> None:
    save_checkpoint(str(tmp_path / "best_model.pt"), create_model("resunet", 3, 1).state_dict(),
                    {"arch": "resunet", "context_slices": 0, "image_size": 256})
    with pytest.raises(ValueError, match="not a U-Net"):
        hubconf.unet_segmentation(pretrained=True, weights_path=str(tmp_path / "best_model.pt"))
