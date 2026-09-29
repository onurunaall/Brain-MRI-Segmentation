"""Tests for predict.py helpers: bundle guard, per-patient grouping and saved probabilities."""

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from predict import _create_backend, _group_by_patient, _save_masks


@pytest.mark.parametrize("split", ["validation", "test"])
def test_bundle_is_refused_on_cross_validation_patients(split: str) -> None:
    cfg = argparse.Namespace(bundle_dir="./models/unet", bundle_file="model.pt", split=split,
                             model_path=None, arch="unet", compile="none", amp=False, image_size=256)
    with pytest.raises(SystemExit, match="--split all"):
        _create_backend(cfg, torch.device("cpu"))


def test_group_by_patient_stacks_consecutive_slices() -> None:
    slices = [np.full((1, 2, 2), value, dtype=np.float32) for value in range(5)]
    flat_index = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]

    volumes = _group_by_patient(slices, flat_index, ["A", "B"])

    assert volumes["A"].shape == (2, 1, 2, 2)
    assert volumes["B"].shape == (3, 1, 2, 2)
    np.testing.assert_array_equal(volumes["B"][:, 0, 0, 0], [2, 3, 4])


def test_save_masks_stores_probabilities_as_uint8(tmp_path: Path) -> None:
    vol_in = np.zeros((1, 3, 1, 4), dtype=np.float32)
    zeros = np.zeros((1, 1, 1, 4), dtype=np.float32)
    probability = np.array([0.0, 0.5, 1.0, 1.2], dtype=np.float32).reshape(1, 1, 1, 4)  # 1.2: clipped to 1

    _save_masks(str(tmp_path / "m.npz"), {"P": (vol_in, zeros, zeros)}, {"P": probability})

    with np.load(tmp_path / "m.npz") as npz:
        assert sorted(npz.files) == ["P__flair", "P__gt", "P__pred", "P__prob"]
        assert npz["P__prob"].dtype == np.uint8
        np.testing.assert_array_equal(npz["P__prob"], [[[0, 128, 255, 255]]])
