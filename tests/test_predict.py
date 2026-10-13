"""Tests for predict.py helpers: bundle guard, post-processing choice, per-patient grouping and saved probabilities."""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import predict
from inference import InferenceBackend
from predict import _choose_postprocess, _create_backend, _group_by_patient, _reassemble_volumes, _save_masks


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


# ------------------------------------------------------ post-processing choice

class ProbabilityDataset(Dataset):
    """One patient whose 'image' channel 0 already holds the probabilities PassThroughBackend returns."""

    context_slices = 0

    def __init__(self, probabilities: np.ndarray, gt: np.ndarray) -> None:
        self.probabilities, self.gt = probabilities, gt
        self.patient_ids = ["P"]
        self.flat_index: List[Tuple[int, int]] = [(0, s) for s in range(gt.shape[0])]

    def __len__(self) -> int:
        return self.gt.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = np.repeat(self.probabilities[idx][None], 3, axis=0)
        return torch.from_numpy(image).float(), torch.from_numpy(self.gt[idx][None]).float()


class PassThroughBackend(InferenceBackend):
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :1]


def _choose(monkeypatch: pytest.MonkeyPatch, probabilities: np.ndarray, gt: np.ndarray) -> str:
    monkeypatch.setattr(predict, "_build_dataset", lambda cfg: ProbabilityDataset(probabilities, gt))
    cfg = argparse.Namespace(split="test", batch_size=2)
    chosen, scores = _choose_postprocess(cfg, PassThroughBackend(), torch.device("cpu"))
    assert set(scores) == {"none", "drop-small", "lcc"}
    return chosen


def test_auto_keeps_separate_lesions_the_model_found(monkeypatch: pytest.MonkeyPatch) -> None:
    gt = np.zeros((3, 12, 12), dtype=np.float32)
    gt[0:2, 1:4, 1:4] = 1   # two separate lesions of similar size
    gt[1:3, 7:10, 7:10] = 1
    # A perfect prediction: 'lcc' would delete the second lesion; 'none' and 'drop-small' tie, the gentler one wins
    assert _choose(monkeypatch, gt * 0.9, gt) == "none"


def test_auto_removes_a_false_blob_when_that_helps(monkeypatch: pytest.MonkeyPatch) -> None:
    gt = np.zeros((3, 12, 12), dtype=np.float32)
    gt[0:2, 1:5, 1:5] = 1   # 32-voxel tumour
    probabilities = gt * 0.9
    probabilities[1:3, 8:10, 8:10] = 0.9  # separate 8-voxel false positive, too big for 'drop-small'
    assert _choose(monkeypatch, probabilities, gt) == "lcc"


def test_auto_postprocess_needs_validation_patients() -> None:
    cfg = argparse.Namespace(postprocess="auto", split="all")
    with pytest.raises(SystemExit, match="--split all"):
        predict.run_inference(cfg)


def test_reassemble_volumes_applies_the_chosen_postprocess() -> None:
    preds = [np.zeros((1, 6, 6), dtype=np.float32) for _ in range(2)]
    preds[0][0, 0:4, 0:4] = 0.9   # 16 voxels
    preds[1][0, 5, 5] = 0.9       # separate single voxel, under 10% of the largest component
    targets = [np.zeros((1, 6, 6), dtype=np.float32) for _ in range(2)]
    inputs = [np.zeros((3, 6, 6), dtype=np.float32) for _ in range(2)]
    flat_index = [(0, 0), (0, 1)]

    for method, expected_voxels in (("none", 17), ("drop-small", 16), ("lcc", 16)):
        volumes = _reassemble_volumes(inputs, preds, targets, flat_index, ["P"], method)
        assert volumes["P"][1].sum() == expected_voxels
