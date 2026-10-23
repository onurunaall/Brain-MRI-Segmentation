"""Tests for the evaluation module: grid Dice for training, original-pixel scores for reporting."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mriseg.evaluation import hd95_px, raw_dice_per_patient, score_patient, write_results
from mriseg.utils import VolumeGeometry


def test_raw_dice_per_patient_groups_slices_by_patient() -> None:
    # Patient A: 2 slices, perfect. Patient B: 1 slice, half overlap (Dice 0.5).
    gt_a = np.zeros((1, 4, 4), dtype=np.float32)
    gt_a[0, 1:3, 1:3] = 1
    gt_b = np.zeros((1, 4, 4), dtype=np.float32)
    gt_b[0, 0, 1:3] = 1
    pred_b = np.zeros((1, 4, 4), dtype=np.float32)
    pred_b[0, 0, 0:2] = 0.9

    preds = [gt_a * 0.8, gt_a * 0.8, pred_b]
    targets = [gt_a, gt_a, gt_b]
    flat_index = [(0, 0), (0, 1), (1, 0)]

    scores = raw_dice_per_patient(preds, targets, flat_index, ["A", "B"])

    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(0.5, abs=1e-6)


def test_raw_dice_rejects_slices_in_the_wrong_order() -> None:
    gt = np.zeros((1, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="out of order"):
        raw_dice_per_patient([gt, gt], [gt, gt], [(0, 1), (0, 0)], ["A"])


def test_raw_dice_keeps_stray_components() -> None:
    """Unlike the reported metric, raw Dice does NOT apply largest-connected-component filtering."""
    gt = np.zeros((1, 8, 8), dtype=np.float32)
    gt[0, 1:3, 1:3] = 1
    pred = gt.copy()
    pred[0, 7, 7] = 1
    scores = raw_dice_per_patient([pred], [gt], [(0, 0)], ["A"])
    assert scores["A"] == pytest.approx(2 * 4 / (5 + 4), abs=1e-6)


def _geometry(n_slices: int, size: int, z_start: int = 0) -> VolumeGeometry:
    """Geometry of a scan whose grid is its crop: no resizing, only (optionally) dropped end slices."""
    return VolumeGeometry(n_slices=n_slices, height=size, width=size, z_start=z_start, y_start=0, x_start=0,
                          crop_slices=n_slices - 2 * z_start, crop_height=size, crop_width=size, grid_size=size)


def test_hd95_edge_cases() -> None:
    empty = np.zeros((2, 8, 8), dtype=bool)
    blob = empty.copy()
    blob[1, 2:5, 2:5] = True

    assert hd95_px(empty, empty) == 0.0
    assert math.isnan(hd95_px(empty, blob))  # missed
    assert math.isnan(hd95_px(blob, empty))  # false alarm
    assert hd95_px(blob, blob) == pytest.approx(0.0)


def test_hd95_shifted_mask_is_positive() -> None:
    gt = np.zeros((1, 12, 12), dtype=bool)
    gt[0, 2:6, 2:6] = True
    assert 0.0 < hd95_px(np.roll(gt, 4, axis=2), gt) <= 4.0 + 1e-6


def test_score_patient_applies_the_postprocessing_only_to_dice() -> None:
    gt = np.zeros((3, 12, 12), dtype=bool)
    gt[0:2, 1:5, 1:5] = True                  # 32-voxel tumour
    probability = gt * 0.9
    probability[2, 10, 10] = 0.9              # one false-positive voxel elsewhere

    scores = score_patient(probability, _geometry(3, 12), gt, "lcc")
    assert scores["dice"] == pytest.approx(1.0)                      # lcc removed the stray voxel
    assert scores["dice_raw"] == pytest.approx(2 * 32 / 65, abs=1e-6)  # without post-processing it counts
    assert scores["hd95_px"] == pytest.approx(0.0)


def test_score_patient_counts_tumour_on_slices_the_model_never_saw() -> None:
    # The first and last slice are dropped by preprocessing; tumour there is missed, not ignored
    gt = np.zeros((5, 8, 8), dtype=bool)
    gt[:, 2:6, 2:6] = True
    perfect_on_grid = gt[1:-1].astype(np.float32)
    scores = score_patient(perfect_on_grid, _geometry(5, 8, z_start=1), gt, "none")
    assert scores["dice"] == pytest.approx(2 * 48 / (48 + 80), abs=1e-6)


def test_write_results_schema(tmp_path: Path) -> None:
    path = tmp_path / "res.json"
    meta = {"arch": "unet", "fold": 0}
    per_patient = {"A": {"dice": 0.9, "dice_raw": 0.85, "hd95_px": 3.0}}
    write_results(str(path), meta, per_patient)

    data = json.loads(path.read_text())
    assert data == {"meta": meta, "per_patient": per_patient}
