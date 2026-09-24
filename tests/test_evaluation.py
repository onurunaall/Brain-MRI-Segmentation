"""Tests for evaluation.PatientEvaluator."""

import json
import math

import numpy as np
import pytest

from evaluation import PatientEvaluator


def test_strip_compile_prefix():
    state = {"_orig_mod.enc1.weight": 1, "_orig_mod.head.bias": 2}
    assert dict(PatientEvaluator.strip_compile_prefix(state)) == {"enc1.weight": 1, "head.bias": 2}


def test_strip_compile_prefix_is_noop_without_prefix():
    state = {"enc1.weight": 1}
    assert dict(PatientEvaluator.strip_compile_prefix(state)) == state


def test_raw_dice_per_patient_groups_slices_by_patient():
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

    scores = PatientEvaluator.raw_dice_per_patient(preds, targets, flat_index, ["A", "B"])

    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(0.5, abs=1e-6)


def test_raw_dice_keeps_stray_components():
    """Unlike the reported metric, raw Dice does NOT apply largest-connected-component filtering."""
    gt = np.zeros((1, 8, 8), dtype=np.float32)
    gt[0, 1:3, 1:3] = 1
    pred = gt.copy()
    pred[0, 7, 7] = 1
    scores = PatientEvaluator.raw_dice_per_patient([pred], [gt], [(0, 0)], ["A"])
    assert scores["A"] == pytest.approx(2 * 4 / (5 + 4), abs=1e-6)


def _volume(pred: np.ndarray, gt: np.ndarray):
    """(input, pred, gt) tuple with the (Z, 1, H, W) layout predict.py builds."""
    return (np.zeros((pred.shape[0], 3) + pred.shape[1:]), pred[:, None], gt[:, None])


def test_hd95_edge_cases():
    empty = np.zeros((2, 8, 8), dtype=np.float32)
    blob = empty.copy()
    blob[1, 2:5, 2:5] = 1

    scores = PatientEvaluator.hd95_per_patient({"both_empty": _volume(empty, empty),
                                                "missed": _volume(empty, blob),
                                                "false_alarm": _volume(blob, empty),
                                                "identical": _volume(blob, blob)})

    assert scores["both_empty"] == 0.0
    assert math.isnan(scores["missed"])
    assert math.isnan(scores["false_alarm"])
    assert scores["identical"] == pytest.approx(0.0)


def test_hd95_shifted_mask_is_positive():
    gt = np.zeros((1, 12, 12), dtype=np.float32)
    gt[0, 2:6, 2:6] = 1
    pred = np.roll(gt, 4, axis=2)
    score = PatientEvaluator.hd95_per_patient({"p": _volume(pred, gt)})["p"]
    assert 0.0 < score <= 4.0 + 1e-6


def test_write_results_schema(tmp_path):
    path = tmp_path / "res.json"
    meta = {"arch": "unet", "fold": 0}
    PatientEvaluator.write_results(str(path), meta, {"A": 0.9}, {"A": 0.85}, {"A": 3.0})

    data = json.loads(path.read_text())
    assert data["meta"] == meta
    assert data["per_patient"] == {"A": {"dice": 0.9, "dice_raw": 0.85, "hd95_vox": 3.0}}
