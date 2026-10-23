"""Tests for predict.py: patients from the checkpoint record, post-processing choice, saved arrays and a full run."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from skimage.io import imread

from conftest import write_patient
from mriseg import predict
from mriseg.inference import save_checkpoint
from mriseg.network import create_model
from mriseg.utils import VolumeGeometry

RECORD = {"arch": "unet", "image_size": 32, "context_slices": 1, "fold": 0, "n_folds": 5, "split_seed": 42,
          "n_validation": 1, "seed": 3,
          "train_patients": ["TCGA_AA_0000_20000101"],
          "val_patients": ["TCGA_AA_0001_20000101"],
          "test_patients": ["TCGA_AA_0002_20000101"]}


def _cfg(**overrides: Any) -> argparse.Namespace:
    settings: Dict[str, Any] = dict(model_path="best_model.pt", bundle_dir=None, bundle_file="model.pt",
                                    compile="none", amp=False, device="cpu", batch_size=4, data_dir="data",
                                    output_dir="predictions", figure_path="dice.png", split="test",
                                    postprocess="auto", results_json=None, masks_npz=None, masks_dir=None,
                                    skip_overlays=True)
    settings.update(overrides)
    return argparse.Namespace(**settings)


# ----------------------------------------------------- which patients

@pytest.mark.parametrize("split", ["validation", "test"])
def test_bundle_is_refused_on_cross_validation_patients(split: str) -> None:
    with pytest.raises(SystemExit, match="--split all"):
        predict._load_model(_cfg(model_path=None, bundle_dir="./models/unet", split=split), torch.device("cpu"))


def test_validation_and_test_patients_come_from_the_checkpoint() -> None:
    assert predict._patients(_cfg(), RECORD, "validation") == ["TCGA_AA_0001_20000101"]
    assert predict._patients(_cfg(), RECORD, "test") == ["TCGA_AA_0002_20000101"]


def test_single_split_model_has_no_test_patients() -> None:
    with pytest.raises(SystemExit, match="without --fold"):
        predict._patients(_cfg(), {**RECORD, "test_patients": []}, "test")


def test_training_patients_are_refused_with_split_all(tmp_path: Path) -> None:
    write_patient(tmp_path, "TCGA_AA_0000_20000101", n_slices=4, tumour_slices=())
    write_patient(tmp_path, "TCGA_NN_0009_20000101", n_slices=4, tumour_slices=())
    with pytest.raises(SystemExit, match="1 of the 2 patients .* were TRAINING patients"):
        predict._patients(_cfg(data_dir=str(tmp_path)), RECORD, "all")
    # A bundle manifest lists no training patients
    assert len(predict._patients(_cfg(data_dir=str(tmp_path)), {"arch": "unet"}, "all")) == 2


def test_auto_postprocess_needs_validation_patients() -> None:
    with pytest.raises(SystemExit, match="--split all"):
        predict.run_inference(_cfg(split="all"))


# ------------------------------------------------- post-processing choice

def _identity_geometry(gt: np.ndarray) -> VolumeGeometry:
    """Geometry of a scan whose grid is its own size: the grid prediction maps back unchanged."""
    z, size = gt.shape[0], gt.shape[1]
    return VolumeGeometry(n_slices=z, height=size, width=size, z_start=0, y_start=0, x_start=0,
                          crop_slices=z, crop_height=size, crop_width=size, grid_size=size)


def _choose(probabilities: np.ndarray, gt: np.ndarray) -> str:
    chosen, scores = predict.best_postprocess({"P": probabilities}, {"P": _identity_geometry(gt)}, {"P": gt > 0.5})
    assert set(scores) == {"none", "drop-small", "lcc"}
    return chosen


def test_auto_keeps_separate_lesions_the_model_found() -> None:
    gt = np.zeros((3, 12, 12), dtype=np.float32)
    gt[0:2, 1:4, 1:4] = 1   # two separate lesions of similar size
    gt[1:3, 7:10, 7:10] = 1
    # A perfect prediction: 'lcc' would delete the second lesion; 'none' and 'drop-small' tie, the gentler one wins
    assert _choose(gt * 0.9, gt) == "none"


def test_auto_removes_a_false_blob_when_that_helps() -> None:
    gt = np.zeros((3, 12, 12), dtype=np.float32)
    gt[0:2, 1:5, 1:5] = 1   # 32-voxel tumour
    probabilities = gt * 0.9
    probabilities[1:3, 8:10, 8:10] = 0.9  # separate 8-voxel false positive, too big for 'drop-small'
    assert _choose(probabilities, gt) == "lcc"


# ------------------------------------------------------------ saved files

def test_save_masks_stores_probabilities_as_uint8(tmp_path: Path) -> None:
    zeros = {"P": np.zeros((1, 1, 4), dtype=np.float32)}
    probability = {"P": np.array([0.0, 0.5, 1.0, 1.2], dtype=np.float32).reshape(1, 1, 4)}  # 1.2: clipped to 1

    predict._save_masks(str(tmp_path / "m.npz"), zeros, zeros, zeros, probability)

    with np.load(tmp_path / "m.npz") as npz:
        assert sorted(npz.files) == ["P__flair", "P__gt", "P__pred", "P__prob"]
        assert npz["P__prob"].dtype == np.uint8
        np.testing.assert_array_equal(npz["P__prob"], [[[0, 128, 255, 255]]])


def test_full_run_on_the_test_patients(tmp_path: Path) -> None:
    """predict.py end to end: the checkpoint names the patients; scores and masks are in original pixels."""
    data_dir = tmp_path / "data"
    for pid in ["TCGA_AA_0000_20000101", "TCGA_AA_0001_20000101", "TCGA_AA_0002_20000101"]:
        write_patient(data_dir, pid, n_slices=6, tumour_slices=(3, 4), size=32)
    torch.manual_seed(0)
    save_checkpoint(str(tmp_path / "best_model.pt"), create_model("unet", 9, 1).state_dict(), RECORD)

    cfg = _cfg(model_path=str(tmp_path / "best_model.pt"), data_dir=str(data_dir),
               output_dir=str(tmp_path / "predictions"), figure_path=str(tmp_path / "dice.png"),
               results_json=str(tmp_path / "results.json"), masks_npz=str(tmp_path / "masks.npz"),
               masks_dir=str(tmp_path / "masks"))
    predict.run_inference(cfg)

    results = json.loads((tmp_path / "results.json").read_text())
    assert list(results["per_patient"]) == ["TCGA_AA_0002_20000101"]
    assert set(results["per_patient"]["TCGA_AA_0002_20000101"]) == {"dice", "dice_raw", "hd95_px"}
    assert results["meta"]["fold"] == 0 and results["meta"]["seed"] == 3 and results["meta"]["arch"] == "unet"
    assert results["meta"]["postprocess"] in ("none", "drop-small", "lcc")

    written = sorted(p.name for p in (tmp_path / "masks" / "TCGA_AA_0002_20000101").iterdir())
    assert written == [f"TCGA_AA_0002_20000101_{i}_pred.tif" for i in range(1, 7)]
    assert imread(str(tmp_path / "masks" / "TCGA_AA_0002_20000101" / written[0])).shape == (32, 32)
    with np.load(tmp_path / "masks.npz") as npz:
        assert npz["TCGA_AA_0002_20000101__pred"].shape == (4, 32, 32)  # grid: first and last slice dropped
