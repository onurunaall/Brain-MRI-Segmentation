"""
Shared pytest fixtures. Tests use small synthetic arrays only: no dataset download, no GPU, no training.
"""

import csv
import json
import os
import sys
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

# Make the repo's top-level modules (utils, compare, ...) importable from tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

HISTORY_HEADER = ["epoch", "train_loss", "val_loss", "val_dice", "lr", "seconds", "new_best"]


def write_run(runs_dir: str,
              arch: str,
              fold: Optional[int],
              seed: int,
              per_patient: Dict[str, Dict[str, float]],
              masks: Optional[Dict[str, np.ndarray]] = None,
              history: Optional[list] = None) -> str:
    """
    Create one run folder in the layout run_cv.sh produces.

    :param per_patient: patient_id -> {"dice", "dice_raw", "hd95_vox"}
    :param masks: npz arrays keyed '<pid>__flair' / '<pid>__gt' / '<pid>__pred' (written to test_masks.npz)
    :param history: rows for logs/history.csv, in HISTORY_HEADER order
    :return: run directory path
    """
    run_dir = os.path.join(runs_dir, arch, f"fold{fold}_seed{seed}")
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    with open(os.path.join(run_dir, "test_results.json"), "w") as fp:
        json.dump({"meta": {"arch": arch, "fold": fold, "seed": seed, "split": "test"},
                   "per_patient": per_patient}, fp)

    if masks is not None:
        np.savez_compressed(os.path.join(run_dir, "test_masks.npz"), **masks)

    if history is not None:
        with open(os.path.join(run_dir, "logs", "history.csv"), "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(HISTORY_HEADER)
            writer.writerows(history)

    return run_dir


def patient_volume(n_slices: int = 5, size: int = 16, tumour_slices=(1, 2, 3), radius: int = 3):
    """
    Synthetic (flair uint8, gt uint8) volume with a square 'tumour' on the given slices.
    Slice 2 gets the largest tumour, the outer tumour slices a smaller one.
    """
    flair = np.full((n_slices, size, size), 60, dtype=np.uint8)
    gt = np.zeros((n_slices, size, size), dtype=np.uint8)
    c = size // 2
    for s in tumour_slices:
        r = radius if s == 2 else max(1, radius - 2)
        gt[s, c - r: c + r, c - r: c + r] = 1
        flair[s, c - r: c + r, c - r: c + r] = 220
    return flair, gt


@pytest.fixture
def runs_dir(tmp_path):
    return str(tmp_path / "runs")
