"""Tests for aggregate.ResultsAggregator."""

import csv

import numpy as np
import pytest

from aggregate import ResultsAggregator
from conftest import write_run


def _metrics(dice: float) -> dict:
    return {"dice": dice, "dice_raw": dice - 0.05, "hd95_vox": 10.0 * (1 - dice)}


@pytest.fixture
def two_arch_runs(runs_dir):
    # unet: 2 folds x 2 patients. resunet: same patients, every Dice +0.1
    write_run(runs_dir, "unet", 0, 0, {"A": _metrics(0.5), "B": _metrics(0.7)})
    write_run(runs_dir, "unet", 1, 0, {"C": _metrics(0.6), "D": _metrics(0.8)})
    write_run(runs_dir, "resunet", 0, 0, {"A": _metrics(0.6), "B": _metrics(0.8)})
    write_run(runs_dir, "resunet", 1, 0, {"C": _metrics(0.7), "D": _metrics(0.9)})
    return runs_dir


def _rows_by_arch(runs_dir: str) -> dict:
    agg = ResultsAggregator(runs_dir)
    agg.load()
    return {r["arch"]: r for r in agg.summarize(baseline="unet")}


def test_summary_statistics(two_arch_runs):
    rows = _rows_by_arch(two_arch_runs)
    unet = rows["unet"]

    assert unet["n_runs"] == 2
    assert unet["n_patients"] == 4
    assert unet["dice_mean"] == pytest.approx(0.65)
    assert unet["dice_median"] == pytest.approx(0.65)
    assert unet["dice_min"] == pytest.approx(0.5)
    assert unet["dice_std"] == pytest.approx(np.std([0.5, 0.7, 0.6, 0.8], ddof=1))
    assert unet["dice_raw_mean"] == pytest.approx(0.60)
    # fold means 0.6 and 0.7
    assert unet["fold_mean_std"] == pytest.approx(np.std([0.6, 0.7], ddof=1))


def test_delta_vs_baseline(two_arch_runs):
    rows = _rows_by_arch(two_arch_runs)
    assert "delta_vs_baseline" not in rows["unet"]
    assert rows["resunet"]["delta_vs_baseline"] == pytest.approx(0.1)
    assert 0.0 < rows["resunet"]["wilcoxon_p"] <= 1.0


def test_identical_architectures_give_p_value_one(runs_dir):
    same = {"A": _metrics(0.5), "B": _metrics(0.7)}
    write_run(runs_dir, "unet", 0, 0, same)
    write_run(runs_dir, "resunet", 0, 0, same)
    rows = _rows_by_arch(runs_dir)
    assert rows["resunet"]["delta_vs_baseline"] == 0.0
    assert rows["resunet"]["wilcoxon_p"] == 1.0


def test_seeds_are_averaged_per_patient(runs_dir):
    write_run(runs_dir, "unet", 0, 0, {"A": _metrics(0.4)})
    write_run(runs_dir, "unet", 0, 1, {"A": _metrics(0.8)})
    agg = ResultsAggregator(runs_dir)
    agg.load()
    assert agg.per_patient("unet", "dice") == {"A": pytest.approx(0.6)}


def test_undefined_hd95_is_counted_not_averaged(runs_dir):
    write_run(runs_dir, "unet", 0, 0, {"A": {"dice": 0.0, "dice_raw": 0.0, "hd95_vox": float("nan")},
                                       "B": _metrics(0.8)})
    row = _rows_by_arch(runs_dir)["unet"]
    assert row["hd95_nan_count"] == 1
    assert row["hd95_vox_median"] == pytest.approx(2.0)


def test_write_csv(two_arch_runs, tmp_path):
    agg = ResultsAggregator(two_arch_runs)
    agg.load()
    rows = agg.summarize("unet")
    path = tmp_path / "summary.csv"
    ResultsAggregator.write_csv(rows, str(path))

    with open(path, newline="") as fp:
        read = list(csv.DictReader(fp))
    assert [r["arch"] for r in read] == [r["arch"] for r in rows]
    assert "delta_vs_baseline" in read[0]
