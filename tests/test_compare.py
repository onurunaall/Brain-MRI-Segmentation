"""Tests for compare.py (comparison figures) and predict._save_masks (the masks it reads)."""

import os

import numpy as np
import pytest

import compare
from compare import ComparisonPlotter
from conftest import HISTORY_HEADER, patient_volume, write_run


# ------------------------------------------------------------ fixtures

def _masks_for(pids, pred_fn):
    """npz dict for the given patients; pred_fn(gt) -> predicted mask."""
    arrays = {}
    for i, pid in enumerate(pids):
        flair, gt = patient_volume(radius=3 + i)
        arrays[f"{pid}__flair"] = flair
        arrays[f"{pid}__gt"] = gt
        arrays[f"{pid}__pred"] = pred_fn(gt).astype(np.uint8)
    return arrays


def _shifted(gt):
    return np.roll(gt, 1, axis=2)


@pytest.fixture
def full_runs(runs_dir):
    """2 architectures x 2 folds with metrics, masks and history.csv."""
    history = [[e, 1.0 / e, 1.2 / e, 0.2 * e, 1e-4, 3.0, 1] for e in range(1, 4)]
    folds = {0: ["P0", "P1"], 1: ["P2", "P3"]}
    for fold, pids in folds.items():
        write_run(runs_dir, "unet", fold, 0,
                  {p: {"dice": 0.6, "dice_raw": 0.55, "hd95_vox": 5.0} for p in pids},
                  masks=_masks_for(pids, _shifted), history=history)
        write_run(runs_dir, "resunet", fold, 0,
                  {p: {"dice": 0.9, "dice_raw": 0.85, "hd95_vox": 1.0} for p in pids},
                  masks=_masks_for(pids, lambda gt: gt), history=history)
    return runs_dir


def _plotter(runs_dir, tmp_path, seed=None) -> ComparisonPlotter:
    plotter = ComparisonPlotter(runs_dir, str(tmp_path / "figures"), baseline="unet", seed=seed)
    plotter.load()
    return plotter


# --------------------------------------------------------- pure helpers

class TestOverlay:

    def test_prediction_colours_tp_fp_fn(self):
        flair = np.zeros((1, 3), dtype=np.uint8)
        gt = np.array([[1, 1, 0]])
        pred = np.array([[1, 0, 1]])  # TP, FN, FP

        rgb = ComparisonPlotter._overlay(flair, gt, pred)
        a = compare.OVERLAY_ALPHA

        np.testing.assert_allclose(rgb[0, 0], a * np.array(compare.TP_COLOR))
        np.testing.assert_allclose(rgb[0, 1], a * np.array(compare.FN_COLOR))
        np.testing.assert_allclose(rgb[0, 2], a * np.array(compare.FP_COLOR))

    def test_ground_truth_only_and_background_untouched(self):
        flair = np.array([[255, 255]], dtype=np.uint8)
        gt = np.array([[1, 0]])
        rgb = ComparisonPlotter._overlay(flair, gt, None)
        a = compare.OVERLAY_ALPHA

        np.testing.assert_allclose(rgb[0, 0], (1 - a) + a * np.array(compare.GT_COLOR))
        np.testing.assert_allclose(rgb[0, 1], [1.0, 1.0, 1.0])


class TestSliceDice:

    def test_values(self):
        gt = np.array([[1, 1, 0, 0]], dtype=bool)
        assert ComparisonPlotter._slice_dice(gt, gt) == pytest.approx(1.0)
        assert ComparisonPlotter._slice_dice(np.array([[0, 1, 1, 0]], dtype=bool), gt) == pytest.approx(0.5)
        assert ComparisonPlotter._slice_dice(np.zeros_like(gt), gt) == 0.0

    def test_both_empty_is_undefined(self):
        empty = np.zeros((2, 2), dtype=bool)
        assert ComparisonPlotter._slice_dice(empty, empty) is None


class TestPickSlices:

    def test_largest_most_errors_and_edge(self):
        _, gt = patient_volume(n_slices=6, tumour_slices=(1, 2, 3))  # slice 2 largest, 1 and 3 smaller
        pred = gt.copy()
        pred[4, 0:4, 0:4] = 1  # 16 false positives on a tumour-free slice

        picks = ComparisonPlotter._pick_slices(gt, [pred])

        assert picks[0] == (2, "largest tumour")
        assert picks[1] == (4, "most errors")
        assert picks[2] == (1, "tumour edge")  # smallest tumour slice (first on ties)

    def test_no_duplicate_slices(self):
        gt = np.zeros((3, 8, 8), dtype=np.uint8)
        gt[1, 2:6, 2:6] = 1
        picks = ComparisonPlotter._pick_slices(gt, [np.zeros_like(gt)])  # errors also peak on slice 1
        assert [p[0] for p in picks] == [1]

    def test_no_tumour_and_no_predictions_uses_middle_slice(self):
        empty = np.zeros((5, 4, 4), dtype=np.uint8)
        picks = ComparisonPlotter._pick_slices(empty, [empty.copy()])
        assert picks == [(2, "middle (no tumour, no predictions)")]


# ------------------------------------------------------ history readers

def test_read_history_csv(runs_dir):
    run_dir = write_run(runs_dir, "unet", 0, 0, {}, history=[[1, 0.9, 0.95, 0.1, 1e-4, 2.0, 1],
                                                                [2, 0.8, 0.85, 0.3, 5e-5, 2.1, 1]])
    hist = ComparisonPlotter._read_history_csv(os.path.join(run_dir, "logs", "history.csv"))

    np.testing.assert_allclose(hist["epoch"], [1, 2])
    np.testing.assert_allclose(hist["train_loss"], [0.9, 0.8])
    np.testing.assert_allclose(hist["val_loss"], [0.95, 0.85])
    np.testing.assert_allclose(hist["val_dice"], [0.1, 0.3])


def test_history_header_matches_train_py():
    """compare.py reads the columns train.py writes; guard against the two drifting apart."""
    with open(os.path.join(os.path.dirname(compare.__file__), "train.py")) as fp:
        source = fp.read()
    assert str(HISTORY_HEADER).replace("'", '"') in source


def _write_tb_events(log_dir: str, val_points, train_points) -> None:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)
    for step, value in train_points:
        writer.add_scalar("train/loss", value, step)
    for step, loss, dice in val_points:
        writer.add_scalar("val/loss", loss, step)
        writer.add_scalar("val/dice", dice, step)
    writer.close()


def test_read_history_tensorboard(tmp_path):
    # Two epochs ending at steps 20 and 40; train loss logged every 10 steps as in train.py
    log_dir = str(tmp_path / "logs")
    _write_tb_events(log_dir,
                     val_points=[(20, 0.9, 0.2), (40, 0.7, 0.5)],
                     train_points=[(9, 1.0), (19, 0.8), (29, 0.6), (39, 0.4)])

    hist = ComparisonPlotter._read_history_tensorboard(log_dir)

    np.testing.assert_allclose(hist["epoch"], [1, 2])
    np.testing.assert_allclose(hist["val_loss"], [0.9, 0.7], rtol=1e-6)
    np.testing.assert_allclose(hist["val_dice"], [0.2, 0.5], rtol=1e-6)
    np.testing.assert_allclose(hist["train_loss"], [0.9, 0.5], rtol=1e-6)  # mean of points inside each epoch


def test_read_history_tensorboard_uses_newest_event_file(tmp_path):
    """A crashed-then-restarted run leaves two event files; only the newest (complete) one counts."""
    log_dir = str(tmp_path / "logs")
    _write_tb_events(log_dir, val_points=[(10, 0.99, 0.01)], train_points=[])
    old_file = [f for f in os.listdir(log_dir) if f.startswith("events")][0]
    os.utime(os.path.join(log_dir, old_file), (1, 1))  # make it clearly older
    _write_tb_events(log_dir, val_points=[(10, 0.5, 0.6), (20, 0.4, 0.7)], train_points=[])

    hist = ComparisonPlotter._read_history_tensorboard(log_dir)
    np.testing.assert_allclose(hist["val_dice"], [0.6, 0.7], rtol=1e-6)


def test_read_history_tensorboard_without_events_returns_none(tmp_path):
    assert ComparisonPlotter._read_history_tensorboard(str(tmp_path)) is None


# ------------------------------------------------------------- plotting

def test_all_figures_are_written(full_runs, tmp_path):
    plotter = _plotter(full_runs, tmp_path)
    plotter.plot_metric_distributions()
    plotter.plot_paired_vs_baseline()
    plotter.plot_fold_means()
    plotter.plot_dice_vs_volume()
    plotter.plot_training_curves()
    plotter.plot_segmentation_grids(max_patients=None)

    out = tmp_path / "figures"
    for name in ["metric_distributions.png", "paired_dice_vs_baseline.png", "fold_mean_dice.png",
                 "dice_vs_tumour_volume.png", "training_curves.png", "segmentation_overview.png"]:
        assert (out / name).stat().st_size > 0, name
    assert sorted(os.listdir(out / "segmentation")) == [f"P{i}.png" for i in range(4)]


def test_baseline_is_ordered_first(full_runs, tmp_path):
    assert _plotter(full_runs, tmp_path).archs == ["unet", "resunet"]


def test_max_patients_limits_segmentation_figures(full_runs, tmp_path):
    _plotter(full_runs, tmp_path).plot_segmentation_grids(max_patients=1)
    assert len(os.listdir(tmp_path / "figures" / "segmentation")) == 1


def test_missing_masks_skip_segmentation_without_error(runs_dir, tmp_path, capsys):
    write_run(runs_dir, "unet", 0, 0, {"A": {"dice": 0.5, "dice_raw": 0.5, "hd95_vox": 1.0}})
    plotter = _plotter(runs_dir, tmp_path)
    plotter.plot_segmentation_grids(max_patients=None)
    plotter.plot_dice_vs_volume()
    assert "skip segmentation grids" in capsys.readouterr().out
    assert not (tmp_path / "figures" / "segmentation_overview.png").exists()


def test_paired_plot_counts_ties_within_tolerance(runs_dir, tmp_path, monkeypatch):
    base = {"A": 0.5, "B": 0.5, "C": 1e-7, "D": 0.5}
    other = {"A": 0.6, "B": 0.4, "C": 2e-7, "D": 0.5}  # better, worse, tie (noise), tie
    for arch, scores in (("unet", base), ("resunet", other)):
        write_run(runs_dir, arch, 0, 0, {p: {"dice": d, "dice_raw": d, "hd95_vox": 1.0} for p, d in scores.items()})

    saved = {}
    monkeypatch.setattr(ComparisonPlotter, "_save", lambda self, fig, name: saved.setdefault(name, fig))
    _plotter(runs_dir, tmp_path).plot_paired_vs_baseline()

    title = saved["paired_dice_vs_baseline.png"].axes[0].get_title()
    assert "better 1, worse 1, tied 2" in title


def test_seed_selection_for_masks(runs_dir, tmp_path):
    for seed in (0, 1):
        write_run(runs_dir, "unet", 0, seed, {"P0": {"dice": 0.5, "dice_raw": 0.5, "hd95_vox": 1.0}},
                  masks=_masks_for(["P0"], lambda gt: gt))

    assert _plotter(runs_dir, tmp_path)._mask_run("unet", 0).seed == 0            # lowest by default
    assert _plotter(runs_dir, tmp_path, seed=1)._mask_run("unet", 0).seed == 1    # explicit
    assert _plotter(runs_dir, tmp_path, seed=7)._mask_run("unet", 0) is None      # unavailable


def test_no_runs_exits(tmp_path):
    with pytest.raises(SystemExit):
        ComparisonPlotter(str(tmp_path), str(tmp_path / "f"), "unet", None).load()


# ------------------------------------------------------ predict._save_masks

def test_save_masks_round_trip(tmp_path):
    from predict import _save_masks

    rng = np.random.default_rng(0)
    vol_in = rng.normal(size=(3, 3, 8, 8)).astype(np.float32)  # (Z, C, H, W), channel 1 = FLAIR
    gt = np.zeros((3, 1, 8, 8), dtype=np.float32)
    gt[1, 0, 2:5, 2:5] = 1.0
    pred = gt.astype(bool)  # LCC output is boolean

    path = tmp_path / "nested" / "masks.npz"
    _save_masks(str(path), {"P0": (vol_in, pred, gt)})

    with np.load(path) as npz:
        assert sorted(npz.files) == ["P0__flair", "P0__gt", "P0__pred"]
        flair = npz["P0__flair"]
        assert flair.dtype == np.uint8 and flair.shape == (3, 8, 8)
        assert flair.min() == 0 and flair.max() == 255
        # min-max scaling preserves the ordering of FLAIR intensities
        assert np.argmax(flair) == np.argmax(vol_in[:, 1])
        np.testing.assert_array_equal(npz["P0__gt"], gt[:, 0].astype(np.uint8))
        np.testing.assert_array_equal(npz["P0__pred"], pred[:, 0].astype(np.uint8))


def test_save_masks_constant_flair(tmp_path):
    from predict import _save_masks

    vol_in = np.ones((2, 3, 4, 4), dtype=np.float32)
    zeros = np.zeros((2, 1, 4, 4))
    _save_masks(str(tmp_path / "m.npz"), {"P": (vol_in, zeros.astype(int), zeros)})
    with np.load(tmp_path / "m.npz") as npz:
        assert np.all(npz["P__flair"] == 0)
