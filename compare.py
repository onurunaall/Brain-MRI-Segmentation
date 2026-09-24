"""
Comparison figures across architectures from finished CV runs (no GPU, no dataset needed).
Usage: python compare.py --runs-dir ./runs --baseline unet
Expects per run: <runs-dir>/<arch>/fold<k>_seed<s>/test_results.json, test_masks.npz (predict.py --masks-npz)
and logs/history.csv (train.py). Runs without history.csv fall back to their TensorBoard event file.
Writes figures to <runs-dir>/figures/ (or --out-dir).
"""

import argparse
import csv
import glob
import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np

# Overlay colours (RGB 0-1): true positive, false positive, false negative, ground truth
TP_COLOR = (0.0, 0.8, 0.0)
FP_COLOR = (1.0, 0.1, 0.1)
FN_COLOR = (0.1, 0.4, 1.0)
GT_COLOR = (1.0, 0.85, 0.0)
OVERLAY_ALPHA = 0.55
TIE_TOLERANCE = 1e-3  # Dice differences below this count as ties in the paired plot


class RunRecord:
    """One finished run: arch, fold, seed and the paths of its saved outputs."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        with open(os.path.join(run_dir, "test_results.json")) as fp:
            self.results = json.load(fp)
        meta = self.results["meta"]
        self.arch: str = meta["arch"]
        self.fold: Optional[int] = meta.get("fold")
        self.seed: int = meta.get("seed", 0)
        self.masks_path = os.path.join(run_dir, "test_masks.npz")
        self.history_path = os.path.join(run_dir, "logs", "history.csv")
        self.log_dir = os.path.join(run_dir, "logs")

    def per_patient(self, metric: str) -> Dict[str, float]:
        return {pid: vals[metric] for pid, vals in self.results["per_patient"].items()}


class ComparisonPlotter:
    """Builds metric, training-curve and segmentation comparison figures."""

    def __init__(self, runs_dir: str, out_dir: str, baseline: str, seed: Optional[int]) -> None:
        self.runs_dir = runs_dir
        self.out_dir = out_dir
        self.baseline = baseline
        self.seed = seed
        self.runs: List[RunRecord] = []

    # ------------------------------------------------------------------ loading

    def load(self) -> None:
        pattern = os.path.join(self.runs_dir, "*", "fold*_seed*", "test_results.json")
        self.runs = [RunRecord(os.path.dirname(p)) for p in sorted(glob.glob(pattern))]
        if not self.runs:
            raise SystemExit(f"No test_results.json found under {self.runs_dir}")
        order = {self.baseline: 0}
        self.archs = sorted({r.arch for r in self.runs}, key=lambda a: (order.get(a, 1), a))
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[Compare] {len(self.runs)} runs, architectures: {', '.join(self.archs)}")

    def _per_patient(self, arch: str, metric: str) -> Dict[str, float]:
        """Metric per patient averaged over seeds (each patient is tested in exactly one fold)."""
        collected: Dict[str, List[float]] = {}
        for run in self.runs:
            if run.arch == arch:
                for pid, value in run.per_patient(metric).items():
                    collected.setdefault(pid, []).append(value)
        return {pid: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
                for pid, v in collected.items()}

    def _mask_run(self, arch: str, fold: Optional[int]) -> Optional[RunRecord]:
        """Run whose masks represent (arch, fold): the requested seed, else the lowest available seed."""
        candidates = [r for r in self.runs
                      if r.arch == arch and r.fold == fold and os.path.exists(r.masks_path)]
        if self.seed is not None:
            candidates = [r for r in candidates if r.seed == self.seed]
        return min(candidates, key=lambda r: r.seed) if candidates else None

    # ------------------------------------------------------------ metric plots

    def plot_metric_distributions(self) -> None:
        """Box + individual points of per-patient Dice and HD95 for every architecture."""
        fig, axes = plt.subplots(1, 2, figsize=(6 + 2.5 * len(self.archs), 5))
        specs = [("dice", "Dice per patient (LCC)", axes[0]),
                 ("hd95_vox", "HD95 per patient (voxels, lower = better)", axes[1])]
        rng = np.random.default_rng(0)

        for metric, title, ax in specs:
            data, n_nan = [], []
            for arch in self.archs:
                values = np.array(list(self._per_patient(arch, metric).values()), dtype=float)
                n_nan.append(int(np.isnan(values).sum()))
                data.append(values[~np.isnan(values)])

            ax.boxplot(data, showfliers=False, widths=0.5)
            for i, values in enumerate(data, start=1):
                ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values, s=12, alpha=0.6, zorder=3)
                if len(values):
                    ax.text(i, 1.01, f"mean {np.mean(values):.3f}\nmedian {np.median(values):.3f}",
                            ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
            labels = [f"{a}\n(n={len(d)}" + (f", {k} undefined)" if k else ")") for a, d, k in zip(self.archs, data, n_nan)]
            ax.set_xticks(range(1, len(self.archs) + 1))
            ax.set_xticklabels(labels)
            ax.set_title(title, pad=30)
            ax.grid(axis="y", alpha=0.3)

        axes[0].set_ylim(-0.02, 1.02)
        fig.tight_layout()
        self._save(fig, "metric_distributions.png")

    def plot_paired_vs_baseline(self) -> None:
        """Per-patient Dice of each architecture against the baseline; points above the diagonal = better."""
        others = [a for a in self.archs if a != self.baseline]
        if self.baseline not in self.archs or not others:
            print(f"[Compare] skip paired plot: need baseline '{self.baseline}' and at least one other arch")
            return

        base = self._per_patient(self.baseline, "dice")
        fig, axes = plt.subplots(1, len(others), figsize=(5 * len(others), 5), squeeze=False)
        for ax, arch in zip(axes[0], others):
            other = self._per_patient(arch, "dice")
            common = sorted(set(base) & set(other))
            x = np.array([base[p] for p in common])
            y = np.array([other[p] for p in common])
            ax.scatter(x, y, s=16, alpha=0.7)
            ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{self.baseline} Dice")
            ax.set_ylabel(f"{arch} Dice")
            n_better = int(np.sum(y - x > TIE_TOLERANCE))
            n_worse = int(np.sum(x - y > TIE_TOLERANCE))
            n_tied = len(common) - n_better - n_worse
            ax.set_title(f"{arch} vs {self.baseline} ({len(common)} patients): better {n_better}, "
                         f"worse {n_worse}, tied {n_tied}\n(|Δ| < {TIE_TOLERANCE} = tie; mean Δ {np.mean(y - x):+.4f})",
                         fontsize=10)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        self._save(fig, "paired_dice_vs_baseline.png")

    def plot_fold_means(self) -> None:
        """Mean test Dice per fold for each architecture (averaged over seeds)."""
        folds = sorted({r.fold for r in self.runs if r.fold is not None})
        if not folds:
            print("[Compare] skip fold plot: runs have no fold index")
            return

        width = 0.8 / len(self.archs)
        fig, ax = plt.subplots(figsize=(2 + 1.5 * len(folds), 4.5))
        for i, arch in enumerate(self.archs):
            means = []
            for fold in folds:
                runs = [r for r in self.runs if r.arch == arch and r.fold == fold]
                means.append(np.mean([np.mean(list(r.per_patient("dice").values())) for r in runs]) if runs else np.nan)
            ax.bar(np.arange(len(folds)) + (i - (len(self.archs) - 1) / 2) * width, means, width, label=arch)
        ax.set_xticks(np.arange(len(folds)))
        ax.set_xticklabels([f"fold {f}" for f in folds])
        ax.set_ylabel("Mean test Dice")
        ax.set_ylim(0, 1)
        ax.set_title("Mean test Dice per fold (each fold = different test patients)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        self._save(fig, "fold_mean_dice.png")

    def plot_dice_vs_volume(self) -> None:
        """Per-patient Dice against ground-truth tumour size, to show whether small lesions are harder."""
        volumes: Dict[str, int] = {}
        for run in self.runs:
            if os.path.exists(run.masks_path):
                with np.load(run.masks_path) as npz:
                    for key in npz.files:
                        if key.endswith("__gt"):
                            volumes.setdefault(key[:-4], int(npz[key].sum()))
        if not volumes:
            print("[Compare] skip Dice-vs-volume plot: no test_masks.npz found")
            return

        fig, ax = plt.subplots(figsize=(7, 5))
        for arch in self.archs:
            dice = self._per_patient(arch, "dice")
            pids = [p for p in dice if p in volumes]
            ax.scatter([volumes[p] for p in pids], [dice[p] for p in pids], s=16, alpha=0.7, label=arch)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("Ground-truth tumour volume (voxels on the resized grid, not mm³)")
        ax.set_ylabel("Dice (LCC)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("Dice vs tumour size")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self._save(fig, "dice_vs_tumour_volume.png")

    # ---------------------------------------------------------- training curves

    @staticmethod
    def _read_history_csv(path: str) -> Dict[str, np.ndarray]:
        with open(path, newline="") as fp:
            rows = list(csv.DictReader(fp))
        return {key: np.array([float(r[key]) for r in rows]) for key in ("epoch", "train_loss", "val_loss", "val_dice")}

    @staticmethod
    def _read_history_tensorboard(log_dir: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Rebuild per-epoch history from the newest TensorBoard event file (runs trained before history.csv existed).
        val_loss / val_dice are exact; train_loss is the mean of the 10-step training-loss points in each epoch.
        """
        event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")), key=os.path.getmtime)
        if not event_files:
            return None
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        acc = EventAccumulator(event_files[-1], size_guidance={"scalars": 0})  # newest = last (re)started run
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if "val/dice" not in tags or "val/loss" not in tags:
            return None

        val_dice = acc.Scalars("val/dice")
        val_loss = acc.Scalars("val/loss")
        train_pts = acc.Scalars("train/loss") if "train/loss" in tags else []
        epoch_ends = [e.step for e in val_dice]

        train_loss, prev_end = [], -1
        for end in epoch_ends:
            pts = [e.value for e in train_pts if prev_end < e.step <= end]
            train_loss.append(np.mean(pts) if pts else np.nan)
            prev_end = end

        return {"epoch": np.arange(1, len(val_dice) + 1, dtype=float),
                "train_loss": np.array(train_loss, dtype=float),
                "val_loss": np.array([e.value for e in val_loss], dtype=float),
                "val_dice": np.array([e.value for e in val_dice], dtype=float)}

    def plot_training_curves(self) -> None:
        """Mean ± std (over folds and seeds) of train loss, val loss and val Dice per epoch."""
        histories: Dict[str, List[Dict[str, np.ndarray]]] = {}
        n_fallback = 0
        for run in self.runs:
            if os.path.exists(run.history_path):
                hist = self._read_history_csv(run.history_path)
            else:
                hist = self._read_history_tensorboard(run.log_dir)
                n_fallback += hist is not None
            if hist is not None and len(hist["epoch"]):
                histories.setdefault(run.arch, []).append(hist)

        if not histories:
            print("[Compare] skip training curves: no history.csv or TensorBoard logs found")
            return
        if n_fallback:
            print(f"[Compare] {n_fallback} run(s) had no history.csv; used TensorBoard "
                  f"(train loss there is approximate)")

        panels = [("train_loss", "Train loss (soft Dice, with augmentation)"),
                  ("val_loss", "Validation loss"),
                  ("val_dice", "Validation Dice (per patient, LCC)")]
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
        for ax, (key, title) in zip(axes, panels):
            for arch in self.archs:
                runs = histories.get(arch, [])
                if not runs:
                    continue
                n_epochs = max(len(h[key]) for h in runs)
                grid = np.full((len(runs), n_epochs), np.nan)
                for i, h in enumerate(runs):
                    grid[i, :len(h[key])] = h[key]
                with warnings.catch_warnings():  # epochs covered by no run -> NaN, not a warning
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean = np.nanmean(grid, axis=0)
                    std = np.nanstd(grid, axis=0)
                epochs = np.arange(1, n_epochs + 1)
                line, = ax.plot(epochs, mean, label=f"{arch} (n={len(runs)} runs)")
                ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.3)
        axes[2].set_ylim(0, 1)
        axes[0].legend()
        fig.suptitle("Training curves: line = mean over folds/seeds, band = ± 1 std")
        fig.tight_layout()
        self._save(fig, "training_curves.png")

    # ------------------------------------------------------ segmentation grids

    @staticmethod
    def _overlay(flair: np.ndarray, gt: np.ndarray, pred: Optional[np.ndarray]) -> np.ndarray:
        """RGB image: FLAIR with GT filled (pred=None) or with TP/FP/FN coloured."""
        base = np.repeat((flair.astype(np.float32) / 255.0)[..., None], 3, axis=2)
        gt = gt.astype(bool)
        if pred is None:
            layers = [(gt, GT_COLOR)]
        else:
            pred = pred.astype(bool)
            layers = [(pred & gt, TP_COLOR), (pred & ~gt, FP_COLOR), (~pred & gt, FN_COLOR)]
        for mask, color in layers:
            base[mask] = (1 - OVERLAY_ALPHA) * base[mask] + OVERLAY_ALPHA * np.array(color)
        return base

    @staticmethod
    def _slice_dice(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
        total = pred.sum() + gt.sum()
        return None if total == 0 else float(2.0 * np.logical_and(pred, gt).sum() / total)

    @staticmethod
    def _pick_slices(gt: np.ndarray, preds: List[np.ndarray]) -> List[Tuple[int, str]]:
        """Largest-tumour slice, the slice with the most model errors, and the smallest tumour (edge) slice."""
        areas = gt.reshape(gt.shape[0], -1).sum(axis=1)
        errors = sum(np.logical_xor(p, gt).reshape(gt.shape[0], -1).sum(axis=1) for p in preds)
        picks: List[Tuple[int, str]] = []

        def add(idx: int, why: str) -> None:
            if idx not in [p[0] for p in picks]:
                picks.append((int(idx), why))

        if areas.max() > 0:
            add(np.argmax(areas), "largest tumour")
        if np.max(errors) > 0:
            add(np.argmax(errors), "most errors")
        tumour_slices = np.nonzero(areas)[0]
        if len(tumour_slices):
            add(tumour_slices[np.argmin(areas[tumour_slices])], "tumour edge")
        if not picks:
            add(gt.shape[0] // 2, "middle (no tumour, no predictions)")
        return picks

    def _load_fold_masks(self, fold: Optional[int]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """FLAIR+GT per patient and predictions per arch for one fold."""
        flair_gt: Dict[str, np.ndarray] = {}
        preds: Dict[str, Dict[str, np.ndarray]] = {}
        for arch in self.archs:
            run = self._mask_run(arch, fold)
            if run is None:
                continue
            with np.load(run.masks_path) as npz:
                for key in npz.files:
                    pid, kind = key.rsplit("__", 1)
                    if kind == "pred":
                        preds.setdefault(pid, {})[arch] = npz[key]
                    elif pid not in flair_gt or kind not in flair_gt[pid]:
                        flair_gt.setdefault(pid, {})[kind] = npz[key]
        return flair_gt, preds

    def plot_segmentation_grids(self, max_patients: Optional[int]) -> None:
        """One figure per test patient: rows = picked slices, columns = FLAIR | GT | each architecture."""
        seg_dir = os.path.join(self.out_dir, "segmentation")
        os.makedirs(seg_dir, exist_ok=True)
        dice_by_arch = {arch: self._per_patient(arch, "dice") for arch in self.archs}
        overview: List[Tuple[str, float, np.ndarray, np.ndarray, Dict[str, np.ndarray], int]] = []
        n_written = 0

        for fold in sorted({r.fold for r in self.runs}, key=lambda f: (f is None, f)):
            flair_gt, preds = self._load_fold_masks(fold)
            for pid in sorted(flair_gt):
                if max_patients is not None and n_written >= max_patients:
                    break
                flair, gt = flair_gt[pid]["flair"], flair_gt[pid]["gt"]
                arch_preds = preds.get(pid, {})
                if any(p.shape != gt.shape for p in arch_preds.values()):
                    print(f"[Compare] skip {pid}: mask shapes differ between architectures (different --image-size?)")
                    continue
                archs = [a for a in self.archs if a in arch_preds]
                picks = self._pick_slices(gt, [arch_preds[a] for a in archs])
                self._draw_patient(pid, fold, flair, gt, arch_preds, archs, picks, dice_by_arch,
                                   os.path.join(seg_dir, f"{pid}.png"))
                n_written += 1

                mean_dice = np.nanmean([dice_by_arch[a].get(pid, np.nan) for a in archs]) if archs else np.nan
                overview.append((pid, float(mean_dice), flair, gt, arch_preds, picks[0][0]))

        if n_written == 0:
            print("[Compare] skip segmentation grids: no test_masks.npz found "
                  "(re-run predict.py with --masks-npz, or re-run run_cv.sh which back-fills them)")
            return
        print(f"[Compare] wrote {n_written} per-patient segmentation figures -> {seg_dir}")
        self._draw_overview(overview, dice_by_arch)

    def _draw_patient(self, pid: str, fold: Optional[int], flair: np.ndarray, gt: np.ndarray,
                      arch_preds: Dict[str, np.ndarray], archs: List[str], picks: List[Tuple[int, str]],
                      dice_by_arch: Dict[str, Dict[str, float]], path: str) -> None:
        n_cols = 2 + len(archs)
        fig, axes = plt.subplots(len(picks), n_cols, figsize=(2.6 * n_cols, 2.8 * len(picks) + 0.8), squeeze=False)

        for row, (s, why) in enumerate(picks):
            panels = [("FLAIR", self._overlay(flair[s], np.zeros_like(gt[s]), None)),
                      ("Ground truth", self._overlay(flair[s], gt[s], None))]
            for arch in archs:
                sd = self._slice_dice(arch_preds[arch][s].astype(bool), gt[s].astype(bool))
                sd_txt = "n/a (both empty)" if sd is None else f"{sd:.3f}"
                panels.append((f"{arch}\nslice Dice {sd_txt}", self._overlay(flair[s], gt[s], arch_preds[arch][s])))

            for col, (title, img) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0 or col >= 2:
                    ax.set_title(title, fontsize=8)
            axes[row, 0].set_ylabel(f"slice {s}\n({why})", fontsize=8)

        patient_scores = ", ".join(f"{a} {dice_by_arch[a].get(pid, float('nan')):.3f}" for a in archs)
        fold_txt = f"fold {fold}" if fold is not None else "no fold"
        fig.suptitle(f"{pid} ({fold_txt}) | patient Dice: {patient_scores}", fontsize=10)
        fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.savefig(path, dpi=110)
        plt.close(fig)

    def _draw_overview(self, overview, dice_by_arch: Dict[str, Dict[str, float]], n_each: int = 2) -> None:
        """Best / median / worst patients (by mean Dice over architectures) on their largest-tumour slice."""
        ranked = sorted([o for o in overview if not np.isnan(o[1])], key=lambda o: o[1])
        if not ranked:
            return
        mid = len(ranked) // 2
        chosen = [("worst", o) for o in ranked[:n_each]]
        chosen += [("median", o) for o in ranked[max(0, mid - n_each // 2): mid - n_each // 2 + n_each]]
        chosen += [("best", o) for o in ranked[-n_each:]]
        seen, rows = set(), []
        for label, o in chosen:
            if o[0] not in seen:
                seen.add(o[0])
                rows.append((label, o))

        n_cols = 2 + len(self.archs)
        fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.6 * n_cols, 2.7 * len(rows) + 0.8), squeeze=False)
        for r, (label, (pid, mean_dice, flair, gt, arch_preds, s)) in enumerate(rows):
            panels = [("FLAIR", self._overlay(flair[s], np.zeros_like(gt[s]), None)),
                      ("Ground truth", self._overlay(flair[s], gt[s], None))]
            for arch in self.archs:
                if arch in arch_preds:
                    panels.append((f"{arch}: {dice_by_arch[arch].get(pid, float('nan')):.3f}",
                                   self._overlay(flair[s], gt[s], arch_preds[arch][s])))
                else:
                    panels.append((f"{arch}: no masks", None))
            for c, (title, img) in enumerate(panels):
                ax = axes[r, c]
                if img is not None:
                    ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(title, fontsize=8)
            axes[r, 0].set_ylabel(f"{label}\n{pid}\nslice {s}", fontsize=7)

        fig.suptitle("Worst / median / best test patients (ranked by mean patient Dice over architectures)", fontsize=10)
        fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        self._save(fig, "segmentation_overview.png")

    # ------------------------------------------------------------------- misc

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = os.path.join(self.out_dir, name)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"[Compare] wrote {path}")

    @classmethod
    def cli(cls) -> None:
        parser = argparse.ArgumentParser(description="Comparison figures across architectures")
        parser.add_argument("--runs-dir", type=str, default="./runs")
        parser.add_argument("--out-dir", type=str, default=None, help="Default: <runs-dir>/figures")
        parser.add_argument("--baseline", type=str, default="unet")
        parser.add_argument("--seed", type=int, default=None,
                            help="Training seed whose masks are drawn (default: lowest available per arch/fold)")
        parser.add_argument("--max-patients", type=int, default=None,
                            help="Limit the number of per-patient segmentation figures (default: all)")
        args = parser.parse_args()

        plotter = cls(args.runs_dir, args.out_dir or os.path.join(args.runs_dir, "figures"), args.baseline, args.seed)
        plotter.load()
        plotter.plot_metric_distributions()
        plotter.plot_paired_vs_baseline()
        plotter.plot_fold_means()
        plotter.plot_dice_vs_volume()
        plotter.plot_training_curves()
        plotter.plot_segmentation_grids(args.max_patients)


def _legend_handles() -> List[Patch]:
    return [Patch(color=GT_COLOR, label="ground truth"),
            Patch(color=TP_COLOR, label="correct (TP)"),
            Patch(color=FP_COLOR, label="false positive"),
            Patch(color=FN_COLOR, label="missed (FN)")]


if __name__ == "__main__":
    ComparisonPlotter.cli()
