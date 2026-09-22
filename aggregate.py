"""
Aggregate K-fold CV results into one comparison table.
Usage: python aggregate.py --runs-dir ./runs --baseline unet
Expects: <runs-dir>/<arch>/fold<k>_seed<s>/test_results.json and .../logs/train_summary.json
"""

import argparse
import csv
import glob
import json
import os
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon


class ResultsAggregator:
    """Collects per-patient test metrics across folds/seeds and compares architectures."""

    def __init__(self, runs_dir: str) -> None:
        self.runs_dir = runs_dir
        self.test_results: Dict[str, List[Dict]] = {}
        self.train_summaries: Dict[str, List[Dict]] = {}

    def load(self) -> None:
        pattern = os.path.join(self.runs_dir, "*", "fold*_seed*", "test_results.json")
        for path in sorted(glob.glob(pattern)):
            with open(path) as fp:
                res = json.load(fp)
            arch = res["meta"]["arch"]
            self.test_results.setdefault(arch, []).append(res)

            summary_path = os.path.join(os.path.dirname(path), "logs", "train_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as fp:
                    self.train_summaries.setdefault(arch, []).append(json.load(fp))

    def per_patient(self, arch: str, metric: str) -> Dict[str, float]:
        """Metric per patient, averaged over seeds (each patient is tested in exactly one fold)."""
        collected: Dict[str, List[float]] = {}
        for res in self.test_results[arch]:
            for pid, vals in res["per_patient"].items():
                collected.setdefault(pid, []).append(vals[metric])
        averaged = {}
        for pid, values in collected.items():
            averaged[pid] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
        return averaged

    def fold_means(self, arch: str) -> List[float]:
        return [float(np.mean([v["dice"] for v in res["per_patient"].values()])) for res in self.test_results[arch]]

    def summarize(self, baseline: str) -> List[Dict]:
        rows = []
        base_dice = self.per_patient(baseline, "dice") if baseline in self.test_results else {}

        for arch in sorted(self.test_results):
            dice = self.per_patient(arch, "dice")
            dice_raw = self.per_patient(arch, "dice_raw")
            hd = self.per_patient(arch, "hd95_vox")
            values = np.array(list(dice.values()))
            hd_values = np.array(list(hd.values()))

            row = {"arch": arch,
                   "n_runs": len(self.test_results[arch]),
                   "n_patients": len(dice),
                   "dice_mean": float(values.mean()),
                   "dice_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                   "dice_median": float(np.median(values)),
                   "dice_min": float(values.min()),
                   "dice_raw_mean": float(np.mean(list(dice_raw.values()))),
                   "fold_mean_std": float(np.std(self.fold_means(arch), ddof=1)) if len(self.fold_means(arch)) > 1 else float("nan"),
                   "hd95_vox_median": float(np.nanmedian(hd_values)),
                   "hd95_nan_count": int(np.isnan(hd_values).sum())}

            summaries = self.train_summaries.get(arch, [])
            if summaries:
                row["n_params"] = summaries[0]["n_params"]
                row["sec_per_epoch_mean"] = float(np.mean([s["seconds_per_epoch"] for s in summaries]))
                mems = [s["peak_train_memory_mb"] for s in summaries if s["peak_train_memory_mb"] is not None]
                row["peak_train_mem_mb"] = float(np.max(mems)) if mems else None

            if base_dice and arch != baseline:
                common = sorted(set(dice) & set(base_dice))
                diffs = np.array([dice[p] - base_dice[p] for p in common])
                row["delta_vs_baseline"] = float(diffs.mean())
                row["wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if np.any(diffs != 0) else 1.0
            rows.append(row)
        return rows

    @staticmethod
    def print_table(rows: List[Dict], baseline: str) -> None:
        header = "| arch | runs | patients | Dice mean ± std | median | min | Dice (no LCC) | fold-mean std | HD95 median (vox) | Δ vs " + baseline + " | Wilcoxon p |"
        print(header)
        print("|" + "---|" * 11)
        for r in rows:
            delta = f"{r['delta_vs_baseline']:+.4f}" if "delta_vs_baseline" in r else "-"
            pval = f"{r['wilcoxon_p']:.3g}" if "wilcoxon_p" in r else "-"
            print(f"| {r['arch']} | {r['n_runs']} | {r['n_patients']} | {r['dice_mean']:.4f} ± {r['dice_std']:.4f} | "
                  f"{r['dice_median']:.4f} | {r['dice_min']:.4f} | {r['dice_raw_mean']:.4f} | {r['fold_mean_std']:.4f} | "
                  f"{r['hd95_vox_median']:.2f} | {delta} | {pval} |")

    @staticmethod
    def write_csv(rows: List[Dict], path: str) -> None:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def cli(cls) -> None:
        parser = argparse.ArgumentParser(description="Aggregate CV results")
        parser.add_argument("--runs-dir", type=str, default="./runs")
        parser.add_argument("--baseline", type=str, default="unet")
        parser.add_argument("--csv", type=str, default="./runs/summary.csv")
        args = parser.parse_args()

        agg = cls(args.runs_dir)
        agg.load()
        if not agg.test_results:
            raise SystemExit(f"No test_results.json found under {args.runs_dir}")
        rows = agg.summarize(args.baseline)
        cls.print_table(rows, args.baseline)
        cls.write_csv(rows, args.csv)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    ResultsAggregator.cli()
