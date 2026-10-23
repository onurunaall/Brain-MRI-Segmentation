"""
Aggregate K-fold CV results into one comparison table.
Usage: python -m mriseg.aggregate --runs-dir ./runs --baseline unet
Expects: <runs-dir>/<arch>/fold<k>_seed<s>/test_results.json and .../logs/train_summary.json
--results-name summarizes another results file per run instead, e.g. test_results.fp16.engine.json (run_backend_eval.sh)
"""

import argparse
import csv
import glob
import json
import os
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy.stats import wilcoxon

HD95_KEY = "hd95_px"


def mean_per_patient(results: Iterable[Dict[str, Any]], metric: str) -> Dict[str, float]:
    """
    Metric per patient averaged over seeds (each patient is tested in exactly one fold).

    An undefined HD95 (NaN: the tumour was missed entirely, or a false alarm) counts as infinitely bad before
    averaging, so one seed that missed the tumour makes the patient's HD95 infinite instead of disappearing.

    :param results: Parsed test_results.json files of one architecture
    :param metric: Per-patient key, e.g. 'dice'
    :return: Dict patient_id -> mean metric
    """
    collected: Dict[str, List[float]] = {}
    for res in results:
        for pid, vals in res["per_patient"].items():
            value = vals[metric]
            if metric == HD95_KEY and np.isnan(value):
                value = float("inf")
            collected.setdefault(pid, []).append(value)
    return {pid: float(np.mean(values)) for pid, values in collected.items()}


class ResultsAggregator:
    """Collects per-patient test metrics across folds/seeds and compares architectures."""

    def __init__(self, runs_dir: str, results_name: str = "test_results.json") -> None:
        """
        :param runs_dir: Output folder of run_cv.sh
        :param results_name: Per-run results file to summarize
        """
        self.runs_dir = runs_dir
        self.results_name = results_name
        self.test_results: Dict[str, List[Dict[str, Any]]] = {}
        self.train_summaries: Dict[str, List[Dict[str, Any]]] = {}

    def load(self) -> None:
        pattern = os.path.join(self.runs_dir, "*", "fold*_seed*", self.results_name)
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
        return mean_per_patient(self.test_results[arch], metric)

    def fold_means(self, arch: str) -> List[float]:
        """Mean test Dice of each fold, first averaged over the fold's training seeds."""
        by_fold: Dict[Any, List[float]] = {}
        for res in self.test_results[arch]:
            run_mean = float(np.mean([v["dice"] for v in res["per_patient"].values()]))
            by_fold.setdefault(res["meta"].get("fold"), []).append(run_mean)
        return [float(np.mean(run_means)) for run_means in by_fold.values()]

    def summarize(self, baseline: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        base_dice = self.per_patient(baseline, "dice") if baseline in self.test_results else {}

        for arch in sorted(self.test_results):
            dice = self.per_patient(arch, "dice")
            dice_raw = self.per_patient(arch, "dice_raw")
            hd = self.per_patient(arch, HD95_KEY)
            values = np.array(list(dice.values()))
            hd_values = np.array(list(hd.values()))
            fold_means = self.fold_means(arch)

            row = {"arch": arch,
                   "n_runs": len(self.test_results[arch]),
                   "n_patients": len(dice),
                   "dice_mean": float(values.mean()),
                   "dice_std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                   "dice_median": float(np.median(values)),
                   "dice_min": float(values.min()),
                   "dice_raw_mean": float(np.mean(list(dice_raw.values()))),
                   "fold_mean_std": float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else float("nan"),
                   # Undefined HD95 (tumour missed entirely, or a false alarm) is infinite: the worst value
                   "hd95_px_median": float(np.median(hd_values)),
                   "hd95_undefined_count": int(np.isinf(hd_values).sum())}

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


def print_table(rows: List[Dict[str, Any]], baseline: str) -> None:
    # No "Δ": Windows writes redirected output in cp1252, which cannot encode it (the "±" it can)
    header = "| arch | runs | patients | Dice mean ± std | median | min | Dice (no post-processing) | fold-mean std | HD95 median (px) | delta vs " + baseline + " | Wilcoxon p |"
    print(header)
    print("|" + "---|" * 11)
    for r in rows:
        delta = f"{r['delta_vs_baseline']:+.4f}" if "delta_vs_baseline" in r else "-"
        pval = f"{r['wilcoxon_p']:.3g}" if "wilcoxon_p" in r else "-"
        print(f"| {r['arch']} | {r['n_runs']} | {r['n_patients']} | {r['dice_mean']:.4f} ± {r['dice_std']:.4f} | "
              f"{r['dice_median']:.4f} | {r['dice_min']:.4f} | {r['dice_raw_mean']:.4f} | {r['fold_mean_std']:.4f} | "
              f"{r['hd95_px_median']:.2f} | {delta} | {pval} |")

def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CV results")
    parser.add_argument("--runs-dir", type=str, default="./runs")
    parser.add_argument("--baseline", type=str, default="unet")
    parser.add_argument("--csv", type=str, default="./runs/summary.csv")
    parser.add_argument("--results-name", type=str, default="test_results.json",
                        help="Per-run results file to summarize (default: test_results.json)")
    args = parser.parse_args()

    agg = ResultsAggregator(args.runs_dir, args.results_name)
    agg.load()
    if not agg.test_results:
        raise SystemExit(f"No {args.results_name} found under {args.runs_dir}")
    rows = agg.summarize(args.baseline)
    print_table(rows, args.baseline)
    write_csv(rows, args.csv)
    print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
