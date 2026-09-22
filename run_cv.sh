#!/usr/bin/env bash
# 5-fold patient-level CV for every architecture. Safe to re-run: finished runs are skipped.
set -euo pipefail

ARCHS=${ARCHS:-"unet resunet swinunetr"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"0"}
EPOCHS=${EPOCHS:-100}
DATA_DIR=${DATA_DIR:-./kaggle_3m}
RUNS_DIR=${RUNS_DIR:-./runs}

for arch in $ARCHS; do
  for fold in $FOLDS; do
    for seed in $SEEDS; do
      run="$RUNS_DIR/$arch/fold${fold}_seed${seed}"

      if [ -f "$run/test_results.json" ]; then
        echo "[skip] $run already done"
        continue
      fi

      echo "[train] $run"
      python train.py --arch "$arch" --fold "$fold" --seed "$seed" --epochs "$EPOCHS" \
        --data-dir "$DATA_DIR" --checkpoint-dir "$run/checkpoints" --log-dir "$run/logs"

      echo "[test] $run"
      python predict.py --arch "$arch" --fold "$fold" --seed "$seed" --split test \
        --data-dir "$DATA_DIR" --model-path "$run/checkpoints/best_model.pt" \
        --results-json "$run/test_results.json" --figure-path "$run/dice_distribution.png" \
        --output-dir "$run/predictions" --skip-overlays
    done
  done
done

python aggregate.py --runs-dir "$RUNS_DIR" --baseline unet --csv "$RUNS_DIR/summary.csv"
