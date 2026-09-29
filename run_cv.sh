#!/usr/bin/env bash
# 5-fold patient-level CV for every architecture. Safe to re-run: finished runs are skipped.
# Ends by bundling each architecture's fold models into $MODELS_DIR (export_models.py) for later prediction.
set -euo pipefail

ARCHS=${ARCHS:-"unet resunet swinunetr"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"0"}
EPOCHS=${EPOCHS:-100}
DATA_DIR=${DATA_DIR:-./kaggle_3m}
RUNS_DIR=${RUNS_DIR:-./runs}
MODELS_DIR=${MODELS_DIR:-./models}
COMPILE=${COMPILE:-default}         # torch.compile mode for training, or "none"
EXPORT_FORMATS=${EXPORT_FORMATS:-}  # e.g. "fp32.onnx fp16.onnx fp32.engine fp16.engine" (engines need a GPU + TensorRT)

for arch in $ARCHS; do
  for fold in $FOLDS; do
    for seed in $SEEDS; do
      run="$RUNS_DIR/$arch/fold${fold}_seed${seed}"

      if [ -f "$run/test_results.json" ] && [ -f "$run/test_masks.npz" ]; then
        echo "[skip] $run already done"
        continue
      fi

      if [ -f "$run/test_results.json" ] && [ -f "$run/checkpoints/best_model.pt" ]; then
        # Finished before masks were saved: re-run inference only, no retraining
        echo "[backfill] $run: saving test masks for comparison figures"
      else
        echo "[train] $run"
        python train.py --arch "$arch" --fold "$fold" --seed "$seed" --epochs "$EPOCHS" --compile "$COMPILE" \
          --data-dir "$DATA_DIR" --checkpoint-dir "$run/checkpoints" --log-dir "$run/logs"
      fi

      echo "[test] $run"
      python predict.py --arch "$arch" --fold "$fold" --seed "$seed" --split test \
        --data-dir "$DATA_DIR" --model-path "$run/checkpoints/best_model.pt" \
        --results-json "$run/test_results.json" --figure-path "$run/dice_distribution.png" \
        --masks-npz "$run/test_masks.npz" --output-dir "$run/predictions" --skip-overlays
    done
  done
done

python aggregate.py --runs-dir "$RUNS_DIR" --baseline unet --csv "$RUNS_DIR/summary.csv"
python compare.py --runs-dir "$RUNS_DIR" --baseline unet --out-dir "$RUNS_DIR/figures"
# shellcheck disable=SC2086  # word splitting of the list variables is intended
python export_models.py --runs-dir "$RUNS_DIR" --out-dir "$MODELS_DIR" --archs $ARCHS --formats $EXPORT_FORMATS --overwrite
