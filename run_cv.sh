#!/usr/bin/env bash
# 5-fold patient-level CV for every architecture. Safe to re-run: finished runs are skipped, and a run whose training
# finished (logs/train_summary.json) but whose test failed is only tested again.
# Ends by bundling each architecture's fold models into $MODELS_DIR (export_models.py) for later prediction.
set -euo pipefail

ARCHS=${ARCHS:-"unet resunet swinunetr"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"0 1 2"}             # several training seeds per fold separate architecture differences from seed noise
EPOCHS=${EPOCHS:-100}
CONTEXT_SLICES=${CONTEXT_SLICES:-1} # neighbour slices per side as extra input channels (2.5D); 0 = plain 2D
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

      if [ ! -f "$run/logs/train_summary.json" ]; then
        echo "[train] $run"
        python -m mriseg.train --arch "$arch" --fold "$fold" --seed "$seed" --epochs "$EPOCHS" --compile "$COMPILE" \
          --context-slices "$CONTEXT_SLICES" \
          --data-dir "$DATA_DIR" --checkpoint-dir "$run/checkpoints" --log-dir "$run/logs"
      fi

      # The checkpoint knows its fold, seed and patients. --postprocess defaults to auto: chosen on this fold's
      # validation patients, never on the test patients
      echo "[test] $run"
      python -m mriseg.predict --model-path "$run/checkpoints/best_model.pt" --split test --data-dir "$DATA_DIR" \
        --results-json "$run/test_results.json" --figure-path "$run/dice_distribution.png" \
        --masks-npz "$run/test_masks.npz" --output-dir "$run/predictions" --skip-overlays
    done
  done
done

python -m mriseg.aggregate --runs-dir "$RUNS_DIR" --baseline unet --csv "$RUNS_DIR/summary.csv"
python -m mriseg.compare --runs-dir "$RUNS_DIR" --baseline unet --out-dir "$RUNS_DIR/figures"
# shellcheck disable=SC2086  # word splitting of the list variables is intended
python -m mriseg.export_models --runs-dir "$RUNS_DIR" --out-dir "$MODELS_DIR" --archs $ARCHS --formats $EXPORT_FORMATS --overwrite
