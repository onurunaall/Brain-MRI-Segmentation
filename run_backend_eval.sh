#!/usr/bin/env bash
# Test-set evaluation of every CV fold model under other inference backends, to measure how much torch.compile,
# ONNX Runtime and TensorRT (fp32 / fp16) change the results compared with eager PyTorch (test_results.json).
# Run after run_cv.sh with the formats exported, e.g. EXPORT_FORMATS="fp32.onnx fp16.onnx fp32.engine fp16.engine".
# Writes <run>/test_results.<variant>.json and $RUNS_DIR/summary.<variant>.csv. Safe to re-run: finished runs are skipped.
set -euo pipefail

ARCHS=${ARCHS:-"unet resunet swinunetr"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
SEEDS=${SEEDS:-"0"}
VARIANTS=${VARIANTS:-"compile fp32.onnx fp16.onnx fp32.engine fp16.engine"}  # "compile" = model.pt + torch.compile
DATA_DIR=${DATA_DIR:-./kaggle_3m}
RUNS_DIR=${RUNS_DIR:-./runs}
MODELS_DIR=${MODELS_DIR:-./models}

for variant in $VARIANTS; do
  for arch in $ARCHS; do
    for fold in $FOLDS; do
      for seed in $SEEDS; do
        run="$RUNS_DIR/$arch/fold${fold}_seed${seed}"
        member="$MODELS_DIR/$arch/fold${fold}_seed${seed}"
        results="$run/test_results.$variant.json"

        if [ -f "$results" ]; then
          echo "[skip] $results already exists"
          continue
        fi

        if [ "$variant" = "compile" ]; then
          model_args=(--model-path "$member/model.pt" --compile default)
        else
          model_args=(--model-path "$member/model.$variant")
        fi

        echo "[$variant] $run"
        python predict.py --arch "$arch" --fold "$fold" --seed "$seed" --split test --data-dir "$DATA_DIR" \
          "${model_args[@]}" --results-json "$results" --figure-path "$run/dice_distribution.$variant.png" \
          --output-dir "$run/predictions" --skip-overlays
      done
    done
  done

  python aggregate.py --runs-dir "$RUNS_DIR" --baseline unet --results-name "test_results.$variant.json" \
    --csv "$RUNS_DIR/summary.$variant.csv"
done
