# Usage reference

The README covers what the project is and what came out of it. This page is the detailed reference for running it:
every script, its options and the files it writes.

- [Install](#install)
- [Train one model](#train-one-model)
- [Predict and score](#predict-and-score)
- [Cross-validation: run_cv.sh](#cross-validation-run_cvsh)
- [Figures: compare.py](#figures-comparepy)
- [Fold ensembles for new patients](#fold-ensembles-for-new-patients)
- [torch.compile](#torchcompile)
- [ONNX and TensorRT](#onnx-and-tensorrt)
- [Accuracy per backend: run_backend_eval.sh](#accuracy-per-backend-run_backend_evalsh)
- [Compute benchmark](#compute-benchmark)
- [Tests and type checks](#tests-and-type-checks)
- [Training options](#training-options)

## Install

Python 3.10 to 3.12.

```bash
uv sync                              # core dependencies, plus pytest and mypy (dev group)
uv sync --group onnx                 # + ONNX export and ONNX Runtime
uv sync --group onnx --group tensorrt  # + TensorRT 11.3 engines (NVIDIA GPU)
python get_data.py                   # Kaggle LGG dataset -> ./kaggle_3m (refuses to overwrite an existing folder)
```

The default PyTorch and TensorRT wheels are built for CUDA 13. On a machine whose driver only supports CUDA 12, install
the CUDA 12 builds instead: PyTorch from `https://download.pytorch.org/whl/cu126`, `tensorrt-cu12`, and
`onnxruntime-gpu` from ONNX Runtime's CUDA 12 index
(`--extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`).
`onnxruntime` and `onnxruntime-gpu` must not be installed together. If ONNX Runtime cannot load its CUDA provider,
`predict.py` says so and runs on the CPU; adding the `nvidia/*/lib` folders of the PyTorch install to
`LD_LIBRARY_PATH` usually fixes it.

## Train one model

```bash
python -m mriseg.train --arch unet --fold 0 --seed 0 --epochs 100 \
  --data-dir ./kaggle_3m --checkpoint-dir ./runs/unet/fold0_seed0/checkpoints --log-dir ./runs/unet/fold0_seed0/logs
```

- `--arch`: `unet`, `resunet` or `swinunetr`.
- `--fold k` holds out fold k of 5 (patient-level, split seed 42) as the test set; 10 of the remaining patients pick
  the checkpoint (`--n-validation`). Without `--fold` there is a single 100 / 10 split and no test set.
- The best checkpoint (`best_model.pt`) is saved whenever the validation Dice improves (per patient, thresholded at
  0.5, no post-processing). It is saved from the uncompiled model, and it holds the record of how it was trained:
  architecture, image size, context slices, fold, seed and the training / validation / test patients. `predict.py` and
  `export_models.py` read every setting from it, so they cannot get a setting wrong or score a model on its own
  training patients. Checkpoints that hold only weights (from early versions of `train.py`) are refused.
- Training samples slices so that half of each patient's weight follows tumour area and half is spread over all its
  slices: tumour-free slices are about 30% of the training samples, so the model learns to predict nothing on them.
- The logs folder gets `config.json`, `history.csv` (per-epoch losses, validation Dice, learning rate, time),
  `train_summary.json` and TensorBoard events.

## Predict and score

```bash
python -m mriseg.predict --model-path ./runs/unet/fold0_seed0/checkpoints/best_model.pt --split test \
  --data-dir ./kaggle_3m --results-json ./test_results.json --masks-npz ./test_masks.npz --skip-overlays
```

- `--split validation` / `test` scores the model's own validation / test patients, read from its checkpoint. The
  validation patients chose the checkpoint, so their score is optimistic. `--split all` predicts every patient in
  `--data-dir`; those patients need no `*_mask.tif` files, but none of them may be one of the model's training patients.
- The backend follows the file: `.pt` -> PyTorch (add `--compile` and/or `--amp` for torch.compile / fp16 autocast),
  `.onnx` -> ONNX Runtime, `.engine` / `.plan` -> TensorRT. Exported files take their settings from the `model.pt`
  next to them.
- Dice and HD95 are measured in original pixels: the prediction is mapped back onto the original slices and compared
  with the `*_mask.tif` files, every slice included (also the first and last, which the model never sees). HD95 counts
  one slice step as one pixel, because the TIFF files carry no pixel size or slice thickness.
- `--postprocess`: `none`, `drop-small` (drop 3D components smaller than 10% of the largest) or `lcc` (keep only the
  largest 3D component). The default `auto` picks the one with the best mean Dice on the model's validation patients
  and records the choice in the results JSON. `auto` does not work with `--split all`, which has no validation patients.
- `--masks-dir DIR` writes the predicted masks as `DIR/<patient>/<slice>_pred.tif` in the size of the input slices
  (must be outside `--data-dir`). `--masks-npz` stores FLAIR, ground truth, prediction and probabilities per patient on
  the preprocessed grid; `compare.py` draws its segmentation figures from it.

## Cross-validation: run_cv.sh

```bash
bash run_cv.sh
RUNS_DIR=./runs_v2 MODELS_DIR=./models_v2 COMPILE=none EXPORT_FORMATS="fp32.onnx fp16.engine" bash run_cv.sh
```

Trains and tests every architecture, fold and seed, then runs `aggregate.py`, `compare.py` and `export_models.py`.
Settings come from the environment:

| Variable | Default | Meaning |
|---|---|---|
| `ARCHS` | `unet resunet swinunetr` | Architectures |
| `FOLDS` | `0 1 2 3 4` | Test folds |
| `SEEDS` | `0 1 2` | Training seeds per fold, to tell architecture differences from seed noise |
| `EPOCHS` | `100` | Epochs per run |
| `CONTEXT_SLICES` | `1` | Neighbour slices per side as extra input channels (2.5D); 0 = plain 2D |
| `DATA_DIR`, `RUNS_DIR`, `MODELS_DIR` | `./kaggle_3m`, `./runs`, `./models` | Folders |
| `COMPILE` | `default` | torch.compile mode for training, or `none` |
| `EXPORT_FORMATS` | empty | e.g. `fp32.onnx fp16.onnx fp32.engine fp16.engine` |

Finished runs are skipped, and a run whose training finished but whose test did not is only tested again, so the
script can be restarted after an interruption. Train with changed settings into a new `RUNS_DIR`.

Per run, `runs/<arch>/fold<k>_seed<s>/` holds `test_results.json` (per-patient Dice after the chosen post-processing,
Dice without it, and HD95; the chosen method and its validation scores are in `meta`), `test_masks.npz`, the logs and
the checkpoint.

`summary.csv` (from `aggregate.py`) has one row per architecture: per-patient Dice averaged over seeds, its mean, SD,
median and minimum, the Dice without post-processing, the SD of the fold means, the HD95 median, parameters, time per
epoch, peak training memory, and the mean difference to the baseline with a Wilcoxon signed-rank p-value (paired by
patient). A patient whose tumour one seed missed entirely gets an infinite HD95 (`hd95_undefined_count`), so a miss is
never averaged away.

## Figures: compare.py

```bash
python -m mriseg.compare --runs-dir ./runs --baseline unet --out-dir ./runs/figures
```

Runs without a GPU or the dataset. Figures that need masks are skipped when `test_masks.npz` is missing.

| File | Shows |
|---|---|
| `metric_distributions.png` | Per-patient Dice and HD95 per architecture (box plot + every patient) |
| `paired_dice_vs_baseline.png` | Each patient's Dice for an architecture against the baseline; above the diagonal = better |
| `fold_mean_dice.png` | Mean test Dice per fold and architecture |
| `dice_vs_tumour_volume.png` | Per-patient Dice against ground-truth tumour size |
| `training_curves.png` | Train loss, validation loss and validation Dice per epoch (mean ± SD over runs) |
| `segmentation_overview.png` | Worst, median and best test patients for every architecture |
| `segmentation/<patient>.png` | Per patient: largest-tumour, most-errors and tumour-edge slices for every architecture |
| `<arch>/all_patients.png` | Every test patient of one architecture on its largest-tumour slice, lowest Dice first |
| `<arch>/cases/*.png` | The worst, median and best patients of one architecture (`--gallery-cases`, default 3) with the probability map |

Overlay colours: green = correct, red = false positive, blue = missed, yellow = ground truth. The per-patient figures
draw one seed per fold (`--seed`, default the lowest) and show that run's Dice.

## Fold ensembles for new patients

`export_models.py` (run at the end of `run_cv.sh`) copies the fold models of each architecture into one bundle:

```
models/<arch>/manifest.json             # arch, image size, context slices; per member: fold, seed, val Dice, best epoch (0-based), test Dice, post-processing
models/<arch>/fold<k>_seed<s>/model.pt  # plus model.fp32.onnx, model.fp16.engine, ... when exported
```

`predict.py --bundle-dir` averages the probabilities of all members:

```bash
python -m mriseg.predict --bundle-dir ./models/unet --split all --postprocess lcc --data-dir ./new_patients --masks-dir ./new_masks
python -m mriseg.predict --bundle-dir ./models/unet --bundle-file model.fp16.engine --split all --postprocess lcc --data-dir ./new_patients
```

Pass the post-processing explicitly (the method most members chose, see `manifest.json`). The ensemble is refused on
`--split validation/test`: every CV patient was a training patient of the other folds' models, so any score there would
be inflated. The cross-validation results estimate the performance of a single fold model; the ensemble has no held-out
estimate of its own.

## torch.compile

`--compile MODE` (train.py, predict.py, benchmark.py) wraps the model with `torch.compile`; `none` runs eager PyTorch.
Training compiles by default; prediction and benchmarking run eagerly by default.

- Same model, not bit-identical numbers: Inductor fuses operations and picks other kernels, so outputs differ from
  eager mode at the level of floating-point rounding (measured on CPU: max |Δp| 6e-8 for U-Net and 8e-7 for
  Swin-UNETR in fp32, no pixel changed class at the 0.5 threshold).
- Training runs diverge slightly, like two runs with non-deterministic cuDNN kernels. Compare architectures with the
  same compile mode.
- The first batches are slow while compiling; switching to eval mode and a smaller last batch each trigger a recompile.

## ONNX and TensorRT

```bash
python -m mriseg.export_models --runs-dir ./runs --out-dir ./models --formats fp32.onnx fp16.onnx fp32.engine fp16.engine --overwrite
```

- ONNX: PyTorch's `torch.export`-based exporter, opset 20, dynamic batch, fixed H × W. `fp16.onnx` is an fp16 copy of
  the model (weights, activations, input and output). The exported models include the sigmoid.
- TensorRT 11.3: networks are always strongly typed, so the engine precision comes from the ONNX file (`fp16.engine`
  is built from `fp16.onnx`). `fp32.engine` lets convolutions use TF32 (`--no-tf32` disables it). Engines accept batch
  sizes 1 to `--max-batch` (default 32; larger batches are split) and only run with the TensorRT version and GPU model
  they were built with.
- After each export, the largest probability difference to the fp32 PyTorch model on a random batch is printed. That
  is a quick check that the export works, not an accuracy measurement; see the next section for that.

## Accuracy per backend: run_backend_eval.sh

```bash
bash run_backend_eval.sh
VARIANTS="amp fp16.engine" SEEDS=0 bash run_backend_eval.sh
```

Scores the test patients of every fold model again with another backend and writes
`<run>/test_results.<variant>.json` and `runs/summary.<variant>.csv`, to compare with eager PyTorch. Variants:
`compile` (torch.compile), `amp` (fp16 autocast), `fp32.onnx`, `fp16.onnx`, `fp32.engine`, `fp16.engine`. Run it after
`run_cv.sh` with the formats exported. `ARCHS`, `FOLDS`, `SEEDS`, `DATA_DIR`, `RUNS_DIR` and `MODELS_DIR` work as in
`run_cv.sh`.

## Compute benchmark

```bash
python -m mriseg.benchmark --backends pytorch fp32.onnx fp16.onnx fp32.engine fp16.engine --out ./runs/compute_benchmark.json
python -m mriseg.benchmark --compile default --out ./runs/compute_benchmark.compile.json
```

Parameters, FLOPs, latency / throughput at batch sizes 1 and 16, and peak inference memory per architecture and
backend. Exports are made from untrained weights (latency does not depend on the weight values). Run every
architecture in one call on the same GPU so the numbers are comparable.

## Tests and type checks

```bash
uv sync --group onnx
pytest        # synthetic data only, no dataset needed; about 2 minutes on a laptop CPU
mypy .        # every function is fully type-annotated (see [tool.mypy] in pyproject.toml)
```

ONNX tests are skipped without the `onnx` group, and the TensorRT test without an NVIDIA GPU and TensorRT.

## Training options

`train.py` saves all of these to `<log-dir>/config.json`.

| Option | Default | Meaning |
|---|---|---|
| `--batch-size` | 16 | Training batch size |
| `--epochs` | 100 | Training epochs |
| `--lr` | 1e-4 | Initial Adam learning rate (cosine decay) |
| `--image-size` | 256 | Side of the square grid after preprocessing |
| `--context-slices` | 1 | Neighbour slices per side as extra channels (2.5D); 0 = plain 2D |
| `--aug-scale` | 0.05 | Random scale range ±5% |
| `--aug-angle` | 15 | Random rotation ±15° |
| `--aug-brightness` | 0.2 | Per-sequence intensity factor in [0.8, 1.2] |
| `--aug-gamma` | 0.4 | Per-sequence gamma in [1/1.4, 1.4] |
| `--n-validation` | 10 | Validation patients that pick the checkpoint |
| `--split-seed` | 42 | Patient split seed; keep it fixed across runs you compare |
| `--compile` | default | torch.compile mode, or `none` |
