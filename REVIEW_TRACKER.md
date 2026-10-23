# Code review tracker

Second review: working tree of branch `claude/segmentation-quality-fixes` (commit `7cf7f40` + uncommitted move into
`mriseg/`), 2026-10-22. Full explanations, proofs and fixes: [REVIEW.md](REVIEW.md).
Fix round: 2026-10-22, same day. Nothing is committed.

Status words: **Fixed** = changed in the code and covered by a test. **Set aside** = not changed; see the section
"Set aside for a second reviewer". Every fix was checked with the full test suite (163 passed, 2 TensorRT tests skipped
without a GPU; 1 test deselected because of a path-length problem in my temporary Python install), `mypy` (clean) and
an end-to-end run of train → predict → aggregate → compare → export → predict-from-ONNX on synthetic data.

## Issues

| ID | Danger | Issue | Status | Where the fix is | Test that guards it |
|---|---|---|---|---|---|
| H1 | High | `--masks-dir` masks shifted 1 px down-right | **Fixed**: `"nearest-exact"` both ways | `mriseg/utils.py` `resize_volume`, `restore_mask_geometry` | `test_utils.py::TestRestoreMaskGeometry` (exact equality, 3 sizes) |
| H2 | High | Training labels 0.5 px off the image (`RandomScale`) | **Fixed**: `NEAREST_EXACT` | `mriseg/augmentations.py` `RandomScale` | `test_augmentations.py::test_random_scale_keeps_the_mask_on_the_image` |
| M1 | Medium | README score optimistic and outdated | Not requested; unchanged | — | — |
| M2 | Medium | Normalization statistics include background | **Fixed**: head voxels only | `mriseg/utils.py` `head_mask`, `normalize_intensity` | `test_utils.py::TestNormalizeIntensity` |
| M3 | Medium | Dice measured on the resized grid | **Fixed**: reported Dice / HD95 in original pixels | `mriseg/evaluation.py` `score_patient`; `mriseg/dataset.py` `original_masks` | `test_evaluation.py::test_score_patient_*`, `test_predict.py::test_full_run_on_the_test_patients` |
| L1 | Low | HD95 seed average hid complete misses | **Fixed**: a miss counts as infinite before averaging | `mriseg/aggregate.py` `mean_per_patient` | `test_aggregate.py::test_a_seed_that_missed_the_tumour_is_not_averaged_away` |
| L2 | Low | Figures showed seed-averaged Dice next to one seed's masks | **Fixed**: the drawn run's Dice | `mriseg/compare.py` `_load_fold_masks` | `test_compare.py::test_segmentation_figures_show_the_dice_of_the_run_drawn` |
| L3 | Low | HD95 mixes pixel and slice units | **Set aside** (see below) | — | — |
| L4 | Low | First/last slices removed from the ground truth | **Fixed** via M3: scored against all original slices | `mriseg/evaluation.py` | `test_evaluation.py::test_score_patient_counts_tumour_on_slices_the_model_never_saw` |
| L5 | Low | Same-name patient folders silently replaced each other | **Fixed** (safe either way): identical copy ignored, different copies raise | `mriseg/dataset.py` `find_patients` | `test_dataset.py::test_identical_copy_*`, `test_different_folders_*` |
| L6 | Low | Geometry tests too loose | **Fixed**: exact tests that fail on the old code (checked) | `tests/test_utils.py`, `tests/test_augmentations.py` | — |
| L7 | Low | `get_data.py` deleted `./kaggle_3m` | **Fixed**: refuses if it exists; `main()` guard | `get_data.py` | — |
| L8 | Low | `fix_model.py` obsolete | **Fixed**: deleted (deletion is staged in git) | — | — |
| L9 | Low | README project structure outdated | **Fixed**, plus README text for every change below | `README.md` | — |
| L10 | Low | `scikit-image` listed twice | **Fixed** | `pyproject.toml` | — |
| L11 | Low | TensorBoard flush after every number | **Set aside** (see below) | — | — |

## Simplification items

| ID | Item | Status | Where |
|---|---|---|---|
| S1 | Settings in 3 places → checkpoint is the single source of truth | **Fixed**. The checkpoint record now also holds `seed` and `test_patients`. `predict.py` lost 8 flags (`--arch`, `--fold`, `--n-folds`, `--n-validation`, `--split-seed`, `--seed`, `--context-slices`, `--image-size`) and the 3 functions that cross-checked them; exported files read the `model.pt` next to them; `export_models.py` no longer reads `config.json` and lost `--image-size`; `hubconf.py` builds the model from the record | `mriseg/inference.py` `load_trained_model`, `model_record`; `mriseg/predict.py`; `mriseg/export_models.py`; `hubconf.py`; `run_cv.sh`; `run_backend_eval.sh` |
| S2 | Remove old-format compatibility code | **Fixed**: weights-only checkpoints are refused with a clear message; removed `strip_compile_prefix`, the `hd95_vox` warning, the TensorBoard history fallback, the manifest `context_slices` default, the `run_cv.sh` backfill branch, `fix_model.py` | several |
| S3 | Dataset constructor did 7 jobs | **Fixed**: `find_patients`, `split_patients`, `slice_files` (regex, pairs by slice number), `read_mask`, `preprocess`, `slice_sampling_weights` + a dataset that takes a patient list | `mriseg/dataset.py` |
| S4 | Training loop with `if phase` branches | **Fixed**: `train_one_epoch`, `validate`, `_log_overlays` | `mriseg/train.py` |
| S5 | `predict.py` did 9 jobs | **Fixed**: `predict_volumes`, `best_postprocess` (pure, testable), `_patients`, `_save_*`; chart saved directly with `savefig` | `mriseg/predict.py` |

## Set aside for a second reviewer

These are the points where I think my review was wrong, overstated, unproven, or where a fix needs a human decision.

| # | Item | Why it is set aside | What to check |
|---|---|---|---|
| A1 | **L3** HD95 mixes units | Not a bug: the code documents it, and the TIFF files have no pixel size or slice thickness. Any "fix" changes what the metric means (e.g. a 2D HD95). | Decide whether to keep 3D HD95 (documented) or switch to a per-slice 2D HD95. |
| A2 | **L11** TensorBoard flush | **I overstated it.** `log_scalar` runs about every 10 training steps, so the flush cost is negligible, and flushing keeps TensorBoard live. Not changed. | Probably close it. |
| A3 | **L5** premise | I could not confirm that the Kaggle download contains two copies of the patient folders (`kaggle_3m/` and `lgg-mri-segmentation/kaggle_3m/`). The fix is safe in both cases. | Look at the real `./kaggle_3m` folder tree. |
| A4 | **H2** size of the effect | The half-pixel shift is proven; how much Dice the fix gains is **not measured** (needs a full CV run). My "1 to 3 Dice points" in REVIEW.md is an estimate. | Retrain with `run_cv.sh` into a new `RUNS_DIR` and compare `summary.csv`. |
| A5 | **M2** effect and one judgement call | The background dependence is proven on synthetic data; the effect on Dice is not measured. I also lowered the clipping percentile from 10 to 1, because the old 10 only made sense with the background included (over head voxels, 10 would clip dark tissue such as CSF in FLAIR). That reasoning is mine, not tested on real scans. | Same CV comparison as A4; maybe try 0.5 / 1 / 2. |
| A6 | **M3** changes the meaning of `dice` | `test_results.json` `dice` / `dice_raw` / `hd95_px` are now in original pixels. Old result files hold grid values, so old and new runs must not be aggregated together. Training's validation Dice (epoch choice) stays on the grid on purpose (cheaper). | Confirm this split is wanted. |
| A7 | Correction to REVIEW.md, H1 | I wrote that the fixed round trip is "exact". That holds when the grid is at least as large as the padded crop (always true at `--image-size 256` with 256-pixel scans), **not** for a smaller grid such as `--image-size 128`. The docstring now says so. | — |
| A8 | Small behaviour changes made during the refactor | (1) A file name whose slice number cannot be parsed now stops with an error instead of silently skipping the patient. (2) TensorBoard image tags are `image/<i>` instead of `image/<batch>/<i>`. (3) The 10-step training-loss points no longer carry the last few losses of one epoch into the next. (4) `hubconf.unet_segmentation(pretrained=True)` returns the model in eval mode and refuses extra keyword arguments. (5) Checkpoints from before this change are refused (retraining is needed anyway for H2 and M2). | Check none of these matters to you. |

## New findings while fixing (not changed)

| # | Danger | Finding | Proof |
|---|---|---|---|
| N1 | Medium (Windows) | `python -m mriseg.aggregate` crashes with `UnicodeEncodeError` when its output is redirected (to a file, a pipe such as `\| tee`, or a terminal like Git Bash's mintty that is not a real Windows console): Python then writes with the Windows code page cp1252, which has no "Δ", and the table header contains "Δ" (`mriseg/aggregate.py`, `print_table`). The crash happens before `write_csv`, so `summary.csv` is not written either, and `run_cv.sh` (`set -e`) stops before `compare.py` and the export. A real console window (cmd, PowerShell, Windows Terminal) prints it fine. This bug is also in the original code, and it is the only printed text with such a character (the "Δ" in `compare.py` is a figure title, not printed). **Fixed** (2026-10-23): the header says "delta"; `test_aggregate.py::test_table_prints_when_output_is_redirected_on_windows` prints the table through a cp1252 stream (fails on the old header, passes now). | Verified: redirected run exited 1 with no `summary.csv` before, exits 0 and writes it now |
| N2 | Low | `dice_similarity_coefficient` returns 2e-8 instead of 0 for an empty prediction on a real tumour (the epsilon is added to the numerator). Cosmetic. | By reading + smoke run output |
| N3 | Low | Dead code: `TensorBoardLogger.log_image` is never called; `crop_to_content` is only used by tests. **Fixed** (2026-10-23): both removed; the tests call `content_box` directly. | grep |

## Further simplifications checked on 2026-10-23

| Item | Verdict | Status |
|---|---|---|
| Remove dead code (N3) | Correct | **Fixed** |
| Define `--device`, `--compile`, `--data-dir` once | **My claim was partly wrong.** `--device` is in 4 scripts, `--compile` in 3 (default `default` for training, `none` for predict and benchmark, different help texts), `--data-dir` in only 2. A shared helper would need parameters for the different defaults and texts, save about 6 lines, and add an import to every script. Not clearly simpler. | Not done |
| Simplify `pad_to_square` | Correct: one `np.pad` per array instead of two branches. Checked identical to the old version (values, shapes, dtypes) on 500 random shapes. | **Done** |

## First review (2026-10-19)

Fixed before this round: old H1, H2, M2, M5, M6, M7, L1, L3, L7, L8, L9, L11. Partly fixed then and finished now: old H3
(→ H1), M3 (→ L1), L2 (→ L8), L5 (→ L4), L6 (→ L7), L10 (→ L10). Still open: old M1 (→ M1, not requested) and M8 (no
preprocessing cache).
