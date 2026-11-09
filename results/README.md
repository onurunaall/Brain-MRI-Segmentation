# Raw results

The numbers and plots in the main README come from these files, so they can be checked and redrawn without a GPU or
the dataset.

`cv_seed0/`: 5-fold cross-validation, 3 architectures, one training seed per fold (15 runs), trained on one
NVIDIA RTX A6000 in October 2026.

```
cv_seed0/summary.csv                              # one row per architecture (aggregate.py)
cv_seed0/<arch>/fold<k>_seed0/test_results.json   # per test patient: Dice, Dice without post-processing, HD95; run metadata
cv_seed0/<arch>/fold<k>_seed0/logs/history.csv    # per epoch: train loss, val loss, val Dice, learning rate, seconds
cv_seed0/<arch>/fold<k>_seed0/logs/train_summary.json
cv_seed0/<arch>/fold<k>_seed0/logs/config.json    # the train.py options of the run
```

Reproduce the summary table and the plots that need no masks:

```bash
python -m mriseg.aggregate --runs-dir results/cv_seed0 --baseline unet --csv summary_check.csv
python -m mriseg.compare --runs-dir results/cv_seed0 --baseline unet --out-dir figures_check
```

The segmentation figures need the predicted masks (`test_masks.npz`, about 26 MB per run), which are not committed.
