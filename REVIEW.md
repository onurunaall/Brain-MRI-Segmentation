# Project review: Brain MRI Segmentation

What I reviewed: the working tree of branch `claude/segmentation-quality-fixes` on 2026-10-22, that is commit
`7cf7f40` plus the uncommitted changes that move the code into the `mriseg/` package.

> **Fix status (2026-10-22):** H1, H2, M2, M3, L1, L2, L4–L10 and S1–S5 are now fixed in the code. L3 and L11 are set
> aside, and so are the open questions about this review. See [REVIEW_TRACKER.md](REVIEW_TRACKER.md). The line
> numbers below refer to the code *before* the fixes.

How I worked:

- I read every source file, the scripts and the tests.
- When I suspected a bug, I ran a small experiment to prove it. Those issues are marked **Verified**. Issues marked
  **By reading** are ones I am confident about from the code but did not run.
- I did **not** change any project code. The experiments ran in a separate temporary folder.
- I ran the test suite: 146 passed, 8 skipped, 2 failed. One failure is a real bug (H1 below). The other
  (`test_every_compile_mode_is_accepted_by_torch`) fails because of a Windows path-length limit in my temporary
  Python install, not because of the project.

Danger levels:

- **High**: the program runs fine, but the numbers or the output masks are wrong.
- **Medium**: the numbers are misleading, or the model is worse than it needs to be.
- **Low**: small, cosmetic, speed, or a trap for later.

---

## Words used in this document

| Word | Meaning here |
|---|---|
| Scan / patient | One person's MRI: a folder of TIFF files, one file per slice. |
| Slice | One 2D picture of the head. A scan is a stack of slices. |
| Mask | A black-and-white picture of the same size as the slice. White = tumour. |
| Ground truth (GT) | The mask drawn by a doctor. The "correct answer". |
| Original pixels | The pixels of the TIFF files exactly as they are on disk (256 × 256). |
| Grid | The 256 × 256 picture the model actually works on. Before the model sees a scan, the code cuts away the empty border (crop), adds black bars to make it square (pad), and stretches it to 256 × 256 (resize). So one grid pixel is **not** one original pixel. |
| Resize / interpolation | Stretching a picture to a new size. For a mask you must copy existing pixel values (no in-between values), which is called "nearest" interpolation. |
| `"nearest"` vs `"nearest-exact"` | Two ways PyTorch does nearest interpolation. `"nearest"` is an old version that picks the source pixel slightly off-centre (it rounds down). `"nearest-exact"` picks the correct, centred source pixel. |
| Dice | Overlap score between prediction and ground truth. 1.0 = exactly the same pixels, 0.0 = no overlap. The main score of this project. |
| HD95 | "How far is the predicted edge from the true edge", in pixels. Lower is better. |
| Training / validation / test patients | Training patients teach the model. Validation patients pick the best epoch. Test patients are only used once, at the very end, to measure the score honestly. |
| Fold, seed | 5-fold cross-validation splits the 110 patients into 5 groups. Each group is the test set once. Each fold is trained 3 times with different random seeds. |
| Checkpoint | The saved model file (`best_model.pt`). |

---

## Short answer

1. **Wrong results:** yes. I found **2 High** issues. Both have the same root cause: masks are resized with
   PyTorch's `"nearest"` mode, which is off-centre by half a pixel.
   - **H1:** the masks written for real patients (`predict.py --masks-dir`) come out shifted by exactly 1 pixel
     down and 1 pixel right. Even a perfect model would only score Dice 0.78 to 0.97 against the real masks.
     The project's own test for this fails.
   - **H2:** during training only, the augmentation shifts every tumour label half a pixel down-right compared with
     the image. Test labels do not have this shift. So the model learns to draw tumours slightly in the wrong
     place, which lowers every reported Dice. I proved the shift; I could not measure the final Dice loss
     without the dataset and a GPU.
   - The fix for both is to write `"nearest-exact"` instead of `"nearest"` in 3 places, then retrain.

   There are also **3 Medium** issues (optimistic README numbers, brightness normalization that depends on the
   amount of black background, scores measured on the stretched grid instead of the real pixels) and **11 Low**
   ones.

   The important things that **are** done right: no patient appears in both training and test, the
   post-processing is chosen without looking at the test patients, and images are matched to their masks by slice
   number. I checked these carefully.

2. **Simpler and more readable code:** the code is careful and well documented, but it is far bigger than it needs
   to be. The main reasons: the same settings live in 3 places (command-line flags, `config.json`, the checkpoint),
   a lot of code exists only to stay compatible with old files, two functions do 7 to 9 jobs each, and the project
   carries a lot of optional features (ONNX, TensorRT, model bundles, benchmark, 4 kinds of segmentation figures).
   See Part 2.

3. **Design patterns:** the project already uses Strategy, Composite, Factory + Registry, Pipeline and Value Object,
   mostly in good places. The patterns that would help most are: one **settings object** (Parameter Object), the
   **checkpoint as the single source of truth**, and a **preprocessing object that knows how to undo itself**
   (this would have prevented H1). See Part 3.

---

## Summary table

| ID | Danger | Issue (one line) | Where | Proof |
|---|---|---|---|---|
| H1 | High | Masks written with `--masks-dir` are shifted 1 px down-right; small tumours lose up to 22% Dice | `mriseg/utils.py:255-256` | Verified |
| H2 | High | Training labels are shifted half a pixel vs the image (augmentation only), so the model learns a wrong position | `mriseg/augmentations.py:113-116` | Verified (shift); effect on final Dice not measured |
| M1 | Medium | README score (~0.90) is optimistic and outdated | `README.md:32-55`, `mriseg/predict.py:437` | By reading |
| M2 | Medium | Intensity normalization changes with the amount of black background | `mriseg/utils.py:270-293` | Verified (synthetic) |
| M3 | Medium | Scores are measured on the stretched 256 grid, not on the real pixels, and some slices are left out | `mriseg/predict.py:77`, `mriseg/dataset.py:180-182` | Verified (synthetic) |
| L1 | Low | HD95 averaged over seeds hides runs that missed the tumour completely | `mriseg/aggregate.py:34` | Verified |
| L2 | Low | Comparison figures show a seed-averaged Dice next to masks from one seed | `mriseg/compare.py:451`, `:485` | By reading |
| L3 | Low | HD95 mixes two units (in-plane pixels and "1 per slice") | `mriseg/predict.py:82-84` | By reading |
| L4 | Low | First and last slice (and empty end slices) are removed from the ground truth | `mriseg/dataset.py:180-182`, `mriseg/utils.py:107-135` | By reading |
| L5 | Low | Patient folders with the same name silently replace each other | `mriseg/dataset.py:119-127` | By reading |
| L6 | Low | The geometry tests are too loose; the one that would catch H1 is failing | `tests/test_utils.py:181-188` | Verified |
| L7 | Low | `get_data.py` deletes `./kaggle_3m` without asking | `get_data.py` | By reading |
| L8 | Low | `fix_model.py` is obsolete | `fix_model.py` | By reading |
| L9 | Low | README "Project structure" lists files that moved | `README.md:253-281` | By reading |
| L10 | Low | `scikit-image` is listed twice | `pyproject.toml:10`, `:18` | By reading |
| L11 | Low | TensorBoard writes to disk after every single number | `mriseg/tb_logger.py:30`, `:42` | By reading |

---

# Part 1: Things that make results wrong

## The root cause behind H1 and H2, explained with a picture

Think of a mask as a drawing on graph paper. To stretch it onto bigger graph paper, every new square must copy
the colour of one old square.

- The **correct** way: look at the **centre** of the new square, find which old square that point falls in, copy
  it. This is `"nearest-exact"`.
- PyTorch's old `"nearest"` mode instead looks at the **top-left corner** of the new square. The corner is half a
  square up and left of the centre, so it sometimes copies the old square above or to the left. The result: the
  stretched mask is drawn **about half a pixel too far down and right**.

The image itself is stretched with a different method (`bicubic` / `bilinear`) that **is** centred correctly. So
after stretching, the mask and the image no longer line up exactly. Half a pixel sounds tiny, but tumour edges are
where all the Dice errors happen, and a tumour of 10 to 20 pixels across has a lot of edge.

This happens in three places:

| Place | When it runs | Effect |
|---|---|---|
| `mriseg/utils.py:208` (`resize_volume`, mask) | Every scan, training and testing | Mask is 0.5 grid px off from the image. Same in training and testing, so the model is not confused by it, but it is half of the H1 problem. |
| `mriseg/augmentations.py:113-116` (`RandomScale`, mask) | Training only, every sample | Another 0.5 px shift that test data does not have → **H2** |
| `mriseg/utils.py:255-256` (`restore_mask_geometry`) | When writing masks for real patients | Another off-centre step → together with the first row, a full **1 px** shift → **H1** |

---

## H1 (High). Masks saved for real patients are shifted by 1 pixel

**In one sentence:** when `predict.py --masks-dir` maps the predicted mask back onto the original slices, every
mask ends up exactly 1 pixel too low and 1 pixel too far right, and small tumours shrink.

**Who is affected:** anyone who uses `--masks-dir`, which is the feature for predicting **new patients** and the only
output in the original scan size. The cross-validation scores are **not** affected by H1 (they are measured on the
grid; see M3 for that).

**Where:**
[mriseg/utils.py:254-256](mriseg/utils.py:254) (going back) together with
[mriseg/utils.py:207-209](mriseg/utils.py:207) (going forward). Called from
[mriseg/predict.py:312](mriseg/predict.py:312).

**What the code does now:**

```python
# mriseg/utils.py, resize_volume (forward: original crop -> 256 grid)
seg_t = F.interpolate(seg_t, size=(target_size, target_size),
                      mode="nearest")

# mriseg/utils.py, restore_mask_geometry (backward: 256 grid -> original crop)
resized = F.interpolate(torch.from_numpy(np.asarray(mask, dtype=np.float32))[:, None],
                        size=(padded, padded), mode="nearest")[:, 0].numpy() > 0.5
```

**Why it is wrong:** each direction rounds down ("top-left corner"). Going forward moves the mask about half a
pixel; going back moves it about half a pixel again, in the same direction. I worked out the maths: for every
original pixel `o`, the round trip gives back pixel `o - 1` (except in rare spots), in both rows and columns.

**Proof (Verified):** I took a fake scan with a round tumour, ran the project's own preprocessing, then pretended
the model was **perfect** (its prediction equals the ground truth on the grid) and mapped it back with
`restore_mask_geometry`:

| Tumour radius | Dice of the restored mask vs the real mask | Shift (rows, columns) |
|---|---|---|
| 4 px | 0.776 | (1, 1) |
| 8 px | 0.883 | (1, 1) |
| 15 px | 0.939 | (1, 1) |
| 30 px | 0.970 | (1, 1) |

A perfect model should give 1.000. The project's own test
[tests/test_utils.py:181-188](tests/test_utils.py:181) also fails on my machine: it expects 36 tumour pixels back
and gets 28 (Dice 0.75).

**Possible fixes:**

- **A.** Use `"nearest-exact"` in both directions (and in the augmentation, see H2). Then the round trip is
  exact. I tested it: Dice 1.000 for every radius, and the failing test gets all 36 of 36 pixels back.
  Downside: the model must be retrained, because the old models learned the shifted labels.
- **B.** Keep the forward step as it is and make the backward step the exact mirror of the old forward step (pick
  grid pixel `ceil(o * grid / padded)`). I tested it: exact for every crop size from 100 to 256. Upside: works with
  the models you already trained. Downside: the image/mask half-pixel mismatch inside the grid stays.
- **C.** Stop saving masks in original size. Not a real option; this is the whole point of `--masks-dir`.

**Recommended: A**, together with the H2 fix, because it removes the root cause in all three places with three
one-word changes, and the code then says what it means. Use **B** only if you must keep using models that are
already trained.

**How I would implement A:**

```python
# mriseg/utils.py, resize_volume
    seg_t = F.interpolate(seg_t, size=(target_size, target_size),
                          mode="nearest-exact")   # was "nearest": off-centre by half a pixel

# mriseg/utils.py, restore_mask_geometry
    resized = F.interpolate(torch.from_numpy(np.asarray(mask, dtype=np.float32))[:, None],
                            size=(padded, padded), mode="nearest-exact")[:, 0].numpy() > 0.5
```

And make the test strict, so this can never come back silently
([tests/test_utils.py:181](tests/test_utils.py:181)):

```python
    def test_round_trip_is_exact_after_upsampling(self) -> None:
        mask, grid_mask, geometry = self._preprocess(height=20, width=14, grid_size=64)
        restored = restore_mask_geometry(grid_mask, geometry)
        np.testing.assert_array_equal(restored, mask > 0.5)  # a perfect prediction must come back unchanged
```

**How I would implement B** (only if you keep old models):

```python
# mriseg/utils.py, restore_mask_geometry: exact mirror of F.interpolate(mode="nearest") upsampling
    padded = geometry.padded_size
    idx = np.minimum(np.ceil(np.arange(padded) * geometry.grid_size / padded).astype(int),
                     geometry.grid_size - 1)
    resized = (np.asarray(mask) > 0.5)[:, idx][:, :, idx]
```

---

## H2 (High). Training labels are half a pixel off from the image

**In one sentence:** the random-zoom augmentation stretches the image correctly but the mask off-centre, so during
training every tumour label sits about half a pixel down-right of the tumour in the image; test data does not have
this shift, so the trained model draws tumours in a slightly wrong place.

**Where:** [mriseg/augmentations.py:108-116](mriseg/augmentations.py:108)

```python
        image = TF.resize(image, [new_h, new_w],
                          interpolation=TF.InterpolationMode.BILINEAR,   # centred correctly
                          antialias=False)
        mask = TF.resize(mask, [new_h, new_w],
                         interpolation=TF.InterpolationMode.NEAREST,     # off-centre (old "nearest")
                         antialias=False)
```

`RandomScale` runs on **every** training sample (the zoom factor is random between 0.95 and 1.05, and it is almost
never exactly 1.0).

**Why it is wrong:** the model learns "the tumour label is half a pixel down-right of the bright spot". At test time
the label is not shifted, so every predicted edge is a little off. This costs Dice on every edge of every tumour,
more for small tumours.

**Proof (Verified for the shift):** I drew the same circle into the image and into the mask, ran `RandomScale` 200
times, and compared where the image puts the circle and where the mask puts it:

| Circle radius | Average shift of the label vs the image | Dice between the training label and the correct outline |
|---|---|---|
| 8 px | 0.47 px down, 0.47 px right | 0.940 |
| 15 px | 0.48 px down, 0.47 px right | 0.966 |
| 30 px | 0.50 px down, 0.47 px right | 0.983 |

With `NEAREST_EXACT` the shift drops to about 0 (between -0.27 and +0.21 px in my tests, no direction preferred).
The rotation augmentation does **not** have this problem (average shift 0.005 px).

**What I could not prove:** how many Dice points the final model loses. That needs training on the real data. My
estimate, from the table above, is roughly 1 to 3 Dice points for typical tumours, more for small ones. That is
about the same size as the differences between architectures you are trying to measure. The comparison between
architectures stays fair (they all get the same handicap), but all absolute numbers are lower than they should be.

**Possible fixes:**

- **A.** Use `NEAREST_EXACT` for the mask. One word. Needs retraining (which you need anyway to see the benefit).
- **B.** Resize the mask with `BILINEAR` like the image, then threshold at 0.5. Also centred, slightly smoother
  edges. One more line.
- **C.** Remove the zoom augmentation. Loses a useful augmentation.

**Recommended: A**, it is the smallest change and matches the H1 fix.

**How I would implement it:**

```python
# mriseg/augmentations.py, RandomScale.__call__
        mask = TF.resize(mask,
                         [new_h, new_w],
                         interpolation=TF.InterpolationMode.NEAREST_EXACT,  # NEAREST is off-centre by half a pixel
                         antialias=False)
```

Add a test so it stays fixed (in `tests/test_augmentations.py`):

```python
def test_random_scale_keeps_mask_on_the_image() -> None:
    yy, xx = np.mgrid[:64, :64]
    circle = torch.from_numpy(((yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 100).astype(np.float32))[None]
    for _ in range(20):
        image, mask = RandomScale(0.05)(circle.clone(), circle.clone())
        image_rows = np.nonzero((image[0] > 0.5).numpy().any(axis=1))[0]
        mask_rows = np.nonzero((mask[0] > 0.5).numpy().any(axis=1))[0]
        assert abs(image_rows.mean() - mask_rows.mean()) < 0.3
```

After H1 + H2 are fixed, **retrain** (`run_cv.sh` into a new `RUNS_DIR`) and compare the CV summary with the old one.
That comparison is also the real measurement of how much H2 cost.

---

## M1 (Medium). The README score is optimistic and outdated

**In one sentence:** the "Mean DSC ~0.90" in the README was measured on the same 10 patients that chose the best
epoch, and with today's default settings those same 10 patients also choose the post-processing.

**Where:** [README.md:32-55](README.md:32) and the default `--split validation` in
[mriseg/predict.py:437](mriseg/predict.py:437) together with `--postprocess auto` in
[mriseg/predict.py:447](mriseg/predict.py:447).

**Why it is wrong, in plain words:** training runs 100 epochs and keeps the one with the best score on the 10
validation patients. That is like taking an exam 100 times and reporting your best attempt: the best attempt is
partly luck. Then `--postprocess auto` picks the best of 3 cleaning methods on the **same** 10 patients, and the
score is again reported on them. So the number is too high. With only 10 patients, the error can easily be a few
Dice points.

The code already knows this: [mriseg/predict.py:206-218](mriseg/predict.py:206) prints a warning. But the README
headline, the README quickstart command and the default flags all lead to exactly this situation. Also, the README
figure `assets/dice_distribution.png` was last changed on 2026-03-05, while Dice+BCE loss, 2.5D input and the new
post-processing were added on 2026-10-13 (commit `7cf7f40`). So the README numbers describe an older model, not
this code.

**Possible fixes:**

- **A.** Report the cross-validation **test** numbers (from `runs/summary.csv` after `run_cv.sh`) in the README,
  and make `--split` a required flag so nobody gets the optimistic number by accident.
- **B.** Keep the README number but label it "validation patients, optimistic".
- **C.** Change the default split to `test`. Does not work for single-split models (they have no test set).

**Recommended: A.** The project already has an honest, well-built evaluation (5 folds × 3 seeds, every patient
tested exactly once by a model that never saw it). The README should show that.

**How I would implement it:**

```python
# mriseg/predict.py, _parse_args
    parser.add_argument("--split", type=str, required=True, choices=["validation", "test", "all"],
                        help="Which patients to evaluate. 'test' (with --fold) gives the honest score; "
                             "'validation' patients chose the checkpoint, so their score is optimistic; "
                             "'all' = every patient in --data-dir (new patients)")
```

README "Results" section, filled from `runs/summary.csv`:

```markdown
## Results (5-fold cross-validation, 3 seeds per fold, 110 patients)

Every patient is scored once, by a model that never saw it during training or model selection.

| Model | Mean Dice ± std | Median | HD95 median (px) |
|---|---|---|---|
| U-Net | ... | ... | ... |
```

---

## M2 (Medium). Brightness normalization depends on how much black background there is

**In one sentence:** each scan's brightness is normalized using **all** voxels, including the black area around the
head, so the same brain tissue gets a different number in different patients.

**Where:** [mriseg/utils.py:270-293](mriseg/utils.py:270), called in
[mriseg/dataset.py:213](mriseg/dataset.py:213).

```python
    for c in range(vol.shape[-1]):
      p_lo = np.percentile(vol[..., c], low_percentile)     # computed over head AND background
      p_hi = np.percentile(vol[..., c], high_percentile)
      vol[..., c] = rescale_intensity(vol[..., c], in_range=(p_lo, p_hi))
    ch_mean = np.mean(vol, axis=(0, 1, 2))                  # also over head AND background
    ch_std = np.std(vol, axis=(0, 1, 2))
    vol = (vol - ch_mean) / ch_std
```

**Why it is a problem:** the mean and spread include the black background. How much background a scan has depends
on the head's shape and on how many top-of-head slices (mostly black) are in the scan. More black → lower mean →
the same tissue looks brighter after normalization.

**Proof (Verified, synthetic data):** the same tissue value (raw 100) after normalization:

| Share of black background | Value after normalization |
|---|---|
| 20% | +0.48 |
| 40% | +0.80 |
| 60% | +1.21 |

So the model sees the same tissue at very different values in different patients. The brightness augmentation
(±20%) hides some of this, but not all. The effect on the final Dice is **not measured**; it needs a training run.

**Possible fixes:**

- **A.** Compute the percentiles, mean and std only over head voxels (raw value above a small threshold, the same
  idea `content_box` already uses).
- **B.** Per-slice normalization. Worse: neighbouring slices (2.5D input) would no longer be on the same scale.
- **C.** Leave it. It is what the original LGG reference code did.

**Recommended: A.** Small change, standard practice (nnU-Net does it), and it removes a source of
patient-to-patient noise.

**How I would implement it:**

```python
def normalize_intensity(vol: npt.NDArray,
                        low_percentile: float = 1.0,
                        high_percentile: float = 99.0) -> npt.NDArray:
    """
    Per channel: clip to percentiles and z-score, with the statistics taken from head voxels only, so the result
    does not depend on how much black background surrounds the head. Background becomes the channel minimum.
    """
    vol = vol.astype(np.float32)
    head = vol.max(axis=-1) > 0.1 * vol.max()          # same background rule as content_box
    for c in range(vol.shape[-1]):
        lo, hi = np.percentile(vol[..., c][head], [low_percentile, high_percentile])
        clipped = np.clip(vol[..., c], lo, hi)         # black background is clipped up to `lo`
        mean, std = clipped[head].mean(), clipped[head].std() or 1.0
        vol[..., c] = (clipped - mean) / std
    return vol
```

Note the low percentile changes from 10 to 1: the old 10 was only needed because the background was included.
I ran this version on the same synthetic test: the raw value 100 becomes +0.00 with 20%, 40% and 60% background.
Retrain and compare on CV before keeping it.

---

## M3 (Medium). Scores are measured on the stretched grid, not on the real pixels

**In one sentence:** every Dice in this project compares two masks that have both been cropped and stretched to
256 × 256, and some slices are left out, so the number is not exactly the Dice on the real scan.

**Where:** [mriseg/predict.py:77](mriseg/predict.py:77) (`_compute_dice_per_patient` on the grid volumes) and the
ground truth that was stretched in [mriseg/dataset.py:211](mriseg/dataset.py:211).

**Why it matters:** stretching a crop of, say, 221 pixels to 256 copies some rows twice and others once. Pixels that
were copied twice count double in the Dice. Also see L4 (removed slices).

**Proof (Verified, synthetic):** the same prediction (the true tumour grown by 1 pixel), scored two ways:

| Scored on | Dice |
|---|---|
| Original pixels (the true answer) | 0.866 |
| 256 grid (what the project reports) | 0.876 |

The difference can go either way and is about ±1 Dice point for small tumours. For comparing architectures with
each other it does not matter (all are scored the same way). For comparing with papers, or for a final report, it
does.

**Possible fixes:**

- **A.** For the final test numbers, map the prediction back with `restore_mask_geometry` (after the H1 fix!) and
  score it against the original `*_mask.tif` files. Keep the grid Dice for training/validation, where it is only
  used to pick epochs.
- **B.** Do not stretch at all: pad every crop to 256 × 256 without resizing. Simpler preprocessing, but larger
  heads would need cropping.

**Recommended: A** after H1 is fixed, because the code to map back already exists.

**How I would implement it:**

```python
# mriseg/dataset.py: keep the original masks (before any cropping or resizing) per patient
        self.original_masks: Dict[str, npt.NDArray] = {}
        ...
            self.original_masks[pid] = masks > 0.5          # all slices, original size

# mriseg/predict.py: score the final prediction on the real pixels
def _dice_on_original_pixels(volumes, dataset) -> Dict[str, float]:
    return {pid: dice_similarity_coefficient(restore_mask_geometry(pred[:, 0], dataset.geometry[pid]),
                                             dataset.original_masks[pid])
            for pid, (_, pred, _) in volumes.items()}
```

Then write this as `"dice"` in `test_results.json` and keep the grid value as `"dice_grid"`.

---

## L1 (Low). HD95 averaging over seeds hides complete misses

**Where:** [mriseg/aggregate.py:34](mriseg/aggregate.py:34)

```python
    return {pid: float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan") ...
```

**What is wrong:** HD95 is "undefined" (NaN) when a run finds no tumour at all. `nanmean` simply skips those runs.
So if seed 1 missed the tumour but seeds 0 and 2 found it, the patient looks fine. **Verified:** HD95 values
(4, missed, 6) average to 5.0. Later, line 103 says "undefined counts as the worst value", so the two places
disagree.

**Fix (recommended):** treat a miss as infinitely bad **before** averaging, for HD95 only:

```python
def mean_per_patient(results, metric):
    collected: Dict[str, List[float]] = {}
    for res in results:
        for pid, vals in res["per_patient"].items():
            value = vals.get(metric, float("nan"))
            if metric == HD95_KEY and np.isnan(value):
                value = float("inf")          # tumour missed entirely = worst possible edge distance
            collected.setdefault(pid, []).append(value)
    return {pid: float(np.mean(values)) for pid, values in collected.items()}
```

The median in the summary already handles `inf`. The box plot in `compare.py:121-123` should then show the count of
`inf` patients instead of dropping them.

## L2 (Low). Figure titles show a different Dice than the masks drawn

**Where:** [mriseg/compare.py:451](mriseg/compare.py:451) and [mriseg/compare.py:485](mriseg/compare.py:485).
The per-patient figures draw the masks of **one** seed (`_mask_run`, line 101-107) but write the Dice **averaged
over 3 seeds** in the title. You can see a perfect-looking mask with a mediocre score, or the other way around. The
galleries (line 529-549) already do it right ("scored with the Dice of the run whose masks are drawn").
**Fix:** use `run.per_patient("dice")` of the run whose masks are drawn, as the gallery does.

## L3 (Low). HD95 mixes two units

**Where:** [mriseg/predict.py:82-84](mriseg/predict.py:82). In-plane distances are in original pixels, but one slice
step counts as 1. Slices are usually several times thicker than a pixel, so up-down errors are under-counted. The
TIFF files carry no physical size, so a perfect fix is impossible. **Fix:** state it next to every HD95 number, or
report a 2D (per-slice) HD95 that has only one unit.

## L4 (Low). Some slices are removed from the ground truth

**Where:** [mriseg/dataset.py:180-182](mriseg/dataset.py:180) drops the first and last slice of every scan, and
`content_box` ([mriseg/utils.py:107-135](mriseg/utils.py:107)) drops empty end slices. Any tumour on those slices
disappears from the "correct answer", so it can never be counted as missed. Usually there is no tumour there, so
the effect is small. **Fix:** with M3-A, score against the original masks; then these slices count again.

## L5 (Low). Patient folders with the same name replace each other

**Where:** [mriseg/dataset.py:119-127](mriseg/dataset.py:119). Patients are keyed by folder name only. The Kaggle
download (as far as I know) contains the data twice (`kaggle_3m/` and `lgg-mri-segmentation/kaggle_3m/`), and
`get_data.py` copies everything. Today that is harmless because both copies are identical, but any other folder
with the same name would silently win. **Fix:** raise an error when a name is seen twice with a different path, and
let `get_data.py` copy only one copy.

## L6 (Low). Tests did not protect the geometry

**Where:** [tests/test_utils.py:181-188](tests/test_utils.py:181). The test for mapping masks back only asks for
"about the same size" (±15%) and Dice > 0.85. That is why H1 could exist at all. On my machine it fails anyway
(28 instead of 36 pixels). **Fix:** exact equality, as shown in H1.

## L7 (Low). `get_data.py` deletes your data folder without asking

**Where:** [get_data.py](get_data.py). It runs `shutil.rmtree("./kaggle_3m")` and has no
`if __name__ == "__main__":`, so importing it starts a download. **Fix:** ask or refuse when the folder exists, and
add the main guard.

## L8 (Low). `fix_model.py` is obsolete

**Where:** [fix_model.py](fix_model.py). Checkpoints are now saved without the `_orig_mod.` prefix, and loading
strips it anyway. The script has a hard-coded path and overwrites the file in place. **Fix:** delete it and its
README mention.

## L9 (Low). README "Project structure" is out of date

**Where:** [README.md:253-281](README.md:253). It lists `train.py`, `predict.py`, ... at the top level; they are in
`mriseg/` now. **Fix:** update the tree.

## L10 (Low). `scikit-image` listed twice

**Where:** [pyproject.toml:10](pyproject.toml:10) and [pyproject.toml:18](pyproject.toml:18). **Fix:** delete line 18.

## L11 (Low). TensorBoard flushes after every number

**Where:** [mriseg/tb_logger.py:30](mriseg/tb_logger.py:30), [mriseg/tb_logger.py:42](mriseg/tb_logger.py:42). Every
`log_scalar` forces a disk write. Slower training, no benefit. **Fix:** remove the `flush()` calls;
`SummaryWriter` flushes by itself every few seconds and on `close()`.

---

## Checked and fine

These are places where bugs often hide in segmentation projects. I checked them and they are correct here:

- **No patient leakage.** Splits are by patient, never by slice ([mriseg/dataset.py:20-59](mriseg/dataset.py:20)).
  The checkpoint stores its training patients, and `predict.py` refuses to score any of them
  ([mriseg/predict.py:184-203](mriseg/predict.py:184)).
- **Post-processing is chosen on validation patients, never on test patients**
  ([mriseg/predict.py:247-271](mriseg/predict.py:247)).
- **Images and masks are paired by slice number**, and a missing mask file is an error
  ([mriseg/dataset.py:166-170](mriseg/dataset.py:166)).
- **Slices are put back in the right order**; any mix-up raises an error
  ([mriseg/utils.py:43-67](mriseg/utils.py:43)).
- **Rotation and flip augmentations keep image and mask aligned** (measured: 0.005 px).
- **Augmentation copies the data**, so the stored dataset is never changed.
- **Loss:** Dice + BCE on logits, computed in float32 even under mixed precision ([mriseg/losses.py](mriseg/losses.py)).
- **Swin-UNETR:** window partition/reverse, shifted-window mask, relative position index and patch merging match
  the reference implementation ([mriseg/network.py:182-306](mriseg/network.py:182)).
- **K-fold maths:** 110 patients → 5 folds of 22; each fold's validation patients are drawn from the other 88.
- **Statistics:** the Wilcoxon test is paired per patient, after averaging seeds. Correct.
- **TensorRT stream order** ([mriseg/inference.py:228-232](mriseg/inference.py:228)): waits are in the right order.

## Old review (2026-10-19): what happened to it

| Old ID | Old issue | Now |
|---|---|---|
| H1 | Model scored on its own training patients | **Fixed** (checkpoint stores patients, predict refuses) |
| H2 | Training crashed on Windows with workers | **Fixed** |
| H3 | Predicting new patients crashed / wrong geometry | **Partly fixed**: runs now, but the output is shifted (new H1) |
| M1 | README number optimistic | **Still open** (now M1; a warning was added) |
| M2 | Masks paired by list position | **Fixed** |
| M3 | HD95 units, NaN left out | **Partly fixed**: units and median fixed; seed averaging still hides misses (new L1) |
| M4 | Normalization includes background | **Still open** (now M2, with numbers) |
| M5 | Workers re-created every epoch | **Fixed** |
| M6 | Slice regrouping copied 4 times | **Fixed** (one function that checks order) |
| M7 | Export deleted old bundle first | **Fixed** |
| M8 | No preprocessing cache | **Still open** (see S8 in Part 2) |
| L1, L3, L7, L8, L9 | Dice post-processed, global random seed, hubconf, slow contours, column name | **Fixed** |
| L2 | Dead code, `fix_model.py` | **Partly**: dead code gone, `fix_model.py` still there (new L8) |
| L4 | Classes that are only bags of functions | **Partly** (see Part 2) |
| L5 | First / last slice removed | **Still open** (new L4) |
| L6 | `get_data.py` deletes without asking | **Still open** (new L7) |
| L10 | `assert` for input, duplicate dependency, flush | **Partly**: dependency and flush still open (new L10, L11) |
| L11 | Seeding is not full determinism | **Fixed** (now documented honestly) |

---

# Part 2: Good parts, bad parts, and how to make it simpler

## The good parts (keep these)

- **Honest evaluation design.** Patient-level 5-fold CV, 3 seeds, paired Wilcoxon tests, post-processing chosen on
  validation only, a guard against scoring training patients. Many published projects get this wrong.
- **Every model returns logits, the sigmoid is applied in one place.** Clear rule, written down.
- **The inference backends share one small interface** ([mriseg/inference.py:103-111](mriseg/inference.py:103)).
  Adding a backend does not touch any caller.
- **Adding a model is one line** in `ARCHITECTURES` ([mriseg/network.py:410-414](mriseg/network.py:410)).
- **Errors are loud and explain what to do** (e.g. [mriseg/inference.py:97-99](mriseg/inference.py:97)).
- **Type hints and docstrings everywhere; tests run on fake data without a GPU.**
- **Comments explain *why*, not *what*** (e.g. the Dice + BCE explanation in
  [mriseg/losses.py:57-59](mriseg/losses.py:57)).

## The bad parts

The code is not messy. It is **too big for what it does**, mostly because it grew feature by feature and every old
format is still supported. About 4,200 lines of source code (without tests) for "train a U-Net and evaluate it".
A clean version of the same science fits in roughly 1,500 to 2,000 lines.

### S1. The same settings live in three places

`arch`, `context_slices`, `image_size`, the split settings and the patient lists are stored in the checkpoint
([mriseg/train.py:201-209](mriseg/train.py:201)), **and** in `config.json`
([mriseg/train.py:108-112](mriseg/train.py:108)), **and** must be typed again as flags for `predict.py`
([mriseg/predict.py:435-446](mriseg/predict.py:435)). `export_models.py` reads `config.json`
([mriseg/export_models.py:251-270](mriseg/export_models.py:251)), `predict.py` reads flags and then checks them
against the checkpoint ([mriseg/predict.py:167-218](mriseg/predict.py:167), ~50 lines that only exist to catch
mismatches).

**Simpler:** the checkpoint is the only truth. `predict.py --model-path X --split test` reads everything else from
the checkpoint, including the list of test patients. Then wrong flags are impossible, and
`_refuse_training_patients`, `_checkpoint_record`, `_warn_if_checkpoint_selection_patients`, `_common_image_size`,
`_common_context_slices` and 6 flags disappear.

```python
# predict.py, simplified
state, meta = read_checkpoint(cfg.model_path, device)
model = create_model(meta["arch"], in_channels=num_input_channels(meta["context_slices"]), out_channels=1)
model.load_state_dict(state)
patients = meta["test_patients"] if cfg.split == "test" else meta["val_patients"]
dataset = SliceDataset(load_patients(cfg.data_dir, patients, meta["image_size"]), meta["context_slices"])
```

(For this, `train.py` must also save `test_patients` in the checkpoint meta.)

### S2. Too much code for old files

Examples: old checkpoints without meta ([mriseg/inference.py:73-75](mriseg/inference.py:73),
[hubconf.py:27](hubconf.py)), results with `hd95_vox` ([mriseg/aggregate.py:18](mriseg/aggregate.py:18),
`:76-79`, `:157-161`), training curves read from TensorBoard when `history.csv` is missing
([mriseg/compare.py:233-264](mriseg/compare.py:233)), bundles without `context_slices`
([mriseg/inference.py:319-321](mriseg/inference.py:319)), the "backfill" branch in
[run_cv.sh:27-30](run_cv.sh:27), `fix_model.py`.

**Simpler:** you will retrain anyway after H1/H2. Delete all of these and say in the README "results from before
version X must be regenerated".

### S3. The dataset constructor does 7 jobs in 160 lines

[mriseg/dataset.py:86-249](mriseg/dataset.py:86) finds patient folders, splits patients, reads files, checks
image/mask pairs, preprocesses, computes sampling weights, and builds the slice index. You cannot preprocess one
patient without building a whole dataset, and you cannot test the parts alone.

**Simpler:** small plain functions plus a thin dataset:

```python
def find_patients(root: str) -> Dict[str, str]: ...                       # patient id -> folder
def read_patient(folder: str) -> Tuple[np.ndarray, np.ndarray, List[str]]: ...  # images, masks, file names
def preprocess(images, masks, size) -> Tuple[np.ndarray, np.ndarray, VolumeGeometry]: ...

class SliceDataset(Dataset):
    """Serves (image with neighbour slices, mask) pairs from already preprocessed volumes."""
    def __init__(self, volumes: List[Tuple[np.ndarray, np.ndarray]], context_slices: int, transform=None): ...
```

`kfold_split` is already a separate function: good, keep it.

### S4. The training loop mixes training and validation with many `if phase == ...`

[mriseg/train.py:228-335](mriseg/train.py:228) loops over `("train", "valid")` and then branches on `phase` six
times. It is hard to see what happens in training versus validation.

**Simpler:** two functions with no branches:

```python
def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    losses = []
    for images, masks in tqdm(loader, leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = criterion(model(images), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, criterion, device) -> Tuple[float, float]:
    """Returns (mean loss, mean per-patient Dice)."""
    model.eval()
    losses, probabilities, targets = [], [], []
    for images, masks in loader:
        logits = model(images.to(device))
        losses.append(criterion(logits, masks.to(device)).item())
        probabilities.extend(torch.sigmoid(logits).cpu().numpy())
        targets.extend(masks.numpy())
    dataset = loader.dataset
    dice = raw_dice_per_patient(probabilities, targets, dataset.flat_index, dataset.patient_ids)
    return float(np.mean(losses)), float(np.mean(list(dice.values())))
```

The main loop then reads like the description of the method: train, validate, save if better, log.

### S5. `predict.py` does 9 jobs

[mriseg/predict.py:32-119](mriseg/predict.py:32): load model or ensemble, choose post-processing, predict, regroup,
save original-size masks, compute Dice, draw a bar chart, compute HD95, write JSON, write NPZ, draw overlay PNGs.

**Simpler:** split into "predict" (images → probability volumes), "evaluate" (volumes → metrics JSON) and "save"
(volumes → files). Also small things:

- `argparse.Namespace(**{**vars(cfg), "split": "validation"})` ([mriseg/predict.py:257](mriseg/predict.py:257)) is
  a hack to reuse a function. With S1 you just pass a patient list.
- The bar chart is drawn to PNG bytes, decoded to an array, then written again as PNG
  ([mriseg/predict.py:374-409](mriseg/predict.py:374)). `fig.savefig(path)` does it in one line.

### S6. `utils.py` is a drawer of unrelated things

It holds Dice, regrouping, post-processing, cropping, padding, resizing, geometry, normalization, drawing and
seeding. **Simpler:** `preprocessing.py` (crop, pad, resize, normalize, geometry), `metrics.py` (Dice, HD95,
post-processing), `viz.py` (drawing). Each file then has one topic.

### S7. Too many optional features in the core

ONNX export, TensorRT engines, fold-ensemble bundles, the compute benchmark, `torch.compile` modes, and four kinds of
segmentation figures (`segmentation/`, `segmentation_overview.png`, `all_patients.png`, `cases/`) are over 1,000
lines.
They are well written, but each one adds flags, tests and README text you must keep correct.

**Ask yourself for each one: "Do I use this result?"** If not, move it to an `extras/` folder or delete it. For a
research comparison of three architectures, you need: train, predict on test fold, aggregate, a few figures.

### S8. Everything is re-read and re-preprocessed on every run

Every `train.py` / `predict.py` call re-reads all TIFF files and redoes cropping, resizing and normalization.
`predict.py --postprocess auto` does it twice (validation, then test). With 5 folds × 3 seeds × 3 architectures
that is 45+ times the same work. **Simpler and faster:** preprocess once into one `.npz` per patient
(`preprocessed/<patient>.npz` with image, mask, geometry) and load those. This is also where an exact geometry test
(H1) naturally lives.

### S9. Small things that make reading harder

- `content_box` repeats the same 4 lines for each axis ([mriseg/utils.py:120-133](mriseg/utils.py:120)):

  ```python
  def content_box(vol, bg_threshold_ratio=0.1):
      foreground = vol.max(axis=-1) >= vol.max() * bg_threshold_ratio
      box = []
      for axis in range(3):
          other_axes = tuple(a for a in range(3) if a != axis)
          present = np.nonzero(foreground.any(axis=other_axes))[0]
          box.append(slice(int(present[0]), int(present[-1]) + 1))
      return tuple(box)
  ```

- The slice number is parsed with `split(".")[-2].split("_")[4]`
  ([mriseg/dataset.py:318-322](mriseg/dataset.py:318)), which breaks if a folder name has a different number of
  underscores. A regular expression says what it means: `re.search(r"_(\d+)(_mask)?\.tif$", name)`.
- `for p_idx, n_sl in enumerate(slices_per_patient)` where `n_sl` is unused
  ([mriseg/dataset.py:246](mriseg/dataset.py:246)); the whole block is `np.concatenate(patient_slice_weights)`.
- The indentation in `normalize_intensity` is 2 spaces inside a 4-space file
  ([mriseg/utils.py:282-284](mriseg/utils.py:282)).
- `TensorBoardLogger` ([mriseg/tb_logger.py](mriseg/tb_logger.py)) only forwards to `SummaryWriter`. Using
  `SummaryWriter` directly removes a file.
- `ResultsAggregator` and `ComparisonPlotter` are classes that mostly hold a list and call functions. Plain functions
  that take the list of runs are simpler.
- The same flags (`--image-size`, `--context-slices`, `--device`, `--compile`, split flags) are defined separately in
  4 scripts. One shared settings object (Part 3) removes this.

## A simpler layout

```
mriseg/
  data.py          find_patients, read_patient, kfold_split, SliceDataset
  preprocessing.py crop / pad / resize / normalize, VolumeGeometry, restore (forward and backward side by side)
  augment.py       the 5 augmentations
  models.py        U-Net, ResU-Net, Swin-UNETR, ARCHITECTURES
  losses.py        Dice + BCE
  metrics.py       dice, hd95, postprocess
  train.py         train_one_epoch, validate, main
  predict.py       predict volumes, evaluate, save
  report.py        aggregate table + figures
extras/            onnx / tensorrt / bundles / benchmark (only if you use them)
```

## Suggested order of work

1. Fix H1 + H2 (3 one-word changes) and make the tests strict. → Retrain once.
2. Fix M1 (README from CV results) and L1.
3. Try M2 (normalization) as a separate CV run; keep it only if it helps.
4. Simplify: S1 (checkpoint = truth) and S2 (drop old formats) first, because they delete the most code.
   Then S3/S4/S5 (split big functions). Then decide about S7.

---

# Part 3: Design patterns

A design pattern is a well-known shape of code for a common problem. Patterns help when they make code easier to
change; they hurt when they add layers you do not need.

## Patterns already in the code

| Pattern | Where | What it means here | Verdict |
|---|---|---|---|
| **Strategy** | `InferenceBackend` + `PyTorchBackend`, `OnnxRuntimeBackend`, `TensorRTBackend` ([mriseg/inference.py:103-234](mriseg/inference.py:103)) | Several interchangeable ways to do one job ("turn images into probabilities"). The caller does not care which one it has. | Good use. |
| **Composite** | `EnsembleBackend` ([mriseg/inference.py:237-252](mriseg/inference.py:237)) | A group of backends that behaves like one backend. `predict.py` does not know it is talking to 15 models. | Good use. |
| **Factory** | `create_backend` ([mriseg/inference.py:255-285](mriseg/inference.py:255)), `create_model` ([mriseg/network.py:422](mriseg/network.py:422)), `build_augmentation_pipeline` ([mriseg/augmentations.py:12](mriseg/augmentations.py:12)) | One function decides which object to build (from a file extension or a name), so callers never write `if ext == ".onnx": ...`. | Good use. |
| **Registry** | `ARCHITECTURES` dict ([mriseg/network.py:410](mriseg/network.py:410)) | A table name → class. Adding a model = one line; argparse choices come from it. | Good use. |
| **Pipeline** (a chain of steps) | `AugmentationPipeline` ([mriseg/augmentations.py:43-83](mriseg/augmentations.py:43)) | Each step takes (image, mask) and returns (image, mask); the pipeline runs them in order. Same idea as torchvision `Compose`. | Good use. |
| **Value Object** | `VolumeGeometry` ([mriseg/utils.py:215-242](mriseg/utils.py:215)), `GalleryEntry` | A small frozen bundle of data with a name, instead of a loose tuple. | Good use. |
| **Adapter / thin wrapper** | `TensorBoardLogger` ([mriseg/tb_logger.py](mriseg/tb_logger.py)) | Wraps `SummaryWriter`. | Not needed: it adds nothing (S9). |
| **Memoization** | `functools.lru_cache` on `_checkpoint_record` ([mriseg/predict.py:167](mriseg/predict.py:167)) | Remember a result so the file is read once. | Works, but only needed because of S1. |

## Where other patterns would help

### 1. Parameter Object (one settings object) — **recommended**

**Problem it solves:** `argparse.Namespace` is passed everywhere, the same flags are defined in 4 scripts, nothing
says which fields exist, and code copies it with `argparse.Namespace(**{**vars(cfg), "split": "validation"})`.

**What it looks like:**

```python
@dataclass(frozen=True)
class DataSettings:
    data_dir: str = "./kaggle_3m"
    image_size: int = 256
    context_slices: int = 1
    n_folds: int = 5
    fold: Optional[int] = None
    n_validation: int = 10
    split_seed: int = 42

# train.py
data = DataSettings(data_dir=args.data_dir, fold=args.fold)      # built once from argparse
torch.save({"state_dict": ..., "data": asdict(data), ...}, path)   # saved in the checkpoint

# predict.py
data = DataSettings(**meta["data"])                                 # read back: identical by construction
validation_data = replace(data, ...)                                # instead of the Namespace hack
```

**Why:** the editor shows you the fields, a typo is an error, and together with S1 the checkpoint can carry the
exact settings.

### 2. Checkpoint as the single source of truth — **recommended**

Not a classic "Gang of Four" pattern, but a well-known rule: **one fact, one place.** See S1. It removes a whole
class of bugs (the old H1 was exactly this: flags differing from training).

### 3. A preprocessing object that can undo itself (Command with undo) — **recommended**

**Problem it solves:** the forward step (crop → pad → resize) is in `dataset.py` + `utils.py`, and the backward step
(`restore_mask_geometry`) is somewhere else. Nobody tested that "backward after forward = nothing changed" exactly,
which is how H1 happened.

**What it looks like:** in the Command pattern, each operation knows how to undo itself. Here:

```python
@dataclass(frozen=True)
class Preprocessing:
    """Crop to the head, pad to a square, resize to `grid` pixels. `inverse` exactly undoes `forward` for masks."""
    geometry: VolumeGeometry

    @classmethod
    def fit(cls, images: np.ndarray, grid: int) -> "Preprocessing": ...      # finds the crop box once

    def forward(self, images: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
    def inverse(self, grid_mask: np.ndarray) -> np.ndarray: ...
```

And one test: `assert np.array_equal(p.inverse(p.forward(images, masks)[1]), masks)`.

**Why:** forward and backward live side by side, so changing one makes you look at the other.

### 4. Strategy as a dictionary for post-processing — small win

[mriseg/utils.py:76-104](mriseg/utils.py:76) chooses the method with `if method == ...`. A table is shorter and
gives the list of names for free:

```python
POSTPROCESS = {"none": keep_all, "drop-small": drop_small_components, "lcc": largest_component}
mask = POSTPROCESS[method](mask)
```

### 5. Cache for preprocessed patients — speed

See S8. "Compute once, store, reuse." A function `load_or_preprocess(patient) -> np.ndarray` that reads
`preprocessed/<patient>.npz` if it exists and otherwise creates it.

## Patterns NOT to add

For a one-person research project these would make the code harder to read, not easier:

- **Observer / callback system** for the training loop (like PyTorch Lightning callbacks). Two plain functions
  (S4) are clearer.
- **Abstract base classes for models, datasets or plotters.** `nn.Module` and `Dataset` already are the
  interfaces.
- **Plugin systems / dependency injection containers.** The `ARCHITECTURES` dict is already enough.
- **Builders** for objects that a dataclass with defaults already handles.

Rule of thumb: add a pattern only when you can name the change it makes easier. If you cannot, plain functions win.
