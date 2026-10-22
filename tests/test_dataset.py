"""Tests for dataset.py on a tiny synthetic folder in the Kaggle layout: patient discovery, splits, 2.5D input and slice sampling."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import write_patient
from mriseg.dataset import MRISegmentationDataset as SegDataset, find_patients, split_patients

SIZE = 16


@pytest.fixture
def data_root(tmp_path: Path) -> str:
    # First and last files are dropped by the dataset: 7 files -> 5 slices per patient
    write_patient(tmp_path, "TCGA_AA_0001_20000101", n_slices=7, tumour_slices=(3, 4))
    write_patient(tmp_path, "TCGA_AA_0002_20000101", n_slices=7, tumour_slices=())
    return str(tmp_path)


def _dataset(data_root: str, context_slices: int) -> SegDataset:
    return SegDataset(data_root, sorted(find_patients(data_root)), resolution=SIZE, context_slices=context_slices)


def test_channel_helpers() -> None:
    assert SegDataset.num_input_channels(0) == 3
    assert SegDataset.num_input_channels(1) == 9
    assert SegDataset.flair_channel(0) == 1  # channels: pre-contrast, FLAIR, post-contrast
    assert SegDataset.flair_channel(2) == 7  # the middle slice of five


def test_context_slices_stack_the_neighbours_in_slice_order(data_root: str) -> None:
    plain, context = _dataset(data_root, 0), _dataset(data_root, 1)
    idx = context.flat_index.index((0, 2))  # a middle slice of the first patient

    image, mask = context[idx]
    assert image.shape == (9, SIZE, SIZE)
    np.testing.assert_array_equal(image[0:3].numpy(), plain[idx - 1][0].numpy())
    np.testing.assert_array_equal(image[3:6].numpy(), plain[idx][0].numpy())
    np.testing.assert_array_equal(image[6:9].numpy(), plain[idx + 1][0].numpy())
    np.testing.assert_array_equal(mask.numpy(), plain[idx][1].numpy())


def test_neighbours_beyond_the_volume_are_background(data_root: str) -> None:
    context = _dataset(data_root, 1)
    first, last = context.flat_index.index((0, 0)), context.flat_index.index((0, 4))
    vol = context.volumes[0][0]
    background = vol.min(axis=(0, 1, 2))

    np.testing.assert_allclose(context.backgrounds[0], background)
    for idx, outside in ((first, slice(0, 3)), (last, slice(6, 9))):
        image = context[idx][0].numpy()
        np.testing.assert_allclose(image[outside], np.broadcast_to(background[:, None, None], (3, SIZE, SIZE)))


def test_sampling_weights_give_tumour_free_slices_half_of_each_patient(data_root: str) -> None:
    dataset = _dataset(data_root, 0)
    weights = np.array(dataset.sample_weights)
    patient = np.array([p for p, _ in dataset.flat_index])
    has_tumour = np.array([bool(dataset.volumes[p][1][s].any()) for p, s in dataset.flat_index])

    for p in (0, 1):
        assert weights[patient == p].sum() == pytest.approx(1.0)  # every patient equally likely

    # Patient 0: 5 slices, 2 with tumour. Tumour-free slices get 1 / (2 * 5) each
    np.testing.assert_allclose(weights[(patient == 0) & ~has_tumour], 0.1)
    # Patient 1 has no tumour: uniform weights instead of all-zero (which would never be sampled)
    np.testing.assert_allclose(weights[patient == 1], 0.2)


def test_negative_context_slices_raises(data_root: str) -> None:
    with pytest.raises(ValueError, match="context_slices"):
        _dataset(data_root, -1)


def test_missing_mask_file_raises_instead_of_shifting_the_masks(tmp_path: Path) -> None:
    write_patient(tmp_path, "TCGA_AA_0001_20000101", n_slices=7, tumour_slices=(4,), mask_slices=(1, 2, 4, 5, 6, 7))
    with pytest.raises(ValueError, match=r"slices \[3\] have no \*_mask.tif file"):
        SegDataset(str(tmp_path), ["TCGA_AA_0001_20000101"], resolution=SIZE)


def test_patients_without_masks_load_for_prediction_only(tmp_path: Path) -> None:
    write_patient(tmp_path, "TCGA_AA_0001_20000101", n_slices=7, tumour_slices=(3,))
    write_patient(tmp_path, "TCGA_NN_0002_20000101", n_slices=6, tumour_slices=(), mask_slices=())
    patients = sorted(find_patients(str(tmp_path)))

    dataset = SegDataset(str(tmp_path), patients, resolution=SIZE, require_masks=False)
    assert not dataset.has_masks
    assert dataset.image_files["TCGA_NN_0002_20000101"] == [f"TCGA_NN_0002_20000101_{i}.tif" for i in range(1, 7)]
    geometry = dataset.geometry["TCGA_NN_0002_20000101"]
    assert (geometry.n_slices, geometry.height, geometry.width) == (6, SIZE, SIZE)
    assert geometry.z_start == 1 and geometry.crop_slices == 4  # first and last slice dropped
    assert not dataset.volumes[1][1].any()  # placeholder masks are empty

    with pytest.raises(ValueError, match="patients without masks can only be predicted"):
        SegDataset(str(tmp_path), patients, resolution=SIZE)


def test_labelled_dataset_has_masks(data_root: str) -> None:
    assert _dataset(data_root, 0).has_masks


def test_unknown_patient_raises(data_root: str) -> None:
    with pytest.raises(ValueError, match="not in"):
        SegDataset(data_root, ["TCGA_XX_9999_20000101"], resolution=SIZE)


def test_original_masks_are_every_slice_at_full_size(data_root: str) -> None:
    dataset = _dataset(data_root, 0)
    masks = dataset.original_masks("TCGA_AA_0001_20000101")
    assert masks.shape == (7, SIZE, SIZE) and masks.dtype == bool  # dropped first / last slice included
    assert [bool(m.any()) for m in masks] == [False, False, True, True, False, False, False]  # tumour on files 3, 4


def test_slices_pair_up_by_number_not_by_name_order(tmp_path: Path) -> None:
    # As text, "..._10.tif" sorts before "..._2.tif"; the slices must still be in numeric order
    write_patient(tmp_path, "TCGA_AA_0001_20000101", n_slices=11, tumour_slices=(10,))
    dataset = SegDataset(str(tmp_path), ["TCGA_AA_0001_20000101"], resolution=SIZE)
    assert dataset.image_files["TCGA_AA_0001_20000101"][1:3] == ["TCGA_AA_0001_20000101_2.tif",
                                                                  "TCGA_AA_0001_20000101_3.tif"]
    assert dataset.mask_files["TCGA_AA_0001_20000101"][9] == "TCGA_AA_0001_20000101_10_mask.tif"


# ------------------------------------------------------- patients and splits

def test_identical_copy_of_a_patient_folder_is_ignored(data_root: str) -> None:
    shutil.copytree(Path(data_root) / "TCGA_AA_0001_20000101", Path(data_root) / "copy" / "TCGA_AA_0001_20000101")
    patients = find_patients(data_root)
    assert sorted(patients) == ["TCGA_AA_0001_20000101", "TCGA_AA_0002_20000101"]
    assert patients["TCGA_AA_0001_20000101"] == os.path.join(data_root, "TCGA_AA_0001_20000101")  # first in path order


def test_different_folders_with_the_same_name_raise(data_root: str) -> None:
    write_patient(Path(data_root) / "other", "TCGA_AA_0001_20000101", n_slices=4, tumour_slices=())
    with pytest.raises(ValueError, match="Two different folders"):
        find_patients(data_root)


def test_single_split_has_no_test_patients() -> None:
    patients = [f"P{i:03d}" for i in range(110)]
    splits = split_patients(patients, fold=None, n_folds=5, n_validation=10, seed=42)
    assert len(splits["train"]) == 100 and len(splits["validation"]) == 10 and splits["test"] == []
    assert not set(splits["train"]) & set(splits["validation"])
    assert split_patients(list(reversed(patients)), None, 5, 10, 42) == splits  # order of the input does not matter


def test_fold_split_is_the_kfold_split() -> None:
    patients = [f"P{i:03d}" for i in range(20)]
    splits = split_patients(patients, fold=1, n_folds=5, n_validation=3, seed=42)
    assert len(splits["test"]) == 4 and len(splits["validation"]) == 3 and len(splits["train"]) == 13
