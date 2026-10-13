"""Tests for dataset.MRISegmentationDataset on a tiny synthetic folder in the Kaggle layout: 2.5D input and slice sampling."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from skimage.io import imsave

from dataset import MRISegmentationDataset as SegDataset

SIZE = 16


def _write_patient(root: Path, pid: str, n_slices: int, tumour_slices: Sequence[int]) -> None:
    """Slices <pid>_<i>.tif (RGB) and <pid>_<i>_mask.tif numbered from 1, as in the Kaggle dataset."""
    folder = root / pid
    folder.mkdir(parents=True)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    head = (yy - 8) ** 2 + (xx - 8) ** 2 <= 36  # round, so the cropped image still has background corners

    for i in range(1, n_slices + 1):
        image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        image[head] = (20 * i, 30 + 10 * i, 200 - 20 * i)  # every slice and channel differs
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        if i in tumour_slices:
            mask[6:10, 6:10] = 255
        imsave(str(folder / f"{pid}_{i}.tif"), image, check_contrast=False)
        imsave(str(folder / f"{pid}_{i}_mask.tif"), mask, check_contrast=False)


@pytest.fixture
def data_root(tmp_path: Path) -> str:
    # First and last files are dropped by the dataset: 7 files -> 5 slices per patient
    _write_patient(tmp_path, "TCGA_AA_0001_20000101", n_slices=7, tumour_slices=(3, 4))
    _write_patient(tmp_path, "TCGA_AA_0002_20000101", n_slices=7, tumour_slices=())
    return str(tmp_path)


def _dataset(data_root: str, context_slices: int) -> SegDataset:
    return SegDataset(data_root, resolution=SIZE, split="all", context_slices=context_slices)


def test_channel_helpers() -> None:
    assert SegDataset.num_input_channels(0) == 3
    assert SegDataset.num_input_channels(1) == 9
    assert SegDataset.center_slice_channels(0) == slice(0, 3)
    assert SegDataset.center_slice_channels(2) == slice(6, 9)


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
