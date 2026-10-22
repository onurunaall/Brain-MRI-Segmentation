"""Tests for augmentations.py: background-filled padding and the intensity augmentations."""

import numpy as np
import pytest
import torch

from mriseg.augmentations import (AugmentationPipeline, RandomBrightness, RandomGamma, RandomRotation, RandomScale,
                                  build_augmentation_pipeline)


def _zscored_slice(size: int = 32, channels: int = 3) -> tuple:
    """(image, mask) like the dataset serves: z-scored, so the background is negative, not 0."""
    rng = np.random.default_rng(0)
    image = np.full((size, size, channels), -0.8, dtype=np.float32)
    image[8:24, 8:24] = rng.uniform(0.0, 2.0, (16, 16, channels))
    mask = np.zeros((size, size, 1), dtype=np.float32)
    mask[12:20, 12:20] = 1.0
    return image, mask


@pytest.mark.parametrize("stage", [RandomScale(0.3), RandomRotation(30.0)])
def test_padding_and_rotation_fill_with_the_background_not_zero(stage: object) -> None:
    torch.manual_seed(1)
    image, mask = _zscored_slice()
    for _ in range(5):
        out, _ = AugmentationPipeline([stage])((image, mask))
        # Without the background shift the corners would be filled with 0, brighter than the -0.8 background
        np.testing.assert_allclose(out[0, 0], image[0, 0], atol=1e-6)
        np.testing.assert_allclose(out[-1, -1], image[-1, -1], atol=1e-6)
        assert out.min() >= image.min() - 1e-6


def test_random_scale_keeps_the_mask_on_the_image() -> None:
    # The same circle in the image and the mask must stay in the same place; nearest resizing that is not centred
    # would shift every training label half a pixel down-right of its image
    torch.manual_seed(0)
    yy, xx = np.mgrid[:64, :64]
    circle = torch.from_numpy(((yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 100).astype(np.float32))[None]
    for _ in range(20):
        image, mask = RandomScale(0.05)(circle.clone(), circle.clone())
        image_rows, image_cols = np.nonzero((image[0] > 0.5).numpy())
        mask_rows, mask_cols = np.nonzero((mask[0] > 0.5).numpy())
        assert abs(image_rows.mean() - mask_rows.mean()) < 0.3
        assert abs(image_cols.mean() - mask_cols.mean()) < 0.3


def test_intensity_stages_keep_background_shape_and_mask() -> None:
    torch.manual_seed(0)
    image, mask = _zscored_slice(channels=9)  # 2.5D input
    pipeline = build_augmentation_pipeline(brightness_range=0.2, gamma_range=0.4)

    out_image, out_mask = pipeline((image, mask))

    assert out_image.shape == image.shape
    np.testing.assert_array_equal(out_mask, mask)
    np.testing.assert_allclose(out_image[0, 0], image[0, 0], atol=1e-6)  # background unchanged
    assert not np.allclose(out_image, image)


def test_brightness_factor_per_sequence_shared_by_context_slices() -> None:
    torch.manual_seed(0)
    image = torch.ones(9, 4, 4)  # 3 context slices x 3 MRI sequences
    out, _ = RandomBrightness(0.2)(image, torch.zeros(1, 4, 4))
    assert 0.8 - 1e-6 <= float(out.min()) and float(out.max()) <= 1.2 + 1e-6
    factors = out[:, 0, 0].reshape(3, 3)  # rows = context slices, columns = sequences
    torch.testing.assert_close(factors, factors[:1].expand(3, 3))  # same factor for a sequence in every slice
    assert float(factors[0].std()) > 0  # but each sequence has its own


def test_gamma_keeps_zero_and_peak_and_is_shared_by_context_slices() -> None:
    torch.manual_seed(0)
    image = torch.linspace(0.0, 3.0, 16).reshape(1, 4, 4).repeat(6, 1, 1)  # 2 context slices x 3 sequences
    out, _ = RandomGamma(0.4)(image, torch.zeros(1, 4, 4))
    torch.testing.assert_close(out[:, 0, 0], torch.zeros(6))
    torch.testing.assert_close(out[:, -1, -1], torch.full((6,), 3.0))
    torch.testing.assert_close(out[:3], out[3:])  # identical input slices stay identical
    assert float(out.min()) >= 0.0


def test_zero_ranges_are_the_identity() -> None:
    image, mask = _zscored_slice()
    out, _ = build_augmentation_pipeline(brightness_range=0.0, gamma_range=0.0)((image, mask))
    np.testing.assert_allclose(out, image, atol=1e-5)
