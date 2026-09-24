"""Tests for utils.py: Dice metric, preprocessing and visualization helpers."""

import numpy as np
import pytest

from utils import (crop_to_content, dice_similarity_coefficient, draw_contour, grayscale_to_rgb,
                   normalize_intensity, pad_to_square, resize_volume)


# ------------------------------------------------------------------ Dice

class TestDiceSimilarityCoefficient:

    def test_identical_masks_score_one(self):
        gt = np.zeros((3, 8, 8), dtype=np.float32)
        gt[1, 2:5, 2:5] = 1
        assert dice_similarity_coefficient(gt.copy(), gt) == pytest.approx(1.0)

    def test_disjoint_masks_score_zero(self):
        pred = np.zeros((1, 8, 8), dtype=np.float32)
        gt = np.zeros((1, 8, 8), dtype=np.float32)
        pred[0, 0:2, 0:2] = 1
        gt[0, 5:7, 5:7] = 1
        assert dice_similarity_coefficient(pred, gt) == pytest.approx(0.0, abs=1e-6)

    def test_known_partial_overlap(self):
        # |P| = 2, |G| = 2, |P ∩ G| = 1  ->  Dice = 2*1 / (2+2) = 0.5
        pred = np.zeros((1, 4, 4), dtype=np.float32)
        gt = np.zeros((1, 4, 4), dtype=np.float32)
        pred[0, 0, 0:2] = 1
        gt[0, 0, 1:3] = 1
        assert dice_similarity_coefficient(pred, gt) == pytest.approx(0.5, abs=1e-6)

    def test_both_empty_is_perfect_agreement(self):
        empty = np.zeros((2, 4, 4), dtype=np.float32)
        assert dice_similarity_coefficient(empty, empty.copy()) == 1.0

    def test_empty_prediction_on_tumour_scores_zero(self):
        gt = np.zeros((1, 4, 4), dtype=np.float32)
        gt[0, 1:3, 1:3] = 1
        assert dice_similarity_coefficient(np.zeros_like(gt), gt) == pytest.approx(0.0, abs=1e-6)

    def test_all_probabilities_below_threshold_does_not_crash(self):
        """Regression: sigmoid outputs are never exactly 0; all < 0.5 used to crash LCC with an empty argmax."""
        rng = np.random.default_rng(0)
        pred = rng.uniform(0.01, 0.4, size=(4, 1, 8, 8)).astype(np.float32)
        gt = np.zeros_like(pred)
        gt[2, 0, 2:5, 2:5] = 1

        assert dice_similarity_coefficient(pred, gt) == pytest.approx(0.0, abs=1e-6)
        assert dice_similarity_coefficient(pred, np.zeros_like(gt)) == 1.0

    def test_lcc_removes_stray_component(self):
        gt = np.zeros((3, 10, 10), dtype=np.float32)
        gt[1, 2:6, 2:6] = 1
        pred = gt * 0.9
        pred[2, 9, 9] = 0.9  # isolated false-positive voxel, not connected to the tumour

        with_lcc = dice_similarity_coefficient(pred, gt, apply_lcc=True)
        without_lcc = dice_similarity_coefficient(np.round(pred), gt, apply_lcc=False)

        assert with_lcc == pytest.approx(1.0)
        assert without_lcc == pytest.approx(2 * 16 / (17 + 16))

    def test_lcc_thresholds_probabilities_at_half(self):
        gt = np.zeros((1, 6, 6), dtype=np.float32)
        gt[0, 1:4, 1:4] = 1
        pred = gt * 0.6  # every tumour voxel just above threshold
        assert dice_similarity_coefficient(pred, gt) == pytest.approx(1.0)


# --------------------------------------------------------- preprocessing

class TestCropToContent:

    def test_crops_to_bounding_box_of_foreground(self):
        vol = np.zeros((6, 20, 30, 3), dtype=np.float32)
        vol[1:4, 5:12, 7:20, :] = 100.0
        seg = np.zeros((6, 20, 30), dtype=np.float32)
        seg[2, 6:8, 8:10] = 1

        vol_c, seg_c = crop_to_content((vol, seg))

        assert vol_c.shape == (3, 7, 13, 3)
        assert seg_c.shape == (3, 7, 13)
        assert seg_c.sum() == seg.sum()  # tumour lies inside the brain box, nothing lost

    def test_ignores_voxels_below_background_threshold(self):
        vol = np.zeros((2, 10, 10, 1), dtype=np.float32)
        vol[:, 2:8, 2:8] = 100.0
        vol[:, 0, 0] = 5.0  # 5% of max, below the default 10% threshold
        vol_c, _ = crop_to_content((vol, np.zeros((2, 10, 10))))
        assert vol_c.shape == (2, 6, 6, 1)


class TestPadToSquare:

    def test_pads_width_symmetrically(self):
        vol = np.ones((2, 6, 3, 3), dtype=np.float32)
        seg = np.ones((2, 6, 3), dtype=np.float32)

        vol_p, seg_p = pad_to_square((vol, seg))

        assert vol_p.shape == (2, 6, 6, 3)
        assert seg_p.shape == (2, 6, 6)
        # gap 3 -> floor 1 column left, ceil 2 columns right
        assert np.all(seg_p[:, :, 0] == 0)
        assert np.all(seg_p[:, :, 1:4] == 1)
        assert np.all(seg_p[:, :, 4:] == 0)

    def test_pads_height(self):
        vol, seg = pad_to_square((np.ones((1, 2, 6, 3)), np.ones((1, 2, 6))))
        assert vol.shape == (1, 6, 6, 3)
        assert seg.shape == (1, 6, 6)
        assert seg.sum() == 12

    def test_square_input_unchanged(self):
        vol = np.random.default_rng(0).random((2, 5, 5, 3))
        seg = np.zeros((2, 5, 5))
        vol_p, seg_p = pad_to_square((vol, seg))
        assert vol_p is vol and seg_p is seg


class TestResizeVolume:

    def test_output_shapes(self):
        vol = np.random.default_rng(0).random((4, 20, 20, 3)).astype(np.float32)
        seg = np.zeros((4, 20, 20), dtype=np.float32)
        vol_r, seg_r = resize_volume((vol, seg), target_size=32)
        assert vol_r.shape == (4, 32, 32, 3)
        assert seg_r.shape == (4, 32, 32)

    def test_mask_stays_binary(self):
        seg = np.zeros((2, 17, 17), dtype=np.float32)
        seg[:, 4:11, 5:9] = 1
        _, seg_r = resize_volume((np.zeros((2, 17, 17, 3), dtype=np.float32), seg), target_size=40)
        assert set(np.unique(seg_r)) <= {0.0, 1.0}
        assert seg_r.sum() > 0

    def test_volume_clamped_to_original_range(self):
        # Bicubic interpolation overshoots at sharp edges; resize_volume clamps it back
        vol = np.zeros((1, 10, 10, 1), dtype=np.float32)
        vol[0, :, 5:, 0] = 255.0
        vol_r, _ = resize_volume((vol, np.zeros((1, 10, 10), dtype=np.float32)), target_size=37)
        assert vol_r.min() >= 0.0
        assert vol_r.max() <= 255.0


class TestNormalizeIntensity:

    def test_each_channel_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        vol = np.stack([rng.normal(100, 20, (4, 16, 16)),
                        rng.uniform(0, 1000, (4, 16, 16)),
                        rng.exponential(5, (4, 16, 16))], axis=-1)

        out = normalize_intensity(vol.copy())

        np.testing.assert_allclose(out.mean(axis=(0, 1, 2)), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.std(axis=(0, 1, 2)), 1.0, atol=1e-6)

    def test_constant_channel_gives_no_nan(self):
        vol = np.random.default_rng(0).random((2, 8, 8, 3))
        vol[..., 2] = 0.0
        out = normalize_intensity(vol.copy())
        assert np.all(np.isfinite(out))


# --------------------------------------------------------- visualization

class TestGrayscaleToRgb:

    def test_output_is_uint8_rgb_spanning_full_range(self):
        img = np.linspace(-3.0, 5.0, 64).reshape(8, 8)
        rgb = grayscale_to_rgb(img)
        assert rgb.shape == (8, 8, 3)
        assert rgb.dtype == np.uint8
        assert rgb.min() == 0 and rgb.max() == 255
        assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])

    def test_constant_image_does_not_divide_by_zero(self):
        rgb = grayscale_to_rgb(np.zeros((4, 4)))
        assert np.all(rgb == 0)


class TestDrawContour:

    def test_colours_only_the_boundary(self):
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        mask = np.zeros((7, 7))
        mask[1:6, 1:6] = 1

        out = draw_contour(image, mask, color=[255, 0, 0])
        coloured = np.all(out == [255, 0, 0], axis=-1)

        expected = np.zeros((7, 7), dtype=bool)
        expected[1:6, 1:6] = True
        expected[2:5, 2:5] = False  # interior pixels have no background neighbour
        assert np.array_equal(coloured, expected)

    def test_empty_mask_leaves_image_unchanged(self):
        image = np.full((5, 5, 3), 7, dtype=np.uint8)
        out = draw_contour(image.copy(), np.zeros((5, 5)), color=[0, 255, 0])
        assert np.array_equal(out, image)
