"""Tests for utils.py: Dice metric, preprocessing and visualization helpers."""

import numpy as np
import pytest

from mriseg.utils import (VolumeGeometry, content_box, dice_similarity_coefficient, draw_contour, grayscale_to_rgb,
                          group_slices_by_patient, normalize_intensity, pad_to_square, postprocess_mask,
                          resize_volume, restore_mask_geometry)


# ------------------------------------------------------------------ Dice

class TestDiceSimilarityCoefficient:

    def test_identical_masks_score_one(self) -> None:
        gt = np.zeros((3, 8, 8), dtype=np.float32)
        gt[1, 2:5, 2:5] = 1
        assert dice_similarity_coefficient(gt.copy(), gt) == pytest.approx(1.0)

    def test_disjoint_masks_score_zero(self) -> None:
        pred = np.zeros((1, 8, 8), dtype=np.float32)
        gt = np.zeros((1, 8, 8), dtype=np.float32)
        pred[0, 0:2, 0:2] = 1
        gt[0, 5:7, 5:7] = 1
        assert dice_similarity_coefficient(pred, gt) == pytest.approx(0.0, abs=1e-6)

    def test_known_partial_overlap(self) -> None:
        # |P| = 2, |G| = 2, |P ∩ G| = 1  ->  Dice = 2*1 / (2+2) = 0.5
        pred = np.zeros((1, 4, 4), dtype=np.float32)
        gt = np.zeros((1, 4, 4), dtype=np.float32)
        pred[0, 0, 0:2] = 1
        gt[0, 0, 1:3] = 1
        assert dice_similarity_coefficient(pred, gt) == pytest.approx(0.5, abs=1e-6)

    def test_both_empty_is_perfect_agreement(self) -> None:
        empty = np.zeros((2, 4, 4), dtype=np.float32)
        assert dice_similarity_coefficient(empty, empty.copy()) == 1.0

    def test_empty_prediction_on_tumour_scores_zero(self) -> None:
        gt = np.zeros((1, 4, 4), dtype=np.float32)
        gt[0, 1:3, 1:3] = 1
        assert dice_similarity_coefficient(np.zeros_like(gt), gt) == pytest.approx(0.0, abs=1e-6)

    def test_all_probabilities_below_threshold_does_not_crash(self) -> None:
        """Regression: sigmoid outputs are never exactly 0; all < 0.5 used to crash LCC with an empty argmax."""
        rng = np.random.default_rng(0)
        pred = rng.uniform(0.01, 0.4, size=(4, 1, 8, 8)).astype(np.float32)
        gt = np.zeros_like(pred)
        gt[2, 0, 2:5, 2:5] = 1

        assert dice_similarity_coefficient(pred, gt) == pytest.approx(0.0, abs=1e-6)
        assert dice_similarity_coefficient(pred, np.zeros_like(gt)) == 1.0

    def test_keeps_stray_components(self) -> None:
        """Dice does no post-processing of its own (postprocess_mask does that)."""
        gt = np.zeros((3, 10, 10), dtype=np.float32)
        gt[1, 2:6, 2:6] = 1
        pred = gt * 0.9
        pred[2, 9, 9] = 0.9  # isolated false-positive voxel, not connected to the tumour
        assert dice_similarity_coefficient(pred, gt) == pytest.approx(2 * 16 / (17 + 16))

    def test_thresholds_probabilities_at_half(self) -> None:
        gt = np.zeros((1, 6, 6), dtype=np.float32)
        gt[0, 1:4, 1:4] = 1
        assert dice_similarity_coefficient(gt * 0.6, gt) == pytest.approx(1.0)  # just above the threshold
        assert dice_similarity_coefficient(gt * 0.5, gt) == pytest.approx(0.0, abs=1e-6)  # 0.5 is background


# ------------------------------------------------------- post-processing

class TestPostprocessMask:

    @staticmethod
    def _three_blobs() -> np.ndarray:
        """Separate 3D components of 27, 8 and 1 voxels."""
        mask = np.zeros((4, 10, 10), dtype=bool)
        mask[0:3, 1:4, 1:4] = True     # 27 voxels over 3 slices
        mask[2:4, 7:9, 7:9] = True     # 8 voxels over 2 slices
        mask[0, 9, 0] = True           # 1 voxel
        return mask

    def test_none_keeps_every_voxel(self) -> None:
        mask = self._three_blobs()
        np.testing.assert_array_equal(postprocess_mask(mask, "none"), mask)

    def test_lcc_keeps_only_the_largest_3d_component(self) -> None:
        out = postprocess_mask(self._three_blobs(), "lcc")
        assert out.sum() == 27 and out[1, 2, 2]

    def test_drop_small_keeps_components_of_at_least_10_percent_of_the_largest(self) -> None:
        out = postprocess_mask(self._three_blobs(), "drop-small")
        assert out.sum() == 27 + 8  # the 1-voxel speck is below 2.7 voxels and dropped
        assert not out[0, 9, 0]

    def test_lcc_matches_medpy_on_probability_volumes(self) -> None:
        """postprocess_mask('lcc') reproduces the medpy-based filter it replaced in predict.py."""
        from medpy.filter.binary import largest_connected_component

        rng = np.random.default_rng(0)
        probabilities = rng.uniform(size=(3, 1, 12, 12))
        binary = probabilities > 0.5
        np.testing.assert_array_equal(postprocess_mask(binary, "lcc"), largest_connected_component(binary))

    def test_empty_mask_stays_empty(self) -> None:
        empty = np.zeros((2, 4, 4), dtype=bool)
        for method in ("none", "lcc", "drop-small"):
            assert not postprocess_mask(empty, method).any()

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown post-processing"):
            postprocess_mask(np.zeros((1, 2, 2)), "largest")


# ---------------------------------------------------------- slice grouping

class TestGroupSlicesByPatient:

    def test_stacks_each_patients_slices(self) -> None:
        slices = [np.full((1, 2, 2), value, dtype=np.float32) for value in range(5)]
        volumes = group_slices_by_patient(slices, [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)], ["A", "B"])
        assert list(volumes) == ["A", "B"]
        assert volumes["A"].shape == (2, 1, 2, 2)
        np.testing.assert_array_equal(volumes["B"][:, 0, 0, 0], [2, 3, 4])

    def test_rejects_missing_or_reordered_slices(self) -> None:
        slices = [np.zeros((1, 2, 2))] * 2
        with pytest.raises(ValueError, match="Got 2 slices for a dataset of 3"):
            group_slices_by_patient(slices, [(0, 0), (0, 1), (0, 2)], ["A"])
        with pytest.raises(ValueError, match="out of order"):
            group_slices_by_patient(slices, [(0, 0), (0, 2)], ["A"])


# --------------------------------------------------------- preprocessing

class TestContentBox:

    def test_box_is_the_bounding_box_of_the_foreground(self) -> None:
        vol = np.zeros((6, 20, 30, 3), dtype=np.float32)
        vol[1:4, 5:12, 7:20, :] = 100.0
        seg = np.zeros((6, 20, 30), dtype=np.float32)
        seg[2, 6:8, 8:10] = 1

        box = content_box(vol)

        assert box == (slice(1, 4), slice(5, 12), slice(7, 20))
        assert seg[box].sum() == seg.sum()  # tumour lies inside the brain box, nothing lost

    def test_ignores_voxels_below_background_threshold(self) -> None:
        vol = np.zeros((2, 10, 10, 1), dtype=np.float32)
        vol[:, 2:8, 2:8] = 100.0
        vol[:, 0, 0] = 5.0  # 5% of max, below the default 10% threshold
        assert vol[content_box(vol)].shape == (2, 6, 6, 1)


class TestRestoreMaskGeometry:

    @staticmethod
    def _preprocess(height: int, width: int, grid_size: int) -> tuple:
        """A 9-slice scan whose content box is 5 slices x height x width; returns (original mask, grid mask, geometry)."""
        vol = np.zeros((9, 40, 40, 3), dtype=np.float32)
        vol[2:7, 5:5 + height, 8:8 + width] = 100.0
        mask = np.zeros((9, 40, 40), dtype=np.float32)
        mask[3:5, 10:16, 12:15] = 1

        inner_vol, inner_mask = vol[1:-1], mask[1:-1]  # the dataset drops the first and last slice
        z, y, x = box = content_box(inner_vol)
        geometry = VolumeGeometry(n_slices=9, height=40, width=40, z_start=z.start + 1, y_start=y.start,
                                  x_start=x.start, crop_slices=z.stop - z.start, crop_height=y.stop - y.start,
                                  crop_width=x.stop - x.start, grid_size=grid_size)
        _, grid_mask = resize_volume(pad_to_square((inner_vol[box], inner_mask[box])), target_size=grid_size)
        return mask, grid_mask, geometry

    def test_round_trip_is_exact_without_resizing(self) -> None:
        mask, grid_mask, geometry = self._preprocess(height=20, width=14, grid_size=20)  # padded to 20 = grid
        restored = restore_mask_geometry(grid_mask, geometry)
        assert restored.shape == (9, 40, 40)
        np.testing.assert_array_equal(restored, mask > 0.5)

    @pytest.mark.parametrize("height, width, grid_size", [(20, 14, 64), (31, 27, 40), (17, 23, 256)])
    def test_round_trip_is_exact_after_resizing(self, height: int, width: int, grid_size: int) -> None:
        # A perfect prediction on the grid must come back on exactly the original pixels: not shifted, none lost
        mask, grid_mask, geometry = self._preprocess(height=height, width=width, grid_size=grid_size)
        restored = restore_mask_geometry(grid_mask, geometry)
        np.testing.assert_array_equal(restored, mask > 0.5)


class TestPadToSquare:

    def test_pads_width_symmetrically(self) -> None:
        vol = np.ones((2, 6, 3, 3), dtype=np.float32)
        seg = np.ones((2, 6, 3), dtype=np.float32)

        vol_p, seg_p = pad_to_square((vol, seg))

        assert vol_p.shape == (2, 6, 6, 3)
        assert seg_p.shape == (2, 6, 6)
        # gap 3 -> floor 1 column left, ceil 2 columns right
        assert np.all(seg_p[:, :, 0] == 0)
        assert np.all(seg_p[:, :, 1:4] == 1)
        assert np.all(seg_p[:, :, 4:] == 0)

    def test_pads_height(self) -> None:
        vol, seg = pad_to_square((np.ones((1, 2, 6, 3)), np.ones((1, 2, 6))))
        assert vol.shape == (1, 6, 6, 3)
        assert seg.shape == (1, 6, 6)
        assert seg.sum() == 12

    def test_square_input_unchanged(self) -> None:
        vol = np.random.default_rng(0).random((2, 5, 5, 3))
        seg = np.zeros((2, 5, 5))
        vol_p, seg_p = pad_to_square((vol, seg))
        assert vol_p is vol and seg_p is seg


class TestResizeVolume:

    def test_output_shapes(self) -> None:
        vol = np.random.default_rng(0).random((4, 20, 20, 3)).astype(np.float32)
        seg = np.zeros((4, 20, 20), dtype=np.float32)
        vol_r, seg_r = resize_volume((vol, seg), target_size=32)
        assert vol_r.shape == (4, 32, 32, 3)
        assert seg_r.shape == (4, 32, 32)

    def test_mask_stays_binary(self) -> None:
        seg = np.zeros((2, 17, 17), dtype=np.float32)
        seg[:, 4:11, 5:9] = 1
        _, seg_r = resize_volume((np.zeros((2, 17, 17, 3), dtype=np.float32), seg), target_size=40)
        assert set(np.unique(seg_r)) <= {0.0, 1.0}
        assert seg_r.sum() > 0

    def test_mask_stays_on_the_image(self) -> None:
        # The same square drawn into the image and the mask must land on the same grid pixels after resizing
        seg = np.zeros((1, 21, 21), dtype=np.float32)
        seg[0, 5:12, 7:15] = 1
        vol = np.repeat(seg[..., None], 3, axis=-1) * 100.0
        vol_r, seg_r = resize_volume((vol, seg), target_size=64)

        image_rows, image_cols = np.nonzero(vol_r[0, ..., 0] > 50)
        mask_rows, mask_cols = np.nonzero(seg_r[0])
        assert abs(image_rows.mean() - mask_rows.mean()) < 0.3
        assert abs(image_cols.mean() - mask_cols.mean()) < 0.3

    def test_volume_clamped_to_original_range(self) -> None:
        # Bicubic interpolation overshoots at sharp edges; resize_volume clamps it back
        vol = np.zeros((1, 10, 10, 1), dtype=np.float32)
        vol[0, :, 5:, 0] = 255.0
        vol_r, _ = resize_volume((vol, np.zeros((1, 10, 10), dtype=np.float32)), target_size=37)
        assert vol_r.min() >= 0.0
        assert vol_r.max() <= 255.0


class TestNormalizeIntensity:

    @staticmethod
    def _scan(n_background: int) -> np.ndarray:
        """The same 5000 head voxels (3 channels, each its own distribution) after n_background black voxels."""
        rng = np.random.default_rng(0)
        head = np.stack([rng.normal(100, 20, 5000).clip(30),
                         rng.uniform(200, 1000, 5000),
                         rng.exponential(50, 5000) + 30], axis=-1)
        return np.concatenate([np.zeros((n_background, 3)), head])[None, None]  # (Z, H, W, C) = (1, 1, N, 3)

    def test_head_voxels_get_zero_mean_unit_std(self) -> None:
        out = normalize_intensity(self._scan(n_background=2000))
        head = out[0, 0, 2000:]
        np.testing.assert_allclose(head.mean(axis=0), 0.0, atol=1e-4)
        np.testing.assert_allclose(head.std(axis=0), 1.0, atol=1e-4)

    def test_result_does_not_depend_on_the_amount_of_background(self) -> None:
        little = normalize_intensity(self._scan(n_background=500))[0, 0, 500:]
        much = normalize_intensity(self._scan(n_background=20000))[0, 0, 20000:]
        np.testing.assert_allclose(little, much, atol=1e-5)

    def test_background_becomes_the_channel_minimum(self) -> None:
        out = normalize_intensity(self._scan(n_background=100))
        np.testing.assert_allclose(out[0, 0, 0], out.min(axis=(0, 1, 2)))

    def test_constant_channel_gives_no_nan(self) -> None:
        vol = np.random.default_rng(0).random((2, 8, 8, 3))
        vol[..., 2] = 0.0
        out = normalize_intensity(vol.copy())
        assert np.all(np.isfinite(out))


# --------------------------------------------------------- visualization

class TestGrayscaleToRgb:

    def test_output_is_uint8_rgb_spanning_full_range(self) -> None:
        img = np.linspace(-3.0, 5.0, 64).reshape(8, 8)
        rgb = grayscale_to_rgb(img)
        assert rgb.shape == (8, 8, 3)
        assert rgb.dtype == np.uint8
        assert rgb.min() == 0 and rgb.max() == 255
        assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])

    def test_constant_image_does_not_divide_by_zero(self) -> None:
        rgb = grayscale_to_rgb(np.zeros((4, 4)))
        assert np.all(rgb == 0)


class TestDrawContour:

    def test_colours_only_the_boundary(self) -> None:
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        mask = np.zeros((7, 7))
        mask[1:6, 1:6] = 1

        out = draw_contour(image, mask, color=[255, 0, 0])
        coloured = np.all(out == [255, 0, 0], axis=-1)

        expected = np.zeros((7, 7), dtype=bool)
        expected[1:6, 1:6] = True
        expected[2:5, 2:5] = False  # interior pixels have no background neighbour
        assert np.array_equal(coloured, expected)

    def test_matches_the_original_pixel_loop(self) -> None:
        """The vectorized boundary equals the old per-pixel check: a mask pixel with a background 8-neighbour."""
        def reference(mask: np.ndarray) -> np.ndarray:
            binary = np.round(mask)
            boundary = np.zeros(mask.shape, dtype=bool)
            for r, c in zip(*np.nonzero(binary)):
                neighbourhood = binary[max(0, r - 1): r + 2, max(0, c - 1): c + 2]
                boundary[r, c] = 0.0 < np.mean(neighbourhood) < 1.0
            return boundary

        rng = np.random.default_rng(0)
        for _ in range(50):
            mask = (rng.random((24, 24)) < 0.5).astype(np.float64)  # many components touching the image border
            out = draw_contour(np.zeros((24, 24, 3), dtype=np.uint8), mask, color=[255, 0, 0])
            np.testing.assert_array_equal(np.all(out == [255, 0, 0], axis=-1), reference(mask))

    def test_empty_mask_leaves_image_unchanged(self) -> None:
        image = np.full((5, 5, 3), 7, dtype=np.uint8)
        out = draw_contour(image.copy(), np.zeros((5, 5)), color=[0, 255, 0])
        assert np.array_equal(out, image)
