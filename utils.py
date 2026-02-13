"""
Utility functions for MRI segmentation preprocessing and evaluation.

Provides helper routines for volume cropping, padding, resizing,
normalization, Dice score computation, and visualization overlays.
"""

from typing import Tuple, List, Optional

import numpy as np
import numpy.typing as npt
from medpy.filter.binary import largest_connected_component
from skimage.exposure import rescale_intensity
from skimage.transform import resize


def dice_similarity_coefficient(prediction: npt.NDArray[np.float32],
                                ground_truth: npt.NDArray[np.float32],
                                apply_lcc: bool = True,
                                epsilon: float = 1e-6) -> float:
    """
    Calculate Dice Similarity Coefficient (DSC) between prediction and ground truth.

    :param prediction: Predicted segmentation mask
    :param ground_truth: Ground truth segmentation mask
    :param apply_lcc: If True, retain only the largest connected component in prediction
    :param epsilon: Small constant for numerical stability
    :return: Dice coefficient in range [0.0, 1.0]
    """
    # Optionally binarize and keep only the largest connected component
    if apply_lcc and np.any(prediction):
        prediction = np.round(prediction).astype(np.int32)
        ground_truth = np.round(ground_truth).astype(np.int32)
        prediction = largest_connected_component(prediction)

    overlap = np.sum(prediction[ground_truth == 1])
    total = np.sum(prediction) + np.sum(ground_truth)

    # Both masks empty → perfect agreement by convention
    if total == 0:
        return 1.0

    return (2.0 * overlap + epsilon) / (total + epsilon)


def crop_to_content(sample: Tuple[npt.NDArray, npt.NDArray],
                    bg_threshold_ratio: float = 0.1) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Crop volume and mask to the tight bounding box enclosing non-background voxels.

    :param sample: Tuple of (volume, mask) arrays with shape (Z, H, W, C) and (Z, H, W)
    :param bg_threshold_ratio: Fraction of max intensity below which voxels are treated as background
    :return: Cropped (volume, mask) tuple
    """
    vol, seg = sample

    # Suppress background voxels by thresholding
    vol_thresh = vol.copy()
    vol_thresh[vol_thresh < np.max(vol) * bg_threshold_ratio] = 0

    # Project along each axis to find non-zero extent
    # Slice axis (Z)
    z_proj = np.max(vol_thresh, axis=(1, 2, 3))
    z_nz = np.nonzero(z_proj)[0]
    z_lo, z_hi = z_nz.min(), z_nz.max() + 1

    # Height axis (Y)
    y_proj = np.max(vol_thresh, axis=(0, 2, 3))
    y_nz = np.nonzero(y_proj)[0]
    y_lo, y_hi = y_nz.min(), y_nz.max() + 1

    # Width axis (X)
    x_proj = np.max(vol_thresh, axis=(0, 1, 3))
    x_nz = np.nonzero(x_proj)[0]
    x_lo, x_hi = x_nz.min(), x_nz.max() + 1

    return (vol[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi], seg[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi])


def pad_to_square(sample: Tuple[npt.NDArray, npt.NDArray]) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Pad volume and mask so that the spatial dimensions (H, W) become square.

    :param sample: Tuple of (volume, mask) arrays
    :return: Padded (volume, mask) tuple with equal H and W
    """
    vol, seg = sample
    h_dim, w_dim = vol.shape[1], vol.shape[2]

    if h_dim == w_dim:
        return vol, seg

    # Compute symmetric padding for the shorter axis
    gap = (max(h_dim, w_dim) - min(h_dim, w_dim)) / 2.0

    if h_dim > w_dim:
        # Pad width dimension
        pad_spec = ((0, 0), (0, 0), (int(np.floor(gap)), int(np.ceil(gap))))
    else:
        # Pad height dimension
        pad_spec = ((0, 0), (int(np.floor(gap)), int(np.ceil(gap))), (0, 0))

    seg = np.pad(seg, pad_spec, mode="constant", constant_values=0)

    # Volume has an extra channel axis → extend the padding tuple
    vol_pad_spec = pad_spec + ((0, 0),)
    vol = np.pad(vol, vol_pad_spec, mode="constant", constant_values=0)

    return vol, seg


def resize_volume(sample: Tuple[npt.NDArray, npt.NDArray],
                  target_size: int = 256) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Resize spatial dimensions of volume and mask to (target_size × target_size).

    :param sample: Tuple of (volume, mask) arrays
    :param target_size: Desired spatial resolution (pixels)
    :return: Resized (volume, mask) tuple
    """
    vol, seg = sample
    n_slices = vol.shape[0]

    # Mask: nearest-neighbour interpolation to preserve binary labels
    seg_out_shape = (n_slices, target_size, target_size)
    seg = resize(
        seg,
        output_shape=seg_out_shape,
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )

    # Volume: bi-quadratic interpolation for smooth intensity resampling
    vol_out_shape = (n_slices, target_size, target_size, vol.shape[3])
    vol = resize(
        vol,
        output_shape=vol_out_shape,
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )

    return vol, seg


def normalize_intensity(vol: npt.NDArray[np.float64],
                        low_percentile: float = 10.0,
                        high_percentile: float = 99.0) -> npt.NDArray[np.float64]:
    """
    Channel-wise intensity normalization: percentile clipping followed by z-score standardization.

    :param vol: Volume array with shape (Z, H, W, C)
    :param low_percentile: Lower percentile for clipping
    :param high_percentile: Upper percentile for clipping
    :return: Normalized volume with zero mean and unit variance per channel
    """
    p_lo = np.percentile(vol, low_percentile)
    p_hi = np.percentile(vol, high_percentile)
    vol = rescale_intensity(vol, in_range=(p_lo, p_hi))

    # Per-channel z-score normalization
    ch_mean = np.mean(vol, axis=(0, 1, 2))
    ch_std = np.std(vol, axis=(0, 1, 2))
    # Guard: constant channels (e.g. all-zero after cropping) produce std=0 → NaN
    ch_std[ch_std == 0] = 1.0
    vol = (vol - ch_mean) / ch_std

    return vol


def compose_visualization(mri_slice: npt.NDArray,
                          gt_mask: npt.NDArray,
                          pred_mask: npt.NDArray,
                          channel_idx: int = 1) -> List[npt.NDArray[np.uint8]]:
    """
    Generate overlay images with prediction and ground-truth contours for a batch.

    :param mri_slice: Input tensor batch, shape (B, C, H, W)
    :param gt_mask: Ground-truth masks, shape (B, 1, H, W)
    :param pred_mask: Predicted masks, shape (B, 1, H, W)
    :param channel_idx: Which MRI channel to display (default 1 = FLAIR)
    :return: List of RGB overlay images as uint8 arrays
    """
    overlays: List[npt.NDArray[np.uint8]] = []

    input_np = mri_slice[:, channel_idx].cpu().numpy()
    gt_np = gt_mask[:, 0].cpu().numpy()
    pred_np = pred_mask[:, 0].cpu().numpy()

    for i in range(input_np.shape[0]):
        rgb = grayscale_to_rgb(np.squeeze(input_np[i]))
        rgb = draw_contour(rgb, pred_np[i], color=[255, 0, 0])   # red = prediction
        rgb = draw_contour(rgb, gt_np[i], color=[0, 255, 0])     # green = ground truth
        overlays.append(rgb)

    return overlays


def grayscale_to_rgb(image: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """
    Convert a single-channel grayscale image to 3-channel RGB (uint8).

    :param image: 2D grayscale array
    :return: RGB array of shape (H, W, 3) with dtype uint8
    """
    h, w = image.shape

    # Shift to non-negative range
    image = image + np.abs(np.min(image))
    peak = np.abs(np.max(image))
    if peak > 0:
        image = image / peak

    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = rgb[:, :, 1] = rgb[:, :, 2] = (image * 255).astype(np.uint8)

    return rgb


def draw_contour(image: npt.NDArray[np.uint8],
                 mask: npt.NDArray[np.float64],
                 color: List[int]) -> npt.NDArray[np.uint8]:
    """
    Draw the boundary contour of a binary mask onto an RGB image.

    :param image: RGB image array of shape (H, W, 3)
    :param mask: 2D mask array (will be rounded to binary)
    :param color: RGB color list for the contour, e.g. [255, 0, 0]
    :return: Image with contour overlay
    """
    binary = np.round(mask)
    rows, cols = np.nonzero(binary)

    for r, c in zip(rows, cols):
        # Check if this pixel is on the boundary (has at least one non-mask neighbour)
        neighbourhood = binary[max(0, r - 1): r + 2, max(0, c - 1): c + 2]
        if 0.0 < np.mean(neighbourhood) < 1.0:
            image[max(0, r): r + 1, max(0, c): c + 1] = color

    return image
