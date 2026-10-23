"""
Utility functions for MRI segmentation preprocessing and evaluation.
Provides helper routines for volume cropping, padding, resizing, normalization, Dice score computation, and visualization overlays.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from scipy import ndimage
import torch
import torch.nn.functional as F


def dice_similarity_coefficient(prediction: npt.NDArray,
                                ground_truth: npt.NDArray,
                                epsilon: float = 1e-6) -> float:
    """
    Dice Similarity Coefficient (DSC) between two masks, each binarized at 0.5 (values of exactly 0.5 count as
    background). It applies no post-processing; filter the prediction with postprocess_mask first if needed.

    :param prediction: Predicted mask or probabilities
    :param ground_truth: Ground truth mask
    :param epsilon: Small constant for numerical stability
    :return: Dice coefficient in range [0.0, 1.0]
    """
    prediction = np.asarray(prediction) > 0.5
    ground_truth = np.asarray(ground_truth) > 0.5

    overlap = np.count_nonzero(prediction & ground_truth)
    total = np.count_nonzero(prediction) + np.count_nonzero(ground_truth)

    # Both masks empty → perfect agreement by convention
    if total == 0:
        return 1.0

    return float((2.0 * overlap + epsilon) / (total + epsilon))


def group_slices_by_patient(slices: Sequence[npt.NDArray],
                            flat_index: Sequence[Tuple[int, int]],
                            patient_ids: Sequence[str]) -> Dict[str, npt.NDArray]:
    """
    Stack per-slice arrays back into one volume per patient.

    slices[i] must belong to flat_index[i], i.e. come from a DataLoader over the dataset without shuffling.

    :param slices: Per-slice arrays in dataset order
    :param flat_index: (patient_idx, slice_idx) of each dataset item
    :param patient_ids: Patient identifiers in dataset order
    :return: Dict patient_id -> array of shape (Z, ...), patients in dataset order
    :raises ValueError: if the counts differ or a patient's slices are not in order 0, 1, 2, ...
    """
    if len(slices) != len(flat_index):
        raise ValueError(f"Got {len(slices)} slices for a dataset of {len(flat_index)} (drop_last or a subset?)")

    grouped: Dict[str, List[npt.NDArray]] = {}
    for (p_idx, s_idx), item in zip(flat_index, slices):
        stack = grouped.setdefault(patient_ids[p_idx], [])
        if s_idx != len(stack):
            raise ValueError(f"Slice {s_idx} of {patient_ids[p_idx]} is out of order (expected {len(stack)})")
        stack.append(item)

    return {pid: np.array(stack) for pid, stack in grouped.items()}


# Post-processing of a thresholded prediction volume, from removing nothing to removing the most; predict.py picks
# one per model on the validation patients
POSTPROCESS_METHODS: List[str] = ["none", "drop-small", "lcc"]
SMALL_COMPONENT_FRACTION = 0.1  # 'drop-small' removes components below this fraction of the largest one


def postprocess_mask(mask: npt.NDArray, method: str) -> npt.NDArray[np.bool_]:
    """
    Filter the 3D connected components of a binary prediction volume.

    'lcc' can delete the whole tumour when a false-positive blob elsewhere is larger, and it always deletes
    separate lesions, so it is one option among several rather than a fixed step.

    :param mask: Binary prediction volume (components are connected across all axes, i.e. across slices too)
    :param method: 'none' keeps every voxel, 'lcc' only the largest connected component,
                   'drop-small' every component with at least SMALL_COMPONENT_FRACTION of the largest one's voxels
    :return: Filtered boolean mask of the same shape
    """
    if method not in POSTPROCESS_METHODS:
        raise ValueError(f"Unknown post-processing '{method}'. Choose from {POSTPROCESS_METHODS}")

    mask = np.asarray(mask).astype(bool)
    if method == "none" or not mask.any():
        return mask

    labels, _ = ndimage.label(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # label 0 is the background

    if method == "lcc":
        return labels == np.argmax(sizes)  # first largest on ties, as medpy's largest_connected_component

    keep = sizes >= SMALL_COMPONENT_FRACTION * sizes.max()
    keep[0] = False
    return keep[labels]


def head_mask(vol: npt.NDArray, bg_threshold_ratio: float = 0.1) -> npt.NDArray[np.bool_]:
    """
    Voxels that belong to the head: at least one channel reaches `bg_threshold_ratio` of the volume's maximum.

    :param vol: Volume of shape (Z, H, W, C)
    :param bg_threshold_ratio: Fraction of max intensity below which voxels are treated as background
    :return: Boolean mask of shape (Z, H, W)
    """
    return np.asarray(vol.max(axis=-1) >= vol.max() * bg_threshold_ratio)


def content_box(vol: npt.NDArray, bg_threshold_ratio: float = 0.1) -> Tuple[slice, slice, slice]:
    """
    Tight bounding box (Z, Y, X slices) enclosing the non-background voxels of a volume (see head_mask).

    :param vol: Volume of shape (Z, H, W, C)
    :param bg_threshold_ratio: Fraction of max intensity below which voxels are treated as background
    :return: (z, y, x) slices; vol[z, y, x] is the cropped volume
    """
    head = head_mask(vol, bg_threshold_ratio)
    box = []
    for axis in range(3):
        other_axes = tuple(a for a in range(3) if a != axis)
        present = np.nonzero(head.any(axis=other_axes))[0]
        box.append(slice(int(present[0]), int(present[-1]) + 1))
    return box[0], box[1], box[2]


def pad_to_square(sample: Tuple[npt.NDArray, npt.NDArray]) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Pad volume and mask with zeros so that the spatial dimensions (H, W) become square. The shorter axis is padded on
    both sides; the smaller half of an odd padding goes before the content (restore_mask_geometry relies on this).

    :param sample: Tuple of (volume (Z, H, W, C), mask (Z, H, W)) arrays
    :return: Padded (volume, mask) tuple with equal H and W
    """
    vol, seg = sample
    height, width = vol.shape[1], vol.shape[2]
    if height == width:
        return vol, seg

    side = max(height, width)
    rows = ((side - height) // 2, side - height - (side - height) // 2)
    cols = ((side - width) // 2, side - width - (side - width) // 2)
    return np.pad(vol, ((0, 0), rows, cols, (0, 0))), np.pad(seg, ((0, 0), rows, cols))


def resize_volume(
    sample: Tuple[npt.NDArray, npt.NDArray],
    target_size: int = 256,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Resize spatial dimensions of volume and mask to (target_size × target_size).
    Uses PyTorch interpolation for speed (10-50× faster than skimage.resize).

    :param sample: Tuple of (volume, mask) arrays
    :param target_size: Desired spatial resolution (pixels)
    :return: Resized (volume, mask) tuple
    """
    vol, seg = sample

    # Volume: (Z, H, W, C) → (Z, C, H, W) for torch
    vol_t = torch.from_numpy(vol.transpose(0, 3, 1, 2)).float()
    orig_min, orig_max = vol_t.min(), vol_t.max()
    vol_t = F.interpolate(vol_t, size=(target_size, target_size),
                          mode="bicubic", align_corners=False)
    vol_t = vol_t.clamp(min=orig_min, max=orig_max)
    vol = vol_t.numpy().transpose(0, 2, 3, 1)  # back to (Z, H, W, C)

    # Mask: (Z, H, W) → (Z, 1, H, W) for torch. "nearest-exact" samples pixel centres like the bicubic resize above;
    # plain "nearest" would shift the mask half a pixel down-right of the image
    seg_t = torch.from_numpy(seg[:, np.newaxis, :, :]).float()
    seg_t = F.interpolate(seg_t, size=(target_size, target_size),
                          mode="nearest-exact")
    seg = seg_t[:, 0].numpy()  # back to (Z, H, W)

    return vol, seg


@dataclass(frozen=True)
class VolumeGeometry:
    """
    Where a preprocessed volume came from in the original scan, so predictions can be mapped back.

    Preprocessing drops the first and last slice, crops to content_box, pads to a square (pad_to_square) and resizes
    to grid_size × grid_size (resize_volume).
    """
    n_slices: int     # slices (image files) of the original scan
    height: int       # original slice size
    width: int
    z_start: int      # first original slice inside the crop
    y_start: int      # top-left corner of the crop in the original slices
    x_start: int
    crop_slices: int  # size of the crop
    crop_height: int
    crop_width: int
    grid_size: int    # preprocessed in-plane size (--image-size)

    @property
    def padded_size(self) -> int:
        """Side of the square the crop is padded to before resizing."""
        return max(self.crop_height, self.crop_width)


def restore_mask_geometry(mask: npt.NDArray, geometry: VolumeGeometry) -> npt.NDArray[np.bool_]:
    """
    Map a mask on the preprocessed grid back onto the original scan: undo the resize (nearest neighbour), the padding,
    the crop and the dropped first / last slices. Everything outside the crop is background.

    When the grid is at least as large as the padded crop (the usual case: crops are at most 256 pixels), this exactly
    undoes resize_volume for masks: both use "nearest-exact".

    :param mask: (crop_slices, grid_size, grid_size) mask, e.g. a thresholded and post-processed prediction
    :param geometry: The patient's VolumeGeometry from the dataset
    :return: Boolean mask of shape (n_slices, height, width)
    """
    padded = geometry.padded_size
    resized = F.interpolate(torch.from_numpy(np.asarray(mask, dtype=np.float32))[:, None],
                            size=(padded, padded), mode="nearest-exact")[:, 0].numpy() > 0.5

    # pad_to_square puts the smaller half of the padding before the content
    top = (padded - geometry.crop_height) // 2
    left = (padded - geometry.crop_width) // 2
    crop = resized[:, top: top + geometry.crop_height, left: left + geometry.crop_width]

    restored = np.zeros((geometry.n_slices, geometry.height, geometry.width), dtype=bool)
    restored[geometry.z_start: geometry.z_start + geometry.crop_slices,
             geometry.y_start: geometry.y_start + geometry.crop_height,
             geometry.x_start: geometry.x_start + geometry.crop_width] = crop
    return restored


def normalize_intensity(vol: npt.NDArray,
                        low_percentile: float = 1.0,
                        high_percentile: float = 99.0) -> npt.NDArray[np.float32]:
    """
    Channel-wise intensity normalization: percentile clipping followed by z-score standardization, with all
    statistics taken from head voxels only (head_mask). Including the black background would make the result depend
    on how much background surrounds the head, so the same tissue would get different values in different patients.
    The background is clipped up to the low percentile, so it becomes each channel's minimum.

    :param vol: Volume array with shape (Z, H, W, C), background 0
    :param low_percentile: Lower percentile of the head voxels for clipping
    :param high_percentile: Upper percentile of the head voxels for clipping
    :return: Normalized float32 volume with zero mean and unit variance per channel over the head voxels
    """
    vol = vol.astype(np.float32)
    head = head_mask(vol)
    for c in range(vol.shape[-1]):
        lo, hi = np.percentile(vol[..., c][head], [low_percentile, high_percentile])
        clipped = np.clip(vol[..., c], lo, hi)
        # A constant channel would give std 0 -> NaN
        mean, std = clipped[head].mean(), clipped[head].std() or 1.0
        vol[..., c] = (clipped - mean) / std
    return vol


def compose_visualization(mri_slice: torch.Tensor,
                          gt_mask: torch.Tensor,
                          pred_mask: torch.Tensor,
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
    binary = np.round(mask).astype(bool)
    # Boundary = mask pixels with at least one background pixel among their 8 neighbours (outside the image counts
    # as mask, so the image border itself is not a boundary)
    interior = ndimage.binary_erosion(binary, structure=np.ones((3, 3), dtype=bool), border_value=1)
    image[binary & ~interior] = color

    return image


def seed_everything(seed: int) -> None:
    """
    Seed every random number generator used during training (Python, NumPy, PyTorch CPU and all CUDA devices).

    This makes weight init, slice sampling and augmentation repeatable. It does NOT make two GPU runs bit-identical:
    cuDNN and torch.compile still pick non-deterministic kernels, so runs with the same seed end slightly apart.

    :param seed: Seed for all generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
