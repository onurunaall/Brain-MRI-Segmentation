from typing import Tuple, List, Optional
import numpy as np
import numpy.typing as npt
from medpy.filter.binary import largest_connected_component
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from PIL import Image


def dsc(y_pred: npt.NDArray[np.float32],
        y_true: npt.NDArray[np.float32],
        lcc: bool = True,
        epsilon: float = 1e-1) -> float:
    """
    Calculate Dice Similarity Coefficient (DSC).
    
    :param y_pred: Predicted segmentation mask
    :param y_true: Ground truth segmentation mask
    :param lcc: If True, apply largest connected component filter to prediction
    :param epsilon: Small constant for numerical stability
    :return: Dice coefficient (0.0 to 1.0)
    """
    # Optionally binarize and keep only the largest connected component in the prediction
    if lcc and np.any(y_pred):
        y_pred = np.round(y_pred).astype(np.int32)
        y_true = np.round(y_true).astype(np.int32)
        y_pred = largest_connected_component(y_pred)
    
    intersection = np.sum(y_pred[y_true == 1])
    denominator = np.sum(y_pred) + np.sum(y_true)

    # If both masks are empty, define DSC=1; if only GT has positives, define DSC=0.
    if denominator == 0:
        return 1.0 if np.sum(y_true) == 0 else 0.0
    
    return (2.0 * intersection + epsilon) / (denominator + epsilon)


def crop_sample(x: Tuple[npt.NDArray, npt.NDArray],
                intensity_threshold: float = 0.1) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Crop volume to the tight bounding box that contains non-background voxels.

    :param x: Tuple of (volume, mask) arrays
    :param intensity_threshold: Threshold relative to max intensity for background detection
    :return: Cropped (volume, mask) tuple
    """
    volume, mask = x
    
    # Threshold to remove background
    volume_thresholded = volume.copy()
    volume_thresholded[volume_thresholded < np.max(volume) * intensity_threshold] = 0  # suppress background voxels

    # Find bounding box in each dimension
    # Z-axis (slices)
    z_projection = np.max(volume_thresholded, axis=(1, 2, 3))
    z_nonzero = np.nonzero(z_projection)[0]
    z_min = np.min(z_nonzero)
    z_max = np.max(z_nonzero) + 1
    
    # Y-axis (height)
    y_projection = np.max(volume_thresholded, axis=(0, 2, 3))
    y_nonzero = np.nonzero(y_projection)[0]
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero) + 1
    
    # X-axis (width)
    x_projection = np.max(volume_thresholded, axis=(0, 1, 3))
    x_nonzero = np.nonzero(x_projection)[0]
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero) + 1
    
    return (volume[z_min:z_max, y_min:y_max, x_min:x_max, mask[z_min:z_max, y_min:y_max, x_min:x_max])


def pad_sample(x: Tuple[npt.NDArray, npt.NDArray]) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Pad volume and mask to make spatial dimensions square (H == W).

    :param x: Tuple of (volume, mask) arrays
    :return: Padded (volume, mask) tuple
    """
    volume, mask = x
    height, width = volume.shape[1], volume.shape[2]
    
    if height == width:
        return volume, mask
    
    # Calculate padding needed
    max_dim = max(height, width)
    diff = (max_dim - min(height, width)) / 2.0
    
    if height > width:
        # Pad width
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        # Pad height
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))
    
    mask = np.pad(mask, padding, mode="constant", constant_values=0)
    
    # Add channel dimension for volume padding
    padding_with_channels = padding + ((0, 0))
    volume = np.pad(volume, padding_with_channels, mode="constant", constant_values=0)
    
    return volume, mask


def resize_sample(x: Tuple[npt.NDArray, npt.NDArray],
                  size: int = 256) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Resize volume and mask to (D, size, size).

    :param x: Tuple of (volume, mask) arrays
    :param size: Target spatial size for height and width
    :return: Resized (volume, mask) tuple
    """
    volume, mask = x
    num_slices, _, _, num_channels = volume.shape
    
    out_shape = (num_slices, size, size)
    
    # Resize mask with nearest neighbor interpolation
    mask = resize(mask,
                  output_shape=out_shape,
                  order=0,  # nearest-neighbor (keeps binary/labels intact)
                  mode="constant",
                  cval=0,
                  anti_aliasing=False,
                  preserve_range=True)
    
    # Resize volume with bilinear interpolation
    volume = resize(volume,
                    output_shape=out_shape + (num_channels),
                    order=1,  # bilinear interpolation for intensities
                    mode="constant",
                    cval=0,
                    anti_aliasing=True,  # reduce aliasing artifacts when downsampling
                    preserve_range=True)

    return volume, mask


def normalize_volume(volume: npt.NDArray,
                     percentile_lower: float = 10.0,
                     percentile_upper: float = 99.0) -> npt.NDArray:
    """
    Normalize volume intensity using percentile scaling + per-channel z-score.

    :param volume: Input volume array
    :param percentile_lower: Lower percentile for intensity rescaling
    :param percentile_upper: Upper percentile for intensity rescaling
    :return: Normalized volume
    """
    # Clip intensities to percentile range
    p_low = np.percentile(volume, percentile_lower)
    p_high = np.percentile(volume, percentile_upper)
    volume = rescale_intensity(volume,
                               in_range=(p_low, p_high),
                               out_range=(0, 1))
    
    # Z-score normalization per channel
    mean = np.mean(volume, axis=(0, 1, 2), keepdims=True)
    std = np.std(volume, axis=(0, 1, 2), keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    volume = (volume - mean) / std
    return volume


def gray2rgb(image: npt.NDArray) -> npt.NDArray:
    """
    Convert a grayscale (H, W) image into an RGB uint8 (H, W, 3) image.

    :param image: Grayscale image array (H, W)
    :return: RGB image array (H, W, 3) with values in [0, 255]
    """
    height, width = image.shape
    
    # Normalize to [0, 1]
    image = image - np.min(image)
    image_max = np.max(image)
    if image_max > 0:
        image = image / image_max
    
    # Convert to uint8 RGB
    channel_u8 = (image * 255).astype(np.uint8)
    rgb_image = np.empty((height, width, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = channel_u8
    rgb_image[:, :, 1] = channel_u8
    rgb_image[:, :, 2] = channel_u8
    
    return rgb_image


def outline(image: npt.NDArray,
            mask: npt.NDArray,
            color: List[int]) -> npt.NDArray:
    """
    Draw a 1-pixel outline of a binary mask onto an RGB image.

    :param image: RGB image array (H, W, 3)
    :param mask: Binary mask array (H, W)
    :param color: RGB color as [R, G, B]
    :return: Image with outlined mask
    """
    mask = np.round(mask).astype(np.uint8)
    yy, xx = np.nonzero(mask)
    
    for y, x in zip(yy, xx):
        # Check if pixel is on the boundary
        y_min, y_max = max(0, y - 1), min(mask.shape[0], y + 2)
        x_min, x_max = max(0, x - 1), min(mask.shape[1], x + 2)
        
        local_region = mask[y_min:y_max, x_min:x_max]
        if 0.0 < np.mean(local_region) < 1.0:
            image[y:y + 1, x:x + 1] = color    # paint boundary pixel
    
    return image


def log_images(x: "torch.Tensor",
               y_true: "torch.Tensor",
               y_pred: "torch.Tensor",
               channel: int = 1) -> List[npt.NDArray]:
    """
    Create visualization images with prediction and ground-truth outlines.

    :param x: Input images tensor (B, C, H, W)
    :param y_true: Ground truth masks tensor (B, 1, H, W)
    :param y_pred: Predicted masks tensor (B, 1, H, W)
    :param channel: Which input channel to visualize
    :return: List of RGB images with overlays
    """
    images = []
    
    x_np = x[:, channel].cpu().numpy()
    y_true_np = y_true[:, 0].cpu().numpy()
    y_pred_np = y_pred[:, 0].cpu().numpy()
    
    for i in range(x_np.shape[0]):
        image = gray2rgb(x_np[i])
        
        # Add prediction outline in red
        image = outline(image, y_pred_np[i], color=[255, 0, 0])
        
        # Add ground truth outline in green
        image = outline(image, y_true_np[i], color=[0, 255, 0])
        
        images.append(image)
    
    return images


def numpy_to_pil(image: npt.NDArray) -> Image.Image:
    """
    Convert a numpy uint8 RGB image to a PIL Image.

    :param image: Numpy array (H, W, C) with values in [0, 255]
    :return: PIL Image
    """
    return Image.fromarray(image.astype(np.uint8))uint8))
