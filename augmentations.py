"""
Data augmentation transforms for MRI segmentation training.
"""

from typing import Tuple, Optional, List

import torch
import torchvision.transforms.v2.functional as TF


def build_augmentation_pipeline(scale_range: Optional[float] = None,
                                rotation_deg: Optional[float] = None,
                                flip_probability: Optional[float] = None) -> "AugmentationPipeline":
    """
    Construct a composed augmentation pipeline.

    :param scale_range: Maximum scale deviation from 1.0 (e.g. 0.05 → [0.95, 1.05])
    :param rotation_deg: Maximum rotation in degrees (symmetric around 0)
    :param flip_probability: Probability of horizontal flip per sample
    :return: Callable pipeline operating on (image, mask) numpy tuples
    """
    stages: List = []

    if scale_range is not None:
        stages.append(RandomScale(scale_range))
    if rotation_deg is not None:
        stages.append(RandomRotation(rotation_deg))
    if flip_probability is not None:
        stages.append(RandomHorizontalFlip(flip_probability))

    return AugmentationPipeline(stages)


class AugmentationPipeline:
    """
    Applies a sequence of augmentations to (image, mask) numpy array pairs.

    Internally converts to torch tensors for the augmentation operations,
    then converts back to numpy to remain compatible with the dataset's
    existing transpose and tensor conversion logic.
    """

    def __init__(self, stages: List) -> None:
        """
        :param stages: List of augmentation callables
        """
        self.stages = stages

    def __call__(self, sample: Tuple) -> Tuple:
        """
        :param sample: Tuple of (image, mask) numpy arrays, shape (H, W, C)
        :return: Augmented (image, mask) numpy arrays, shape (H, W, C)
        """
        import numpy as np

        image_np, mask_np = sample

        # (H, W, C) → (C, H, W) tensor
        image_t = torch.from_numpy(image_np.transpose(2, 0, 1).copy()).float()
        mask_t = torch.from_numpy(mask_np.transpose(2, 0, 1).copy()).float()

        for stage in self.stages:
            image_t, mask_t = stage(image_t, mask_t)

        image_out = image_t.numpy().transpose(1, 2, 0)
        mask_out = mask_t.numpy().transpose(1, 2, 0)

        return image_out, mask_out


class RandomScale:
    """Randomly rescale the image and mask, then crop or pad back to original size."""

    def __init__(self, scale_range: float) -> None:
        """
        :param scale_range: Maximum deviation from 1.0 (e.g. 0.05 → uniform in [0.95, 1.05])
        """
        self.scale_range = scale_range

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param image: (C, H, W) tensor
        :param mask: (C, H, W) tensor
        :return: Scaled (image, mask) tensors at original spatial size
        """
        _, orig_h, orig_w = image.shape

        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale_range
        new_h = int(round(orig_h * factor))
        new_w = int(round(orig_w * factor))

        # Bilinear for image, nearest for mask
        image = TF.resize(image,
                          [new_h, new_w],
                          interpolation=TF.InterpolationMode.BILINEAR,
                          antialias=False)
        
        mask = TF.resize(mask,
                         [new_h, new_w],
                         interpolation=TF.InterpolationMode.NEAREST,
                         antialias=False)

        if factor < 1.0:
            pad_h = orig_h - new_h
            pad_w = orig_w - new_w
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            padding = [pad_left, pad_top, pad_w - pad_left, pad_h - pad_top]
            image = TF.pad(image, padding, fill=0)
            mask = TF.pad(mask, padding, fill=0)
        else:
            image = TF.center_crop(image, [orig_h, orig_w])
            mask = TF.center_crop(mask, [orig_h, orig_w])

        return image, mask


class RandomRotation:
    """Apply a random in-plane rotation to image and mask."""

    def __init__(self, max_angle: float) -> None:
        """
        :param max_angle: Maximum rotation angle in degrees (symmetric range)
        """
        self.max_angle = max_angle

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param image: (C, H, W) tensor
        :param mask: (C, H, W) tensor
        :return: Rotated (image, mask) tensors
        """
        theta = (torch.rand(1).item() * 2 - 1) * self.max_angle

        image = TF.rotate(image, theta, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, theta, interpolation=TF.InterpolationMode.NEAREST, fill=0)

        return image, mask


class RandomHorizontalFlip:
    """Randomly flip the image and mask along the horizontal axis."""

    def __init__(self, probability: float) -> None:
        """
        :param probability: Probability of applying the flip
        """
        self.probability = probability

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param image: (C, H, W) tensor
        :param mask: (C, H, W) tensor
        :return: Possibly flipped (image, mask) tensors
        """
        if torch.rand(1).item() < self.probability:
            image = TF.horizontal_flip(image)
            mask = TF.horizontal_flip(mask)

        return image, mask
