"""
Data augmentation transforms for MRI segmentation training.

Each transform operates on (image, mask) tuples and returns the same.
Masks always use nearest-neighbour interpolation to preserve binary labels.
"""

from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose


def build_augmentation_pipeline(scale_range: Optional[float] = None,
                                rotation_deg: Optional[float] = None,
                                flip_probability: Optional[float] = None) -> Compose:
    """
    Construct a composed augmentation pipeline.

    :param scale_range: Maximum scale deviation from 1.0 (e.g. 0.05 → [0.95, 1.05])
    :param rotation_deg: Maximum rotation in degrees (symmetric around 0)
    :param flip_probability: Probability of horizontal flip per sample
    :return: Composed transform callable
    """
    stages = []

    if scale_range is not None:
        stages.append(RandomScale(scale_range))
    
    if rotation_deg is not None:
        stages.append(RandomRotation(rotation_deg))
    
    if flip_probability is not None:
        stages.append(RandomHorizontalFlip(flip_probability))

    return Compose(stages)


class RandomScale:
    """Randomly rescale the image and mask, then crop or pad back to original size."""

    def __init__(self,
                 scale_range: float) -> None:
        """
        :param scale_range: Maximum deviation from 1.0 (e.g. 0.05 → uniform in [0.95, 1.05])
        """
        self.scale_range = scale_range

    def __call__(self,
                 sample: Tuple[npt.NDArray, npt.NDArray]) -> Tuple[npt.NDArray, npt.NDArray]:
        image, mask = sample
        orig_h, orig_w = image.shape[0], image.shape[1]

        factor = np.random.uniform(low=1.0 - self.scale_range, high=1.0 + self.scale_range)

        image = rescale(image,
                        (factor, factor),
                        channel_axis=2,
                        preserve_range=True,
                        mode="constant",
                        anti_aliasing=False)
        
        mask = rescale(mask,
                       (factor, factor),
                       order=0,
                       channel_axis=2,
                       preserve_range=True,
                       mode="constant",
                       anti_aliasing=False)

        if factor < 1.0:
            delta_h = (orig_h - image.shape[0]) / 2.0
            delta_w = (orig_w - image.shape[1]) / 2.0
            
            pad_spec = ((int(np.floor(delta_h)), int(np.ceil(delta_h))),
                        (int(np.floor(delta_w)), int(np.ceil(delta_w))),
                        (0, 0))
            
            image = np.pad(mask, pad_spec, mode="constant", constant_values=0)
            mask = np.pad(mask, pad_spec, mode="constant", constant_values=0)
        else:
            off_h = (image.shape[0] - orig_h) // 2
            off_w = (image.shape[1] - orig_w) // 2
            
            image = image[off_h: off_h + orig_h, off_w: off_w + orig_w, ...]
            mask = mask[off_h: off_h + orig_h, off_w: off_w + orig_w, ...]

        return image, mask


class RandomRotation:
    """Apply a random in-plane rotation to image and mask."""

    def __init__(self,
                 max_angle: float) -> None:
        """
        :param max_angle: Maximum rotation angle in degrees (symmetric range)
        """
        self.max_angle = max_angle

    def __call__(self,
                 sample: Tuple[npt.NDArray, npt.NDArray]) -> Tuple[npt.NDArray, npt.NDArray]:
        image, mask = sample

        theta = np.random.uniform(low=-self.max_angle,
                                  high=self.max_angle)

        image = rotate(image, theta, resize=False, preserve_range=True, mode="constant")
        mask = rotate(mask, theta, resize=False, order=0, preserve_range=True, mode="constant")

        return image, mask


class RandomHorizontalFlip:
    """Randomly flip the image and mask along the horizontal axis."""

    def __init__(self,
                 probability: float) -> None:
        """
        :param probability: Probability of applying the flip
        """
        self.probability = probability

    def __call__(self,
                 sample: Tuple[npt.NDArray, npt.NDArray]) -> Tuple[npt.NDArray, npt.NDArray]:
        image, mask = sample

        if np.random.rand() > self.probability:
            return image, mask

        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask
