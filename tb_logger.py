"""
TensorBoard logging wrapper for training metrics and image visualization.
"""

from typing import List, Optional

import numpy as np
import numpy.typing as npt
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Lightweight TensorBoard logger for scalars and images."""

    def __init__(self, log_dir: str) -> None:
        """
        :param log_dir: Directory where TensorBoard event files will be written
        """
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag: str, value: float, global_step: int) -> None:
        """
        Log a scalar metric.

        :param tag: Metric name (e.g. 'train/loss')
        :param value: Scalar value to record
        :param global_step: Training step counter
        """
        self.writer.add_scalar(tag, value, global_step)
        self.writer.flush()

    def log_image(self, tag: str, image: npt.NDArray[np.uint8], global_step: int) -> None:
        """
        Log a single image.

        :param tag: Image label
        :param image: HWC uint8 array
        :param global_step: Training step counter
        """
        # SummaryWriter expects (H, W, C) for add_image with dataformats='HWC'
        self.writer.add_image(tag, image, global_step, dataformats="HWC")
        self.writer.flush()

    def log_image_batch(self,
                        tag: str,
                        images: List[npt.NDArray[np.uint8]],
                        global_step: int) -> None:
        """
        Log a list of images under a common tag prefix.

        :param tag: Base tag; each image gets '{tag}/{index}'
        :param images: List of HWC uint8 arrays
        :param global_step: Training step counter
        """
        if not images:
            return

        for idx, img in enumerate(images):
            self.writer.add_image(f"{tag}/{idx}", img, global_step, dataformats="HWC")

        self.writer.flush()

    def close(self) -> None:
        """Flush and close the underlying SummaryWriter."""
        self.writer.close()
