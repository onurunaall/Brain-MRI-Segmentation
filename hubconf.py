from typing import Any

import torch
import torch.nn as nn
from mriseg.inference import load_trained_model
from mriseg.network import UNetModel


def unet_segmentation(pretrained: bool = False,
                      weights_path: str = "",
                      **kwargs: Any) -> nn.Module:
    """
    Load the U-Net segmentation model, optionally from a local checkpoint.

    :param pretrained: If True, load weights from weights_path
    :param weights_path: Path to a .pt checkpoint file saved by train.py
    :param kwargs: Forwarded to UNetModel (in_channels, out_channels, base_filters) when pretrained=False; a
                   pretrained model is built exactly as its checkpoint describes
    :return: UNetModel instance (in eval mode when pretrained); it returns logits, apply torch.sigmoid for
             probabilities
    """
    if not pretrained:
        return UNetModel(**kwargs)
    if not weights_path:
        raise ValueError("weights_path must be provided when pretrained=True")
    if kwargs:
        raise ValueError(f"{sorted(kwargs)} cannot be changed for a pretrained model; its checkpoint sets them")

    model, record = load_trained_model(weights_path, torch.device("cpu"))
    if record["arch"] != "unet":
        raise ValueError(f"{weights_path} is a '{record['arch']}' model, not a U-Net")
    return model
