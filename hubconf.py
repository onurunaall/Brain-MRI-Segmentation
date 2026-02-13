dependencies = ["torch"]

import torch
from network import UNetModel


def unet_segmentation(pretrained: bool = False,
                      weights_path: str = "",
                      **kwargs) -> UNetModel:
    """
    Load the U-Net segmentation model, optionally from a local checkpoint.

    :param pretrained: If True, load weights from weights_path
    :param weights_path: Path to a .pt checkpoint file
    :param kwargs: Forwarded to UNetModel (in_channels, out_channels, base_filters)
    :return: UNetModel instance
    """
    model = UNetModel(**kwargs)

    if pretrained:
        if not weights_path:
            raise ValueError("weights_path must be provided when pretrained=True")
        
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

    return model
