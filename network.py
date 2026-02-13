"""
U-Net architecture for biomedical image segmentation.

Encoder–decoder network with skip connections and batch normalization,
designed for pixel-wise binary segmentation of MRI volumes.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class UNetModel(nn.Module):
    """
    U-Net with four encoder stages, a bottleneck, and four decoder stages.

    Each stage is a double-convolution block (Conv → BN → ReLU) × 2.
    Skip connections concatenate encoder features with decoder features.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 32) -> None:
        """
        :param in_channels: Number of input channels (e.g. 3 for RGB-like MRI)
        :param out_channels: Number of output segmentation classes
        :param base_filters: Feature map count in the first encoder stage (doubled each level)
        """
        super().__init__()

        f = base_filters

        # Encoder path
        self.enc_block1 = self._conv_block(in_channels, f, tag="e1")
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_block2 = self._conv_block(f, f * 2, tag="e2")
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_block3 = self._conv_block(f * 2, f * 4, tag="e3")
        self.downsample3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_block4 = self._conv_block(f * 4, f * 8, tag="e4")
        self.downsample4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge
        self.bridge = self._conv_block(f * 8, f * 16, tag="bridge")

        # Decoder path
        self.upsample4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec_block4 = self._conv_block(f * 8 * 2, f * 8, tag="d4")

        self.upsample3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec_block3 = self._conv_block(f * 4 * 2, f * 4, tag="d3")

        self.upsample2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec_block2 = self._conv_block(f * 2 * 2, f * 2, tag="d2")

        self.upsample1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec_block1 = self._conv_block(f * 2, f, tag="d1")

        # Output head
        self.head = nn.Conv2d(in_channels=f, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        :param x: Input tensor of shape (B, C, H, W)
        :return: Sigmoid-activated segmentation map of shape (B, out_channels, H, W)
        """
        # Encoder
        e1 = self.enc_block1(x)
        e2 = self.enc_block2(self.downsample1(e1))
        e3 = self.enc_block3(self.downsample2(e2))
        e4 = self.enc_block4(self.downsample3(e3))

        # Bridge
        latent = self.bridge(self.downsample4(e4))

        # Decoder with skip connections
        d4 = self.upsample4(latent)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec_block4(d4)

        d3 = self.upsample3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec_block3(d3)

        d2 = self.upsample2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec_block2(d2)

        d1 = self.upsample1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec_block1(d1)

        return torch.sigmoid(self.head(d1))

    @staticmethod
    def _conv_block(ch_in: int, ch_out: int, tag: str) -> nn.Sequential:
        """
        Double convolution block: (Conv3×3 → BN → ReLU) × 2.

        :param ch_in: Number of input channels
        :param ch_out: Number of output channels
        :param tag: Prefix for named layers (aids debugging / state-dict readability)
        :return: Sequential module implementing the block
        """
        return nn.Sequential(
            OrderedDict(
                [(f"{tag}_conv1", nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False)),
                 (f"{tag}_bn1", nn.BatchNorm2d(ch_out)),
                 (f"{tag}_act1", nn.ReLU(inplace=True)),
                 (f"{tag}_conv2", nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False)),
                 (f"{tag}_bn2", nn.BatchNorm2d(ch_out)),
                 (f"{tag}_act2", nn.ReLU(inplace=True))]
            )
        )
