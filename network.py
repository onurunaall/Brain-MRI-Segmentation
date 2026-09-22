""" U-Net architecture for biomedical image segmentation """
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class _ResidualBlock(nn.Module):
    """(Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN) + shortcut -> ReLU."""

    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.act = nn.ReLU(inplace=True)

        if ch_in == ch_out:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False),
                                          nn.BatchNorm2d(ch_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ResUNetModel(nn.Module):
    """U-Net with residual blocks. Pooling, upsampling and widths match UNetModel."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 32) -> None:
        super().__init__()
        f = base_filters

        self.enc_block1 = _ResidualBlock(in_channels, f)
        self.enc_block2 = _ResidualBlock(f, f * 2)
        self.enc_block3 = _ResidualBlock(f * 2, f * 4)
        self.enc_block4 = _ResidualBlock(f * 4, f * 8)
        self.bridge = _ResidualBlock(f * 8, f * 16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upsample4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec_block4 = _ResidualBlock(f * 8 * 2, f * 8)
        self.upsample3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec_block3 = _ResidualBlock(f * 4 * 2, f * 4)
        self.upsample2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec_block2 = _ResidualBlock(f * 2 * 2, f * 2)
        self.upsample1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec_block1 = _ResidualBlock(f * 2, f)

        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc_block1(x)
        e2 = self.enc_block2(self.pool(e1))
        e3 = self.enc_block3(self.pool(e2))
        e4 = self.enc_block4(self.pool(e3))
        latent = self.bridge(self.pool(e4))

        d4 = self.dec_block4(torch.cat((self.upsample4(latent), e4), dim=1))
        d3 = self.dec_block3(torch.cat((self.upsample3(d4), e3), dim=1))
        d2 = self.dec_block2(torch.cat((self.upsample2(d3), e2), dim=1))
        d1 = self.dec_block1(torch.cat((self.upsample1(d2), e1), dim=1))

        return torch.sigmoid(self.head(d1))


class _WindowAttention(nn.Module):
    """Multi-head self-attention inside a window, with learned relative position bias."""

    def __init__(self, dim: int, num_heads: int, ws: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.rel_bias_table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.rel_bias_table, std=0.02)

        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (ws - 1)
        self.register_buffer("rel_index", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        bw, n, c = x.shape
        qkv = self.qkv(x).reshape(bw, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.rel_bias_table[self.rel_index.reshape(-1)].reshape(n, n, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            n_win = mask.shape[0]
            attn = attn.view(bw // n_win, n_win, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(bw, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bw, n, c)
        return self.proj(out)


class _SwinBlock(nn.Module):
    """Pre-norm Swin block: (shifted) window attention + MLP, both residual."""

    def __init__(self, dim: int, num_heads: int, ws: int, shift: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.ws = ws
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, num_heads, ws)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    @staticmethod
    def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
        """(B, H, W, C) -> (B * nW, ws*ws, C)"""
        b, h, w, c = x.shape
        x = x.view(b, h // ws, ws, w // ws, ws, c)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)

    @staticmethod
    def window_reverse(windows: torch.Tensor, ws: int, b: int, h: int, w: int) -> torch.Tensor:
        """(B * nW, ws*ws, C) -> (B, H, W, C)"""
        c = windows.shape[-1]
        x = windows.view(b, h // ws, w // ws, ws, ws, c)
        return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)

    @staticmethod
    def shifted_window_mask(h: int, w: int, ws: int, shift: int, device: torch.device) -> torch.Tensor:
        """(nW, ws*ws, ws*ws) mask blocking attention across regions merged by the cyclic shift."""
        region = torch.zeros(1, h, w, 1, device=device)
        cnt = 0
        for hs in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
            for wsl in (slice(0, -ws), slice(-ws, -shift), slice(-shift, None)):
                region[:, hs, wsl, :] = cnt
                cnt += 1
        ids = _SwinBlock.window_partition(region, ws).squeeze(-1)
        mask = ids.unsqueeze(1) - ids.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        shortcut = x
        x = self.norm1(x)

        mask = None
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
            mask = self.shifted_window_mask(h, w, self.ws, self.shift, x.device).to(x.dtype)

        windows = self.attn(self.window_partition(x, self.ws), mask)
        x = self.window_reverse(windows, self.ws, b, h, w)

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        x = shortcut + x
        return x + self.mlp(self.norm2(x))
