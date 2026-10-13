import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Differentiable Dice loss for binary or multi-class segmentation.

    Computes per-sample, per-channel Dice and returns the mean loss.
    This avoids large-foreground samples dominating the gradient signal
    (as happens with batch-level Dice) and generalises to multi-channel outputs.
    """

    def __init__(self, smoothing: float = 1.0) -> None:
        """
        :param smoothing: Laplace smoothing constant to avoid division by zero
        """
        super().__init__()
        self.smoothing = smoothing

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Compute 1 − mean(Dice) as the loss value.

        :param pred: Predicted probabilities, shape (B, C, H, W)
        :param target: Ground-truth binary masks, shape (B, C, H, W)
        :return: Scalar loss tensor
        """
        assert pred.size() == target.size(), (f"Shape mismatch: pred {pred.size()} vs target {target.size()}")

        pred = pred.float()
        target = target.float()
        
        batch_size, n_channels = pred.shape[0], pred.shape[1]

        # Flatten spatial dims per sample per channel → (B, C, H*W)
        pred_flat = pred.view(batch_size, n_channels, -1)
        target_flat = target.view(batch_size, n_channels, -1)

        # Per-sample, per-channel Dice
        overlap = (pred_flat * target_flat).sum(dim=2)  # (B, C)
        cardinality = pred_flat.sum(dim=2) + target_flat.sum(dim=2)  # (B, C)

        dice_per_sample = (2.0 * overlap + self.smoothing) / (cardinality + self.smoothing)  # (B, C)

        # Mean over samples and channels
        return 1.0 - dice_per_sample.mean()


class DiceBCELoss(nn.Module):
    """
    Soft Dice loss plus binary cross-entropy, both on the same logits (as in nnU-Net).

    Soft Dice alone gives almost no gradient to false positives on a tumour-free slice: there its loss is
    1 - 1 / (sum(p) + 1), whose gradient 1 / (sum(p) + 1)^2 vanishes once a false blob is predicted. BCE gives
    every wrong pixel a gradient of size |p - target|, so false blobs on empty slices are pushed down.
    """

    def __init__(self, smoothing: float = 1.0) -> None:
        """
        :param smoothing: Laplace smoothing of the Dice term
        """
        super().__init__()
        self.dice = SoftDiceLoss(smoothing)

    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        :param logits: Raw model outputs (before sigmoid), shape (B, C, H, W)
        :param target: Ground-truth binary masks, shape (B, C, H, W)
        :return: Scalar loss tensor (Dice loss + mean BCE)
        """
        logits = logits.float()
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return self.dice(torch.sigmoid(logits), target) + bce
