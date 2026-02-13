import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """
    Differentiable Dice loss for binary or multi-class segmentation.

    Computes per-sample, per-channel Dice and returns the mean loss.
    This avoids large-foreground samples dominating the gradient signal
    (as happens with batch-level Dice) and generalises to multi-channel outputs.
    """

    def __init__(self,
                 smoothing: float = 1.0) -> None:
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
