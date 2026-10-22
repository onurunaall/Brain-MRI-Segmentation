"""Tests for losses.SoftDiceLoss and losses.DiceBCELoss."""

import pytest
import torch

from mriseg.losses import DiceBCELoss, SoftDiceLoss


def _mask(batch: int = 2, size: int = 8) -> torch.Tensor:
    target = torch.zeros(batch, 1, size, size)
    target[:, :, 2:6, 2:6] = 1.0
    return target


def test_perfect_prediction_has_zero_loss() -> None:
    target = _mask()
    assert SoftDiceLoss()(target.clone(), target).item() == pytest.approx(0.0, abs=1e-6)


def test_empty_prediction_on_tumour_is_near_one() -> None:
    target = _mask()
    # Dice per sample = (0 + 1) / (16 + 1) with smoothing 1
    expected = 1.0 - 1.0 / 17.0
    assert SoftDiceLoss()(torch.zeros_like(target), target).item() == pytest.approx(expected, abs=1e-6)


def test_empty_slice_predicted_empty_has_zero_loss() -> None:
    # Smoothing makes an empty/empty slice count as a perfect match instead of 0/0
    empty = torch.zeros(1, 1, 8, 8)
    assert SoftDiceLoss()(empty, empty.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_mean_of_per_sample_dice() -> None:
    target = _mask(batch=2)
    pred = target.clone()
    pred[1] = 0.0  # sample 0 perfect (Dice 1), sample 1 empty (Dice 1/17)
    expected = 1.0 - (1.0 + 1.0 / 17.0) / 2.0
    assert SoftDiceLoss()(pred, target).item() == pytest.approx(expected, abs=1e-6)


def test_loss_is_differentiable() -> None:
    target = _mask()
    pred = torch.full_like(target, 0.5, requires_grad=True)
    SoftDiceLoss()(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_shape_mismatch_raises() -> None:
    with pytest.raises(AssertionError):
        SoftDiceLoss()(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4))


# ------------------------------------------------------------- Dice + BCE

def test_dice_bce_of_confident_correct_logits_is_near_zero() -> None:
    target = _mask()
    logits = (target * 2 - 1) * 20.0  # +20 inside the tumour, -20 outside
    assert DiceBCELoss()(logits, target).item() == pytest.approx(0.0, abs=1e-6)


def test_dice_bce_is_dice_of_sigmoid_plus_bce() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    target = _mask()
    expected = (SoftDiceLoss()(torch.sigmoid(logits), target)
                + torch.nn.functional.binary_cross_entropy(torch.sigmoid(logits), target))
    assert DiceBCELoss()(logits, target).item() == pytest.approx(expected.item(), abs=1e-5)


def test_false_blob_on_empty_slice_gets_a_strong_gradient() -> None:
    """
    Regression for the false positives at the top and bottom of the head: on a tumour-free slice, soft Dice alone
    gives a confident false blob almost no gradient, while Dice + BCE pushes every blob pixel down.
    """
    target = torch.zeros(1, 1, 32, 32)
    blob = torch.full((1, 1, 32, 32), -10.0)
    blob[:, :, 4:28, 4:28] = 5.0  # 576 false-positive pixels at p ~ 0.993

    dice_logits = blob.clone().requires_grad_(True)
    SoftDiceLoss()(torch.sigmoid(dice_logits), target).backward()
    combined_logits = blob.clone().requires_grad_(True)
    DiceBCELoss()(combined_logits, target).backward()

    assert dice_logits.grad is not None and combined_logits.grad is not None
    dice_push = dice_logits.grad[:, :, 4:28, 4:28].mean().item()
    combined_push = combined_logits.grad[:, :, 4:28, 4:28].mean().item()
    assert 0.0 < dice_push < 1e-7
    assert combined_push > 1000 * dice_push
