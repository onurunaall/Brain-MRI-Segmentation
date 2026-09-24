"""Tests for losses.SoftDiceLoss."""

import pytest
import torch

from losses import SoftDiceLoss


def _mask(batch: int = 2, size: int = 8) -> torch.Tensor:
    target = torch.zeros(batch, 1, size, size)
    target[:, :, 2:6, 2:6] = 1.0
    return target


def test_perfect_prediction_has_zero_loss():
    target = _mask()
    assert SoftDiceLoss()(target.clone(), target).item() == pytest.approx(0.0, abs=1e-6)


def test_empty_prediction_on_tumour_is_near_one():
    target = _mask()
    # Dice per sample = (0 + 1) / (16 + 1) with smoothing 1
    expected = 1.0 - 1.0 / 17.0
    assert SoftDiceLoss()(torch.zeros_like(target), target).item() == pytest.approx(expected, abs=1e-6)


def test_empty_slice_predicted_empty_has_zero_loss():
    # Smoothing makes an empty/empty slice count as a perfect match instead of 0/0
    empty = torch.zeros(1, 1, 8, 8)
    assert SoftDiceLoss()(empty, empty.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_loss_is_mean_of_per_sample_dice():
    target = _mask(batch=2)
    pred = target.clone()
    pred[1] = 0.0  # sample 0 perfect (Dice 1), sample 1 empty (Dice 1/17)
    expected = 1.0 - (1.0 + 1.0 / 17.0) / 2.0
    assert SoftDiceLoss()(pred, target).item() == pytest.approx(expected, abs=1e-6)


def test_loss_is_differentiable():
    target = _mask()
    pred = torch.full_like(target, 0.5, requires_grad=True)
    SoftDiceLoss()(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        SoftDiceLoss()(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 4, 4))
