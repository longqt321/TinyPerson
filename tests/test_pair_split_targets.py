"""Self-tests for the PairSplit target builder and loss.

Run with: uv run --extra cpu --group dev pytest tests/ -q
(pytest is not installed locally; fallback:
  .venv/bin/python scratch/run_pair_split_tests.py)
"""

from __future__ import annotations

import math

import torch

from tinydet.modules.pair_split import box_iou, build_pair_targets, pair_split_loss


def test_box_iou_matches_hand_computation() -> None:
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 10.0], [0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    iou = box_iou(a, b)
    assert torch.allclose(iou, torch.tensor([[50.0 / 150.0, 1.0, 0.0]]), atol=1e-6)


def test_box_iou_empty_is_zero() -> None:
    a = torch.zeros((0, 4))
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert box_iou(a, b).shape == (0, 1)


def test_isolated_gt_gets_zero_target() -> None:
    det = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    batch = torch.tensor([0])
    gt = torch.tensor([[0.0, 0.0, 4.0, 4.0], [100.0, 100.0, 104.0, 104.0]])
    gt_batch = torch.tensor([0, 0])
    targets, valid = build_pair_targets(det, batch, gt, gt_batch)
    assert valid.tolist() == [True]
    assert torch.allclose(targets, torch.zeros(1, 4))


def test_close_pair_gets_relative_target() -> None:
    # detection matches gt0 exactly; gt1 is 6 px to the right, same size
    det = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    batch = torch.tensor([0])
    gt = torch.tensor([[0.0, 0.0, 4.0, 4.0], [6.0, 0.0, 10.0, 4.0]])
    gt_batch = torch.tensor([0, 0])
    targets, valid = build_pair_targets(det, batch, gt, gt_batch)
    assert valid.tolist() == [True]
    assert targets[0, 0].item() == 1.0
    assert math.isclose(targets[0, 1].item(), 6.0 / 4.0, rel_tol=1e-5)  # dx in units of w
    assert math.isclose(targets[0, 2].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(targets[0, 3].item(), 0.0, abs_tol=1e-6)  # same area -> ratio 1


def test_size_ratio_is_log_of_sqrt_area_ratio() -> None:
    det = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    batch = torch.tensor([0])
    gt = torch.tensor([[0.0, 0.0, 4.0, 4.0], [3.0, 0.0, 11.0, 8.0]])  # 8x8 vs 4x4, centre 5.4px away
    gt_batch = torch.tensor([0, 0])
    targets, _ = build_pair_targets(det, batch, gt, gt_batch)
    assert math.isclose(targets[0, 3].item(), math.log(2.0), rel_tol=1e-5)


def test_unmatched_detection_is_invalid() -> None:
    det = torch.tensor([[50.0, 50.0, 54.0, 54.0]])
    batch = torch.tensor([0])
    gt = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    gt_batch = torch.tensor([0])
    targets, valid = build_pair_targets(det, batch, gt, gt_batch)
    assert valid.tolist() == [False]
    assert torch.allclose(targets, torch.zeros(1, 4))


def test_targets_are_per_image() -> None:
    # two images, each with one isolated GT: no cross-image pairing allowed
    det = torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 0.0, 5.0, 4.0]])
    batch = torch.tensor([0, 1])
    gt = torch.tensor([[0.0, 0.0, 4.0, 4.0], [1.0, 0.0, 5.0, 4.0]])
    gt_batch = torch.tensor([0, 1])
    targets, valid = build_pair_targets(det, batch, gt, gt_batch)
    assert valid.tolist() == [True, True]
    assert torch.allclose(targets[:, 0], torch.zeros(2))


def test_empty_inputs() -> None:
    det = torch.zeros((0, 4))
    batch = torch.zeros((0,), dtype=torch.long)
    gt = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    gt_batch = torch.tensor([0])
    targets, valid = build_pair_targets(det, batch, gt, gt_batch)
    assert targets.shape == (0, 4) and valid.shape == (0,)


def test_loss_regression_only_on_positives() -> None:
    # two valid rows: one positive with wrong offsets, one negative with garbage
    # offsets that must be ignored
    out = torch.tensor([[5.0, 0.5, 0.0, 0.0], [-5.0, 99.0, 99.0, 99.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([True, True])
    _loss, parts = pair_split_loss(out, targets, valid)
    assert math.isclose(parts["reg"], 0.5 / 3, abs_tol=1e-5)  # mean of |0.5|,|0|,|0|
    assert parts["cls"] < 0.01  # both logits already on the right side


def test_loss_ignores_invalid_rows() -> None:
    out = torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([True, False])
    loss_a, _ = pair_split_loss(out, targets, valid)
    loss_b, _ = pair_split_loss(out[:1], targets[:1], valid[:1])
    assert math.isclose(loss_a.item(), loss_b.item(), rel_tol=1e-6)


def test_loss_is_zero_when_nothing_is_valid() -> None:
    out = torch.zeros(2, 4, requires_grad=True)
    targets = torch.zeros(2, 4)
    valid = torch.zeros(2, dtype=torch.bool)
    loss, parts = pair_split_loss(out, targets, valid)
    assert loss.item() == 0.0
    assert parts == {"cls": 0.0, "reg": 0.0}
    loss.backward()  # must not raise
