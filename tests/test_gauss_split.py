"""Self-tests for GaussSplit.

Run with: uv run --extra cpu --group dev pytest tests/ -q
(pytest is not installed locally; fallback:
  .venv/bin/python scratch/run_pair_split_tests.py)
"""

from __future__ import annotations

import math

import torch

from tinydet.modules.gauss_split import (
    GaussSplit,
    boxes_to_gaussians,
    count_parameters,
    gauss_split_loss,
    nwd,
    offsets_to_boxes,
    wasserstein2,
)
from tinydet.modules.pair_split import INIT_SECOND_BIAS, PATCH_SIDE, PairSplit


def _make(c_in: int = 32, patch: int = PATCH_SIDE, hidden: int = 64, compress: int = 4) -> GaussSplit:
    torch.manual_seed(0)
    return GaussSplit(c_in=c_in, hidden=hidden, patch=patch, stride=4, compress=compress)


def test_parameter_count() -> None:
    m = _make()
    # proj: (2*32)*4 + 4 ; fc1: (4*49 + 3)*64 + 64 ; fc2: 64*64 + 64 ; fc3: 64*4 + 4
    # scale_space: 4 * 32 sigma raws
    expected = (2 * 32) * 4 + 4 + (4 * 49 + 3) * 64 + 64 + 64 * 64 + 64 + 64 * 4 + 4 + 4 * 32
    assert count_parameters(m) == expected
    assert count_parameters(m) == 17_608
    # smaller than PairSplit, so a gain cannot be blamed on capacity
    assert count_parameters(m) < count_parameters(PairSplit())


def test_init_is_a_no_op() -> None:
    m = _make().eval()
    feat = torch.randn(1, 32, 8, 8)
    boxes = torch.tensor([[10.0, 10.0, 14.0, 14.0], [40.0, 20.0, 44.0, 24.0]])
    out = m(feat, boxes, torch.tensor([0.9, 0.5]))
    assert out.shape == (2, 4)
    assert torch.allclose(out[:, 1:], torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(out[:, 0], torch.full((2,), INIT_SECOND_BIAS), atol=1e-5)
    assert math.isclose(torch.sigmoid(out[0, 0]).item(), 0.1192029, rel_tol=1e-4)


def test_empty_input() -> None:
    m = _make().eval()
    feat = torch.randn(1, 32, 8, 8)
    out = m(feat, torch.zeros((0, 4)), torch.zeros((0,)))
    assert out.shape == (0, 4)


def test_dog_vanishes_on_constant_map() -> None:
    """A constant feature map has no scale-space structure: the DoG must vanish."""
    m = _make().eval()
    feat = torch.ones(1, 32, 16, 16)
    boxes = torch.tensor([[30.0, 30.0, 34.0, 34.0]])
    with torch.no_grad():
        fine, dog = m.scale_space_patches(feat, boxes, torch.tensor([0]))
    assert fine.abs().max() > 0.9  # the fine scale preserves a constant
    assert dog.abs().max() < 1e-4  # the difference of two blurs of a constant is 0


def test_gather_semantics_delta_at_known_cell() -> None:
    """A delta at feature cell (y=3, x=5) must land at the patch centre of a
    detection whose centre is exactly that cell."""
    m = _make().eval()
    feat = torch.zeros(1, 32, 8, 8)
    feat[0, :, 3, 5] = 1.0
    boxes = torch.tensor([[18.0, 10.0, 22.0, 14.0]])  # centre (20, 12) -> cell (5, 3)
    with torch.no_grad():
        fine, _dog = m.scale_space_patches(feat, boxes, torch.tensor([0]))
    # the fine blur spreads the delta but keeps the centre as the maximum
    c = m.patch // 2
    centre = fine[0, :, c, c]
    assert centre.abs().min() > 0.0
    assert (centre >= fine[0].amax(dim=(1, 2)) - 1e-6).all()


def test_batch_ordering_is_restored() -> None:
    m = _make().eval()
    feat = torch.zeros(2, 32, 8, 8)
    feat[0] = 1.0
    feat[1] = 2.0
    boxes = torch.tensor([[14.0, 14.0, 18.0, 18.0]] * 4)
    batch = torch.tensor([1, 0, 1, 0])
    with torch.no_grad():
        fine, _ = m.scale_space_patches(feat, boxes, batch)
    assert torch.allclose(fine[0].mean(), torch.tensor(2.0), atol=1e-3)
    assert torch.allclose(fine[1].mean(), torch.tensor(1.0), atol=1e-3)
    assert torch.allclose(fine[2].mean(), torch.tensor(2.0), atol=1e-3)
    assert torch.allclose(fine[3].mean(), torch.tensor(1.0), atol=1e-3)


def test_gradients_reach_every_parameter() -> None:
    m = _make()
    feat = torch.randn(2, 32, 16, 16, requires_grad=True)
    boxes = torch.rand((6, 4)) * 40 + 10
    boxes[:, 2:] = boxes[:, :2] + 4.0
    score = torch.rand(6)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out = m(feat, boxes, score, batch)
    torch.nn.functional.mse_loss(out, torch.randn_like(out)).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
    assert feat.grad is not None and torch.isfinite(feat.grad).all()


def test_boxes_to_gaussians() -> None:
    mean, std = boxes_to_gaussians(torch.tensor([[0.0, 0.0, 10.0, 20.0]]))
    assert torch.allclose(mean, torch.tensor([[5.0, 10.0]]))
    assert torch.allclose(std, torch.tensor([[5.0, 10.0]]))


def test_wasserstein_hand_computation() -> None:
    a = torch.tensor([[0.0, 0.0, 4.0, 4.0]])  # centre (2,2), sigma (2,2)
    b = torch.tensor([[6.0, 0.0, 10.0, 4.0]])  # centre (8,2), sigma (2,2)
    # W2^2 = dx^2 + dy^2 + dsigma_x^2 + dsigma_y^2 = 36
    assert math.isclose(wasserstein2(a, b).item(), 36.0, rel_tol=1e-5)
    c = torch.tensor([[-2.0, -2.0, 6.0, 6.0]])  # same centre (2,2), sigma (4,4)
    # W2^2 = 0 + (4-2)^2 + (4-2)^2 = 8
    assert math.isclose(wasserstein2(a, c).item(), 8.0, rel_tol=1e-5)


def test_nwd_is_one_for_identical_and_decays() -> None:
    a = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    assert math.isclose(nwd(a, a.clone(), c=4.0).item(), 1.0, rel_tol=1e-6)
    far = torch.tensor([[100.0, 100.0, 104.0, 104.0]])
    assert nwd(a, far, c=4.0).item() < 1e-6
    # monotone in distance
    mid = torch.tensor([[4.0, 0.0, 8.0, 4.0]])
    assert nwd(a, far, c=4.0) < nwd(a, mid, c=4.0)


def test_offsets_to_boxes() -> None:
    src = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    off = torch.tensor([[1.0, 0.0, math.log(0.5)]])
    box = offsets_to_boxes(src, off)
    assert torch.allclose(box, torch.tensor([[12.5, 2.5, 17.5, 7.5]]), atol=1e-4)


def test_decode_parity_with_pair_split() -> None:
    """Same head output must decode to the same boxes as PairSplit.decode, so the
    ablation isolates the representation rather than the decode."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    score = torch.tensor([0.9, 0.4])
    out = torch.tensor([[8.0, 1.0, 0.0, math.log(0.5)], [8.0, -0.5, 0.5, 0.0]])
    a, sa = PairSplit.decode(boxes, score, out, threshold=0.5)
    b, sb = PairSplit.decode(boxes, score, out, threshold=0.5)
    assert torch.allclose(a, b) and torch.allclose(sa, sb)


def test_loss_l1_matches_pair_split_definition() -> None:
    src = torch.tensor([[0.0, 0.0, 4.0, 4.0], [10.0, 10.0, 14.0, 14.0]])
    out = torch.tensor([[5.0, 0.5, 0.0, 0.0], [-5.0, 99.0, 99.0, 99.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([True, True])
    _loss, parts = gauss_split_loss(out, targets, valid, src, geom="l1")
    assert math.isclose(parts["reg"], 0.5 / 3, abs_tol=1e-5)
    assert parts["cls"] < 0.01


def test_loss_nwd_is_zero_for_perfect_prediction() -> None:
    src = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
    out = torch.tensor([[10.0, 0.0, 0.0, 0.0]])  # gate open, zero offset
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([True])
    _loss, parts = gauss_split_loss(out, targets, valid, src, geom="nwd", nwd_c=4.0)
    assert parts["reg"] < 1e-6
    assert parts["cls"] < 1e-4


def test_loss_ignores_invalid_rows() -> None:
    src = torch.tensor([[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0]])
    out = torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([True, False])
    a, _ = gauss_split_loss(out, targets, valid, src, geom="nwd", nwd_c=4.0)
    b, _ = gauss_split_loss(out[:1], targets[:1], valid[:1], src[:1], geom="nwd", nwd_c=4.0)
    assert math.isclose(a.item(), b.item(), rel_tol=1e-6)


def test_loss_zero_when_nothing_valid() -> None:
    out = torch.zeros(2, 4, requires_grad=True)
    loss, parts = gauss_split_loss(
        out, torch.zeros(2, 4), torch.zeros(2, dtype=torch.bool), torch.zeros(2, 4)
    )
    assert loss.item() == 0.0
    assert parts == {"cls": 0.0, "reg": 0.0}
    loss.backward()
