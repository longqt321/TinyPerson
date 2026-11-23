"""Self-tests for PairSplit.

Run with: uv run --extra cpu --group dev pytest tests/ -q
(pytest is not installed locally; fallback:
  .venv/bin/python scratch/run_pair_split_tests.py)
"""

from __future__ import annotations

import math

import numpy as np
import torch

from tinydet.modules.pair_split import INIT_SECOND_BIAS, PATCH_SIDE, PairSplit, count_parameters


def _make_model(c_in: int = 32, patch: int = PATCH_SIDE, hidden: int = 64) -> PairSplit:
    torch.manual_seed(0)
    return PairSplit(c_in=c_in, patch=patch, hidden=hidden, stride=4)


def test_parameter_count_matches_docstring() -> None:
    m = _make_model()
    # proj 1x1: 32 -> 6 ; fc1: (6*49 + 3) -> 64 ; fc2: 64 -> 64 ; fc3: 64 -> 4
    expected = 32 * 6 + 6 + (6 * 49 + 3) * 64 + 64 + 64 * 64 + 64 + 64 * 4 + 4
    assert count_parameters(m) == expected
    assert count_parameters(m) == 23_690


def test_output_shape_and_init_behaviour() -> None:
    m = _make_model().eval()
    feat = torch.randn(1, 32, 8, 8)
    boxes = torch.tensor([[10.0, 10.0, 14.0, 14.0], [40.0, 20.0, 44.0, 24.0]])
    score = torch.tensor([0.9, 0.5])
    out = m(feat, boxes, score)
    assert out.shape == (2, 4)
    # at init: logit = INIT_SECOND_BIAS, offsets 0, log-ratio 0
    assert torch.allclose(out[:, 1:], torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(out[:, 0], torch.full((2,), INIT_SECOND_BIAS), atol=1e-5)
    assert math.isclose(torch.sigmoid(out[0, 0]).item(), 0.1192029, rel_tol=1e-4)


def test_empty_input_returns_empty() -> None:
    m = _make_model().eval()
    feat = torch.randn(1, 32, 8, 8)
    boxes = torch.zeros((0, 4))
    score = torch.zeros((0,))
    out = m(feat, boxes, score)
    assert out.shape == (0, 4)
    extra, sc = PairSplit.decode(boxes, score, out)
    assert extra.shape == (0, 4) and sc.shape == (0,)


def test_gather_semantics_delta_at_known_cell() -> None:
    """A delta placed at feature cell (y=3, x=5) must appear at the patch centre
    of a detection whose centre maps exactly onto that cell, and nowhere else."""
    m = _make_model().eval()
    h = w = 8
    feat = torch.zeros(1, 32, h, w)
    feat[0, :, 3, 5] = 1.0
    # centre at pixel (20, 12) -> feature (5.0, 3.0): exactly the centre of cell
    # (y=3, x=5) under align_corners=False, where cell i sits at (2i+1)/8 - 1.
    boxes = torch.tensor([[18.0, 10.0, 22.0, 14.0]])
    score = torch.tensor([0.5])
    # expose the gathered patch directly (pre-projection)
    captured = {}
    orig = m._gather_patches

    def spy(feat_, boxes_, batch_):
        out = orig(feat_, boxes_, batch_)
        captured["x"] = out[0].detach()
        return out

    m._gather_patches = spy
    m(feat, boxes, score)
    del m._gather_patches
    x = captured["x"]
    patch = x.reshape(32, m.patch, m.patch)  # dims: (channel, x-offset, y-offset)
    # the delta lands exactly at the patch centre and nowhere else
    c = m.patch // 2
    assert torch.allclose(patch[:, c, c], torch.ones(32), atol=1e-4)
    others = patch.clone()
    others[:, c, c] = 0.0
    assert others.abs().max() == 0.0


def test_batch_ordering_is_restored() -> None:
    """Interleaved batch indices must not permute the output rows."""
    m = _make_model().eval()
    feat = torch.zeros(2, 32, 8, 8)
    feat[0] = 1.0
    feat[1] = 2.0
    boxes = torch.tensor([[14.0, 14.0, 18.0, 18.0]] * 4)
    batch = torch.tensor([1, 0, 1, 0])
    with torch.no_grad():
        patches = m._gather_patches(feat, boxes, batch)
    # every sampled point is interior, so each row is a constant equal to the
    # image it came from
    flat = 32 * m.patch * m.patch
    assert torch.allclose(patches[0], torch.full((flat,), 2.0))
    assert torch.allclose(patches[1], torch.full((flat,), 1.0))
    assert torch.allclose(patches[2], torch.full((flat,), 2.0))
    assert torch.allclose(patches[3], torch.full((flat,), 1.0))


def test_gradients_reach_every_parameter() -> None:
    m = _make_model()
    feat = torch.randn(2, 32, 16, 16, requires_grad=True)
    boxes = torch.rand((6, 4)) * 40 + 10
    boxes[:, 2:] = boxes[:, :2] + 4.0
    score = torch.rand(6)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out = m(feat, boxes, score, batch)
    target = torch.randn_like(out)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
    assert feat.grad is not None and torch.isfinite(feat.grad).all()


def test_decode_emits_shifted_scaled_box() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    score = torch.tensor([0.8])
    # logit large -> p ~ 1 ; dx = 1.0 (one width right), dy = 0, ratio = log(0.5)
    out = torch.tensor([[8.0, 1.0, 0.0, math.log(0.5)]])
    extra, sc = PairSplit.decode(boxes, score, out, threshold=0.5)
    assert extra.shape == (1, 4)
    # centre moved by dx*w = 10 -> new centre (15, 5); side 5
    assert torch.allclose(extra[0], torch.tensor([12.5, 2.5, 17.5, 7.5]), atol=1e-4)
    assert math.isclose(sc[0].item(), 0.8 * torch.sigmoid(torch.tensor(8.0)).item(), rel_tol=1e-5)


def test_decode_threshold_blocks_low_probability() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    score = torch.tensor([0.8])
    out = torch.tensor([[-4.0, 1.0, 0.0, 0.0]])  # p = 0.018
    extra, _sc = PairSplit.decode(boxes, score, out, threshold=0.5)
    assert extra.shape == (0, 4)


def test_numpy_reference_of_mlp() -> None:
    """The learnable part (MLP) must agree with a plain numpy implementation."""
    m = _make_model().eval()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, m.fc1.in_features)).astype(np.float32)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        y_torch = m.fc3(torch.relu(m.fc2(torch.relu(m.fc1(xt))))).numpy()

    w1 = m.fc1.weight.detach().numpy()
    b1 = m.fc1.bias.detach().numpy()
    w2 = m.fc2.weight.detach().numpy()
    b2 = m.fc2.bias.detach().numpy()
    w3 = m.fc3.weight.detach().numpy()
    b3 = m.fc3.bias.detach().numpy()
    h1 = np.maximum(x @ w1.T + b1, 0.0)
    h2 = np.maximum(h1 @ w2.T + b2, 0.0)
    y_np = h2 @ w3.T + b3
    assert np.allclose(y_np, y_torch, atol=1e-5)


def test_border_detection_does_not_crash() -> None:
    """Detections at the image edge sample zero padding, not garbage."""
    m = _make_model().eval()
    feat = torch.randn(1, 32, 8, 8)
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [28.0, 28.0, 32.0, 32.0]])
    score = torch.tensor([0.5, 0.5])
    out = m(feat, boxes, score)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 4)


def test_decode_is_scale_invariant() -> None:
    """The DECODE is scale-invariant: ``dx`` is in units of the box width, so the
    same head output produces the same *relative* shift at two box scales.

    The head input is not scale-free (it sees ``log w``), so this test only
    asserts the decode property, not identical logits across scales.
    """
    m = _make_model().eval()
    feat = torch.zeros(1, 32, 64, 64)
    b_small = torch.tensor([[8.0, 8.0, 12.0, 12.0]])  # 4 px box at (10, 10)
    b_large = torch.tensor([[80.0, 80.0, 120.0, 120.0]])  # 40 px box at (100, 100)
    score = torch.tensor([0.7])
    out = m(feat, b_small, score)
    out2 = m(feat, b_large, score)
    assert torch.isfinite(out).all() and torch.isfinite(out2).all()
    e1, _ = PairSplit.decode(b_small, score, out, threshold=0.0)
    e2, _ = PairSplit.decode(b_large, score, out2, threshold=0.0)
    rel1 = (e1[0, 0] - 10.0) / 4.0   # centre shift / width
    rel2 = (e2[0, 0] - 100.0) / 40.0
    assert math.isclose(rel1.item(), rel2.item(), rel_tol=1e-4)
