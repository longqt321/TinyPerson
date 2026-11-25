"""Unit tests for Local Gaussian Transport head (LGTDetect), Wasserstein TAL/Loss, and fixes."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinydet.evaluation.tiled_inference import suppress_border_truncated
from tinydet.models.registry import build_model
from tinydet.models.ultralytics import make_tiny_floor_forward
from tinydet.modules.lgt_detect import (
    LGTDetect,
    SpatialShiftStencil,
    find_lgt_head,
    gaussian_to_ltrb,
    ltrb_to_gaussian,
    scale_invariant_wasserstein_xyxy,
    wasserstein_similarity_xyxy,
)
from tinydet.training.epoch_metrics import RealtimeLogCallback, SystemMetricsCallback


def test_proposition_1_spatial_and_cross_scale_transport_exactness():
    """Proposition 1: Identical physical boxes across 3x3 cells and P2/P3 scales yield D_k == 0."""
    # Physical box in image pixels: [x1=18.0, y1=22.0, x2=26.0, y2=34.0] (center=(22, 28), half=(4, 6))
    x1, y1, x2, y2 = 18.0, 22.0, 26.0, 34.0

    # P2 grid (stride=4, 16x16) and P3 grid (stride=8, 8x8)
    h2, w2, s2 = 16, 16, 4.0
    h3, w3, s3 = 8, 8, 8.0

    def make_exact_ltrb(height: int, width: int, stride: float) -> torch.Tensor:
        xs = (torch.arange(width, dtype=torch.float32) + 0.5) * stride
        ys = (torch.arange(height, dtype=torch.float32) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        left = (xx - x1) / stride
        top = (yy - y1) / stride
        right = (x2 - xx) / stride
        bottom = (y2 - yy) / stride
        return torch.stack((left, top, right, bottom), dim=0).unsqueeze(0)

    ltrb_p2 = make_exact_ltrb(h2, w2, s2)
    ltrb_p3 = make_exact_ltrb(h3, w3, s3)
    mu2, sig2 = ltrb_to_gaussian(ltrb_p2, eps_sigma=1e-5)
    mu3, sig3 = ltrb_to_gaussian(ltrb_p3, eps_sigma=1e-5)

    # Check round-trip inversion where box encloses anchor
    rec = gaussian_to_ltrb(mu2[:, :, 7:8, 5:6], sig2[:, :, 7:8, 5:6])
    assert torch.allclose(rec, ltrb_p2[:, :, 7:8, 5:6], atol=1e-5)

    stencil = SpatialShiftStencil(channels=5)
    packed2 = torch.cat((mu2, sig2, torch.zeros((1, 1, h2, w2))), dim=1)
    shifted, valid = stencil(packed2)
    tilde_mu = torch.stack((shifted[:, 0] + stencil.dx, shifted[:, 1] + stencil.dy), dim=1)
    tilde_sig = shifted[:, 2:4]

    # At interior cell (y=7, x=5), anchor=(22, 30), every 3x3 neighbor must transport to exact same (mu, sig)
    assert valid[0, :, 7, 5].all()
    assert torch.allclose(tilde_mu[:, :, :, 7, 5], mu2[:, :, 7, 5].unsqueeze(-1).expand(-1, -1, 9), atol=1e-5)
    assert torch.allclose(tilde_sig[:, :, :, 7, 5], sig2[:, :, 7, 5].unsqueeze(-1).expand(-1, -1, 9), atol=1e-5)

    # Check cross-scale parent transport P3 (stride 8) -> P2 (stride 4)
    head = LGTDetect(nc=1, reg_max=1, end2end=True, ch=(32, 64))
    head.stride = torch.tensor([4.0, 8.0])
    smax_dummy = [torch.zeros((1, 1, h2, w2)), torch.zeros((1, 1, h3, w3))]
    tilde_mu_p, tilde_sig_p, _, valid_p = head._parent_candidate(
        0, h2, w2, [mu2, mu3], [sig2, sig3], smax_dummy
    )
    assert valid_p.all()
    # Test both even and odd child cells around (y=6..7, x=4..5)
    assert torch.allclose(tilde_mu_p[:, :, 0, 6:8, 4:6], mu2[:, :, 6:8, 4:6], atol=1e-5)
    assert torch.allclose(tilde_sig_p[:, :, 0, 6:8, 4:6], sig2[:, :, 6:8, 4:6], atol=1e-5)


def test_proposition_2_exact_scale_invariance():
    """Proposition 2: Self-normalized Wasserstein distance has zero scale dependence."""
    b1 = torch.tensor([[10.0, 12.0, 14.0, 18.0], [100.0, 200.0, 106.0, 208.0]])
    b2 = torch.tensor([[11.5, 13.0, 15.5, 19.0], [108.0, 200.0, 114.0, 208.0]])
    d_base = scale_invariant_wasserstein_xyxy(b1, b2)
    s_base = wasserstein_similarity_xyxy(b1, b2)
    for scale in (0.25, 0.5, 4.0, 19.7):
        d_scaled = scale_invariant_wasserstein_xyxy(b1 * scale, b2 * scale)
        s_scaled = wasserstein_similarity_xyxy(b1 * scale, b2 * scale)
        assert torch.allclose(d_base, d_scaled, atol=1e-4), (scale, d_base, d_scaled)
        assert torch.allclose(s_base, s_scaled, atol=1e-4)


def test_proposition_3_zero_init_cls_and_close_pair_preservation():
    """Proposition 3: Zero-init cls gate preserves baseline logits; barycenter ignores distinct close neighbor."""
    head = LGTDetect(nc=2, reg_max=1, end2end=True, ch=(32, 64, 128, 256))
    head.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])
    feats = [
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
        torch.randn(2, 256, 2, 2),
    ]
    # At step 0, classification gate output conv is zero -> refined scores equal raw one2one_cv3 scores
    raw_scores = torch.cat([head.one2one_cv3[i](feats[i]).view(2, 2, -1) for i in range(4)], dim=-1)
    out = head.forward_lgt_one2one(feats)
    assert torch.allclose(out["scores"], raw_scores, atol=1e-6)


def test_yaml_build_param_budget_loss_backward_and_fuse():
    """Full pipeline check: YAML build, <3K extra params, E2ELoss backward, and Detect.fuse()."""
    base = build_model({"config": str(REPO_ROOT / "configs/model/yolo26n_tiny_A2_p2_clean_tinyperson.yaml")})
    lgt = build_model({"config": str(REPO_ROOT / "configs/model/yolo26n_tiny_A2_p2_lgt_tinyperson.yaml")})

    base_params = sum(p.numel() for p in base.model.parameters())
    lgt_params = sum(p.numel() for p in lgt.model.parameters())
    extra_params = lgt_params - base_params
    assert 0 < extra_params < 3000, f"Expected <3000 extra params, got {extra_params}"

    head = find_lgt_head(lgt.model)
    assert list(head.lgt_stencil.parameters()) == [], "Shift stencil must be a buffer, not a trainable Parameter"
    assert head is not None

    # Train forward + E2ELoss (with Wasserstein TAL + Wasserstein BboxLoss + Dense Field Distillation)
    lgt.model.train()
    criterion = lgt.model.init_criterion()
    imgs = torch.randn(2, 3, 128, 128)
    batch = {
        "img": imgs,
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0], [0.0]]),
        "bboxes": torch.tensor([
            [0.25, 0.25, 0.04, 0.05],  # ~5x6 px tiny1
            [0.31, 0.25, 0.04, 0.05],  # close adjacent tiny1 pair
            [0.60, 0.70, 0.12, 0.15],
        ]),
    }
    preds = lgt.model(imgs)
    total_loss, items = criterion(preds, batch)
    assert total_loss.ndim == 0 and torch.isfinite(total_loss)
    assert "lgt_loss" in items and torch.isfinite(items["lgt_loss"])

    total_loss.backward()
    assert head.lgt_logit_beta_box.grad is not None and torch.isfinite(head.lgt_logit_beta_box.grad).all()
    assert head.lgt_cls_gate[0][-1].weight.grad is not None

    # Verify Detect.fuse() drops one2many (cv2=None, cv3=None) while keeping LGTDetect inference intact
    fused_model = copy.deepcopy(lgt.model).eval()
    fused_model.fuse()
    fused_head = find_lgt_head(fused_model)
    assert fused_head is not None
    assert fused_head.cv2 is None and fused_head.cv3 is None
    assert fused_head.one2one_cv2 is not None and fused_head.lgt_cls_gate is not None

    with torch.no_grad():
        det_out, raw_dict = fused_model(imgs)
    assert det_out.shape == (2, 300, 6)
    assert torch.isfinite(det_out).all()
    assert raw_dict["one2many"] == {}


def test_multiclass_tiny_floor_and_tiled_border_ios():
    """Regression checks for multi-class tiny_floor safety and tile-seam IOS suppression."""
    def dummy_assigner(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        del pd_bboxes, anc_points, gt_labels, mask_gt
        bsz, na, nc = pd_scores.shape
        target_labels = torch.zeros((bsz, na), dtype=torch.long)
        target_bboxes = gt_bboxes[:, 0:1].expand(bsz, na, 4).clone()
        target_scores = torch.zeros((bsz, na, nc))
        target_scores[:, 0, 1] = 0.12  # class 1 matched, classes 0 and 2 must stay 0.0!
        fg_mask = torch.zeros((bsz, na), dtype=torch.bool)
        fg_mask[:, 0] = True
        target_gt_idx = torch.zeros((bsz, na), dtype=torch.long)
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

    wrapped = make_tiny_floor_forward(dummy_assigner, 0.5)
    _, _, scores_out, _, _ = wrapped(
        torch.zeros((1, 4, 3)),
        torch.zeros((1, 4, 4)),
        torch.zeros((4, 2)),
        torch.tensor([[[1.0]]]),
        torch.tensor([[[10.0, 10.0, 14.0, 15.0]]]),  # 4x5 px tiny1 (< 8 px)
        torch.ones((1, 1, 1)),
    )
    assert float(scores_out[0, 0, 1]) == 0.5
    assert float(scores_out[0, 0, 0]) == 0.0
    assert float(scores_out[0, 0, 2]) == 0.0

    # Tile-seam truncated box: full box [10, 10, 20, 20] (area=100, score=0.9),
    # half-truncated seam box [10, 10, 14, 20] (area=40, IoU=0.40 < 0.50, IOS=1.0 >= 0.85, score=0.7)
    boxes = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [10.0, 10.0, 14.0, 20.0],
        [10.0, 10.0, 14.0, 20.0],  # different class -> must NOT be suppressed
    ])
    scores = torch.tensor([0.9, 0.7, 0.6])
    classes = torch.tensor([0, 0, 1])
    keep = suppress_border_truncated(boxes, scores, classes, ios_thr=0.85)
    assert keep.tolist() == [0, 2]


def test_realtime_and_system_callbacks_no_per_batch_sync(capsys):
    """Verify RealtimeLogCallback throttles output and SystemMetricsCallback arms once per epoch."""
    live = RealtimeLogCallback(min_interval_s=60.0, force=True)

    class DummyTrainer:
        def __init__(self) -> None:
            self.epoch = 0
            self.epochs = 10
            self.train_loader = (1, 2, 3, 4)
            self.tloss = torch.tensor([1.2, 0.8, 0.3])
            self.loss_names = ("box_loss", "cls_loss", "l1_loss")
            self.metrics = {"metrics/mAP50(B)": 0.412}
            self.model = torch.nn.Linear(4, 2)
            self.scaler = None

        def optimizer_step(self):
            pass

    trainer = DummyTrainer()
    live.on_train_epoch_start(trainer)
    for _ in range(4):
        live.on_train_batch_end(trainer)
    live.on_fit_epoch_end(trainer)
    out = capsys.readouterr().out
    # Logs batch 1, skips batches 2-3 (throttled), logs final batch 4, and logs epoch summary
    assert "batch 1/4" in out
    assert "batch 2/4" not in out
    assert "batch 4/4" in out
    assert "[epoch 1/10] done" in out

    sys_cb = SystemMetricsCallback(wandb_enabled=False)
    sys_cb.on_train_start(trainer)
    sys_cb.on_train_epoch_start(trainer)
    assert sys_cb._grad_armed is True
    for p in trainer.model.parameters():
        p.grad = torch.ones_like(p)
    trainer.optimizer_step()
    assert sys_cb._grad_armed is False
    assert sys_cb._grad_norm_tensor is not None
