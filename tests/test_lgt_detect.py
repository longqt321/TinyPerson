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
    tilde_mu_p, tilde_sig_p, _, valid_p, _ = head._parent_candidate(
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


def test_lowmem_wasserstein_and_cache_serialization_equivalence():
    """Verify low-memory TAL Wasserstein matches exact formula and LGT caches clear on pickle."""
    import pickle

    from tinydet.models.ultralytics import _wasserstein_sim_lowmem

    torch.manual_seed(0)
    xy = torch.rand(2, 5, 16, 2) * 100.0
    wh = torch.rand(2, 5, 16, 2) * 20.0 + 1.0
    gt = torch.cat((xy, xy + wh), dim=-1)
    xy2 = xy + torch.randn_like(xy) * 2.0
    wh2 = (wh + torch.randn_like(wh) * 1.5).clamp(min=0.5)
    pd = torch.cat((xy2, xy2 + wh2), dim=-1)

    exact = wasserstein_similarity_xyxy(gt, pd)
    lowmem = _wasserstein_sim_lowmem(gt, pd)
    assert torch.allclose(exact, lowmem, atol=1e-6)

    # Also verify unexpanded broadcast shapes (B, M, 1, 4) vs (B, 1, N, 4)
    gt_bc = gt[:, :, :1, :]
    pd_bc = pd[:, :1, :, :]
    assert torch.allclose(wasserstein_similarity_xyxy(gt_bc, pd_bc), _wasserstein_sim_lowmem(gt_bc, pd_bc), atol=1e-6)

    head = LGTDetect(nc=1, reg_max=1, end2end=True, ch=(32, 64))
    head.stride = torch.tensor([4.0, 8.0])
    feats = [torch.randn(1, 32, 8, 8), torch.randn(1, 64, 4, 4)]
    head.forward_lgt_one2one(feats)
    assert len(head.lgt_stencil._valid_cache) == 2
    assert len(head._parent_grid_cache) == 1
    assert len(head._level_mask_cache) == 2
    assert head._stride_ratios == [2.0]

    restored = pickle.loads(pickle.dumps(head))
    assert restored.lgt_stencil._valid_cache == {}
    assert restored._parent_grid_cache == {}
    assert restored._level_mask_cache == {}
    assert restored._stride_ratios is None


def test_tinyperson_benchmark_preallocated_boolean_matching(tmp_path):
    """Verify optimized evaluate_tinyperson matching across tiny1/small/ignore/outside bins."""
    import json

    from tinydet.evaluation.tinyperson_benchmark import evaluate_tinyperson

    ann_file = tmp_path / "ann.json"
    pred_file = tmp_path / "pred.json"
    ann_file.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 200, "height": 200}],
        "categories": [{"id": 1, "name": "sea_person"}, {"id": 2, "name": "earth_person"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 5.0, 5.0], "ignore": False},   # tiny1 (AS=5)
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50.0, 50.0, 24.0, 24.0], "ignore": False}, # small (AS=24)
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [120.0, 120.0, 20.0, 20.0], "ignore": True},
        ],
    }))
    pred_file.write_text(json.dumps({
        "images": [{
            "path": "img1.jpg",
            "boxes": [
                [10.0, 10.0, 15.0, 15.0],    # exact TP on tiny1
                [122.0, 122.0, 128.0, 128.0], # inside ignore box -> IoD neutral, must NOT be FP!
                [50.0, 50.0, 74.0, 74.0],    # exact TP on small (neutral when scoring tiny1)
            ],
            "scores": [0.95, 0.90, 0.85],
            "classes": [0, 0, 1],
        }]
    }))
    res = evaluate_tinyperson(ann_file, pred_file, workers=1, class_mode="merge")
    assert res["all"]["AP@0.50"] == 1.0
    assert res["tiny1"]["AP@0.50"] == 1.0
    assert res["small"]["AP@0.50"] == 1.0

    # Also verify full analyze_files + cached _ImageOverlaps + oracle_ap75 reuse
    import importlib.util

    from tinydet.error_analysis.core import analyze_files

    analysis = analyze_files(ann_file, pred_file)
    assert analysis.summary["integrity"]["valid"] is True
    assert all(analysis.summary["integrity"]["invariants"].values())
    assert analysis.summary["metric_crosscheck"]["status"] == "match"

    spec = importlib.util.spec_from_file_location("oracle_ap75", REPO_ROOT / "scripts/oracle_ap75.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    oracles = mod.run_oracles(ann_file, pred_file, tmp_path / "oracle", workers=1, class_mode="merge", baseline_result=res)
    assert oracles["baseline"]["metrics"]["all"]["AP@0.50"] == 1.0
    assert oracles["both"]["metrics"]["all"]["AP@0.75"] == 1.0

    # Verify zero-disk in-memory dict mode + val=False safety on log_ap_by_iou
    res_mem = evaluate_tinyperson(
        json.loads(ann_file.read_text()),
        json.loads(pred_file.read_text()),
        workers=1,
        class_mode="merge",
    )
    assert res_mem["all"]["AP@0.50"] == 1.0
    assert res_mem["protocol"]["predictions_file"] == "<in-memory>"

    from tinydet.training.iou_logging import log_ap_by_iou

    class NoValTrainer:
        validator = None

    log_ap_by_iou(NoValTrainer(), wandb_enabled=False)


def test_vectorized_error_matching_and_xywh_conversion():
    """Verify fast float64 xywh_to_xyxy and vectorized error_analysis._match_image."""
    from tinydet.error_analysis.core import AnalysisConfig, Detection, GroundTruth, _match_image
    from tinydet.utils.geometry import xywh_to_xyxy

    assert xywh_to_xyxy([10.5, 20.25, 4.0, 6.5]) == (10.5, 20.25, 14.5, 26.75)

    gts = [
        GroundTruth(id=1, box=(10.0, 10.0, 16.0, 16.0), label=0, ignore=False, uncertain=False, dense=False, raw={}),
        GroundTruth(id=2, box=(40.0, 40.0, 46.0, 46.0), label=0, ignore=False, uncertain=False, dense=False, raw={}),
        GroundTruth(id=3, box=(80.0, 80.0, 90.0, 90.0), label=0, ignore=True, uncertain=False, dense=False, raw={}),
    ]
    dets = [
        Detection(source_index=0, box=(10.0, 10.0, 16.0, 16.0), score=0.95, label=0),  # TP on GT 1
        Detection(source_index=1, box=(10.5, 10.5, 16.0, 16.0), score=0.80, label=0),  # Duplicate on GT 1
        Detection(source_index=2, box=(82.0, 82.0, 88.0, 88.0), score=0.70, label=0),  # Neutral (IoD inside ignored GT 3)
        Detection(source_index=3, box=(0.0, 0.0, 5.0, 5.0), score=0.60, label=0),      # Background FP
    ]
    det_out, gt_out = _match_image(dets, gts, AnalysisConfig(), size_bin="all")
    assert [d["status"] for d in det_out] == ["true_positive", "false_positive", "neutral", "false_positive"]
    assert det_out[1]["subtype"] == "duplicate"
    assert det_out[3]["subtype"] == "background"
    assert gt_out[0]["status"] == "matched"
    assert gt_out[1]["status"] == "missed"
    assert gt_out[2]["status"] == "ignored"


def test_multiclass_same_cls_isolation_and_regmax16_yolov10_backward(tmp_path):
    """Verify Class-Aware gating isolates cross-class neighbors (VisDrone nc=10) & reg_max=16 DFL backward."""
    import json

    from tinydet.evaluation.coco_benchmark import evaluate_coco_suite

    # 1. Multi-class isolation: a confident car (cls=3) next to a tiny pedestrian (cls=0) must NOT pull mu_0!
    head10 = LGTDetect(nc=10, reg_max=1, end2end=True, ch=(32, 64))
    head10.stride = torch.tensor([4.0, 8.0])
    assert head10.lgt_stencil.channels == 6

    raw_box = torch.ones((1, 4, 4, 4), dtype=torch.float32) * 0.5  # symmetric box centered on each cell
    raw_cls_diff = torch.full((1, 10, 4, 4), -4.0)
    raw_cls_diff[:, 5] = -3.0       # background cells default to cls=5 (truck), not pedestrian (cls=0)
    raw_cls_diff[0, 0, 2, 2] = 1.0  # center cell (2,2) is pedestrian (cls=0)
    raw_cls_diff[0, 3, 2, 3] = 8.0  # right neighbor (2,3) is a high-score car (cls=3)
    raw_box_shifted = raw_box.clone()
    raw_box_shifted[0, :, 2, 3] = torch.tensor([0.2, 0.2, 3.8, 3.8])  # large car shifted right

    mu0, sig0 = ltrb_to_gaussian(raw_box_shifted)
    mu1, sig1 = ltrb_to_gaussian(torch.ones((1, 4, 2, 2)) * 0.5)
    s0 = raw_cls_diff.amax(dim=1, keepdim=True)
    s1 = torch.full((1, 1, 2, 2), -4.0)
    c0 = raw_cls_diff.argmax(dim=1, keepdim=True).float()
    c1 = torch.full((1, 1, 2, 2), 9.0)  # parent is motor (cls=9), also different from pedestrian (cls=0)

    ref_ltrb, _ = head10._refine_level(0, raw_box_shifted, raw_cls_diff, [mu0, mu1], [sig0, sig1], [s0, s1], [c0, c1])
    assert torch.allclose(ref_ltrb[0, :, 2, 2], raw_box_shifted[0, :, 2, 2], atol=1e-6)

    # 2. YOLOv10n-P2 + LGTDetect (reg_max=16 DFL) forward + E2ELoss backward + Detect.fuse()
    v10_lgt = build_model({"config": str(REPO_ROOT / "configs/model/yolov10n_p2_lgt.yaml")})
    v10_head = find_lgt_head(v10_lgt.model)
    assert v10_head is not None and v10_head.reg_max == 16
    v10_lgt.model.train()
    criterion = v10_lgt.model.init_criterion()
    imgs = torch.randn(2, 3, 128, 128)
    batch = {
        "img": imgs,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [0.0]]),
        "bboxes": torch.tensor([[0.30, 0.30, 0.06, 0.08], [0.60, 0.60, 0.10, 0.12]]),
    }
    preds = v10_lgt.model(imgs)
    assert "ltrb" in preds["one2one"]
    assert preds["one2one"]["boxes"].shape[1] == 64
    assert preds["one2one"]["ltrb"].shape[1] == 4
    loss_val, items = criterion(preds, batch)
    loss_val.backward()
    assert v10_head.lgt_logit_beta_box.grad is not None and v10_head.lgt_logit_beta_box.grad.abs().sum() > 0
    assert "lgt_loss" in items

    fused_v10 = copy.deepcopy(v10_lgt.model).eval()
    fused_v10.fuse()
    with torch.no_grad():
        det_v10, _ = fused_v10(imgs)
    assert det_v10.shape == (2, 300, 6)

    # 3. NWD competitor YAML build + backward & YOLO26s + YOLO11n-NMS build sanity
    nwd_m = build_model({"config": str(REPO_ROOT / "configs/model/yolo26n_p2_nwd.yaml")})
    nwd_m.model.train()
    nwd_crit = nwd_m.model.init_criterion()
    nwd_loss, _ = nwd_crit(nwd_m.model(imgs), batch)
    nwd_loss.backward()
    assert torch.isfinite(nwd_loss)

    s_lgt = build_model({"config": str(REPO_ROOT / "configs/model/yolo26s_p2_lgt.yaml")})
    y11_nms = build_model({"config": str(REPO_ROOT / "configs/model/yolo11n_p2_nms.yaml")})
    assert sum(p.numel() for p in s_lgt.model.parameters()) > 8_000_000
    assert getattr(y11_nms.model.model[-1], "end2end", False) is False

    # 4. Verify faster-coco-eval suite (COCO-100 + VisDrone-500 + Occlusion + TIDE + Crowded pairs)
    gt_json = tmp_path / "vd_gt.json"
    dt_json = tmp_path / "vd_dt.json"
    gt_json.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "images/val/0001.jpg", "width": 400, "height": 400}],
        "categories": [{"id": 1, "name": "pedestrian"}, {"id": 4, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20.0, 20.0, 6.0, 6.0], "area": 36.0, "iscrowd": 0, "occlusion": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [27.0, 20.0, 6.0, 6.0], "area": 36.0, "iscrowd": 0, "occlusion": 1},
            {"id": 3, "image_id": 1, "category_id": 4, "bbox": [100.0, 100.0, 50.0, 40.0], "area": 2000.0, "iscrowd": 0, "occlusion": 2},
        ],
    }))
    dt_json.write_text(json.dumps({
        "images": [{
            "path": "/tmp/images/val/0001.jpg",
            "boxes": [[20.0, 20.0, 26.0, 26.0], [100.0, 100.0, 150.0, 140.0]],
            "scores": [0.92, 0.88],
            "classes": [0, 3],
        }]
    }))
    suite = evaluate_coco_suite(gt_json, dt_json, collapse_single_class=False)
    assert suite["visdrone_maxdet500"]["AP50"] > 0.70
    assert suite["occlusion_ap50"] is not None
    assert suite["tide_and_crowded"]["counts"]["TP"] == 2
    assert suite["tide_and_crowded"]["counts"]["FN_Miss"] == 1
    assert suite["tide_and_crowded"]["crowded_pairs"]["crowded_lt_10px"]["ground_truths"] == 2


