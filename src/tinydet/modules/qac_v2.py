"""Distributional Quality-Aware Calibration (D-QAC / QAC v2) for tiny person detection.

Upgrades QAC v1 with two theoretically grounded, parameter-efficient mechanisms:
1. Distributional Uncertainty-Gated Calibration (DUGC): Extracts top-k boundary
   distribution certainty from DFL logits (only 52 parameters across 4 scales) and
   applies Semantic-Anchored Multiplicative Gating to prevent False Positive leakage.
2. Scale-Adaptive Task Alignment Recalibration (SATAR): Replaces hard-clipped confidence
   floors with a smooth NWD-IoU hybrid quality metric and scale-adaptive exponent
   recalibration using vectorized scatter_reduce operations.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class QACDetectV2(Detect):
    """Distributional Quality-Aware Calibrated Detection Head (QAC v2).

    Extracts statistical sharpness (top-k probabilities) from the 4-boundary DFL
    regression distributions to form a geometric certainty gate, modulating classification
    logits with semantic-veto protection against structured background false positives.
    """

    def __init__(
        self,
        nc: int = 1,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] | list[int] = (),
        topk_stat: int = 3,
        quality_gamma: float = 1.0,
        quality_min_conf: float = 0.25,
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.topk_stat = min(int(topk_stat), max(1, self.reg_max))
        # Kept for backward config compatibility with v1 YAML signatures
        self.quality_gamma = float(quality_gamma)
        self.quality_min_conf = float(quality_min_conf)

        # Each of the 4 box boundaries (l, t, r, b) provides `topk_stat` probability values
        stat_channels = 4 * self.topk_stat
        self.quality_gates = nn.ModuleList(
            [nn.Conv2d(stat_channels, 1, kernel_size=1, bias=True) for _ in range(self.nl)]
        )

        # Near-identity initialization: small weights and negative bias (-2.5 -> sigmoid ~0.076)
        for gate in self.quality_gates:
            nn.init.normal_(gate.weight, mean=0.0, std=0.01)
            nn.init.constant_(gate.bias, -2.5)

        # Per-level learnable calibration scale parameter (initialized to 0.5 for semantic-anchored modulation)
        self.calib_scale = nn.Parameter(torch.full((self.nl, 1, 1, 1), 0.50))

    def _extract_dist_stats(self, b: Tensor) -> Tensor:
        """Vectorized extraction of top-k DFL boundary probabilities without full-tensor softmax allocation."""
        bs, _, h, w = b.shape
        if self.reg_max <= 1:
            return b.sigmoid()

        # Reshape to (B, 4, reg_max, H, W) view without copying memory
        b_dist = b.view(bs, 4, self.reg_max, h, w)

        # LogSumExp over reg_max (dim=2) for exact normalization denominator
        lse = torch.logsumexp(b_dist, dim=2, keepdim=True)

        # Since exp(x) is strictly monotonic, top-k on logits equals top-k on probabilities
        topk_logits = b_dist.topk(self.topk_stat, dim=2, largest=True, sorted=False).values

        # Convert top-k logits to exact probabilities: exp(z_top - lse)
        topk_probs = (topk_logits - lse).exp()
        return topk_probs.reshape(bs, 4 * self.topk_stat, h, w)

    def forward_head(
        self,
        x: list[Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, Tensor]:
        """Concatenate and return calibrated bounding boxes and semantic-anchored class logits."""
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        box_list, cls_list = [], []

        for i in range(self.nl):
            b = box_head[i](x[i])
            c = cls_head[i](x[i])

            # 1. Extract 12-channel boundary distribution sharpness statistics
            dist_stat = self._extract_dist_stats(b)

            # 2. Compute spatial geometric quality gate Q_geo in (0, 1)
            q_geo = self.quality_gates[i](dist_stat).sigmoid()

            # 3. Semantic-Veto Modulation: c = c + scale * (Q_geo * sigmoid(c))
            # Prevents background false positives (where sigmoid(c) ~ 0) from being boosted by sharp edges
            semantic_consensus = c.sigmoid() * q_geo
            c = c + self.calib_scale[i] * semantic_consensus

            box_list.append(b.view(bs, 4 * self.reg_max, -1))
            cls_list.append(c.view(bs, self.nc, -1))

        boxes = torch.cat(box_list, dim=-1)
        scores = torch.cat(cls_list, dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x}

    def forward(self, x: list[Tensor]) -> Any:
        """Forward pass using Detect base pipeline with distributional quality-calibrated head."""
        return super().forward(x)

    @classmethod
    def from_detect(
        cls,
        head: Detect,
        topk_stat: int = 3,
        quality_gamma: float = 1.0,
        quality_min_conf: float = 0.25,
    ) -> QACDetectV2:
        """Construct QACDetectV2 from an existing Detect head, preserving pretrained weights."""
        channels = [
            int(b[0].conv.in_channels) if hasattr(b[0], "conv") else int(b[0].in_channels)
            for b in head.cv2
        ]
        reg = getattr(head, "reg_max", 16)
        e2e = getattr(head, "end2end", getattr(head, "_end2end", hasattr(head, "one2one_cv2")))
        new_head = cls(
            nc=head.nc,
            reg_max=reg,
            end2end=e2e,
            ch=channels,
            topk_stat=topk_stat,
            quality_gamma=quality_gamma,
            quality_min_conf=quality_min_conf,
        )
        new_head._end2end = bool(getattr(head, "_end2end", e2e))
        new_head.load_state_dict(head.state_dict(), strict=False)
        for attr in ("i", "f", "type", "np", "inplace", "stride", "anchors", "strides", "shape"):
            if hasattr(head, attr):
                val = getattr(head, attr)
                setattr(new_head, attr, val.clone() if isinstance(val, Tensor) else val)
        return new_head


class QualityAwareDetectionLossV2(v8DetectionLoss):
    """Detection Loss with Scale-Adaptive Task Alignment Recalibration (SATAR).

    Replaces the rigid TAL exponential penalty on tiny objects with a smooth
    Normalized Wasserstein Distance (NWD) + IoU hybrid quality metric and an
    instance-wise scale-adaptive exponent, fully vectorized via GPU scatter_reduce.
    """

    def __init__(
        self,
        model: nn.Module,
        tal_topk: int = 10,
        tal_topk2: int = 0,
        quality_gamma: float = 1.0,
        quality_min_conf: float = 0.25,
        nwd_constant: float = 12.8,
        nwd_weight: float = 0.5,
        beta_min: float = 1.5,
        beta_max: float = 6.0,
        area_ref: float = 576.0,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.quality_gamma = float(quality_gamma)
        self.quality_min_conf = float(quality_min_conf)
        self.nwd_constant = float(nwd_constant)
        self.nwd_weight = float(nwd_weight)
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.area_ref = float(area_ref)

    @staticmethod
    def _compute_hybrid_iou_nwd(
        pbox: Tensor,
        tbox: Tensor,
        nwd_c: float,
        nwd_weight: float,
    ) -> tuple[Tensor, Tensor]:
        """Vectorized computation of exact IoU, NWD, and ground-truth area on foreground boxes."""
        px1, py1, px2, py2 = pbox.unbind(dim=-1)
        tx1, ty1, tx2, ty2 = tbox.unbind(dim=-1)

        pw = (px2 - px1).clamp_min_(1e-4)
        ph = (py2 - py1).clamp_min_(1e-4)
        tw = (tx2 - tx1).clamp_min_(1e-4)
        th = (ty2 - ty1).clamp_min_(1e-4)

        area_p = pw * ph
        area_t = tw * th

        # 1. Exact IoU calculation
        inter_w = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp_min_(0.0)
        inter_h = (torch.minimum(py2, ty2) - torch.maximum(py1, ty1)).clamp_min_(0.0)
        inter = inter_w.mul_(inter_h)
        union = (area_p + area_t).sub_(inter).clamp_min_(1e-4)
        ious = inter.div_(union)

        # 2. Vectorized Normalized Wasserstein Distance (NWD) for 2D Gaussian bounding boxes
        # Center coordinates (cx, cy)
        pcx = (px1 + px2).mul_(0.5)
        pcy = (py1 + py2).mul_(0.5)
        tcx = (tx1 + tx2).mul_(0.5)
        tcy = (ty1 + ty2).mul_(0.5)

        # Wasserstein-2 squared distance: (dcx^2 + dcy^2) + 0.25 * (dw^2 + dh^2)
        dcx = pcx.sub_(tcx)
        dcy = pcy.sub_(tcy)
        dw = pw.sub_(tw)
        dh = ph.sub_(th)

        w2_sq = dcx.square_().add_(dcy.square_())
        wh_sq = dw.square_().add_(dh.square_()).mul_(0.25)
        w2_dist = w2_sq.add_(wh_sq).clamp_min_(1e-8).sqrt_()

        # NWD = exp(-sqrt(W2^2) / C)
        nwd = w2_dist.div_(-nwd_c).exp_()

        # 3. Blend IoU and NWD into a smooth scale-invariant quality metric
        q_hybrid = ious.mul(1.0 - nwd_weight).add_(nwd.mul_(nwd_weight))
        return q_hybrid, area_t

    def get_assigned_targets_and_loss(
        self, preds: dict[str, Tensor], batch: dict[str, Any]
    ) -> tuple:
        """Calculate loss with Scale-Adaptive Task Alignment Recalibration (SATAR) for foreground anchors."""
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )

        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # SATAR: Smooth Scale-Adaptive Target Recalibration on foreground anchors
        if fg_mask.any():
            pbox = (pred_bboxes.detach() * stride_tensor)[fg_mask]
            tbox = target_bboxes[fg_mask]

            # 1. Compute NWD-IoU hybrid quality and ground-truth pixel area
            q_hybrid, area_t = self._compute_hybrid_iou_nwd(
                pbox, tbox, self.nwd_constant, self.nwd_weight
            )

            # 2. Compute scale-adaptive exponent beta(A) in [beta_min, beta_max]
            # Small objects (< 32x32 px) receive beta ~ 1.5 - 2.2 instead of harsh 6.0
            scale_factor = area_t.div(self.area_ref).sqrt_().clamp_(0.0, 1.0)
            beta_adaptive = scale_factor.mul_(self.beta_max - self.beta_min).add_(self.beta_min)

            # 3. Extract foreground classification confidence on assigned class
            fg_scores = target_scores[fg_mask]
            _, fg_cls_idx = fg_scores.max(dim=-1)
            one_hot_mask = fg_scores.gt(0.0)

            # Extract detached predicted probabilities for the assigned ground-truth classes
            fg_pred_prob = (
                pred_scores.detach()[fg_mask]
                .sigmoid()
                .gather(dim=-1, index=fg_cls_idx.unsqueeze(-1))
                .squeeze(-1)
            )

            # 4. Compute scale-adaptive alignment metric: t_i = s^alpha * q_hybrid^beta(A)
            alpha = getattr(self.assigner, "alpha", 0.5)
            align_metric = fg_pred_prob.pow(alpha).mul_(q_hybrid.pow(beta_adaptive))

            # 5. Instance-wise Peak Normalization via vectorized GPU scatter_reduce
            # Unique global GT ID across batch: batch_idx * max_num_obj + gt_idx
            batch_ids = (
                torch.arange(batch_size, device=self.device)
                .unsqueeze(1)
                .expand_as(fg_mask)[fg_mask]
            )
            max_gt_per_img = max(1, gt_bboxes.shape[1])
            global_gt_id = batch_ids * max_gt_per_img + target_gt_idx[fg_mask]

            num_total_gt = batch_size * max_gt_per_img
            max_align_per_gt = torch.zeros(
                num_total_gt, dtype=align_metric.dtype, device=self.device
            )
            max_align_per_gt.scatter_reduce_(
                0, global_gt_id, align_metric, reduce="amax", include_self=False
            )
            inst_peak = max_align_per_gt.gather(0, global_gt_id).clamp_min_(1e-6)

            # Normalized target score scaled by geometric quality: (t_i / max_t) * q_hybrid^gamma
            recalibrated_conf = (
                align_metric.div_(inst_peak).mul_(q_hybrid.pow(self.quality_gamma)).unsqueeze_(-1)
            )

            # Smoothly blend with TAL target (guarantees non-suppression without hard-clipping artifacts)
            target_scores[fg_mask] = torch.maximum(
                fg_scores, recalibrated_conf.type_as(fg_scores)
            ).mul_(one_hot_mask)

        target_scores_sum = target_scores.sum().clamp_(min=1.0)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )

        h_box = self.hyp.box if hasattr(self.hyp, "box") else self.hyp.get("box", 7.5)
        h_cls = self.hyp.cls if hasattr(self.hyp, "cls") else self.hyp.get("cls", 0.5)
        h_dfl = self.hyp.dfl if hasattr(self.hyp, "dfl") else self.hyp.get("dfl", 1.5)
        loss[0] *= h_box
        loss[1] *= h_cls
        loss[2] *= h_dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )


class QualityAwareE2ELossV2(E2ELoss):
    """End-to-end wrapper for Quality-Aware Detection Loss V2 (SATAR)."""

    def __init__(
        self,
        model: nn.Module,
        quality_gamma: float = 1.0,
        quality_min_conf: float = 0.25,
        nwd_constant: float = 12.8,
        nwd_weight: float = 0.5,
        beta_min: float = 1.5,
        beta_max: float = 6.0,
        area_ref: float = 576.0,
    ) -> None:
        super().__init__(model, loss_fn=v8DetectionLoss)
        self.one2many = QualityAwareDetectionLossV2(
            model,
            tal_topk=10,
            quality_gamma=quality_gamma,
            quality_min_conf=quality_min_conf,
            nwd_constant=nwd_constant,
            nwd_weight=nwd_weight,
            beta_min=beta_min,
            beta_max=beta_max,
            area_ref=area_ref,
        )
        self.one2one = QualityAwareDetectionLossV2(
            model,
            tal_topk=7,
            tal_topk2=1,
            quality_gamma=quality_gamma,
            quality_min_conf=quality_min_conf,
            nwd_constant=nwd_constant,
            nwd_weight=nwd_weight,
            beta_min=beta_min,
            beta_max=beta_max,
            area_ref=area_ref,
        )


# Backward-compatible aliases
QACDetect = QACDetectV2
QualityAwareDetectionLoss = QualityAwareDetectionLossV2
QualityAwareE2ELoss = QualityAwareE2ELossV2

__all__ = [
    "QACDetect",
    "QACDetectV2",
    "QualityAwareDetectionLoss",
    "QualityAwareDetectionLossV2",
    "QualityAwareE2ELoss",
    "QualityAwareE2ELossV2",
]
