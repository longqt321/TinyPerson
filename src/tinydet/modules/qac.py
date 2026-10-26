"""Quality-Aware Calibration (QAC) module for tiny person detection.

Addresses the Confidence-Localization Misalignment where high-IoU tiny targets
receive suppressed classification confidence scores.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class QACDetect(Detect):
    """Quality-Aware Calibrated Detection Head.

    Augments the standard Decoupled Detect head with a cross-branch calibration
    path that allows high-resolution boundary regression features to modulate and
    boost classification confidence for tiny persons.
    """

    def __init__(
        self,
        nc: int = 1,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] | list[int] = (),
        quality_gamma: float = 1.0,
        quality_min_conf: float = 0.25,
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.quality_gamma = float(quality_gamma)
        self.quality_min_conf = float(quality_min_conf)

        # Cross-branch quality projection: 1x1 conv mapping box features to 1-channel quality factor
        # Each scale level (e.g. P2, P3, P4, P5) has its own lightweight gate
        c_boxes = [
            int(b[-1].conv.out_channels) if hasattr(b[-1], "conv") else int(b[-1].out_channels)
            for b in self.cv2
        ]
        self.quality_gates = nn.ModuleList([nn.Conv2d(c_b, 1, 1, bias=True) for c_b in c_boxes])

        # Initialize gate weights small and bias negative (-2.5) so gate is initially ~0.07 (near-identity)
        for gate in self.quality_gates:
            nn.init.normal_(gate.weight, mean=0.0, std=0.01)
            nn.init.constant_(gate.bias, -2.5)

        # Per-level learnable calibration scale parameter (initialized to 0.2)
        self.calib_scale = nn.Parameter(torch.full((len(ch), 1, 1, 1), 0.20))

    def forward_head(
        self,
        x: list[Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
    ) -> dict[str, Tensor]:
        """Concatenate and return calibrated bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:
            return {}
        bs = x[0].shape[0]
        box_list, cls_list = [], []
        for i in range(self.nl):
            b = box_head[i](x[i])
            c = cls_head[i](x[i])
            # In-place GPU quality calibration:
            gate = self.quality_gates[i](b).sigmoid_()
            c = c.addcmul(gate, self.calib_scale[i], value=1.0)
            box_list.append(b.view(bs, 4 * self.reg_max, -1))
            cls_list.append(c.view(bs, self.nc, -1))
        boxes = torch.cat(box_list, dim=-1)
        scores = torch.cat(cls_list, dim=-1)
        return {"boxes": boxes, "scores": scores, "feats": x}

    def forward(self, x: list[Tensor]) -> Any:
        """Forward pass using Detect base pipeline with quality-calibrated head."""
        return super().forward(x)

    @classmethod
    def from_detect(
        cls,
        head: Detect,
        quality_gamma: float = 1.0,
        quality_min_conf: float = 0.25,
    ) -> QACDetect:
        """Construct QACDetect from an existing Detect head, copying all weights."""
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


class QualityAwareDetectionLoss(v8DetectionLoss):
    """Detection Loss with Quality-Aware Classification Target (IDEA-05).

    Recalibrates the assigned classification target scores using the actual geometric
    IoU overlap on foreground anchors, preventing high-quality tiny person detections
    from having their confidence suppressed.
    """

    def __init__(
        self,
        model: nn.Module,
        tal_topk: int = 10,
        tal_topk2: int = 0,
        quality_gamma: float = 1.0,
        quality_min_conf: float = 0.25,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.quality_gamma = float(quality_gamma)
        self.quality_min_conf = float(quality_min_conf)

    def get_assigned_targets_and_loss(
        self, preds: dict[str, Tensor], batch: dict[str, Any]
    ) -> tuple:
        """Calculate loss with quality-aware target calibration for foreground anchors."""
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

        # Quality-Aware Target Alignment (IDEA-05):
        # Prevent high-IoU tiny person boxes from having confidence crushed by TAL power metric
        if fg_mask.any():
            pbox = (pred_bboxes.detach() * stride_tensor)[fg_mask]
            tbox = target_bboxes[fg_mask]
            inter_x1 = torch.maximum(pbox[:, 0], tbox[:, 0])
            inter_y1 = torch.maximum(pbox[:, 1], tbox[:, 1])
            inter_x2 = torch.minimum(pbox[:, 2], tbox[:, 2])
            inter_y2 = torch.minimum(pbox[:, 3], tbox[:, 3])
            inter_w = (inter_x2 - inter_x1).clamp_min_(0.0)
            inter_h = (inter_y2 - inter_y1).clamp_min_(0.0)
            inter = inter_w * inter_h
            area_p = (pbox[:, 2] - pbox[:, 0]).clamp_min_(1e-4) * (
                pbox[:, 3] - pbox[:, 1]
            ).clamp_min_(1e-4)
            area_t = (tbox[:, 2] - tbox[:, 0]).clamp_min_(1e-4) * (
                tbox[:, 3] - tbox[:, 1]
            ).clamp_min_(1e-4)
            ious = inter / (area_p + area_t - inter).clamp_min_(1e-4)

            # Modulate target score by geometric IoU quality
            quality_target = ious.pow(self.quality_gamma).unsqueeze_(-1)
            # Guarantee baseline confidence reward for accurate localizations
            target_scores[fg_mask] = torch.maximum(
                target_scores[fg_mask] * quality_target,
                quality_target * self.quality_min_conf,
            )

        target_scores_sum = target_scores.sum().clamp_(min=1)

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


class QualityAwareE2ELoss(E2ELoss):
    """End-to-end wrapper for Quality-Aware Detection Loss."""

    def __init__(
        self,
        model: nn.Module,
        quality_gamma: float = 1.0,
        quality_min_conf: float = 0.25,
    ) -> None:
        super().__init__(model, loss_fn=v8DetectionLoss)
        # Replace sub-losses with QualityAwareDetectionLoss
        self.one2many = QualityAwareDetectionLoss(
            model,
            tal_topk=10,
            quality_gamma=quality_gamma,
            quality_min_conf=quality_min_conf,
        )
        self.one2one = QualityAwareDetectionLoss(
            model,
            tal_topk=7,
            tal_topk2=1,
            quality_gamma=quality_gamma,
            quality_min_conf=quality_min_conf,
        )
