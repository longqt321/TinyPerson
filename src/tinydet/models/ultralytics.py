"""Explicit registration and hooks for custom TinyDet modules parsed from YAML."""

import torch
from ultralytics.utils.tal import dist2bbox

from tinydet.modules.lgt_detect import (
    LGTDetect,
    SpatialShiftStencil,
    find_lgt_head,
    nwd_similarity_xyxy,
    scale_invariant_wasserstein_xyxy,
)


def make_tiny_floor_forward(orig_fn, floor_val):
    """Wrap a TAL assigner forward so tiny ground truths keep a score floor on their matched class only."""

    def patched_forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kwargs):
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = orig_fn(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kwargs
        )
        area = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp(min=0.0) * (
            target_bboxes[..., 3] - target_bboxes[..., 1]
        ).clamp(min=0.0)
        is_tiny_pos = ((area < 64.0) & fg_mask).unsqueeze(-1) & (target_scores > 0)
        target_scores = torch.where(
            is_tiny_pos,
            target_scores.clamp(min=floor_val),
            target_scores,
        )
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

    return patched_forward


def _wasserstein_sim_lowmem(gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
    """Memory-minimal scale-invariant Wasserstein similarity ``1 / (1 + D)`` for TAL."""
    g1 = gt_bboxes[..., 0].float()
    g2 = gt_bboxes[..., 2].float()
    p1 = pd_bboxes[..., 0].float()
    p2 = pd_bboxes[..., 2].float()
    sig_g = (g2 - g1).mul_(0.5).clamp_(min=1e-3)
    sig_p = (p2 - p1).mul_(0.5).clamp_(min=1e-3)
    dist = ((g2 + g1) - (p2 + p1)).mul_(0.5).square_()
    dist.add_((sig_g - sig_p).square_())
    dist.div_((sig_g.square() + sig_p.square()).add_(1e-6))

    g1 = gt_bboxes[..., 1].float()
    g2 = gt_bboxes[..., 3].float()
    p1 = pd_bboxes[..., 1].float()
    p2 = pd_bboxes[..., 3].float()
    sig_g = (g2 - g1).mul_(0.5).clamp_(min=1e-3)
    sig_p = (p2 - p1).mul_(0.5).clamp_(min=1e-3)
    dy = ((g2 + g1) - (p2 + p1)).mul_(0.5).square_()
    dy.add_((sig_g - sig_p).square_())
    dy.div_((sig_g.square() + sig_p.square()).add_(1e-6))
    dist.add_(dy)
    del dy, g1, g2, p1, p2, sig_g, sig_p
    return dist.add_(1.0).reciprocal_()


def _nwd_sim_lowmem(gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor, constant_c: float = 12.8) -> torch.Tensor:
    """Memory-minimal NWD similarity ``exp(-sqrt(W_2^2) / C)`` (Wang et al., 2021) in pixel coordinates."""
    inv_c = -1.0 / max(float(constant_c), 1e-3)
    g1 = gt_bboxes[..., 0].float()
    g2 = gt_bboxes[..., 2].float()
    p1 = pd_bboxes[..., 0].float()
    p2 = pd_bboxes[..., 2].float()
    sig_g = (g2 - g1).mul_(0.5).clamp_(min=1e-3)
    sig_p = (p2 - p1).mul_(0.5).clamp_(min=1e-3)
    w2_sq = ((g2 + g1) - (p2 + p1)).mul_(0.5).square_()
    w2_sq.add_((sig_g - sig_p).square_())

    g1 = gt_bboxes[..., 1].float()
    g2 = gt_bboxes[..., 3].float()
    p1 = pd_bboxes[..., 1].float()
    p2 = pd_bboxes[..., 3].float()
    sig_g = (g2 - g1).mul_(0.5).clamp_(min=1e-3)
    sig_p = (p2 - p1).mul_(0.5).clamp_(min=1e-3)
    w2_sq.add_(((g2 + g1) - (p2 + p1)).mul_(0.5).square_())
    w2_sq.add_((sig_g - sig_p).square_())
    del g1, g2, p1, p2, sig_g, sig_p
    return w2_sq.add_(1e-6).sqrt_().mul_(inv_c).exp_()


def _logistic_scale_gate(
    gt_bboxes: torch.Tensor,
    max_weight: float,
    tau_px: float = 20.0,
    beta_px: float = 3.5,
) -> torch.Tensor:
    """Logistic scale gate routing Wasserstein treatment to tiny GTs while preserving pure CIoU for >=28px GTs."""
    w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=0.0)
    h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=0.0)
    size_px = (w * h).sqrt()
    return torch.sigmoid((float(tau_px) - size_px) / max(float(beta_px), 0.5)).mul_(float(max_weight))


def make_wasserstein_iou_calculation(
    orig_iou_fn,
    weight: float,
    ref_scale_px: float = 32.0,
    tau_px: float | None = None,
    beta_px: float = 3.5,
):
    """Blend CIoU with scale-invariant Wasserstein similarity smoothly for tiny/small GTs."""
    ref_area = max(ref_scale_px * ref_scale_px, 1.0)

    def patched_iou_calculation(gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        ciou = orig_iou_fn(gt_bboxes, pd_bboxes)
        if gt_bboxes.numel() == 0:
            return ciou
        if tau_px is not None:
            alpha = _logistic_scale_gate(gt_bboxes, weight, tau_px=tau_px, beta_px=beta_px).to(dtype=ciou.dtype)
        else:
            gt_area = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_(min=0.0)
            gt_area.mul_((gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_(min=0.0))
            alpha = gt_area.div_(ref_area).add_(1.0).reciprocal_().mul_(float(weight)).to(dtype=ciou.dtype)
        w_sim = _wasserstein_sim_lowmem(gt_bboxes, pd_bboxes).to(dtype=ciou.dtype)
        return ciou.add_(w_sim.sub_(ciou).mul_(alpha))

    return patched_iou_calculation


def _pairwise_iou_aligned(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Elementwise IoU in ``[0, 1]`` for aligned ``(K, 4)`` pixel ``xyxy`` boxes."""
    lt = torch.maximum(box1[:, :2], box2[:, :2])
    rb = torch.minimum(box1[:, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0.0) * (box1[:, 3] - box1[:, 1]).clamp(min=0.0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0.0) * (box2[:, 3] - box2[:, 1]).clamp(min=0.0)
    return inter / (area1 + area2 - inter + eps)


def make_decoupled_tal_forward(
    orig_fn,
    weight: float,
    tau_px: float = 20.0,
    beta_px: float = 3.5,
    tiny_calib_floor: float = 0.08,
):
    """Decouple Wasserstein candidate selection from BCE soft-label calibration (DW-TAL).

    ``assigner.iou_calculation`` uses hybrid Wasserstein similarity so sub-stride tiny GTs
    do not suffer positive-sample starvation when ``IoU == 0``. Immediately afterward,
    ``target_scores`` on selected positive anchors are recalibrated against true ``IoU``
    (with a small ``tiny_calib_floor * r_gate`` floor only when ``IoU ~ 0``) so poorly
    localized or near-background anchors never receive inflated classification targets.
    """

    def decoupled_forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kwargs):
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = orig_fn(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kwargs
        )
        fg_idx = fg_mask.nonzero(as_tuple=True)
        if fg_idx[0].numel() == 0:
            return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

        pd_fg = pd_bboxes[fg_idx].detach().float()
        gt_fg = target_bboxes[fg_idx].detach().float()
        true_iou = _pairwise_iou_aligned(pd_fg, gt_fg)
        r_gate = _logistic_scale_gate(gt_fg, 1.0, tau_px=tau_px, beta_px=beta_px)
        calib_cap = torch.maximum(true_iou, float(tiny_calib_floor) * r_gate).unsqueeze(-1).to(target_scores.dtype)
        w_sim = _wasserstein_sim_lowmem(gt_fg, pd_fg)
        alpha = r_gate * float(weight)
        blended = ((1.0 - alpha) * true_iou + alpha * w_sim).clamp(min=1e-4).unsqueeze(-1).to(target_scores.dtype)
        ratio = (calib_cap / blended).clamp(max=1.15)
        target_scores[fg_idx] = torch.minimum(target_scores[fg_idx] * ratio, calib_cap)
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

    return decoupled_forward


def make_nwd_iou_calculation(orig_iou_fn, weight: float = 0.50, constant_c: float = 12.8):
    """Blend CIoU with fixed-constant NWD similarity (Wang et al., 2021) for controlled comparison."""
    w = float(weight)

    def patched_iou_calculation(gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        ciou = orig_iou_fn(gt_bboxes, pd_bboxes)
        if gt_bboxes.numel() == 0:
            return ciou
        nwd = _nwd_sim_lowmem(gt_bboxes, pd_bboxes, constant_c=constant_c).to(dtype=ciou.dtype)
        return ciou.mul_(1.0 - w).add_(nwd.mul_(w))

    return patched_iou_calculation


def make_wasserstein_bbox_forward(
    orig_bbox_forward,
    weight: float,
    ref_scale_px: float = 32.0,
    tau_px: float | None = None,
    beta_px: float = 3.5,
):
    """Augment ``BboxLoss.forward`` with bounded scale-invariant Wasserstein regression loss."""
    ref_area = max(ref_scale_px * ref_scale_px, 1.0)

    def patched_bbox_forward(
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    ):
        loss_iou, loss_dfl = orig_bbox_forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )
        fg_idx = fg_mask.nonzero(as_tuple=True)
        if fg_idx[0].numel() == 0:
            return loss_iou, loss_dfl
        fg_pred = pred_bboxes[fg_idx]
        fg_target = target_bboxes[fg_idx]
        fg_stride = stride[fg_idx[1]]
        gt_area_px = (
            (fg_target[:, 2] - fg_target[:, 0]).clamp(min=0.0)
            * (fg_target[:, 3] - fg_target[:, 1]).clamp(min=0.0)
            * fg_stride.squeeze(-1).square()
        )
        if tau_px is not None:
            size_px = gt_area_px.sqrt()
            scale_gate = (float(weight) * torch.sigmoid((float(tau_px) - size_px) / max(float(beta_px), 0.5))).unsqueeze(-1)
        else:
            scale_gate = (float(weight) / (1.0 + gt_area_px / ref_area)).unsqueeze(-1)
        pos_weight = target_scores[fg_idx].sum(-1, keepdim=True)
        dist = scale_invariant_wasserstein_xyxy(fg_pred, fg_target).unsqueeze(-1)
        w_loss = ((dist / (1.0 + dist)) * scale_gate.to(dist.dtype) * pos_weight.to(dist.dtype)).sum()
        loss_iou = loss_iou + (w_loss / target_scores_sum.to(dist.dtype)).to(loss_iou.dtype)
        return loss_iou, loss_dfl

    return patched_bbox_forward


def make_nwd_bbox_forward(orig_bbox_forward, weight: float = 0.50, constant_c: float = 12.8):
    """Hybrid CIoU + NWD regression loss ``(1 - w) * L_CIoU + w * (1 - NWD)`` in pixel coordinates."""
    w = float(weight)

    def patched_bbox_forward(
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
        imgsz,
        stride,
    ):
        loss_iou, loss_dfl = orig_bbox_forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )
        fg_idx = fg_mask.nonzero(as_tuple=True)
        if fg_idx[0].numel() == 0:
            return loss_iou, loss_dfl
        fg_stride = stride[fg_idx[1]]
        fg_pred_px = pred_bboxes[fg_idx] * fg_stride
        fg_target_px = target_bboxes[fg_idx] * fg_stride
        pos_weight = target_scores[fg_idx].sum(-1, keepdim=True)
        nwd_sim = nwd_similarity_xyxy(fg_pred_px, fg_target_px, constant_c=constant_c).unsqueeze(-1)
        nwd_loss = (((1.0 - nwd_sim) * pos_weight.to(nwd_sim.dtype)).sum() / target_scores_sum.to(nwd_sim.dtype)).to(
            loss_iou.dtype
        )
        return loss_iou * (1.0 - w) + nwd_loss * w, loss_dfl

    return patched_bbox_forward


def _patch_ltrb_aware_loss(loss_obj) -> None:
    """Allow ``v8DetectionLoss`` to consume Barycenter-refined ``preds['ltrb']`` when ``reg_max > 1``."""
    if getattr(loss_obj, "_ltrb_patched", False):
        return
    orig_loss = loss_obj.loss
    orig_decode = loss_obj.bbox_decode

    def bbox_decode(anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        active_ltrb = getattr(loss_obj, "_active_ltrb", None)
        if active_ltrb is not None:
            return dist2bbox(active_ltrb.permute(0, 2, 1).contiguous(), anchor_points, xywh=False)
        return orig_decode(anchor_points, pred_dist)

    def loss(preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        loss_obj._active_ltrb = preds.get("ltrb")
        try:
            return orig_loss(preds, batch)
        finally:
            loss_obj._active_ltrb = None

    loss_obj.bbox_decode = bbox_decode
    loss_obj.loss = loss
    loss_obj._ltrb_patched = True


def _install_hooks() -> None:
    """Apply model YAML hooks once after Ultralytics constructs a detector."""
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import E2ELoss

    if getattr(DetectionModel, "_tinydet_hooks", False):
        return

    original_criterion = DetectionModel.init_criterion

    def init_criterion(self):
        criterion = original_criterion(self)
        beta = float(self.yaml.get("tal_beta", 6.0))
        tiny_floor = float(self.yaml.get("tal_tiny_floor", 0.0))
        tal_wasserstein = float(self.yaml.get("tal_wasserstein", 0.0))
        tal_decoupled = bool(self.yaml.get("tal_decoupled", False))
        loss_wasserstein = float(self.yaml.get("loss_wasserstein", 0.0))
        ref_scale_px = float(self.yaml.get("wasserstein_ref_px", 32.0))
        raw_tau = self.yaml.get("wasserstein_tau_px")
        tau_px = float(raw_tau) if raw_tau is not None else None
        beta_px = float(self.yaml.get("wasserstein_beta_px", 3.5))
        tal_nwd = float(self.yaml.get("tal_nwd", 0.0))
        loss_nwd = float(self.yaml.get("loss_nwd", 0.0))
        nwd_constant = float(self.yaml.get("nwd_constant", 12.8))

        losses = (criterion.one2many, criterion.one2one) if hasattr(criterion, "one2many") else (criterion,)
        for loss in losses:
            if hasattr(loss, "assigner"):
                assigner = loss.assigner
                assigner.beta = beta
                if tiny_floor > 0.0 and not getattr(assigner, "_tiny_floor_patched", False):
                    assigner.forward = make_tiny_floor_forward(assigner.forward, tiny_floor)
                    assigner._tiny_floor_patched = True
                if tal_wasserstein > 0.0 and not getattr(assigner, "_wasserstein_patched", False):
                    assigner.iou_calculation = make_wasserstein_iou_calculation(
                        assigner.iou_calculation,
                        tal_wasserstein,
                        ref_scale_px=ref_scale_px,
                        tau_px=tau_px,
                        beta_px=beta_px,
                    )
                    if tal_decoupled:
                        assigner.forward = make_decoupled_tal_forward(
                            assigner.forward,
                            tal_wasserstein,
                            tau_px=tau_px if tau_px is not None else 20.0,
                            beta_px=beta_px,
                        )
                    assigner._wasserstein_patched = True
                elif tal_nwd > 0.0 and not getattr(assigner, "_nwd_patched", False):
                    assigner.iou_calculation = make_nwd_iou_calculation(
                        assigner.iou_calculation, tal_nwd, constant_c=nwd_constant
                    )
                    assigner._nwd_patched = True
            if hasattr(loss, "bbox_loss"):
                bbox_loss = loss.bbox_loss
                if loss_wasserstein > 0.0 and not getattr(bbox_loss, "_wasserstein_patched", False):
                    bbox_loss.forward = make_wasserstein_bbox_forward(
                        bbox_loss.forward,
                        loss_wasserstein,
                        ref_scale_px=ref_scale_px,
                        tau_px=tau_px,
                        beta_px=beta_px,
                    )
                    bbox_loss._wasserstein_patched = True
                elif loss_nwd > 0.0 and not getattr(bbox_loss, "_nwd_patched", False):
                    bbox_loss.forward = make_nwd_bbox_forward(
                        bbox_loss.forward, loss_nwd, constant_c=nwd_constant
                    )
                    bbox_loss._nwd_patched = True

        head = find_lgt_head(self)
        if head is not None:
            if not isinstance(criterion, E2ELoss):
                raise TypeError(
                    f"a model with {type(head).__name__} needs E2ELoss, got {type(criterion).__name__}; "
                    "set end2end: True in the model YAML"
                )
            _patch_ltrb_aware_loss(criterion.one2one)
            criterion._tinydet_head = head
        return criterion

    DetectionModel.init_criterion = init_criterion

    original_e2e_call = E2ELoss.__call__

    def e2e_call(self, preds, batch):
        total, items = original_e2e_call(self, preds, batch)
        head = getattr(self, "_tinydet_head", None)
        if head is None or not getattr(head, "training", True):
            return total.sum(), items
        o2m_cur = float(getattr(self, "o2m", 0.8))
        o2m_init = max(float(getattr(self, "o2m_copy", 0.8)), 1e-6)
        head._distill_decay = max(0.05, min(1.0, o2m_cur / o2m_init))
        raw = preds[1] if isinstance(preds, tuple) else preds
        aux = head.aux_loss(raw, batch)
        bsz = max(raw["one2one"]["boxes"].shape[0], 1)
        return total.sum() + aux, {**items, "lgt_loss": aux.detach() / bsz}

    E2ELoss.__call__ = e2e_call

    _install_tile_train_only_hook()
    DetectionModel._tinydet_hooks = True


def _install_tile_train_only_hook() -> None:
    """Apply ``args.tile`` to the train loader only; validation stays full-frame."""
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    if getattr(DetectionTrainer, "_tinydet_tile_hook", False):
        return
    original_build_dataset = DetectionTrainer.build_dataset

    def build_dataset(self, img_path, mode="train", batch=None):
        tile = getattr(self.args, "tile", False)
        if mode == "train" or not tile:
            return original_build_dataset(self, img_path, mode, batch)
        self.args.tile = False
        try:
            return original_build_dataset(self, img_path, mode, batch)
        finally:
            self._tiled_train = tile

    DetectionTrainer.build_dataset = build_dataset
    DetectionTrainer._tinydet_tile_hook = True


def register_ultralytics_modules() -> None:
    """Register LGTDetect before Ultralytics YAML parsing and checkpoint loading."""
    from ultralytics.nn import tasks

    tasks.LGTDetect = LGTDetect

    if not getattr(tasks, "_tinydet_lgt_patched", False):
        import inspect

        src = inspect.getsource(tasks.parse_model)
        target = "        else:\n            c2 = ch[f]"
        replacement = """        elif m is LGTDetect:
            args = [args[0], reg_max, end2end, [ch[x] for x in f], *args[1:]]
            m.legacy = legacy
        else:
            c2 = ch[f]"""
        if target not in src:
            raise RuntimeError(
                "Ultralytics tasks.parse_model signature changed; expected branch 'else: c2 = ch[f]' not found."
            )
        new_src = src.replace(target, replacement, 1)
        glb = dict(tasks.__dict__)
        glb["LGTDetect"] = LGTDetect
        exec(new_src, glb)  # noqa: S102
        tasks.parse_model = glb["parse_model"]
        tasks._tinydet_lgt_patched = True

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            LGTDetect,
            SpatialShiftStencil,
        ])

    _install_hooks()
