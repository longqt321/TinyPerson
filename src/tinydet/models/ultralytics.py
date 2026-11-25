"""Explicit registration and hooks for custom TinyDet modules parsed from YAML."""

import torch

from tinydet.modules.gauss_split_detect import DetectGaussSplit, find_gauss_split_head
from tinydet.modules.lgt_detect import (
    LGTDetect,
    find_lgt_head,
    scale_invariant_wasserstein_xyxy,
)


def make_tiny_floor_forward(orig_fn, floor_val):
    """Wrap a TAL assigner forward so tiny ground truths keep a score floor on their matched class only.

    ``orig_fn`` returns ``(target_labels, target_bboxes, target_scores, fg_mask,
    target_gt_idx)`` where ``target_bboxes`` is ``(b, h*w, 4)`` in pixel units
    and ``fg_mask`` is ``(b, h*w)``. Only positive class channels (``target_scores > 0``)
    are floored so false classes in multi-class training remain strictly zero.
    """

    def patched_forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kwargs):
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = orig_fn(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kwargs
        )
        if fg_mask.any():
            bw = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp(min=0)
            bh = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp(min=0)
            sqrt_area = torch.sqrt(bw * bh + 1e-6)
            is_tiny_pos = ((sqrt_area < 8.0) & fg_mask).unsqueeze(-1) & (target_scores > 0)
            target_scores = torch.where(
                is_tiny_pos,
                target_scores.clamp(min=floor_val),
                target_scores,
            )
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

    return patched_forward


def _wasserstein_sim_lowmem(gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
    """Memory-minimal scale-invariant Wasserstein similarity ``1 / (1 + D)`` for TAL.

    Computes x and y axes sequentially with in-place ops to cut peak VRAM ~3x inside
    ``TaskAlignedAssigner`` where tensors reach ``(B, max_gt, 136k, 4)``.
    """
    g1 = gt_bboxes[..., 0].float()
    g2 = gt_bboxes[..., 2].float()
    p1 = pd_bboxes[..., 0].float()
    p2 = pd_bboxes[..., 2].float()
    sig_g = (g2 - g1).mul_(0.5).clamp_(min=1e-3)
    sig_p = (p2 - p1).mul_(0.5).clamp_(min=1e-3)
    dist = (g2 + g1).sub_(p2 + p1).mul_(0.5).square_()
    dist.add_((sig_g - sig_p).square_())
    dist.div_(sig_g.square_().add_(sig_p.square_()).add_(1e-6))

    g1 = gt_bboxes[..., 1].float()
    g2 = gt_bboxes[..., 3].float()
    p1 = pd_bboxes[..., 1].float()
    p2 = pd_bboxes[..., 3].float()
    sig_g = (g2 - g1).mul_(0.5).clamp_(min=1e-3)
    sig_p = (p2 - p1).mul_(0.5).clamp_(min=1e-3)
    dy = (g2 + g1).sub_(p2 + p1).mul_(0.5).square_()
    dy.add_((sig_g - sig_p).square_())
    dy.div_(sig_g.square_().add_(sig_p.square_()).add_(1e-6))
    dist.add_(dy)
    del dy, g1, g2, p1, p2, sig_g, sig_p
    return dist.add_(1.0).reciprocal_()


def make_wasserstein_iou_calculation(orig_iou_fn, weight: float, ref_scale_px: float = 32.0):
    """Blend CIoU with scale-invariant Wasserstein similarity smoothly for tiny/small GTs."""
    ref_area = max(ref_scale_px * ref_scale_px, 1.0)

    def patched_iou_calculation(gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        ciou = orig_iou_fn(gt_bboxes, pd_bboxes)
        if gt_bboxes.numel() == 0:
            return ciou
        gt_area = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_(min=0.0)
        gt_area.mul_((gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_(min=0.0))
        alpha = gt_area.div_(ref_area).add_(1.0).reciprocal_().mul_(float(weight)).to(dtype=ciou.dtype)
        w_sim = _wasserstein_sim_lowmem(gt_bboxes, pd_bboxes).to(dtype=ciou.dtype)
        return ciou.add(w_sim.sub_(ciou).mul_(alpha))

    return patched_iou_calculation


def make_wasserstein_bbox_forward(orig_bbox_forward, weight: float, ref_scale_px: float = 32.0):
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
        scale_gate = (float(weight) / (1.0 + gt_area_px / ref_area)).unsqueeze(-1)
        pos_weight = target_scores[fg_idx].sum(-1, keepdim=True)
        dist = scale_invariant_wasserstein_xyxy(fg_pred, fg_target).unsqueeze(-1)
        w_loss = ((dist / (1.0 + dist)) * scale_gate.to(dist.dtype) * pos_weight.to(dist.dtype)).sum()
        loss_iou = loss_iou + (w_loss / target_scores_sum.to(dist.dtype)).to(loss_iou.dtype)
        return loss_iou, loss_dfl

    return patched_bbox_forward


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
        loss_wasserstein = float(self.yaml.get("loss_wasserstein", 0.0))
        ref_scale_px = float(self.yaml.get("wasserstein_ref_px", 32.0))
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
                        assigner.iou_calculation, tal_wasserstein, ref_scale_px=ref_scale_px
                    )
                    assigner._wasserstein_patched = True
            if loss_wasserstein > 0.0 and hasattr(loss, "bbox_loss"):
                bbox_loss = loss.bbox_loss
                if not getattr(bbox_loss, "_wasserstein_patched", False):
                    bbox_loss.forward = make_wasserstein_bbox_forward(
                        bbox_loss.forward, loss_wasserstein, ref_scale_px=ref_scale_px
                    )
                    bbox_loss._wasserstein_patched = True
        head = find_gauss_split_head(self) or find_lgt_head(self)
        if head is not None:
            if not isinstance(criterion, E2ELoss):
                raise TypeError(
                    f"a model with {type(head).__name__} needs E2ELoss, got {type(criterion).__name__}; "
                    "the auxiliary loss would silently not train otherwise"
                )
            criterion._tinydet_head = head
        return criterion

    DetectionModel.init_criterion = init_criterion

    original_e2e_call = E2ELoss.__call__

    def e2e_call(self, preds, batch):
        total, items = original_e2e_call(self, preds, batch)
        head = getattr(self, "_tinydet_head", None)
        if head is None:
            return total, items
        if not getattr(head, "training", True):
            return total, items
        raw = preds[1] if isinstance(preds, tuple) else preds
        aux = head.aux_loss(raw, batch)
        loss_key = "lgt_loss" if isinstance(head, LGTDetect) else "gs_loss"
        if isinstance(head, LGTDetect):
            bsz = max(raw["one2one"]["boxes"].shape[0], 1)
            return total.sum() + aux, {**items, loss_key: aux.detach() / bsz}
        return total.sum() + aux, {**items, loss_key: aux.detach()}

    E2ELoss.__call__ = e2e_call

    _install_predictor_hooks()
    DetectionModel._tinydet_hooks = True


def _install_predictor_hooks() -> None:
    """Route inference through the head when the model carries a DetectGaussSplit."""
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.torch_utils import unwrap_model

    original_construct = DetectionPredictor.construct_results

    def construct_results(self, preds, img, orig_imgs):
        head = find_gauss_split_head(getattr(self.model, "model", self.model))
        if head is not None:
            preds = head.add_second_boxes(preds)
        return original_construct(self, preds, img, orig_imgs)

    DetectionPredictor.construct_results = construct_results

    # The validator receives the model as a local in __call__, so the head has to
    # be stashed in init_metrics (called once per validation, right before the
    # prediction loop) for postprocess to use.
    original_init_metrics = DetectionValidator.init_metrics

    def init_metrics(self, model):
        self._tinydet_head = find_gauss_split_head(unwrap_model(model))
        return original_init_metrics(self, model)

    original_validator_postprocess = DetectionValidator.postprocess

    def validator_postprocess(self, preds):
        outputs = original_validator_postprocess(self, preds)
        head = getattr(self, "_tinydet_head", None)
        if head is None:
            return outputs
        rows = [
            torch.cat([o["bboxes"], o["conf"][:, None], o["cls"][:, None]], 1) for o in outputs
        ]
        return [
            {"bboxes": t[:, :4], "conf": t[:, 4], "cls": t[:, 5], "extra": t[:, 6:]}
            for t in head.add_second_boxes(rows)
        ]

    DetectionValidator.init_metrics = init_metrics
    DetectionValidator.postprocess = validator_postprocess

    _install_tile_train_only_hook()


def _install_tile_train_only_hook() -> None:
    """Apply `args.tile` to the train loader only; validation stays full-frame.

    Ultralytics passes `cfg.tile` to every `build_yolo_dataset` call, so enabling
    tiling also tiles validation. That breaks the per-epoch TinyPerson benchmark:
    `build_prediction_row` emits one row per *tile* keyed by the source image, and
    the benchmark refuses duplicate images. Training the head on 1:1 tiles while
    validating on full frames keeps both the intended distribution (tile space,
    same as `evaluate_tiled`) and the per-epoch size-stratified diagnostics.
    """
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
            # Keep validator in full-frame mode while preserving provenance on trainer._tiled_train
            self._tiled_train = tile

    DetectionTrainer.build_dataset = build_dataset
    DetectionTrainer._tinydet_tile_hook = True


def register_ultralytics_modules() -> None:
    """Register TinyDet modules (LGTDetect, GauSS, P2Refine, DetectGaussSplit) before YAML parsing."""
    from ultralytics.nn import tasks

    from tinydet.modules.gauss import (
        BandBranch,
        DilatedContextBranch,
        DySampleUp,
        GauSS,
        LearnableGaussianScaleSpace,
        RefinementBranch,
    )
    from tinydet.modules.lgt_detect import SpatialShiftStencil
    from tinydet.modules.p2_refine import P2Refine

    tasks.GauSS = GauSS
    tasks.BandBranch = BandBranch
    tasks.DilatedContextBranch = DilatedContextBranch
    tasks.LearnableGaussianScaleSpace = LearnableGaussianScaleSpace
    tasks.RefinementBranch = RefinementBranch
    tasks.P2Refine = P2Refine
    tasks.DetectGaussSplit = DetectGaussSplit
    tasks.LGTDetect = LGTDetect

    if not getattr(tasks, "_tinydet_gauss_patched", False):
        import inspect

        src = inspect.getsource(tasks.parse_model)
        target = "        else:\n            c2 = ch[f]"
        replacement = """        elif m is GauSS:
            c1 = [ch[x] for x in f]
            c2 = args[0] if len(args) > 0 else 32
            args = [c1, c2, *args[1:]]
        elif m is P2Refine:
            c1 = [ch[x] for x in f]
            c2 = args[0] if len(args) > 0 else 32
            args = [c1, c2, *args[1:]]
        elif m in {DetectGaussSplit, LGTDetect}:
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
        glb["GauSS"] = GauSS
        glb["P2Refine"] = P2Refine
        glb["DetectGaussSplit"] = DetectGaussSplit
        glb["LGTDetect"] = LGTDetect
        exec(new_src, glb)  # noqa: S102
        tasks.parse_model = glb["parse_model"]
        tasks._tinydet_gauss_patched = True

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            GauSS,
            BandBranch,
            DilatedContextBranch,
            DySampleUp,
            LearnableGaussianScaleSpace,
            RefinementBranch,
            P2Refine,
            DetectGaussSplit,
            LGTDetect,
            SpatialShiftStencil,
        ])

    _install_hooks()
