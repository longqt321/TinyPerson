"""Explicit registration and hooks for custom TinyDet modules parsed from YAML."""

import torch


def make_tiny_floor_forward(orig_fn, floor_val):
    """Wrap a TAL assigner forward so tiny ground truths keep a score floor.

    ``orig_fn`` returns ``(target_labels, target_bboxes, target_scores, fg_mask,
    target_gt_idx)`` where ``target_bboxes`` is ``(b, h*w, 4)`` in pixel units
    and ``fg_mask`` is ``(b, h*w)``. Anchors whose assigned box has a
    sqrt-area below 8 px get ``target_scores`` clamped to ``floor_val`` so they
    are not drowned out by the background.
    """

    def patched_forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kwargs):
        # **kwargs keeps this wrapper compatible with assigner variants that take
        # extra arguments (e.g. anc_strides) instead of raising TypeError.
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = orig_fn(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **kwargs
        )
        if fg_mask.any():
            bw = (target_bboxes[..., 2] - target_bboxes[..., 0]).clamp(min=0)
            bh = (target_bboxes[..., 3] - target_bboxes[..., 1]).clamp(min=0)
            sqrt_area = torch.sqrt(bw * bh + 1e-6)  # (b, h*w)
            # fg_mask is (b, h*w); broadcast it against sqrt_area first, then
            # lift to the class dimension of target_scores.
            is_tiny = (sqrt_area < 8.0) & fg_mask
            target_scores = torch.where(
                is_tiny.unsqueeze(-1),
                target_scores.clamp(min=floor_val),
                target_scores,
            )
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

    return patched_forward


def _install_hooks() -> None:
    """Apply model YAML hooks once after Ultralytics constructs a detector."""
    from ultralytics.nn.tasks import DetectionModel

    if getattr(DetectionModel, "_tinydet_hooks", False):
        return

    original_criterion = DetectionModel.init_criterion

    def init_criterion(self):
        criterion = original_criterion(self)
        beta = float(self.yaml.get("tal_beta", 6.0))
        tiny_floor = float(self.yaml.get("tal_tiny_floor", 0.0))
        losses = (criterion.one2many, criterion.one2one) if hasattr(criterion, "one2many") else (criterion,)
        for loss in losses:
            if hasattr(loss, "assigner"):
                assigner = loss.assigner
                assigner.beta = beta
                if tiny_floor > 0.0 and not getattr(assigner, "_tiny_floor_patched", False):
                    assigner.forward = make_tiny_floor_forward(assigner.forward, tiny_floor)
                    assigner._tiny_floor_patched = True
        return criterion

    DetectionModel.init_criterion = init_criterion
    DetectionModel._tinydet_hooks = True


def register_ultralytics_modules() -> None:
    """Register TinyDet modules (GauSS, P2Refine) before Ultralytics parses a model YAML."""
    import torch
    from ultralytics.nn import tasks

    from tinydet.modules.gauss import (
        BandBranch,
        DilatedContextBranch,
        DySampleUp,
        GauSS,
        LearnableGaussianScaleSpace,
        RefinementBranch,
    )
    from tinydet.modules.p2_refine import P2Refine

    tasks.GauSS = GauSS
    tasks.BandBranch = BandBranch
    tasks.DilatedContextBranch = DilatedContextBranch
    tasks.LearnableGaussianScaleSpace = LearnableGaussianScaleSpace
    tasks.RefinementBranch = RefinementBranch
    tasks.P2Refine = P2Refine

    # Patch parse_model to support multi-input layers:
    #   GauSS: [ [c2, p3, p4], 1, GauSS, [32] ]
    #   P2Refine:     [ [c2, p3], 1, P2Refine, [32] ]
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
        else:
            c2 = ch[f]"""
        if target in src:
            new_src = src.replace(target, replacement, 1)
            glb = dict(tasks.__dict__)
            glb["GauSS"] = GauSS
            glb["P2Refine"] = P2Refine
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
        ])

    _install_hooks()
