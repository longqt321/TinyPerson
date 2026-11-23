"""Explicit registration and hooks for custom TinyDet modules parsed from YAML."""

import torch

from tinydet.modules.gauss_split_detect import DetectGaussSplit, find_gauss_split_head


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
    from ultralytics.utils.loss import E2ELoss

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
        head = find_gauss_split_head(self)
        if head is not None:
            if not isinstance(criterion, E2ELoss):
                raise TypeError(
                    f"a model with DetectGaussSplit needs E2ELoss, got {type(criterion).__name__}; "
                    "the auxiliary loss would silently not train otherwise"
                )
            criterion._tinydet_head = head
        return criterion

    DetectionModel.init_criterion = init_criterion

    # The auxiliary loss is added at the criterion level (once, not per branch).
    # The extra "gs_loss" key flows into results.csv and W&B automatically: the
    # trainer derives `loss_names` from the criterion's loss dict on first batch.
    original_e2e_call = E2ELoss.__call__

    def e2e_call(self, preds, batch):
        total, items = original_e2e_call(self, preds, batch)
        head = getattr(self, "_tinydet_head", None)
        if head is None:
            return total, items
        raw = preds[1] if isinstance(preds, tuple) else preds
        aux = head.aux_loss(raw, batch)
        return total.sum() + aux, {**items, "gs_loss": aux.detach()}

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
        # `_setup_train` builds the train loader (tiled) before the val loader, so
        # clearing `args.tile` here leaves every later consumer — the validator's
        # `merge_tiles` branch included — in full-frame mode.
        self.args.tile = False
        return original_build_dataset(self, img_path, mode, batch)

    DetectionTrainer.build_dataset = build_dataset
    DetectionTrainer._tinydet_tile_hook = True


def register_ultralytics_modules() -> None:
    """Register TinyDet modules (GauSS, P2Refine, DetectGaussSplit) before Ultralytics parses a model YAML."""
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
    tasks.DetectGaussSplit = DetectGaussSplit

    # Patch parse_model to support multi-input layers:
    #   GauSS: [ [c2, p3, p4], 1, GauSS, [32] ]
    #   P2Refine:     [ [c2, p3], 1, P2Refine, [32] ]
    # and the Detect subclass (which needs the reg_max/end2end/ch detection args
    # plus its own extra args after them):
    #   DetectGaussSplit: [ [p2, p3, p4, p5], 1, DetectGaussSplit, [nc, "pair"] ]
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
        elif m is DetectGaussSplit:
            args = [args[0], reg_max, end2end, [ch[x] for x in f], *args[1:]]
            m.legacy = legacy
        else:
            c2 = ch[f]"""
        if target in src:
            new_src = src.replace(target, replacement, 1)
            glb = dict(tasks.__dict__)
            glb["GauSS"] = GauSS
            glb["P2Refine"] = P2Refine
            glb["DetectGaussSplit"] = DetectGaussSplit
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
        ])

    _install_hooks()
