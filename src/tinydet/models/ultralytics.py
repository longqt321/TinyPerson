"""Explicit registration and hooks for custom modules parsed from YAML."""


def _replace_composite_head(model) -> None:
    """Swap a stock Detect head for the composite head named by the model YAML."""
    from ultralytics.nn.modules.head import Detect

    from tinydet.modules.composite import COMPOSITE_MODES, CompositeDetect

    mode = getattr(model, "yaml", {}).get("composite")
    if mode not in COMPOSITE_MODES or getattr(model, "model", None) is None:
        return
    head = model.model[-1]
    if isinstance(head, CompositeDetect) or not isinstance(head, Detect):
        return
    model.model[-1] = CompositeDetect.from_detect(
        head,
        mode,
        topk=int(model.yaml.get("composite_topk", 300)),
        hidden=int(model.yaml.get("composite_hidden", 128)),
        aux_weight=float(model.yaml.get("composite_aux_weight", 0.5)),
    )


def _install_hooks() -> None:
    """Apply model YAML hooks once after Ultralytics constructs a detector."""
    from ultralytics.nn.tasks import DetectionModel

    if getattr(DetectionModel, "_tinydet_hooks", False):
        return
    original_init = DetectionModel.__init__
    original_criterion = DetectionModel.init_criterion

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _replace_composite_head(self)

    def init_criterion(self):
        criterion = original_criterion(self)
        from tinydet.modules.composite import CompositeCriterion, CompositeDetect

        beta = float(self.yaml.get("tal_beta", 6.0))
        losses = (criterion.one2many, criterion.one2one) if hasattr(criterion, "one2many") else (criterion,)
        for loss in losses:
            if hasattr(loss, "assigner"):
                loss.assigner.beta = beta
        if isinstance(self.model[-1], CompositeDetect):
            criterion = CompositeCriterion(self, criterion)
        return criterion

    DetectionModel.__init__ = init
    DetectionModel.init_criterion = init_criterion
    DetectionModel._tinydet_hooks = True


def register_ultralytics_modules() -> None:
    """Register TinyDet modules before Ultralytics parses a model YAML."""
    from ultralytics.nn import tasks

    from tinydet.modules.btrm import BTRM
    from tinydet.modules.composite import CompositeDetect
    from tinydet.modules.msr import MSR

    tasks.MSR = MSR
    tasks.BTRM = BTRM
    tasks.CompositeDetect = CompositeDetect
    _install_hooks()
