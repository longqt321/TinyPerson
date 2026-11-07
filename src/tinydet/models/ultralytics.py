"""Explicit registration and hooks for custom TinyDet modules parsed from YAML."""


def _install_hooks() -> None:
    """Apply model YAML hooks once after Ultralytics constructs a detector."""
    from ultralytics.nn.tasks import DetectionModel

    if getattr(DetectionModel, "_tinydet_hooks", False):
        return

    original_criterion = DetectionModel.init_criterion

    def init_criterion(self):
        criterion = original_criterion(self)
        beta = float(self.yaml.get("tal_beta", 6.0))
        losses = (criterion.one2many, criterion.one2one) if hasattr(criterion, "one2many") else (criterion,)
        for loss in losses:
            if hasattr(loss, "assigner"):
                loss.assigner.beta = beta
        return criterion

    DetectionModel.init_criterion = init_criterion
    DetectionModel._tinydet_hooks = True


def register_ultralytics_modules() -> None:
    """Register TinyDet modules (DCAF, P2Refine) before Ultralytics parses a model YAML."""
    import torch
    from ultralytics.nn import tasks

    from tinydet.modules.dcaf import (
        DCAF,
        BandBranch,
        DilatedContextBranch,
        LearnableGaussianScaleSpace,
        RefinementBranch,
    )
    from tinydet.modules.p2_refine import P2Refine

    tasks.DCAF = DCAF
    tasks.BandBranch = BandBranch
    tasks.DilatedContextBranch = DilatedContextBranch
    tasks.LearnableGaussianScaleSpace = LearnableGaussianScaleSpace
    tasks.RefinementBranch = RefinementBranch
    tasks.P2Refine = P2Refine

    # Patch parse_model to support multi-input layers:
    #   DCAF:     [ [c2, p3, p4], 1, DCAF, [32] ]
    #   P2Refine: [ [c2, p3], 1, P2Refine, [32] ]
    if not getattr(tasks, "_tinydet_dcaf_patched", False):
        import inspect

        src = inspect.getsource(tasks.parse_model)
        target = "        else:\n            c2 = ch[f]"
        replacement = """        elif m is DCAF:
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
            glb["DCAF"] = DCAF
            glb["P2Refine"] = P2Refine
            exec(new_src, glb)  # noqa: S102
            tasks.parse_model = glb["parse_model"]
            tasks._tinydet_dcaf_patched = True

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            DCAF,
            BandBranch,
            DilatedContextBranch,
            LearnableGaussianScaleSpace,
            RefinementBranch,
            P2Refine,
        ])

    _install_hooks()
