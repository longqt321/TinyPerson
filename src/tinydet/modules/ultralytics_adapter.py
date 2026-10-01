"""Per-model training adapter for Ultralytics horizontal-box detection.

No global monkeypatching. Plugin settings live in model YAML so checkpoint and
EMA models initialize the same criterion. Inference layers remain unchanged.
Tested against Ultralytics 8.4.164; fail explicitly for incompatible loss APIs.
"""

import inspect
import math
from copy import deepcopy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner

from tinydet.modules.nwd import NWDBboxLoss, nwd_similarity


class NWDTaskAlignedAssigner(TaskAlignedAssigner):
    """Use pixel-space NWD wherever TAL uses its CIoU overlap."""

    def __init__(self, *args, constant=12.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        return nwd_similarity(gt_bboxes, pd_bboxes, self.constant)


def validate_modules(options):
    options = {} if options is None else deepcopy(options)
    if not isinstance(options, dict) or set(options) - {"nwd_tal", "nwd"}:
        raise ValueError("modules accepts only nwd_tal and nwd mappings")
    allowed = {"nwd_tal": {"enabled", "constant"}, "nwd": {"enabled", "constant", "weight"}}
    for name, config in options.items():
        if not isinstance(config, dict) or set(config) - allowed[name]:
            raise ValueError(f"Invalid {name} options; allowed keys: {sorted(allowed[name])}")
        if type(config.get("enabled", False)) is not bool:
            raise ValueError(f"{name}.enabled must be a boolean")
        constant = config.get("constant", 12.8)
        if not isinstance(constant, (int, float)) or not math.isfinite(constant) or constant <= 0:
            raise ValueError(f"{name}.constant must be positive and finite")
    nwd = options.get("nwd", {})
    if not 0 <= nwd.get("weight", 1.0) <= 1:
        raise ValueError("nwd.weight must lie in [0, 1]")
    return options


class ModuleDetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        if "stride" not in inspect.signature(BboxLoss.forward).parameters:
            raise RuntimeError("Unsupported Ultralytics BboxLoss API; tested version is 8.4.164")
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        options = validate_modules(model.yaml.get("tinydet_modules", {}))
        tal, nwd = options.get("nwd_tal", {}), options.get("nwd", {})
        if tal.get("enabled", False):
            original = self.assigner
            self.assigner = NWDTaskAlignedAssigner(
                topk=original.topk, topk2=original.topk2, num_classes=original.num_classes,
                alpha=original.alpha, beta=original.beta, stride=original.stride,
                eps=original.eps, constant=tal.get("constant", 12.8),
            )
        if nwd.get("enabled", False):
            self.bbox_loss = NWDBboxLoss(self.bbox_loss, nwd.get("constant", 12.8), nwd.get("weight", 1.0))


class ModuleDetectionModel(DetectionModel):
    def init_criterion(self):
        options = self.yaml.get("tinydet_modules", {})
        if not any(value.get("enabled", False) for value in options.values()):
            return super().init_criterion()
        if getattr(self.model[-1], "one2one_cv2", None) is not None:
            return E2ELoss(self, loss_fn=ModuleDetectionLoss)
        return ModuleDetectionLoss(self)


class ModuleDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, modules=None, **kwargs):
        self.modules_config = validate_modules(modules) if modules is not None else None
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        from ultralytics.nn.tasks import yaml_model_load

        config = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        if self.modules_config is not None:
            config["tinydet_modules"] = self.modules_config
        model = self.set_model_names_for_load(ModuleDetectionModel(
            config, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1,
        ))
        if weights is not None:
            model.load(weights)
        return model
