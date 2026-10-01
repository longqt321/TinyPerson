"""Per-model training adapter for Ultralytics horizontal-box detection.

No global monkeypatching. Plugin settings live in model YAML so checkpoint and
EMA models initialize the same criterion. Inference layers remain unchanged.
Tested against Ultralytics 8.4.164; fail explicitly for incompatible loss APIs.
"""

import inspect
import math
from copy import deepcopy

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss

from tinydet.modules.nwd import NWDBboxLoss
from tinydet.modules.rfla import RFLAAssigner


def validate_modules(options):
    options = {} if options is None else deepcopy(options)
    if not isinstance(options, dict) or set(options) - {"rfla", "nwd"}:
        raise ValueError("modules accepts only rfla and nwd mappings")
    allowed = {
        "rfla": {"enabled", "topk", "extra_topk", "shrink", "chunk_size", "receptive_fields",
                 "erf_fraction", "sigma_stride_ratio", "inside_only"},
        "nwd": {"enabled", "constant", "weight"},
    }
    for name, config in options.items():
        if not isinstance(config, dict) or set(config) - allowed[name]:
            raise ValueError(f"Invalid {name} options; allowed keys: {sorted(allowed[name])}")
        if type(config.get("enabled", False)) is not bool:
            raise ValueError(f"{name}.enabled must be a boolean")
    nwd = options.get("nwd", {})
    if not math.isfinite(nwd.get("constant", 12.8)) or nwd.get("constant", 12.8) <= 0:
        raise ValueError("nwd.constant must be positive and finite")
    if not 0 <= nwd.get("weight", 1.0) <= 1:
        raise ValueError("nwd.weight must lie in [0, 1]")
    rfla = options.get("rfla", {})
    for key, default in (("erf_fraction", 0.5), ("sigma_stride_ratio", 4.0)):
        if not math.isfinite(rfla.get(key, default)) or rfla.get(key, default) <= 0:
            raise ValueError(f"rfla.{key} must be positive and finite")
    fields = rfla.get("receptive_fields")
    if fields is not None and (not isinstance(fields, list) or not fields or
                              any(not math.isfinite(x) or x <= 0 for x in fields)):
        raise ValueError("rfla.receptive_fields must be positive pixel diameters, one per Detect level")
    if type(rfla.get("inside_only", False)) is not bool:
        raise ValueError("rfla.inside_only must be a boolean")
    # Validate assignment options before allocating GPU resources.
    RFLAAssigner(1, **{key: rfla[key] for key in ("topk", "extra_topk", "shrink", "chunk_size") if key in rfla})
    return options


class ModuleDetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        if "stride" not in inspect.signature(BboxLoss.forward).parameters:
            raise RuntimeError("Unsupported Ultralytics BboxLoss API; tested version is 8.4.164")
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        options = validate_modules(model.yaml.get("tinydet_modules", {}))
        rfla, nwd = options.get("rfla", {}), options.get("nwd", {})
        self.rfla_options = rfla if rfla.get("enabled", False) else None
        if self.rfla_options is not None:
            if self.use_dfl and not rfla.get("inside_only", False):
                raise ValueError("RFLA outside-GT assignment requires signed box distances (YOLO26 reg_max=1). "
                                 "For DFL heads set rfla.inside_only=true for the constrained adaptation.")
            fields = rfla.get("receptive_fields")
            if fields is not None and len(fields) != len(self.stride):
                raise ValueError("rfla.receptive_fields must match the number of Detect levels")
            self.assigner = RFLAAssigner(
                self.nc, one_to_one=(tal_topk2 == 1 or tal_topk == 1),
                **{key: rfla[key] for key in ("topk", "extra_topk", "shrink", "chunk_size", "inside_only") if key in rfla},
            )
        if nwd.get("enabled", False):
            self.bbox_loss = NWDBboxLoss(self.bbox_loss, nwd.get("constant", 12.8), nwd.get("weight", 1.0))

    def get_assigned_targets_and_loss(self, preds, batch):
        if self.rfla_options is not None:
            options = self.rfla_options
            fields = options.get("receptive_fields")
            sigmas = (torch.tensor(fields, device=self.device) * options.get("erf_fraction", 0.5) / 2
                      if fields is not None else self.stride * options.get("sigma_stride_ratio", 4.0))
            self.assigner.sigma = torch.cat([
                torch.ones(feat.shape[-2] * feat.shape[-1], device=feat.device) * sigma
                for feat, sigma in zip(preds["feats"], sigmas)
            ])
        return super().get_assigned_targets_and_loss(preds, batch)


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
