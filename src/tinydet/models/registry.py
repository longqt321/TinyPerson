from pathlib import Path

import yaml
from ultralytics import YOLO

from tinydet.models.checkpoint import load_p2_relocated_weights
from tinydet.models.ultralytics import _replace_composite_head, register_ultralytics_modules


def build_model(config: dict):
    register_ultralytics_modules()
    model_path = Path(config["config"])
    if not model_path.is_file():
        raise ValueError(f"Model config does not exist: {model_path}")
    model_yaml = yaml.safe_load(model_path.read_text())
    has_p2_module = any(
        row[2] in {"MSR", "BTRM"}
        for row in model_yaml.get("backbone", []) + model_yaml.get("head", [])
    )
    has_composite = model_yaml.get("composite") in {"cglr", "cass", "sure"}
    model = YOLO(str(model_path))
    _replace_composite_head(model.model)
    pretrained = config.get("pretrained")
    if pretrained:
        weights = Path(pretrained)
        if not weights.is_file():
            raise ValueError(f"Pretrained checkpoint does not exist: {weights}")
        # YOLO.train rebuilds from YAML unless the wrapper holds a checkpoint.
        model.load(str(weights))
        if has_composite:
            model.model.yaml["composite"] = model_yaml["composite"]
            for key in ("composite_topk", "composite_hidden", "composite_aux_weight"):
                if key in model_yaml:
                    model.model.yaml[key] = model_yaml[key]
        _replace_composite_head(model.model)
        if has_p2_module or has_composite:
            from ultralytics.nn.tasks import load_checkpoint

            source, _ = load_checkpoint(str(weights))
            load_p2_relocated_weights(model.model, source)
    return model
