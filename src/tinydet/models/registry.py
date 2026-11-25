from pathlib import Path

from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from tinydet.models.ultralytics import register_ultralytics_modules


def build_model(config: dict):
    register_ultralytics_modules()
    model_path = Path(config["config"])
    if not model_path.is_file():
        raise ValueError(f"Model config does not exist: {model_path}")

    model = YOLO(str(model_path))
    if isinstance(getattr(model.model, "args", None), dict):
        model.model.args = get_cfg(overrides=model.model.args)
    if getattr(model.model, "yaml", {}).get("e2e_eval", False):
        model.model.model[-1].end2end = True

    pretrained = config.get("pretrained")
    if pretrained:
        weights = Path(pretrained)
        if not weights.is_file():
            raise ValueError(f"Pretrained checkpoint does not exist: {weights}")
        model.load(str(weights))

    return model
