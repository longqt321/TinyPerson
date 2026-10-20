from pathlib import Path

import yaml
from ultralytics import YOLO

from tinydet.modules.msr import load_p2_msr_weights


def build_model(config: dict):
    model_path = Path(config["config"])
    if not model_path.is_file():
        raise ValueError(f"Model config does not exist: {model_path}")
    model_yaml = yaml.safe_load(model_path.read_text())
    has_p2_msr = any(
        row[2] == "MSR" for row in model_yaml.get("backbone", []) + model_yaml.get("head", [])
    )
    model = YOLO(str(model_path))
    pretrained = config.get("pretrained")
    if pretrained:
        weights = Path(pretrained)
        if not weights.is_file():
            raise ValueError(f"Pretrained checkpoint does not exist: {weights}")
        if has_p2_msr:
            from ultralytics.nn.tasks import load_checkpoint

            source, _ = load_checkpoint(str(weights))
            load_p2_msr_weights(model.model, source)
        else:
            model.load(str(weights))
    return model
