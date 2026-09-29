from pathlib import Path

from ultralytics import YOLO


def build_model(config: dict):
    model_path = Path(config["config"])
    if not model_path.is_file():
        raise ValueError(f"Model config does not exist: {model_path}")
    model = YOLO(str(model_path))
    pretrained = config.get("pretrained")
    if pretrained:
        weights = Path(pretrained)
        if not weights.is_file():
            raise ValueError(f"Pretrained checkpoint does not exist: {weights}")
        model.load(str(weights))
    return model
