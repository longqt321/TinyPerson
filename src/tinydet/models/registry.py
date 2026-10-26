from pathlib import Path

import yaml
from ultralytics import YOLO

from tinydet.models.ultralytics import register_ultralytics_modules
from tinydet.modules.qac import QACDetect
from tinydet.modules.qac_v2 import QACDetectV2


def build_model(config: dict):
    register_ultralytics_modules()
    model_path = Path(config["config"])
    if not model_path.is_file():
        raise ValueError(f"Model config does not exist: {model_path}")
    model_yaml = yaml.safe_load(model_path.read_text())
    has_qac = bool(model_yaml.get("qac", False))
    has_qac_v2 = bool(model_yaml.get("qac_v2", False))
    model = YOLO(str(model_path))

    def _swap_head(net):
        if has_qac and hasattr(net, "model") and type(net.model[-1]).__name__ == "Detect":
            gamma = float(model_yaml.get("quality_gamma", 1.0))
            min_conf = float(model_yaml.get("quality_min_conf", 0.25))
            net.model[-1] = QACDetect.from_detect(
                head=net.model[-1], quality_gamma=gamma, quality_min_conf=min_conf
            )
        elif has_qac_v2 and hasattr(net, "model") and type(net.model[-1]).__name__ == "Detect":
            gamma = float(model_yaml.get("quality_gamma", 1.0))
            min_conf = float(model_yaml.get("quality_min_conf", 0.25))
            topk_stat = int(model_yaml.get("topk_stat", 5))
            net.model[-1] = QACDetectV2.from_detect(
                head=net.model[-1], topk_stat=topk_stat, quality_gamma=gamma, quality_min_conf=min_conf
            )

    # Sync custom hyperparameters to model.model.yaml so criterion can access them
    if hasattr(model.model, "yaml") and isinstance(model.model.yaml, dict):
        if has_qac_v2:
            model.model.yaml["qac_v2"] = True
            for key in ("quality_gamma", "quality_min_conf", "topk_stat", "nwd_constant", "nwd_weight", "area_ref", "beta_min", "beta_max"):
                if key in model_yaml:
                    model.model.yaml[key] = model_yaml[key]
        elif has_qac:
            model.model.yaml["qac"] = True
            for key in ("quality_gamma", "quality_min_conf"):
                if key in model_yaml:
                    model.model.yaml[key] = model_yaml[key]

    # Lần 1: Tráo đổi Head ngay sau khi khởi tạo từ YAML
    _swap_head(model.model)

    pretrained = config.get("pretrained")
    if pretrained:
        weights = Path(pretrained)
        if not weights.is_file():
            raise ValueError(f"Pretrained checkpoint does not exist: {weights}")
        # Load weights gốc của YOLO
        model.load(str(weights))
        if has_qac:
            model.model.yaml["qac"] = True
            for key in ("quality_gamma", "quality_min_conf"):
                if key in model_yaml:
                    model.model.yaml[key] = model_yaml[key]
        if has_qac_v2:
            model.model.yaml["qac_v2"] = True
            for key in ("quality_gamma", "quality_min_conf", "topk_stat", "nwd_constant", "nwd_weight", "area_ref", "beta_min", "beta_max"):
                if key in model_yaml:
                    model.model.yaml[key] = model_yaml[key]

        # Lần 2: Tráo đổi Head lại sau khi Ultralytics có thể đã ghi đè cấu trúc model.model lúc load weights
        _swap_head(model.model)

    return model
