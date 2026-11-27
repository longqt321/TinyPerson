from pathlib import Path
from typing import Any, cast

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.downloads import attempt_download_asset

from tinydet.models.ultralytics import register_ultralytics_modules


def build_model(config: dict):
    register_ultralytics_modules()
    model_path = Path(config["config"])
    if not model_path.is_file():
        raise ValueError(f"Model config does not exist: {model_path}")

    model = YOLO(str(model_path))
    net = cast(Any, model.model)
    if isinstance(getattr(net, "args", None), dict):
        net.args = get_cfg(overrides=net.args)
    if getattr(net, "yaml", {}).get("e2e_eval", False):
        net.model[-1].end2end = True

    pretrained = config.get("pretrained")
    if pretrained:
        weights = Path(str(pretrained)).expanduser()
        if not weights.is_file():
            downloaded = Path(attempt_download_asset(str(pretrained)))
            if downloaded.is_file():
                weights = downloaded
        if not weights.is_file():
            raise ValueError(f"Pretrained checkpoint does not exist: {pretrained}")
        model.load(str(weights))

    return model
