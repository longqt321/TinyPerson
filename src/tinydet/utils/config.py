from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    model: dict[str, Any]
    data: dict[str, Any]
    train: dict[str, Any]
    hardware: dict[str, Any]
    output: dict[str, Any]


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Experiment config does not exist: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("Experiment config must be a YAML mapping")
    required = {"name", "seed", "model", "data", "train", "hardware", "output"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Missing config fields: {', '.join(sorted(missing))}")
    if not isinstance(raw["name"], str) or not raw["name"].replace("-", "").replace("_", "").isalnum():
        raise ValueError("name must contain only letters, digits, '-' or '_'")
    if type(raw["seed"]) is not int or raw["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    fields = {"model": {"config", "pretrained"}, "data": {"config"}, "train": {"epochs", "imgsz", "batch", "optimizer", "lr0", "amp", "deterministic"}, "hardware": {"gpu"}, "output": {"root"}}
    for section, keys in fields.items():
        value = raw[section]
        if not isinstance(value, dict):
            raise TypeError(f"{section} must be a mapping")
        missing = keys - value.keys()
        if missing:
            raise ValueError(f"Missing {section} fields: {', '.join(sorted(missing))}")
    if raw["train"]["epochs"] <= 0 or raw["train"]["imgsz"] <= 0 or raw["train"]["batch"] <= 0:
        raise ValueError("epochs, imgsz and batch must be positive")
    if "modules" in raw["model"]:
        from tinydet.modules.ultralytics_adapter import validate_modules

        raw["model"]["modules"] = validate_modules(raw["model"]["modules"])
    return ExperimentConfig(**{key: raw[key] for key in required})
