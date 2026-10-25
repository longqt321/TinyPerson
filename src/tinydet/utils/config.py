from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tinydet.diagnostics.config import DiagnosticsConfig


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    model: dict[str, Any]
    data: dict[str, Any]
    train: dict[str, Any]
    hardware: dict[str, Any]
    output: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Experiment config does not exist: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("Experiment config must be a YAML mapping")
    unknown = set(raw) - {"name", "seed", "model", "data", "train", "hardware", "output", "diagnostics"}
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
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
        if section == "model" and set(value) - {"config", "pretrained"}:
            raise ValueError("model accepts only config and pretrained")
        missing = keys - value.keys()
        if missing:
            raise ValueError(f"Missing {section} fields: {', '.join(sorted(missing))}")
    if raw["train"]["epochs"] <= 0 or raw["train"]["imgsz"] <= 0 or raw["train"]["batch"] <= 0:
        raise ValueError("epochs, imgsz and batch must be positive")
    diagnostics = DiagnosticsConfig.parse(raw.get("diagnostics")).to_dict()
    return ExperimentConfig(**{**{key: raw[key] for key in required}, "diagnostics": diagnostics})
