from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import yaml


def resolve_dataset_yaml(path: str | Path) -> Path:
    """Resolve a dataset path (either a YAML file or a directory containing a YAML file) to a Path."""
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        for name in ("data.yaml", "dataset.yaml"):
            candidate = p / name
            if candidate.is_file():
                return candidate
        candidates = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No YAML configuration file found in dataset directory: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {p}")
    return p


def dataset_images(dataset_file: Path, split: str = "val") -> list[Path]:
    resolved_file = resolve_dataset_yaml(dataset_file)
    data = yaml.safe_load(resolved_file.read_text())
    root = Path(data["path"]) if data.get("path") else resolved_file.parent
    if data.get("path") and not root.is_absolute():
        root = resolved_file.parent / root
    source = root / data[split]
    if source.is_file():
        return [Path(line) if Path(line).is_absolute() else root / line
                for line in source.read_text().splitlines() if line.strip()]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError(f"Image path lacks images directory: {image_path}")
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def normalize_dataset_yaml(path: str | Path) -> Path:
    """Return a portable YAML copy whose relative split paths resolve beside source YAML."""
    source = resolve_dataset_yaml(path)
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict):
        raise TypeError(f"Dataset YAML must be a mapping: {source}")
    configured_root = Path(config.get("path", source.parent)).expanduser()
    if not configured_root.is_absolute():
        configured_root = source.parent / configured_root
    if not configured_root.is_dir():
        configured_root = source.parent
    config["path"] = str(configured_root.resolve())
    key = hashlib.sha256(str(source).encode() + source.read_bytes()).hexdigest()[:16]
    target = Path(tempfile.gettempdir()) / "tinydet-dataset-yaml" / key / "data.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    return target
