import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
import yaml

from tinydet.utils.config import ExperimentConfig


def git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL))
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def metadata(run: Path) -> dict:
    return json.loads((run / "metadata.json").read_text())


def update_status(run: Path, status: str, error: str | None = None) -> None:
    value = metadata(run)
    value["status"] = status
    if error:
        value["error"] = error
    if status in {"COMPLETED", "FAILED"}:
        value["finished_at"] = datetime.now(UTC).isoformat()
    write_json(run / "metadata.json", value)


def create_run(config: ExperimentConfig, source_git: tuple[str, bool] | None = None) -> Path:
    commit, dirty = source_git if source_git is not None else git_info()
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%d-%H%M%S}-{commit}-seed{config.seed}"
    root = Path(config.output["root"]) / config.name
    run = root / run_id
    suffix = 1
    while run.exists():
        run = root / f"{run_id}-{suffix}"
        suffix += 1
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    packages = {name: version(name) for name in ("torch", "torchvision", "ultralytics", "numpy", "opencv-python-headless")}
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    info = {"experiment": config.name, "run_id": run.name, "seed": config.seed, "git_commit": commit, "git_dirty": dirty, "python": platform.python_version(), "torch": packages["torch"], "torchvision": packages["torchvision"], "ultralytics": packages["ultralytics"], "cuda": torch.version.cuda, "gpu": gpu, "dataset": config.data["config"], "model_config_sha256": hashlib.sha256(Path(config.model["config"]).read_bytes()).hexdigest(), "started_at": now.isoformat(), "status": "CREATED"}
    write_json(run / "metadata.json", info)
    (run / "environment.txt").write_text("\n".join([f"python=={platform.python_version()}", *(f"{key}=={value}" for key, value in packages.items()), f"cuda={torch.version.cuda}", f"gpu={gpu}"]) + "\n")
    return run


def resolve_run(root: str | Path, reference: str) -> Path:
    parts = Path(reference).parts
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        raise ValueError("Run must be experiment/run-id")
    run = Path(root).joinpath(*parts)
    if not (run / "metadata.json").is_file():
        raise ValueError(f"Run does not exist: {run}")
    return run
