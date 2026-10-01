import json
from pathlib import Path

import yaml


def find_reusable_run(root: Path, expected: dict, model_sha256: str) -> dict | None:
    """Find newest complete run with an identical experiment config and checkpoint."""
    directory = root / expected["name"]
    if not directory.is_dir():
        return None
    for run in sorted(directory.iterdir(), reverse=True):
        try:
            saved = yaml.safe_load((run / "config.yaml").read_text())
            metadata = json.loads((run / "metadata.json").read_text())
        except (OSError, ValueError, yaml.YAMLError):
            continue
        if saved != expected or metadata.get("status") != "COMPLETED":
            continue
        if not (run / "checkpoints" / "best.pt").is_file():
            continue
        recorded_hash = metadata.get("model_config_sha256")
        if recorded_hash is not None and recorded_hash != model_sha256:
            continue
        return {
            "run": f"{expected['name']}/{run.name}",
            "legacy": recorded_hash is None,
            "val": (run / "metrics" / "val.json").is_file(),
            "benchmark": (run / "metrics" / "benchmark.json").is_file(),
        }
    return None
