"""Measure Modal volume sizes to plan a cross-account migration.

Run:
    uv run --with modal --extra cpu --group dev modal run modal/volume_sizes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import app as base_app

app = base_app.app
image = base_app.image
data = base_app.data
models = base_app.models
runs = base_app.runs
DATA_ROOT = base_app.DATA_ROOT
MODEL_ROOT = base_app.MODEL_ROOT
RUN_ROOT = base_app.RUN_ROOT


def _dir_size(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
                count += 1
            except OSError:
                continue
    return total, count


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    max_containers=1,
    volumes={str(DATA_ROOT): data, str(MODEL_ROOT): models, str(RUN_ROOT): runs},
    timeout=1800,
)
def _sizes() -> dict:
    data.reload()
    models.reload()
    runs.reload()

    def _breakdown_with_total(root: Path) -> tuple[dict[str, int | float], dict[str, dict[str, int | float]]]:
        total_bytes = 0
        total_files = 0
        per_dir: dict[str, dict[str, int | float]] = {}
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                size, count = _dir_size(entry)
                per_dir[entry.name] = {"gb": round(size / 1e9, 2), "files": count}
                total_bytes += size
                total_files += count
            elif entry.is_file():
                try:
                    total_bytes += entry.stat().st_size
                    total_files += 1
                except OSError:
                    continue
        return {"bytes": total_bytes, "gb": round(total_bytes / 1e9, 2), "files": total_files}, per_dir

    data_total, ds = _breakdown_with_total(DATA_ROOT)
    models_size, models_count = _dir_size(MODEL_ROOT)
    runs_total, per_run = _breakdown_with_total(RUN_ROOT)
    return {
        "data": data_total,
        "models": {"bytes": models_size, "gb": round(models_size / 1e9, 2), "files": models_count},
        "runs": runs_total,
        "runs_per_dir": per_run,
        "data_per_dir": ds,
    }


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(_sizes.remote(), indent=2), flush=True)
