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
    out: dict = {}
    for label, root in (("data", DATA_ROOT), ("models", MODEL_ROOT), ("runs", RUN_ROOT)):
        size, count = _dir_size(root)
        out[label] = {"bytes": size, "gb": round(size / 1e9, 2), "files": count}
    # Per-run breakdown so a partial migration can pick only what is needed.
    per_run = {}
    for entry in sorted(RUN_ROOT.iterdir()):
        if entry.is_dir():
            size, count = _dir_size(entry)
            per_run[entry.name] = {"gb": round(size / 1e9, 2), "files": count}
    out["runs_per_dir"] = per_run
    # Dataset breakdown.
    ds = {}
    for entry in sorted(DATA_ROOT.iterdir()):
        if entry.is_dir():
            size, count = _dir_size(entry)
            ds[entry.name] = {"gb": round(size / 1e9, 2), "files": count}
    out["data_per_dir"] = ds
    return out


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(_sizes.remote(), indent=2), flush=True)
