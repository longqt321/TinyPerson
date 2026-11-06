"""Report sizes needed to migrate the ablation study to another Modal account.

Prints one compact JSON line per item so the output survives log interleaving.

Run:
    uv run --with modal --extra cpu --group dev modal run modal/migrate_sizes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import app as base_app

app = base_app.app
image = base_app.image
data = base_app.data
runs = base_app.runs
DATA_ROOT = base_app.DATA_ROOT
RUN_ROOT = base_app.RUN_ROOT

ABLATIONS = ("no_high_band", "no_mid_band", "no_low_band", "no_router", "no_saliency")


def _size(path: Path) -> tuple[int, int]:
    total = count = 0
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
    volumes={str(DATA_ROOT): data, str(RUN_ROOT): runs},
    timeout=1800,
)
def _measure() -> list[str]:
    import json

    runs.reload()
    data.reload()
    lines = []

    ds = DATA_ROOT / "tinyperson-yolo"
    size, count = _size(ds)
    lines.append(json.dumps({
        "item": "dataset/tinyperson-yolo",
        "gb": round(size / 1e9, 2),
        "files": count,
        "migrate": "re-upload (needed to train on the new account)",
    }))

    total = 0
    for abl in ABLATIONS:
        name = f"abl_gauss_{abl}_seed42"
        run_dir = RUN_ROOT / name / name
        if not run_dir.is_dir():
            lines.append(json.dumps({"item": name, "status": "MISSING"}))
            continue
        size, count = _size(run_dir)
        total += size
        weights = run_dir / "training" / "weights"
        wsize = 0
        if weights.is_dir():
            wsize = sum(f.stat().st_size for f in weights.iterdir() if f.is_file())
        lines.append(json.dumps({
            "item": name,
            "gb": round(size / 1e9, 3),
            "files": count,
            "weights_gb": round(wsize / 1e9, 3),
        }))

    lines.append(json.dumps({
        "item": "TOTAL_5_ABLATION_RUNS",
        "gb": round(total / 1e9, 2),
    }))
    return lines


@app.local_entrypoint()
def main() -> None:
    for line in _measure.remote():
        print(line, flush=True)
