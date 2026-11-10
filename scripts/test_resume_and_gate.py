"""CPU integration test for the two pre-train risks of the G1 launch (ADR-025):

1. Resume: Ultralytics must continue an aborted run from last.pt (epoch counter
   advances instead of restarting).
2. Gate wiring: GaussGateCallback must fire inside a real Ultralytics run with
   a real GauSS model (the earlier test used a fake trainer).

Run: uv run --extra cpu python scripts/test_resume_and_gate.py
Cost: no GPU, ~2-5 min.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IMG_SIZE = 64
N_IMAGES = 8
MODEL_YAML = "configs/model/yolo26n_tiny_A2_p2_gauss_tinyperson.yaml"


def make_dataset(root: Path) -> Path:
    import numpy as np
    from PIL import Image

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(N_IMAGES):
            rng = np.random.default_rng(i)
            arr = (rng.random((IMG_SIZE, IMG_SIZE, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(root / "images" / split / f"img{i}.jpg")
            cx, cy, w, h = 0.5, 0.5, 0.05, 0.05
            (root / "labels" / split / f"img{i}.txt").write_text(f"0 {cx} {cy} {w} {h}\n")
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n"
    )
    return root / "data.yaml"


def train_args(epochs: int, project: Path, name: str, data_yaml: Path) -> dict:
    return dict(
        data=str(data_yaml), project=str(project), name=name, exist_ok=True,
        epochs=epochs, imgsz=IMG_SIZE, batch=2, device="cpu", verbose=False,
        plots=False, save_period=-1, workers=0, val=True,
    )


def main() -> int:
    from ultralytics import YOLO

    from tinydet.models.registry import build_model  # registers custom modules

    tmp = Path(tempfile.mkdtemp(prefix="resume_gate_"))
    data_yaml = make_dataset(tmp / "data")
    project = tmp / "runs"

    # --- 1. resume: train 2 epochs, then resume for 2 more ---
    model = build_model({"config": MODEL_YAML})
    model.train(**train_args(2, project, "r", data_yaml))
    last = project / "r" / "weights" / "last.pt"
    if not last.is_file():
        print("FAIL: last.pt missing after first train")
        return 1
    resumed = YOLO(str(last))
    resumed.train(resume=True, **train_args(4, project, "r", data_yaml))
    rows = list(csv.DictReader(open(project / "r" / "results.csv")))
    epochs = [int(r["epoch"]) for r in rows]
    if max(epochs) < 3:
        print(f"FAIL: resume did not continue (max epoch {max(epochs)})")
        return 1
    print(f"PASS resume: epochs {epochs}")

    # --- 2. gate callback fires in a real run with a real GauSS model ---
    from tinydet.training.gauss_gate import GaussGateCallback

    gate = GaussGateCallback({
        "epochs": [1], "val_map50_floor": 0.0, "sigma2_std_min": 0.0,
        "alpha_min": 0.0, "residual_share_min": 0.0,
    })
    fired = []
    original = gate.on_fit_epoch_end

    def spy(trainer):
        fired.append(int(getattr(trainer, "epoch", -1)) + 1)
        original(trainer)

    gate.on_fit_epoch_end = spy  # type: ignore[method-assign]
    m = build_model({"config": MODEL_YAML})
    m.callbacks["on_train_start"].append(gate.on_train_start)
    m.callbacks["on_fit_epoch_end"].append(gate.on_fit_epoch_end)
    m.train(**train_args(1, project, "g", data_yaml))
    if not fired:
        print("FAIL: gate callback never fired in a real Ultralytics run")
        return 1
    verdict_path = project / "g" / "metrics" / "gauss_gate_verdict.json"
    if verdict_path.is_file():
        print("verdict:", json.loads(verdict_path.read_text())[0]["checks"])
    print(f"PASS gate: fired at epoch {fired}")
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
