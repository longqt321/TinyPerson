"""Evaluate ablation checkpoints with the frozen TOV tiled protocol on CPU.

Same protocol as `modal/app.py::_run_tiled_checkpoint_evaluation`:
tile 640x512, overlap 100, global NMS IoU 0.50, top-2000 pre-NMS,
max 1000 detections/image, IoD > 0.50 ignore handling (inside the benchmark).

Only the device differs (cpu instead of cuda:0), because Modal is over its
spend limit. Nothing in the protocol is changed.

The ablation flag is NOT passed on the command line: it is read from the
checkpoint itself, where `GauSS.ablation_flags` was pickled during training.
This script asserts the flag matches the expected value so a silent
full-model evaluation cannot happen.

Usage:
    uv run --extra cpu python scripts/eval_ablation_local.py --runs no_high_band
    uv run --extra cpu python scripts/eval_ablation_local.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from PIL import Image
from ultralytics import YOLO

from tinydet.evaluation.tiled_inference import predict_tiled_image
from tinydet.evaluation.tinyperson_benchmark import evaluate_tinyperson
from tinydet.models.ultralytics import register_ultralytics_modules

register_ultralytics_modules()

BACKUP_ROOT = Path("backups/ablation_seed42")
DATA_ROOT = Path("backups/data/tinyperson-yolo")
OUT_ROOT = Path("reports/eval/ablation")
ABLATIONS = ("no_high_band", "no_mid_band", "no_low_band", "no_router", "no_saliency")


def _subset_annotations(source: Path, test_dir: Path, destination: Path) -> Path:
    """Restrict the official COCO JSON to the 786 test-split images.

    Mirrors `modal/app.py::_subset_annotations` so the ignore/uncertain/dense
    metadata is preserved exactly as in the paper evaluation.
    """
    wanted = {p.name for p in test_dir.glob("*.jpg")}
    payload = json.loads(source.read_text())
    keep_ids = {
        int(row["id"]) for row in payload["images"]
        if Path(row["file_name"]).name in wanted
    }
    images = [row for row in payload["images"] if int(row["id"]) in keep_ids]
    annotations = [
        row for row in payload["annotations"] if int(row["image_id"]) in keep_ids
    ]
    covered = {Path(row["file_name"]).name for row in images}
    missing = wanted - covered
    if missing:
        raise ValueError(
            f"{len(missing)} test image(s) have no annotation; "
            f"examples={sorted(missing)[:5]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({**payload, "images": images, "annotations": annotations}, indent=2)
        + "\n"
    )
    return destination


def _checkpoint(ablation: str) -> Path:
    run = f"abl_gauss_{ablation}_seed42"
    path = BACKUP_ROOT / run / run / "training" / "weights" / "last.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _flags(model) -> frozenset[str]:
    for _, module in model.model.named_modules():
        if type(module).__name__ == "GauSS":
            return frozenset(getattr(module, "ablation_flags", frozenset()))
    raise RuntimeError("No GauSS module found in the loaded model")


def _trained_epochs(checkpoint: Path) -> int:
    """Number of epochs actually completed, read from the sibling results.csv.

    The five seed-42 ablation runs were interrupted when Modal credit ran out,
    so the requested 150 epochs were never reached. Recording the real count
    prevents a partial run from being reported as a full-length one.
    """
    results_csv = checkpoint.parent.parent / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(results_csv)
    lines = results_csv.read_text().strip().splitlines()
    if len(lines) < 2:
        return 0
    return int(lines[-1].split(",")[0])


def evaluate(ablation: str, annotations: Path, test_dir: Path, limit: int | None,
             class_mode: str = "merge") -> dict:
    out_dir = OUT_ROOT / f"abl_gauss_{ablation}_seed42"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions_tiled.json"

    ckpt = _checkpoint(ablation)
    model = YOLO(ckpt)
    flags = _flags(model)
    if flags != frozenset({ablation}):
        raise RuntimeError(
            f"Checkpoint {ckpt} carries flags {sorted(flags)}, expected "
            f"['{ablation}']. Refusing to evaluate: the score would not describe "
            "the intended ablation."
        )
    print(f"[{ablation}] flags={sorted(flags)} ckpt={ckpt}", flush=True)

    payload = json.loads(annotations.read_text())
    images = payload["images"]
    if limit:
        images = images[:limit]

    predictions = []
    started = time.perf_counter()
    for idx, info in enumerate(images):
        name = Path(info["file_name"]).name
        path = test_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as pil:
            boxes, scores, classes = predict_tiled_image(
                model=model,
                img=pil,
                tile_w=640,
                tile_h=512,
                overlap=100,
                conf=0.001,
                merge_iou=0.50,
                nms_pre=2000,
                max_per_img=1000,
                device="cpu",
            )
        predictions.append(
            {"path": str(path), "boxes": boxes, "scores": scores, "classes": classes}
        )
        if (idx + 1) % 50 == 0:
            rate = (time.perf_counter() - started) / (idx + 1)
            print(
                f"  [{ablation}] {idx + 1}/{len(images)} "
                f"({rate:.2f} s/img, eta {rate * (len(images) - idx - 1) / 60:.1f} min)",
                flush=True,
            )

    pred_file.write_text(json.dumps({"images": predictions}))
    elapsed = time.perf_counter() - started
    print(f"[{ablation}] inference done in {elapsed / 60:.1f} min", flush=True)

    if limit:
        print(f"[{ablation}] --limit set; skipping benchmark (partial predictions)", flush=True)
        return {"ablation": ablation, "inference_seconds": elapsed, "limited": True}

    bench = evaluate_tinyperson(annotations, pred_file, workers=4, class_mode=class_mode)
    summary = {
        "name": f"abl_gauss_{ablation}_seed42",
        "checkpoint": str(ckpt),
        "ablation": ablation,
        "ablation_flags": sorted(flags),
        "epoch": _trained_epochs(ckpt),
        "epochs_requested": 150,
        "training_complete": _trained_epochs(ckpt) >= 150,
        "protocol": f"Tiled crop 640x512 + NMS merge 0.50 (CPU); class_mode={class_mode}",
        "inference_seconds": round(elapsed, 1),
        "benchmark": bench,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # The per-class breakdown is also written on its own so that every ablation
    # run exposes the same artifact regardless of when it was scored.
    (out_dir / "summary_per_class.json").write_text(
        json.dumps(bench, indent=2) + "\n"
    )
    print(f"[{ablation}] wrote {out_dir / 'summary.json'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", choices=ABLATIONS, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only first N images (smoke test; skips benchmark)")
    parser.add_argument("--class-mode", default="merge",
                        choices=["per_class", "merge", "primary"],
                        help="Single-class TinyPerson models must use 'merge'.")
    args = parser.parse_args()

    runs = list(ABLATIONS) if args.all else (args.runs or [])
    if not runs:
        parser.error("pass --all or --runs <name> [...]")

    test_dir = DATA_ROOT / "images" / "test"
    source = DATA_ROOT / "annotations" / "tiny_set_test_with_dense.json"
    annotations = _subset_annotations(
        source, test_dir, OUT_ROOT / "annotations_test.json"
    )
    print(f"annotations subset -> {annotations}", flush=True)

    results = []
    for ablation in runs:
        try:
            results.append(evaluate(ablation, annotations, test_dir, args.limit, args.class_mode))
        except Exception as exc:  # noqa: BLE001 - report and continue to next run
            print(f"[{ablation}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"ablation": ablation, "error": f"{type(exc).__name__}: {exc}"})

    print(json.dumps(results, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
