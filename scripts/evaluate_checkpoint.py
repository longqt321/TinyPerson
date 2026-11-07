"""Post-training evaluation pipeline for TinyPerson checkpoints.

Runs the full evaluation stack against any trained `.pt` checkpoint (e.g. the
`best.pt` produced by a training run, or a standalone checkpoint file):

1. Prediction generation (source coordinates, official protocol).
2. TinyPerson benchmark evaluation (ignore-aware, size ranges).
3. AP75 diagnostic oracles (score / box / both variants).
4. Sample-level error analysis (size bins, density bins, geometry).
5. Unified summary report.

Can be invoked right after a training run finishes, or later on any `.pt` file
produced by a previous run. All steps write into one output directory.

Usage:
    uv run --extra cpu --group dev python scripts/evaluate_checkpoint.py \
        --checkpoint runs/<run>/best.pt \
        --annotations /mnt/data/tinyperson-yolo/annotations/tiny_set_test_with_dense.json \
        --data /mnt/data/tinyperson-yolo/data.yaml \
        --output eval_results/<run_name> \
        --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tinydet.error_analysis.core import AnalysisConfig, analyze_files
from tinydet.error_analysis.inference import predict_split
from tinydet.error_analysis.report import write_outputs
from tinydet.evaluation.tinyperson_benchmark import IOU_THRESHOLDS, evaluate_tinyperson
from tinydet.models.ultralytics import register_ultralytics_modules
from tinydet.utils.dataset import dataset_images

register_ultralytics_modules()


def _run_oracles(annotations: Path, predictions: Path, output_dir: Path, workers: int,
                 class_mode: str = "per_class") -> dict[str, Any]:
    """Run the AP75 diagnostic oracles (baseline / score / box / both)."""
    import subprocess
    import sys

    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable, str(Path(__file__).parent / "oracle_ap75.py"),
            "--annotations", str(annotations),
            "--predictions", str(predictions),
            "--output-dir", str(output_dir / "oracle"),
            "--workers", str(workers),
            "--class-mode", class_mode,
        ],
        capture_output=True, text=True, check=False,
    )
    summary_path = output_dir / "oracle" / "oracle_summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    return {
        "error": result.stderr.strip() or result.stdout.strip() or "oracle script failed",
        "returncode": result.returncode,
    }


def _run_error_analysis(
    annotations: Path,
    predictions: Path,
    output_dir: Path,
    *,
    iou: float = 0.50,
    localization_iou: float = 0.10,
    score_threshold: float = 0.001,
    plots: bool = False,
) -> dict[str, Any]:
    """Run sample-level error analysis (size bins, density bins, geometry)."""
    config = AnalysisConfig(
        iou_thresholds=tuple(sorted({*IOU_THRESHOLDS, iou})),
        primary_iou=iou,
        localization_iou=localization_iou,
        score_threshold=score_threshold,
    )
    result = analyze_files(
        annotations, predictions, config=config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(result, output_dir, plots=plots)
    return result.summary


def _benchmark_metrics(annotations: Path, predictions: Path, workers: int,
                       class_mode: str = "per_class") -> dict[str, Any]:
    """Run the official TinyPerson benchmark evaluator."""
    return evaluate_tinyperson(annotations, predictions, workers, class_mode=class_mode)


def _geometry_summary(analysis_summary: dict[str, Any]) -> dict[str, Any]:
    """Extract geometry-related aggregates from the error-analysis summary."""
    return {
        "size_bins": analysis_summary.get("by_size_bin"),
        "density_bins": analysis_summary.get("by_density"),
        "counts": analysis_summary.get("counts"),
        "rates": analysis_summary.get("rates"),
        "diagnostic_ap": analysis_summary.get("diagnostic_ap"),
        "operating_points": analysis_summary.get("confidence_operating_points"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to a trained .pt checkpoint (e.g. best.pt)")
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Original TinyPerson COCO annotation JSON")
    parser.add_argument("--data", required=True, type=Path,
                        help="YOLO data.yaml for the split being evaluated")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for all evaluation artifacts")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0")
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU processes for the benchmark evaluator; 0 uses up to 7 cores")
    parser.add_argument("--iou", type=float, default=0.50, help="Primary IoU threshold")
    parser.add_argument("--localization-iou", type=float, default=0.10)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--plots", action="store_true",
                        help="Write diagnostic PNGs from the validated records")
    parser.add_argument("--skip-oracle", action="store_true",
                        help="Skip the AP75 oracle step")
    parser.add_argument("--skip-error-analysis", action="store_true",
                        help="Skip the sample-level error analysis step")
    parser.add_argument("--class-mode", default="per_class",
                        choices=["per_class", "merge", "primary"],
                        help="How detector classes map onto TinyPerson categories. "
                             "A single-class model must use 'merge'.")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    images = dataset_images(args.data, args.split)
    if not images:
        raise ValueError(f"No images found for split {args.split!r} in {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)

    # Step 1: predictions
    predictions_path = args.output / "predictions.json"
    print(f"[1/4] Generating predictions for split={args.split!r} ({len(images)} images)")
    inference = predict_split(
        args.checkpoint, args.data, args.split, predictions_path,
        imgsz=args.imgsz, conf=args.conf, batch_size=args.batch_size,
        device=args.device,
    )
    (args.output / "inference.json").write_text(json.dumps(inference, indent=2) + "\n")

    # Step 2: benchmark
    print("[2/4] Running TinyPerson benchmark evaluator")
    benchmark = _benchmark_metrics(args.annotations, predictions_path, args.workers,
                                   args.class_mode)
    (args.output / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")

    # Step 3: oracles
    oracles: dict[str, Any] | None = None
    if not args.skip_oracle:
        print("[3/4] Running AP75 diagnostic oracles")
        oracles = _run_oracles(args.annotations, predictions_path, args.output, args.workers,
                               args.class_mode)
        (args.output / "oracle_summary.json").write_text(json.dumps(oracles, indent=2) + "\n")

    # Step 4: error analysis
    analysis_summary: dict[str, Any] | None = None
    if not args.skip_error_analysis:
        print("[4/4] Running sample-level error analysis")
        analysis_summary = _run_error_analysis(
            args.annotations, predictions_path, args.output / f"analysis-{args.split}",
            iou=args.iou,
            localization_iou=args.localization_iou,
            score_threshold=args.score_threshold,
            plots=args.plots,
        )

    # Unified summary
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "annotations": str(args.annotations.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "class_mode": args.class_mode,
        "benchmark": benchmark,
        "oracle": oracles,
        "error_analysis": _geometry_summary(analysis_summary) if analysis_summary else None,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "summary": str(summary_path.resolve()),
        "benchmark_all": benchmark.get("all"),
        "oracle_all_AP75": {
            k: v["metrics"]["all"]["AP@0.75"]
            for k, v in (oracles or {}).items()
            if isinstance(v, dict) and "metrics" in v
        } if oracles else None,
    }, indent=2))


def _sha256(path: Path, chunk: int = 8 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
