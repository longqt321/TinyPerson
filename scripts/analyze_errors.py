"""Build sample-level error records, aggregates, cross-checks, and a local report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinydet.error_analysis.core import AnalysisConfig, analyze_files
from tinydet.error_analysis.inference import predict_split
from tinydet.error_analysis.report import write_outputs
from tinydet.evaluation.tinyperson_benchmark import IOU_THRESHOLDS


def _checkpoint(run: Path | None, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    if run is None:
        return None
    for candidate in (run / "checkpoints" / "best.pt", run / "training" / "weights" / "best.pt"):
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Original TinyPerson COCO annotation JSON")
    parser.add_argument("--manifest", type=Path,
                        help="Derivative manifest.json; selects the requested split")
    parser.add_argument("--split", default="val")
    parser.add_argument("--predictions", type=Path,
                        help="Prediction JSON or evaluation predictions.pt")
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint used to generate predictions when --predictions is absent")
    parser.add_argument("--data", type=Path,
                        help="YOLO data.yaml used with --checkpoint to generate predictions")
    parser.add_argument("--run", type=Path, help="Run directory for provenance and checkpoint lookup")
    parser.add_argument("--output", type=Path,
                        help="Output directory; defaults to RUN/analysis-SPLIT")
    parser.add_argument("--limit", type=int, default=0,
                        help="Analyze only the first N images sorted by basename")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--iou", type=float, default=0.50, help="Primary IoU threshold")
    parser.add_argument("--localization-iou", type=float, default=0.10)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--baseline-samples", type=Path,
                        help="Earlier samples.jsonl for paired corrected/regressed counts")
    parser.add_argument("--plots", action="store_true",
                        help="Write diagnostic PNGs from the validated records")
    args = parser.parse_args()

    if args.output is None:
        if args.run is None:
            parser.error("--output is required unless --run is provided")
        args.output = args.run / f"analysis-{args.split}"
    checkpoint = _checkpoint(args.run, args.checkpoint)
    predictions = args.predictions
    if predictions is None:
        if checkpoint is None or args.data is None:
            parser.error("provide --predictions, or --checkpoint/--run and --data to infer")
        predictions = args.output / "predictions.json"
        inference = predict_split(
            checkpoint, args.data, args.split, predictions,
            imgsz=args.imgsz, conf=args.conf, batch_size=args.batch_size,
            limit=args.limit, device=args.device, cpu_threads=args.cpu_threads,
        )
        print(json.dumps(inference, indent=2))

    config = AnalysisConfig(
        iou_thresholds=tuple(sorted({*IOU_THRESHOLDS, args.iou})),
        primary_iou=args.iou,
        localization_iou=args.localization_iou,
        score_threshold=args.score_threshold,
    )
    result = analyze_files(
        args.annotations,
        predictions,
        manifest_path=args.manifest,
        split=args.split,
        config=config,
        limit=args.limit,
        baseline_samples=args.baseline_samples,
        run=args.run,
        checkpoint=checkpoint,
    )
    paths = write_outputs(result, args.output, plots=args.plots)
    print(json.dumps({
        "valid": result.summary["integrity"]["valid"],
        "analyzed_images": result.summary["dataset"]["analyzed_images"],
        "counts": result.summary["counts"],
        "rates": result.summary["rates"],
        "crosscheck": result.summary["metric_crosscheck"].get("status"),
        "outputs": paths,
    }, indent=2))


if __name__ == "__main__":
    main()
