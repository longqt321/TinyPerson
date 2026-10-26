"""Compute diagnostic AP75 oracles without changing the TinyPerson evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tinydet.evaluation.tinyperson_benchmark import evaluate_tinyperson


def _iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(second[3] - second[1], 0.0)
    return intersection / (first_area + second_area - intersection) if intersection else 0.0


def _variant_predictions(
    predictions: dict[str, Any], annotations: dict[str, Any], variant: str
) -> dict[str, Any]:
    positives: dict[int, list[list[float]]] = {}
    for image_id, anns in annotations.items():
        if image_id == "images":
            continue
        positives[image_id] = [
            ann["box"] for ann in anns if not (ann["ignore"] or ann["uncertain"])
        ]

    images = []
    matched_gt = 0
    matched_detections = 0
    for row in predictions["images"]:
        boxes = [list(map(float, box)) for box in row["boxes"]]
        scores = list(map(float, row["scores"]))
        image_id = next(
            image["id"] for image in annotations["images"]
            if Path(image["file_name"]).name == Path(row["path"]).name
        )
        gt = positives[image_id]
        order = sorted(range(len(boxes)), key=lambda index: scores[index], reverse=True)
        assignments: dict[int, int] = {}
        for detection_index in order:
            overlaps = [
                (index, _iou(boxes[detection_index], box))
                for index, box in enumerate(gt)
            ]
            candidates = [
                (index, overlap) for index, overlap in overlaps
                if overlap >= 0.75 and index not in assignments
            ]
            if candidates:
                gt_index, _ = max(candidates, key=lambda pair: pair[1])
                assignments[detection_index] = gt_index

        matched_gt += len(set(assignments.values()))
        matched_detections += len(assignments)
        output_boxes, output_scores = [], []
        for detection_index, (box, score) in enumerate(zip(boxes, scores)):
            if variant in {"box", "both"} and detection_index in assignments:
                box = gt[assignments[detection_index]]
            if variant in {"score", "both"}:
                score = max(
                    (_iou(box, candidate) for candidate in gt), default=0.0
                )
                if variant == "both":
                    score = 1.0 if detection_index in assignments else 0.0
            output_boxes.append(box)
            output_scores.append(score)
        images.append({
            "path": row["path"], "boxes": output_boxes,
            "scores": output_scores, "classes": [0] * len(output_boxes),
        })
    return {"images": images, "matched_gt": matched_gt, "matched_detections": matched_detections}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    raw_annotations = json.loads(args.annotations.read_text())
    gt_annotations: dict[int, list[dict[str, Any]]] = {
        int(image["id"]): [] for image in raw_annotations["images"]
    }
    for ann in raw_annotations["annotations"]:
        x, y, width, height = map(float, ann["bbox"])
        gt_annotations[int(ann["image_id"])].append({
            "box": [x, y, x + width, y + height],
            "ignore": bool(ann.get("ignore", False)),
            "uncertain": bool(ann.get("uncertain", False)),
        })
    predictions = json.loads(args.predictions.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    rows: list[str] = ["variant,size_bin,AP@0.25,AP@0.50,AP@0.75"]
    for variant in ("baseline", "box", "score", "both"):
        if variant == "baseline":
            payload, matched_gt, matched_detections = predictions, None, None
        else:
            payload = _variant_predictions(
                predictions, {"images": raw_annotations["images"], **gt_annotations}, variant
            )
            matched_gt, matched_detections = payload.pop("matched_gt"), payload.pop("matched_detections")
        temporary = args.output_dir / f"{variant}.json"
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        result = evaluate_tinyperson(args.annotations, temporary, args.workers)
        temporary.unlink()
        summary[variant] = {
            "metrics": result, "matched_gt_iou_075": matched_gt,
            "matched_detections_iou_075": matched_detections,
        }
        for size_bin, metrics in result.items():
            if size_bin == "protocol":
                continue
            rows.append(
                f"{variant},{size_bin},{metrics['AP@0.25']:.9f},"
                f"{metrics['AP@0.50']:.9f},{metrics['AP@0.75']:.9f}"
            )

    (args.output_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "oracle_by_size.csv").write_text("\n".join(rows) + "\n")
    compact = {
        variant: {
            "all_AP75": summary[variant]["metrics"]["all"]["AP@0.75"],
            "matched_gt_iou_075": summary[variant]["matched_gt_iou_075"],
        }
        for variant in summary
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
