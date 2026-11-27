"""Compute diagnostic AP75 oracles without changing the TinyPerson evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from tinydet.evaluation.tinyperson_benchmark import _load_gt, evaluate_tinyperson
from tinydet.utils.atomic import atomic_write_text
from tinydet.utils.geometry import greedy_assign_first_max, iou_matrix


def _precompute_image_matches(
    predictions: dict[str, Any], raw_annotations: dict[str, Any]
) -> tuple[list[dict[str, Any]], int, int]:
    """Compute IoU>=0.75 greedy matches and IoU matrices once across all oracle arms."""
    positives: dict[int, list[list[float]]] = {
        int(image["id"]): [] for image in raw_annotations["images"]
    }
    for ann in raw_annotations["annotations"]:
        if bool(ann.get("ignore", False)) or bool(ann.get("uncertain", False)):
            continue
        x, y, width, height = map(float, ann["bbox"])
        positives[int(ann["image_id"])].append([x, y, x + width, y + height])
    basename_to_id = {
        Path(image["file_name"]).name: int(image["id"]) for image in raw_annotations["images"]
    }

    prepared: list[dict[str, Any]] = []
    matched_gt = 0
    matched_detections = 0
    for row in predictions["images"]:
        boxes = [list(map(float, box)) for box in row["boxes"]]
        scores = list(map(float, row["scores"]))
        image_id = basename_to_id[Path(row["path"]).name]
        gt = positives[image_id]
        overlap_mat = np.ascontiguousarray(iou_matrix(boxes, gt), dtype=np.float64)
        scores_arr = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-scores_arr, kind="stable").astype(np.int64, copy=False)
        assigned_arr = greedy_assign_first_max(overlap_mat, order, 0.75)
        assignments = {
            int(det_idx): int(gt_idx)
            for det_idx, gt_idx in enumerate(assigned_arr)
            if gt_idx >= 0
        }
        matched_gt += len(assignments)
        matched_detections += len(assignments)
        prepared.append({
            "path": row["path"],
            "boxes": boxes,
            "scores": scores,
            "gt": gt,
            "max_iou": overlap_mat.max(axis=1).tolist() if overlap_mat.size else [0.0] * len(boxes),
            "assignments": assignments,
        })
    return prepared, matched_gt, matched_detections


def _build_variant_payload(prepared: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    images = []
    for item in prepared:
        boxes = item["boxes"]
        gt = item["gt"]
        assignments = item["assignments"]
        output_boxes = [
            gt[assignments[idx]] if (variant in {"box", "both"} and idx in assignments) else box
            for idx, box in enumerate(boxes)
        ]
        if variant == "score":
            output_scores = item["max_iou"]
        elif variant == "both":
            output_scores = [1.0 if idx in assignments else 0.0 for idx in range(len(output_boxes))]
        else:
            output_scores = item["scores"]
        images.append({
            "path": item["path"],
            "boxes": output_boxes,
            "scores": output_scores,
            "classes": [0] * len(output_boxes),
        })
    return {"images": images}


def run_oracles(
    annotations_path: Path,
    predictions_path: Path | dict[str, Any],
    output_dir: Path,
    workers: int = 0,
    class_mode: str = "per_class",
    baseline_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_annotations = orjson.loads(annotations_path.read_bytes())
    predictions = (
        predictions_path
        if isinstance(predictions_path, dict)
        else orjson.loads(Path(predictions_path).read_bytes())
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_cached = _load_gt(raw_annotations)
    gt_cached["source"] = str(annotations_path.resolve())
    prepared, matched_gt_all, matched_det_all = _precompute_image_matches(
        predictions, raw_annotations
    )

    summary: dict[str, Any] = {}
    rows: list[str] = ["variant,size_bin,AP@0.25,AP@0.50,AP@0.75"]
    for variant in ("baseline", "box", "score", "both"):
        if variant == "baseline":
            result = (
                baseline_result
                if baseline_result is not None
                else evaluate_tinyperson(
                    gt_cached, predictions, workers, class_mode=class_mode
                )
            )
            matched_gt, matched_detections = None, None
        else:
            payload = _build_variant_payload(prepared, variant)
            result = evaluate_tinyperson(
                gt_cached, payload, workers, class_mode=class_mode
            )
            matched_gt, matched_detections = matched_gt_all, matched_det_all
        summary[variant] = {
            "metrics": result,
            "matched_gt_iou_075": matched_gt,
            "matched_detections_iou_075": matched_detections,
        }
        for size_bin, metrics in result.items():
            if size_bin == "protocol":
                continue
            ap25 = f"{metrics['AP@0.25']:.9f}" if metrics["AP@0.25"] is not None else ""
            ap50 = f"{metrics['AP@0.50']:.9f}" if metrics["AP@0.50"] is not None else ""
            ap75 = f"{metrics['AP@0.75']:.9f}" if metrics["AP@0.75"] is not None else ""
            rows.append(f"{variant},{size_bin},{ap25},{ap50},{ap75}")

    atomic_write_text(
        output_dir / "oracle_summary.json",
        orjson.dumps(summary, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode() + "\n",
    )
    atomic_write_text(output_dir / "oracle_by_size.csv", "\n".join(rows) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--class-mode", default="per_class",
                        choices=["per_class", "merge", "primary"])
    args = parser.parse_args()

    summary = run_oracles(
        args.annotations,
        args.predictions,
        args.output_dir,
        workers=args.workers,
        class_mode=args.class_mode,
    )
    compact = {
        variant: {
            "all_AP75": summary[variant]["metrics"]["all"]["AP@0.75"],
            "matched_gt_iou_075": summary[variant]["matched_gt_iou_075"],
        }
        for variant in summary
    }
    print(orjson.dumps(compact, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
