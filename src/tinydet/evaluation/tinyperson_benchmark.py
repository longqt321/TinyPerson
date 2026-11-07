"""TinyPerson benchmark-compatible AP on original-coordinate predictions.

Class handling
--------------
TinyPerson annotates two person categories (``sea_person`` = category 1,
``earth_person`` = category 2) which the detector emits as class ids 0 and 1.
``class_mode`` selects how predictions are matched against ground truth:

``per_class`` (default)
    Correct COCO/TOV behaviour. Every prediction is kept and matched only
    against ground truth of its own category. ``result[bin]`` holds the
    class-agnostic aggregate over all categories, and ``result["per_class"]``
    holds the per-category breakdown.
``merge``
    All categories are collapsed into a single class before matching.
``primary``
    Legacy behaviour: predictions whose class id is not 0 are discarded while
    ground truth still contains both categories. This silently drops ~40% of
    the ground truth from the recall denominator and is retained only to
    reproduce numbers published before 2026-11-06. Do not use it for new
    results.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

IOU_THRESHOLDS = (0.25, 0.50, 0.75)
MAX_DETECTIONS = 200
CLASS_MODES = ("per_class", "merge", "primary")
# TinyBenchmark official size ranges (AS = sqrt(w * h)):
# tiny1: [2, 8), tiny2: [8, 12), tiny3: [12, 20], tiny: [2, 20], small: (20, 32], all: [2, 100000]
SIZE_RANGES: dict[str, tuple[float, float, bool, bool]] = {
    "all": (2.0, 100000.0, True, True),
    "tiny": (2.0, 20.0, True, True),
    "tiny1": (2.0, 8.0, True, False),
    "tiny2": (8.0, 12.0, True, False),
    "tiny3": (12.0, 20.0, True, True),
    "small": (20.0, 32.0, False, True),
    "reasonable": (32.0, 100000.0, False, True),
}
_EVAL_CONTEXT: tuple[list[int], dict, list[tuple[float, int, list[float], int]]] | None = None
MERGED_CATEGORY = 0


def _evaluate_size_bin(bin_name: str) -> tuple[str, dict[str, Any]]:
    if _EVAL_CONTEXT is None:
        raise RuntimeError("Evaluator worker context was not initialized")
    eval_ids, gt_by_image, scored = _EVAL_CONTEXT
    size_spec = SIZE_RANGES[bin_name]
    bin_gt: dict[int, dict[str, Any]] = {}
    positive_count = 0
    positive_count_by_cat: dict[int, int] = defaultdict(int)
    for image_id in eval_ids:
        positives = gt_by_image[image_id]["positive"]
        inside: dict[int, list[list[float]]] = {}
        outside: dict[int, list[list[float]]] = {}
        for category, boxes in positives.items():
            inside[category] = [b for b in boxes if _in_range(b, size_spec)]
            outside[category] = [b for b in boxes if not _in_range(b, size_spec)]
            positive_count += len(inside[category])
            positive_count_by_cat[category] += len(inside[category])
        bin_gt[image_id] = {
            "positive": inside, "outside": outside,
            "ignored": gt_by_image[image_id]["ignored"],
        }

    metrics: dict[str, Any] = {}
    per_category: dict[int, dict[str, Any]] = defaultdict(dict)
    num_eval_images = len(eval_ids)
    for threshold in IOU_THRESHOLDS:
        matched: dict[tuple[int, int], set[int]] = defaultdict(set)
        matched_outside: dict[tuple[int, int], set[int]] = defaultdict(set)
        matched_ignored_iou = {image_id: set() for image_id in eval_ids}
        tp, fp = [], []
        tp_by_cat: dict[int, list[int]] = defaultdict(list)
        fp_by_cat: dict[int, list[int]] = defaultdict(list)
        for _, image_id, box, category in scored:
            target = bin_gt[image_id]["positive"].get(category, [])
            key = (image_id, category)
            overlaps = [_iou(box, candidate) for candidate in target]
            candidates = [i for i, overlap in enumerate(overlaps)
                          if overlap >= threshold and i not in matched[key]]
            if candidates:
                best = max(candidates, key=lambda i: overlaps[i])
                matched[key].add(best)
                tp.append(1)
                fp.append(0)
                tp_by_cat[category].append(1)
                fp_by_cat[category].append(0)
                continue

            outside = bin_gt[image_id]["outside"].get(category, [])
            outside_overlaps = [_iou(box, candidate) for candidate in outside]
            outside_candidates = [
                i for i, overlap in enumerate(outside_overlaps)
                if overlap >= threshold and i not in matched_outside[key]
            ]
            if outside_candidates:
                best = max(outside_candidates, key=lambda i: outside_overlaps[i])
                matched_outside[key].add(best)
                continue

            ignored = bin_gt[image_id]["ignored"]
            ignored_iou = [
                i for i, other in enumerate(ignored)
                if _iou(box, other) >= threshold and i not in matched_ignored_iou[image_id]
            ]
            if ignored_iou:
                matched_ignored_iou[image_id].add(ignored_iou[0])
                continue
            neutral_iod = any(_iod(box, other) >= threshold for other in ignored)
            if neutral_iod or not _in_range(box, size_spec):
                continue
            tp.append(0)
            fp.append(1)
            tp_by_cat[category].append(0)
            fp_by_cat[category].append(1)
        metrics[f"AP@{threshold:.2f}"] = _ap(tp, fp, positive_count)
        metrics[f"MR@{threshold:.2f}"] = _mr(tp, fp, positive_count, num_eval_images)
        for category, category_gt in positive_count_by_cat.items():
            per_category[category][f"AP@{threshold:.2f}"] = _ap(
                tp_by_cat[category], fp_by_cat[category], category_gt
            )
            per_category[category][f"MR@{threshold:.2f}"] = _mr(
                tp_by_cat[category], fp_by_cat[category],
                category_gt, num_eval_images,
            )

    valid_ap = [metrics[f"AP@{t:.2f}"] for t in IOU_THRESHOLDS if metrics[f"AP@{t:.2f}"] is not None]
    for category, values in per_category.items():
        valid = [values[f"AP@{t:.2f}"] for t in IOU_THRESHOLDS if values[f"AP@{t:.2f}"] is not None]
        values["AP_mean_IoU_0.25_0.50_0.75"] = float(np.mean(valid)) if valid else None
        values["positive_ground_truths"] = positive_count_by_cat[category]
    return bin_name, {
        **metrics,
        "AP_mean_IoU_0.25_0.50_0.75": float(np.mean(valid_ap)) if valid_ap else None,
        "positive_ground_truths": positive_count,
        "per_category": {str(k): v for k, v in sorted(per_category.items())},
    }


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = map(float, box)
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive COCO bbox: {box}")
    return [x, y, x + width, y + height]


def _sqrt_area(box: list[float]) -> float:
    return math.sqrt(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))


def _in_range(box: list[float], size_spec: tuple[float, float, bool, bool]) -> bool:
    """Check if sqrt(w * h) falls into the half-open or closed interval."""
    low, high, left_closed, right_closed = size_spec
    size = _sqrt_area(box)
    left_ok = (size >= low) if left_closed else (size > low)
    right_ok = (size <= high) if right_closed else (size < high)
    return left_ok and right_ok


def _iou(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(right-left, 0.0) * max(bottom-top, 0.0)
    area_a = max(a[2]-a[0], 0.0) * max(a[3]-a[1], 0.0)
    area_b = max(b[2]-b[0], 0.0) * max(b[3]-b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iod(detection: list[float], ignored: list[float]) -> float:
    """Intersection divided by detection area; used for ignore/uncertain boxes."""
    left, top = max(detection[0], ignored[0]), max(detection[1], ignored[1])
    right, bottom = min(detection[2], ignored[2]), min(detection[3], ignored[3])
    inter = max(right-left, 0.0) * max(bottom-top, 0.0)
    area = max(detection[2]-detection[0], 0.0) * max(detection[3]-detection[1], 0.0)
    return inter / area if area > 0 else 0.0


def _ap(tp: list[int], fp: list[int], positive_count: int) -> float | None:
    if positive_count == 0:
        return None
    if not tp:
        return 0.0
    tp_cum = np.cumsum(np.asarray(tp, dtype=float))
    fp_cum = np.cumsum(np.asarray(fp, dtype=float))
    recall = tp_cum / positive_count
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0.0, 1.0, 101)
    return float(np.mean([
        precision[recall >= point].max() if np.any(recall >= point) else 0.0
        for point in points
    ]))


def _mr(tp: list[int], fp: list[int], positive_count: int, num_images: int) -> float | None:
    """Log-average miss rate across 9 points evenly spaced in log10(FPPI) in [-2, 0]."""
    if positive_count == 0 or num_images == 0:
        return None
    if not tp:
        return 1.0
    tp_cum = np.cumsum(np.asarray(tp, dtype=float))
    fp_cum = np.cumsum(np.asarray(fp, dtype=float))
    rec = tp_cum / positive_count
    fppi = fp_cum / num_images
    miss_rate = 1.0 - rec

    fppi_targets = np.logspace(-2, 0, 9)
    mr_at_fppi = []
    for target in fppi_targets:
        valid = miss_rate[fppi <= target]
        if len(valid) > 0:
            mr_at_fppi.append(valid[-1])
        else:
            mr_at_fppi.append(1.0)
    return float(np.mean(mr_at_fppi))


def _load_gt(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not all(key in data for key in ("images", "annotations", "categories")):
        raise ValueError(f"{path} is not COCO-style JSON")
    images, basename_to_id = {}, {}
    for image in data["images"]:
        image_id = int(image["id"])
        basename = Path(str(image["file_name"])).name
        if basename in basename_to_id:
            raise ValueError(f"Duplicate image basename in annotation JSON: {basename}")
        images[image_id] = {**image, "basename": basename}
        basename_to_id[basename] = image_id
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_counts = Counter()
    dense_ids: set[int] = set()
    for ann in data["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in images:
            raise ValueError(f"Unknown image_id={image_id} in annotation")
        row = {
            "box": _xywh_to_xyxy(ann["bbox"]),
            "category": int(ann.get("category_id", 0)),
            "ignore": bool(ann.get("ignore", False)),
            "uncertain": bool(ann.get("uncertain", False)),
            "dense": bool(ann.get("in_dense_image", False)),
        }
        annotations[image_id].append(row)
        raw_counts[image_id] += 1
        if row["dense"]:
            dense_ids.add(image_id)
    # The release defines a dense image as one containing more than 220 annotations.
    dense_ids.update(image_id for image_id, count in raw_counts.items() if count > 220)
    return {
        "images": images, "basename_to_id": basename_to_id, "annotations": annotations,
        "dense_ids": dense_ids, "source": str(path.resolve()),
    }


def evaluate_tinyperson(annotation_file: str | Path,
                        predictions_file: str | Path, workers: int = 0,
                        class_mode: str = "per_class") -> dict[str, Any]:
    """Evaluate predictions/test.json against the original TinyPerson COCO test JSON.

    Predictions must contain an images list with path, boxes (xyxy in source pixels),
    scores and classes, matching the repository's exported test predictions.

    ``class_mode`` controls how the detector's two output classes are matched
    against the two TinyPerson categories; see the module docstring.
    """
    if workers < 0:
        raise ValueError("workers must be non-negative")
    if class_mode not in CLASS_MODES:
        raise ValueError(f"class_mode must be one of {CLASS_MODES}, got {class_mode!r}")
    annotation_file, predictions_file = Path(annotation_file), Path(predictions_file)
    gt = _load_gt(annotation_file)
    payload = json.loads(predictions_file.read_text())
    rows = payload.get("images")
    if not isinstance(rows, list):
        raise TypeError("Prediction JSON must contain an images list")

    # Map the detector's zero-based class ids onto TinyPerson category ids.
    # The detector emits class 0 for sea_person (category 1) and class 1 for
    # earth_person (category 2), so the offset is +1. If the annotation file
    # does not use category ids 1..N we fall back to identity.
    gt_categories = sorted({ann["category"] for anns in gt["annotations"].values()
                            for ann in anns})
    use_offset = bool(gt_categories) and min(gt_categories) == 1

    # A single-class detector emits only class 0. Under `per_class` that would map
    # every detection onto one TinyPerson category and silently halve the recall
    # denominator, so refuse the combination instead of reporting a wrong number.
    predicted_classes = {
        int(class_id) for row in rows for class_id in row.get("classes", [])
    }
    if (class_mode == "per_class" and len(gt_categories) > 1
            and predicted_classes and predicted_classes <= {0}):
        raise ValueError(
            "Single-class predictions cannot be scored with class_mode='per_class' "
            f"against {len(gt_categories)} TinyPerson categories: every detection would "
            "be matched to one category only. Use class_mode='merge' for a single-class "
            "model, or train a detector with one output per category."
        )

    def _category_of(class_id: int) -> int:
        if class_mode in ("merge", "primary"):
            return MERGED_CATEGORY
        return class_id + 1 if use_offset else class_id

    pred_by_id: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        basename = Path(str(row["path"])).name
        image_id = gt["basename_to_id"].get(basename)
        if image_id is None:
            raise ValueError(
                f"Prediction image {basename!r} is absent from official annotation JSON; "
                "check the selected test split."
            )
        if image_id in pred_by_id:
            raise ValueError(f"Duplicate predictions for image {basename}")
        boxes, scores = row.get("boxes", []), row.get("scores", [])
        classes = row.get("classes", [0] * len(boxes))
        if not (len(boxes) == len(scores) == len(classes)):
            raise ValueError(f"boxes/scores/classes length mismatch for {basename}")
        detections = []
        for box, score, class_id in zip(boxes, scores, classes):
            if class_mode == "primary" and int(class_id) != 0:
                continue
            box = list(map(float, box))
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError(f"Invalid prediction box for {basename}: {box}")
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append({
                "box": box, "score": float(score),
                "category": _category_of(int(class_id)),
            })
        detections.sort(key=lambda d: d["score"], reverse=True)
        pred_by_id[image_id] = detections[:MAX_DETECTIONS]

    expected = set(gt["images"])
    missing = expected - set(pred_by_id)
    if missing:
        sample = [gt["images"][i]["basename"] for i in sorted(missing)[:5]]
        raise ValueError(
            f"Missing predictions for {len(missing)} images in annotation JSON; examples={sample}. "
            "A partial/custom split must not be reported as official test."
        )

    eval_ids = sorted(expected - gt["dense_ids"])
    gt_by_image: dict[int, dict[str, Any]] = {}
    ann_counts = Counter()
    for image_id in eval_ids:
        positives: dict[int, list[list[float]]] = defaultdict(list)
        ignored = []
        for ann in gt["annotations"].get(image_id, []):
            category = MERGED_CATEGORY if class_mode in ("merge", "primary") else ann["category"]
            if ann["ignore"]:
                ignored.append(ann["box"])
                ann_counts["ignore"] += 1
            elif ann["uncertain"]:
                ignored.append(ann["box"])
                ann_counts["uncertain"] += 1
            else:
                positives[category].append(ann["box"])
        gt_by_image[image_id] = {"positive": dict(positives), "ignored": ignored}

    scored = sorted(
        ((d["score"], image_id, d["box"], d["category"])
         for image_id in eval_ids for d in pred_by_id[image_id]),
        key=lambda item: item[0], reverse=True,
    )
    global _EVAL_CONTEXT
    _EVAL_CONTEXT = (eval_ids, gt_by_image, scored)
    worker_count = min(workers or min(len(SIZE_RANGES), os.cpu_count() or 1), len(SIZE_RANGES))
    if (len(scored) >= 5000 and worker_count > 1
            and "fork" in multiprocessing.get_all_start_methods()):
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=multiprocessing.get_context("fork")
        ) as executor:
            output = dict(executor.map(_evaluate_size_bin, SIZE_RANGES))
    else:
        output = dict(map(_evaluate_size_bin, SIZE_RANGES))
    _EVAL_CONTEXT = None

    output["protocol"] = {
        "name": "TinyPerson documented protocol; standalone implementation",
        "annotation_file": gt["source"],
        "predictions_file": str(predictions_file.resolve()),
        "source_images": len(gt["images"]),
        "dense_images_excluded": len(gt["dense_ids"]),
        "images_evaluated": len(eval_ids),
        "max_detections_per_image": MAX_DETECTIONS,
        "iou_thresholds": list(IOU_THRESHOLDS),
        "size_ranges_sqrt_area_px_inclusive": {
            key: list(value) for key, value in SIZE_RANGES.items()
        },
        "ignored_annotations": dict(ann_counts),
        "class_mode": class_mode,
        "categories": gt_categories,
        "matching": {
            "normal_positive": "IoU",
            "class": {
                "per_class": "detections match only same-category ground truth",
                "merge": "all categories collapsed into one class",
                "primary": "non-zero class ids discarded (legacy, incorrect recall)",
            }[class_mode],
            "ignore_uncertain": "IoU match or intersection over detection (IoD)",
            "out_of_size_range": "neutral by IoU for size-specific AP; unmatched detections outside the range are ignored",
        },
        "ap_interpolation": "101-point precision envelope",
        "warning": (
            "Validate this standalone evaluator against the pinned TinyBenchmark reference "
            "before using its scores as publication results."
        ),
    }
    return output
