"""TinyPerson benchmark-compatible AP on original-coordinate predictions."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

IOU_THRESHOLDS = (0.25, 0.50, 0.75)
MAX_DETECTIONS = 200
# TinyBenchmark Params uses squared pixel-area bounds, inclusive at both ends.
# These are expressed as sqrt(area) for readability.
SIZE_RANGES: dict[str, tuple[float, float]] = {
    "all": (1.0, 100000.0), "tiny": (1.0, 20.0), "tiny1": (1.0, 8.0),
    "tiny2": (8.0, 12.0), "tiny3": (12.0, 20.0),
    "small": (20.0, 32.0), "reasonable": (32.0, 100000.0),
}


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = map(float, box)
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive COCO bbox: {box}")
    return [x, y, x + width, y + height]


def _sqrt_area(box: list[float]) -> float:
    return math.sqrt(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))


def _in_range(box: list[float], size_range: tuple[float, float]) -> bool:
    # The reference Params uses area < min or area > max, so boundaries are inclusive.
    size = _sqrt_area(box)
    return size_range[0] <= size <= size_range[1]


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
                        predictions_file: str | Path) -> dict[str, Any]:
    """Evaluate predictions/test.json against the original TinyPerson COCO test JSON.

    Predictions must contain an images list with path, boxes (xyxy in source pixels),
    scores and classes, matching the repository's exported test predictions.
    """
    annotation_file, predictions_file = Path(annotation_file), Path(predictions_file)
    gt = _load_gt(annotation_file)
    payload = json.loads(predictions_file.read_text())
    rows = payload.get("images")
    if not isinstance(rows, list):
        raise ValueError("Prediction JSON must contain an images list")

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
            if int(class_id) != 0:
                continue
            box = list(map(float, box))
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                raise ValueError(f"Invalid prediction box for {basename}: {box}")
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append({"box": box, "score": float(score)})
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
    gt_by_image: dict[int, dict[str, list[list[float]]]] = {}
    ann_counts = Counter()
    for image_id in eval_ids:
        positives, ignored = [], []
        for ann in gt["annotations"].get(image_id, []):
            if ann["ignore"]:
                ignored.append(ann["box"])
                ann_counts["ignore"] += 1
            elif ann["uncertain"]:
                ignored.append(ann["box"])
                ann_counts["uncertain"] += 1
            else:
                positives.append(ann["box"])
        gt_by_image[image_id] = {"positive": positives, "ignored": ignored}

    output: dict[str, Any] = {}
    for bin_name, size_range in SIZE_RANGES.items():
        bin_gt, positive_count = {}, 0
        for image_id in eval_ids:
            positives = gt_by_image[image_id]["positive"]
            inside = [b for b in positives if _in_range(b, size_range)]
            outside = [b for b in positives if not _in_range(b, size_range)]
            bin_gt[image_id] = {
                "positive": inside, "outside": outside,
                "ignored": gt_by_image[image_id]["ignored"],
            }
            positive_count += len(inside)

        metrics = {}
        for threshold in IOU_THRESHOLDS:
            scored = sorted(
                ((d["score"], image_id, d["box"])
                 for image_id in eval_ids for d in pred_by_id[image_id]),
                key=lambda item: item[0], reverse=True,
            )
            matched = {image_id: set() for image_id in eval_ids}
            matched_outside = {image_id: set() for image_id in eval_ids}
            tp, fp = [], []
            for _, image_id, box in scored:
                target = bin_gt[image_id]["positive"]
                overlaps = [_iou(box, candidate) for candidate in target]
                candidates = [i for i, overlap in enumerate(overlaps)
                              if overlap >= threshold and i not in matched[image_id]]
                if candidates:
                    best = max(candidates, key=lambda i: overlaps[i])
                    matched[image_id].add(best)
                    tp.append(1)
                    fp.append(0)
                    continue

                outside = bin_gt[image_id]["outside"]
                outside_overlaps = [_iou(box, candidate) for candidate in outside]
                outside_candidates = [
                    i for i, overlap in enumerate(outside_overlaps)
                    if overlap >= threshold and i not in matched_outside[image_id]
                ]
                if outside_candidates:
                    best = max(outside_candidates, key=lambda i: outside_overlaps[i])
                    matched_outside[image_id].add(best)
                    continue

                ignored = bin_gt[image_id]["ignored"]
                # Official TinyBenchmark first accepts an IoU match to an ignored
                # GT; its fallback uses IoD >= the current IoU threshold.
                neutral_ignore = any(
                    _iou(box, other) >= threshold or _iod(box, other) >= threshold
                    for other in ignored
                )
                # COCOeval also ignores unmatched detections outside the selected
                # area range. A matched in-range GT remains a TP even if its
                # predicted box has a different size.
                if neutral_ignore or not _in_range(box, size_range):
                    continue
                tp.append(0)
                fp.append(1)
            metrics[f"AP@{threshold:.2f}"] = _ap(tp, fp, positive_count)
        valid = [value for value in metrics.values() if value is not None]
        output[bin_name] = {
            **metrics,
            "AP_mean_IoU_0.25_0.50_0.75": float(np.mean(valid)) if valid else None,
            "positive_ground_truths": positive_count,
        }

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
        "matching": {
            "normal_positive": "IoU",
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
