"""TinyPerson benchmark-compatible AP on original-coordinate predictions.

Single-process vectorized NumPy + TorchVision C++ implementation (zero Numba JIT,
zero fork/multiprocessing) preserving bit-exact parity with the WACV 2020 protocol.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from tinydet.utils.geometry import iod_matrix, iou_matrix

IOU_THRESHOLDS = (0.25, 0.50, 0.75)
MAX_DETECTIONS = 200
CLASS_MODES = ("per_class", "merge", "primary")
SIZE_RANGES: dict[str, tuple[float, float, bool, bool]] = {
    "all": (2.0, 100000.0, True, True),
    "tiny": (2.0, 20.0, True, True),
    "tiny1": (2.0, 8.0, True, False),
    "tiny2": (8.0, 12.0, True, False),
    "tiny3": (12.0, 20.0, True, True),
    "small": (20.0, 32.0, False, True),
    "reasonable": (32.0, 100000.0, False, True),
}
MERGED_CATEGORY = 0


def _boxes_sqrt_area(boxes: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if arr.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    w = np.maximum(arr[:, 2] - arr[:, 0], 0.0)
    h = np.maximum(arr[:, 3] - arr[:, 1], 0.0)
    return np.sqrt(w * h)


def _sizes_in_range(sizes: np.ndarray, size_spec: tuple[float, float, bool, bool]) -> np.ndarray:
    low, high, left_closed, right_closed = size_spec
    left_ok = (sizes >= low) if left_closed else (sizes > low)
    right_ok = (sizes <= high) if right_closed else (sizes < high)
    return left_ok & right_ok


def _precompute_overlap_matrices(
    eval_ids: list[int],
    gt_by_image: dict[int, dict[str, Any]],
    scored: list[tuple[float, int, list[float], int]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, dict[int, np.ndarray]],
    dict[int, dict[int, np.ndarray]],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    """Compute per-image IoU/IoD matrices and sqrt-area vectors once across all 7 size bins."""
    rows_of_image: dict[int, list[int]] = defaultdict(list)
    for row, (_, image_id, _, _) in enumerate(scored):
        rows_of_image[image_id].append(row)
    local_of_row = np.empty(len(scored), dtype=np.intp)
    det_sizes = (
        _boxes_sqrt_area([item[2] for item in scored])
        if scored
        else np.empty(0, dtype=np.float64)
    )
    gt_sizes: dict[int, dict[int, np.ndarray]] = {}
    positive_iou: dict[int, dict[int, np.ndarray]] = {}
    ignored_iou: dict[int, np.ndarray] = {}
    ignored_iod: dict[int, np.ndarray] = {}
    for image_id in eval_ids:
        rows = rows_of_image.get(image_id, [])
        local_of_row[rows] = np.arange(len(rows), dtype=np.intp)
        detections = np.asarray(
            [scored[row][2] for row in rows], dtype=np.float64
        ).reshape(-1, 4)
        positives = gt_by_image[image_id]["positive"]
        positive_iou[image_id] = {
            category: iou_matrix(detections, boxes)
            for category, boxes in positives.items()
        }
        gt_sizes[image_id] = {
            category: _boxes_sqrt_area(boxes)
            for category, boxes in positives.items()
        }
        ignored_boxes = gt_by_image[image_id]["ignored"]
        ignored_iou[image_id] = iou_matrix(detections, ignored_boxes)
        ignored_iod[image_id] = iod_matrix(detections, ignored_boxes)
    return local_of_row, det_sizes, gt_sizes, positive_iou, ignored_iou, ignored_iod


def _match_threshold(
    local_of_row: np.ndarray,
    in_slot: np.ndarray,
    out_slot: np.ndarray,
    ign_slot: np.ndarray,
    det_in_bin: np.ndarray,
    in_mats: list[np.ndarray],
    out_mats: list[np.ndarray],
    ign_iou_mats: list[np.ndarray],
    ign_iod_any: list[np.ndarray],
    threshold: float,
) -> np.ndarray:
    """Bit-exact TinyPerson greedy matcher using vectorized NumPy column masking."""
    n = local_of_row.shape[0]
    status = np.zeros(n, dtype=np.int8)
    if n == 0:
        return status

    in_work = [np.where(m >= threshold, m, -1.0) for m in in_mats]
    out_work = [np.where(m >= threshold, m, -1.0) for m in out_mats]
    ign_work = [m >= threshold for m in ign_iou_mats]

    local_list = local_of_row.tolist()
    in_list = in_slot.tolist()
    out_list = out_slot.tolist()
    ign_list = ign_slot.tolist()
    det_bin_list = det_in_bin.tolist()

    for r in range(n):
        local = local_list[r]

        slot = in_list[r]
        if slot >= 0:
            row = in_work[slot][local]
            best_j = int(row.argmax())
            if row[best_j] >= threshold:
                in_work[slot][:, best_j] = -1.0
                status[r] = 1
                continue

        oslot = out_list[r]
        if oslot >= 0:
            row = out_work[oslot][local]
            best_j = int(row.argmax())
            if row[best_j] >= threshold:
                out_work[oslot][:, best_j] = -1.0
                continue

        islot = ign_list[r]
        if islot >= 0:
            irow = ign_work[islot][local]
            if irow.any():
                hit_j = int(irow.argmax())
                ign_work[islot][:, hit_j] = False
                continue
            if ign_iod_any[islot][local]:
                continue

        if det_bin_list[r]:
            status[r] = -1
    return status


def _evaluate_size_bin(
    bin_name: str,
    eval_ids: list[int],
    gt_sizes: dict[int, dict[int, np.ndarray]],
    scored: list[tuple[float, int, list[float], int]],
    local_of_row: np.ndarray,
    det_sizes: np.ndarray,
    positive_iou: dict[int, dict[int, np.ndarray]],
    ignored_iou: dict[int, np.ndarray],
    ignored_iod: dict[int, np.ndarray],
) -> tuple[str, dict[str, Any]]:
    size_spec = SIZE_RANGES[bin_name]
    det_in_bin = np.ascontiguousarray(_sizes_in_range(det_sizes, size_spec), dtype=np.bool_)
    positive_count = 0
    positive_count_by_cat: dict[int, int] = defaultdict(int)

    in_mats: list[np.ndarray] = []
    out_mats: list[np.ndarray] = []
    ign_iou_mats: list[np.ndarray] = []
    ign_iod_mats: list[np.ndarray] = []
    in_slot_map: dict[tuple[int, int], int] = {}
    out_slot_map: dict[tuple[int, int], int] = {}
    ign_slot_map: dict[int, int] = {}

    for image_id in eval_ids:
        for category, sizes in gt_sizes[image_id].items():
            flags = _sizes_in_range(sizes, size_spec)
            inside_idx = np.flatnonzero(flags)
            outside_idx = np.flatnonzero(~flags)
            n_in = int(inside_idx.size)
            positive_count += n_in
            positive_count_by_cat[category] += n_in
            if n_in:
                in_slot_map[(image_id, category)] = len(in_mats)
                in_mats.append(np.ascontiguousarray(positive_iou[image_id][category][:, inside_idx], dtype=np.float64))
            if outside_idx.size:
                out_slot_map[(image_id, category)] = len(out_mats)
                out_mats.append(np.ascontiguousarray(positive_iou[image_id][category][:, outside_idx], dtype=np.float64))
        n_ign = ignored_iou[image_id].shape[1]
        if n_ign > 0:
            ign_slot_map[image_id] = len(ign_iou_mats)
            ign_iou_mats.append(np.ascontiguousarray(ignored_iou[image_id], dtype=np.float64))
            ign_iod_mats.append(np.ascontiguousarray(ignored_iod[image_id], dtype=np.float64))

    n_scored = len(scored)
    in_slot_arr = np.full(n_scored, -1, dtype=np.int32)
    out_slot_arr = np.full(n_scored, -1, dtype=np.int32)
    ign_slot_arr = np.full(n_scored, -1, dtype=np.int32)
    cat_arr = np.empty(n_scored, dtype=np.int64)
    for r, (_, image_id, _, category) in enumerate(scored):
        key = (image_id, category)
        in_slot_arr[r] = in_slot_map.get(key, -1)
        out_slot_arr[r] = out_slot_map.get(key, -1)
        ign_slot_arr[r] = ign_slot_map.get(image_id, -1)
        cat_arr[r] = category

    local_arr = np.ascontiguousarray(local_of_row, dtype=np.int64)
    metrics: dict[str, Any] = {}
    per_category: dict[int, dict[str, Any]] = defaultdict(dict)
    num_eval_images = len(eval_ids)
    for threshold in IOU_THRESHOLDS:
        thr = float(threshold)
        ign_iod_any = [(m >= thr).any(axis=1) for m in ign_iod_mats]
        status = _match_threshold(
            local_arr,
            in_slot_arr,
            out_slot_arr,
            ign_slot_arr,
            det_in_bin,
            in_mats,
            out_mats,
            ign_iou_mats,
            ign_iod_any,
            thr,
        )
        active = status != 0
        tp = (status[active] == 1).astype(np.float64)
        fp = (status[active] == -1).astype(np.float64)
        metrics[f"AP@{threshold:.2f}"] = _ap(tp, fp, positive_count)
        metrics[f"MR@{threshold:.2f}"] = _mr(tp, fp, positive_count, num_eval_images)
        for category, category_gt in positive_count_by_cat.items():
            cat_active = active & (cat_arr == category)
            tp_c = (status[cat_active] == 1).astype(np.float64)
            fp_c = (status[cat_active] == -1).astype(np.float64)
            per_category[category][f"AP@{threshold:.2f}"] = _ap(tp_c, fp_c, category_gt)
            per_category[category][f"MR@{threshold:.2f}"] = _mr(
                tp_c, fp_c, category_gt, num_eval_images
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
    from tinydet.utils.geometry import xywh_to_xyxy

    return list(xywh_to_xyxy(box, validate=True))


def _ap(tp: Sequence[int] | np.ndarray, fp: Sequence[int] | np.ndarray, positive_count: int) -> float | None:
    if positive_count == 0:
        return None
    if len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(np.asarray(tp, dtype=float))
    fp_cum = np.cumsum(np.asarray(fp, dtype=float))
    recall = tp_cum / positive_count
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0.0, 1.0, 101)
    index = np.searchsorted(recall, points, side="left")
    valid = index < recall.size
    sampled = np.zeros(points.size)
    sampled[valid] = precision[index[valid]]
    return float(sampled.mean())


def _mr(
    tp: Sequence[int] | np.ndarray,
    fp: Sequence[int] | np.ndarray,
    positive_count: int,
    num_images: int,
) -> float | None:
    """Log-average miss rate across 9 points evenly spaced in log10(FPPI) in [-2, 0]."""
    if positive_count == 0 or num_images == 0:
        return None
    if len(tp) == 0:
        return 1.0
    tp_cum = np.cumsum(np.asarray(tp, dtype=float))
    fp_cum = np.cumsum(np.asarray(fp, dtype=float))
    rec = tp_cum / positive_count
    fppi = fp_cum / num_images
    miss_rate = 1.0 - rec

    fppi_targets = np.logspace(-2, 0, 9)
    index = np.searchsorted(fppi, fppi_targets, side="right") - 1
    valid = index >= 0
    sampled = np.ones(fppi_targets.size)
    sampled[valid] = miss_rate[index[valid]]
    return float(sampled.mean())


def subset_annotations_to_split(
    annotation_source: str | Path, data_yaml: str | Path, split: str, destination: str | Path
) -> Path:
    """Restrict an official COCO JSON to the images of one YOLO split."""
    from tinydet.utils.dataset import dataset_images

    annotation_source, data_yaml, destination = Path(annotation_source), Path(data_yaml), Path(destination)
    wanted = {path.name for path in dataset_images(data_yaml, split)}
    if not wanted:
        raise ValueError(f"No images for split {split!r} in {data_yaml}")
    payload = orjson.loads(annotation_source.read_bytes())
    keep_ids = {int(row["id"]) for row in payload["images"] if Path(row["file_name"]).name in wanted}
    images = [row for row in payload["images"] if int(row["id"]) in keep_ids]
    annotations = [row for row in payload["annotations"] if int(row["image_id"]) in keep_ids]
    covered = {Path(row["file_name"]).name for row in images}
    missing = wanted - covered
    if missing:
        raise ValueError(
            f"{len(missing)} {split} image(s) have no annotation in {annotation_source}; "
            f"examples={sorted(missing)[:5]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        orjson.dumps({**payload, "images": images, "annotations": annotations}, option=orjson.OPT_INDENT_2).decode() + "\n"
    )
    return destination


def _load_gt(source: Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict) and "basename_to_id" in source and "dense_ids" in source:
        return source
    if isinstance(source, dict):
        data = source
        source_label = "<in-memory>"
    else:
        path = Path(source)
        data = orjson.loads(path.read_bytes())
        source_label = str(path.resolve())
    if not all(key in data for key in ("images", "annotations", "categories")):
        raise ValueError(f"{source_label} is not COCO-style JSON")
    images, basename_to_id = {}, {}
    for image in data["images"]:
        image_id = int(image["id"])
        basename = Path(str(image["file_name"])).name
        if basename in basename_to_id:
            raise ValueError(f"Duplicate image basename in annotation JSON: {basename}")
        images[image_id] = {**image, "basename": basename}
        basename_to_id[basename] = image_id
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    raw_counts: Counter[int] = Counter()
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
    dense_ids.update(image_id for image_id, count in raw_counts.items() if count > 220)
    return {
        "images": images, "basename_to_id": basename_to_id, "annotations": annotations,
        "dense_ids": dense_ids, "source": source_label,
    }


def evaluate_tinyperson(
    annotation_file: str | Path | dict[str, Any],
    predictions_file: str | Path | dict[str, Any],
    workers: int = 0,
    class_mode: str = "per_class",
) -> dict[str, Any]:
    """Evaluate predictions against the original TinyPerson COCO JSON (or in-memory dicts)."""
    if workers < 0:
        raise ValueError("workers must be non-negative")
    if class_mode not in CLASS_MODES:
        raise ValueError(f"class_mode must be one of {CLASS_MODES}, got {class_mode!r}")
    gt = _load_gt(annotation_file if isinstance(annotation_file, dict) else Path(annotation_file))
    if isinstance(predictions_file, dict):
        payload = predictions_file
        pred_source = "<in-memory>"
    else:
        pred_path = Path(predictions_file)
        payload = orjson.loads(pred_path.read_bytes())
        pred_source = str(pred_path.resolve())
    rows = payload.get("images")
    if not isinstance(rows, list):
        raise TypeError("Prediction JSON must contain an images list")

    gt_categories = sorted({ann["category"] for anns in gt["annotations"].values()
                            for ann in anns})
    use_offset = bool(gt_categories) and min(gt_categories) == 1

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
        detections: list[dict[str, Any]] = []
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
    ann_counts: Counter[str] = Counter()
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
    local_of_row, det_sizes, gt_sizes, positive_iou, ignored_iou, ignored_iod = (
        _precompute_overlap_matrices(eval_ids, gt_by_image, scored)
    )
    output = dict(
        _evaluate_size_bin(
            bin_name,
            eval_ids,
            gt_sizes,
            scored,
            local_of_row,
            det_sizes,
            positive_iou,
            ignored_iou,
            ignored_iod,
        )
        for bin_name in SIZE_RANGES
    )

    output["protocol"] = {
        "name": "TinyPerson documented protocol; standalone implementation",
        "annotation_file": gt["source"],
        "predictions_file": pred_source,
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
