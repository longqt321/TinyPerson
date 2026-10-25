"""Deterministic sample-level error decomposition and metric cross-checks.

The canonical records store source-coordinate boxes, annotations, raw
postprocessed predictions, and one primary matching outcome per entity. The
official TinyPerson evaluator remains the metric source of truth; diagnostic
AP is derived from the same matched records and cross-checked against it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tinydet.evaluation.tinyperson_benchmark import (
    IOU_THRESHOLDS,
    MAX_DETECTIONS,
    SIZE_RANGES,
    _ap,
    _in_range,
    _iod,
    _iou,
)
from tinydet.utils.experiment import git_info

SCHEMA_VERSION = 1
IMPLEMENTATION_VERSION = "tinydet-error-analysis-v1"
DIAGNOSTIC_SIZE_BINS: dict[str, tuple[float, float | None]] = {
    "tiny1": (1.0, 8.0),
    "tiny2": (8.0, 12.0),
    "tiny3": (12.0, 20.0),
    "small": (20.0, 32.0),
    "reasonable": (32.0, None),
}
DENSITY_BINS = (
    ("single", 1, 1),
    ("sparse", 2, 10),
    ("crowded", 11, 50),
    ("very_crowded", 51, None),
)


@dataclass(frozen=True)
class AnalysisConfig:
    """All rules that can change matching, error categories, or denominators."""

    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS
    primary_iou: float = 0.50
    localization_iou: float = 0.10
    score_threshold: float = 0.001
    max_detections: int = MAX_DETECTIONS
    boundary_fraction: float = 0.02

    def __post_init__(self) -> None:
        if not self.iou_thresholds or any(not 0.0 < value <= 1.0 for value in self.iou_thresholds):
            raise ValueError("iou_thresholds must be nonempty and inside (0, 1]")
        if self.primary_iou not in self.iou_thresholds:
            raise ValueError("primary_iou must be one of iou_thresholds")
        if not 0.0 <= self.localization_iou < self.primary_iou:
            raise ValueError("localization_iou must be below primary_iou")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be inside [0, 1]")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if not 0.0 <= self.boundary_fraction < 0.5:
            raise ValueError("boundary_fraction must be inside [0, 0.5)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_thresholds": list(self.iou_thresholds),
            "primary_iou": self.primary_iou,
            "localization_iou": self.localization_iou,
            "score_threshold": self.score_threshold,
            "max_detections": self.max_detections,
            "boundary_fraction": self.boundary_fraction,
            "diagnostic_size_bins_sqrt_area_px": {
                name: [low, high] for name, (low, high) in DIAGNOSTIC_SIZE_BINS.items()
            },
            "density_bins_positive_objects": {
                name: [low, high] for name, low, high in DENSITY_BINS
            },
        }


@dataclass(frozen=True)
class GroundTruth:
    id: int
    box: tuple[float, float, float, float]
    label: int
    ignore: bool
    uncertain: bool
    dense: bool
    raw: dict[str, Any] = field(compare=False, repr=False)


@dataclass(frozen=True)
class Detection:
    source_index: int
    box: tuple[float, float, float, float]
    score: float
    label: int


@dataclass
class AnnotationSet:
    images: dict[int, dict[str, Any]]
    basename_to_id: dict[str, int]
    annotations: dict[int, list[GroundTruth]]
    dense_ids: set[int]
    raw: dict[str, Any]
    source: str
    sha256: str
    issues: list[dict[str, Any]]


@dataclass
class ManifestSet:
    expected_ids: list[int]
    rows: list[dict[str, Any]]
    source: str
    sha256: str
    issues: list[dict[str, Any]]


@dataclass
class PredictionSet:
    rows: dict[str, dict[str, Any]]
    source: str
    sha256: str
    metadata: dict[str, Any] | None = None

    @property
    def payload(self) -> dict[str, Any]:
        return {"images": list(self.rows.values())}


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    samples: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _issue(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **extra}


def _xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"bbox must have four values: {box}")
    x, y, width, height = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError(f"bbox values must be finite: {box}")
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive COCO bbox: {box}")
    return x, y, x + width, y + height


def _xyxy(box: list[float] | tuple[float, ...], context: str) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"{context}: box must have four values: {box}")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{context}: non-finite box: {box}")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError(f"{context}: non-positive box area: {box}")
    return values


def _diagnostic_size_bin(box: tuple[float, float, float, float]) -> str:
    size = math.sqrt(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0))
    for name, (low, high) in DIAGNOSTIC_SIZE_BINS.items():
        if size >= low and (high is None or size < high):
            return name
    return "below_1px"


def _density_bin(count: int) -> str:
    for name, low, high in DENSITY_BINS:
        if count >= low and (high is None or count <= high):
            return name
    return "none"


def _boundary(box: tuple[float, float, float, float], width: int, height: int,
              fraction: float) -> bool:
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return (x <= width * fraction or x >= width * (1 - fraction)
            or y <= height * fraction or y >= height * (1 - fraction))


def _label_map(raw: dict[str, Any], issues: list[dict[str, Any]]) -> dict[int, int]:
    categories = raw.get("categories", [])
    labels: dict[int, int] = {}
    for category in categories:
        category_id = int(category["id"])
        name = str(category.get("name", "")).lower()
        if "person" in name:
            labels[category_id] = 0
        else:
            labels[category_id] = category_id
            issues.append(_issue(
                "non_person_category",
                f"Category {name!r} is not merged into the person class",
                category_id=category_id,
            ))
    return labels


def load_annotations(path: str | Path) -> AnnotationSet:
    source = Path(path).expanduser().resolve()
    raw = json.loads(source.read_text())
    if not isinstance(raw, dict) or not all(
        key in raw for key in ("images", "annotations", "categories")
    ):
        raise ValueError(f"{source} is not COCO-style JSON")
    issues: list[dict[str, Any]] = []
    images: dict[int, dict[str, Any]] = {}
    basename_to_id: dict[str, int] = {}
    for image in raw["images"]:
        image_id = int(image["id"])
        basename = Path(str(image["file_name"])).name
        width, height = int(image["width"]), int(image["height"])
        if image_id in images:
            issues.append(_issue("duplicate_image_id", f"Duplicate image id {image_id}"))
            continue
        if basename in basename_to_id:
            issues.append(_issue(
                "duplicate_image_basename",
                f"Duplicate image basename {basename!r}; predictions cannot be mapped uniquely",
            ))
            continue
        if width <= 0 or height <= 0:
            issues.append(_issue(
                "invalid_image_dimensions",
                f"Image {basename!r} has non-positive dimensions {width}x{height}",
            ))
            continue
        images[image_id] = {
            "id": image_id,
            "basename": basename,
            "path": str(image["file_name"]),
            "width": width,
            "height": height,
        }
        basename_to_id[basename] = image_id

    labels = _label_map(raw, issues)
    annotations: dict[int, list[GroundTruth]] = defaultdict(list)
    raw_counts: Counter[int] = Counter()
    dense_ids: set[int] = set()
    duplicate_keys: Counter[tuple[int, int, tuple[float, ...]]] = Counter()
    for ann in raw["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in images:
            issues.append(_issue(
                "annotation_unknown_image",
                f"Annotation {ann.get('id')} references unknown image_id={image_id}",
            ))
            continue
        raw_counts[image_id] += 1
        try:
            box = _xywh_to_xyxy(ann["bbox"])
        except (KeyError, TypeError, ValueError) as error:
            issues.append(_issue(
                "invalid_annotation_box",
                f"Image {images[image_id]['basename']}: {error}",
                image_id=image_id,
                annotation_id=ann.get("id"),
            ))
            continue
        category_id = int(ann["category_id"])
        if category_id not in labels:
            issues.append(_issue(
                "unknown_annotation_category",
                f"Image {images[image_id]['basename']}: category {category_id} is absent",
                image_id=image_id,
                annotation_id=ann.get("id"),
            ))
            continue
        label = labels[category_id]
        ignore = bool(ann.get("ignore", False))
        uncertain = bool(ann.get("uncertain", False))
        dense = bool(ann.get("in_dense_image", False))
        if dense:
            dense_ids.add(image_id)
        key = (image_id, label, tuple(round(value, 6) for value in box))
        duplicate_keys[key] += 1
        annotations[image_id].append(GroundTruth(
            id=int(ann.get("id", len(annotations[image_id]))),
            box=box,
            label=label,
            ignore=ignore,
            uncertain=uncertain,
            dense=dense,
            raw=ann,
        ))
    for (image_id, label, box), count in duplicate_keys.items():
        if count > 1:
            issues.append(_issue(
                "duplicate_annotation",
                f"Image {images[image_id]['basename']}: duplicate label {label} box {list(box)}",
                image_id=image_id,
                count=count,
            ))
    dense_ids.update(image_id for image_id, count in raw_counts.items() if count > 220)
    return AnnotationSet(
        images=images,
        basename_to_id=basename_to_id,
        annotations=dict(annotations),
        dense_ids=dense_ids,
        raw=raw,
        source=str(source),
        sha256=sha256(source),
        issues=issues,
    )


def load_manifest(path: str | Path, split: str, annotations: AnnotationSet) -> ManifestSet:
    source = Path(path).expanduser().resolve()
    raw = json.loads(source.read_text())
    rows = raw.get("images_manifest")
    if not isinstance(rows, list):
        raise TypeError(f"{source} does not contain images_manifest")
    issues: list[dict[str, Any]] = []
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"Manifest has no images for split {split!r}")
    expected_ids: list[int] = []
    seen: set[int] = set()
    for row in selected:
        image_id = int(row["image_id"])
        if image_id in seen:
            issues.append(_issue("duplicate_manifest_image", f"Duplicate manifest image {image_id}"))
            continue
        seen.add(image_id)
        annotation_image = annotations.images.get(image_id)
        if annotation_image is None:
            issues.append(_issue(
                "manifest_image_absent_from_annotations",
                f"Manifest image_id={image_id} is absent from COCO annotations",
            ))
            continue
        if Path(str(row["source_file_name"])).name != annotation_image["basename"]:
            issues.append(_issue(
                "manifest_basename_mismatch",
                f"image_id={image_id}: manifest {row['source_file_name']!r} disagrees with "
                f"annotations {annotation_image['basename']!r}",
            ))
        expected = (int(row["source_width"]), int(row["source_height"]))
        actual = (annotation_image["width"], annotation_image["height"])
        if expected != actual:
            issues.append(_issue(
                "manifest_dimension_mismatch",
                f"image_id={image_id}: manifest {expected} disagrees with annotations {actual}",
            ))
        expected_ids.append(image_id)
    return ManifestSet(
        expected_ids=expected_ids,
        rows=selected,
        source=str(source),
        sha256=sha256(source),
        issues=issues,
    )


def _predictions_from_pt(path: Path) -> dict[str, dict[str, Any]]:
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator

    payload = torch.load(path, map_location="cpu", weights_only=True)
    validator = DetectionValidator(save_dir=path.parent, args={"plots": False})
    validator.metrics.names = payload.get("names", {0: "person"})
    rows: dict[str, dict[str, Any]] = {}
    for batch, pred in payload["samples"]:
        scaled = validator.scale_preds(pred, batch)
        row = {
            "path": str(batch["im_file"]),
            "boxes": scaled["bboxes"].tolist(),
            "scores": pred["conf"].tolist(),
            "classes": pred["cls"].tolist(),
        }
        basename = Path(row["path"]).name
        if basename in rows:
            raise ValueError(f"Duplicate prediction image {basename}")
        rows[basename] = row
    return rows


def load_predictions(path: str | Path) -> PredictionSet:
    source = Path(path).expanduser().resolve()
    if source.suffix == ".pt":
        rows = _predictions_from_pt(source)
        return PredictionSet(rows=rows, source=str(source), sha256=sha256(source))
    payload = json.loads(source.read_text())
    rows = payload.get("images")
    if not isinstance(rows, list):
        raise TypeError("Prediction JSON must contain an images list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        basename = Path(str(row["path"])).name
        if basename in indexed:
            raise ValueError(f"Duplicate prediction image {basename}")
        boxes = row.get("boxes", [])
        scores = row.get("scores", [])
        classes = row.get("classes", [0] * len(boxes))
        if not (len(boxes) == len(scores) == len(classes)):
            raise ValueError(f"boxes/scores/classes length mismatch for {basename}")
        indexed[basename] = {**row, "path": str(row["path"]), "classes": classes}
    metadata = payload.get("inference")
    return PredictionSet(
        rows=indexed,
        source=str(source),
        sha256=sha256(source),
        metadata=metadata if isinstance(metadata, dict) else None,
    )


def _match_image(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    config: AnalysisConfig,
    size_bin: str = "all",
    iou_threshold: float | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Reproduce TinyPerson per-image greedy matching and classify each outcome."""
    threshold = config.primary_iou if iou_threshold is None else iou_threshold
    size_range = SIZE_RANGES[size_bin]
    positives = [
        (index, gt) for index, gt in enumerate(ground_truths)
        if not (gt.ignore or gt.uncertain) and _in_range(list(gt.box), size_range)
    ]
    outside = [
        (index, gt) for index, gt in enumerate(ground_truths)
        if not (gt.ignore or gt.uncertain) and not _in_range(list(gt.box), size_range)
    ]
    ignored = [gt for gt in ground_truths if gt.ignore or gt.uncertain]
    matched: set[int] = set()
    matched_outside: set[int] = set()
    matched_ignored: set[int] = set()
    outcomes: dict[int, dict[str, Any]] = {}
    for detection in detections:
        same_label = [(index, gt) for index, gt in positives if gt.label == detection.label]
        overlaps = [_iou(list(detection.box), list(gt.box)) for _, gt in same_label]
        candidates = [
            position for position, overlap in enumerate(overlaps)
            if overlap >= threshold and same_label[position][0] not in matched
        ]
        if candidates:
            best = max(candidates, key=lambda position: overlaps[position])
            gt_index = same_label[best][0]
            matched.add(gt_index)
            outcomes[detection.source_index] = {
                "status": "true_positive",
                "subtype": "matched",
                "gt_index": gt_index,
                "gt_id": ground_truths[gt_index].id,
                "iou": round(overlaps[best], 6),
            }
            continue

        outside_overlaps = [(_iou(list(detection.box), list(gt.box)), index)
                            for index, gt in outside]
        outside_candidates = [item for item in outside_overlaps
                              if item[0] >= threshold and item[1] not in matched_outside]
        if outside_candidates:
            overlap, gt_index = max(outside_candidates)
            matched_outside.add(gt_index)
            outcomes[detection.source_index] = {
                "status": "neutral",
                "subtype": "outside_size_range",
                "gt_index": gt_index,
                "gt_id": ground_truths[gt_index].id,
                "iou": round(overlap, 6),
            }
            continue

        ignored_iou = [(index, _iou(list(detection.box), list(gt.box)))
                       for index, gt in enumerate(ignored)]
        ignored_candidates = [item for item in ignored_iou
                              if item[1] >= threshold and item[0] not in matched_ignored]
        if ignored_candidates:
            gt_index, overlap = max(ignored_candidates, key=lambda item: item[1])
            matched_ignored.add(gt_index)
            outcomes[detection.source_index] = {
                "status": "neutral",
                "subtype": "ignored_annotation",
                "gt_index": gt_index,
                "iou": round(overlap, 6),
            }
            continue
        if any(_iod(list(detection.box), list(gt.box)) >= threshold for gt in ignored):
            outcomes[detection.source_index] = {
                "status": "neutral",
                "subtype": "ignored_annotation",
                "iou": 0.0,
            }
            continue
        if not _in_range(list(detection.box), size_range):
            outcomes[detection.source_index] = {
                "status": "neutral",
                "subtype": "out_of_size_range",
                "iou": 0.0,
            }
            continue

        best_iou = max(overlaps, default=0.0)
        duplicate = any(overlap >= threshold for overlap in overlaps)
        if duplicate:
            subtype = "duplicate"
        elif best_iou >= config.localization_iou:
            subtype = "localization"
        else:
            subtype = "background"
        outcomes[detection.source_index] = {
            "status": "false_positive",
            "subtype": subtype,
            "iou": round(best_iou, 6),
        }

    gt_outcomes: dict[int, dict[str, Any]] = {}
    for index, gt in enumerate(ground_truths):
        if gt.ignore or gt.uncertain:
            gt_outcomes[index] = {"status": "ignored", "iou": 0.0}
            continue
        if (index, gt) in outside:
            gt_outcomes[index] = {"status": "outside_size_range", "iou": 0.0}
            continue
        best_iou = max(
            (_iou(list(detection.box), list(gt.box)) for detection in detections
             if detection.label == gt.label),
            default=0.0,
        )
        gt_outcomes[index] = {
            "status": "matched" if index in matched else "missed",
            "iou": round(best_iou, 6),
        }
    return [outcomes[detection.source_index] for detection in detections], gt_outcomes


def _record_error(record: dict[str, Any], category: str, subtype: str, *,
                  box: tuple[float, float, float, float], iou: float,
                  score: float | None = None, object_id: int | None = None,
                  gt_id: int | None = None,
                  gt_box: tuple[float, float, float, float] | None = None,
                  boundary: bool | None = None,
                  ) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": record["image"]["id"],
        "image": record["image"]["basename"],
        "category": category,
        "subtype": subtype,
        "object_id": object_id,
        "gt_id": gt_id,
        "box": list(box),
        "gt_box": list(gt_box) if gt_box is not None else None,
        "score": score,
        "iou": iou,
        "sqrt_area_px": round(math.sqrt((box[2] - box[0]) * (box[3] - box[1])), 4),
        "size_bin": _diagnostic_size_bin(box),
        "density_bin": record["image"]["density_bin"],
        "boundary": boundary,
    }


def _validated_detections(row: dict[str, Any], config: AnalysisConfig,
                          issues: list[dict[str, Any]], basename: str) -> list[Detection]:
    detections: list[Detection] = []
    boxes, scores, classes = row["boxes"], row["scores"], row["classes"]
    for source_index, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        try:
            label = int(class_id)
            box_value = _xyxy(box, f"prediction {source_index}")
            score_value = float(score)
            if not math.isfinite(score_value) or not 0.0 <= score_value <= 1.0:
                raise ValueError(f"confidence must be finite and inside [0, 1]: {score}")
        except (TypeError, ValueError) as error:
            issues.append(_issue(
                "invalid_prediction",
                f"Image {basename}: {error}",
                image=basename,
                prediction_index=source_index,
            ))
            continue
        if label != 0:
            issues.append(_issue(
                "non_person_prediction",
                f"Image {basename}: prediction class {label} is outside the evaluated person class",
                image=basename,
                prediction_index=source_index,
            ))
            continue
        if score_value < config.score_threshold:
            continue
        detections.append(Detection(source_index, box_value, score_value, label))
    detections.sort(key=lambda detection: detection.score, reverse=True)
    if len(detections) > config.max_detections:
        issues.append(_issue(
            "predictions_truncated",
            f"Image {basename}: {len(detections)} predictions exceed max_detections; "
            f"keeping {config.max_detections}",
            image=basename,
        ))
        detections = detections[:config.max_detections]
    return detections


def _ap_from_outcomes(
    entries: list[tuple[float, int, int, str]],
    positive_count: int,
) -> float | None:
    ordered = sorted(entries, key=lambda item: item[0], reverse=True)
    tp = [1 if item[3] == "true_positive" else 0 for item in ordered]
    fp = [1 if item[3] == "false_positive" else 0 for item in ordered]
    return _ap(tp, fp, positive_count)


def _crosscheck_reference(
    annotations: AnnotationSet,
    predictions: PredictionSet,
    expected_ids: list[int],
) -> dict[str, Any]:
    import tempfile

    from tinydet.evaluation.tinyperson_benchmark import evaluate_tinyperson

    expected = set(expected_ids)
    image_ids = {annotations.images[image_id]["basename"]: image_id for image_id in expected}
    predicted = {
        basename: row for basename, row in predictions.rows.items()
        if basename in image_ids
    }
    try:
        with tempfile.TemporaryDirectory(prefix="tinydet-error-crosscheck-") as temporary:
            root = Path(temporary)
            annotation_file = root / "annotations.json"
            prediction_file = root / "predictions.json"
            annotation_file.write_text(json.dumps({
                "images": [image for image in annotations.raw["images"]
                           if int(image["id"]) in expected],
                "annotations": [ann for ann in annotations.raw["annotations"]
                                if int(ann["image_id"]) in expected],
                "categories": annotations.raw["categories"],
            }))
            prediction_file.write_text(json.dumps({"images": list(predicted.values())}))
            reference = evaluate_tinyperson(annotation_file, prediction_file, workers=0)
    except (KeyError, TypeError, ValueError, OSError) as error:
        return {"available": False, "reason": f"reference evaluator failed: {error}"}
    return {"available": True, "reference": reference}


def _compare_samples(
    current: list[dict[str, Any]], baseline_path: str | Path,
    dataset_key: dict[str, Any],
) -> dict[str, Any]:
    baseline = [json.loads(line) for line in
                Path(baseline_path).read_text().splitlines() if line.strip()]
    baseline_by_image = {record["image"]["basename"]: record for record in baseline}
    current_by_image = {record["image"]["basename"]: record for record in current}
    if not all(record.get("dataset_key") == dataset_key for record in baseline):
        return {"available": False, "reason": "baseline dataset provenance differs"}
    shared = sorted(set(current_by_image) & set(baseline_by_image))
    corrected, regressed = [], []
    deltas = Counter()
    for basename in shared:
        before = baseline_by_image[basename]["summary"]
        after = current_by_image[basename]["summary"]
        error_before = before["misses"] + before["false_positives"]
        error_after = after["misses"] + after["false_positives"]
        if error_before > 0 and error_after == 0:
            corrected.append(basename)
        if error_before == 0 and error_after > 0:
            regressed.append(basename)
        deltas["misses"] += after["misses"] - before["misses"]
        deltas["false_positives"] += after["false_positives"] - before["false_positives"]
    return {
        "available": True,
        "baseline_samples": str(Path(baseline_path).resolve()),
        "shared_images": len(shared),
        "corrected_images": len(corrected),
        "regressed_images": len(regressed),
        "unchanged_images": len(shared) - len(corrected) - len(regressed),
        "corrected_examples": corrected[:20],
        "regressed_examples": regressed[:20],
        "count_deltas": dict(deltas),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for record in records:
        counts["true_positives"] += record["summary"]["true_positives"]
        counts["false_positives"] += record["summary"]["false_positives"]
        counts["duplicates"] += record["summary"]["duplicates"]
        counts["localization_errors"] += record["summary"]["localization_errors"]
        counts["background_false_positives"] += record["summary"]["background_false_positives"]
        counts["misses"] += record["summary"]["misses"]
        counts["neutral_predictions"] += record["summary"]["neutral_predictions"]

    by_size: dict[str, dict[str, Any]] = {}
    by_density: dict[str, dict[str, Any]] = {}
    by_boundary: dict[str, dict[str, Any]] = {}
    by_confidence: dict[str, dict[str, Any]] = {}
    by_class: dict[str, dict[str, Any]] = {}
    positive_gt = 0
    for record in records:
        density = record["image"]["density_bin"]
        density_row = by_density.setdefault(density, {
            "images": 0, "positive_ground_truths": 0, "matched": 0, "misses": 0,
            "false_positives": 0, "duplicates": 0, "localization_errors": 0,
            "background_false_positives": 0,
        })
        density_row["images"] += 1
        density_row["false_positives"] += record["summary"]["false_positives"]
        density_row["duplicates"] += record["summary"]["duplicates"]
        density_row["localization_errors"] += record["summary"]["localization_errors"]
        density_row["background_false_positives"] += record["summary"][
            "background_false_positives"]
        for gt in record["ground_truth"]:
            if gt["status"] == "ignored":
                continue
            if gt["outcome"]["status"] == "outside_size_range":
                continue
            positive_gt += 1
            size_name = gt["size_bin"]
            size_row = by_size.setdefault(size_name, {
                "positive_ground_truths": 0, "matched": 0, "misses": 0,
            })
            size_row["positive_ground_truths"] += 1
            if gt["outcome"]["status"] == "matched":
                size_row["matched"] += 1
            else:
                size_row["misses"] += 1
            density_row["positive_ground_truths"] += 1
            if gt["outcome"]["status"] == "matched":
                density_row["matched"] += 1
            else:
                density_row["misses"] += 1
            boundary_name = "border" if gt["boundary"] else "interior"
            boundary_row = by_boundary.setdefault(boundary_name, {
                "positive_ground_truths": 0, "matched": 0, "misses": 0,
            })
            boundary_row["positive_ground_truths"] += 1
            boundary_row["matched" if gt["outcome"]["status"] == "matched" else "misses"] += 1
            class_row = by_class.setdefault(str(gt["label"]), {
                "positive_ground_truths": 0, "matched": 0, "misses": 0,
            })
            class_row["positive_ground_truths"] += 1
            class_row["matched" if gt["outcome"]["status"] == "matched" else "misses"] += 1
        for prediction in record["predictions"]:
            for low, high in ((0.0, 0.01), (0.01, 0.05), (0.05, 0.10),
                              (0.10, 0.25), (0.25, 0.50), (0.50, 1.0)):
                in_bucket = low <= prediction["score"] and (
                    prediction["score"] <= high if high == 1.0
                    else prediction["score"] < high
                )
                if in_bucket:
                    bucket = f"[{low:.2f},{high:.2f}{']' if high == 1.0 else ')'}"
                    row = by_confidence.setdefault(bucket, Counter())
                    row[prediction["outcome"]["status"]] += 1
                    row[f"subtype:{prediction['outcome']['subtype']}"] += 1
                    break

    for row in by_size.values():
        row["recall"] = _rate(row["matched"], row["positive_ground_truths"])
        row["miss_rate"] = _rate(row["misses"], row["positive_ground_truths"])
    for row in by_density.values():
        row["recall"] = _rate(row["matched"], row["positive_ground_truths"])
        row["false_positives_per_image"] = _rate(row["false_positives"], row["images"])
    for row in by_boundary.values():
        row["recall"] = _rate(row["matched"], row["positive_ground_truths"])
    for row in by_class.values():
        row["recall"] = _rate(row["matched"], row["positive_ground_truths"])
    for bucket, row in by_confidence.items():
        row.update({key: int(value) for key, value in row.items()})
        total = row.get("true_positive", 0) + row.get("false_positive", 0) + row.get(
            "neutral", 0)
        row["predictions"] = total
        row["precision"] = _rate(row.get("true_positive", 0),
                                 row.get("true_positive", 0) + row.get("false_positive", 0))

    operating_points = []
    all_predictions = [prediction for record in records for prediction in record["predictions"]]
    for threshold in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
        selected = [prediction for prediction in all_predictions
                    if prediction["score"] >= threshold]
        tp = sum(prediction["outcome"]["status"] == "true_positive" for prediction in selected)
        fp = sum(prediction["outcome"]["status"] == "false_positive" for prediction in selected)
        operating_points.append({
            "score_threshold": threshold,
            "predictions": len(selected),
            "true_positives": tp,
            "false_positives": fp,
            "recall": _rate(tp, positive_gt),
            "precision": _rate(tp, tp + fp),
        })

    priorities = []
    for size_name, row in by_size.items():
        priorities.append({
            "category": "miss", "group": f"size_bin:{size_name}",
            "count": row["misses"], "denominator": row["positive_ground_truths"],
            "rate": row["miss_rate"], "metric_impact": "recall",
        })
    subtype_keys = {
        "duplicate": "duplicates",
        "localization": "localization_errors",
        "background": "background_false_positives",
    }
    for subtype, label in subtype_keys.items():
        count = counts[label]
        priorities.append({
            "category": "false_positive", "group": subtype, "count": counts[label],
            "denominator": counts["false_positives"],
            "rate": _rate(count, counts["false_positives"]),
            "metric_impact": "precision",
        })
    for density_name, row in by_density.items():
        priorities.append({
            "category": "miss", "group": f"density:{density_name}",
            "count": row["misses"], "denominator": row["positive_ground_truths"],
            "rate": (1 - row["recall"]) if row["recall"] is not None else None,
            "metric_impact": "recall",
        })
    priorities.sort(key=lambda item: (item["count"], item["rate"] or 0.0), reverse=True)
    return {
        "record_source": "samples.jsonl",
        "error_source": "errors.jsonl",
        "counts": dict(counts),
        "positive_ground_truths": positive_gt,
        "by_size_bin": by_size,
        "by_density": by_density,
        "by_boundary": by_boundary,
        "by_confidence": dict(sorted(by_confidence.items())),
        "by_class": by_class,
        "confidence_operating_points": operating_points,
        "priorities": priorities[:12],
    }
def analyze_files(
    annotations_path: str | Path,
    predictions_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    split: str = "val",
    config: AnalysisConfig | None = None,
    limit: int = 0,
    baseline_samples: str | Path | None = None,
    run: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> AnalysisResult:
    config = config or AnalysisConfig()
    annotations = load_annotations(annotations_path)
    manifest = (load_manifest(manifest_path, split, annotations)
                if manifest_path is not None else None)
    predictions = load_predictions(predictions_path)
    if limit < 0:
        raise ValueError("limit must be non-negative")

    issues = [*annotations.issues, *(manifest.issues if manifest else [])]
    expected_ids = (manifest.expected_ids if manifest is not None
                    else sorted(annotations.images))
    expected_ids = sorted(expected_ids, key=lambda image_id: annotations.images[image_id]["basename"])
    if limit:
        expected_ids = expected_ids[:limit]
    expected = {
        annotations.images[image_id]["basename"]: image_id for image_id in expected_ids
    }
    for basename, image_id in expected.items():
        if basename not in predictions.rows:
            issues.append(_issue(
                "missing_predictions", f"No predictions for image {basename}",
                image_id=image_id,
            ))
    for basename in sorted(set(predictions.rows) - set(expected)):
        issues.append(_issue(
            "unexpected_predictions",
            f"Predictions for {basename} are outside the selected split or limit",
            image=basename,
        ))

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    ap_entries: dict[str, dict[float, list[tuple[float, int, int, str]]]] = {
        name: {threshold: [] for threshold in config.iou_thresholds} for name in SIZE_RANGES
    }
    positive_counts = {name: 0 for name in SIZE_RANGES}
    excluded_dense = 0
    for basename, image_id in expected.items():
        if image_id in annotations.dense_ids:
            excluded_dense += 1
            continue
        image = annotations.images[image_id]
        row = predictions.rows.get(basename)
        if row is None:
            continue
        detections = _validated_detections(row, config, issues, basename)
        ground_truths = annotations.annotations.get(image_id, [])
        positive_objects = [gt for gt in ground_truths if not (gt.ignore or gt.uncertain)]
        density_name = _density_bin(len(positive_objects))
        boundary = _boundary(
            (0.0, 0.0, float(image["width"]), float(image["height"])),
            image["width"], image["height"], 0.0,
        )
        del boundary
        primary_outcomes, gt_outcomes = _match_image(
            detections, ground_truths, config, "all", config.primary_iou,
        )
        image_boundary = any(
            _boundary(gt.box, image["width"], image["height"], config.boundary_fraction)
            for gt in positive_objects
        )
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "dataset_key": {
                "annotations_sha256": annotations.sha256,
                "manifest_sha256": manifest.sha256 if manifest else None,
                "split": split,
            },
            "image": {
                "id": image_id,
                "basename": basename,
                "path": image["path"],
                "width": image["width"],
                "height": image["height"],
                "positive_objects": len(positive_objects),
                "density_bin": density_name,
                "boundary": image_boundary,
                "dense": image_id in annotations.dense_ids,
            },
            "ground_truth": [],
            "predictions": [],
            "summary": {},
        }
        for index, gt in enumerate(ground_truths):
            size_name = _diagnostic_size_bin(gt.box)
            record["ground_truth"].append({
                "id": gt.id,
                "box": list(gt.box),
                "label": gt.label,
                "ignore": gt.ignore,
                "uncertain": gt.uncertain,
                "dense": gt.dense,
                "sqrt_area_px": round(math.sqrt(
                    (gt.box[2] - gt.box[0]) * (gt.box[3] - gt.box[1])), 4),
                "size_bin": size_name,
                "boundary": _boundary(
                    gt.box, image["width"], image["height"], config.boundary_fraction),
                "status": "ignored" if gt.ignore or gt.uncertain else "positive",
                "outcome": gt_outcomes[index],
            })
            if not (gt.ignore or gt.uncertain) and gt_outcomes[index]["status"] == "missed":
                errors.append(_record_error(
                    record, "miss", "no_matching_detection",
                    box=gt.box, iou=gt_outcomes[index]["iou"], gt_id=gt.id,
                    boundary=_boundary(gt.box, image["width"], image["height"],
                                       config.boundary_fraction),
                ))
            elif gt_outcomes[index]["status"] == "matched":
                pass
        for detection, outcome in zip(detections, primary_outcomes):
            item = {
                "source_index": detection.source_index,
                "box": list(detection.box),
                "score": detection.score,
                "label": detection.label,
                "sqrt_area_px": round(math.sqrt(
                    (detection.box[2] - detection.box[0])
                    * (detection.box[3] - detection.box[1])), 4),
                "size_bin": _diagnostic_size_bin(detection.box),
                "outcome": outcome,
            }
            record["predictions"].append(item)
            if outcome["status"] == "false_positive":
                gt_index = outcome.get("gt_index")
                gt = ground_truths[gt_index] if gt_index is not None else None
                errors.append(_record_error(
                    record, "false_positive", outcome["subtype"],
                    box=detection.box, iou=outcome["iou"], score=detection.score,
                    object_id=detection.source_index,
                    gt_id=gt.id if gt else None,
                    gt_box=gt.box if gt else None,
                    boundary=_boundary(detection.box, image["width"], image["height"],
                                       config.boundary_fraction),
                ))
        tp = sum(item["outcome"]["status"] == "true_positive"
                 for item in record["predictions"])
        fp = sum(item["outcome"]["status"] == "false_positive"
                 for item in record["predictions"])
        neutral = sum(item["outcome"]["status"] == "neutral"
                      for item in record["predictions"])
        misses = sum(gt["outcome"]["status"] == "missed"
                     for gt in record["ground_truth"] if gt["status"] == "positive")
        record["summary"] = {
            "positive_ground_truths": len(positive_objects),
            "predictions": len(detections),
            "true_positives": tp,
            "false_positives": fp,
            "duplicates": sum(item["outcome"]["subtype"] == "duplicate"
                              for item in record["predictions"]),
            "localization_errors": sum(item["outcome"]["subtype"] == "localization"
                                       for item in record["predictions"]),
            "background_false_positives": sum(
                item["outcome"]["subtype"] == "background"
                for item in record["predictions"]),
            "misses": misses,
            "neutral_predictions": neutral,
        }
        records.append(record)

        for size_name in SIZE_RANGES:
            positive_counts[size_name] += sum(
                not (gt.ignore or gt.uncertain) and _in_range(list(gt.box), SIZE_RANGES[size_name])
                for gt in ground_truths
            )
            for threshold in config.iou_thresholds:
                if size_name == "all" and threshold == config.primary_iou:
                    outcomes = primary_outcomes
                else:
                    outcomes, _ = _match_image(
                        detections, ground_truths, config, size_name, threshold)
                for detection, outcome in zip(detections, outcomes):
                    if outcome["status"] in {"true_positive", "false_positive"}:
                        ap_entries[size_name][threshold].append(
                            (detection.score, image_id, detection.source_index,
                             outcome["status"]))
    if excluded_dense:
        issues.append(_issue(
            "dense_images_excluded",
            f"{excluded_dense} dense images excluded, matching the TinyPerson protocol",
        ))

    ap_by_bin: dict[str, dict[str, Any]] = {}
    for size_name, thresholds in ap_entries.items():
        row: dict[str, Any] = {}
        for threshold, entries in thresholds.items():
            row[f"AP@{threshold:.2f}"] = _ap_from_outcomes(
                entries, positive_counts[size_name])
        row["positive_ground_truths"] = positive_counts[size_name]
        valid = [value for value in row.values() if isinstance(value, float)]
        row["AP_mean_IoU_0.25_0.50_0.75"] = (
            sum(valid) / len(valid) if valid else None)
        ap_by_bin[size_name] = row

    crosscheck = _crosscheck_reference(annotations, predictions, expected_ids)
    if crosscheck.get("available"):
        reference = crosscheck.pop("reference")
        comparisons = {}
        mismatch = False
        for size_name, values in ap_by_bin.items():
            for key in ("AP@0.25", "AP@0.50", "AP@0.75"):
                ours = values.get(key)
                theirs = reference.get(size_name, {}).get(key)
                if isinstance(ours, float) and isinstance(theirs, float):
                    difference = abs(ours - theirs)
                    comparisons[f"{size_name}:{key}"] = {
                        "diagnostic": ours, "reference": theirs, "abs_diff": difference,
                    }
                    mismatch = mismatch or difference > 1e-9
        crosscheck = {
            "available": True,
            "status": "mismatch" if mismatch else "match",
            "comparisons": comparisons,
            "reference_protocol": reference.get("protocol"),
        }

    aggregation = _aggregate(records)
    counts = aggregation["counts"]
    positives = aggregation["positive_ground_truths"]
    counts["misses"] = sum(record["summary"]["misses"] for record in records)
    counts["predictions"] = sum(record["summary"]["predictions"] for record in records)
    invariants = {
        "ground_truth_partition": counts["misses"] + counts["true_positives"] == positives,
        "prediction_partition": (
            counts["true_positives"] + counts["false_positives"] + counts["neutral_predictions"]
            == counts["predictions"]
        ),
        "false_positive_partition": (
            counts["duplicates"] + counts["localization_errors"]
            + counts["background_false_positives"] == counts["false_positives"]
        ),
        "records_reconcile": sum(record["summary"]["predictions"] for record in records)
        == counts["predictions"],
    }
    if not all(invariants.values()):
        issues.append(_issue("reconciliation_failed", "Record counts do not reconcile"))

    provenance = _provenance(annotations, manifest, predictions, config, split,
                             run, checkpoint)
    dataset_key = {
        "annotations_sha256": annotations.sha256,
        "manifest_sha256": manifest.sha256 if manifest else None,
        "split": split,
    }
    comparison = (_compare_samples(records, baseline_samples, dataset_key)
                  if baseline_samples is not None else None)
    limitations = [
        ("Diagnostic AP is derived from sample records; the repository TinyPerson "
         "evaluator remains the metric source of truth."),
        "The derivative validation split is not the official test set.",
        ("The TinyPerson evaluator is a repository implementation, not a vendored "
         "reference evaluator; miss rate still requires the reference evaluator."),
        "Associations in this report are descriptive, not causal.",
    ]
    if predictions.metadata is None:
        limitations.append(
            "The prediction artifact carries no inference settings; imgsz, confidence, "
            "batch, and device are not independently verified."
        )
    if limit:
        limitations.append(
            f"Analysis is a deterministic basename-sorted subset of {limit} images, not a "
            "random sample; rates are not population estimates."
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "dataset": {
            "split": split,
            "annotations": annotations.source,
            "manifest": manifest.source if manifest else None,
            "expected_images": len(expected_ids),
            "analyzed_images": len(records),
            "dense_images_excluded": excluded_dense,
            "predictions_source": predictions.source,
        },
        "config": config.to_dict(),
        "integrity": {
            "valid": not any(issue["code"] in {
                "missing_predictions", "unexpected_predictions", "invalid_prediction",
                "invalid_annotation_box", "annotation_unknown_image",
                "unknown_annotation_category", "duplicate_image_basename",
                "duplicate_image_id", "invalid_image_dimensions",
                "manifest_image_absent_from_annotations", "manifest_basename_mismatch",
                "manifest_dimension_mismatch", "reconciliation_failed",
                "non_person_prediction",
            } for issue in issues),
            "issues": issues,
            "invariants": invariants,
        },
        "counts": counts,
        "positive_ground_truths": positives,
        "rates": {
            "recall": _rate(counts["true_positives"], positives),
            "precision": _rate(counts["true_positives"],
                               counts["true_positives"] + counts["false_positives"]),
            "miss_rate": _rate(counts["misses"], positives),
            "duplicate_share_of_false_positives": _rate(
                counts["duplicates"], counts["false_positives"]),
            "localization_share_of_false_positives": _rate(
                counts["localization_errors"], counts["false_positives"]),
            "background_share_of_false_positives": _rate(
                counts["background_false_positives"], counts["false_positives"]),
        },
        "diagnostic_ap": ap_by_bin,
        "metric_crosscheck": crosscheck,
        "aggregation": aggregation,
        "comparison": comparison,
        "limitations": limitations,
    }
    return AnalysisResult(summary=summary, samples=records, errors=errors)


def _provenance(annotations: AnnotationSet, manifest: ManifestSet | None,
                predictions: PredictionSet, config: AnalysisConfig, split: str,
                run: str | Path | None, checkpoint: str | Path | None) -> dict[str, Any]:
    commit, dirty = git_info()
    run_metadata = None
    run_path = Path(run).expanduser().resolve() if run is not None else None
    if run_path is not None and (run_path / "metadata.json").is_file():
        run_metadata = json.loads((run_path / "metadata.json").read_text())
    checkpoint_path = Path(checkpoint).expanduser().resolve() if checkpoint is not None else None
    if checkpoint_path is None and run_path is not None:
        for candidate in (run_path / "checkpoints" / "best.pt",
                          run_path / "training" / "weights" / "best.pt"):
            if candidate.is_file():
                checkpoint_path = candidate.resolve()
                break
    config_json = json.dumps(config.to_dict(), sort_keys=True)
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "run": str(run_path) if run_path else None,
        "run_metadata": run_metadata,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path else None,
        "annotations_sha256": annotations.sha256,
        "manifest_sha256": manifest.sha256 if manifest else None,
        "predictions_sha256": predictions.sha256,
        "predictions_metadata": predictions.metadata,
        "split": split,
        "analysis_config_sha256": hashlib.sha256(config_json.encode()).hexdigest(),
        "analysis_config": config.to_dict(),
    }
