"""Object-size AP with excluded GT treated as ignored detections."""

from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from tinydet.analysis.heavy_bottom import BINS, dataset_images, label_path, size_bin


def iou(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.empty(0)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    overlap = np.maximum(rb - lt, 0).prod(1)
    area = max((box[2]-box[0]) * (box[3]-box[1]), 0)
    other = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(1)
    return overlap / np.maximum(area + other - overlap, 1e-9)


def ap_for_bin(samples, bin_name: str, threshold: float, prepared=None):
    """Match target GT first; predictions matching excluded GT are ignored, not FP."""
    target_count = sum(sum(gt["bin"] == bin_name for gt in sample["gt"]) for sample in samples)
    if target_count == 0:
        return None
    matched = [set() for _ in samples]
    predictions = prepare_matches(samples) if prepared is None else prepared
    true_positive, false_positive = [], []
    for _, sample_index, overlaps in predictions:
        ground_truth = samples[sample_index]["gt"]
        eligible = [j for j, gt in enumerate(ground_truth) if gt["bin"] == bin_name
                    and j not in matched[sample_index]]
        overlap = overlaps[eligible]
        if len(overlap) and overlap.max() >= threshold:
            matched[sample_index].add(eligible[int(overlap.argmax())])
            true_positive.append(1)
            false_positive.append(0)
        elif len(ground_truth) and np.any(overlaps[[j for j, gt in enumerate(ground_truth)
                                                 if gt["bin"] != bin_name]] >= threshold):
            continue
        else:
            true_positive.append(0)
            false_positive.append(1)
    if not true_positive:
        return 0.0
    recall = np.cumsum(true_positive) / target_count
    precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive) + np.cumsum(false_positive), 1)
    return float(np.mean([precision[recall >= point].max(initial=0) for point in np.linspace(0, 1, 101)]))


def prepare_matches(samples):
    return sorted(((float(score), index, iou(np.asarray(box), [gt["box"] for gt in sample["gt"]]))
                   for index, sample in enumerate(samples) for box, score in sample["pred"]),
                  reverse=True, key=lambda item: item[0])


def size_ap(samples):
    prepared = prepare_matches(samples)
    result = {}
    for name in BINS:
        values = [ap_for_bin(samples, name, float(t), prepared) for t in np.linspace(.5, .95, 10)]
        result[name] = {"AP50": values[0],
                        "AP50-95": float(np.mean(values)) if values[0] is not None else None}
    return result


def collect_size_predictions(checkpoint: Path, dataset_file: Path, artifact: Path,
                             split: str = "val", imgsz: int = 1280):
    """One streaming predictor; preserve the original size experiment's settings."""
    images = dataset_images(dataset_file, split)
    if not images or any(not path.is_file() for path in images):
        raise ValueError(f"Missing or empty dataset split: {split}")
    predictions = []
    for result in YOLO(str(checkpoint)).predict(
        source=[str(path) for path in images], stream=True, batch=1,
        imgsz=imgsz, conf=.001, max_det=1000, verbose=False,
    ):
        predictions.append({"image": result.path, "shape": result.orig_shape,
                            "boxes": result.boxes.xyxy.cpu(), "scores": result.boxes.conf.cpu(),
                            "classes": result.boxes.cls.cpu()})
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"predictions": predictions, "imgsz": imgsz}, artifact)
    return artifact


def parse_annotation_box(line: str):
    """Parse one normalized YOLO detection box OR YOLO segmentation polygon.

    TinyPerson Roboflow labels in this project are segmentation polygons.
    Treating their first four coordinates as xywh corrupts tiny-bin AP.
    """
    values = [float(v) for v in line.split()]
    if len(values) < 5 or int(values[0]) != 0:
        return None
    if len(values) == 5:
        cx, cy, bw, bh = values[1:]
        return [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
    if (len(values) - 1) % 2 or len(values) < 7:
        raise ValueError("YOLO polygon must contain at least three xy pairs")
    x = values[1::2]
    y = values[2::2]
    return [min(x), min(y), max(x), max(y)]


def recall_for_bin(samples, bin_name: str, threshold: float = 0.5) -> float | None:
    """Max recall among predictions retained at collection conf=0.001."""
    total = sum(gt["bin"] == bin_name for sample in samples for gt in sample["gt"])
    if not total:
        return None
    matched = [set() for _ in samples]
    for _, sample_idx, overlaps in prepare_matches(samples):
        eligible = [
            j for j, gt in enumerate(samples[sample_idx]["gt"])
            if gt["bin"] == bin_name and j not in matched[sample_idx]
        ]
        if not eligible:
            continue
        group_ious = overlaps[eligible]
        if len(group_ious) and group_ious.max() >= threshold:
            matched[sample_idx].add(eligible[int(group_ious.argmax())])
    return sum(map(len, matched)) / total


def finalize_size_predictions(artifact: Path):
    """Read original-scale polygons and calculate AP entirely on CPU."""
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    samples = []
    for result in payload["predictions"]:
        height, width = result["shape"]
        scale = min(payload["imgsz"] / width, payload["imgsz"] / height)
        gt = []
        label = label_path(Path(result["image"]))
        if label.is_file():
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                normalized = parse_annotation_box(line)
                if normalized is None:
                    continue
                x1, y1, x2, y2 = normalized
                box = [x1 * width, y1 * height, x2 * width, y2 * height]
                gt.append({
                    "box": box,
                    "bin": size_bin((x2-x1) * width * scale,
                                    (y2-y1) * height * scale),
                })
        pred = [
            (box.numpy(), float(score))
            for box, score, cls in zip(result["boxes"], result["scores"], result["classes"])
            if int(cls) == 0
        ]
        samples.append({"gt": gt, "pred": pred})
    result = size_ap(samples)
    for bin_name, metrics in result.items():
        metrics["GT"] = sum(
            gt["bin"] == bin_name for sample in samples for gt in sample["gt"]
        )
        metrics["Recall50_conf0.001"] = recall_for_bin(samples, bin_name)
    return result

def evaluate_size(checkpoint: Path, dataset_file: Path, split: str = "val", imgsz: int = 1280):
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="tinydet-size-") as directory:
        artifact = collect_size_predictions(checkpoint, dataset_file, Path(directory) / "predictions.pt",
                                             split, imgsz)
        return finalize_size_predictions(artifact)
