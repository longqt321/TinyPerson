"""Frozen YOLO26 spatial-probe experiment.

GT ROIAlign is an oracle diagnostic, never an inference-time detector component.
The probe is trained only on group-disjoint feature vectors.
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.ops import roi_align
from ultralytics import YOLO

from tinydet.analysis.representation import (
    FeatureCapture, center, gt_entries, input_tensor, validate_output_directory, write_csv,
)
from tinydet.utils.dataset import dataset_images
from tinydet.utils.images import letterbox
from tinydet.utils.model import resolve_stages

def resolve_spatial_checkpoint(run):
    """Locate best.pt in finalized or raw Ultralytics run artifacts."""
    run = Path(run)
    candidates = (
        run / "checkpoints" / "best.pt",
        run / "training" / "weights" / "best.pt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find best.pt; checked: " + ", ".join(map(str, candidates))
    )


BINS = ("0-8", "8-16", "16-32", ">32")
METHODS = ("legacy_center", "center", "local_mean_3x3", "roi_align")


def source_group(path, kind):
    path = Path(path)
    if kind == "image":
        return str(path.resolve())
    # TinyPerson filenames commonly include V0031 (original video identity).
    match = re.search(r"(?:^|[_-])V(\d+)(?:[_-]|$)", path.stem, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Cannot derive source video from {path.name}; use --group-by image explicitly"
        )
    return f"video:{match.group(1)}"


def sample_feature(feature, box, size, method):
    """Return C-vector; coordinates refer to square network input."""
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("Expected NCHW feature tensor with batch size 1")
    _, _, h, w = feature.shape
    x, y = center(box)
    if method == "legacy_center":
        ix = min(w - 1, max(0, int(x * w / size)))
        iy = min(h - 1, max(0, int(y * h / size)))
        return feature[0, :, iy, ix].detach().float().cpu().numpy()
    if method in ("center", "local_mean_3x3"):
        source = (F.avg_pool2d(feature, 3, stride=1, padding=1)
                  if method == "local_mean_3x3" else feature)
        # align_corners=False: normalized coordinates of a pixel center follow x / W.
        grid = torch.tensor([[[[2 * x / size - 1, 2 * y / size - 1]]]],
                            dtype=source.dtype, device=source.device)
        vec = F.grid_sample(source, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
        return vec[0, :, 0, 0].detach().float().cpu().numpy()
    if method == "roi_align":
        if w != h:
            raise ValueError("Expected square feature map and square letterboxed input")
        tensor_box = torch.as_tensor([box], dtype=feature.dtype, device=feature.device)
        vec = roi_align(feature, [tensor_box], output_size=(1, 1),
                        spatial_scale=w / size, sampling_ratio=2, aligned=True)
        return vec[0, :, 0, 0].detach().float().cpu().numpy()
    raise ValueError(f"Unknown sampler: {method}")


def sample_features(feature, boxes, size, method):
    """Batch sampler; one PyTorch operator call per stage/method per image."""
    if not boxes:
        return np.empty((0, feature.shape[1]), dtype=np.float32)
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise ValueError("Expected NCHW feature with batch size 1")
    _, channels, height, width = feature.shape
    boxes_tensor = torch.as_tensor(boxes, device=feature.device, dtype=feature.dtype)
    xy = (boxes_tensor[:, :2] + boxes_tensor[:, 2:]) / 2
    if method == "legacy_center":
        ix = (xy[:, 0] * width / size).long().clamp(0, width - 1)
        iy = (xy[:, 1] * height / size).long().clamp(0, height - 1)
        out = feature[0, :, iy, ix].T
    elif method in ("center", "local_mean_3x3"):
        source = (F.avg_pool2d(feature, 3, stride=1, padding=1)
                  if method == "local_mean_3x3" else feature)
        grid = (2 * xy / size - 1).reshape(1, -1, 1, 2)
        out = F.grid_sample(source, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)[0, :, :, 0].T
    elif method == "roi_align":
        if width != height:
            raise ValueError("ROIAlign requires square feature map in this experiment")
        out = roi_align(feature, [boxes_tensor], output_size=(1, 1),
                        spatial_scale=width / size, sampling_ratio=2,
                        aligned=True)[:, :, 0, 0]
    else:
        raise ValueError(f"Unknown sampler: {method}")
    return out.detach().float().cpu().numpy().reshape(-1, channels)


def overlaps(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return w * h


def matched_negative(target, gt, size, rng, attempts=300):
    """Background ROI of equal size; reject every overlap with all annotated objects."""
    x1, y1, x2, y2 = target["box"]
    w, h = x2 - x1, y2 - y1
    if w >= size or h >= size:
        return None
    for _ in range(attempts):
        xx = float(rng.uniform(0, size - w))
        yy = float(rng.uniform(0, size - h))
        box = [xx, yy, xx + w, yy + h]
        if all(overlaps(box, other["box"]) <= 0 for other in gt):
            return box
    return None


def extract(checkpoint, dataset, split, imgsz, seed, group_by, methods=METHODS,
            device="cpu", max_images=None):
    """Single backbone forward per image, producing common sample IDs for all probes."""
    rng = np.random.default_rng(seed)
    detector = YOLO(str(checkpoint))
    net = detector.model.to(device).eval()
    selected = resolve_stages(net)
    capture = FeatureCapture(net, selected)
    records, vectors = [], {stage: {method: [] for method in methods} for stage in selected}
    paths = dataset_images(Path(dataset), split)
    if max_images is not None:
        paths = paths[:max_images]
    try:
        for index, path in enumerate(paths):
            with Image.open(path) as original:
                width, height = original.size
                canvas, scale, left, top = letterbox(original.convert("RGB"), imgsz)
            all_gt = gt_entries(path, width, height, scale, left, top)
            if not all_gt:
                continue
            tensor = input_tensor(canvas, device)
            features = capture.forward(net, tensor)
            group = source_group(path, group_by)
            image_boxes = []
            for row in all_gt:
                negative = matched_negative(row, all_gt, imgsz, rng)
                if negative is None:
                    continue
                for positive, box in ((1, row["box"]), (0, negative)):
                    sample_id = len(records)
                    records.append({"sample_id": sample_id, "image": str(path),
                                    "group": group, "gt_id": row["gt_id"],
                                    "size_bin": row["size_bin"], "label": positive,
                                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
                    image_boxes.append(box)
            for stage, feature in features.items():
                for method in methods:
                    vectors[stage][method].extend(
                        sample_features(feature, image_boxes, imgsz, method)
                    )
            if (index + 1) % 25 == 0:
                print(f"Extracted {index+1}/{len(paths)} images", flush=True)
    finally:
        capture.close()
    return records, {s: {m: np.stack(v).astype("float32")
                         for m, v in methods_dict.items()}
                     for s, methods_dict in vectors.items()}


def split_groups(records, seed, train_fraction):
    groups = np.asarray([r["group"] for r in records])
    labels = np.asarray([r["label"] for r in records])
    unique = set(groups)
    if len(unique) < 2:
        raise ValueError("At least two distinct image/video groups are required")
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
    train, test = next(splitter.split(np.zeros(len(records)), labels, groups))
    if set(groups[train]) & set(groups[test]):
        raise AssertionError("Train/test group leakage")
    if len(set(labels[train])) != 2 or len(set(labels[test])) != 2:
        raise ValueError("Both train and test must have positive and negative examples")
    return train, test


def paired_group_bootstrap(rows, a, b, repeats, seed):
    """Bootstrap AP difference with group-level resampling and shared sampled groups."""
    rng = np.random.default_rng(seed)
    by_group = defaultdict(list)
    for i, row in enumerate(rows):
        by_group[row["group"]].append(i)
    keys = sorted(by_group)
    if len(keys) < 2:
        return float("nan"), float("nan")
    deltas = []
    for _ in range(repeats):
        selected = rng.choice(len(keys), size=len(keys), replace=True)
        ids = np.concatenate([by_group[keys[j]] for j in selected])
        labels = np.array([rows[int(i)]["label"] for i in ids])
        if labels.min() == labels.max():
            continue
        deltas.append(float(average_precision_score(labels, b[ids])
                            - average_precision_score(labels, a[ids])))
    if not deltas:
        return float("nan"), float("nan")
    return tuple(map(float, np.percentile(deltas, [2.5, 97.5])))


def analyze(records, vectors, seed, train_fraction=0.6, bootstrap=1000):
    train, test = split_groups(records, seed, train_fraction)
    labels = np.asarray([r["label"] for r in records], dtype=int)
    records_test = [records[int(i)] for i in test]
    predictions, metrics, contrasts = [], [], []
    for stage in vectors:
        scores = {}
        for method, vectors_method in vectors[stage].items():
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=seed),
            )
            model.fit(vectors_method[train], labels[train])
            pred = model.predict_proba(vectors_method[test])[:, 1]
            scores[method] = pred
            for i, row in enumerate(records_test):
                predictions.append({"stage": stage, "method": method,
                                    **row, "score": float(pred[i])})
            for size_bin in ("all", *BINS):
                ids = [i for i, row in enumerate(records_test)
                       if size_bin == "all" or row["size_bin"] == size_bin]
                y = labels[test][ids]
                if not len(ids) or y.sum() == 0:
                    continue
                metrics.append({
                    "stage": stage, "method": method, "size_bin": size_bin,
                    "positives": int(y.sum()), "negatives": int(len(y) - y.sum()),
                    "ap_random_baseline": float(y.mean()),
                    "auprc": float(average_precision_score(y, pred[ids])),
                })
        for method in ("center", "local_mean_3x3", "roi_align"):
            if method not in scores or "legacy_center" not in scores:
                continue
            for size_bin in ("all", *BINS):
                ids = [i for i, row in enumerate(records_test)
                       if size_bin == "all" or row["size_bin"] == size_bin]
                if not ids:
                    continue
                subset = [records_test[i] for i in ids]
                y = np.asarray([row["label"] for row in subset])
                if y.sum() == 0:
                    continue
                base = scores["legacy_center"][ids]
                new = scores[method][ids]
                low, high = paired_group_bootstrap(subset, base, new, bootstrap, seed)
                contrasts.append({
                    "stage": stage, "method": method, "baseline": "legacy_center",
                    "size_bin": size_bin,
                    "delta_ap": float(average_precision_score(y, new)
                                      - average_precision_score(y, base)),
                    "ci_low": low, "ci_high": high,
                    "groups": len({row["group"] for row in subset}),
                })
    return train, test, predictions, metrics, contrasts


def run(checkpoint, dataset, output, config, allowed_output_root):
    output = validate_output_directory(allowed_output_root, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory not empty: {output}")
    settings = config.get("experiment", {})
    seed = int(settings.get("seed", 42))
    split = settings.get("split", "val")
    imgsz = int(settings.get("imgsz", 1280))
    group_by = settings.get("group_by", "video")
    if group_by not in ("image", "video"):
        raise ValueError("group_by must be image or video")
    methods = tuple(settings.get("methods", METHODS))
    if len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
        raise ValueError("Unknown or duplicate sampler")
    train_fraction = float(settings.get("train_fraction", 0.6))
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    output.mkdir(parents=True, exist_ok=True)
    records, vectors = extract(
        checkpoint, dataset, split, imgsz, seed, group_by, methods=methods,
        device=settings.get("device", "cpu"), max_images=settings.get("max_images"),
    )
    if not records:
        raise ValueError("No probe samples; check segmentation annotations")
    train, test, predictions, metrics, contrasts = analyze(
        records, vectors, seed, train_fraction, int(settings.get("bootstrap", 1000))
    )
    train_set = set(train.tolist())
    for i, record in enumerate(records):
        record["partition"] = "train" if i in train_set else "test"
    write_csv(output / "samples.csv", records)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "paired_deltas.csv", contrasts)
    meta = {
        "checkpoint": str(checkpoint), "dataset": str(dataset), "split": split,
        "input_size": imgsz, "seed": seed, "group_by": group_by,
        "train_count": len(train), "test_count": len(test),
        "train_groups": len({records[int(i)]["group"] for i in train}),
        "test_groups": len({records[int(i)]["group"] for i in test}),
        "total_instances_in_probe": len(records) // 2,
        "size_distribution": dict(Counter(r["size_bin"] for r in records if r["label"] == 1)),
        "methods": methods, "stages": list(vectors),
        "limitations": [
            "GT ROIAlign is oracle-only; cannot imply inference-time alignment.",
            "Probes measured on val after detector checkpoint selection; not an independent YOLO benchmark.",
            "AP of subgroup contrasts depends on negative sampling; compare matched cohorts only.",
            "Feature-map probing is correlational, not causal evidence of information loss.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta

def save_feature_cache(checkpoint, dataset, output, config, allowed_output_root):
    """GPU phase: extract vectors and save to volume, with no probe training."""
    output = validate_output_directory(allowed_output_root, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory not empty: {output}")
    settings = config.get("experiment", {})
    device = settings.get("device", "cuda")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Modal GPU extraction requires a working CUDA device")
    output.mkdir(parents=True, exist_ok=True)
    records, vectors = extract(
        checkpoint, dataset, settings.get("split", "val"),
        int(settings.get("imgsz", 1280)), int(settings.get("seed", 42)),
        settings.get("group_by", "video"),
        methods=tuple(settings.get("methods", METHODS)),
        device=device, max_images=settings.get("max_images"),
    )
    if not records:
        raise ValueError("No probe samples")
    write_csv(output / "samples.csv", records)
    flattened = {f"{stage}__{method}": arr for stage, values in vectors.items()
                 for method, arr in values.items()}
    np.savez_compressed(output / "features.npz", **flattened)
    (output / "config.yaml").write_text(__import__("yaml").safe_dump(config))
    return {"instances": len(records) // 2, "images": len({r["image"] for r in records}),
            "output": str(output)}


def finish_feature_cache(output, config, allowed_output_root):
    """CPU phase: fit probes and bootstrap, no GPU or YOLO dependency in execution."""
    output = validate_output_directory(allowed_output_root, output)
    with (output / "samples.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    for row in records:
        row["sample_id"] = int(row["sample_id"])
        row["gt_id"] = int(row["gt_id"])
        row["label"] = int(row["label"])
    vectors = defaultdict(dict)
    with np.load(output / "features.npz", allow_pickle=False) as archive:
        for name in archive.files:
            stage, method = name.split("__", 1)
            vectors[stage][method] = archive[name]
    settings = config.get("experiment", {})
    seed = int(settings.get("seed", 42))
    train, test, predictions, metrics, contrasts = analyze(
        records, vectors, seed, float(settings.get("train_fraction", 0.6)),
        int(settings.get("bootstrap", 1000)),
    )
    train_set = set(train.tolist())
    for i, record in enumerate(records):
        record["partition"] = "train" if i in train_set else "test"
    write_csv(output / "samples.csv", records)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "paired_deltas.csv", contrasts)
    meta = {
        "train_count": len(train), "test_count": len(test),
        "train_groups": len({records[int(i)]["group"] for i in train}),
        "test_groups": len({records[int(i)]["group"] for i in test}),
        "total_instances_in_probe": len(records) // 2,
        "size_distribution": dict(Counter(r["size_bin"] for r in records if r["label"] == 1)),
        "stages": list(vectors), "group_by": settings.get("group_by", "video"),
        "samplers": list(next(iter(vectors.values()))),
        "device_extraction": settings.get("device", "cuda"),
        "note": "GT ROIAlign is an oracle diagnostic; AP differences are not causal evidence.",
    }
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
