"""Representation-breakdown experiments on frozen YOLO26 features.

Outputs are diagnostics, not causal proofs of information loss. All geometry is
defined in letterboxed network-input pixels. No YOLO parameters are updated.
"""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from tinydet.analysis.heavy_bottom import (
    dataset_images, label_path, letterbox, map_box, resolve_stages, size_bin,
)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def split_image(path, seed=42, train_fraction=0.7):
    """Deterministic image-group split; never split instances within an image."""
    digest = hashlib.sha256(f"{seed}:{Path(path).as_posix()}".encode()).digest()
    return "train" if int.from_bytes(digest[:8], "big") / 2**64 < train_fraction else "test"


def input_tensor(image, device):
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).to(device)


def gt_entries(path, size, width, height, scale, left, top):
    labels = label_path(path)
    entries = []
    if not labels.is_file():
        return entries
    for i, line in enumerate(labels.read_text().splitlines()):
        values = [float(x) for x in line.split()]
        if len(values) < 5 or int(values[0]) != 0:
            continue
        cx, cy, bw, bh = values[1:5]
        box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
               (cx + bw / 2) * width, (cy + bh / 2) * height]
        mapped = map_box(box, scale, left, top)
        if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
            continue
        entries.append({"gt_id": i, "box": mapped,
                        "size_bin": size_bin(mapped[2] - mapped[0], mapped[3] - mapped[1])})
    return entries


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def vector_at(feature, xy, input_size):
    _, _, height, width = feature.shape
    ix = max(0, min(width - 1, int(xy[0] * width / input_size)))
    iy = max(0, min(height - 1, int(xy[1] * height / input_size)))
    return feature[0, :, iy, ix].detach().float().cpu().numpy()


def background_centers(image, gt, count, size, rng):
    """Hard-ish local negatives and uniform negatives, all away from GT boxes."""
    boxes = [row["box"] for row in gt]
    centers = []
    for row in gt:
        x, y = center(row["box"])
        radius = max(row["box"][2] - row["box"][0], row["box"][3] - row["box"][1], 8)
        for t in range(count):
            if t % 2:
                candidate = (rng.uniform(0, size), rng.uniform(0, size))
            else:
                angle = rng.uniform(0, 2 * math.pi)
                distance = rng.uniform(radius * 1.8, radius * 4.0)
                candidate = (x + math.cos(angle) * distance,
                             y + math.sin(angle) * distance)
            a, b = candidate
            if not (2 <= a < size - 2 and 2 <= b < size - 2):
                continue
            if any(x1 - 3 <= a <= x2 + 3 and y1 - 3 <= b <= y2 + 3
                   for x1, y1, x2, y2 in boxes):
                continue
            centers.append(candidate)
    return centers


def average_precision(y, score):
    """Ranking AP; ties follow a stable ordering."""
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / (np.arange(len(y)) + 1)
    return float((precision * sorted_y).sum() / positives)


def linear_probe(train_x, train_y, test_x, steps=200, lr=0.15, l2=1e-3):
    """Tiny deterministic logistic probe with train-only standardization."""
    x = np.asarray(train_x, np.float64)
    z = np.asarray(test_x, np.float64)
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-5)
    x = np.clip((x - mu) / sd, -8, 8)
    z = np.clip((z - mu) / sd, -8, 8)
    y = np.asarray(train_y, np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(steps):
        logits = np.clip(x @ weights + bias, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        gradient = probabilities - y
        weights -= lr * (x.T @ gradient / len(x) + l2 * weights)
        bias -= lr * gradient.mean()
    return z @ weights + bias


def relative_feature_delta(a, b, box, size, eps=1e-6):
    """Normalized difference in the box-aligned feature region."""
    _, _, h, w = a.shape
    x1, y1, x2, y2 = box
    xa = max(0, min(w - 1, int(math.floor(x1 * w / size))))
    ya = max(0, min(h - 1, int(math.floor(y1 * h / size))))
    xb = max(xa + 1, min(w, int(math.ceil(x2 * w / size))))
    yb = max(ya + 1, min(h, int(math.ceil(y2 * h / size))))
    before = a[..., ya:yb, xa:xb].float()
    after = b[..., ya:yb, xa:xb].float()
    numerator = torch.linalg.vector_norm(before - after)
    denominator = torch.linalg.vector_norm(after)
    return float((numerator / (denominator + eps)).detach().cpu())


def shift_tensor(tensor, dx, dy, fill=114 / 255):
    """Translation without wraparound; mask affected padded area in interpretation."""
    shifted = torch.full_like(tensor, fill)
    h, w = tensor.shape[-2:]
    x1, x2 = max(0, dx), min(w, w + dx)
    y1, y2 = max(0, dy), min(h, h + dy)
    if x2 > x1 and y2 > y1:
        shifted[..., y1:y2, x1:x2] = tensor[..., y1-dy:y2-dy, x1-dx:x2-dx]
    return shifted


def detection_rows(detector, tensor, boxes, offset, names, size):
    """Match class-0 detections to GT with a fixed confidence threshold."""
    with torch.no_grad():
        result = detector.predict(source=tensor, imgsz=size, conf=0.001,
                                  iou=0.7, max_det=1000, verbose=False)[0]
    predictions = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy()
    records = []
    dx, dy = offset
    for row in boxes:
        x1, y1, x2, y2 = row["box"]
        target = np.array([x1 + dx, y1 + dy, x2 + dx, y2 + dy])
        overlaps = []
        for pred, cls in zip(predictions, classes):
            if int(cls) != 0:
                overlaps.append(0.0)
                continue
            left, top = np.maximum(target[:2], pred[:2])
            right, bottom = np.minimum(target[2:], pred[2:])
            area = max(0.0, right-left) * max(0.0, bottom-top)
            union = np.prod(target[2:] - target[:2]) + np.prod(pred[2:] - pred[:2]) - area
            overlaps.append(float(area / max(union, 1e-9)))
        matched = [float(score) for score, overlap in zip(scores, overlaps)
                   if overlap >= 0.5 and score >= 0.25]
        records.append({"gt_id": row["gt_id"], "size_bin": row["size_bin"],
                        "dx": dx, "dy": dy, "detected": int(bool(matched)),
                        "best_confidence": max(matched, default=0.0)})
    return records


class FeatureCapture:
    def __init__(self, model, selected):
        self.features = {}
        self.handles = []
        for name, index in selected.items():
            def store(_module, _args, output, stage=name):
                self.features[stage] = output.detach()
            self.handles.append(model.model[index].register_forward_hook(store))

    def forward(self, model, tensor):
        self.features.clear()
        with torch.no_grad():
            model(tensor)
        return dict(self.features)

    def close(self):
        for handle in self.handles:
            handle.remove()


def study(checkpoint, dataset_file, output, cfg):
    """GPU/CPU single-process study; outputs compact CSV, plots and report."""
    checkpoint, dataset_file, output = Path(checkpoint), Path(dataset_file), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["study"].get("seed", 42))
    rng = np.random.default_rng(seed)
    size = int(cfg["data"].get("imgsz", 1280))
    count = int(cfg["data"].get("sample_count", 400))
    split = cfg["data"].get("split", "val")
    stages_requested = cfg["model"].get("stages", ["C2", "C3", "C4", "C5"])
    enabled = cfg["experiments"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(checkpoint))
    net = detector.model.to(device).eval()
    stages = resolve_stages(net)
    selected = {k: stages[k] for k in stages_requested}
    collector = FeatureCapture(net, selected)
    dataset = dataset_images(dataset_file, split)
    if not dataset:
        raise ValueError("Dataset split has no images")
    grouped = defaultdict(list)
    for path in dataset:
        with Image.open(path) as im:
            width, height = im.size
        from tinydet.analysis.heavy_bottom import letterbox_geometry
        scale, left, top = letterbox_geometry(width, height, size)
        for row in gt_entries(path, size, width, height, scale, left, top):
            grouped[row["size_bin"]].append((path, row))
    for group in grouped.values():
        rng.shuffle(group)
    chosen = []
    while len(chosen) < count and any(grouped.values()):
        for key in ("0-8", "8-16", "16-32", ">32"):
            if grouped[key] and len(chosen) < count:
                chosen.append(grouped[key].pop())
    by_image = defaultdict(list)
    for path, row in chosen:
        by_image[path].append(row)
    samples, vector_rows, deltas, shift_rows = [], [], [], []
    stage_vectors = defaultdict(list)
    try:
        for path, targets in by_image.items():
            with Image.open(path) as im:
                letterboxed, _, _, _ = letterbox(im.convert("RGB"), size)
            tensor = input_tensor(letterboxed, device)
            features = collector.forward(net, tensor)
            for row in targets:
                samples.append({"image": str(path), **row})
            if enabled.get("separability", {}).get("enabled", True):
                negatives = background_centers(letterboxed, targets,
                    int(enabled["separability"].get("background_samples_per_object", 3)),
                    size, rng)
                training_group = split_image(path, seed)
                for stage, feature in features.items():
                    for row in targets:
                        stage_vectors[stage].append((training_group, 1, row["size_bin"],
                                                      vector_at(feature, center(row["box"]), size)))
                    for xy in negatives:
                        stage_vectors[stage].append((training_group, 0, "background",
                                                      vector_at(feature, xy, size)))
            if enabled.get("counterfactual", {}).get("enabled", True):
                # A simple background-fill control, explicitly not a clean causal intervention.
                for row in targets:
                    x1, y1, x2, y2 = [int(v) for v in row["box"]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(size, max(x1 + 1, x2)), min(size, max(y1 + 1, y2))
                    erased = tensor.clone()
                    surround = tensor[..., max(0, y1-5):min(size, y2+5),
                                      max(0, x1-5):min(size, x2+5)]
                    fill = surround.mean(dim=(-1, -2), keepdim=True)
                    erased[..., y1:y2, x1:x2] = fill
                    changed = collector.forward(net, erased)
                    for stage in features:
                        deltas.append({"image": str(path), "gt_id": row["gt_id"],
                                       "size_bin": row["size_bin"], "stage": stage,
                                       "delta": relative_feature_delta(
                                           features[stage], changed[stage], row["box"], size),
                                       "control": "mean_fill_proxy"})
            if enabled.get("shift_stability", {}).get("enabled", True):
                offsets = enabled["shift_stability"].get("offsets", [0, 1, 2, 3])
                for dx in offsets:
                    for dy in offsets:
                        if any(not (6 <= row["box"][0] + dx and 6 <= row["box"][1] + dy
                                   and row["box"][2] + dx < size-6
                                   and row["box"][3] + dy < size-6) for row in targets):
                            continue
                        shifted = shift_tensor(tensor, dx, dy)
                        # Ultralytics tensor-input prediction: batch BCHW float [0,1].
                        try:
                            detected = detection_rows(detector, shifted, targets, (dx, dy),
                                                      None, size)
                        except Exception as exc:
                            raise RuntimeError("Shift detection inference failed") from exc
                        for row in detected:
                            shift_rows.append({"image": str(path), **row})
    finally:
        collector.close()
    write_csv(output / "samples.csv", samples)
    outputs = {"checkpoint": str(checkpoint), "dataset": str(dataset_file), "split": split,
               "input_size": size, "stages": selected, "device": device,
               "sample_count": len(samples), "image_count": len(by_image),
               "counterfactual_validity": "mean-fill proxy; NOT clean object removal"}
    if stage_vectors:
        probe_rows = []
        for stage, items in stage_vectors.items():
            train = [(label, vector) for group, label, _, vector in items if group == "train"]
            test = [(label, vector, bin_name) for group, label, bin_name, vector
                    in items if group == "test"]
            if len({label for label, _ in train}) < 2 or not test:
                continue
            score = linear_probe([v for _, v in train], [y for y, _ in train],
                                 [v for _, v, _ in test])
            labels = np.array([y for y, _, _ in test])
            for size_group in ("all", "0-8", "8-16", "16-32", ">32"):
                if size_group == "all":
                    take = np.ones(len(test), bool)
                else:
                    take = np.array([label == 0 or group == size_group
                                     for label, _, group in test])
                if not np.any(labels[take] == 1):
                    continue
                probe_rows.append({"stage": stage, "size_bin": size_group,
                                   "positives": int(labels[take].sum()),
                                   "negatives": int((1-labels[take]).sum()),
                                   "auprc": average_precision(labels[take], score[take])})
        write_csv(output / "separability" / "metrics.csv", probe_rows)
        outputs["probe_records"] = len(probe_rows)
        plot_by_stage(probe_rows, "auprc", output / "separability" / "auprc_by_stage.png")
    if deltas:
        write_csv(output / "counterfactual" / "feature_delta.csv", deltas)
        plot_by_stage(deltas, "delta", output / "counterfactual" / "delta_by_stage.png")
    if shift_rows:
        write_csv(output / "shift_stability" / "predictions.csv", shift_rows)
        baseline = {(r["image"], r["gt_id"]): r["detected"]
                    for r in shift_rows if r["dx"] == 0 and r["dy"] == 0}
        flips = []
        for row in shift_rows:
            key = row["image"], row["gt_id"]
            if key not in baseline:
                continue
            flips.append({**row, "changed": int(baseline[key] != row["detected"])})
        write_csv(output / "shift_stability" / "miss_flip_rate.csv", flips)
        plot_shift(flips, output / "shift_stability" / "phase_heatmap.png")
        outputs["shift_flip_rate_nonzero"] = float(np.mean(
            [r["changed"] for r in flips if r["dx"] or r["dy"]])) if any(
            r["dx"] or r["dy"] for r in flips) else None
    (output / "metadata.json").write_text(json.dumps(outputs, indent=2) + "\n")
    report = [
        "# Representation breakdown: exploratory diagnostics",
        "", f"- Checkpoint: {checkpoint}", f"- Dataset split: {split}",
        f"- Input size: {size}", f"- Instances: {len(samples)}",
        f"- Unique images: {len(by_image)}", "",
        "## Important limitations", "",
        "- A linear-probe decline is not proof that information was destroyed.",
        "- Mean-filled object removal is a proxy confounded by edit artifacts.",
        "- Translation sensitivity mixes feature response with detector decisions.",
        "- Do not interpret observed effects as novel without prior-art comparisons.",
    ]
    (output / "report").mkdir(exist_ok=True)
    (output / "report" / "findings.md").write_text("\n".join(report) + "\n")
    return outputs


def plot_by_stage(rows, value, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    stages = sorted({r["stage"] for r in rows})
    if value == "auprc":
        for bin_name in ("all", "0-8", "8-16", "16-32", ">32"):
            values = [next((r[value] for r in rows
                            if r["stage"] == stage and r["size_bin"] == bin_name), np.nan)
                      for stage in stages]
            ax.plot(stages, values, marker="o", label=bin_name)
        ax.legend()
    else:
        for bin_name in ("0-8", "8-16", "16-32", ">32"):
            values = [np.median([r[value] for r in rows if r["stage"] == stage
                                 and r["size_bin"] == bin_name])
                      if any(r["stage"] == stage and r["size_bin"] == bin_name for r in rows)
                      else np.nan for stage in stages]
            ax.plot(stages, values, marker="o", label=bin_name)
        ax.legend()
    ax.set_xlabel("Backbone stage")
    ax.set_ylabel(value)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shift(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = sorted({r["dx"] for r in rows} | {r["dy"] for r in rows})
    grid = np.full((len(offsets), len(offsets)), np.nan)
    for i, dy in enumerate(offsets):
        for j, dx in enumerate(offsets):
            selected = [r["changed"] for r in rows if r["dx"] == dx and r["dy"] == dy]
            if selected:
                grid[i, j] = np.mean(selected)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(grid, vmin=0, vmax=1)
    ax.set_xticks(range(len(offsets)), offsets)
    ax.set_yticks(range(len(offsets)), offsets)
    ax.set_xlabel("dx (input pixels)")
    ax.set_ylabel("dy (input pixels)")
    fig.colorbar(image, ax=ax, label="Detection-state flip rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
