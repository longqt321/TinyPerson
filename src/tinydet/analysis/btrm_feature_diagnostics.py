"""Observe trained YOLO26n P3 background evidence without retraining.

This is an oracle-location *diagnostic*, not a full detector evaluation:
GT centers are used ONLY for feature sampling, never inference-time detection.
The original checkpoint, P3 gate and numerical implementation remain unchanged.
"""
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

from tinydet.analysis.heavy_bottom import dataset_images, letterbox_geometry
from tinydet.analysis.representation import gt_entries
from tinydet.modules.btrm import GATE_INDEX, LocalBackgroundEvidence, evidence_score
from tinydet.utils.experiment import write_json


def _summary(values):
    items = np.asarray(values, dtype=np.float64)
    if not len(items):
        return {"count": 0}
    return {
        "count": int(len(items)), "mean": float(items.mean()),
        "median": float(np.median(items)),
        "p05": float(np.quantile(items, .05)),
        "p95": float(np.quantile(items, .95)),
        "p99": float(np.quantile(items, .99)),
    }


def _random_background(box, boxes, w, h, rng, padding):
    """Try same-image near-center negatives, explicitly excluding ALL GTs."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0, x1, y1 = padding
    for index in range(100):
        if index < 75:
            theta = rng.uniform(0, math.tau)
            dist = rng.uniform(24, 80)
            x, y = cx + dist * math.cos(theta), cy + dist * math.sin(theta)
            sampling = "nearby"
        else:
            x, y = rng.uniform(x0 + 8, x1 - 8), rng.uniform(y0 + 8, y1 - 8)
            sampling = "global"
        if not (max(x0 + 8, 8) <= x < min(x1 - 8, w - 8)
                and max(y0 + 8, 8) <= y < min(y1 - 8, h - 8)):
            continue
        if any(a - 16 <= x <= c + 16 and b - 16 <= y <= d + 16
               for a, b, c, d in boxes):
            continue
        return x, y, sampling
    return None


def _scalar(tensor: torch.Tensor, x: float, y: float, size: int) -> float:
    height, width = tensor.shape[-2:]
    col = max(0, min(width - 1, int(x * width / size)))
    row = max(0, min(height - 1, int(y * height / size)))
    return float(tensor[0, 0, row, col])


def _feature_summary(records: list[dict], per_image: list[dict], gain: float | None):
    groups = defaultdict(list)
    for row in records:
        groups[(row["group"], row["size_bin"])].append(row)
    by_size = {}
    for (group, size), subset in groups.items():
        by_size.setdefault(size, {})[group] = {
            "count": len(subset),
            "z_contrast": _summary([r["z_contrast"] for r in subset]),
            "z_btrm": _summary([r["z_btrm"] for r in subset]),
            "g_contrast": _summary([r["g_contrast"] for r in subset]),
            "g_btrm": _summary([r["g_btrm"] for r in subset]),
        }
    global_stats = {}
    for field in ("g_contrast_all_sat95", "g_btrm_all_sat95",
                  "g_contrast_all_p50", "g_btrm_all_p50",
                  "g_contrast_all_p95", "g_btrm_all_p95",
                  "z_ratio_median", "modulation_abs_max_error"):
        global_stats[field] = _summary([r[field] for r in per_image])
    return {
        "effective_gain": gain, "records": len(records),
        "images": len(per_image), "by_size": by_size,
        "all_feature_maps": global_stats,
        "note": ("GT-center feature samples are oracle observations, not "
                 "independent detector predictions. A3 and A2 features come "
                 "from different trained checkpoints; scores for both modes "
                 "are also computed on the SAME checkpoint for paired control."),
    }


def collect_features(run: Path, max_images: int = 128, split: str = "val",
                     max_gt_per_image: int = 64, seed: int = 42) -> dict:
    """Use L4 batch=1; compute the numerical module gate again on captured P3."""
    if max_images < 1 or max_gt_per_image < 1 or split not in ("val", "test"):
        raise ValueError("Invalid feature diagnostic configuration")
    config = yaml.safe_load((run / "config.yaml").read_text())
    checkpoint = run / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    images = dataset_images(Path(config["data"]["config"]), split)
    if not images or any(not p.is_file() for p in images):
        raise ValueError("Missing full evaluation dataset")
    rng = random.Random(seed)
    sample_indices = sorted(rng.sample(range(len(images)), min(max_images, len(images))))
    selected = [images[index] for index in sample_indices]
    imgsz = int(config["train"]["imgsz"])

    if not torch.cuda.is_available():
        raise RuntimeError("Feature diagnostic expects a Modal L4 GPU")
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    model = YOLO(str(checkpoint)).model.to(device).eval()
    gate = model.model[GATE_INDEX]
    mode = config["model"].get("feature_module", {}).get("mode", "none")
    if mode == "none" and not isinstance(gate, torch.nn.Identity):
        raise ValueError("A0 must use a true Identity gate")
    if mode != "none" and (
        not isinstance(gate, LocalBackgroundEvidence) or gate.mode != mode
    ):
        raise ValueError("Saved gate differs from experiment YAML")
    gain = float(gate.effective_gain.detach()) if mode != "none" else None
    captured = {}

    def on_p3(_module, _inputs, output):
        captured["p3"] = output.detach().float().cpu()

    def on_gate(_module, _inputs, output):
        captured["after"] = output.detach().float().cpu()

    handles = [
        model.model[16].register_forward_hook(on_p3),
        model.model[GATE_INDEX].register_forward_hook(on_gate),
    ]
    records, per_image = [], []
    try:
        for path in selected:
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if raw is None:
                raise FileNotFoundError(f"Unable to decode source image: {path}")
            height, width = raw.shape[:2]
            scale, left, top = letterbox_geometry(width, height, imgsz)
            resized = LetterBox(
                new_shape=(imgsz, imgsz), auto=False, stride=32
            )(image=raw)
            inp = (
                torch.from_numpy(resized[..., ::-1].transpose(2, 0, 1).copy())
                .unsqueeze(0).to(device).float().div_(255)
            )
            all_annotations = gt_entries(path, width, height, scale, left, top)
            annotations = all_annotations
            if len(annotations) > max_gt_per_image:
                annotations = rng.sample(annotations, max_gt_per_image)
            captured.clear()
            with torch.inference_mode():
                _ = model(inp)
            if set(captured) != {"p3", "after"}:
                raise RuntimeError("Feature hooks did not capture P3 and gate outputs")
            before, after = captured["p3"], captured["after"]
            # Always compute both candidate mechanisms on the SAME P3 tensor.
            params = {
                "radius": gate.radius, "ridge_relative": gate.ridge_relative,
                "eps": gate.eps
            } if mode != "none" else {}
            with torch.no_grad():
                zc = evidence_score(before, "contrast", **params)
                zb = evidence_score(before, "btrm", **params)
                gc = torch.tanh(zc / (gate.saturation if mode != "none" else 3.0))
                gb = torch.tanh(zb / (gate.saturation if mode != "none" else 3.0))
                if mode == "none":
                    expected_after = before
                else:
                    expected_gate = gc if mode == "contrast" else gb
                    expected_after = before * (1 + gain * expected_gate)
                error = float((after - expected_after).abs().max())
                # CPU vs CUDA reduction-order differences are bounded, but
                # an actual wiring bug produces much larger discrepancies.
                tolerance = 1e-4 + 2e-4 * float(after.abs().max())
                if error > tolerance:
                    raise AssertionError(
                        f"Checkpoint gate does not match recomputed equation: "
                        f"{mode} max_error={error:.5g}, tolerance={tolerance:.5g}"
                    )
            for map_name, scores in (("zc", zc), ("zb", zb)):
                if not torch.isfinite(scores).all():
                    raise AssertionError(f"Nonfinite {map_name} on real features {path}")
            # Gate saturation across actual feature map including zeroed
            # boundary (interior only for statistics).
            rad = gate.radius if mode != "none" else 1
            interior_gc = gc[..., rad:-rad, rad:-rad].flatten().numpy()
            interior_gb = gb[..., rad:-rad, rad:-rad].flatten().numpy()
            nz = zc[..., rad:-rad, rad:-rad].flatten().numpy()
            dz = zb[..., rad:-rad, rad:-rad].flatten().numpy()
            per_image.append({
                "image": str(path), "gt_count": len(annotations),
                "g_contrast_all_sat95": float(np.mean(interior_gc >= .95)),
                "g_btrm_all_sat95": float(np.mean(interior_gb >= .95)),
                "g_contrast_all_p50": float(np.median(interior_gc)),
                "g_btrm_all_p50": float(np.median(interior_gb)),
                "g_contrast_all_p95": float(np.quantile(interior_gc, .95)),
                "g_btrm_all_p95": float(np.quantile(interior_gb, .95)),
                "z_ratio_median": float(np.median(
                    dz[nz > 1e-5] / nz[nz > 1e-5]
                )) if (nz > 1e-5).any() else 0.0,
                "modulation_abs_max_error": error,
            })
            gt_boxes = [r["box"] for r in all_annotations]
            padding = (left, top, left + round(width * scale),
                       top + round(height * scale))
            for row in annotations:
                box = row["box"]
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                negative = _random_background(
                    box, gt_boxes, imgsz, imgsz, rng, padding
                )
                for group, px, py, method in (
                    [("GT_center", cx, cy, "oracle")]
                    + ([("BG", negative[0], negative[1], negative[2])]
                       if negative else [])
                ):
                    a = _scalar(zc, px, py, imgsz)
                    b = _scalar(zb, px, py, imgsz)
                    records.append({
                        "image": str(path), "gt_id": row["gt_id"],
                        "size_bin": row["size_bin"], "group": group,
                        "bg_sampler": method, "x_input": px, "y_input": py,
                        "z_contrast": a, "z_btrm": b,
                        "g_contrast": math.tanh(a / (gate.saturation if mode != "none" else 3.0)),
                        "g_btrm": math.tanh(b / (gate.saturation if mode != "none" else 3.0)),
                    })
    finally:
        for handle in handles:
            handle.remove()
    destination = run / "diagnostics" / "btrm_mechanism" / split
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "samples.csv").open("w", newline="") as stream:
        fieldnames = list(records[0]) if records else [
            "image", "gt_id", "size_bin", "group", "bg_sampler",
            "x_input", "y_input", "z_contrast", "z_btrm", "g_contrast", "g_btrm",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with (destination / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    summary = _feature_summary(records, per_image, gain)
    summary.update({"run": str(run), "mode": mode, "selected_images": len(selected),
                    "dataset_images": len(images), "seed": seed,
                    "max_gt_per_image": max_gt_per_image,
                    "checkpoint": str(checkpoint)})
    write_json(destination / "summary.json", summary)
    return summary
