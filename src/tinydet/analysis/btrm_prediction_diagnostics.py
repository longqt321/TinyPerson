"""Size-binned precision/recall diagnostics from SAVED predictions.pt files.

No training and no prediction jobs. Matching mirrors the custom size-AP
protocol: target-bin GT first, excluded-bin GT cause ignored detections.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from tinydet.evaluation.size_metrics import (
    iou,
    label_path,
    parse_annotation_box,
    size_bin,
)
from tinydet.utils.experiment import write_json

BINS = ("0-8", "8-16", "16-32", ">32")
THRESHOLDS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5)


def load_saved_samples(run: Path, split: str = "val"):
    artifact = run / f"evaluation-{split}-size" / "predictions.pt"
    if not artifact.is_file():
        raise FileNotFoundError(f"Missing saved size predictions: {artifact}")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if "imgsz" not in payload or not isinstance(payload.get("predictions"), list):
        raise ValueError("Invalid size prediction artifact")
    config = yaml.safe_load((run / "config.yaml").read_text())
    if int(payload["imgsz"]) != int(config["train"]["imgsz"]):
        raise ValueError("Incompatible saved image size")
    samples = []
    for result in payload["predictions"]:
        h, w = result["shape"]
        scale = min(payload["imgsz"] / w, payload["imgsz"] / h)
        gt = []
        label = label_path(Path(result["image"]))
        if not label.is_file():
            raise FileNotFoundError(f"GT label missing: {label}")
        for index, line in enumerate(label.read_text().splitlines()):
            if not line.strip():
                continue
            box = parse_annotation_box(line)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            gt.append({
                "id": index, "box": [x1 * w, y1 * h, x2 * w, y2 * h],
                "bin": size_bin((x2-x1) * w * scale, (y2-y1) * h * scale),
            })
        pred = [
            (box.numpy(), float(score))
            for box, score, cls in zip(result["boxes"], result["scores"], result["classes"])
            if int(cls) == 0
        ]
        samples.append({"image": str(Path(result["image"])), "gt": gt, "pred": pred})
    return samples


def _prepare(samples):
    return sorted(
        ((score, image_idx, iou(box, [item["box"] for item in sample["gt"]]))
         for image_idx, sample in enumerate(samples)
         for box, score in sample["pred"]),
        key=lambda t: t[0], reverse=True,
    )


def confidence_counts(samples, prepared, size: str, conf: float) -> dict:
    matched = [set() for _ in samples]
    tp, fp, ignored = 0, 0, 0
    tp_confidences, fp_confidences = [], []
    for score, image_idx, overlaps in prepared:
        if score < conf:
            break
        gt = samples[image_idx]["gt"]
        eligible = [
            j for j, item in enumerate(gt)
            if item["bin"] == size and j not in matched[image_idx]
        ]
        candidates = overlaps[eligible]
        if len(candidates) and candidates.max() >= .5:
            j = eligible[int(candidates.argmax())]
            matched[image_idx].add(j)
            tp += 1
            tp_confidences.append(score)
        elif any(item["bin"] != size and overlaps[j] >= .5
                 for j, item in enumerate(gt)):
            ignored += 1
        else:
            fp += 1
            fp_confidences.append(score)
    n_gt = sum(item["bin"] == size for sample in samples for item in sample["gt"])
    hits = {
        (sample["image"], sample["gt"][j]["id"])
        for idx, sample in enumerate(samples)
        for j in matched[idx]
    }
    summary = {
        "bin": size, "confidence_threshold": conf, "images": len(samples),
        "GT": n_gt, "TP": tp, "FP": fp, "ignored_predictions": ignored,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / n_gt if n_gt else None,
        "false_positives_per_image": fp / len(samples) if samples else None,
        "TP_conf_median": float(np.median(tp_confidences)) if tp_confidences else None,
        "FP_conf_median": float(np.median(fp_confidences)) if fp_confidences else None,
    }
    return summary, hits


def analyze_prediction_runs(references: dict[str, Path], out: Path, split: str = "val") -> dict:
    if len(references) != 3:
        raise ValueError("Require A0/A2/A3 completed experiment artifacts")
    results, gt_sets, samples_by_model = {}, {}, {}
    for model, run in references.items():
        samples = load_saved_samples(run, split)
        prepared = _prepare(samples)
        results[model] = []
        for size in BINS:
            for threshold in THRESHOLDS:
                row, hits = confidence_counts(samples, prepared, size, threshold)
                row["model"] = model
                results[model].append(row)
                gt_sets[(model, size, threshold)] = hits
        samples_by_model[model] = samples
    # All arms must evaluate identical images and identical GT labels.
    source = samples_by_model["btrm_a0"]
    for model in ("btrm_a2", "btrm_a3"):
        other = samples_by_model[model]
        if [(x["image"], x["gt"]) for x in source] != [
            (x["image"], x["gt"]) for x in other
        ]:
            raise ValueError("Dataset/GT differs between experiment runs")

    deltas = []
    model_to_comp = (
        ("A2_minus_A0", "btrm_a2", "btrm_a0"),
        ("A3_minus_A2", "btrm_a3", "btrm_a2"),
        ("A3_minus_A0", "btrm_a3", "btrm_a0"),
    )
    lookup = {
        (row["model"], row["bin"], row["confidence_threshold"]): row
        for items in results.values() for row in items
    }
    gained = []
    for label, after, before in model_to_comp:
        for size in BINS:
            for threshold in THRESHOLDS:
                aa = lookup[(after, size, threshold)]
                bb = lookup[(before, size, threshold)]
                ra = gt_sets[(after, size, threshold)]
                rb = gt_sets[(before, size, threshold)]
                deltas.append({
                    "comparison": label, "bin": size, "confidence_threshold": threshold,
                    "delta_TP": aa["TP"] - bb["TP"], "delta_FP": aa["FP"] - bb["FP"],
                    "delta_precision": ((aa["precision"] - bb["precision"])
                                        if aa["precision"] is not None and bb["precision"] is not None
                                        else None),
                    "delta_recall": ((aa["recall"] - bb["recall"])
                                     if aa["recall"] is not None and bb["recall"] is not None
                                     else None),
                    "newly_matched_GT": len(ra - rb),
                    "previously_matched_GT_lost": len(rb - ra),
                })
                if label == "A3_minus_A0" and size == "0-8" and threshold == .001:
                    for image, gt_id in sorted(ra - rb):
                        gained.append({
                            "change": "gained_by_A3", "image": image,
                            "gt_id": gt_id, "bin": size, "confidence": threshold,
                        })
                    for image, gt_id in sorted(rb - ra):
                        gained.append({
                            "change": "lost_by_A3", "image": image,
                            "gt_id": gt_id, "bin": size, "confidence": threshold,
                        })
    out.mkdir(parents=True, exist_ok=True)
    rows = [row for items in results.values() for row in items]
    for name, values in (
        ("precision_recall_thresholds.csv", rows),
        ("matched_deltas.csv", deltas),
        ("tiny_gt_changes.csv", gained),
    ):
        if not values:
            continue
        with (out / name).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    output = {
        "models": list(references),
        "PR_thresholds": len(THRESHOLDS),
        "size_bins": list(BINS),
        "tiny_recall_A3_minus_A0_at_0.001": next(
            x for x in deltas if x["comparison"] == "A3_minus_A0"
            and x["bin"] == "0-8" and x["confidence_threshold"] == .001
        ),
        "caveats": [
            "FP depends on the custom size-bin ignored-GT matching protocol.",
            "Counts use the saved conf>=0.001 predictions, not raw logits or every model proposal.",
            "Prediction files must come from the same dataset and postprocessing settings.",
            "Any gained GT counts are one-seed descriptive observations, not significance.",
        ],
    }
    write_json(out / "prediction_summary.json", output)
    return output
