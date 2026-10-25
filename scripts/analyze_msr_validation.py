"""Run matched per-object error analysis for trained TinyPerson checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

SIZE_BINS = ((0, 8, "0-8px"), (8, 16, "8-16px"), (16, 32, "16-32px"), (32, float("inf"), "32px+"))
GEOMETRY_BINS = ((0, 1, "<=1px"), (1, 2, "1-2px"), (2, 3, "2-3px"), (3, float("inf"), ">3px"))


def read_labels(path: Path, width: int, height: int) -> list[list[float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected YOLO detection row in {path}: {line!r}")
        cls, cx, cy, bw, bh = map(float, fields)
        if int(cls) != 0:
            continue
        boxes.append([
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ])
    return boxes


def iou_matrix(predictions: list[dict], targets: list[list[float]]) -> list[list[float]]:
    matrix = []
    for prediction in predictions:
        px1, py1, px2, py2 = prediction["box"]
        row = []
        for gx1, gy1, gx2, gy2 in targets:
            ix1, iy1 = max(px1, gx1), max(py1, gy1)
            ix2, iy2 = min(px2, gx2), min(py2, gy2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            p_area = max(0, px2 - px1) * max(0, py2 - py1)
            g_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
            row.append(intersection / (p_area + g_area - intersection + 1e-9))
        matrix.append(row)
    return matrix


def size_bin(box: list[float], width: int, height: int, imgsz: int) -> tuple[str, float]:
    side = max(box[2] - box[0], box[3] - box[1])
    side *= min(imgsz / width, imgsz / height)
    for lower, upper, name in SIZE_BINS:
        if lower <= side < upper:
            return name, side
    raise AssertionError(side)


def box_geometry_error(prediction: list[float], target: list[float]) -> dict[str, float]:
    px1, py1, px2, py2 = prediction
    gx1, gy1, gx2, gy2 = target
    return {
        "best_dx_px": round(((px1 + px2) - (gx1 + gx2)) / 2, 3),
        "best_dy_px": round(((py1 + py2) - (gy1 + gy2)) / 2, 3),
        "best_dw_px": round((px2 - px1) - (gx2 - gx1), 3),
        "best_dh_px": round((py2 - py1) - (gy2 - gy1), 3),
        "best_center_error_px": round(
            ((((px1 + px2) - (gx1 + gx2)) / 2) ** 2
             + (((py1 + py2) - (gy1 + gy2)) / 2) ** 2) ** 0.5,
            3,
        ),
    }


def geometry_summary(rows: list[dict]) -> list[dict]:
    summary = []
    for model_name in sorted({row["model"] for row in rows}):
        for bin_name in ("all", *(name for _, _, name in SIZE_BINS)):
            selected = [row for row in rows if row["model"] == model_name
                        and (bin_name == "all" or row["size_bin"] == bin_name)
                        and row["best_center_error_px"] is not None]
            if not selected:
                continue
            center_errors = sorted(row["best_center_error_px"] for row in selected)
            row = {
                "model": model_name,
                "size_bin": bin_name,
                "objects_with_prediction": len(selected),
                "best_iou_ge_0.50": sum(r["best_iou"] >= 0.50 for r in selected),
                "best_iou_ge_0.75": sum(r["best_iou"] >= 0.75 for r in selected),
                "center_error_median_px": round(statistics.median(center_errors), 3),
                "center_error_p90_px": round(center_errors[int(0.9 * (len(center_errors) - 1))], 3),
                "dx_median_px": round(statistics.median(r["best_dx_px"] for r in selected), 3),
                "dy_median_px": round(statistics.median(r["best_dy_px"] for r in selected), 3),
                "dw_median_px": round(statistics.median(r["best_dw_px"] for r in selected), 3),
                "dh_median_px": round(statistics.median(r["best_dh_px"] for r in selected), 3),
            }
            for lower, upper, name in GEOMETRY_BINS:
                row[f"center_error_{name}"] = sum(
                    lower <= error < upper for error in center_errors
                )
            summary.append(row)
    return summary


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze_image(
    model_name: str,
    image_path: Path,
    target_boxes: list[list[float]],
    predictions: list[dict],
    image_size: tuple[int, int],
    imgsz: int,
    conf: float,
    iou_threshold: float,
) -> tuple[list[dict], list[dict]]:
    width, height = image_size
    ious = iou_matrix(predictions, target_boxes)
    assigned_targets: set[int] = set()
    assigned_predictions: set[int] = set()
    for pred_idx in sorted(range(len(predictions)), key=lambda i: predictions[i]["score"], reverse=True):
        if predictions[pred_idx]["score"] < conf or not target_boxes:
            continue
        target_idx = max(range(len(target_boxes)), key=lambda j: ious[pred_idx][j])
        if ious[pred_idx][target_idx] >= iou_threshold and target_idx not in assigned_targets:
            assigned_targets.add(target_idx)
            assigned_predictions.add(pred_idx)

    object_rows = []
    for target_idx, target in enumerate(target_boxes):
        bin_name, side = size_bin(target, width, height, imgsz)
        candidate_idxs = sorted(range(len(predictions)), key=lambda i: ious[i][target_idx], reverse=True)
        best_idx = candidate_idxs[0] if candidate_idxs else None
        matched_idx = next((i for i in candidate_idxs
                            if i in assigned_predictions and ious[i][target_idx] >= iou_threshold), None)
        if matched_idx is not None:
            outcome = "detected"
            selected_idx = matched_idx
        else:
            localized = next((i for i in candidate_idxs
                              if predictions[i]["score"] >= 0.05 and ious[i][target_idx] >= 0.1), None)
            outcome = "localization_error" if localized is not None else "miss"
            selected_idx = localized if localized is not None else best_idx
        geometry = (box_geometry_error(predictions[best_idx]["box"], target)
                    if best_idx is not None else {
                        "best_dx_px": None, "best_dy_px": None, "best_dw_px": None,
                        "best_dh_px": None, "best_center_error_px": None,
                    })
        object_rows.append({
            "model": model_name,
            "image": image_path.name,
            "target_index": target_idx,
            "size_bin": bin_name,
            "target_max_side_input_px": round(side, 3),
            "gt_xyxy": json.dumps([round(v, 3) for v in target]),
            "outcome": outcome,
            "best_iou": round(ious[best_idx][target_idx], 6) if best_idx is not None else 0.0,
            "best_confidence": predictions[selected_idx]["score"] if selected_idx is not None else 0.0,
            "matched_iou": round(ious[matched_idx][target_idx], 6) if matched_idx is not None else 0.0,
            "matched_confidence": predictions[matched_idx]["score"] if matched_idx is not None else 0.0,
            **geometry,
        })

    prediction_rows = []
    for pred_idx, prediction in enumerate(predictions):
        if pred_idx in assigned_predictions or prediction["score"] < conf:
            continue
        best_iou = max(ious[pred_idx], default=0.0)
        outcome = "duplicate" if best_iou >= iou_threshold else (
            "localization_fp" if best_iou >= 0.1 else "background_fp"
        )
        prediction_rows.append({
            "model": model_name,
            "image": image_path.name,
            "prediction_index": pred_idx,
            "confidence": prediction["score"],
            "max_gt_iou": round(best_iou, 6),
            "outcome": outcome,
            "pred_xyxy": json.dumps([round(v, 3) for v in prediction["box"]]),
        })
    return object_rows, prediction_rows


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=/path/to/best.pt")
    name, path = value.split("=", 1)
    checkpoint = Path(path)
    if not name or not checkpoint.is_file():
        raise argparse.ArgumentTypeError(f"Invalid checkpoint: {value}")
    return name, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001, help="Low prediction floor for error analysis")
    parser.add_argument("--match-conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--max-hard-cases", type=int, default=12)
    args = parser.parse_args()
    if args.imgsz <= 0 or args.cpu_threads <= 0 or args.max_det <= 0:
        parser.error("imgsz, cpu-threads and max-det must be positive")
    if not args.images.is_dir() or not args.labels.is_dir():
        parser.error("images and labels must be directories")
    images = sorted(path for path in args.images.iterdir() if path.is_file())
    if not images:
        parser.error("no images found")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions").mkdir(exist_ok=True)
    torch.set_num_threads(args.cpu_threads)

    all_objects: list[dict] = []
    all_predictions: list[dict] = []
    model_summaries = []
    size_summaries = []
    comparison: dict[tuple[str, int], dict] = {}
    for model_name, checkpoint in args.checkpoint:
        print(f"Predicting {model_name} from {checkpoint}", flush=True)
        model = YOLO(str(checkpoint))
        raw_predictions: dict[str, list[dict]] = {}
        results = model.predict(
            source=[str(path.resolve()) for path in images],
            stream=True,
            imgsz=args.imgsz,
            batch=1,
            conf=args.conf,
            max_det=args.max_det,
            classes=[0],
            device=args.device,
            verbose=False,
        )
        for result in results:
            boxes = result.boxes
            raw_predictions[Path(result.path).name] = [] if boxes is None else [
                {"box": [float(v) for v in box], "score": float(score)}
                for box, score in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist())
            ]
        if len(raw_predictions) != len(images):
            raise RuntimeError(f"{model_name}: predicted {len(raw_predictions)} of {len(images)} images")
        (args.output / "predictions" / f"{model_name}.json").write_text(
            json.dumps({"checkpoint": str(checkpoint.resolve()), "imgsz": args.imgsz,
                        "conf_floor": args.conf, "max_det": args.max_det,
                        "images": raw_predictions}, indent=2) + "\n"
        )

        model_objects = []
        model_preds = []
        image_lookup = {path.name: path for path in images}
        for image_path in images:
            with Image.open(image_path) as image:
                width, height = image.size
            targets = read_labels(args.labels / f"{image_path.stem}.txt", width, height)
            object_rows, prediction_rows = analyze_image(
                model_name, image_path, targets, raw_predictions[image_path.name],
                (width, height), args.imgsz, args.match_conf, args.iou,
            )
            model_objects.extend(object_rows)
            model_preds.extend(prediction_rows)
            for row in object_rows:
                comparison.setdefault((row["image"], row["target_index"]), {
                    "image": row["image"], "target_index": row["target_index"],
                    "size_bin": row["size_bin"],
                    "target_max_side_input_px": row["target_max_side_input_px"],
                })[f"{model_name}_outcome"] = row["outcome"]
                comparison[(row["image"], row["target_index"])][f"{model_name}_best_iou"] = row["best_iou"]
                comparison[(row["image"], row["target_index"])][f"{model_name}_confidence"] = row["best_confidence"]
        all_objects.extend(model_objects)
        all_predictions.extend(model_preds)
        by_bin = []
        for _, _, bin_name in SIZE_BINS:
            rows = [r for r in model_objects if r["size_bin"] == bin_name]
            detected = sum(r["outcome"] == "detected" for r in rows)
            by_bin.append({"size_bin": bin_name, "objects": len(rows), "detected": detected,
                           "misses": sum(r["outcome"] == "miss" for r in rows),
                           "localization_errors": sum(r["outcome"] == "localization_error" for r in rows),
                           "recall_conf025_iou050": detected / len(rows) if rows else None})
        size_summaries.extend({"model": model_name, **row} for row in by_bin)
        run_dir = checkpoint.parents[2]
        metadata_path = run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        results_csv = checkpoint.parents[1] / "results.csv"
        epoch_rows = list(csv.DictReader(results_csv.open())) if results_csv.exists() else []
        completed_epochs = len(epoch_rows)
        model_summaries.append({
            "model": model_name,
            "checkpoint": str(checkpoint.resolve()),
            "run_status": metadata.get("status", "unknown"),
            "completed_epochs": completed_epochs,
            "images": len(images),
            "objects": len(model_objects),
            "detected": sum(r["outcome"] == "detected" for r in model_objects),
            "misses": sum(r["outcome"] == "miss" for r in model_objects),
            "localization_errors": sum(r["outcome"] == "localization_error" for r in model_objects),
            "false_positives": len(model_preds),
            "background_fp": sum(r["outcome"] == "background_fp" for r in model_preds),
            "duplicate": sum(r["outcome"] == "duplicate" for r in model_preds),
            "localization_fp": sum(r["outcome"] == "localization_fp" for r in model_preds),
            "recall_conf025_iou050": sum(r["outcome"] == "detected" for r in model_objects) / max(1, len(model_objects)),
        })
        # Render the smallest misses first so hard-case review is deterministic.
        misses = sorted((r for r in model_objects if r["outcome"] == "miss"),
                        key=lambda r: (r["target_max_side_input_px"], r["image"]))
        hard_dir = args.output / "hard_cases" / model_name
        for row in misses[:args.max_hard_cases]:
            image_path = image_lookup[row["image"]]
            with Image.open(image_path) as image:
                width, height = image.size
            targets = read_labels(args.labels / f"{image_path.stem}.txt", width, height)
            # Save the rendered copy beside the report without modifying source images.
            rendered = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(rendered)
            for index, box in enumerate(targets):
                draw.rectangle(box, outline="#00d26a", width=2)
                draw.text((box[0], max(0, box[1] - 12)), f"GT {index}", fill="#00d26a")
            for prediction in raw_predictions[image_path.name]:
                if prediction["score"] >= 0.05:
                    box = prediction["box"]
                    draw.rectangle(box, outline="#ff3434", width=2)
                    draw.text((box[0], max(0, box[1] - 12)), f"P {prediction['score']:.2f}", fill="#ff3434")
            hard_dir.mkdir(parents=True, exist_ok=True)
            rendered.save(hard_dir / f"{image_path.stem}_miss.png")
        del model
        print(f"{model_name}: objects={len(model_objects)}, misses={model_summaries[-1]['misses']}, "
              f"recall@conf0.25/IoU0.5={model_summaries[-1]['recall_conf025_iou050']:.4f} "
              f"(epoch rows={completed_epochs}, status={model_summaries[-1]['run_status']})", flush=True)

    write_csv(args.output / "per_object_errors.csv", all_objects)
    write_csv(args.output / "per_prediction_errors.csv", all_predictions)
    write_csv(args.output / "geometry_summary.csv", geometry_summary(all_objects))
    write_csv(args.output / "error_summary.csv", model_summaries)
    write_csv(args.output / "size_summary.csv", size_summaries)
    write_csv(args.output / "model_comparison.csv", list(comparison.values()))
    report = {
        "images": len(images),
        "imgsz": args.imgsz,
        "prediction_confidence_floor": args.conf,
        "matching_confidence_threshold": args.match_conf,
        "matching_iou_threshold": args.iou,
        "max_detections": args.max_det,
        "size_bins": [name for _, _, name in SIZE_BINS],
        "matching_policy": "descending-confidence greedy one-to-one match at confidence >= match-conf and IoU >= threshold",
        "models": model_summaries,
        "limitations": [
            "YOLO validation labels do not encode official COCO ignore/uncertain regions or dense-image handling.",
            "The four checkpoints have different completed epoch counts; this is descriptive, not a controlled architecture comparison.",
            "FP type thresholds (0.05 confidence, 0.1 IoU) are diagnostic conventions, not official benchmark rules.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "summary": model_summaries}, indent=2))


if __name__ == "__main__":
    main()
