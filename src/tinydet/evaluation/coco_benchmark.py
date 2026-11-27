"""Comprehensive C++-accelerated COCO / VisDrone / Tiny-Scale evaluation & diagnostics.

Powered by ``faster-coco-eval`` (C++17 pybind11) and ``torchvision.ops.box_iou``:
1. Standard COCO-100 (``maxDets=100``) + Official VisDrone-500 (``maxDets=500``) AP/AR.
2. Per-class AP50 & AP50-95 across all categories.
3. Micro-scale stratification: ``AP_vt`` (<8^2), ``AP_t`` (8^2..16^2), ``AP_s`` (<32^2), ``AP_m``, ``AP_l``.
4. Occlusion-stratified AP50 (``none``, ``partial``, ``heavy``) when GT annotations include ``occlusion``.
5. TIDE-style 6-component error decomposition (``Cls``, ``Loc``, ``Both``, ``Dupe``, ``Bkg``, ``Miss``).
6. Crowded-pair recall & duplicate rate stratified by nearest-neighbor GT center distance ``d_NN``.
"""

from __future__ import annotations

import io
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from faster_coco_eval import COCO, COCOeval_faster

from tinydet.utils.geometry import iod_matrix, iou_matrix, xywh_to_xyxy

MICRO_AREA_RANGES: list[list[float]] = [
    [0.0, 1e10],       # all
    [0.0, 32.0**2],    # small (<32x32)
    [32.0**2, 96.0**2],  # medium
    [96.0**2, 1e10],   # large
    [0.0, 8.0**2],     # very_tiny (<8x8)
    [8.0**2, 16.0**2],  # tiny (8x8..16x16)
    [16.0**2, 32.0**2],  # sub_small (16x16..32x32)
]
MICRO_AREA_LABELS: list[str] = ["all", "small", "medium", "large", "very_tiny", "tiny", "sub_small"]


def _build_coco_gt(raw_gt: dict[str, Any], *, collapse_single_class: bool = False) -> tuple[COCO, dict[str, int], dict[int, str]]:
    basename_to_id: dict[str, int] = {}
    images = []
    for img in raw_gt["images"]:
        img_id = int(img["id"])
        basename = Path(str(img["file_name"])).name
        basename_to_id[basename] = img_id
        images.append({
            "id": img_id,
            "file_name": str(img["file_name"]),
            "width": int(img["width"]),
            "height": int(img["height"]),
        })

    if collapse_single_class:
        categories = [{"id": 1, "name": "object"}]
        cat_names = {1: "object"}
    else:
        categories = [{"id": int(c["id"]), "name": str(c["name"])} for c in raw_gt["categories"]]
        cat_names = {int(c["id"]): str(c["name"]) for c in raw_gt["categories"]}

    annotations = []
    for idx, ann in enumerate(raw_gt["annotations"], start=1):
        w = float(ann["bbox"][2])
        h = float(ann["bbox"][3])
        if w <= 0.0 or h <= 0.0:
            continue
        is_ign = int(
            bool(ann.get("iscrowd", 0))
            or bool(ann.get("ignore", False))
            or bool(ann.get("uncertain", False))
        )
        cat_id = 1 if collapse_single_class else int(ann["category_id"])
        if cat_id not in cat_names:
            continue
        annotations.append({
            "id": int(ann.get("id", idx)),
            "image_id": int(ann["image_id"]),
            "category_id": cat_id,
            "bbox": [float(v) for v in ann["bbox"]],
            "area": float(ann.get("area", w * h)),
            "iscrowd": is_ign,
            "occlusion": ann.get("occlusion"),
        })

    coco_gt = COCO()
    with redirect_stdout(io.StringIO()):
        coco_gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
        coco_gt.createIndex()
    return coco_gt, basename_to_id, cat_names


def _convert_predictions_to_coco(
    pred_payload: dict[str, Any],
    basename_to_id: dict[str, int],
    cat_ids_sorted: list[int],
    *,
    collapse_single_class: bool = False,
) -> list[dict[str, Any]]:
    use_offset = bool(cat_ids_sorted) and min(cat_ids_sorted) == 1
    coco_dets: list[dict[str, Any]] = []
    for row in pred_payload.get("images", []):
        basename = Path(str(row["path"])).name
        img_id = basename_to_id.get(basename)
        if img_id is None:
            continue
        boxes = row.get("boxes", [])
        scores = row.get("scores", [])
        classes = row.get("classes", [0] * len(boxes))
        for box, score, cls_idx in zip(boxes, scores, classes):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            w = x2 - x1
            h = y2 - y1
            if w <= 0.0 or h <= 0.0:
                continue
            cat_id = 1 if collapse_single_class else (int(cls_idx) + 1 if use_offset else int(cls_idx))
            coco_dets.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "score": float(score),
            })
    return coco_dets


def _run_cocoeval(coco_gt: COCO, coco_dets: list[dict[str, Any]], max_det: int) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    if not coco_dets:
        zero = {
            "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
            "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0,
            "AP_very_tiny": 0.0, "AP_tiny": 0.0, "AP_sub_small": 0.0,
            f"AR@{max_det}": 0.0,
        }
        return zero, {}

    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_dets)
        ev = COCOeval_faster(coco_gt, coco_dt, iouType="bbox")
        ev.params.maxDets = [1, 10, max_det]
        ev.params.areaRng = MICRO_AREA_RANGES
        ev.params.areaRngLbl = MICRO_AREA_LABELS
        ev.evaluate()
        ev.accumulate()

    precision = ev.eval["precision"]  # (T=10, R=101, K, A=7, M=3)
    recall = ev.eval["recall"]        # (T=10, K, A=7, M=3)

    def _mean_valid(arr: np.ndarray) -> float | None:
        valid = arr[arr > -1]
        return float(np.mean(valid)) if valid.size > 0 else None

    summary: dict[str, Any] = {
        "AP": _mean_valid(precision[:, :, :, 0, 2]),
        "AP50": _mean_valid(precision[0, :, :, 0, 2]),
        "AP75": _mean_valid(precision[5, :, :, 0, 2]),
        "AP_small": _mean_valid(precision[:, :, :, 1, 2]),
        "AP50_small": _mean_valid(precision[0, :, :, 1, 2]),
        "AP_medium": _mean_valid(precision[:, :, :, 2, 2]),
        "AP_large": _mean_valid(precision[:, :, :, 3, 2]),
        "AP_very_tiny": _mean_valid(precision[:, :, :, 4, 2]),
        "AP50_very_tiny": _mean_valid(precision[0, :, :, 4, 2]),
        "AP_tiny": _mean_valid(precision[:, :, :, 5, 2]),
        "AP50_tiny": _mean_valid(precision[0, :, :, 5, 2]),
        "AP_sub_small": _mean_valid(precision[:, :, :, 6, 2]),
        f"AR@{max_det}": _mean_valid(recall[:, :, 0, 2]),
        f"AR_small@{max_det}": _mean_valid(recall[:, :, 1, 2]),
    }

    per_class: dict[str, dict[str, Any]] = {}
    cat_ids = ev.params.catIds
    for k_idx, cat_id in enumerate(cat_ids):
        name = coco_gt.cats[cat_id]["name"]
        per_class[name] = {
            "AP": _mean_valid(precision[:, :, k_idx, 0, 2]),
            "AP50": _mean_valid(precision[0, :, k_idx, 0, 2]),
            "AP75": _mean_valid(precision[5, :, k_idx, 0, 2]),
            "AP_small": _mean_valid(precision[:, :, k_idx, 1, 2]),
        }
    return summary, per_class


def _compute_occlusion_ap50(raw_gt: dict[str, Any], coco_dets: list[dict[str, Any]], collapse_single_class: bool) -> dict[str, float | None] | None:
    has_occ = any(ann.get("occlusion") is not None for ann in raw_gt["annotations"])
    if not has_occ:
        return None
    occ_bins = {"no_occlusion": 0, "partial_occlusion": 1, "heavy_occlusion": 2}
    out: dict[str, float | None] = {}
    for label, occ_val in occ_bins.items():
        filtered_anns = []
        for ann in raw_gt["annotations"]:
            is_ign = bool(ann.get("iscrowd", 0)) or bool(ann.get("ignore", False)) or bool(ann.get("uncertain", False))
            if is_ign or ann.get("occlusion") == occ_val:
                filtered_anns.append(ann)
            else:
                filtered_anns.append({**ann, "iscrowd": 1})
        sub_gt, _, _ = _build_coco_gt({**raw_gt, "annotations": filtered_anns}, collapse_single_class=collapse_single_class)
        sub_summary, _ = _run_cocoeval(sub_gt, coco_dets, max_det=500)
        out[label] = sub_summary.get("AP50")
    return out


def _compute_tide_and_crowded_diagnostics(
    raw_gt: dict[str, Any],
    pred_payload: dict[str, Any],
    basename_to_id: dict[str, int],
    *,
    collapse_single_class: bool = False,
    pos_iou: float = 0.50,
    bg_iou: float = 0.10,
    score_floor: float = 0.05,
) -> dict[str, Any]:
    """Compute TIDE 6-category error rates and crowded nearest-neighbor recall/duplicate stratification."""
    cat_ids = sorted({int(c["id"]) for c in raw_gt["categories"]})
    use_offset = bool(cat_ids) and min(cat_ids) == 1

    gt_pos_by_img: dict[int, list[tuple[tuple[float, float, float, float], int]]] = defaultdict(list)
    gt_ign_by_img: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for ann in raw_gt["annotations"]:
        w, h = float(ann["bbox"][2]), float(ann["bbox"][3])
        if w <= 0.0 or h <= 0.0:
            continue
        img_id = int(ann["image_id"])
        xyxy = xywh_to_xyxy(ann["bbox"], validate=False)
        is_ign = bool(ann.get("iscrowd", 0)) or bool(ann.get("ignore", False)) or bool(ann.get("uncertain", False))
        if is_ign:
            gt_ign_by_img[img_id].append(xyxy)
        else:
            cid = 1 if collapse_single_class else int(ann["category_id"])
            gt_pos_by_img[img_id].append((xyxy, cid))

    counts = {
        "TP": 0,
        "FP_Cls": 0,
        "FP_Loc": 0,
        "FP_Both": 0,
        "FP_Dupe": 0,
        "FP_Bkg": 0,
        "FN_Miss": 0,
    }
    nn_bins = {
        "crowded_lt_10px": {"gt": 0, "recalled": 0, "dupes": 0},
        "close_10_20px": {"gt": 0, "recalled": 0, "dupes": 0},
        "isolated_ge_20px": {"gt": 0, "recalled": 0, "dupes": 0},
    }

    pred_by_img: dict[int, list[tuple[list[float], float, int]]] = {}
    for row in pred_payload.get("images", []):
        matched_id = basename_to_id.get(Path(str(row["path"])).name)
        if matched_id is None:
            continue
        dets = []
        for box, score, cls_idx in zip(row.get("boxes", []), row.get("scores", []), row.get("classes", [])):
            if float(score) < score_floor:
                continue
            cid = 1 if collapse_single_class else (int(cls_idx) + 1 if use_offset else int(cls_idx))
            dets.append(([float(v) for v in box], float(score), cid))
        dets.sort(key=lambda item: item[1], reverse=True)
        pred_by_img[matched_id] = dets[:500]

    for img in raw_gt["images"]:
        img_id = int(img["id"])
        gt_items = gt_pos_by_img.get(img_id, [])
        ign_boxes = gt_ign_by_img.get(img_id, [])
        dets = pred_by_img.get(img_id, [])
        n_gt = len(gt_items)
        n_det = len(dets)

        gt_boxes = np.asarray([g[0] for g in gt_items], dtype=np.float64).reshape(-1, 4)
        gt_clses = np.asarray([g[1] for g in gt_items], dtype=np.int64)
        gt_matched = np.zeros(n_gt, dtype=np.bool_)
        gt_dupes = np.zeros(n_gt, dtype=np.int64)

        # Nearest-neighbor center distances among positive GTs
        nn_dist = np.full(n_gt, np.inf, dtype=np.float64)
        if n_gt >= 2:
            centers = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:])
            diff = centers[:, None, :] - centers[None, :, :]
            dmat = np.hypot(diff[..., 0], diff[..., 1])
            np.fill_diagonal(dmat, np.inf)
            nn_dist = dmat.min(axis=1)

        if n_det > 0:
            det_boxes = np.asarray([d[0] for d in dets], dtype=np.float64).reshape(-1, 4)
            det_clses = np.asarray([d[2] for d in dets], dtype=np.int64)
            ious = iou_matrix(det_boxes, gt_boxes) if n_gt > 0 else np.zeros((n_det, 0), dtype=np.float64)
            ign_iod = iod_matrix(det_boxes, ign_boxes) if ign_boxes else np.zeros((n_det, 0), dtype=np.float64)

            for i in range(n_det):
                c_det = det_clses[i]
                if n_gt > 0:
                    same_c = gt_clses == c_det
                    avail_same = same_c & (~gt_matched)
                    if avail_same.any():
                        cand_ious = np.where(avail_same, ious[i], -1.0)
                        best_j = int(cand_ious.argmax())
                        if cand_ious[best_j] >= pos_iou:
                            gt_matched[best_j] = True
                            counts["TP"] += 1
                            continue

                if ign_iod.shape[1] > 0 and (ign_iod[i] >= pos_iou).any():
                    continue

                if n_gt == 0:
                    counts["FP_Bkg"] += 1
                    continue

                row_iou = ious[i]
                max_j = int(row_iou.argmax())
                max_iou = float(row_iou[max_j])
                if max_iou < bg_iou:
                    counts["FP_Bkg"] += 1
                elif max_iou >= pos_iou:
                    if gt_clses[max_j] == c_det and gt_matched[max_j]:
                        counts["FP_Dupe"] += 1
                        gt_dupes[max_j] += 1
                    else:
                        counts["FP_Cls"] += 1
                else:
                    if gt_clses[max_j] == c_det:
                        counts["FP_Loc"] += 1
                    else:
                        counts["FP_Both"] += 1

        counts["FN_Miss"] += int((~gt_matched).sum())
        for j in range(n_gt):
            d = nn_dist[j]
            bkey = "crowded_lt_10px" if d < 10.0 else ("close_10_20px" if d < 20.0 else "isolated_ge_20px")
            nn_bins[bkey]["gt"] += 1
            if gt_matched[j]:
                nn_bins[bkey]["recalled"] += 1
            nn_bins[bkey]["dupes"] += int(gt_dupes[j])

    total_gt = max(sum(b["gt"] for b in nn_bins.values()), 1)
    total_fp = max(
        counts["FP_Cls"] + counts["FP_Loc"] + counts["FP_Both"] + counts["FP_Dupe"] + counts["FP_Bkg"],
        1,
    )
    crowded_summary = {}
    for bkey, bvals in nn_bins.items():
        gt_n = bvals["gt"]
        crowded_summary[bkey] = {
            "ground_truths": gt_n,
            "recall_at_50": float(bvals["recalled"] / gt_n) if gt_n > 0 else None,
            "duplicate_rate_per_gt": float(bvals["dupes"] / gt_n) if gt_n > 0 else None,
        }
    return {
        "operating_score_floor": score_floor,
        "counts": counts,
        "fp_share": {
            k: round(counts[k] / total_fp, 4)
            for k in ("FP_Cls", "FP_Loc", "FP_Both", "FP_Dupe", "FP_Bkg")
        },
        "miss_rate_at_floor": round(counts["FN_Miss"] / total_gt, 4),
        "crowded_pairs": crowded_summary,
    }


def evaluate_coco_suite(
    annotation_file: str | Path,
    predictions_file: str | Path,
    *,
    collapse_single_class: bool = False,
) -> dict[str, Any]:
    """Run the full COCO-100 + VisDrone-500 + Micro-Scale + TIDE + Crowded-Pair suite."""
    raw_gt = orjson.loads(Path(annotation_file).read_bytes())
    pred_payload = orjson.loads(Path(predictions_file).read_bytes())
    coco_gt, basename_to_id, cat_names = _build_coco_gt(
        raw_gt, collapse_single_class=collapse_single_class
    )
    coco_dets = _convert_predictions_to_coco(
        pred_payload,
        basename_to_id,
        sorted(cat_names),
        collapse_single_class=collapse_single_class,
    )
    coco_100, per_class_100 = _run_cocoeval(coco_gt, coco_dets, max_det=100)
    visdrone_500, per_class_500 = _run_cocoeval(coco_gt, coco_dets, max_det=500)
    occlusion_ap50 = _compute_occlusion_ap50(raw_gt, coco_dets, collapse_single_class=collapse_single_class)
    diagnostics = _compute_tide_and_crowded_diagnostics(
        raw_gt, pred_payload, basename_to_id, collapse_single_class=collapse_single_class
    )
    return {
        "annotation_file": str(Path(annotation_file).resolve()),
        "predictions_file": str(Path(predictions_file).resolve()),
        "num_images": len(raw_gt["images"]),
        "num_detections": len(coco_dets),
        "coco_maxdet100": coco_100,
        "visdrone_maxdet500": visdrone_500,
        "per_class_maxdet100": per_class_100,
        "per_class_maxdet500": per_class_500,
        "occlusion_ap50": occlusion_ap50,
        "tide_and_crowded": diagnostics,
    }
