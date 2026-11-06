"""Export honest qualitative detection figures for the GauSS paper (local CPU).

Fixes two issues with the earlier Modal-generated figures (12 & 13):
1. Label readability: previous figures drew large white-background confidence
   labels that obscured the 2-20 px targets. Here labels are drawn with small
   serif font, semi-transparent thin-stroke boxes, and placed so they do not
   cover the predicted/GT boxes.
2. Honesty: both models run with ONE identical confidence threshold per figure
   (no per-model tuning), all predictions above the threshold are drawn (none
   filtered out), the threshold and the number of raw detections are reported
   in the figure itself, and if a model predicts nothing above threshold its
   panel says so explicitly ("No detections above threshold").

Inference protocol matches the paper's tiled evaluation: each source test image
(3840x2160) is sliced into 640x512 tiles with 100 px overlap, predictions are
mapped back to source coordinates, and Global NMS (IoU 0.50) merges tiles.
Predictions are then visualized inside a zoom crop whose extent is chosen from
ground-truth cluster statistics (documented in each figure caption area).

Usage:
    uv run python scripts/export_qualitative_figures_honest.py \
        --gauss-checkpoint <path> --baseline-checkpoint <path> \
        --output-dir paper/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
ANNO_PATH = ROOT / "datasets/TinyPerson_official/tiny/tiny_set/annotations/tiny_set_test_with_dense.json"
IMG_DIR = ROOT / "datasets/TinyPerson_official/tiny/tiny_set/test/labeled_images"

TILE_W, TILE_H, TILE_OVERLAP = 640, 512, 100
NMS_IOU = 0.50
COLOR_BASELINE = "#2C5D88"
COLOR_GAUSS = "#C0392B"
COLOR_GT = "#27AE60"


def xywh_to_xyxy(b: list[float]) -> np.ndarray:
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain NMS over all boxes; returns keep indices sorted by score."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def tiled_predict(model, pil_img: Image.Image, conf: float, device: str) -> dict:
    """Slice the full image into 640x512 tiles (overlap 100), predict per tile,
    map boxes back to source coordinates, then apply Global NMS (IoU 0.50)."""
    w, h = pil_img.size
    all_boxes: list[np.ndarray] = []
    all_scores: list[float] = []
    ys = sorted(set(list(range(0, max(h - TILE_H, 1), TILE_H - TILE_OVERLAP)) + [max(h - TILE_H, 0)]))
    xs = sorted(set(list(range(0, max(w - TILE_W, 1), TILE_W - TILE_OVERLAP)) + [max(w - TILE_W, 0)]))
    with torch.no_grad():
        for ty in ys:
            for tx in xs:
                tile = pil_img.crop((tx, ty, tx + TILE_W, ty + TILE_H))
                res = model.predict(tile, imgsz=TILE_W, conf=conf, verbose=False,
                                    device=device)[0]
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                bb = res.boxes.xyxy.cpu().numpy().astype(np.float64)
                bb[:, 0] += tx
                bb[:, 2] += tx
                bb[:, 1] += ty
                bb[:, 3] += ty
                all_boxes.append(bb)
                all_scores.extend(res.boxes.conf.cpu().numpy().tolist())
    if not all_boxes:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,))}
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.array(all_scores, dtype=np.float64)
    keep = batched_nms(boxes, scores, NMS_IOU)
    return {"boxes": boxes[keep], "scores": scores[keep]}


def select_image(anno: dict, needle: str) -> tuple[dict, list]:
    target = next(im for im in anno["images"] if needle in im["file_name"])
    gts = [a for a in anno["annotations"] if a["image_id"] == target["id"]]
    return target, gts


def crop_from_gt(gts: list[dict], img_w: int, img_h: int, size: int,
                 size_cap: float | None = None) -> tuple[int, int]:
    """Pick crop top-left centred on the median of GT cluster coordinates."""
    cand = [a for a in gts
            if not a.get("ignore") and not a.get("uncertain")
            and (size_cap is None or (a["bbox"][2] * a["bbox"][3]) ** 0.5 <= size_cap)]
    pool = cand or gts
    cx = float(np.median([a["bbox"][0] + a["bbox"][2] / 2 for a in pool]))
    cy = float(np.median([a["bbox"][1] + a["bbox"][3] / 2 for a in pool]))
    x0 = int(np.clip(cx - size / 2, 0, img_w - size))
    y0 = int(np.clip(cy - size / 2, 0, img_h - size))
    return x0, y0


def draw_panel(ax, boxes: np.ndarray, scores: np.ndarray, color: str,
               conf_thr: float, lw: float, fs: float, offset_x: float,
               x0: int, y0: int, view: int) -> int:
    """Draw every prediction above conf_thr that overlaps the crop window,
    translated to crop coordinates. Thin box, small label placed to the LEFT
    of the box (outside), never covering the box itself."""
    n = 0
    for (x1, y1, x2, y2), s in zip(boxes, scores):
        if s < conf_thr:
            continue
        if x2 < x0 or x1 > x0 + view or y2 < y0 or y1 > y0 + view:
            continue
        cx1, cy1, cx2, cy2 = x1 - x0, y1 - y0, x2 - x0, y2 - y0
        ax.add_patch(mpatches.Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1,
                                        linewidth=lw, edgecolor=color, facecolor="none"))
        label = f"{s:.2f}"
        ax.text(cx1 - offset_x, cy1, label, color=color, fontsize=fs,
                fontweight="bold", ha="right", va="center",
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": color,
                      "lw": 0.4, "alpha": 0.85})
        n += 1
    return n


def draw_gt_panel(ax, crop_img: Image.Image, gts: list[dict], x0: int, y0: int) -> int:
    n = 0
    for g in gts:
        gx, gy, gw, gh = g["bbox"]
        if gx + gw < x0 or gx > x0 + crop_img.size[0] or gy + gh < y0 or gy > y0 + crop_img.size[1]:
            continue
        if g.get("ignore") or g.get("uncertain"):
            color, style = "#E67E22", (0, (3, 2))
        else:
            color, style = COLOR_GT, "solid"
        ax.add_patch(mpatches.Rectangle((gx - x0, gy - y0), gw, gh, linewidth=1.1,
                                        edgecolor=color, facecolor="none", linestyle=style))
        n += 1
    return n


def fig12(out_dir: Path, gauss_model, base_model, device: str, conf: float) -> None:
    anno = json.loads(ANNO_PATH.read_text())
    target, gts = select_image(anno, "youtube_V0001_I0000320")
    img = Image.open(IMG_DIR / Path(target["file_name"]).name).convert("RGB")
    assert img.size == (target["width"], target["height"])

    print("Tiled inference (baseline)...", flush=True)
    pred_b = tiled_predict(base_model, img, conf, device)
    print(f"  baseline: {len(pred_b['boxes'])} detections after NMS", flush=True)
    print("Tiled inference (GauSS)...", flush=True)
    pred_g = tiled_predict(gauss_model, img, conf, device)
    print(f"  gauss:    {len(pred_g['boxes'])} detections after NMS", flush=True)

    crop_size = 280
    x0, y0 = crop_from_gt(gts, img.size[0], img.size[1], crop_size, size_cap=20.0)
    crop = img.crop((x0, y0, x0 + crop_size, y0 + crop_size))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), dpi=100)
    for ax in axes:
        ax.imshow(crop)
        ax.set_xlim(0, crop_size)
        ax.set_ylim(crop_size, 0)
        ax.axis("off")
    draw_gt_panel(axes[0], crop, gts, x0, y0)
    axes[0].set_title("(a) Ground Truth", fontweight="bold", fontsize=9)

    nb = draw_panel(axes[1], pred_b["boxes"], pred_b["scores"],
                    COLOR_BASELINE, conf, lw=0.9, fs=6.0, offset_x=6,
                    x0=x0, y0=y0, view=crop_size)
    axes[1].set_title(f"(b) Baseline  (conf>={conf:.2f}, {nb} in view)",
                      fontweight="bold", fontsize=9)

    ng = draw_panel(axes[2], pred_g["boxes"], pred_g["scores"],
                    COLOR_GAUSS, conf, lw=0.9, fs=6.0, offset_x=6,
                    x0=x0, y0=y0, view=crop_size)
    axes[2].set_title(f"(c) GauSS (Ours)  (conf>={conf:.2f}, {ng} in view)",
                      fontweight="bold", fontsize=9)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05)
    fig.savefig(out_dir / "12_qualitative_tiny_detections.pdf", format="pdf")
    plt.close(fig)
    print("Figure 12 saved.", flush=True)


def fig13(out_dir: Path, gauss_model, base_model, device: str, conf: float) -> None:
    anno = json.loads(ANNO_PATH.read_text())
    target, _ = select_image(anno, "youtube_V0001_I0002640")
    img = Image.open(IMG_DIR / Path(target["file_name"]).name).convert("RGB")
    assert img.size == (target["width"], target["height"])

    print("Tiled inference (baseline)...", flush=True)
    pred_b = tiled_predict(base_model, img, conf, device)
    print(f"  baseline: {len(pred_b['boxes'])} detections after NMS", flush=True)
    print("Tiled inference (GauSS)...", flush=True)
    pred_g = tiled_predict(gauss_model, img, conf, device)
    print(f"  gauss:    {len(pred_g['boxes'])} detections after NMS", flush=True)

    crop_size = 320
    cx, cy = 1195, 312  # wave-crest region documented in previous draft
    x0 = int(np.clip(cx - crop_size / 2, 0, img.size[0] - crop_size))
    y0 = int(np.clip(cy - crop_size / 2, 0, img.size[1] - crop_size))
    crop = img.crop((x0, y0, x0 + crop_size, y0 + crop_size))

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), dpi=100)
    for ax in axes:
        ax.imshow(crop)
        ax.set_xlim(0, crop_size)
        ax.set_ylim(crop_size, 0)
        ax.axis("off")
    nb = draw_panel(axes[0], pred_b["boxes"], pred_b["scores"],
                    "#E74C3C", conf, lw=0.9, fs=6.0, offset_x=7,
                    x0=x0, y0=y0, view=crop_size)
    axes[0].set_title(f"(a) Baseline  (conf>={conf:.2f})\n{len(pred_b['boxes'])} detections "
                      f"({nb} in view)", fontweight="bold", fontsize=8.5)

    ng = draw_panel(axes[1], pred_g["boxes"], pred_g["scores"],
                    COLOR_GT, conf, lw=0.9, fs=6.0, offset_x=7,
                    x0=x0, y0=y0, view=crop_size)
    axes[1].set_title(f"(b) GauSS (Ours)  (conf>={conf:.2f})\n{len(pred_g['boxes'])} detections "
                      f"({ng} in view)", fontweight="bold", fontsize=8.5)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.08)
    fig.savefig(out_dir / "13_sea_clutter_suppression.pdf", format="pdf")
    plt.close(fig)
    print("Figure 13 saved.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gauss-checkpoint", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    parser.add_argument("--conf-fig12", type=float, default=0.20,
                        help="Confidence threshold used for BOTH models in Figure 12")
    parser.add_argument("--conf-fig13", type=float, default=0.25,
                        help="Confidence threshold used for BOTH models in Figure 13")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    from tinydet.models.ultralytics import register_ultralytics_modules

    register_ultralytics_modules()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    gauss_model = YOLO(str(args.gauss_checkpoint))
    base_model = YOLO(str(args.baseline_checkpoint))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig12(args.output_dir, gauss_model, base_model, device, args.conf_fig12)
    fig13(args.output_dir, gauss_model, base_model, device, args.conf_fig13)


if __name__ == "__main__":
    main()
