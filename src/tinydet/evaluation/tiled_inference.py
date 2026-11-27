"""Sliding window tiled inference and coordinate merge matching TOV_mmdetection protocol.

Tile parameters:
    Tile size: 640x512 (width=640, height=512)
    Overlap: 100x100
    NMS merge IoU threshold: 0.50
    Pre-NMS max candidates: 2000
    Post-NMS max detections: 1000
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms as tv_batched_nms
from torchvision.ops import box_area, nms
from ultralytics import YOLO

# Canonical tile grid shared by every tiled evaluation / training entry point
# (must match the protocol that produced the 0.3540 tiled baseline).
TILE_W, TILE_H, TILE_OVERLAP = 640, 512, 100
NMS_IOU = 0.50
MERGE_IOS = 0.85


def tile_origins(width: int, height: int) -> list[tuple[int, int]]:
    """Top-left (x, y) corners of the canonical tile grid for a WxH image.

    Same rule as :func:`get_image_slices`, expressed as origins.
    """
    return [(x1, y1) for x1, y1, _, _ in get_image_slices(width, height)]


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float = NMS_IOU,
    classes: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy NMS via TorchVision (class-aware when ``classes`` is given)."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    boxes_t = torch.from_numpy(boxes).float()
    scores_t = torch.from_numpy(scores).float()
    if classes is not None:
        classes_t = torch.from_numpy(classes).long()
        keep = tv_batched_nms(boxes_t, scores_t, classes_t, iou_thr).numpy()
        return keep[np.argsort(-scores[keep], kind="stable")].astype(np.int64, copy=False)
    return nms(boxes_t, scores_t, iou_thr).numpy().astype(np.int64, copy=False)


def suppress_border_truncated(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    ios_thr: float = MERGE_IOS,
) -> torch.Tensor:
    """Suppress lower-scoring same-class tile-border fragments contained inside larger boxes.

    At tile seams, a truncated half-person prediction and the adjacent tile's full-person
    prediction often have ``IoU < 0.50`` despite ``IOS = inter / min(area_i, area_j) ~ 1.0``.
    ``boxes``/``scores``/``classes`` must already be sorted in descending score order.
    """
    del scores
    n = boxes.shape[0]
    if n <= 1 or ios_thr <= 0.0 or ios_thr > 1.0:
        return torch.arange(n, device=boxes.device, dtype=torch.long)

    areas = box_area(boxes).clamp(min=1e-6)
    lt = torch.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = torch.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    min_area = torch.minimum(areas[:, None], areas[None, :])
    sup_mat = ((inter >= ios_thr * min_area) & (classes[:, None] == classes[None, :])).cpu().numpy()

    keep_np = np.ones(n, dtype=bool)
    for i in range(n):
        if keep_np[i]:
            keep_np[i + 1 :] &= ~sup_mat[i, i + 1 :]
    return torch.from_numpy(np.flatnonzero(keep_np)).to(device=boxes.device, dtype=torch.long)


def get_image_slices(img_w: int, img_h: int, tile_w: int = 640, tile_h: int = 512, overlap: int = 100) -> list[tuple[int, int, int, int]]:
    """Generate (x1, y1, x2, y2) bounding slices covering the entire image with overlap."""
    step_x = tile_w - overlap
    step_y = tile_h - overlap

    x_starts = list(range(0, max(1, img_w - tile_w + 1), step_x))
    if not x_starts or x_starts[-1] + tile_w < img_w:
        x_starts.append(max(0, img_w - tile_w))

    y_starts = list(range(0, max(1, img_h - tile_h + 1), step_y))
    if not y_starts or y_starts[-1] + tile_h < img_h:
        y_starts.append(max(0, img_h - tile_h))

    slices = []
    for ys in y_starts:
        for xs in x_starts:
            slices.append((xs, ys, min(img_w, xs + tile_w), min(img_h, ys + tile_h)))
    return slices


def predict_tiled_image(
    model: YOLO,
    img: Image.Image,
    tile_w: int = 640,
    tile_h: int = 512,
    overlap: int = 100,
    conf: float = 0.001,
    merge_iou: float = 0.50,
    merge_ios: float | None = MERGE_IOS,
    nms_pre: int = 2000,
    max_per_img: int = 1000,
    device: str = "cuda:0",
) -> tuple[list[list[float]], list[float], list[int]]:
    """Run tiled inference on a single PIL image and merge detections via class-aware NMS + IOS."""
    img_w, img_h = img.size
    slices = get_image_slices(img_w, img_h, tile_w, tile_h, overlap)

    all_boxes = []
    all_scores = []
    all_classes = []

    rgb = img.convert("RGB") if img.mode != "RGB" else img
    img_bgr = np.asarray(rgb)[..., ::-1]
    crops = []
    offsets = []
    for x1, y1, x2, y2 in slices:
        crops.append(np.ascontiguousarray(img_bgr[y1:y2, x1:x2]))
        offsets.append((x1, y1))

    use_half = torch.cuda.is_available() and not str(device).startswith("cpu")
    results: Any = model.predict(
        crops, imgsz=(tile_h, tile_w), conf=conf, device=device, half=use_half, verbose=False
    )

    for (ox, oy), res in zip(offsets, results):
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        xyxy[:, 0] += ox
        xyxy[:, 2] += ox
        xyxy[:, 1] += oy
        xyxy[:, 3] += oy

        all_boxes.append(xyxy)
        all_scores.append(scores)
        all_classes.append(clses)

    if not all_boxes:
        return [], [], []

    cat_boxes = np.concatenate(all_boxes, axis=0)
    cat_scores = np.concatenate(all_scores, axis=0)
    cat_classes = np.concatenate(all_classes, axis=0)

    order = np.argsort(-cat_scores)[:nms_pre]
    cat_boxes = cat_boxes[order]
    cat_scores = cat_scores[order]
    cat_classes = cat_classes[order]

    boxes_t = torch.from_numpy(cat_boxes).float()
    scores_t = torch.from_numpy(cat_scores).float()
    classes_t = torch.from_numpy(cat_classes).long()
    keep_t = tv_batched_nms(boxes_t, scores_t, classes_t, merge_iou)
    if keep_t.numel() > 1:
        keep_t = keep_t[torch.argsort(scores_t[keep_t], descending=True, stable=True)]

    if merge_ios is not None and merge_ios > 0.0 and keep_t.numel() > 1:
        sub_keep = suppress_border_truncated(
            boxes_t[keep_t], scores_t[keep_t], classes_t[keep_t], ios_thr=merge_ios
        )
        keep_t = keep_t[sub_keep]

    keep_indices = keep_t[:max_per_img].numpy()

    final_boxes = cat_boxes[keep_indices].tolist()
    final_scores = cat_scores[keep_indices].tolist()
    final_classes = cat_classes[keep_indices].tolist()

    return final_boxes, final_scores, final_classes
