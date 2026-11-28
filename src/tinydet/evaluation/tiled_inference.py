"""Sliding window tiled inference and coordinate merge matching TOV_mmdetection protocol.

Tile parameters:
    Tile size: 640x512 (width=640, height=512)
    Overlap: 100x100
    NMS merge IoU threshold: 0.50
    Pre-NMS max candidates: 2000
    Post-NMS max detections: 1000
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms as tv_batched_nms
from torchvision.ops import box_area, box_iou, nms
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


def _greedy_suppress_mask(sup_mat: np.ndarray) -> np.ndarray:
    """Vectorized row-wise greedy suppression over an ``(N, N)`` boolean conflict matrix."""
    n = sup_mat.shape[0]
    keep = np.ones(n, dtype=np.bool_)
    for i in range(n - 1):
        if keep[i]:
            keep[i + 1 :] &= ~sup_mat[i, i + 1 :]
    return keep


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
    ious = box_iou(boxes, boxes)
    union = areas[:, None] + areas[None, :]
    # Exact intersection recovered from C++/CUDA torchvision.ops.box_iou: inter = iou * (a1 + a2) / (1 + iou)
    inter = (ious * union) / (1.0 + ious)
    min_area = torch.minimum(areas[:, None], areas[None, :])
    upper_sup = ((inter >= ios_thr * min_area) & (classes[:, None] == classes[None, :])).triu(diagonal=1)
    has_sup = upper_sup.any(dim=0)
    if not bool(has_sup.any()):
        return torch.arange(n, device=boxes.device, dtype=torch.long)
    # Fast vectorized path when no suppressed box itself suppresses a lower-scoring box (99%+ of images)
    if not bool((upper_sup & has_sup[:, None]).any()):
        return (~has_sup).nonzero(as_tuple=False).squeeze(-1)
    sup_mat = np.ascontiguousarray(upper_sup.cpu().numpy())
    keep_np = _greedy_suppress_mask(sup_mat)
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
    img: Image.Image | np.ndarray | Path | str,
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
    """Run batched tiled inference on an image (path, BGR ndarray, or PIL) with single GPU->CPU sync."""
    if isinstance(img, (str, Path)):
        decoded = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if decoded is None:
            with Image.open(img) as pil_im:
                rgb = pil_im.convert("RGB") if pil_im.mode != "RGB" else pil_im
                img_bgr = np.ascontiguousarray(np.asarray(rgb)[..., ::-1])
        else:
            img_bgr = decoded
    elif isinstance(img, np.ndarray):
        img_bgr = img
    else:
        rgb = img.convert("RGB") if img.mode != "RGB" else img
        img_bgr = np.asarray(rgb)[..., ::-1]

    img_h, img_w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    slices = get_image_slices(img_w, img_h, tile_w, tile_h, overlap)

    crops = []
    offsets = []
    for x1, y1, x2, y2 in slices:
        crops.append(np.ascontiguousarray(img_bgr[y1:y2, x1:x2]))
        offsets.append((x1, y1))

    use_half = torch.cuda.is_available() and not str(device).startswith("cpu")
    results: Any = model.predict(
        crops,
        imgsz=(tile_h, tile_w),
        batch=max(1, min(len(crops), 32)),
        conf=conf,
        device=device,
        half=use_half,
        verbose=False,
    )

    gpu_boxes: list[torch.Tensor] = []
    gpu_scores: list[torch.Tensor] = []
    gpu_classes: list[torch.Tensor] = []
    for (ox, oy), res in zip(offsets, results):
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy_t = boxes.xyxy.float()
        if ox != 0 or oy != 0:
            shift = xyxy_t.new_tensor((float(ox), float(oy), float(ox), float(oy)))
            xyxy_t = xyxy_t + shift
        gpu_boxes.append(xyxy_t)
        gpu_scores.append(boxes.conf.float())
        gpu_classes.append(boxes.cls.long())

    if not gpu_boxes:
        return [], [], []

    boxes_t = torch.cat(gpu_boxes, dim=0)
    scores_t = torch.cat(gpu_scores, dim=0)
    classes_t = torch.cat(gpu_classes, dim=0)

    if scores_t.numel() > nms_pre:
        order = torch.argsort(scores_t, descending=True, stable=True)[:nms_pre]
    else:
        order = torch.argsort(scores_t, descending=True, stable=True)
    boxes_t = boxes_t[order]
    scores_t = scores_t[order]
    classes_t = classes_t[order]

    keep_t = tv_batched_nms(boxes_t, scores_t, classes_t, merge_iou)
    if keep_t.numel() > 1:
        keep_t = keep_t[torch.argsort(scores_t[keep_t], descending=True, stable=True)]

    if merge_ios is not None and merge_ios > 0.0 and keep_t.numel() > 1:
        sub_keep = suppress_border_truncated(
            boxes_t[keep_t], scores_t[keep_t], classes_t[keep_t], ios_thr=merge_ios
        )
        keep_t = keep_t[sub_keep]

    keep_t = keep_t[:max_per_img]
    return (
        boxes_t[keep_t].cpu().tolist(),
        scores_t[keep_t].cpu().tolist(),
        classes_t[keep_t].cpu().tolist(),
    )
