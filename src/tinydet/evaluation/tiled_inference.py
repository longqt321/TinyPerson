"""Sliding window tiled inference and coordinate merge matching TOV_mmdetection protocol.

Tile parameters:
    Tile size: 640x512 (width=640, height=512)
    Overlap: 100x100
    NMS merge IoU threshold: 0.50
    Pre-NMS max candidates: 2000
    Post-NMS max detections: 1000
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLO


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
    nms_pre: int = 2000,
    max_per_img: int = 1000,
    device: str = "cuda:0",
) -> tuple[list[list[float]], list[float], list[int]]:
    """Run tiled inference on a single PIL image and merge detections via NMS."""
    img_w, img_h = img.size
    slices = get_image_slices(img_w, img_h, tile_w, tile_h, overlap)

    all_boxes = []
    all_scores = []
    all_classes = []

    # Batch process crops
    crops = []
    offsets = []
    for x1, y1, x2, y2 in slices:
        crop = img.crop((x1, y1, x2, y2))
        crops.append(crop)
        offsets.append((x1, y1))

    # Predict in chunks
    results = model.predict(crops, imgsz=(tile_h, tile_w), conf=conf, device=device, verbose=False)

    for (ox, oy), res in zip(offsets, results):
        boxes = res.boxes
        if len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        # Shift to full-frame coordinates
        xyxy[:, [0, 2]] += ox
        xyxy[:, [1, 3]] += oy

        all_boxes.append(xyxy)
        all_scores.append(scores)
        all_classes.append(clses)

    if not all_boxes:
        return [], [], []

    cat_boxes = np.concatenate(all_boxes, axis=0)
    cat_scores = np.concatenate(all_scores, axis=0)
    cat_classes = np.concatenate(all_classes, axis=0)

    # Sort and take top nms_pre
    order = np.argsort(-cat_scores)[:nms_pre]
    cat_boxes = cat_boxes[order]
    cat_scores = cat_scores[order]
    cat_classes = cat_classes[order]

    # Global NMS via Torchvision
    boxes_t = torch.from_numpy(cat_boxes).float()
    scores_t = torch.from_numpy(cat_scores).float()
    keep_indices = nms(boxes_t, scores_t, merge_iou).numpy()[:max_per_img]

    final_boxes = cat_boxes[keep_indices].tolist()
    final_scores = cat_scores[keep_indices].tolist()
    final_classes = cat_classes[keep_indices].tolist()

    return final_boxes, final_scores, final_classes
