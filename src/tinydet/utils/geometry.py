"""Box geometry shared by the evaluator, error analysis and report scripts.

Delegates standard IoU and box-format conversion to ``torchvision.ops`` while
preserving float64 arithmetic for TinyPerson protocol reproducibility.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torchvision.ops import box_iou

Box = Sequence[float]


def as_boxes(boxes: Iterable[Box] | np.ndarray) -> np.ndarray:
    """Return an ``(n, 4)`` float64 array, preserving float64 arithmetic."""
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        return array.reshape(0, 4)
    return array.reshape(-1, 4)


def xywh_to_xyxy(box: Sequence[float], *, validate: bool = True) -> tuple[float, float, float, float]:
    """Convert a single ``[x, y, w, h]`` COCO box to ``(x1, y1, x2, y2)`` in IEEE-754 float64."""
    if len(box) != 4:
        raise ValueError(f"bbox must have four values: {box}")
    x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    if validate:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and math.isfinite(h)):
            raise ValueError(f"bbox values must be finite: {box}")
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"non-positive COCO bbox: {box}")
    return (x, y, x + w, y + h)


def iou_matrix(boxes: Iterable[Box] | np.ndarray, gt: Iterable[Box] | np.ndarray) -> np.ndarray:
    """Full ``(n, m)`` IoU matrix between two box sets via ``torchvision.ops.box_iou``."""
    first = as_boxes(boxes)
    second = as_boxes(gt)
    if first.shape[0] == 0 or second.shape[0] == 0:
        return np.zeros((first.shape[0], second.shape[0]), dtype=np.float64)
    overlaps = box_iou(
        torch.from_numpy(np.ascontiguousarray(first)),
        torch.from_numpy(np.ascontiguousarray(second)),
    )
    return torch.nan_to_num(overlaps, nan=0.0).numpy()


def iou(box: Box, boxes: Iterable[Box] | np.ndarray) -> np.ndarray:
    """Intersection-over-union of one box against every box in ``boxes``."""
    return iou_matrix([box], boxes)[0]


def iod_matrix(
    detections: Iterable[Box] | np.ndarray, ignored: Iterable[Box] | np.ndarray
) -> np.ndarray:
    """Full ``(n, m)`` intersection-over-detection matrix (TinyPerson ignore matching)."""
    first = as_boxes(detections)
    second = as_boxes(ignored)
    if first.shape[0] == 0 or second.shape[0] == 0:
        return np.zeros((first.shape[0], second.shape[0]), dtype=np.float64)
    left = np.maximum(first[:, None, 0], second[None, :, 0])
    top = np.maximum(first[:, None, 1], second[None, :, 1])
    right = np.minimum(first[:, None, 2], second[None, :, 2])
    bottom = np.minimum(first[:, None, 3], second[None, :, 3])
    intersection = np.maximum(right - left, 0.0) * np.maximum(bottom - top, 0.0)
    area = (
        np.maximum(first[:, 2] - first[:, 0], 0.0)
        * np.maximum(first[:, 3] - first[:, 1], 0.0)
    )[:, None]
    overlaps = np.zeros_like(intersection)
    positive = area[:, 0] > 0
    overlaps[positive] = intersection[positive] / area[positive]
    return overlaps


def iod(detection: Box, ignored: Iterable[Box] | np.ndarray) -> np.ndarray:
    """Intersection-over-detection-area of one box against every ignored box."""
    return iod_matrix([detection], ignored)[0]


def best_first_max(overlaps: np.ndarray, candidates: np.ndarray) -> int:
    """Index of the highest overlap, first index on ties."""
    return int(candidates[int(np.argmax(overlaps[candidates]))])


def best_last_max(overlaps: np.ndarray, candidates: np.ndarray) -> int:
    """Index of the highest overlap, last index on ties."""
    peak = overlaps[candidates].max()
    return int(candidates[overlaps[candidates] == peak].max())


def greedy_assign_first_max(overlap_mat: np.ndarray, order: np.ndarray, threshold: float) -> np.ndarray:
    """Vectorized C-numpy greedy 1-to-1 GT assignment at ``IoU >= threshold`` (first-max tie breaking)."""
    n_det, n_gt = overlap_mat.shape
    assigned = np.full(n_det, -1, dtype=np.int64)
    if n_det == 0 or n_gt == 0:
        return assigned
    masked = overlap_mat.copy()
    masked[masked < threshold] = -1.0
    for det_idx in order.tolist():
        row = masked[det_idx]
        best_gt = int(np.argmax(row))
        if row[best_gt] >= threshold:
            assigned[det_idx] = best_gt
            masked[:, best_gt] = -1.0
    return assigned
