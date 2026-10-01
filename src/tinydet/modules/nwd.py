"""Normalized Wasserstein distance, Wang et al., arXiv:2110.13389, Eq. 7–9."""

import math

import torch
from torch import nn


def nwd_similarity(pred: torch.Tensor, target: torch.Tensor, constant: float = 12.8) -> torch.Tensor:
    """Aligned/broadcast xyxy boxes in pixels; output lies in [0, 1]."""
    if not math.isfinite(constant) or constant <= 0:
        raise ValueError("NWD constant must be finite and positive (pixels)")
    pred, target = pred.float(), target.float()
    center_delta = (pred[..., :2] + pred[..., 2:] - target[..., :2] - target[..., 2:]) / 2
    size_delta = ((pred[..., 2:] - pred[..., :2]) - (target[..., 2:] - target[..., :2])) / 2
    # vector_norm has a finite zero subgradient for identical boxes.
    distance = torch.linalg.vector_norm(torch.cat((center_delta, size_delta), dim=-1), dim=-1)
    return torch.exp(-distance / constant)


class NWDBboxLoss(nn.Module):
    """Replace/blend the native IoU term; preserve native DFL or signed L1."""

    def __init__(self, native: nn.Module, constant: float = 12.8, weight: float = 1.0):
        super().__init__()
        if not math.isfinite(constant) or constant <= 0 or not 0 <= weight <= 1:
            raise ValueError("NWD requires constant > 0 and weight in [0, 1]")
        self.native = native
        self.constant = constant
        self.weight = weight

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        # Native losses receive feature-grid units. Restore pixels before NWD.
        similarity = nwd_similarity((pred_bboxes * stride)[fg_mask],
                                    (target_bboxes * stride)[fg_mask], self.constant)
        weights = target_scores[fg_mask].sum(-1)
        nwd = ((1 - similarity) * weights).sum() / target_scores_sum
        iou, auxiliary = self.native(pred_dist, pred_bboxes, anchor_points, target_bboxes,
                                     target_scores, target_scores_sum, fg_mask, imgsz, stride)
        return (1 - self.weight) * iou + self.weight * nwd, auxiliary
