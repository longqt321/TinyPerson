"""AMP-Safe Unified & Multi-Hypothesis Composite Detection Heads for YOLO26n.

Fixes all NaN/Inf issues under PyTorch Mixed Precision (FP16) and unlocks gradient flow:
  - Forced FP32 execution inside `compute_aux_loss` via `autocast(enabled=False)`.
  - Eliminated singular derivatives in `.sqrt()`, `.norm()`, `.log()`, and area ratios.
  - Replaced dead zero-weight cascades with small normal initialization (std=0.01) + LayerScale.
  - Detached unstable raw box coordinates during grid sampling to prevent gradient explosion
    against stock Ultralytics TAL/CIoU/DFL losses.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import dist2bbox, make_anchors

COMPOSITE_MODES = ("cglr", "cass", "sure", "prism")


# ============================================================================
# 1. AMP-Safe Geometry & Spatial Grid Sampling Utilities
# ============================================================================


def _sample_roi_grid(feature: Tensor, boxes: Tensor, stride: float, points: int = 3) -> Tensor:
    """Sample a ``points x points`` grid inside each pixel-space xyxy box.

    Safe against out-of-bound coordinates and FP16 precision issues.
    Returns tensor of shape [B, K, C, points * points].
    """
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B, K, 4]")
    b = boxes.shape[0]
    if feature.shape[0] != b:
        raise ValueError("feature and boxes must have the same batch size")
    height, width = feature.shape[-2:]
    image_w = max(float(width * stride), 1.0)
    image_h = max(float(height * stride), 1.0)

    # Detach sampling coordinates from raw box regression to prevent grid_sample gradient spikes
    safe_boxes = boxes.detach().float()
    x1 = safe_boxes[..., 0].clamp(0.0, image_w)
    y1 = safe_boxes[..., 1].clamp(0.0, image_h)
    x2 = safe_boxes[..., 2].clamp(0.0, image_w)
    y2 = safe_boxes[..., 3].clamp(0.0, image_h)

    axis = torch.linspace(-1.0, 1.0, points, device=feature.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    local_x = xx.reshape(-1)[None, None, :]
    local_y = yy.reshape(-1)[None, None, :]

    pixel_x = x1[..., None] + (local_x + 1.0) * 0.5 * (x2 - x1).clamp_min(1e-3)[..., None]
    pixel_y = y1[..., None] + (local_y + 1.0) * 0.5 * (y2 - y1).clamp_min(1e-3)[..., None]
    grid = (
        torch.stack((pixel_x / image_w * 2.0 - 1.0, pixel_y / image_h * 2.0 - 1.0), dim=-1)
        .clamp(-1.0, 1.0)
        .to(dtype=feature.dtype)
    )

    values = F.grid_sample(
        feature, grid, mode="bilinear", padding_mode="border", align_corners=False
    )  # [B, C, K, P*P]
    return values.permute(0, 2, 1, 3).contiguous()  # [B, K, C, P*P]


def _sample_roi(feature: Tensor, boxes: Tensor, stride: float, points: int = 3) -> Tensor:
    """Mean-pooled ROI feature of shape [B, K, C]."""
    return _sample_roi_grid(feature, boxes, stride, points=points).mean(dim=-1)


def _cxcywh_norm(boxes: Tensor, width: float, height: float) -> Tensor:
    w_img = max(float(width), 1.0)
    h_img = max(float(height), 1.0)
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = ((x1 + x2) * 0.5 / w_img).clamp(0.0, 1.0)
    cy = ((y1 + y2) * 0.5 / h_img).clamp(0.0, 1.0)
    bw = ((x2 - x1).clamp_min(1e-2) / w_img).clamp(1e-4, 1.0)
    bh = ((y2 - y1).clamp_min(1e-2) / h_img).clamp(1e-4, 1.0)
    return torch.stack((cx, cy, bw, bh), dim=-1)


def _apply_delta(boxes: Tensor, delta: Tensor, width: float, height: float) -> Tensor:
    values = _cxcywh_norm(boxes, width, height)
    d = delta.clamp(-0.25, 0.25)
    cx = (values[..., 0] + d[..., 0] * values[..., 2]).clamp(0.0, 1.0)
    cy = (values[..., 1] + d[..., 1] * values[..., 3]).clamp(0.0, 1.0)
    w = (values[..., 2] * torch.exp(d[..., 2])).clamp(1e-4, 1.0)
    h = (values[..., 3] * torch.exp(d[..., 3])).clamp(1e-4, 1.0)
    return torch.stack(
        (
            (cx - w * 0.5) * width,
            (cy - h * 0.5) * height,
            (cx + w * 0.5) * width,
            (cy + h * 0.5) * height,
        ),
        dim=-1,
    )


def _pair_iou(a: Tensor, b: Tensor) -> Tensor:
    a_f = a.float()
    b_f = b.float()
    left = torch.maximum(a_f[:, None, :2], b_f[None, :, :2])
    right = torch.minimum(a_f[:, None, 2:], b_f[None, :, 2:])
    intersection = (right - left).clamp_min(0.0).prod(-1)
    area_a = (a_f[:, 2] - a_f[:, 0]).clamp_min(1e-3) * (a_f[:, 3] - a_f[:, 1]).clamp_min(1e-3)
    area_b = (b_f[:, 2] - b_f[:, 0]).clamp_min(1e-3) * (b_f[:, 3] - b_f[:, 1]).clamp_min(1e-3)
    union = (area_a[:, None] + area_b[None, :] - intersection).clamp_min(1e-4)
    return (intersection / union).clamp(0.0, 1.0)


def _safe_dist(a: Tensor, b: Tensor) -> Tensor:
    """Smooth L2 distance without NaN gradient at a == b."""
    return ((a.float() - b.float()).square().sum(dim=-1) + 1e-6).sqrt()


def _padded_targets(
    batch: dict[str, Tensor], batch_size: int, width: float, height: float, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    batch_idx = batch.get("batch_idx")
    if batch_idx is None or batch_idx.numel() == 0:
        empty = torch.zeros(batch_size, 0, 4, device=device, dtype=torch.float32)
        return (
            empty,
            torch.zeros(batch_size, 0, device=device, dtype=torch.float32),
            torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
        )
    rows = torch.cat(
        (batch_idx.view(-1, 1).float(), batch["cls"].view(-1, 1).float(), batch["bboxes"].float()),
        dim=1,
    ).to(device)
    counts = torch.bincount(rows[:, 0].long().clamp_min(0), minlength=batch_size)
    max_gt = int(counts.max().item())
    boxes = torch.zeros(batch_size, max_gt, 4, device=device, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_gt, device=device, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_gt, device=device, dtype=torch.bool)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)))
    scale = torch.tensor((width, height, width, height), device=device, dtype=torch.float32)
    for image in range(batch_size):
        start, end = int(offsets[image]), int(offsets[image + 1])
        if start == end:
            continue
        xywh = rows[start:end, 2:6]
        xyxy = torch.cat((xywh[:, :2] - xywh[:, 2:] * 0.5, xywh[:, :2] + xywh[:, 2:] * 0.5), 1)
        boxes[image, : end - start] = xyxy * scale
        labels[image, : end - start] = rows[start:end, 1]
        mask[image, : end - start] = True
    return boxes, labels, mask


# ============================================================================
# 2. Shared Building Blocks (Safe Group Conv, ResBottleneck, LayerScaled SwiGLU)
# ============================================================================


class ConvNormAct(nn.Sequential):
    """Conv2d + BatchNorm2d + SiLU with automatic safe group divisibility."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        safe_groups = 1
        if groups > 1:
            if int(in_channels) % int(groups) == 0 and int(out_channels) % int(groups) == 0:
                safe_groups = int(groups)
            else:
                safe_groups = math.gcd(int(in_channels), int(out_channels))
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=safe_groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResBottleneck2d(nn.Module):
    """2D Residual Bottleneck with Depthwise Separable Conv, SE Attention & LayerScale."""

    def __init__(self, channels: int, expansion: float = 0.5, dilation: int = 1) -> None:
        super().__init__()
        mid = max(16, int(channels * expansion))
        se_mid = max(8, channels // 4)
        self.block = nn.Sequential(
            ConvNormAct(channels, mid, 1, act=True),
            ConvNormAct(mid, mid, 3, padding=dilation, dilation=dilation, groups=mid, act=True),
            ConvNormAct(mid, channels, 1, act=False),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, se_mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_mid, channels, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        out = out * self.se(out)
        return self.act(x + self.gamma * out)


class GatedResMLPBlock(nn.Module):
    """Pre-Norm SwiGLU Residual Block with non-zero initialization so gradients always flow."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or dim
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc_value = nn.Linear(dim, hidden)
        self.fc_gate = nn.Linear(dim, hidden)
        self.fc_out = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 0.1))
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm(x)
        gated = F.silu(self.fc_gate(normed)) * self.fc_value(normed)
        return x + self.gamma * self.fc_out(gated)


class FiveRegionROIExtractor(nn.Module):
    """Decompose a 3x3 ROI grid into 5 explicit geometric sub-regions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        basis = torch.zeros(5, 9)
        basis[0, [0, 3, 6]] = 1.0 / 3.0  # Left
        basis[1, [2, 5, 8]] = 1.0 / 3.0  # Right
        basis[2, [0, 1, 2]] = 1.0 / 3.0  # Top
        basis[3, [6, 7, 8]] = 1.0 / 3.0  # Bottom
        basis[4, 4] = 1.0
        basis[4, [0, 1, 2, 3, 5, 6, 7, 8]] = -1.0 / 8.0  # Center - Surround
        self.register_buffer("basis", basis, persistent=False)
        self.delta_basis = nn.Parameter(torch.zeros(5, 9))

        self.region_fuse = nn.Sequential(
            nn.Linear(channels * 5, channels),
            nn.LayerNorm(channels, eps=1e-5),
            nn.SiLU(inplace=True),
            GatedResMLPBlock(channels, channels),
        )

    def forward(
        self, roi_grid: Tensor, return_center: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        weights = self.basis.to(
            dtype=roi_grid.dtype, device=roi_grid.device
        ) + 0.1 * self.delta_basis.to(roi_grid.dtype)
        regions = torch.matmul(roi_grid, weights.T)  # [B, K, C, 5]
        center_contrast = regions[..., 4]  # [B, K, C]
        mean_pooled = roi_grid.mean(dim=-1)
        fused = mean_pooled + 0.25 * self.region_fuse(regions.flatten(-2))
        if return_center:
            return fused, center_contrast
        return fused


SpatialGridProjector = FiveRegionROIExtractor


# ============================================================================
# 3. Feature Stages for CGLR, CASS, SURE, and PRISM
# ============================================================================


class ContextEvidenceGate(nn.Module):
    """CGLR Module 1: Cascaded Local-Contrast & Cross-Scale Bottleneck Evidence Gate."""

    def __init__(
        self, c2: int = 32, c3: int = 64, c4: int = 128, hidden: int = 32, alpha: float = 0.05
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.project3 = nn.Sequential(
            ConvNormAct(c3, hidden, 1, act=True),
            ResBottleneck2d(hidden, expansion=0.5),
        )
        self.project4 = nn.Sequential(
            ConvNormAct(c4, hidden, 1, act=True),
            ResBottleneck2d(hidden, expansion=0.5),
        )
        self.local_core = ConvNormAct(c2, c2, 3, padding=1, groups=c2, act=True)
        self.local_surround = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.evidence = nn.Sequential(
            ConvNormAct(c2 * 2, c2, 1, act=True),
            ResBottleneck2d(c2, expansion=0.5, dilation=1),
            ResBottleneck2d(c2, expansion=0.5, dilation=2),
        )
        in_gate = c2 + 2 * hidden + c2
        self.gate = nn.Sequential(
            ConvNormAct(in_gate, hidden, 1, act=True),
            ResBottleneck2d(hidden, expansion=0.5, dilation=1),
            ResBottleneck2d(hidden, expansion=0.5, dilation=2),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, p2: Tensor, p3: Tensor, p4: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        size = p2.shape[-2:]
        context3 = F.interpolate(self.project3(p3), size=size, mode="bilinear", align_corners=False)
        context4 = F.interpolate(self.project4(p4), size=size, mode="bilinear", align_corners=False)
        core = self.local_core(p2)
        contrast = core - self.local_surround(p2)
        evidence = self.evidence(torch.cat((core, contrast), dim=1))
        logits = self.gate(torch.cat((p2, context3, context4, evidence), dim=1)).clamp(-10.0, 10.0)
        gate = logits.sigmoid()
        return p2 + self.alpha * gate * evidence, gate, evidence, logits


class DensityRouter(nn.Module):
    """CASS Module 1: ASPP-Lite Multi-Dilation Crowd Density Router with FiLM Modulation."""

    def __init__(
        self, c2: int = 32, c3: int = 64, c4: int = 128, hidden: int = 32, beta: float = 0.1
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.project3 = nn.Sequential(
            ConvNormAct(c3, c2, 1, act=True),
            ResBottleneck2d(c2, expansion=0.5),
        )
        self.project4 = nn.Sequential(
            ConvNormAct(c4, hidden, 1, act=True),
            ResBottleneck2d(hidden, expansion=0.5),
        )
        in_ch = c2 + hidden
        branch_ch = max(8, hidden // 2)
        self.stem = ConvNormAct(in_ch, hidden, 1, act=True)
        self.branch_d1 = ConvNormAct(hidden, branch_ch, 3, padding=1, dilation=1, groups=branch_ch)
        self.branch_d2 = ConvNormAct(hidden, branch_ch, 3, padding=2, dilation=2, groups=branch_ch)
        self.branch_d3 = ConvNormAct(hidden, branch_ch, 3, padding=3, dilation=3, groups=branch_ch)
        self.density_neck = nn.Sequential(
            ConvNormAct(branch_ch * 3, hidden, 1, act=True),
            ResBottleneck2d(hidden, expansion=0.5),
        )
        self.density_head = nn.Conv2d(hidden, 1, 1)
        nn.init.normal_(self.density_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.density_head.bias, -3.0)

        self.crowd_refine = nn.Sequential(
            ResBottleneck2d(c2, expansion=0.5, dilation=1),
            ConvNormAct(c2, c2 * 2, 1, act=False),
        )
        nn.init.normal_(self.crowd_refine[-1][0].weight, mean=0.0, std=0.01)

    def forward(self, p2: Tensor, p3: Tensor, p4: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        size = p2.shape[-2:]
        context3 = F.interpolate(self.project3(p3), size=size, mode="bilinear", align_corners=False)
        context4 = F.interpolate(self.project4(p4), size=size, mode="bilinear", align_corners=False)
        feat = self.stem(torch.cat((context3, context4), dim=1))
        ms_feat = torch.cat(
            (self.branch_d1(feat), self.branch_d2(feat), self.branch_d3(feat)), dim=1
        )
        neck = self.density_neck(ms_feat)
        logits = self.density_head(neck).clamp(-10.0, 10.0)
        density = logits.sigmoid()

        gamma_beta = self.crowd_refine(context3 - p2)
        gamma, shift = gamma_beta.chunk(2, dim=1)
        modulated = p2 * (1.0 + self.beta * density * gamma.tanh()) + self.beta * density * (
            (context3 - p2) + 0.1 * shift.tanh()
        )
        return modulated, density, context3, logits


class ScaleAligner(nn.Module):
    """SURE Module 1: Scale-Conditioned Deformable Residual Aligner."""

    def __init__(self, channels: int = 32, points: int = 9, max_offset_cells: float = 0.75) -> None:
        super().__init__()
        if points != 9:
            raise ValueError("ScaleAligner currently supports the fixed 3x3 grid")
        self.points = points
        self.max_offset_cells = float(max_offset_cells)
        self.scale_net = nn.Sequential(
            ResBottleneck2d(channels, expansion=0.5, dilation=1),
            ResBottleneck2d(channels, expansion=0.5, dilation=2),
        )
        self.scale_head = ConvNormAct(channels, 1, 1, act=False)
        self.offset_mask = nn.Sequential(
            ConvNormAct(channels + 1, channels, 1, act=True),
            ConvNormAct(channels, channels, 3, padding=1, groups=channels, act=True),
            nn.Conv2d(channels, 3 * points, 1),
        )
        nn.init.normal_(self.offset_mask[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.offset_mask[-1].bias)
        self.post_align = ResBottleneck2d(channels, expansion=0.5)
        self.blend = nn.Parameter(torch.tensor(0.05))

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        b, c, h, w = feature.shape
        scale_feat = self.scale_net(feature)
        scale_map = 0.5 + 2.0 * self.scale_head(scale_feat).clamp(-10.0, 10.0).sigmoid()

        raw_om = self.offset_mask(torch.cat((scale_feat, scale_map), dim=1))
        pred_offsets = raw_om[:, : 2 * self.points].view(b, self.points, 2, h, w)
        pred_masks = raw_om[:, 2 * self.points :].view(b, self.points, h, w).clamp(-10.0, 10.0)

        axis = torch.linspace(-1.0, 1.0, 3, device=feature.device, dtype=feature.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        base = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

        predicted = pred_offsets.permute(0, 3, 4, 1, 2).tanh() * self.max_offset_cells
        scale_factor = scale_map.permute(0, 2, 3, 1)[..., None] * 0.5
        offsets = (base[None, None, None] * scale_factor + predicted) * torch.tensor(
            (2.0 / max(w, 1), 2.0 / max(h, 1)), device=feature.device, dtype=feature.dtype
        )
        grid_x = (torch.arange(w, device=feature.device, dtype=feature.dtype) + 0.5) * 2.0 / max(
            w, 1
        ) - 1.0
        grid_y = (torch.arange(h, device=feature.device, dtype=feature.dtype) + 0.5) * 2.0 / max(
            h, 1
        ) - 1.0
        center = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="xy"), dim=-1)
        grid = (center[None, :, :, None, :] + offsets).clamp(-1.0, 1.0)

        values = F.grid_sample(
            feature,
            grid.reshape(b, h, w * self.points, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).view(b, c, h, w, self.points)

        prior_weights = torch.exp(-0.5 * base.square().sum(-1)).view(1, 1, 1, self.points)
        dyn_weights = pred_masks.permute(0, 2, 3, 1).sigmoid() * prior_weights
        dyn_weights = (
            dyn_weights / dyn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        ).unsqueeze(1)
        aligned = self.post_align((values * dyn_weights).sum(-1))
        return feature + self.blend.tanh() * (aligned - feature), scale_map


class PRISMFeatureStage(nn.Module):
    """PRISM Unified Stage 1: High-Frequency Contrast + Crowd FiLM + Sub-Pixel Aligner."""

    def __init__(
        self, c2: int = 32, c3: int = 64, c4: int = 128, hidden: int = 32, alpha: float = 0.1
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.proj3 = nn.Sequential(
            ConvNormAct(c3, hidden, 1, act=True),
            ResBottleneck2d(hidden, expansion=0.5, dilation=1),
        )
        self.proj4 = nn.Sequential(
            ConvNormAct(c4, hidden, 1, act=True),
            ResBottleneck2d(hidden, expansion=0.5, dilation=1),
        )
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.hf_encoder = nn.Sequential(
            ConvNormAct(c2 * 3, c2, 1, act=True),
            ResBottleneck2d(c2, expansion=0.5, dilation=1),
            ResBottleneck2d(c2, expansion=0.5, dilation=2),
        )
        in_fuse = c2 + c2 + 2 * hidden
        self.fuse_stem = ConvNormAct(in_fuse, hidden, 1, act=True)
        self.d1 = ConvNormAct(hidden, hidden, 3, padding=1, dilation=1, groups=hidden)
        self.d2 = ConvNormAct(hidden, hidden, 3, padding=2, dilation=2, groups=hidden)
        self.d3 = ConvNormAct(hidden, hidden, 3, padding=3, dilation=3, groups=hidden)
        self.gate_neck = nn.Sequential(
            ConvNormAct(hidden * 3, hidden, 1, act=True),
            ResBottleneck2d(hidden, expansion=0.5, dilation=1),
        )
        self.aux_head = nn.Conv2d(hidden, 2, 1)
        nn.init.normal_(self.aux_head.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.aux_head.bias.copy_(torch.tensor([-2.0, -3.0]))

        self.film_gen = nn.Sequential(ConvNormAct(hidden, c2 * 2, 1, act=False))
        nn.init.normal_(self.film_gen[0][0].weight, mean=0.0, std=0.01)

        self.micro_offset = nn.Sequential(
            ConvNormAct(c2 + hidden, hidden, 1, act=True),
            ConvNormAct(hidden, hidden, 3, padding=1, groups=hidden, act=True),
            nn.Conv2d(hidden, 2, 1),
        )
        nn.init.normal_(self.micro_offset[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.micro_offset[-1].bias)

    def _subpixel_align(self, feat: Tensor, offset_map: Tensor) -> Tensor:
        _, _, h, w = feat.shape
        dx = offset_map[:, 0].tanh() * (0.75 * 2.0 / max(w, 1))
        dy = offset_map[:, 1].tanh() * (0.75 * 2.0 / max(h, 1))
        grid_x = (torch.arange(w, device=feat.device, dtype=feat.dtype) + 0.5) * 2.0 / max(
            w, 1
        ) - 1.0
        grid_y = (torch.arange(h, device=feat.device, dtype=feat.dtype) + 0.5) * 2.0 / max(
            h, 1
        ) - 1.0
        base_grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="xy"), dim=-1)
        shifted_grid = (base_grid.unsqueeze(0) + torch.stack((dx, dy), dim=-1)).clamp(-1.0, 1.0)
        return F.grid_sample(
            feat, shifted_grid, mode="bilinear", padding_mode="border", align_corners=False
        )

    def forward(self, p2: Tensor, p3: Tensor, p4: Tensor) -> dict[str, Tensor]:
        size = p2.shape[-2:]
        c3 = F.interpolate(self.proj3(p3), size=size, mode="bilinear", align_corners=False)
        c4 = F.interpolate(self.proj4(p4), size=size, mode="bilinear", align_corners=False)

        hf3 = p2 - self.pool3(p2)
        hf5 = p2 - self.pool5(p2)
        evidence = self.hf_encoder(torch.cat((p2, hf3, hf5), dim=1))

        stem = self.fuse_stem(torch.cat((p2, evidence, c3, c4), dim=1))
        ms_context = self.gate_neck(torch.cat((self.d1(stem), self.d2(stem), self.d3(stem)), dim=1))

        offsets = self.micro_offset(torch.cat((evidence, ms_context), dim=1))
        aligned_evidence = self._subpixel_align(evidence, offsets)

        aux_logits = self.aux_head(ms_context).clamp(-10.0, 10.0)
        gate_logits = aux_logits[:, 0:1]
        density_logits = aux_logits[:, 1:2]
        gate = gate_logits.sigmoid()
        density = density_logits.sigmoid()

        gamma, beta = self.film_gen(ms_context).chunk(2, dim=1)
        active_weight = torch.maximum(gate, density)
        enhanced_p2 = p2 * (1.0 + self.alpha * active_weight * gamma.tanh()) + self.alpha * gate * (
            aligned_evidence + 0.1 * beta.tanh()
        )
        return {
            "p2": enhanced_p2,
            "gate": gate,
            "gate_logits": gate_logits,
            "density": density,
            "density_logits": density_logits,
            "scale_map": 0.5 + 2.0 * gate,
            "evidence": aligned_evidence,
        }


# ============================================================================
# 4. Downstream Heads (Non-Zero Initialized for Smooth Gradient Flow)
# ============================================================================


class CandidateRefiner(nn.Module):
    """Cascaded Residual Refiner with non-zero output initialization."""

    def __init__(
        self, in_dim: int, hidden: int = 128, sigma_dim: int = 1, num_bins: int = 7
    ) -> None:
        super().__init__()
        self.sigma_dim = int(sigma_dim)
        self.num_bins = int(num_bins)
        self.is_distributional = self.sigma_dim == 4
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.SiLU(inplace=True),
        )
        self.stage1 = GatedResMLPBlock(hidden, hidden)
        self.coarse_head = nn.Linear(hidden, 4)
        nn.init.normal_(self.coarse_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.coarse_head.bias)

        self.stage2_proj = nn.Linear(hidden + 4, hidden)
        self.stage2 = GatedResMLPBlock(hidden, hidden)
        self.stage3 = GatedResMLPBlock(hidden, hidden)

        if self.is_distributional:
            self.dist_head = nn.Linear(hidden, 4 * self.num_bins + 4)
            nn.init.normal_(self.dist_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.dist_head.bias)
            bins = torch.linspace(-0.20, 0.20, self.num_bins)
            self.register_buffer("bin_values", bins, persistent=False)
        else:
            self.fine_head = nn.Linear(hidden, 4 + self.sigma_dim)
            nn.init.normal_(self.fine_head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.fine_head.bias)

        self._last_dist_stats: Tensor | None = None

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        h1 = self.stage1(self.stem(features))
        coarse_delta = torch.tanh(self.coarse_head(h1)) * 0.12

        h2 = self.stage3(self.stage2(self.stage2_proj(torch.cat((h1, coarse_delta), dim=-1))))
        if not self.is_distributional:
            out = self.fine_head(h2)
            fine_delta = (coarse_delta + torch.tanh(out[..., :4]) * 0.15).clamp(-0.25, 0.25)
            log_sigma = out[..., 4:].clamp(-3.0, 2.0)
            self._last_dist_stats = None
            return fine_delta, log_sigma

        raw = self.dist_head(h2)
        dist_logits = (
            raw[..., : 4 * self.num_bins]
            .view(*features.shape[:-1], 4, self.num_bins)
            .clamp(-10.0, 10.0)
        )
        probs = F.softmax(dist_logits, dim=-1)
        bins = self.bin_values.to(dtype=probs.dtype, device=probs.device)
        expected_delta = (probs * bins).sum(dim=-1)
        delta = (coarse_delta + expected_delta).clamp(-0.25, 0.25)

        variance = (
            (probs * (bins - expected_delta.unsqueeze(-1)).square()).sum(dim=-1).clamp_min(1e-4)
        )
        entropy = -(probs * probs.clamp_min(1e-5).log()).sum(dim=-1)
        top2_mass = probs.topk(min(2, self.num_bins), dim=-1).values.sum(dim=-1)
        self._last_dist_stats = torch.cat((probs.flatten(-2), entropy, top2_mass), dim=-1)

        log_sigma = (raw[..., 4 * self.num_bins :] + 0.5 * variance.log()).clamp(-3.0, 2.0)
        return delta, log_sigma


class QualityHead(nn.Module):
    """Cross-Gated Residual Quality Calibrator."""

    def __init__(
        self, in_dim: int, hidden: int = 64, sigma_dim: int = 1, dist_stat_dim: int = 0
    ) -> None:
        super().__init__()
        self.sigma_dim = int(sigma_dim)
        self.dist_stat_dim = int(dist_stat_dim)
        roi_dim = in_dim - self.sigma_dim
        self.roi_stem = nn.Sequential(
            nn.Linear(roi_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.SiLU(inplace=True),
            GatedResMLPBlock(hidden, hidden),
        )
        unc_in = self.sigma_dim + 5 + self.dist_stat_dim
        self.unc_gate = nn.Sequential(
            nn.Linear(unc_in, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.SiLU(inplace=True),
            GatedResMLPBlock(hidden, hidden),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            GatedResMLPBlock(hidden, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features: Tensor, dist_stats: Tensor | None = None) -> Tensor:
        roi = features[..., : -self.sigma_dim]
        log_sigma = features[..., -self.sigma_dim :]
        geom = roi[..., -5:]
        unc_parts = [log_sigma, geom]
        if self.dist_stat_dim > 0:
            if dist_stats is None:
                dist_stats = features.new_zeros(*features.shape[:-1], self.dist_stat_dim)
            unc_parts.append(dist_stats)
        gate = self.unc_gate(torch.cat(unc_parts, dim=-1))
        fused = self.roi_stem(roi) * gate
        return self.head(fused).clamp(-10.0, 10.0)


class AssignerMLP(nn.Module):
    """Context-Normalized Visual-Geometric Residual Assigner."""

    def __init__(self, in_dim: int = 6, hidden: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.SiLU(inplace=True),
        )
        self.block1 = GatedResMLPBlock(hidden, hidden)
        self.cand_norm = nn.LayerNorm(hidden, eps=1e-5)
        self.block2 = GatedResMLPBlock(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: Tensor) -> Tensor:
        h = self.block1(self.stem(features))
        if h.ndim == 3 and h.shape[1] > 1:
            h = h + 0.25 * self.cand_norm(h - h.mean(dim=1, keepdim=True))
        return self.out(self.block2(h)).squeeze(-1).clamp(-10.0, 10.0)


class DuplicateScorer(nn.Module):
    """Cascaded Relational Residual Scorer for pair-wise duplicate suppression."""

    def __init__(self, in_dim: int = 5, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.SiLU(inplace=True),
            GatedResMLPBlock(hidden, hidden),
            GatedResMLPBlock(hidden, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.net[-1].bias, -3.0)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1).clamp(-10.0, 10.0)


# ============================================================================
# 5. Master CompositeDetect & CompositeCriterion (FP32 Protected)
# ============================================================================


class CompositeDetect(Detect):
    """Detect head supporting 'cglr', 'cass', 'sure', and 'prism' with full AMP safety."""

    def __init__(
        self,
        nc: int,
        reg_max: int,
        end2end: bool,
        ch: list[int],
        mode: str = "prism",
        topk: int = 300,
        hidden: int = 128,
        aux_weight: float = 0.5,
    ) -> None:
        if mode not in COMPOSITE_MODES:
            raise ValueError(f"mode must be one of {COMPOSITE_MODES}")
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.composite_mode = mode
        self.topk = int(topk)
        self.aux_weight = float(aux_weight)
        self.stage: nn.Module | None = None
        self.roi_proj_p2: FiveRegionROIExtractor | None = None
        self.roi_proj_p3: FiveRegionROIExtractor | None = None
        self.refiner: CandidateRefiner | None = None
        self.quality: QualityHead | None = None
        self.assign: AssignerMLP | None = None
        self.duplicate: DuplicateScorer | None = None
        self._last_aux: dict[str, Tensor] = {}

        c2, c3, c4 = ch[0], ch[1], ch[2]
        self.roi_proj_p2 = FiveRegionROIExtractor(c2)
        self.roi_proj_p3 = FiveRegionROIExtractor(c3)
        num_bins = 7

        if mode == "cglr":
            self.stage = ContextEvidenceGate(c2, c3, c4)
            in_dim = c2 + c3 + 1 + 5
            self.refiner = CandidateRefiner(in_dim, hidden, sigma_dim=1)
            self.quality = QualityHead(in_dim + 1, hidden=64, sigma_dim=1, dist_stat_dim=0)
        elif mode == "cass":
            self.stage = DensityRouter(c2, c3, c4)
            self.assign = AssignerMLP(6, hidden=48)
            self.duplicate = DuplicateScorer(5, hidden=48)
        elif mode == "sure":
            self.stage = ScaleAligner(c2)
            in_dim = c2 + c3 + 1 + 5
            self.refiner = CandidateRefiner(in_dim, hidden, sigma_dim=4, num_bins=num_bins)
            self.quality = QualityHead(
                in_dim + 4, hidden=64, sigma_dim=4, dist_stat_dim=4 * num_bins + 8
            )
        else:  # mode == "prism"
            self.stage = PRISMFeatureStage(c2, c3, c4, hidden=max(32, c2))
            in_dim = c2 + c3 + 2 + 5
            self.refiner = CandidateRefiner(in_dim, hidden, sigma_dim=4, num_bins=num_bins)
            self.quality = QualityHead(
                in_dim + 4, hidden=64, sigma_dim=4, dist_stat_dim=4 * num_bins + 8
            )
            self.assign = AssignerMLP(6, hidden=48)
            self.duplicate = DuplicateScorer(5, hidden=48)

    @classmethod
    def from_detect(
        cls,
        head: Detect,
        mode: str = "prism",
        topk: int = 300,
        hidden: int = 128,
        aux_weight: float = 0.5,
    ) -> CompositeDetect:
        """Convert a stock head without changing its detection weights."""
        channels = []
        for branch in head.cv2:
            first = branch[0]
            conv = first.conv if hasattr(first, "conv") else first
            channels.append(int(conv.in_channels))
        result = cls(
            head.nc,
            head.reg_max,
            hasattr(head, "one2one_cv2"),
            channels,
            mode,
            topk,
            hidden,
            aux_weight,
        )
        result._end2end = bool(getattr(head, "_end2end", False))
        result.load_state_dict(head.state_dict(), strict=False)
        for name in ("i", "f", "type", "np", "inplace"):
            if hasattr(head, name):
                setattr(result, name, getattr(head, name))
        result.stride = head.stride.clone()
        result.anchors = head.anchors.clone() if isinstance(head.anchors, Tensor) else head.anchors
        result.strides = head.strides.clone() if isinstance(head.strides, Tensor) else head.strides
        result.shape = head.shape
        return result

    def _prepare_features(self, features: list[Tensor]) -> list[Tensor]:
        p2, p3, p4 = features[:3]
        if self.composite_mode == "cglr":
            assert isinstance(self.stage, ContextEvidenceGate)
            p2, gate, evidence, logits = self.stage(p2, p3, p4)
            self._last_aux = {"gate": gate, "gate_logits": logits, "evidence": evidence}
        elif self.composite_mode == "cass":
            assert isinstance(self.stage, DensityRouter)
            p2, density, context, logits = self.stage(p2, p3, p4)
            self._last_aux = {"density": density, "density_logits": logits, "context": context}
        elif self.composite_mode == "sure":
            assert isinstance(self.stage, ScaleAligner)
            p2, scale = self.stage(p2)
            self._last_aux = {"scale_map": scale}
        else:
            assert isinstance(self.stage, PRISMFeatureStage)
            stage_out = self.stage(p2, p3, p4)
            p2 = stage_out["p2"]
            self._last_aux = {k: v for k, v in stage_out.items() if k != "p2"}
        return [p2, *features[1:]]

    def forward(self, x: list[Tensor]):
        prepared = self._prepare_features(x)
        output = super().forward(prepared)
        if self.training and isinstance(output, dict):
            output["composite"] = {"mode": self.composite_mode, "feats": prepared, **self._last_aux}
        return output

    def _decode(self, prediction: dict[str, Tensor]) -> Tensor:
        anchors, strides = make_anchors(prediction["feats"], self.stride, 0.5)
        anchors = anchors.transpose(0, 1).unsqueeze(0)
        strides = strides.transpose(0, 1).unsqueeze(0)
        distances = self.dfl(prediction["boxes"].float())
        boxes = dist2bbox(distances, anchors.float(), xywh=False, dim=1)
        return (boxes * strides.float()).permute(0, 2, 1)

    def _candidates(
        self, prediction: dict[str, Tensor], limit: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            boxes = self._decode(prediction)
            scores = prediction["scores"].float().sigmoid().amax(dim=1)
            index = scores.topk(min(limit, scores.shape[1]), dim=1).indices
            selected = boxes.gather(1, index[..., None].expand(-1, -1, 4)).detach()
            selected_scores = scores.gather(1, index).detach()
        return selected, selected_scores, index

    def _roi_features(
        self, features: list[Tensor], boxes: Tensor, auxiliary: Tensor | None
    ) -> Tensor:
        stride0 = float(self.stride[0])
        stride1 = float(self.stride[1])
        height = features[0].shape[-2] * stride0
        width = features[0].shape[-1] * stride0
        assert self.roi_proj_p2 is not None and self.roi_proj_p3 is not None
        grid_p2 = _sample_roi_grid(features[0].float(), boxes, stride0, points=3)
        grid_p3 = _sample_roi_grid(features[1].float(), boxes, stride1, points=3)
        parts = [
            self.roi_proj_p2(grid_p2, return_center=False),
            self.roi_proj_p3(grid_p3, return_center=False),
        ]
        if auxiliary is not None:
            parts.append(_sample_roi(auxiliary.float(), boxes, stride0))
        normalized = _cxcywh_norm(boxes, float(width), float(height))
        # Safe sqrt with minimum clamp >= 1e-6 to avoid infinite derivative at 0
        size = (normalized[..., 2] * normalized[..., 3]).clamp_min(1e-6).sqrt().unsqueeze(-1)
        parts.append(torch.cat((normalized, size), dim=-1))
        return torch.cat(parts, dim=-1)

    def _extract_prism_roi(
        self, features: list[Tensor], boxes: Tensor, gate: Tensor, density: Tensor
    ) -> tuple[Tensor, Tensor]:
        stride0 = float(self.stride[0])
        stride1 = float(self.stride[1])
        height = features[0].shape[-2] * stride0
        width = features[0].shape[-1] * stride0
        assert self.roi_proj_p2 is not None and self.roi_proj_p3 is not None
        f_p2, center_p2 = self.roi_proj_p2(
            _sample_roi_grid(features[0].float(), boxes, stride0, points=3), return_center=True
        )
        f_p3 = self.roi_proj_p3(
            _sample_roi_grid(features[1].float(), boxes, stride1, points=3), return_center=False
        )
        aux_map = torch.cat((gate.float(), density.float()), dim=1)
        f_aux = _sample_roi(aux_map, boxes, stride0, points=3)
        normalized = _cxcywh_norm(boxes, float(width), float(height))
        size = (normalized[..., 2] * normalized[..., 3]).clamp_min(1e-6).sqrt().unsqueeze(-1)
        geom = torch.cat((normalized, size), dim=-1)
        return torch.cat((f_p2, f_p3, f_aux, geom), dim=-1), center_p2

    def _inference(self, x: dict[str, Tensor]) -> Tensor:
        raw = super()._inference(x)
        with torch.cuda.amp.autocast(enabled=False):
            raw_f = raw.float()
            feats_f = [f.float() for f in x["feats"]]
            if self.composite_mode == "prism":
                out = self._apply_prism_inference(raw_f, feats_f)
            elif self.composite_mode == "cass":
                out = self._apply_duplicate_penalty(raw_f, feats_f)
            else:
                out = self._apply_refinement(raw_f, feats_f)
        return out.to(dtype=raw.dtype)

    def _apply_refinement(self, raw: Tensor, features: list[Tensor]) -> Tensor:
        scores = raw[:, 4:].amax(dim=1)
        index = scores.topk(min(self.topk, scores.shape[1]), dim=1).indices
        boxes = raw[:, :4].gather(2, index[:, None, :].expand(-1, 4, -1)).transpose(1, 2)
        auxiliary = (
            self._last_aux.get("gate")
            if self.composite_mode == "cglr"
            else self._last_aux.get("scale_map")
        )
        roi = self._roi_features(features, boxes, auxiliary)
        assert self.refiner is not None and self.quality is not None
        delta, log_sigma = self.refiner(roi)
        stride = float(self.stride[0])
        height, width = features[0].shape[-2] * stride, features[0].shape[-1] * stride
        refined = _apply_delta(boxes, delta, width, height)
        quality = self.quality(
            torch.cat((roi, log_sigma), dim=-1), dist_stats=self.refiner._last_dist_stats
        ).sigmoid()
        result = raw.clone()
        result[:, :4].scatter_(2, index[:, None, :].expand(-1, 4, -1), refined.transpose(1, 2))
        selected = result[:, 4:].gather(2, index[:, None, :].expand(-1, self.nc, -1))
        result[:, 4:].scatter_(
            2, index[:, None, :].expand(-1, self.nc, -1), selected * quality.transpose(1, 2)
        )
        return result

    def _apply_duplicate_penalty(self, raw: Tensor, features: list[Tensor]) -> Tensor:
        scores = raw[:, 4:].amax(dim=1)
        index = scores.topk(min(self.topk, scores.shape[1]), dim=1).indices
        boxes = raw[:, :4].gather(2, index[:, None, :].expand(-1, 4, -1)).transpose(1, 2)
        stride = float(self.stride[0])
        width, height = features[0].shape[-1] * stride, features[0].shape[-2] * stride
        assert self.roi_proj_p2 is not None and self.duplicate is not None
        pooled = self.roi_proj_p2(
            _sample_roi_grid(features[0], boxes, stride, points=3), return_center=False
        )
        dense = _sample_roi(self._last_aux["density"].float(), boxes, stride).squeeze(-1)
        result = raw.clone()
        penalties = boxes.new_zeros(boxes.shape[:2])
        for image in range(boxes.shape[0]):
            iou = _pair_iou(boxes[image], boxes[image])
            centers = (boxes[image, :, :2] + boxes[image, :, 2:]) * 0.5
            distance = _safe_dist(centers[:, None, :], centers[None, :, :]) / (
                width + height + 1e-4
            )
            area = (boxes[image, :, 2] - boxes[image, :, 0]).clamp_min(1e-2) * (
                boxes[image, :, 3] - boxes[image, :, 1]
            ).clamp_min(1e-2)
            ratio = (area[:, None] / area[None, :]).clamp(0.1, 10.0)
            norm_p = F.normalize(pooled[image], dim=-1, eps=1e-5)
            cosine = (norm_p @ norm_p.T).clamp(-1.0, 1.0)
            pair = torch.stack(
                (
                    iou,
                    distance,
                    ratio,
                    cosine,
                    (dense[image, :, None] + dense[image, None, :]) * 0.5,
                ),
                dim=-1,
            )
            dup_prob = self.duplicate(pair).sigmoid()
            img_scores = scores[image, index[image]]
            lower_score_mask = (img_scores[:, None] < img_scores[None, :]).float()
            mask = (iou > 0.3).float() * lower_score_mask
            mask.fill_diagonal_(0)
            penalties[image] = (dup_prob * mask).max(dim=-1).values
        selected = raw[:, 4:].gather(2, index[:, None, :].expand(-1, self.nc, -1))
        result[:, 4:].scatter_(
            2,
            index[:, None, :].expand(-1, self.nc, -1),
            selected * (1.0 - 0.75 * penalties[..., None]).transpose(1, 2),
        )
        return result

    def _apply_prism_inference(self, raw: Tensor, features: list[Tensor]) -> Tensor:
        scores = raw[:, 4:].amax(dim=1)
        index = scores.topk(min(self.topk, scores.shape[1]), dim=1).indices
        boxes = raw[:, :4].gather(2, index[:, None, :].expand(-1, 4, -1)).transpose(1, 2)

        gate = self._last_aux["gate"].float()
        density = self._last_aux["density"].float()
        roi, center_sig = self._extract_prism_roi(features, boxes, gate, density)
        assert self.refiner is not None and self.quality is not None and self.duplicate is not None

        delta, log_sigma = self.refiner(roi)
        stride = float(self.stride[0])
        height, width = features[0].shape[-2] * stride, features[0].shape[-1] * stride
        refined = _apply_delta(boxes, delta, width, height)

        quality = self.quality(
            torch.cat((roi, log_sigma), dim=-1), dist_stats=self.refiner._last_dist_stats
        ).sigmoid()

        dense_roi = _sample_roi(density, refined, stride).squeeze(-1)
        penalties = refined.new_zeros(refined.shape[:2])
        for image in range(refined.shape[0]):
            iou = _pair_iou(refined[image], refined[image])
            centers = (refined[image, :, :2] + refined[image, :, 2:]) * 0.5
            distance = _safe_dist(centers[:, None, :], centers[None, :, :]) / (
                width + height + 1e-4
            )
            area = (refined[image, :, 2] - refined[image, :, 0]).clamp_min(1e-2) * (
                refined[image, :, 3] - refined[image, :, 1]
            ).clamp_min(1e-2)
            ratio = (area[:, None] / area[None, :]).clamp(0.1, 10.0)
            norm_c = F.normalize(center_sig[image], dim=-1, eps=1e-5)
            cosine = (norm_c @ norm_c.T).clamp(-1.0, 1.0)
            pair = torch.stack(
                (
                    iou,
                    distance,
                    ratio,
                    cosine,
                    (dense_roi[image, :, None] + dense_roi[image, None, :]) * 0.5,
                ),
                dim=-1,
            )
            dup_prob = self.duplicate(pair).sigmoid()
            cand_rank_score = scores[image, index[image]] * quality[image, :, 0]
            lower_mask = (cand_rank_score[:, None] < cand_rank_score[None, :]).float()
            overlap_mask = (iou > 0.25).float() * lower_mask
            overlap_mask.fill_diagonal_(0)
            penalties[image] = (dup_prob * overlap_mask).max(dim=-1).values

        modulation = quality * (1.0 - 0.65 * penalties.unsqueeze(-1))
        result = raw.clone()
        result[:, :4].scatter_(2, index[:, None, :].expand(-1, 4, -1), refined.transpose(1, 2))
        selected = result[:, 4:].gather(2, index[:, None, :].expand(-1, self.nc, -1))
        result[:, 4:].scatter_(
            2, index[:, None, :].expand(-1, self.nc, -1), selected * modulation.transpose(1, 2)
        )
        return result

    def _match(
        self, boxes: Tensor, gt_boxes: Tensor, gt_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if gt_boxes.shape[1] == 0:
            shape = boxes.shape[:2] + (0,)
            return (
                boxes.new_zeros(boxes.shape[:2]),
                boxes.new_zeros(boxes.shape[:2], dtype=torch.long),
                boxes.new_zeros(shape),
            )
        ious = boxes.new_zeros(boxes.shape[:2] + (gt_boxes.shape[1],))
        for image in range(boxes.shape[0]):
            if gt_mask[image].any():
                valid_gt = gt_boxes[image][gt_mask[image]]
                ious[image, :, : valid_gt.shape[0]] = _pair_iou(boxes[image], valid_gt)
        best_iou, best_index = ious.max(dim=-1)
        return best_iou, best_index, ious

    def _zero_aux(self, device: torch.device) -> Tensor:
        values = [parameter.sum() * 0.0 for parameter in self.parameters()]
        return sum(values) if values else torch.zeros((), device=device)

    def compute_aux_loss(
        self, preds: dict[str, Any], batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        info = preds.get("composite")
        if info is None:
            return self._zero_aux(next(self.parameters()).device), {}
        branch = preds.get("one2many", preds)

        # CRITICAL: Force FP32 inside auxiliary loss to prevent AMP FP16 overflow/underflow
        with torch.cuda.amp.autocast(enabled=False):
            boxes, _, _ = self._candidates(branch, self.topk)
            stride = float(self.stride[0])
            height = float(branch["feats"][0].shape[-2] * stride)
            width = float(branch["feats"][0].shape[-1] * stride)
            gt_boxes, gt_labels, gt_mask = _padded_targets(
                batch, boxes.shape[0], width, height, boxes.device
            )
            if gt_boxes.shape[1] == 0 or not gt_mask.any():
                return self._zero_aux(boxes.device), {}

            info_f32 = {
                k: (
                    [t.float() for t in v]
                    if isinstance(v, list)
                    else (v.float() if isinstance(v, Tensor) else v)
                )
                for k, v in info.items()
            }

            if self.composite_mode == "cglr":
                loss, items = self._aux_cglr(info_f32, boxes, gt_boxes, gt_mask, height, width)
            elif self.composite_mode == "cass":
                loss, items = self._aux_cass(
                    info_f32, boxes, gt_boxes, gt_labels, gt_mask, height, width
                )
            elif self.composite_mode == "sure":
                loss, items = self._aux_sure(info_f32, boxes, gt_boxes, gt_mask, height, width)
            else:
                loss, items = self._aux_prism(info_f32, boxes, gt_boxes, gt_mask, height, width)

            # Final numerical guard: if any unforeseen NaN/Inf occurs, replace with 0 to never crash training
            if not torch.isfinite(loss):
                loss = self._zero_aux(boxes.device)
            safe_items = {
                k: torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in items.items()
            }
            return loss, safe_items

    def _refine(
        self,
        info: dict[str, Tensor],
        boxes: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
        height: float,
        width: float,
        auxiliary_key: str,
    ) -> tuple:
        auxiliary = info.get(auxiliary_key)
        roi = self._roi_features([info["feats"][0], info["feats"][1]], boxes, auxiliary)
        assert self.refiner is not None and self.quality is not None
        delta, log_sigma = self.refiner(roi)
        refined = _apply_delta(boxes, delta, width, height)
        best_iou, best_index, _ = self._match(boxes, gt_boxes, gt_mask)
        valid = (best_iou > 0.10).float()
        target = gt_boxes.gather(1, best_index[..., None].expand(-1, -1, 4))
        return roi, delta, log_sigma, refined, best_iou, target, valid

    def _aux_cglr(
        self,
        info: dict[str, Tensor],
        boxes: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
        height: float,
        width: float,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        logits = info["gate_logits"].clamp(-10.0, 10.0)
        target = torch.zeros_like(logits)
        yy = torch.arange(logits.shape[-2], device=logits.device, dtype=torch.float32)
        xx = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float32)
        for image in range(gt_boxes.shape[0]):
            for box in gt_boxes[image][gt_mask[image]]:
                cx = ((box[0] + box[2]) * 0.5) / width * logits.shape[-1] - 0.5
                cy = ((box[1] + box[3]) * 0.5) / height * logits.shape[-2] - 0.5
                radius = max(
                    1.2, math.sqrt(max(float((box[2] - box[0]) * (box[3] - box[1])), 1.0)) / 4.0
                )
                distance = (xx[None] - cx).square() + (yy[:, None] - cy).square()
                soft_kernel = torch.exp(-0.5 * distance / (radius * radius)).masked_fill(
                    distance > (radius * 1.5) ** 2, 0.0
                )
                target[image, 0] = torch.maximum(target[image, 0], soft_kernel)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = logits.sigmoid()
        focal = (
            bce * (1.0 - probability).square() * (target > 0.1).float()
            + bce * probability.square() * (target <= 0.1).float()
        ).mean()

        roi, delta, log_sigma, refined, best_iou, target_box, valid = self._refine(
            info, boxes, gt_boxes, gt_mask, height, width, "gate"
        )
        iou = (
            _pair_iou(refined.reshape(-1, 4), target_box.reshape(-1, 4))
            .diagonal()
            .reshape(best_iou.shape)
        )
        near_miss_boost = torch.where((best_iou > 0.10) & (best_iou < 0.65), 1.5, 0.5)
        weight = torch.where(best_iou >= 0.75, 0.25, near_miss_boost) * valid
        normalizer = weight.sum().clamp_min(1.0)
        box_loss = ((1.0 - iou) * weight).sum() / normalizer

        raw_cxcywh = _cxcywh_norm(boxes, width, height)
        gt_cxcywh = _cxcywh_norm(target_box, width, height)
        target_delta = torch.cat(
            (
                (gt_cxcywh[..., :2] - raw_cxcywh[..., :2]) / raw_cxcywh[..., 2:].clamp_min(1e-3),
                (gt_cxcywh[..., 2:].clamp_min(1e-3) / raw_cxcywh[..., 2:].clamp_min(1e-3)).log(),
            ),
            dim=-1,
        ).clamp(-0.25, 0.25)
        residual = F.smooth_l1_loss(delta, target_delta, reduction="none").mean(-1)
        residual_loss = (residual * weight).sum() / normalizer

        assert self.quality is not None
        q_logits = self.quality(torch.cat((roi, log_sigma), -1))
        q_target = (iou.detach() * valid).clamp(0.0, 1.0)
        quality_loss = F.binary_cross_entropy_with_logits(
            q_logits.squeeze(-1), q_target, reduction="none"
        )
        quality_loss = (quality_loss * (valid + 0.1)).sum() / normalizer

        sigma = F.softplus(log_sigma).clamp(0.25, 2.0) * float(self.stride[0])
        error = ((refined - target_box).abs().mean(-1, keepdim=True) / float(self.stride[0])).clamp(
            0.0, 10.0
        )
        sigma_Norm = sigma / float(self.stride[0])
        sigma_terms = 0.5 * (error / sigma_Norm).square() + sigma_Norm.log()
        sigma_loss = (sigma_terms * valid[..., None]).sum() / valid.sum().clamp_min(1.0)

        loss = (
            (focal + box_loss + residual_loss + quality_loss + 0.1 * sigma_loss)
            * boxes.shape[0]
            * self.aux_weight
        )
        return loss, {
            "gate_loss": focal.detach(),
            "refine_loss": (box_loss + residual_loss).detach(),
            "quality_loss": quality_loss.detach(),
            "sigma_loss": sigma_loss.detach(),
        }

    def _aux_cass(
        self,
        info: dict[str, Tensor],
        boxes: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        gt_mask: Tensor,
        height: float,
        width: float,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        logits = info["density_logits"].clamp(-10.0, 10.0)
        centers = torch.zeros_like(logits)
        for image in range(gt_boxes.shape[0]):
            for box in gt_boxes[image][gt_mask[image]]:
                cx = int(((box[0] + box[2]) * 0.5) / width * logits.shape[-1])
                cy = int(((box[1] + box[3]) * 0.5) / height * logits.shape[-2])
                if 0 <= cx < logits.shape[-1] and 0 <= cy < logits.shape[-2]:
                    centers[image, 0, cy, cx] += 1.0
        density_target = F.avg_pool2d(centers, 4, stride=4).mul(16.0).clamp(0.0, 4.0) / 4.0
        density_target = F.interpolate(
            density_target, size=logits.shape[-2:], mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        bce = F.binary_cross_entropy_with_logits(logits, density_target, reduction="none")
        probability = logits.sigmoid()
        density_loss = (
            bce * (1.0 - probability).square() * density_target
            + bce * probability.square() * (1.0 - density_target)
        ).mean()

        assert (
            self.roi_proj_p2 is not None and self.assign is not None and self.duplicate is not None
        )
        roi = self.roi_proj_p2(
            _sample_roi_grid(info["feats"][0], boxes, float(self.stride[0]), points=3),
            return_center=False,
        )

        assign_terms = []
        best_iou, best_index, _ = self._match(boxes, gt_boxes, gt_mask)
        for image in range(boxes.shape[0]):
            valid_gt = gt_boxes[image][gt_mask[image]]
            count = int(valid_gt.shape[0])
            if count == 0:
                continue
            ious = _pair_iou(boxes[image], valid_gt)
            top = min(16, boxes.shape[1])
            idx = ious.topk(top, dim=0).indices.T
            candidate = boxes[image][idx]
            target = valid_gt[:, None, :].expand_as(candidate)
            iou = (
                _pair_iou(candidate.reshape(-1, 4), target.reshape(-1, 4))
                .diagonal()
                .reshape(idx.shape)
            )
            cand_centers = (candidate[..., :2] + candidate[..., 2:]) * 0.5
            gt_centers = (valid_gt[:, :2] + valid_gt[:, 2:]) * 0.5
            distance = (
                (cand_centers - gt_centers[:, None]).abs().mean(-1) / (width + height + 1e-4)
            ).clamp(0.0, 2.0)
            area = (candidate[..., 2] - candidate[..., 0]).clamp_min(1e-2) * (
                candidate[..., 3] - candidate[..., 1]
            ).clamp_min(1e-2)
            ratio = (area / area.mean(-1, keepdim=True).clamp_min(1e-2)).clamp(0.1, 10.0)
            dense = (
                _sample_roi(
                    info["density"][image : image + 1],
                    candidate.reshape(1, -1, 4),
                    float(self.stride[0]),
                )
                .reshape(count, top, -1)[..., 0]
                .clamp(0.0, 1.0)
            )

            gt_roi = self.roi_proj_p2(
                _sample_roi_grid(
                    info["feats"][0][image : image + 1],
                    valid_gt[None],
                    float(self.stride[0]),
                    points=3,
                ),
                return_center=False,
            ).squeeze(0)
            cand_roi = roi[image][idx]
            vis_sim = F.cosine_similarity(cand_roi, gt_roi[:, None, :], dim=-1, eps=1e-5).clamp(
                -1.0, 1.0
            )

            prior = (iou.clamp_min(1e-3).log() - distance + 0.5 * vis_sim).clamp(-6.0, 6.0)
            features = torch.stack((iou, distance, ratio, dense, vis_sim, prior), dim=-1)
            assign_logits = (self.assign(features) + prior).clamp(-8.0, 8.0)

            log_resp = assign_logits / 0.5
            for _ in range(3):
                log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
                log_resp = log_resp - torch.logsumexp(log_resp, dim=0, keepdim=True)
            responsibility = log_resp.exp()
            responsibility = responsibility / responsibility.sum(1, keepdim=True).clamp_min(1e-5)
            assign_terms.append(
                -(responsibility.detach() * F.log_softmax(assign_logits, dim=1)).sum(1).mean()
            )

        assign_loss = (
            torch.stack(assign_terms).mean() if assign_terms else density_loss.new_zeros(())
        )

        duplicate_terms = []
        dense_all = (
            _sample_roi(info["density"], boxes, float(self.stride[0])).squeeze(-1).clamp(0.0, 1.0)
        )
        for image in range(boxes.shape[0]):
            sub = boxes[image]
            iou = _pair_iou(sub, sub)
            candidate_mask = torch.triu(iou > 0.3, diagonal=1)
            pairs = candidate_mask.nonzero(as_tuple=False)
            if pairs.numel() == 0:
                continue
            pairs = pairs[:256]
            left, right = pairs[:, 0], pairs[:, 1]
            target_pair = (
                (best_index[image, left] == best_index[image, right])
                & (best_iou[image, left] > 0.5)
                & (best_iou[image, right] > 0.5)
            ).float()
            centers = (sub[:, :2] + sub[:, 2:]) * 0.5
            area_l = (sub[left, 2] - sub[left, 0]).clamp_min(1e-2) * (
                sub[left, 3] - sub[left, 1]
            ).clamp_min(1e-2)
            area_r = (sub[right, 2] - sub[right, 0]).clamp_min(1e-2) * (
                sub[right, 3] - sub[right, 1]
            ).clamp_min(1e-2)
            pair = torch.stack(
                (
                    iou[left, right],
                    _safe_dist(centers[left], centers[right]) / (width + height + 1e-4),
                    (area_l / area_r).clamp(0.1, 10.0),
                    F.cosine_similarity(
                        roi[image, left], roi[image, right], dim=-1, eps=1e-5
                    ).clamp(-1.0, 1.0),
                    (dense_all[image, left] + dense_all[image, right]) * 0.5,
                ),
                dim=-1,
            )
            duplicate_terms.append(
                F.binary_cross_entropy_with_logits(self.duplicate(pair), target_pair)
            )
        duplicate_loss = (
            torch.stack(duplicate_terms).mean() if duplicate_terms else density_loss.new_zeros(())
        )
        loss = (density_loss + assign_loss + duplicate_loss) * boxes.shape[0] * self.aux_weight
        return loss, {
            "density_loss": density_loss.detach(),
            "assign_loss": assign_loss.detach(),
            "duplicate_loss": duplicate_loss.detach(),
        }

    def _aux_sure(
        self,
        info: dict[str, Tensor],
        boxes: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
        height: float,
        width: float,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        roi, delta, log_sigma, refined, best_iou, target_box, valid = self._refine(
            info, boxes, gt_boxes, gt_mask, height, width, "scale_map"
        )
        iou = (
            _pair_iou(refined.reshape(-1, 4), target_box.reshape(-1, 4))
            .diagonal()
            .reshape(best_iou.shape)
        )
        weight = torch.where(best_iou >= 0.75, 0.25, torch.ones_like(best_iou)) * valid
        normalizer = weight.sum().clamp_min(1.0)

        raw_cxcywh = _cxcywh_norm(boxes, width, height)
        gt_cxcywh = _cxcywh_norm(target_box, width, height)
        target_delta = torch.cat(
            (
                (gt_cxcywh[..., :2] - raw_cxcywh[..., :2]) / raw_cxcywh[..., 2:].clamp_min(1e-3),
                (gt_cxcywh[..., 2:].clamp_min(1e-3) / raw_cxcywh[..., 2:].clamp_min(1e-3)).log(),
            ),
            dim=-1,
        ).clamp(-0.25, 0.25)
        delta_loss = (
            F.smooth_l1_loss(delta, target_delta, reduction="none").mean(-1) * weight
        ).sum() / normalizer
        box_loss = ((1.0 - iou) * weight).sum() / normalizer + delta_loss

        pred_ltrb = (
            torch.stack(
                (
                    refined[..., 0] - target_box[..., 0],
                    refined[..., 1] - target_box[..., 1],
                    target_box[..., 2] - refined[..., 2],
                    target_box[..., 3] - refined[..., 3],
                ),
                dim=-1,
            ).abs()
            / float(self.stride[0])
        ).clamp(0.0, 10.0)
        sigma_norm = F.softplus(log_sigma).clamp(0.25, 2.0)
        sigma_terms = 0.5 * (pred_ltrb / sigma_norm).square() + sigma_norm.log()
        sigma_loss = (sigma_terms * valid[..., None]).sum() / (valid.sum() * 4.0).clamp_min(1.0)

        assert self.quality is not None and self.refiner is not None
        q_logits = self.quality(
            torch.cat((roi, log_sigma), -1), dist_stats=self.refiner._last_dist_stats
        )
        q_target = (iou.detach() * valid).clamp(0.0, 1.0)
        quality_loss = F.binary_cross_entropy_with_logits(
            q_logits.squeeze(-1), q_target, reduction="none"
        )
        quality_loss = (quality_loss * (valid + 0.1)).sum() / normalizer

        loss = (box_loss + sigma_loss + quality_loss) * boxes.shape[0] * self.aux_weight
        return loss, {
            "sure_box_loss": box_loss.detach(),
            "sure_sigma_loss": sigma_loss.detach(),
            "sure_quality_loss": quality_loss.detach(),
        }

    def _aux_prism(
        self,
        info: dict[str, Tensor],
        boxes: Tensor,
        gt_boxes: Tensor,
        gt_mask: Tensor,
        height: float,
        width: float,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        gate_logits = info["gate_logits"].clamp(-10.0, 10.0)
        density_logits = info["density_logits"].clamp(-10.0, 10.0)
        gate_target = torch.zeros_like(gate_logits)
        centers_map = torch.zeros_like(density_logits)

        yy = torch.arange(gate_logits.shape[-2], device=gate_logits.device, dtype=torch.float32)
        xx = torch.arange(gate_logits.shape[-1], device=gate_logits.device, dtype=torch.float32)
        for image in range(gt_boxes.shape[0]):
            for box in gt_boxes[image][gt_mask[image]]:
                cx = ((box[0] + box[2]) * 0.5) / width * gate_logits.shape[-1] - 0.5
                cy = ((box[1] + box[3]) * 0.5) / height * gate_logits.shape[-2] - 0.5
                radius = max(
                    1.2, math.sqrt(max(float((box[2] - box[0]) * (box[3] - box[1])), 1.0)) / 4.0
                )
                dist_sq = (xx[None] - cx).square() + (yy[:, None] - cy).square()
                soft_gauss = torch.exp(-0.5 * dist_sq / (radius * radius)).masked_fill(
                    dist_sq > (radius * 1.5) ** 2, 0.0
                )
                gate_target[image, 0] = torch.maximum(gate_target[image, 0], soft_gauss)
                ix, iy = int(cx + 0.5), int(cy + 0.5)
                if 0 <= ix < density_logits.shape[-1] and 0 <= iy < density_logits.shape[-2]:
                    centers_map[image, 0, iy, ix] += 1.0

        gate_bce = F.binary_cross_entropy_with_logits(gate_logits, gate_target, reduction="none")
        gate_prob = gate_logits.sigmoid()
        gate_loss = (
            gate_bce * (1.0 - gate_prob).square() * (gate_target > 0.1).float()
            + gate_bce * gate_prob.square() * (gate_target <= 0.1).float()
        ).mean()

        density_target = F.avg_pool2d(centers_map, 4, stride=4).mul(16.0).clamp(0.0, 4.0) / 4.0
        density_target = F.interpolate(
            density_target, size=density_logits.shape[-2:], mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        dens_bce = F.binary_cross_entropy_with_logits(
            density_logits, density_target, reduction="none"
        )
        dens_prob = density_logits.sigmoid()
        density_loss = (
            dens_bce * (1.0 - dens_prob).square() * density_target
            + dens_bce * dens_prob.square() * (1.0 - density_target)
        ).mean()

        roi, center_sig = self._extract_prism_roi(
            [info["feats"][0], info["feats"][1]], boxes, info["gate"], info["density"]
        )
        assert self.refiner is not None and self.quality is not None
        assert (
            self.assign is not None and self.duplicate is not None and self.roi_proj_p2 is not None
        )

        delta, log_sigma = self.refiner(roi)
        refined = _apply_delta(boxes, delta, width, height)
        best_iou, best_index, _ = self._match(boxes, gt_boxes, gt_mask)
        valid = (best_iou > 0.10).float()
        target_box = gt_boxes.gather(1, best_index[..., None].expand(-1, -1, 4))

        iou = (
            _pair_iou(refined.reshape(-1, 4), target_box.reshape(-1, 4))
            .diagonal()
            .reshape(best_iou.shape)
        )
        near_miss_weight = torch.where((best_iou > 0.10) & (best_iou < 0.65), 1.5, 0.6)
        weight = torch.where(best_iou >= 0.75, 0.25, near_miss_weight) * valid
        normalizer = weight.sum().clamp_min(1.0)

        raw_cxcywh = _cxcywh_norm(boxes, width, height)
        gt_cxcywh = _cxcywh_norm(target_box, width, height)
        target_delta = torch.cat(
            (
                (gt_cxcywh[..., :2] - raw_cxcywh[..., :2]) / raw_cxcywh[..., 2:].clamp_min(1e-3),
                (gt_cxcywh[..., 2:].clamp_min(1e-3) / raw_cxcywh[..., 2:].clamp_min(1e-3)).log(),
            ),
            dim=-1,
        ).clamp(-0.25, 0.25)

        residual_loss = (
            F.smooth_l1_loss(delta, target_delta, reduction="none").mean(-1) * weight
        ).sum() / normalizer
        box_loss = ((1.0 - iou) * weight).sum() / normalizer + residual_loss

        pred_ltrb = (
            torch.stack(
                (
                    refined[..., 0] - target_box[..., 0],
                    refined[..., 1] - target_box[..., 1],
                    target_box[..., 2] - refined[..., 2],
                    target_box[..., 3] - refined[..., 3],
                ),
                dim=-1,
            ).abs()
            / float(self.stride[0])
        ).clamp(0.0, 10.0)
        sigma_norm = F.softplus(log_sigma).clamp(0.25, 2.0)
        sigma_terms = 0.5 * (pred_ltrb / sigma_norm).square() + sigma_norm.log()
        sigma_loss = (sigma_terms * valid[..., None]).sum() / (valid.sum() * 4.0).clamp_min(1.0)

        q_logits = self.quality(
            torch.cat((roi, log_sigma), dim=-1), dist_stats=self.refiner._last_dist_stats
        )
        q_target = (iou.detach() * valid).clamp(0.0, 1.0)
        quality_loss = F.binary_cross_entropy_with_logits(
            q_logits.squeeze(-1), q_target, reduction="none"
        )
        quality_loss = (quality_loss * (valid + 0.1)).sum() / normalizer

        assign_terms = []
        duplicate_terms = []
        dense_all = (
            _sample_roi(info["density"], boxes, float(self.stride[0])).squeeze(-1).clamp(0.0, 1.0)
        )
        for image in range(boxes.shape[0]):
            valid_gt = gt_boxes[image][gt_mask[image]]
            count = int(valid_gt.shape[0])
            if count > 0:
                ious_g = _pair_iou(boxes[image], valid_gt)
                top = min(16, boxes.shape[1])
                idx = ious_g.topk(top, dim=0).indices.T
                cand = boxes[image][idx]
                targ = valid_gt[:, None, :].expand_as(cand)
                c_iou = (
                    _pair_iou(cand.reshape(-1, 4), targ.reshape(-1, 4))
                    .diagonal()
                    .reshape(idx.shape)
                )
                c_centers = (cand[..., :2] + cand[..., 2:]) * 0.5
                g_centers = (valid_gt[:, :2] + valid_gt[:, 2:]) * 0.5
                dist_c = (
                    (c_centers - g_centers[:, None]).abs().mean(-1) / (width + height + 1e-4)
                ).clamp(0.0, 2.0)
                c_area = (cand[..., 2] - cand[..., 0]).clamp_min(1e-2) * (
                    cand[..., 3] - cand[..., 1]
                ).clamp_min(1e-2)
                c_ratio = (c_area / c_area.mean(-1, keepdim=True).clamp_min(1e-2)).clamp(0.1, 10.0)
                c_dense = dense_all[image][idx]

                _, gt_center_sig = self.roi_proj_p2(
                    _sample_roi_grid(
                        info["feats"][0][image : image + 1],
                        valid_gt[None],
                        float(self.stride[0]),
                        points=3,
                    ),
                    return_center=True,
                )
                vis_sim = F.cosine_similarity(
                    center_sig[image][idx], gt_center_sig.squeeze(0)[:, None, :], dim=-1, eps=1e-5
                ).clamp(-1.0, 1.0)
                prior = (c_iou.clamp_min(1e-3).log() - dist_c + 0.5 * vis_sim).clamp(-6.0, 6.0)
                a_feats = torch.stack((c_iou, dist_c, c_ratio, c_dense, vis_sim, prior), dim=-1)
                a_logits = (self.assign(a_feats) + prior).clamp(-8.0, 8.0)

                log_resp = a_logits / 0.5
                for _ in range(3):
                    log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
                    log_resp = log_resp - torch.logsumexp(log_resp, dim=0, keepdim=True)
                resp = log_resp.exp()
                resp = resp / resp.sum(1, keepdim=True).clamp_min(1e-5)
                assign_terms.append(-(resp.detach() * F.log_softmax(a_logits, dim=1)).sum(1).mean())

            sub = refined[image].detach()
            pair_iou = _pair_iou(sub, sub)
            cand_mask = torch.triu(pair_iou > 0.25, diagonal=1)
            pairs = cand_mask.nonzero(as_tuple=False)
            if pairs.numel() > 0:
                pairs = pairs[:256]
                left, right = pairs[:, 0], pairs[:, 1]
                target_pair = (
                    (best_index[image, left] == best_index[image, right])
                    & (best_iou[image, left] > 0.45)
                    & (best_iou[image, right] > 0.45)
                ).float()
                centers = (sub[:, :2] + sub[:, 2:]) * 0.5
                area_l = (sub[left, 2] - sub[left, 0]).clamp_min(1e-2) * (
                    sub[left, 3] - sub[left, 1]
                ).clamp_min(1e-2)
                area_r = (sub[right, 2] - sub[right, 0]).clamp_min(1e-2) * (
                    sub[right, 3] - sub[right, 1]
                ).clamp_min(1e-2)
                pair_feat = torch.stack(
                    (
                        pair_iou[left, right],
                        _safe_dist(centers[left], centers[right]) / (width + height + 1e-4),
                        (area_l / area_r).clamp(0.1, 10.0),
                        F.cosine_similarity(
                            center_sig[image, left], center_sig[image, right], dim=-1, eps=1e-5
                        ).clamp(-1.0, 1.0),
                        (dense_all[image, left] + dense_all[image, right]) * 0.5,
                    ),
                    dim=-1,
                )
                duplicate_terms.append(
                    F.binary_cross_entropy_with_logits(self.duplicate(pair_feat), target_pair)
                )

        assign_loss = torch.stack(assign_terms).mean() if assign_terms else gate_loss.new_zeros(())
        duplicate_loss = (
            torch.stack(duplicate_terms).mean() if duplicate_terms else gate_loss.new_zeros(())
        )

        total_aux = (
            (
                0.75 * gate_loss
                + 0.50 * density_loss
                + 1.25 * box_loss
                + 0.20 * sigma_loss
                + 1.00 * quality_loss
                + 0.35 * assign_loss
                + 0.50 * duplicate_loss
            )
            * boxes.shape[0]
            * self.aux_weight
        )

        return total_aux, {
            "prism_gate": (gate_loss + density_loss).detach(),
            "prism_box": box_loss.detach(),
            "prism_qual": quality_loss.detach(),
            "prism_crowd": (assign_loss + duplicate_loss).detach(),
        }


class CompositeCriterion:
    """Wrap the YOLO criterion and add the selected composite auxiliary loss with warmup."""

    def __init__(self, model: nn.Module, base: Any) -> None:
        self.base = base
        self.head = model.model[-1]
        self.updates = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, preds: Any, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        loss, items = self.base(preds, batch)
        raw = preds[1] if isinstance(preds, tuple) else preds
        auxiliary, auxiliary_items = self.head.compute_aux_loss(raw, batch)
        # Smooth warmup over the first 200 iterations so backbone/head stabilize first
        warmup = min(1.0, float(self.updates + 1) / 200.0)
        self.updates += 1
        return loss + warmup * auxiliary, {**items, **auxiliary_items}

    def update(self) -> None:
        if hasattr(self.base, "update"):
            self.base.update()


__all__ = [
    "AssignerMLP",
    "CandidateRefiner",
    "CompositeCriterion",
    "CompositeDetect",
    "ContextEvidenceGate",
    "DensityRouter",
    "DuplicateScorer",
    "FiveRegionROIExtractor",
    "GatedResMLPBlock",
    "PRISMFeatureStage",
    "QualityHead",
    "ResBottleneck2d",
    "ScaleAligner",
    "SpatialGridProjector",
]
