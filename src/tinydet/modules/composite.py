"""Orthogonal Multi-Hypothesis Composite Detection Heads for tinydet (Tiny Object Detection).

Four strictly independent scientific hypotheses (`COMPOSITE_MODES`):
  1. "cglr"  (Spatial Context)
  2. "cass"  (Crowd Topology)
  3. "sure"  (Scale & Distributional)
  4. "prism" (Spectral-Phase Frequency)
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
# 1. AMP-Safe Geometry & Spatial Grid Sampling
# ============================================================================


_GRID_CACHE: dict[tuple[int, torch.device], tuple[Tensor, Tensor]] = {}


def _get_local_grid(points: int, device: torch.device) -> tuple[Tensor, Tensor]:
    key = (points, device)
    cached = _GRID_CACHE.get(key)
    if cached is None:
        axis = torch.linspace(-1.0, 1.0, points, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        cached = (xx.reshape(-1)[None, None, :], yy.reshape(-1)[None, None, :])
        _GRID_CACHE[key] = cached
    return cached


def _sample_roi_grid(feature: Tensor, boxes: Tensor, stride: float, points: int = 3) -> Tensor:
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [B, K, 4]")
    b = boxes.shape[0]
    if feature.shape[0] != b:
        raise ValueError("feature and boxes must have the same batch size")

    height, width = feature.shape[-2:]
    image_w = max(float(width * stride), 1.0)
    image_h = max(float(height * stride), 1.0)

    safe_boxes = boxes.detach().float()
    x1 = safe_boxes[..., 0].clamp(0.0, image_w)
    y1 = safe_boxes[..., 1].clamp(0.0, image_h)
    x2 = safe_boxes[..., 2].clamp(0.0, image_w)
    y2 = safe_boxes[..., 3].clamp(0.0, image_h)

    local_x, local_y = _get_local_grid(points, feature.device)

    pixel_x = x1[..., None] + (local_x + 1.0) * 0.5 * (x2 - x1).clamp_min(1e-3)[..., None]
    pixel_y = y1[..., None] + (local_y + 1.0) * 0.5 * (y2 - y1).clamp_min(1e-3)[..., None]
    grid = (
        torch.stack((pixel_x / image_w * 2.0 - 1.0, pixel_y / image_h * 2.0 - 1.0), dim=-1)
        .clamp(-1.0, 1.0)
        .to(dtype=feature.dtype)
    )

    values = F.grid_sample(
        feature, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return values.permute(0, 2, 1, 3).contiguous()


def _sample_roi(feature: Tensor, boxes: Tensor, stride: float, points: int = 3) -> Tensor:
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


def _target_delta(raw_boxes: Tensor, target_boxes: Tensor, width: float, height: float) -> Tensor:
    raw_c = _cxcywh_norm(raw_boxes, width, height)
    targ_c = _cxcywh_norm(target_boxes, width, height)
    return torch.cat(
        (
            (targ_c[..., :2] - raw_c[..., :2]) / raw_c[..., 2:].clamp_min(1e-3),
            (targ_c[..., 2:].clamp_min(1e-3) / raw_c[..., 2:].clamp_min(1e-3)).log(),
        ),
        dim=-1,
    ).clamp(-0.25, 0.25)


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


def _elementwise_iou(a: Tensor, b: Tensor) -> Tensor:
    a_f = a.float()
    b_f = b.float()
    left = torch.maximum(a_f[..., :2], b_f[..., :2])
    right = torch.minimum(a_f[..., 2:], b_f[..., 2:])
    intersection = (right - left).clamp_min(0.0).prod(-1)
    area_a = (a_f[..., 2] - a_f[..., 0]).clamp_min(1e-3) * (a_f[..., 3] - a_f[..., 1]).clamp_min(1e-3)
    area_b = (b_f[..., 2] - b_f[..., 0]).clamp_min(1e-3) * (b_f[..., 3] - b_f[..., 1]).clamp_min(1e-3)
    union = (area_a + area_b - intersection).clamp_min(1e-4)
    return (intersection / union).clamp(0.0, 1.0)


def _safe_dist(a: Tensor, b: Tensor) -> Tensor:
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
    counts_cpu = counts.cpu().tolist()
    max_gt = max(counts_cpu) if counts_cpu else 0
    boxes = torch.zeros(batch_size, max_gt, 4, device=device, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_gt, device=device, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_gt, device=device, dtype=torch.bool)
    scale = torch.tensor((width, height, width, height), device=device, dtype=torch.float32)
    xywh = rows[:, 2:6]
    all_xyxy = (
        torch.cat((xywh[:, :2] - xywh[:, 2:] * 0.5, xywh[:, :2] + xywh[:, 2:] * 0.5), dim=1)
        * scale
    )

    offset = 0
    for image in range(batch_size):
        c = counts_cpu[image]
        if c == 0:
            continue
        start, end = offset, offset + c
        offset = end
        boxes[image, :c] = all_xyxy[start:end]
        labels[image, :c] = rows[start:end, 1]
        mask[image, :c] = True
    return boxes, labels, mask


# ============================================================================
# 2. Deep Core Blocks
# ============================================================================


class ConvNormAct(nn.Sequential):
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
    def __init__(self, channels: int, expansion: float = 1.5, dilation: int = 1) -> None:
        super().__init__()
        mid = max(16, round(channels * expansion))
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
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.15))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return self.act(x + self.gamma * (out * self.se(out)))


class GatedResMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: float = 1.5) -> None:
        super().__init__()
        hidden = max(32, round(dim * expansion))
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        # Fused GEMM: Compute gate and value in a single fused linear projection
        self.fc_in = nn.Linear(dim, hidden * 2)
        self.fc_out = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 0.15))
        nn.init.normal_(self.fc_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm(x)
        gate, val = self.fc_in(normed).chunk(2, dim=-1)
        gated = F.silu(gate) * val
        return x + self.gamma * self.fc_out(gated)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        gate_w, val_w = prefix + "fc_gate.weight", prefix + "fc_value.weight"
        if gate_w in state_dict and val_w in state_dict:
            state_dict[prefix + "fc_in.weight"] = torch.cat(
                (state_dict.pop(gate_w), state_dict.pop(val_w)), dim=0
            )
            gate_b, val_b = prefix + "fc_gate.bias", prefix + "fc_value.bias"
            if gate_b in state_dict and val_b in state_dict:
                state_dict[prefix + "fc_in.bias"] = torch.cat(
                    (state_dict.pop(gate_b), state_dict.pop(val_b)), dim=0
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class FiveRegionROIExtractor(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        basis = torch.zeros(7, 9)
        basis[0, [0, 3, 6]] = 1.0 / 3.0
        basis[1, [2, 5, 8]] = 1.0 / 3.0
        basis[2, [0, 1, 2]] = 1.0 / 3.0
        basis[3, [6, 7, 8]] = 1.0 / 3.0
        basis[4, 4] = 1.0
        basis[4, [0, 1, 2, 3, 5, 6, 7, 8]] = -1.0 / 8.0
        basis[5] = basis[1] - basis[0]
        basis[6] = basis[3] - basis[2]
        self.register_buffer("basis", basis, persistent=False)
        self.delta_basis = nn.Parameter(torch.zeros(7, 9))

        self.region_fuse = nn.Sequential(
            nn.Linear(channels * 7, channels),
            nn.LayerNorm(channels, eps=1e-5),
            nn.SiLU(inplace=True),
            GatedResMLPBlock(channels, expansion=1.5),
        )

    def forward(
        self, roi_grid: Tensor, return_details: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        param_dtype = self.delta_basis.dtype
        roi_grid_param = roi_grid.to(param_dtype)
        weights = self.basis.to(
            dtype=param_dtype, device=roi_grid.device
        ) + 0.1 * self.delta_basis
        regions = torch.matmul(roi_grid_param, weights.T)
        center_contrast = regions[..., 4]
        fused = roi_grid_param.mean(dim=-1) + 0.30 * self.region_fuse(regions.flatten(-2))
        if return_details:
            return fused.to(roi_grid.dtype), center_contrast.to(roi_grid.dtype)
        return fused.to(roi_grid.dtype)


SpatialGridProjector = FiveRegionROIExtractor


# ============================================================================
# 3. Orthogonal Feature Stages
# ============================================================================


class ContextEvidenceGate(nn.Module):
    def __init__(
        self, c2: int = 32, c3: int = 64, c4: int = 128, hidden: int = 32, alpha: float = 0.10
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.project3 = nn.Sequential(ConvNormAct(c3, hidden, 1), ResBottleneck2d(hidden))
        self.project4 = nn.Sequential(ConvNormAct(c4, hidden, 1), ResBottleneck2d(hidden))
        self.local_core = ConvNormAct(c2, c2, 3, padding=1, groups=c2)
        self.local_surround = nn.AvgPool2d(5, 1, 2)
        self.evidence = nn.Sequential(
            ConvNormAct(c2 * 2, c2, 1), ResBottleneck2d(c2), ResBottleneck2d(c2, dilation=2)
        )
        self.c2 = c2
        self.gate_neck = nn.Sequential(
            ConvNormAct(c2 + 2 * hidden + c2, hidden, 1), ResBottleneck2d(hidden)
        )
        # Fused 1x1 convolution projecting both channel gate (c2) and spatial gate (1)
        self.gate_proj = nn.Conv2d(hidden, c2 + 1, 1)
        nn.init.normal_(self.gate_proj.weight[:c2], 0, 0.01)
        nn.init.zeros_(self.gate_proj.bias[:c2])
        nn.init.normal_(self.gate_proj.weight[c2:], 0, 0.01)
        nn.init.constant_(self.gate_proj.bias[c2:], -2.0)

    def forward(self, p2: Tensor, p3: Tensor, p4: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sz = p2.shape[-2:]
        c3 = F.interpolate(self.project3(p3), size=sz, mode="bilinear", align_corners=False)
        c4 = F.interpolate(self.project4(p4), size=sz, mode="bilinear", align_corners=False)
        core = self.local_core(p2)
        ev = self.evidence(torch.cat((core, core - self.local_surround(p2)), dim=1))
        neck = self.gate_neck(torch.cat((p2, c3, c4, ev), dim=1))
        fused_gate = self.gate_proj(neck)
        ch_mod = fused_gate[:, : self.c2].clamp(-10.0, 10.0).sigmoid()
        logits = fused_gate[:, self.c2 :].clamp(-10.0, 10.0)
        sp_gate = logits.sigmoid()
        return p2 + self.alpha * (sp_gate * ch_mod) * ev, sp_gate, ev, logits

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        ch_w, sp_w = prefix + "channel_gate.weight", prefix + "spatial_gate.weight"
        if ch_w in state_dict and sp_w in state_dict:
            state_dict[prefix + "gate_proj.weight"] = torch.cat(
                (state_dict.pop(ch_w), state_dict.pop(sp_w)), dim=0
            )
            ch_b, sp_b = prefix + "channel_gate.bias", prefix + "spatial_gate.bias"
            if ch_b in state_dict and sp_b in state_dict:
                state_dict[prefix + "gate_proj.bias"] = torch.cat(
                    (state_dict.pop(ch_b), state_dict.pop(sp_b)), dim=0
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class DensityRouter(nn.Module):
    def __init__(
        self, c2: int = 32, c3: int = 64, c4: int = 128, hidden: int = 32, beta: float = 0.12
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.project3 = nn.Sequential(ConvNormAct(c3, c2, 1), ResBottleneck2d(c2))
        self.project4 = nn.Sequential(ConvNormAct(c4, hidden, 1), ResBottleneck2d(hidden))
        b_ch = max(16, hidden // 2)
        self.stem = ConvNormAct(c2 + hidden, hidden, 1)
        self.branch_d1 = ConvNormAct(hidden, b_ch, 3, padding=1, dilation=1, groups=b_ch)
        self.branch_d2 = ConvNormAct(hidden, b_ch, 3, padding=2, dilation=2, groups=b_ch)
        self.branch_d3 = ConvNormAct(hidden, b_ch, 3, padding=3, dilation=3, groups=b_ch)
        self.density_neck = nn.Sequential(ConvNormAct(b_ch * 3, hidden, 1), ResBottleneck2d(hidden))
        self.density_head = nn.Conv2d(hidden, 1, 1)
        nn.init.normal_(self.density_head.weight, 0, 0.01)
        nn.init.constant_(self.density_head.bias, -3.0)
        self.crowd_film = nn.Sequential(
            ResBottleneck2d(c2 + hidden), ConvNormAct(c2 + hidden, c2 * 2, 1, act=False)
        )
        nn.init.normal_(self.crowd_film[-1][0].weight, 0, 0.01)

    def forward(self, p2: Tensor, p3: Tensor, p4: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        sz = p2.shape[-2:]
        c3 = F.interpolate(self.project3(p3), size=sz, mode="bilinear", align_corners=False)
        c4 = F.interpolate(self.project4(p4), size=sz, mode="bilinear", align_corners=False)
        f = self.stem(torch.cat((c3, c4), dim=1))
        neck = self.density_neck(
            torch.cat((self.branch_d1(f), self.branch_d2(f), self.branch_d3(f)), dim=1)
        )
        logits = self.density_head(neck).clamp(-10.0, 10.0)
        density = logits.sigmoid()
        gamma, shift = self.crowd_film(torch.cat((c3 - p2, neck), dim=1)).chunk(2, dim=1)
        mod = p2 * (1.0 + self.beta * density * gamma.tanh()) + self.beta * density * (
            (c3 - p2) + 0.15 * shift.tanh()
        )
        return mod, density, c3, logits


_SCALE_ALIGN_CACHE: dict[
    tuple[int, int, torch.device, torch.dtype], tuple[Tensor, Tensor, Tensor, Tensor]
] = {}


class ScaleAligner(nn.Module):
    def __init__(self, channels: int = 32, points: int = 9, max_offset_cells: float = 0.75) -> None:
        super().__init__()
        self.points, self.max_offset_cells = int(points), float(max_offset_cells)
        self.scale_net = nn.Sequential(
            ResBottleneck2d(channels), ResBottleneck2d(channels, dilation=2)
        )
        self.scale_head = ConvNormAct(channels, 1, 1, act=False)
        self.offset_mask = nn.Sequential(
            ConvNormAct(channels + 1, channels, 1),
            ResBottleneck2d(channels),
            nn.Conv2d(channels, 3 * points, 1),
        )
        nn.init.normal_(self.offset_mask[-1].weight, 0, 0.01)
        nn.init.zeros_(self.offset_mask[-1].bias)
        self.post_align = nn.Sequential(ResBottleneck2d(channels), ResBottleneck2d(channels))
        self.channel_blend = nn.Parameter(torch.full((1, channels, 1, 1), 0.10))

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        b, c, h, w = feature.shape
        sf = self.scale_net(feature)
        scale_map = 0.5 + 2.0 * self.scale_head(sf).clamp(-10.0, 10.0).sigmoid()
        raw_om = self.offset_mask(torch.cat((sf, scale_map), dim=1))
        pred_off = (
            raw_om[:, : 2 * self.points].view(b, self.points, 2, h, w).permute(0, 3, 4, 1, 2).tanh()
            * self.max_offset_cells
        )
        pred_msk = (
            raw_om[:, 2 * self.points :]
            .view(b, self.points, h, w)
            .clamp(-10.0, 10.0)
            .permute(0, 2, 3, 1)
            .sigmoid()
        )

        cache_key = (w, h, feature.device, feature.dtype)
        cached = _SCALE_ALIGN_CACHE.get(cache_key)
        if cached is None:
            axis = torch.linspace(-1.0, 1.0, 3, device=feature.device, dtype=feature.dtype)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            base = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
            scale_factor = torch.tensor(
                (2.0 / max(w, 1), 2.0 / max(h, 1)), device=feature.device, dtype=feature.dtype
            )
            gx, gy = torch.meshgrid(
                (torch.arange(w, device=feature.device, dtype=feature.dtype) + 0.5) * 2.0 / max(w, 1)
                - 1.0,
                (torch.arange(h, device=feature.device, dtype=feature.dtype) + 0.5) * 2.0 / max(h, 1)
                - 1.0,
                indexing="xy",
            )
            grid_base = torch.stack((gx, gy), dim=-1)[None, :, :, None, :]
            base_gauss = torch.exp(-0.5 * base.square().sum(-1)).view(1, 1, 1, self.points)
            cached = (base, scale_factor, grid_base, base_gauss)
            _SCALE_ALIGN_CACHE[cache_key] = cached

        base, scale_factor, grid_base, base_gauss = cached
        offsets = (
            base[None, None, None] * (scale_map.permute(0, 2, 3, 1)[..., None] * 0.5) + pred_off
        ) * scale_factor
        grid = (grid_base + offsets).clamp(-1.0, 1.0)
        values = F.grid_sample(
            feature,
            grid.reshape(b, h, w * self.points, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        values = values.view(b, c, h, w, self.points)
        dyn_weights = pred_msk * base_gauss
        aligned = self.post_align(
            (
                values
                * (dyn_weights / dyn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-5)).unsqueeze(1)
            ).sum(-1)
        )
        return feature + self.channel_blend.tanh() * (aligned - feature), scale_map


class PRISMSpectralStage(nn.Module):
    def __init__(
        self, c2: int = 32, c3: int = 64, c4: int = 128, hidden: int = 32, alpha: float = 0.12
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.proj3 = nn.Sequential(ConvNormAct(c3, hidden, 1), ResBottleneck2d(hidden))
        self.proj4 = nn.Sequential(ConvNormAct(c4, hidden, 1), ResBottleneck2d(hidden))
        self.pool3 = nn.AvgPool2d(3, 1, 1)
        self.pool5 = nn.AvgPool2d(5, 1, 2)
        self.spectral_encoder = nn.Sequential(
            ConvNormAct(c2 * 2, c2, 1),
            ResBottleneck2d(c2, expansion=2.0),
            ResBottleneck2d(c2, expansion=1.5),
        )
        self.spectral_neck = nn.Sequential(
            ConvNormAct(c2 * 2 + 2 * hidden, hidden, 1), ResBottleneck2d(hidden)
        )
        self.c2 = c2
        self.energy_gate_proj = nn.Conv2d(hidden, c2 + 1, 1)
        nn.init.normal_(self.energy_gate_proj.weight[:c2], 0, 0.01)
        nn.init.zeros_(self.energy_gate_proj.bias[:c2])
        nn.init.normal_(self.energy_gate_proj.weight[c2:], 0, 0.01)
        nn.init.constant_(self.energy_gate_proj.bias[c2:], -2.0)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        cfg_w = prefix + "channel_freq_gate.weight"
        eh_w = prefix + "energy_head.weight"
        fg_w = prefix + "energy_gate_proj.weight"
        if cfg_w in state_dict and eh_w in state_dict and fg_w not in state_dict:
            state_dict[fg_w] = torch.cat((state_dict.pop(cfg_w), state_dict.pop(eh_w)), dim=0)
            cfg_b = prefix + "channel_freq_gate.bias"
            eh_b = prefix + "energy_head.bias"
            fg_b = prefix + "energy_gate_proj.bias"
            if cfg_b in state_dict and eh_b in state_dict and fg_b not in state_dict:
                state_dict[fg_b] = torch.cat((state_dict.pop(cfg_b), state_dict.pop(eh_b)), dim=0)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, p2: Tensor, p3: Tensor, p4: Tensor) -> dict[str, Tensor]:
        sz = p2.shape[-2:]
        c3 = F.interpolate(self.proj3(p3), size=sz, mode="bilinear", align_corners=False)
        c4 = F.interpolate(self.proj4(p4), size=sz, mode="bilinear", align_corners=False)
        p2_s3 = self.pool3(p2)
        f_high = p2 - p2_s3
        f_mid = p2_s3 - self.pool5(p2)
        spec = self.spectral_encoder(torch.cat((f_mid, f_high), dim=1))
        neck = self.spectral_neck(torch.cat((p2, spec, c3, c4), dim=1))
        fused_gate = self.energy_gate_proj(neck).clamp(-10.0, 10.0)
        ch_gate = fused_gate[:, : self.c2].sigmoid()
        energy_logits = fused_gate[:, self.c2 :]
        energy = energy_logits.sigmoid()
        return {
            "p2": p2 + self.alpha * (ch_gate * energy) * spec,
            "spec_energy": energy,
            "spec_logits": energy_logits,
            "gate": energy,
            "gate_logits": energy_logits,
            "density": energy,
            "density_logits": energy_logits,
            "scale_map": 0.5 + 2.0 * energy,
            "spectral_feat": spec,
        }


PRISMFeatureStage = PRISMSpectralStage


# ============================================================================
# 4. Downstream Heads
# ============================================================================


class CandidateRefiner(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        sigma_dim: int = 1,
        num_bins: int = 7,
        distributional: bool = False,
    ) -> None:
        super().__init__()
        self.num_bins, self.is_dist, self.sigma_dim = (
            int(num_bins),
            bool(distributional),
            int(sigma_dim),
        )
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden, eps=1e-5), nn.SiLU(inplace=True)
        )
        self.stage1 = GatedResMLPBlock(hidden)
        self.coarse_head = nn.Linear(hidden, 4)
        nn.init.normal_(self.coarse_head.weight, 0, 0.01)
        nn.init.zeros_(self.coarse_head.bias)

        self.stage2 = nn.Sequential(
            nn.Linear(hidden + 4, hidden), GatedResMLPBlock(hidden), GatedResMLPBlock(hidden)
        )
        if self.is_dist:
            self.dist_head = nn.Linear(hidden, 4 * num_bins + sigma_dim)
            nn.init.normal_(self.dist_head.weight, 0, 0.01)
            nn.init.zeros_(self.dist_head.bias)
            self.register_buffer(
                "bin_values", torch.linspace(-0.20, 0.20, num_bins), persistent=False
            )
        else:
            self.fine_head = nn.Linear(hidden, 4 + sigma_dim)
            nn.init.normal_(self.fine_head.weight, 0, 0.01)
            nn.init.zeros_(self.fine_head.bias)
        self._last_dist_stats: Tensor | None = None

    def forward(self, f: Tensor) -> tuple[Tensor, Tensor]:
        param_dtype = self.coarse_head.weight.dtype
        f_in = f.to(param_dtype)
        h1 = self.stage1(self.stem(f_in))
        coarse = torch.tanh(self.coarse_head(h1)) * 0.12
        h2 = self.stage2(torch.cat((h1, coarse), dim=-1))

        if not self.is_dist:
            out = self.fine_head(h2)
            self._last_dist_stats = None
            return (coarse + torch.tanh(out[..., :4]) * 0.15).clamp(-0.25, 0.25).to(f.dtype), out[
                ..., 4:
            ].clamp(-3.0, 2.0).to(f.dtype)

        raw = self.dist_head(h2)
        d_log = (
            raw[..., : 4 * self.num_bins].view(*f_in.shape[:-1], 4, self.num_bins).clamp(-10.0, 10.0)
        )
        probs = F.softmax(d_log, dim=-1)
        bins = self.bin_values.to(dtype=probs.dtype, device=probs.device)
        fine = (probs * bins).sum(-1)
        var = (probs * (bins - fine.unsqueeze(-1)).square()).sum(-1).clamp_min(1e-4)
        ent = -(probs * probs.clamp_min(1e-5).log()).sum(-1)
        self._last_dist_stats = torch.cat(
            (probs.flatten(-2), ent, probs.topk(2, dim=-1).values.sum(-1)), dim=-1
        ).to(f.dtype)
        return (coarse + fine).clamp(-0.25, 0.25).to(f.dtype), (
            raw[..., 4 * self.num_bins :] + 0.5 * var.log()
        ).clamp(-3.0, 2.0).to(f.dtype)


class QualityHead(nn.Module):
    def __init__(
        self, in_dim: int, hidden: int = 64, sigma_dim: int = 1, dist_stat_dim: int = 0
    ) -> None:
        super().__init__()
        self.sd, self.dsd = int(sigma_dim), int(dist_stat_dim)
        self.roi_stem = nn.Sequential(
            nn.Linear(in_dim - sigma_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.SiLU(),
            GatedResMLPBlock(hidden),
        )
        self.unc_gate = nn.Sequential(
            nn.Linear(sigma_dim + 5 + dist_stat_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.SiLU(),
            GatedResMLPBlock(hidden),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            GatedResMLPBlock(hidden), nn.LayerNorm(hidden, eps=1e-5), nn.Linear(hidden, 1)
        )
        nn.init.normal_(self.head[-1].weight, 0, 0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, f: Tensor, dist_stats: Tensor | None = None) -> Tensor:
        param_dtype = self.head[-1].weight.dtype
        f_in = f.to(param_dtype)
        dist_stats_in = dist_stats.to(param_dtype) if dist_stats is not None else None
        roi, log_s, geom = f_in[..., : -self.sd], f_in[..., -self.sd :], f_in[..., -self.sd - 5 : -self.sd]
        parts = [log_s, geom]
        if self.dsd > 0:
            parts.append(
                dist_stats_in if dist_stats_in is not None else f_in.new_zeros(*f_in.shape[:-1], self.dsd)
            )
        out = self.head(self.roi_stem(roi) * self.unc_gate(torch.cat(parts, dim=-1))).clamp(
            -10.0, 10.0
        )
        return out.to(f.dtype)


class AssignerMLP(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.SiLU(),
            GatedResMLPBlock(hidden),
        )
        self.norm = nn.LayerNorm(hidden, eps=1e-5)
        self.out = nn.Sequential(GatedResMLPBlock(hidden), nn.Linear(hidden, 1))
        nn.init.normal_(self.out[-1].weight, 0, 0.01)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, f: Tensor) -> Tensor:
        param_dtype = self.out[-1].weight.dtype
        f_in = f.to(param_dtype)
        h = self.net(f_in)
        if h.ndim == 3 and h.shape[1] > 1:
            h = h + 0.25 * self.norm(h - h.mean(1, keepdim=True))
        return self.out(h).squeeze(-1).clamp(-10.0, 10.0).to(f.dtype)


class DuplicateScorer(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.SiLU(),
            GatedResMLPBlock(hidden),
            GatedResMLPBlock(hidden),
            nn.LayerNorm(hidden, eps=1e-5),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.net[-1].weight, 0, 0.01)
        nn.init.constant_(self.net[-1].bias, -3.0)

    def forward(self, f: Tensor) -> Tensor:
        param_dtype = self.net[-1].weight.dtype
        return self.net(f.to(param_dtype)).squeeze(-1).clamp(-10.0, 10.0).to(f.dtype)


# ============================================================================
# 5. Master Detect Head & Criterion (Deepcopy, DDP & AMP Safe)
# ============================================================================


class CompositeDetect(Detect):
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
            raise ValueError(f"mode must be in {COMPOSITE_MODES}")
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.composite_mode, self.topk, self.aux_weight = mode, int(topk), float(aux_weight)

        self._last_aux_state: dict[str, Tensor] = {}
        self._last_aux: dict[str, Tensor] = self._last_aux_state
        self.ablation_flags: dict[str, bool] = {"m1": True, "m2": True, "m3": True}

        c2, c3, c4 = ch[0], ch[1], ch[2]
        self.roi_proj_p2 = FiveRegionROIExtractor(c2)
        self.roi_proj_p3 = FiveRegionROIExtractor(c3)

        if mode == "cglr":
            self.stage = ContextEvidenceGate(c2, c3, c4, hidden=max(32, c2))
            in_d = c2 + c3 + 1 + 5
            self.refiner = CandidateRefiner(in_d, hidden, sigma_dim=1, distributional=False)
            self.quality = QualityHead(in_d + 1, 64, 1, 0)
        elif mode == "cass":
            self.stage = DensityRouter(c2, c3, c4, hidden=max(32, c2))
            self.assign = AssignerMLP(6, 48)
            self.duplicate = DuplicateScorer(5, 48)
        elif mode == "sure":
            self.stage = ScaleAligner(c2)
            in_d = c2 + c3 + 1 + 5
            self.refiner = CandidateRefiner(in_d, hidden, 4, 7, True)
            self.quality = QualityHead(in_d + 4, 64, 4, 4 * 7 + 8)
        else:
            self.stage = PRISMSpectralStage(c2, c3, c4, hidden=max(32, c2))
            in_d = c2 + c3 + 1 + 5
            self.refiner = CandidateRefiner(in_d, hidden, 4, 7, False)
            self.quality = QualityHead(in_d + 4, 64, 4, 0)

    @classmethod
    def from_detect(
        cls,
        head: Detect,
        mode: str = "prism",
        topk: int = 300,
        hidden: int = 128,
        aux_weight: float = 0.5,
    ) -> CompositeDetect:
        channels = [
            int(b[0].conv.in_channels) if hasattr(b[0], "conv") else int(b[0].in_channels)
            for b in head.cv2
        ]
        reg = getattr(head, "reg_max", 16)
        e2e = getattr(head, "end2end", getattr(head, "_end2end", hasattr(head, "one2one_cv2")))
        res = cls(head.nc, reg, e2e, channels, mode, topk, hidden, aux_weight)
        res._end2end = bool(getattr(head, "_end2end", e2e))
        res.load_state_dict(head.state_dict(), strict=False)
        for attr in ("i", "f", "type", "np", "inplace", "stride", "anchors", "strides", "shape"):
            if hasattr(head, attr):
                val = getattr(head, attr)
                setattr(res, attr, val.clone() if isinstance(val, Tensor) else val)
        return res

    def set_ablation(self, m1: bool = True, m2: bool = True, m3: bool = True) -> None:
        self.ablation_flags = {"m1": bool(m1), "m2": bool(m2), "m3": bool(m3)}

    def _prepare_features(self, features: list[Tensor]) -> list[Tensor]:
        p2, p3, p4 = features[:3]
        if self.training or self.ablation_flags.get("m1", True):
            if self.composite_mode == "cglr":
                p2_mod, gate, ev, logit = self.stage(p2, p3, p4)
                self._last_aux_state = {"gate": gate, "gate_logits": logit, "evidence": ev}
            elif self.composite_mode == "cass":
                p2_mod, dens, ctx, logit = self.stage(p2, p3, p4)
                self._last_aux_state = {"density": dens, "density_logits": logit, "context": ctx}
            elif self.composite_mode == "sure":
                p2_mod, scale = self.stage(p2)
                self._last_aux_state = {"scale_map": scale}
            else:
                out = self.stage(p2, p3, p4)
                p2_mod = out["p2"]
                self._last_aux_state = {k: v for k, v in out.items() if k != "p2"}
            self._last_aux = self._last_aux_state
            return [p2_mod, *features[1:]]
        return features

    def forward(self, x: list[Tensor]):
        prepared = self._prepare_features(x)
        out = super().forward(prepared)
        if self.training and isinstance(out, dict):
            out["composite"] = {
                "mode": self.composite_mode,
                "feats": prepared,
                **self._last_aux_state,
            }
        return out

    def _decode(self, prediction: dict[str, Tensor]) -> Tensor:
        anchors, strides = make_anchors(prediction["feats"], self.stride, 0.5)
        anchors = anchors.transpose(0, 1).unsqueeze(0)
        strides = strides.transpose(0, 1).unsqueeze(0)
        if hasattr(self, "dfl"):
            distances = self.dfl(prediction["boxes"].float())
        else:
            distances = (
                prediction["boxes"]
                .float()
                .view(prediction["boxes"].shape[0], 4, self.reg_max, -1)
                .mean(2)
            )
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
        c0 = features[0].shape[1]

        # Fused Grid Sample: Sample features[0] (c0 channels) and auxiliary (1 channel) in ONE kernel
        if auxiliary is not None:
            fused_p2_aux = torch.cat((features[0].float(), auxiliary.float()), dim=1)
            sampled = _sample_roi_grid(fused_p2_aux, boxes, stride0, points=3)
            f_p2_grid = sampled[:, :, :c0, :]
            f_aux = sampled[:, :, c0:, :].mean(dim=-1)
        else:
            f_p2_grid = _sample_roi_grid(features[0].float(), boxes, stride0, points=3)
            f_aux = boxes.new_zeros(boxes.shape[0], boxes.shape[1], 1)

        f_p2 = self.roi_proj_p2(f_p2_grid)
        f_p3 = self.roi_proj_p3(_sample_roi_grid(features[1].float(), boxes, stride1, points=3))

        geom = _cxcywh_norm(boxes, float(width), float(height))
        size = (geom[..., 2] * geom[..., 3]).clamp_min(1e-6).sqrt().unsqueeze(-1)

        return torch.cat([f_p2, f_p3, f_aux, geom, size], dim=-1)

    def _inference(self, x: dict[str, Tensor]) -> Tensor:
        raw = super()._inference(x)
        device_type = "cuda" if raw.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            rf = raw.float()
            ff = [f.float() for f in x["feats"]]
            if self.composite_mode == "cass":
                out = self._apply_duplicate_penalty(rf, ff)
            else:
                out = self._apply_refinement(rf, ff)
        return out.to(raw.dtype)

    def _apply_refinement(self, raw: Tensor, features: list[Tensor]) -> Tensor:
        if not (self.ablation_flags.get("m2", True) or self.ablation_flags.get("m3", True)):
            return raw
        scr = raw[:, 4:].amax(1)
        idx = scr.topk(min(self.topk, scr.shape[1]), 1).indices
        box = raw[:, :4].gather(2, idx[:, None, :].expand(-1, 4, -1)).transpose(1, 2)

        aux_key = (
            "gate"
            if self.composite_mode == "cglr"
            else ("scale_map" if self.composite_mode == "sure" else "spec_energy")
        )
        s0 = float(self.stride[0])
        h, w = float(features[0].shape[-2] * s0), float(features[0].shape[-1] * s0)

        auxiliary = self._last_aux_state.get(aux_key)
        roi = self._roi_features(features, box, auxiliary)

        delta, log_sig = self.refiner(roi)
        refined = _apply_delta(box, delta, w, h) if self.ablation_flags.get("m2", True) else box
        qual = (
            self.quality(torch.cat((roi, log_sig), -1), self.refiner._last_dist_stats).sigmoid()
            if self.ablation_flags.get("m3", True)
            else torch.ones_like(scr[:, : refined.shape[1], None])
        )

        res = raw.clone()
        new_vals = torch.cat(
            (
                refined.transpose(1, 2),
                res[:, 4:].gather(2, idx[:, None, :].expand(-1, self.nc, -1)) * qual.transpose(1, 2),
            ),
            dim=1,
        )
        res.scatter_(2, idx[:, None, :].expand(-1, 4 + self.nc, -1), new_vals)
        return res

    def _apply_duplicate_penalty(self, raw: Tensor, features: list[Tensor]) -> Tensor:
        if not self.ablation_flags.get("m3", True):
            return raw
        scr = raw[:, 4:].amax(1)
        idx = scr.topk(min(self.topk, scr.shape[1]), 1).indices
        box = raw[:, :4].gather(2, idx[:, None, :].expand(-1, 4, -1)).transpose(1, 2)
        s0 = float(self.stride[0])
        _, c_p2 = self.roi_proj_p2(_sample_roi_grid(features[0], box, s0, 3), return_details=True)
        dens_map = self._last_aux_state.get("density")
        if dens_map is None:
            return raw
        den = _sample_roi(dens_map.float(), box, s0).squeeze(-1)
        h, w = float(features[0].shape[-2] * s0), float(features[0].shape[-1] * s0)

        box_f = box.float()
        left = torch.maximum(box_f[:, :, None, :2], box_f[:, None, :, :2])
        right = torch.minimum(box_f[:, :, None, 2:], box_f[:, None, :, 2:])
        intersection = (right - left).clamp_min(0.0).prod(-1)
        area = (box_f[:, :, 2] - box_f[:, :, 0]).clamp_min(1e-3) * (
            box_f[:, :, 3] - box_f[:, :, 1]
        ).clamp_min(1e-3)
        union = (area[:, :, None] + area[:, None, :] - intersection).clamp_min(1e-4)
        iou = (intersection / union).clamp(0.0, 1.0)

        cen = (box_f[:, :, :2] + box_f[:, :, 2:]) * 0.5
        dst = ((cen[:, :, None, :] - cen[:, None, :, :]).square().sum(dim=-1) + 1e-6).sqrt() / (
            w + h + 1e-4
        )
        are = (box_f[:, :, 2] - box_f[:, :, 0]).clamp_min(1e-2) * (
            box_f[:, :, 3] - box_f[:, :, 1]
        ).clamp_min(1e-2)
        rat = (are[:, :, None] / are[:, None, :]).clamp(0.1, 10.0)

        n_p = F.normalize(c_p2, dim=-1, eps=1e-5)
        sim = (n_p @ n_p.transpose(-1, -2)).clamp(-1.0, 1.0)
        den_pair = (den[:, :, None] + den[:, None, :]) * 0.5
        pair = torch.stack((iou, dst, rat, sim, den_pair), -1)

        sel_scr = scr.gather(1, idx)
        mask = (iou > 0.3).float() * (sel_scr[:, :, None] < sel_scr[:, None, :]).float()
        mask.diagonal(dim1=-2, dim2=-1).zero_()

        pen = (self.duplicate(pair).sigmoid() * mask).max(-1).values

        res = raw.clone()
        res[:, 4:].scatter_(
            2,
            idx[:, None, :].expand(-1, self.nc, -1),
            res[:, 4:].gather(2, idx[:, None, :].expand(-1, self.nc, -1))
            * (1.0 - 0.75 * pen[..., None]).transpose(1, 2),
        )
        return res

    def _zero_aux(self, device: torch.device) -> Tensor:
        values = [parameter.sum() * 0.0 for parameter in self.parameters()]
        return sum(values) if values else torch.zeros((), device=device, requires_grad=True)

    def compute_aux_loss(
        self, preds: Any, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        device = next(self.parameters()).device
        if not self._last_aux_state:
            return self._zero_aux(device), {}

        branch = (
            preds.get("one2many", preds)
            if isinstance(preds, dict)
            else (preds[1] if isinstance(preds, tuple) else preds)
        )
        feats = (
            branch["feats"]
            if isinstance(branch, dict) and "feats" in branch
            else (branch if isinstance(branch, list) else None)
        )

        if feats is None:
            return self._zero_aux(device), {}

        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            feats_f32 = [f.float() for f in feats]

            if isinstance(branch, dict) and "boxes" in branch and "scores" in branch:
                box_preds = branch["boxes"].float()
                cls_preds = branch["scores"].float()
            else:
                box_list, cls_list = [], []
                for xi in feats_f32:
                    b, _, _, _ = xi.shape
                    box_list.append(xi[:, : 4 * self.reg_max].view(b, 4 * self.reg_max, -1))
                    cls_list.append(
                        xi[:, 4 * self.reg_max : 4 * self.reg_max + self.nc].view(b, self.nc, -1)
                    )
                box_preds = torch.cat(box_list, dim=-1)
                cls_preds = torch.cat(cls_list, dim=-1)

            shape_key = tuple(f.shape[-2:] for f in feats_f32) + (feats_f32[0].device,)
            if not hasattr(self, "_anchor_cache"):
                self._anchor_cache = {}
            cached = self._anchor_cache.get(shape_key)
            if cached is None:
                anchors, strides = make_anchors(feats_f32, self.stride, 0.5)
                anchors = anchors.transpose(0, 1).unsqueeze(0).float()
                strides = strides.transpose(0, 1).unsqueeze(0).float()
                cached = (anchors, strides)
                self._anchor_cache[shape_key] = cached
            else:
                anchors, strides = cached

            if hasattr(self, "dfl"):
                distances = self.dfl(box_preds)
            else:
                distances = box_preds.view(box_preds.shape[0], 4, self.reg_max, -1).mean(2)

            decoded_boxes = dist2bbox(distances, anchors, xywh=False, dim=1)
            decoded_boxes = (decoded_boxes * strides).transpose(1, 2)

            scores = cls_preds.sigmoid().amax(dim=1)
            idx = scores.topk(min(self.topk, scores.shape[1]), dim=1).indices
            boxes = decoded_boxes.gather(1, idx[:, :, None].expand(-1, -1, 4)).detach()

            s0 = float(self.stride[0])
            h_img, w_img = float(feats_f32[0].shape[-2] * s0), float(feats_f32[0].shape[-1] * s0)

            gt_b, gt_l, gt_m = _padded_targets(batch, boxes.shape[0], w_img, h_img, boxes.device)
            if gt_b.shape[1] == 0 or not gt_m.any():
                return self._zero_aux(boxes.device), {}

            info_f32 = {
                k: v.float() if isinstance(v, Tensor) else v
                for k, v in self._last_aux_state.items()
            }
            info_f32["feats"] = feats_f32

            if self.composite_mode == "cglr":
                loss, items = self._aux_cglr(info_f32, boxes, gt_b, gt_m, h_img, w_img)
            elif self.composite_mode == "cass":
                loss, items = self._aux_cass(info_f32, boxes, gt_b, gt_l, gt_m, h_img, w_img)
            elif self.composite_mode == "sure":
                loss, items = self._aux_sure(info_f32, boxes, gt_b, gt_m, h_img, w_img)
            else:
                loss, items = self._aux_prism(info_f32, boxes, gt_b, gt_m, h_img, w_img)

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
        aux_key: str,
    ) -> tuple:
        aux = info.get(aux_key)
        roi = self._roi_features(info["feats"], boxes, aux)
        delta, log_sigma = self.refiner(roi)
        refined = _apply_delta(boxes, delta, width, height)
        best_iou, best_index, _ = self._match(boxes, gt_boxes, gt_mask)
        valid = (best_iou > 0.1).float()
        target = gt_boxes.gather(1, best_index[..., None].expand(-1, -1, 4))
        return roi, delta, log_sigma, refined, best_iou, target, valid

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
        b_f = boxes.float()
        g_f = gt_boxes.float()
        left = torch.maximum(b_f[:, :, None, :2], g_f[:, None, :, :2])
        right = torch.minimum(b_f[:, :, None, 2:], g_f[:, None, :, 2:])
        inter = (right - left).clamp_min(0.0).prod(-1)
        area_b = (b_f[:, :, 2] - b_f[:, :, 0]).clamp_min(1e-3) * (
            b_f[:, :, 3] - b_f[:, :, 1]
        ).clamp_min(1e-3)
        area_g = (g_f[:, :, 2] - g_f[:, :, 0]).clamp_min(1e-3) * (
            g_f[:, :, 3] - g_f[:, :, 1]
        ).clamp_min(1e-3)
        union = (area_b[:, :, None] + area_g[:, None, :] - inter).clamp_min(1e-4)
        ious = (inter / union).clamp(0.0, 1.0) * gt_mask[:, None, :].float()
        best_iou, best_index = ious.max(dim=-1)
        return best_iou, best_index, ious

    def _gaussian_focal_loss(
        self, logits: Tensor, gt_boxes: Tensor, gt_mask: Tensor, width: float, height: float
    ) -> Tensor:
        target = torch.zeros_like(logits)
        h_m, w_m = logits.shape[-2], logits.shape[-1]
        for image in range(gt_boxes.shape[0]):
            m = gt_mask[image]
            if not m.any():
                continue
            valid_boxes = gt_boxes[image][m]
            cx = ((valid_boxes[:, 0] + valid_boxes[:, 2]) * 0.5) / width * w_m - 0.5
            cy = ((valid_boxes[:, 1] + valid_boxes[:, 3]) * 0.5) / height * h_m - 0.5
            rads = (
                ((valid_boxes[:, 2] - valid_boxes[:, 0]) * (valid_boxes[:, 3] - valid_boxes[:, 1]))
                .clamp_min(1.0)
                .sqrt()
                / 4.0
            ).clamp_min(1.2)
            for k in range(valid_boxes.shape[0]):
                c_x, c_y, rad = float(cx[k]), float(cy[k]), float(rads[k])
                r = math.ceil(rad * 1.5)
                x0, x1 = max(0, int(c_x - r)), min(w_m, int(c_x + r + 1))
                y0, y1 = max(0, int(c_y - r)), min(h_m, int(c_y + r + 1))
                if x0 >= x1 or y0 >= y1:
                    continue
                yy, xx = torch.meshgrid(
                    torch.arange(y0, y1, device=logits.device, dtype=torch.float32),
                    torch.arange(x0, x1, device=logits.device, dtype=torch.float32),
                    indexing="ij",
                )
                dist = (xx - c_x).square() + (yy - c_y).square()
                kernel = torch.exp(-0.5 * dist / (rad * rad)).masked_fill(dist > (rad * 1.5) ** 2, 0.0)
                target[image, 0, y0:y1, x0:x1] = torch.maximum(target[image, 0, y0:y1, x0:x1], kernel)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = logits.sigmoid()
        return (
            bce * (1.0 - prob).square() * (target > 0.1).float()
            + bce * prob.square() * (target <= 0.1).float()
        ).mean()

    def _aux_cglr(
        self, info: dict[str, Tensor], boxes: Tensor, gt_b: Tensor, gt_m: Tensor, h: float, w: float
    ) -> tuple[Tensor, dict[str, Tensor]]:
        focal = self._gaussian_focal_loss(info["gate_logits"].clamp(-10.0, 10.0), gt_b, gt_m, w, h)
        roi, delta, log_sig, ref, best_iou, targ, valid = self._refine(
            info, boxes, gt_b, gt_m, h, w, "gate"
        )
        iou = _elementwise_iou(ref, targ)
        weight = (
            torch.where(
                best_iou >= 0.75, 0.25, torch.where((best_iou > 0.1) & (best_iou < 0.65), 1.5, 0.5)
            )
            * valid
        )
        norm = weight.sum().clamp_min(1.0)
        box_loss = ((1.0 - iou) * weight).sum() / norm
        t_delta = _target_delta(boxes, targ, w, h)
        res_loss = (
            F.smooth_l1_loss(delta, t_delta, reduction="none").mean(-1) * weight
        ).sum() / norm
        q_log = self.quality(torch.cat((roi, log_sig), -1))
        qual_loss = (
            F.binary_cross_entropy_with_logits(
                q_log.squeeze(-1), (iou.detach() * valid).clamp(0.0, 1.0), reduction="none"
            )
            * (valid + 0.1)
        ).sum() / norm
        sig_n = F.softplus(log_sig).clamp(0.25, 2.0)
        err = ((ref - targ).abs().mean(-1, keepdim=True) / float(self.stride[0])).clamp(0.0, 10.0)
        sig_loss = (
            (0.5 * (err / sig_n).square() + sig_n.log()) * valid[..., None]
        ).sum() / valid.sum().clamp_min(1.0)
        loss = (
            (focal + box_loss + res_loss + qual_loss + 0.1 * sig_loss)
            * boxes.shape[0]
            * self.aux_weight
        )
        return loss, {
            "gate_loss": focal.detach(),
            "refine_loss": (box_loss + res_loss).detach(),
            "quality_loss": qual_loss.detach(),
            "sigma_loss": sig_loss.detach(),
        }

    def _aux_cass(
        self,
        info: dict[str, Tensor],
        boxes: Tensor,
        gt_b: Tensor,
        gt_l: Tensor,
        gt_m: Tensor,
        h: float,
        w: float,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        logits = info["density_logits"].clamp(-10.0, 10.0)
        centers = torch.zeros_like(logits)
        h_m, w_m = logits.shape[-2], logits.shape[-1]
        for img in range(gt_b.shape[0]):
            m = gt_m[img]
            if not m.any():
                continue
            b = gt_b[img][m]
            cxs = (((b[:, 0] + b[:, 2]) * 0.5) / w * w_m).long().clamp(0, w_m - 1)
            cys = (((b[:, 1] + b[:, 3]) * 0.5) / h * h_m).long().clamp(0, h_m - 1)
            centers[img, 0].index_put_((cys, cxs), torch.ones_like(cxs, dtype=centers.dtype), accumulate=True)
        d_targ = F.interpolate(
            F.avg_pool2d(centers, 4, stride=4).mul(16.0).clamp(0.0, 4.0) / 4.0,
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        bce = F.binary_cross_entropy_with_logits(logits, d_targ, reduction="none")
        dens_loss = (
            bce * (1.0 - logits.sigmoid()).square() * d_targ
            + bce * logits.sigmoid().square() * (1.0 - d_targ)
        ).mean()

        assign_terms, dup_terms = [], []
        s0 = float(self.stride[0])
        _, c_p2 = self.roi_proj_p2(
            _sample_roi_grid(info["feats"][0], boxes, s0, 3), return_details=True
        )
        den_all = _sample_roi(info["density"], boxes, s0).squeeze(-1).clamp(0.0, 1.0)
        best_iou, best_index, _ = self._match(boxes, gt_b, gt_m)

        for img in range(boxes.shape[0]):
            valid_gt = gt_b[img][gt_m[img]]
            if valid_gt.shape[0] > 0:
                ious = _pair_iou(boxes[img], valid_gt)
                idx = ious.topk(min(16, boxes.shape[1]), 0).indices.T
                cand = boxes[img][idx]
                targ = valid_gt[:, None, :].expand_as(cand)
                c_iou = _elementwise_iou(cand, targ)
                cand_cen = (cand[..., :2] + cand[..., 2:]) * 0.5
                gt_cen = ((valid_gt[:, :2] + valid_gt[:, 2:]) * 0.5)[:, None, :]
                dist = (cand_cen - gt_cen).abs().mean(-1) / (w + h + 1e-4)
                are = (cand[..., 2] - cand[..., 0]).clamp_min(1e-2) * (
                    cand[..., 3] - cand[..., 1]
                ).clamp_min(1e-2)
                rat = (are / are.mean(-1, keepdim=True).clamp_min(1e-2)).clamp(0.1, 10.0)
                _, gt_c = self.roi_proj_p2(
                    _sample_roi_grid(info["feats"][0][img : img + 1], valid_gt[None], s0, 3),
                    return_details=True,
                )
                vis = F.cosine_similarity(
                    c_p2[img][idx], gt_c.squeeze(0)[:, None, :], dim=-1, eps=1e-5
                ).clamp(-1.0, 1.0)
                prior = (c_iou.clamp_min(1e-3).log() - dist.clamp(0.0, 2.0) + 0.5 * vis).clamp(
                    -6.0, 6.0
                )
                a_log = (
                    self.assign(
                        torch.stack(
                            (c_iou, dist.clamp(0.0, 2.0), rat, den_all[img][idx], vis, prior), -1
                        )
                    )
                    + prior
                ).clamp(-8.0, 8.0)
                log_r = a_log / 0.5
                for _ in range(3):
                    log_r = log_r - torch.logsumexp(log_r, 1, keepdim=True)
                    log_r = log_r - torch.logsumexp(log_r, 0, keepdim=True)
                resp = log_r.exp()
                resp = (resp / resp.sum(1, keepdim=True).clamp_min(1e-5)).detach()
                assign_terms.append(-(resp * F.log_softmax(a_log, 1)).sum(1).mean())

            iou_p = _pair_iou(boxes[img], boxes[img])
            pairs = torch.triu(iou_p > 0.3, 1).nonzero(as_tuple=False)
            if pairs.numel() > 0:
                l, r = pairs[:256, 0], pairs[:256, 1]
                t_pair = (
                    (best_index[img, l] == best_index[img, r])
                    & (best_iou[img, l] > 0.5)
                    & (best_iou[img, r] > 0.5)
                ).float()
                cen = (boxes[img, :, :2] + boxes[img, :, 2:]) * 0.5
                a_l = (boxes[img, l, 2] - boxes[img, l, 0]).clamp_min(1e-2) * (
                    boxes[img, l, 3] - boxes[img, l, 1]
                ).clamp_min(1e-2)
                a_r = (boxes[img, r, 2] - boxes[img, r, 0]).clamp_min(1e-2) * (
                    boxes[img, r, 3] - boxes[img, r, 1]
                ).clamp_min(1e-2)
                p_feat = torch.stack(
                    (
                        iou_p[l, r],
                        _safe_dist(cen[l], cen[r]) / (w + h + 1e-4),
                        (a_l / a_r).clamp(0.1, 10.0),
                        F.cosine_similarity(c_p2[img, l], c_p2[img, r], dim=-1, eps=1e-5).clamp(
                            -1.0, 1.0
                        ),
                        (den_all[img, l] + den_all[img, r]) * 0.5,
                    ),
                    -1,
                )
                dup_terms.append(F.binary_cross_entropy_with_logits(self.duplicate(p_feat), t_pair))

        assign_l = torch.stack(assign_terms).mean() if assign_terms else dens_loss.new_zeros(())
        dup_l = torch.stack(dup_terms).mean() if dup_terms else dens_loss.new_zeros(())
        loss = (dens_loss + assign_l + dup_l) * boxes.shape[0] * self.aux_weight
        return loss, {
            "density_loss": dens_loss.detach(),
            "assign_loss": assign_l.detach(),
            "duplicate_loss": dup_l.detach(),
        }

    def _aux_sure(
        self, info: dict[str, Tensor], boxes: Tensor, gt_b: Tensor, gt_m: Tensor, h: float, w: float
    ) -> tuple[Tensor, dict[str, Tensor]]:
        roi, delta, log_sig, ref, best_iou, targ, valid = self._refine(
            info, boxes, gt_b, gt_m, h, w, "scale_map"
        )
        iou = _elementwise_iou(ref, targ)
        weight = torch.where(best_iou >= 0.75, 0.25, torch.ones_like(best_iou)) * valid
        norm = weight.sum().clamp_min(1.0)
        t_delta = _target_delta(boxes, targ, w, h)
        d_loss = (F.smooth_l1_loss(delta, t_delta, reduction="none").mean(-1) * weight).sum() / norm
        b_loss = ((1.0 - iou) * weight).sum() / norm + d_loss
        p_ltrb = (
            torch.stack(
                (
                    ref[..., 0] - targ[..., 0],
                    ref[..., 1] - targ[..., 1],
                    targ[..., 2] - ref[..., 2],
                    targ[..., 3] - ref[..., 3],
                ),
                -1,
            ).abs()
            / float(self.stride[0])
        ).clamp(0.0, 10.0)
        s_norm = F.softplus(log_sig).clamp(0.25, 2.0)
        s_loss = ((0.5 * (p_ltrb / s_norm).square() + s_norm.log()) * valid[..., None]).sum() / (
            valid.sum() * 4.0
        ).clamp_min(1.0)
        q_log = self.quality(torch.cat((roi, log_sig), -1), self.refiner._last_dist_stats)
        q_loss = (
            F.binary_cross_entropy_with_logits(
                q_log.squeeze(-1), (iou.detach() * valid).clamp(0.0, 1.0), reduction="none"
            )
            * (valid + 0.1)
        ).sum() / norm
        loss = (b_loss + s_loss + q_loss) * boxes.shape[0] * self.aux_weight
        return loss, {
            "sure_box_loss": b_loss.detach(),
            "sure_sigma_loss": s_loss.detach(),
            "sure_quality_loss": q_loss.detach(),
        }

    def _aux_prism(
        self, info: dict[str, Tensor], boxes: Tensor, gt_b: Tensor, gt_m: Tensor, h: float, w: float
    ) -> tuple[Tensor, dict[str, Tensor]]:
        spec_loss = self._gaussian_focal_loss(
            info["spec_logits"].clamp(-10.0, 10.0), gt_b, gt_m, w, h
        )
        roi, delta, log_sig, ref, best_iou, targ, valid = self._refine(
            info, boxes, gt_b, gt_m, h, w, "spec_energy"
        )
        iou = _elementwise_iou(ref, targ)
        weight = (
            torch.where(
                best_iou >= 0.75, 0.25, torch.where((best_iou > 0.1) & (best_iou < 0.65), 1.5, 0.6)
            )
            * valid
        )
        norm = weight.sum().clamp_min(1.0)
        t_delta = _target_delta(boxes, targ, w, h)
        res_loss = (
            F.smooth_l1_loss(delta, t_delta, reduction="none").mean(-1) * weight
        ).sum() / norm
        b_loss = ((1.0 - iou) * weight).sum() / norm + res_loss
        p_ltrb = (
            torch.stack(
                (
                    ref[..., 0] - targ[..., 0],
                    ref[..., 1] - targ[..., 1],
                    targ[..., 2] - ref[..., 2],
                    targ[..., 3] - ref[..., 3],
                ),
                -1,
            ).abs()
            / float(self.stride[0])
        ).clamp(0.0, 10.0)
        s_norm = F.softplus(log_sig).clamp(0.25, 2.0)
        s_loss = ((0.5 * (p_ltrb / s_norm).square() + s_norm.log()) * valid[..., None]).sum() / (
            valid.sum() * 4.0
        ).clamp_min(1.0)
        q_log = self.quality(torch.cat((roi, log_sig), -1))
        q_loss = (
            F.binary_cross_entropy_with_logits(
                q_log.squeeze(-1), (iou.detach() * valid).clamp(0.0, 1.0), reduction="none"
            )
            * (valid + 0.1)
        ).sum() / norm
        loss = (spec_loss + b_loss + 0.15 * s_loss + q_loss) * boxes.shape[0] * self.aux_weight
        return loss, {
            "prism_spec_loss": spec_loss.detach(),
            "prism_box_loss": b_loss.detach(),
            "prism_sigma_loss": s_loss.detach(),
            "prism_qual_loss": q_loss.detach(),
        }


class CompositeCriterion:
    def __init__(self, model: nn.Module, base: Any) -> None:
        self.base = base
        self.head = model.model[-1] if hasattr(model, "model") else model
        self.updates = 0

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def __call__(self, preds: Any, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        loss, items = self.base(preds, batch)
        auxiliary, aux_items = self.head.compute_aux_loss(preds, batch)
        warmup = min(1.0, float(self.updates + 1) / 200.0)
        self.updates += 1
        return loss + warmup * auxiliary, {**items, **aux_items}

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
    "PRISMSpectralStage",
    "QualityHead",
    "ResBottleneck2d",
    "ScaleAligner",
    "SpatialGridProjector",
]
