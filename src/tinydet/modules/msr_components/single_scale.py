"""Single-scale local-statistics evidence extraction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SingleScaleEvidence(nn.Module):
    """Normalize local residuals and project positive/negative evidence."""

    def __init__(
        self, channels: int, kernel: int, eps: float = 1e-6,
        sigma_min: float = 1e-3, z_max: float = 10.0,
    ) -> None:
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        if type(kernel) is not int or kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be a positive odd integer")
        for name, value in (("eps", eps), ("sigma_min", sigma_min), ("z_max", z_max)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self.channels = channels
        self.kernel = kernel
        self.eps = float(eps)
        self.sigma_min = float(sigma_min)
        self.z_max = float(z_max)
        self.projection = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(
        self, x: Tensor, *, x32: Tensor | None = None, x32_sq: Tensor | None = None
    ) -> Tensor:
        if x.ndim != 4 or not x.is_floating_point() or x.shape[1] != self.channels:
            raise ValueError(f"Expected floating BCHW input with C={self.channels}")
        if x32 is not None and (x32.shape != x.shape or x32.dtype != torch.float32):
            raise ValueError("x32 must be an FP32 tensor with the same shape as x")
        with torch.autocast(device_type=x.device.type, enabled=False):
            values = x.float() if x32 is None else x32
            values_sq = values.square() if x32_sq is None else x32_sq
            pad = self.kernel // 2
            mean = F.avg_pool2d(
                values, self.kernel, stride=1, padding=pad, count_include_pad=False
            )
            second = F.avg_pool2d(
                values_sq, self.kernel, stride=1, padding=pad, count_include_pad=False
            )
            std = (second.sub(mean.square()).clamp_min(0).add(self.eps).sqrt()
                   .clamp_min(self.sigma_min))
            z = ((values - mean) / std).clamp(-self.z_max, self.z_max)
            signed = torch.cat((F.relu(z), F.relu(-z)), dim=1).to(x.dtype)
        return self.projection(signed)
