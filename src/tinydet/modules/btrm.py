"""Boundary-aware multi-scale trend removal for small-object features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class BTRM(nn.Module):
    """Encode local contrast and edges, starting as an exact identity."""

    def __init__(self, channels: int, radius: int = 1, reduction: int = 4) -> None:
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        if type(radius) is not int or radius < 1:
            raise ValueError("radius must be a positive integer")
        if type(reduction) is not int or reduction < 1:
            raise ValueError("reduction must be a positive integer")
        self.channels = channels
        self.radius = radius
        hidden = max(channels // reduction, 8)
        self.project = nn.Sequential(
            nn.Conv2d(5 * channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.gate = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.gate.bias)
        self.gamma = nn.Parameter(torch.zeros(()))
        kernel_x = torch.tensor([[[-0.5, 0.0, 0.5]]])
        kernel_y = torch.tensor([[[-0.5], [0.0], [0.5]]])
        self.register_buffer("kernel_x", kernel_x.repeat(channels, 1, 1, 1))
        self.register_buffer("kernel_y", kernel_y.repeat(channels, 1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or not x.is_floating_point() or x.shape[1] != self.channels:
            raise ValueError(f"Expected floating BCHW input with C={self.channels}")
        values = x
        near = F.avg_pool2d(
            F.pad(values, (self.radius,) * 4, mode="replicate"),
            2 * self.radius + 1,
            stride=1,
        )
        far_radius = 2 * self.radius
        far = F.avg_pool2d(
            F.pad(values, (far_radius,) * 4, mode="replicate"),
            2 * far_radius + 1,
            stride=1,
        )
        gradient_x = F.conv2d(
            F.pad(values, (1, 1, 0, 0), mode="replicate"),
            self.kernel_x.to(dtype=values.dtype),
            groups=self.channels,
        )
        gradient_y = F.conv2d(
            F.pad(values, (0, 0, 1, 1), mode="replicate"),
            self.kernel_y.to(dtype=values.dtype),
            groups=self.channels,
        )
        magnitude = gradient_x.abs() + gradient_y.abs()
        evidence = torch.cat(
            (values - near, values - far, gradient_x, gradient_y, magnitude),
            dim=1,
        )
        projected = self.project(evidence)
        residual = self.gate(projected).sigmoid() * projected
        return x + self.gamma.to(dtype=x.dtype) * residual.to(dtype=x.dtype)


__all__ = ["BTRM"]
