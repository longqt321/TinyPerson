"""Depthwise residual refinement."""

import math

import torch
from torch import Tensor, nn

from tinydet.diagnostics.metrics import sampled_summary


class ResidualRefinement(nn.Module):
    """Apply depthwise 3x3 refinement and scalar residual update."""

    def __init__(self, channels: int, residual_scale: float = 1e-3) -> None:
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        if not isinstance(residual_scale, (int, float)) or not math.isfinite(residual_scale):
            raise ValueError("residual_scale must be finite")
        self.channels = channels
        self.refine = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.gamma = nn.Parameter(torch.tensor(float(residual_scale)))
        self.diagnostics_enabled = False
        self.sampled_elements = 4096
        self._last_diagnostics: dict[str, Tensor] = {}

    def forward(self, x: Tensor, modulated: Tensor) -> Tensor:
        if (x.ndim != 4 or x.shape[1] != self.channels
                or modulated.shape != x.shape):
            raise ValueError("x and modulated must have the same BCHW shape and configured channels")
        residual = self.refine(modulated)
        if not self.diagnostics_enabled:
            return x + self.gamma.to(residual.dtype) * residual
        update = self.gamma.to(residual.dtype) * residual
        output = x + update
        for name, value in sampled_summary(residual, self.sampled_elements).items():
            if name in {"mean", "std", "rms", "finite_frac"}:
                self._last_diagnostics[f"residual/refinement_{name}"] = value
        input_rms = sampled_summary(x, self.sampled_elements)["rms"]
        update_stats = sampled_summary(update, self.sampled_elements)
        output_stats = sampled_summary(output, self.sampled_elements)
        self._last_diagnostics.update({
            "residual/gamma": self.gamma.detach(),
            "residual/update_rms": update_stats["rms"],
            "residual/input_rms": input_rms,
            "residual/output_rms": output_stats["rms"],
            "residual/update_input_ratio": update_stats["rms"] / (input_rms + 1e-12),
            "residual/output_finite_frac": output_stats["finite_frac"],
            "residual/gamma_finite": torch.isfinite(self.gamma.detach()).to(torch.float32),
        })
        return output

    def consume_diagnostics(self) -> dict[str, Tensor]:
        values = self._last_diagnostics
        self._last_diagnostics = {}
        return values
