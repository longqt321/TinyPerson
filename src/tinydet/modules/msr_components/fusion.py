"""Fusion operators for evidence maps from multiple scales."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tinydet.diagnostics.metrics import sampled_summary


class MultiScaleEvidenceFusion(nn.Module):
    """Fuse equal-shaped evidence maps using a fixed or learned scale rule."""

    MODES = ("mean", "max", "adaptive", "sk_softmax")

    def __init__(
        self, channels: int, num_scales: int = 3, reduction: int = 4,
        fusion_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        if type(num_scales) is not int or num_scales < 1:
            raise ValueError("num_scales must be a positive integer")
        if type(reduction) is not int or reduction < 1:
            raise ValueError("reduction must be a positive integer")
        if fusion_mode not in self.MODES:
            raise ValueError(f"fusion_mode must be one of {self.MODES}; got {fusion_mode!r}")
        self.channels = channels
        self.num_scales = num_scales
        self.reduction = reduction
        self.fusion_mode = fusion_mode
        self.scale_ids = tuple(range(num_scales))
        self.diagnostics_enabled = False
        self._last_diagnostics: dict[str, Tensor | float] = {}
        hidden = max(channels // reduction, 8)
        if fusion_mode == "adaptive":
            self.scale_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(channels, hidden), nn.SiLU(),
                    nn.Linear(hidden, channels), nn.Sigmoid(),
                )
                for _ in range(num_scales)
            ])
        elif fusion_mode == "sk_softmax":
            self.sk_reduce = nn.Linear(channels, hidden)
            self.sk_expand = nn.Linear(hidden, num_scales * channels)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scale maps, got {len(features)}")
        shape = features[0].shape
        if any(feature.ndim != 4 or feature.shape != shape for feature in features):
            raise ValueError("All scale maps must have the same BCHW shape")
        if shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {shape[1]}")
        if self.diagnostics_enabled:
            self._last_diagnostics = {}
        if self.fusion_mode == "mean":
            if self.diagnostics_enabled:
                self._record_fixed_weights()
            return sum(features) / self.num_scales
        if self.fusion_mode == "max":
            return torch.stack(features, dim=0).amax(dim=0)
        if self.fusion_mode == "adaptive":
            gates = [
                gate(F.adaptive_avg_pool2d(feature, 1).flatten(1))
                .view(-1, self.channels, 1, 1)
                for gate, feature in zip(self.scale_gates, features)
            ]
            denominator = torch.stack(gates, dim=0).sum(dim=0).clamp_min(1e-6)
            if self.diagnostics_enabled:
                weights = [gate / denominator for gate in gates]
                for scale_id, gate, weight in zip(self.scale_ids, gates, weights):
                    self._record_gate(scale_id, gate, weight)
                self._record_entropy(torch.stack(weights, dim=0))
            return sum(feature * gate for feature, gate in zip(features, gates)) / denominator
        summary = sum(features)
        pooled = F.adaptive_avg_pool2d(summary, 1).flatten(1)
        logits = self.sk_expand(F.silu(self.sk_reduce(pooled)))
        weights = logits.view(-1, self.num_scales, self.channels, 1, 1).softmax(dim=1)
        if self.diagnostics_enabled:
            for index, scale_id in enumerate(self.scale_ids):
                self._record_weight(scale_id, weights[:, index])
            self._record_entropy(weights)
        return sum(feature * weights[:, index] for index, feature in enumerate(features))

    def _record_gate(self, scale_id: int, gate: Tensor, weight: Tensor) -> None:
        for name, value in sampled_summary(gate, gate.numel(), thresholds=(0.05, 0.95)).items():
            if name in {"near_zero_frac", "near_one_frac"}:
                suffix = "saturation_low_frac" if name == "near_zero_frac" else "saturation_high_frac"
                self._last_diagnostics[f"fusion/scale_{scale_id}/gate_raw_{suffix}"] = value
            elif name in {"mean", "std", "min", "max", "finite_frac"}:
                self._last_diagnostics[f"fusion/scale_{scale_id}/gate_raw_{name}"] = value
        self._record_weight(scale_id, weight)

    def _record_weight(self, scale_id: int, weight: Tensor) -> None:
        for name, value in sampled_summary(weight, weight.numel()).items():
            if name in {"mean", "std", "min", "max", "finite_frac"}:
                self._last_diagnostics[f"fusion/scale_{scale_id}/weight_{name}"] = value

    def _record_fixed_weights(self) -> None:
        weight = 1.0 / self.num_scales
        for scale_id in self.scale_ids:
            for name, value in (("mean", weight), ("std", 0.0), ("min", weight), ("max", weight)):
                self._last_diagnostics[f"fusion/scale_{scale_id}/weight_{name}"] = value
        self._last_diagnostics["fusion/entropy"] = math.log(self.num_scales)
        self._last_diagnostics["fusion/normalized_entropy"] = 1.0 if self.num_scales > 1 else 0.0

    def _record_entropy(self, weights: Tensor) -> None:
        # weights: [K, B, C, 1, 1] or [B, K, C, 1, 1].
        if self.fusion_mode == "adaptive":
            by_scale = weights
            entropy = -(by_scale * by_scale.clamp_min(1e-8).log()).sum(dim=0)
        else:
            by_scale = weights.movedim(1, 0)
            entropy = -(by_scale * by_scale.clamp_min(1e-8).log()).sum(dim=0)
        mean_entropy = entropy.mean().detach()
        self._last_diagnostics["fusion/entropy"] = mean_entropy
        self._last_diagnostics["fusion/normalized_entropy"] = (
            mean_entropy / math.log(self.num_scales) if self.num_scales > 1 else mean_entropy * 0
        )

    def consume_diagnostics(self) -> dict[str, Tensor | float]:
        values = self._last_diagnostics
        self._last_diagnostics = {}
        return values
