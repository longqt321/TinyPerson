"""Multi-scale residual block and backward-compatible public API."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from tinydet.models.checkpoint import load_p2_relocated_weights
from tinydet.modules.msr_components import (
    EvidenceGuidedModulation,
    MultiScaleEvidenceFusion,
    ResidualRefinement,
    SingleScaleEvidence,
)


class MSR(nn.Module):
    """Coordinate multi-scale evidence extraction and residual refinement."""

    def __init__(
        self,
        channels: int,
        kernels: tuple[int, ...] | list[int] = (3, 5, 7),
        eps: float = 1e-6,
        sigma_min: float = 1e-3,
        z_max: float = 10.0,
        reduction: int = 4,
        residual_scale: float = 1e-3,
        fusion_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        if (not isinstance(kernels, Sequence) or isinstance(kernels, (str, bytes))
                or not kernels
                or any(type(k) is not int or k < 1 or k % 2 == 0 for k in kernels)):
            raise ValueError("kernels must be a nonempty sequence of positive odd integers")
        if len(set(kernels)) != len(kernels):
            raise ValueError("kernels must not contain duplicates")
        for name, value in (("eps", eps), ("sigma_min", sigma_min), ("z_max", z_max)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if type(reduction) is not int or reduction < 1:
            raise ValueError("reduction must be a positive integer")
        if fusion_mode not in MultiScaleEvidenceFusion.MODES:
            raise ValueError(
                f"fusion_mode must be one of {MultiScaleEvidenceFusion.MODES}; got {fusion_mode!r}"
            )
        if not isinstance(residual_scale, (int, float)) or not math.isfinite(residual_scale):
            raise ValueError("residual_scale must be finite")

        self.channels = channels
        self.kernels = tuple(kernels)
        self.eps = float(eps)
        self.sigma_min = float(sigma_min)
        self.z_max = float(z_max)
        # Preserve original initialization order: all projections, all scale gates,
        # feature gate, depthwise convolution, then scalar residual parameter.
        self.single_scales = nn.ModuleList([
            SingleScaleEvidence(channels, kernel, self.eps, self.sigma_min, self.z_max)
            for kernel in self.kernels
        ])
        self.fusion = MultiScaleEvidenceFusion(
            channels, len(self.kernels), reduction, fusion_mode
        )
        self.fusion.scale_ids = self.kernels
        self.modulation = EvidenceGuidedModulation(channels)
        self.refinement = ResidualRefinement(channels, residual_scale)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or not x.is_floating_point() or x.shape[1] != self.channels:
            raise ValueError(f"Expected floating BCHW input with C={self.channels}")
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
        evidence_by_scale = [scale(x, x32=x32) for scale in self.single_scales]
        evidence = self.fusion(evidence_by_scale)
        modulated = self.modulation(x, evidence)
        return self.refinement(x, modulated)

    @property
    def gamma(self) -> nn.Parameter:
        """Compatibility accessor for the learned residual scale."""
        return self.refinement.gamma

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ) -> None:
        """Load legacy flat MSR keys into the refactored component layout."""
        legacy_prefixes = {
            "projections.": "single_scales.",
            "scale_gates.": "fusion.scale_gates.",
            "feature_gate.": "modulation.gate.",
            "refine.": "refinement.refine.",
            "gamma": "refinement.gamma",
        }
        for key in list(state_dict):
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix):]
            for old, new in legacy_prefixes.items():
                if local_key.startswith(old):
                    suffix = local_key[len(old):]
                    if old == "projections.":
                        index, _, rest = suffix.partition(".")
                        suffix = f"{index}.projection.{rest}"
                    state_dict[prefix + new + suffix] = state_dict.pop(key)
                    break
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    def __setstate__(self, state) -> None:
        """Adapt full-object checkpoints saved with the former flat module layout."""
        super().__setstate__(state)
        if "single_scales" in self._modules:
            return
        modules = self._modules
        projections = modules.pop("projections", None)
        gates = modules.pop("scale_gates", None)
        feature_gate = modules.pop("feature_gate", None)
        refine = modules.pop("refine", None)
        gamma = self._parameters.pop("gamma", None)
        if any(value is None for value in (projections, gates, feature_gate, refine, gamma)):
            raise RuntimeError("Unsupported legacy MSR checkpoint layout")

        def component(cls, **attrs):
            value = cls.__new__(cls)
            nn.Module.__init__(value)
            for name, item in attrs.items():
                setattr(value, name, item)
            return value

        self.single_scales = nn.ModuleList([
            component(
                SingleScaleEvidence, channels=self.channels, kernel=kernel,
                eps=self.eps, sigma_min=self.sigma_min, z_max=self.z_max,
                projection=projection,
            )
            for kernel, projection in zip(self.kernels, projections)
        ])
        self.fusion = component(
            MultiScaleEvidenceFusion,
            channels=self.channels, num_scales=len(self.kernels),
            reduction=4,
            fusion_mode="adaptive", scale_gates=gates, scale_ids=self.kernels,
            diagnostics_enabled=False, _last_diagnostics={},
        )
        self.modulation = component(EvidenceGuidedModulation, channels=self.channels, gate=feature_gate)
        self.refinement = component(
            ResidualRefinement, channels=self.channels, refine=refine, gamma=gamma,
            diagnostics_enabled=False, sampled_elements=4096, _last_diagnostics={},
        )
        self.fusion.scale_ids = self.kernels


__all__ = [
    "MSR",
    "EvidenceGuidedModulation",
    "MultiScaleEvidenceFusion",
    "ResidualRefinement",
    "SingleScaleEvidence",
    "load_p2_relocated_weights",
]
