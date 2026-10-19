"""Multi-scale standardized residual modulation for tiny-object features.

The fixed ring statistics are computed in FP32 for AMP stability. The module
uses a scalar, bounded residual gain initialized to zero, so insertion is an
exact identity until training learns to use the response.
"""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MSR_GATE_INDEX = 20
P2_SOURCE_INDEX = 19
DETECT_INDEX = 30


class MultiScaleStandardizedResidual(nn.Module):
    """Dilated local outlier response with identity-preserving residual modulation.

    For each dilation d, estimate mean and variance from the eight surrounding
    positions (excluding the center), compute a positive residual, then max
    over dilations. A single bounded scalar gain keeps the high-resolution P2
    path inexpensive.
    """

    def __init__(
        self,
        mode: str = "standardized",
        dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        eps: float = 1e-4,
        max_gain: float = 0.1,
        score_clip: float = 10.0,
    ):
        super().__init__()
        if mode not in {"raw", "standardized"}:
            raise ValueError("mode must be 'raw' or 'standardized'")
        if not dilations or any(type(d) is not int or d < 1 for d in dilations):
            raise ValueError("dilations must be a nonempty sequence of positive integers")
        if len(set(dilations)) != len(dilations):
            raise ValueError("dilations must not contain duplicates")
        for name, value in (("eps", eps), ("max_gain", max_gain), ("score_clip", score_clip)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if max_gain > 0.5:
            raise ValueError("max_gain must not exceed 0.5")

        self.mode = mode
        self.dilations = tuple(dilations)
        self.eps = float(eps)
        self.max_gain = float(max_gain)
        self.score_clip = float(score_clip)
        self.gain_raw = nn.Parameter(torch.zeros(()))

        # Uniform 3x3 ring kernel: its center is deliberately excluded.
        kernel = torch.ones(1, 1, 3, 3)
        kernel[..., 1, 1] = 0.0
        kernel /= 8.0
        self.register_buffer("_neighbor_kernel", kernel, persistent=False)

    @property
    def effective_gain(self) -> Tensor:
        """A bounded scalar in (-max_gain, max_gain), exactly zero at init."""
        return self.max_gain * torch.tanh(self.gain_raw)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or not x.is_floating_point():
            raise ValueError("Expected a floating-point BCHW feature tensor")

        input_dtype = x.dtype
        # Variance from second moments and the reciprocal square root are more
        # stable in FP32 than in FP16/BF16 under mixed-precision training.
        with torch.autocast(device_type=x.device.type, enabled=False):
            feature = x.float()
            channels = feature.shape[1]
            kernel = self._neighbor_kernel.to(
                device=feature.device, dtype=feature.dtype
            ).expand(channels, 1, 3, 3)
            response = torch.zeros_like(feature)

            for dilation in self.dilations:
                padded = F.pad(
                    feature, (dilation, dilation, dilation, dilation), mode="replicate"
                )
                mean = F.conv2d(padded, kernel, dilation=dilation, groups=channels)
                second_moment = F.conv2d(
                    padded.square(), kernel, dilation=dilation, groups=channels
                )
                variance = (second_moment - mean.square()).clamp_min(0.0)
                residual = feature - mean
                if self.mode == "standardized":
                    residual = residual * torch.rsqrt(variance + self.eps)

                current = F.relu(residual).clamp_max(self.score_clip)
                response = torch.maximum(response, current)

            output = feature + self.effective_gain.float() * response

        return output.to(dtype=input_dtype)


def validate_feature_module(options: dict | None) -> dict:
    """Validate and normalize the experiment-level feature-module settings."""
    if options is None:
        return {"mode": "none"}
    if not isinstance(options, dict) or set(options) - {
        "mode", "dilations", "eps", "max_gain", "score_clip"
    }:
        raise ValueError("feature_module contains unknown options")
    mode = options.get("mode")
    if mode not in {"none", "raw", "standardized"}:
        raise ValueError("feature_module.mode must be none, raw or standardized")
    if mode == "none":
        if set(options) != {"mode"}:
            raise ValueError("feature_module mode 'none' does not take other options")
        return {"mode": "none"}

    dilations = options.get("dilations", [1, 2, 4])
    if not isinstance(dilations, (list, tuple)) or not dilations:
        raise ValueError("feature_module.dilations must be a nonempty list of integers")
    eps = options.get("eps", 1e-4)
    max_gain = options.get("max_gain", 0.1)
    score_clip = options.get("score_clip", 10.0)
    # Let the module perform the canonical numeric/range checks.
    MultiScaleStandardizedResidual(
        mode=mode, dilations=dilations, eps=eps,
        max_gain=max_gain, score_clip=score_clip,
    )
    return {
        "mode": mode,
        "dilations": list(dilations),
        "eps": float(eps),
        "max_gain": float(max_gain),
        "score_clip": float(score_clip),
    }


def install_msr_feature_module(model: nn.Module, options: dict | None) -> None:
    """Replace the explicit P2 Identity slot in the vetted MSR model YAML."""
    validated = validate_feature_module(options)
    layers = getattr(model, "model", None)
    if layers is None or len(layers) != DETECT_INDEX + 1:
        raise ValueError("MSR requires configs/model/yolo26n-msr-p2.yaml")
    gate = layers[MSR_GATE_INDEX]
    detect = layers[DETECT_INDEX]
    if not isinstance(gate, nn.Identity) or gate.f != -1 or gate.i != MSR_GATE_INDEX:
        raise ValueError("Expected the P2 Identity gate at layer 20")
    if layers[MSR_GATE_INDEX + 1].f != P2_SOURCE_INDEX or detect.f != [
        MSR_GATE_INDEX, 23, 26, 29
    ]:
        raise ValueError("P2/P3/P4/P5 routing differs from the vetted MSR architecture")

    if validated["mode"] != "none":
        replacement = MultiScaleStandardizedResidual(
            mode=validated["mode"],
            dilations=validated["dilations"],
            eps=validated["eps"],
            max_gain=validated["max_gain"],
            score_clip=validated["score_clip"],
        )
        # Ultralytics' graph interpreter relies on these parser-populated fields.
        for attr in ("i", "f"):
            setattr(replacement, attr, getattr(gate, attr))
        replacement.type = type(replacement).__name__
        replacement.np = sum(parameter.numel() for parameter in replacement.parameters())
        layers[MSR_GATE_INDEX] = replacement

    model.yaml["tinydet_feature_module"] = validated
