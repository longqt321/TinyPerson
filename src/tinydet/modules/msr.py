"""Multi-scale standardized residual evidence and cross-level fusion.

The module combines three distinct operations:
1. standardized local evidence computed from ring statistics at several dilations;
2. fixed max or response-adaptive scale aggregation; and
3. evidence-guided feature fusion at the six YOLO26n-P2 neck concatenations.

No Ultralytics global registry is modified. The model YAML contains Identity slots,
which this installer replaces per model/training run.
"""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FUSION_SPECS = {
    # layer index: (high-level channels, low-level channels, high branch is first)
    # Explicit YOLO26n width scaling makes top-down P3/P2 concatenations equal-width.
    13: (256, 128, True),   # P5 -> P4 (top-down)
    17: (128, 128, True),   # P4 -> P3 (top-down)
    21: (64, 64, True),     # P3 -> P2 (top-down)
    25: (64, 32, False),    # P2 detail is first in Concat; P3 semantics is second
    29: (128, 64, False),   # P3 -> P4 (bottom-up)
    33: (256, 128, False),  # P4 -> P5 (bottom-up)
}
DETECT_INDEX = 35
DETECT_SOURCES = [22, 26, 30, 34]


class MultiScaleStandardizedResidual(nn.Module):
    """Estimate positive local outlier evidence at multiple dilation scales.

    Each channel is normalized by the local variance of the eight-neighbour
    ring, with the center excluded. Scale aggregation is either the ALCNet-like
    fixed maximum or a spatially adaptive softmax over each scale's evidence.
    The operation is parameter-free except for one learned bias per scale.
    """

    def __init__(
        self,
        dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        aggregation: str = "adaptive",
        eps: float = 1e-4,
        score_clip: float = 10.0,
        scale_temperature: float = 1.0,
    ):
        super().__init__()
        if not dilations or any(type(d) is not int or d < 1 for d in dilations):
            raise ValueError("dilations must be a nonempty sequence of positive integers")
        if len(set(dilations)) != len(dilations):
            raise ValueError("dilations must not contain duplicates")
        if aggregation not in {"max", "adaptive"}:
            raise ValueError("aggregation must be 'max' or 'adaptive'")
        for name, value in (
            ("eps", eps),
            ("score_clip", score_clip),
            ("scale_temperature", scale_temperature),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

        self.dilations = tuple(dilations)
        self.aggregation = aggregation
        self.eps = float(eps)
        self.score_clip = float(score_clip)
        self.scale_temperature = float(scale_temperature)
        if aggregation == "adaptive":
            self.scale_bias = nn.Parameter(torch.zeros(len(self.dilations)))
        else:
            self.register_parameter("scale_bias", None)

        ring = torch.ones(1, 1, 3, 3)
        ring[..., 1, 1] = 0
        ring /= 8.0
        self.register_buffer("_ring_kernel", ring, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or not x.is_floating_point():
            raise ValueError("Expected a floating-point BCHW feature tensor")

        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            feature = x.float()
            channels = feature.shape[1]
            kernel = self._ring_kernel.to(
                device=feature.device, dtype=feature.dtype
            ).expand(channels, 1, 3, 3)
            scale_responses = []

            for dilation in self.dilations:
                padded = F.pad(
                    feature,
                    (dilation, dilation, dilation, dilation),
                    mode="replicate",
                )
                local_mean = F.conv2d(
                    padded, kernel, dilation=dilation, groups=channels
                )
                local_second = F.conv2d(
                    padded.square(), kernel, dilation=dilation, groups=channels
                )
                local_variance = (local_second - local_mean.square()).clamp_min(0.0)
                standardized = (feature - local_mean) * torch.rsqrt(
                    local_variance + self.eps
                )
                scale_responses.append(F.relu(standardized).clamp_max(self.score_clip))

            stacked = torch.stack(scale_responses, dim=1)  # [B, D, C, H, W]
            if self.aggregation == "max":
                response = stacked.max(dim=1).values
            else:
                # Response-adaptive local scale pooling. The scale weights vary
                # spatially and are biased by trainable per-scale priors.
                scale_score = stacked.mean(dim=2) / self.scale_temperature
                scale_score = scale_score + self.scale_bias.view(1, -1, 1, 1)
                weights = torch.softmax(scale_score, dim=1)
                response = (stacked * weights.unsqueeze(2)).sum(dim=1)

        return response.to(dtype=input_dtype)


class MSRFusionGate(nn.Module):
    """MSR neck block for a concatenated high-/low-level feature pair.

    The local-only controls refine the low-level branch with MSR evidence.
    The full mode computes local evidence on the low-level branch, derives a
    point-wise attention map from both the evidence and original low-level
    features, then fuses the projected evidence with the gated high-level
    feature. The original low-level branch is retained in the concatenation.
    This is not a scalar-gain residual inserted into a single feature map.
    """

    def __init__(
        self,
        high_channels: int,
        low_channels: int,
        high_first: bool,
        mode: str,
        dilations: tuple[int, ...] | list[int] = (1, 2, 4),
        eps: float = 1e-4,
        score_clip: float = 10.0,
        hidden_ratio: int = 4,
        scale_temperature: float = 1.0,
    ):
        super().__init__()
        if mode not in {"local_max", "local_adaptive", "full_max", "full_adaptive"}:
            raise ValueError(f"Unsupported MSR fusion mode: {mode}")
        if high_channels not in {low_channels, 2 * low_channels}:
            raise ValueError("MSR neck fusion expects equal or 2:1 nano-scale channel widths")
        if type(hidden_ratio) is not int or hidden_ratio < 1:
            raise ValueError("hidden_ratio must be a positive integer")

        self.high_channels = high_channels
        self.low_channels = low_channels
        self.high_first = bool(high_first)
        self.mode = mode
        aggregation = "adaptive" if mode.endswith("adaptive") else "max"
        self.evidence = MultiScaleStandardizedResidual(
            dilations=dilations,
            aggregation=aggregation,
            eps=eps,
            score_clip=score_clip,
            scale_temperature=scale_temperature,
        )
        hidden = max(low_channels // hidden_ratio, 8)

        if mode.startswith("local_"):
            self.local_attention = nn.Sequential(
                nn.Conv2d(low_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, low_channels, 1, bias=False),
                nn.BatchNorm2d(low_channels),
                nn.Sigmoid(),
            )
            # Start with a small but nonzero effect so the attention network
            # receives gradients from the first optimization step.
            self.local_gain = nn.Parameter(torch.tensor(0.5))
            self.max_gain = 0.1
        else:
            self.local_attention = None
            self.register_parameter("local_gain", None)
            self.register_buffer("max_gain", torch.tensor(0.1), persistent=False)
            # Project the evidence into the high-level channel space.
            self.evidence_projection = nn.Sequential(
                nn.Conv2d(low_channels, high_channels, 1, bias=False),
                nn.BatchNorm2d(high_channels),
            )
            # ALCNet-inspired bottleneck local attention: low-level features and
            # MSR evidence guide element-wise modulation of the high-level map.
            self.attention = nn.Sequential(
                nn.Conv2d(low_channels * 2, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, high_channels, 1, bias=False),
                nn.BatchNorm2d(high_channels),
                nn.Sigmoid(),
            )
            # Near-identity initialization: the attention starts near one while
            # the evidence projection starts at zero, then both are learned.
            with torch.no_grad():
                self.evidence_projection[1].weight.zero_()
                self.evidence_projection[1].bias.zero_()
                self.attention[4].bias.fill_(3.0)

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 4 or x.shape[1] != self.high_channels + self.low_channels:
            raise ValueError(
                "Unexpected concatenated feature shape for MSR gate: "
                f"expected {self.high_channels + self.low_channels} channels"
            )
        if self.high_first:
            return x[:, :self.high_channels], x[:, self.high_channels:]
        return x[:, self.low_channels:], x[:, :self.low_channels]

    def forward(self, x: Tensor) -> Tensor:
        high, low = self._split(x)
        evidence = self.evidence(low)

        if self.mode.startswith("local_"):
            attention = self.local_attention(evidence)
            gain = self.max_gain * torch.tanh(self.local_gain)
            low_refined = low + gain.to(low.dtype) * attention * evidence
            fused_high = high
        else:
            attention = self.attention(torch.cat((low, evidence), dim=1))
            projected_evidence = self.evidence_projection(evidence)
            # Learned source selection: cross-layer evidence contributes the
            # local component and attention weights the semantic high-level map.
            fused_high = projected_evidence + attention * high
            low_refined = low

        if self.high_first:
            return torch.cat((fused_high, low_refined), dim=1)
        return torch.cat((low_refined, fused_high), dim=1)


def validate_feature_module(options: dict | None) -> dict:
    """Validate and normalize experiment-level MSR settings."""
    if options is None:
        return {"mode": "none"}
    if not isinstance(options, dict) or set(options) - {
        "mode", "dilations", "eps", "score_clip", "hidden_ratio", "scale_temperature"
    }:
        raise ValueError("feature_module contains unknown options")

    mode = options.get("mode")
    modes = {"none", "local_max", "local_adaptive", "full_max", "full_adaptive"}
    if mode not in modes:
        raise ValueError(f"feature_module.mode must be one of {sorted(modes)}")
    if mode == "none":
        if set(options) != {"mode"}:
            raise ValueError("feature_module mode 'none' does not take other options")
        return {"mode": "none"}

    dilations = options.get("dilations", [1, 2, 4])
    if not isinstance(dilations, (list, tuple)) or not dilations:
        raise ValueError("feature_module.dilations must be a nonempty list of integers")
    eps = options.get("eps", 1e-4)
    score_clip = options.get("score_clip", 10.0)
    hidden_ratio = options.get("hidden_ratio", 4)
    scale_temperature = options.get("scale_temperature", 1.0)
    # Construct once for early validation of scale settings.
    MultiScaleStandardizedResidual(
        dilations=dilations,
        aggregation="adaptive" if mode.endswith("adaptive") else "max",
        eps=eps,
        score_clip=score_clip,
        scale_temperature=scale_temperature,
    )
    if type(hidden_ratio) is not int or hidden_ratio < 1:
        raise ValueError("hidden_ratio must be a positive integer")
    return {
        "mode": mode,
        "dilations": list(dilations),
        "eps": float(eps),
        "score_clip": float(score_clip),
        "hidden_ratio": hidden_ratio,
        "scale_temperature": float(scale_temperature),
    }


def install_msr_feature_module(model: nn.Module, options: dict | None) -> None:
    """Replace six vetted Identity slots in the YOLO26n-P2 neck."""
    validated = validate_feature_module(options)
    layers = getattr(model, "model", None)
    if layers is None or len(layers) != DETECT_INDEX + 1:
        raise ValueError("MSR requires configs/model/yolo26n-msr-p2.yaml with six fusion slots")
    detect = layers[DETECT_INDEX]
    if detect.f != DETECT_SOURCES:
        raise ValueError(f"Expected Detect sources {DETECT_SOURCES}, got {detect.f}")

    for index, (high_channels, low_channels, high_first) in FUSION_SPECS.items():
        gate = layers[index]
        if not isinstance(gate, nn.Identity) or gate.f != -1 or gate.i != index:
            raise ValueError(f"Expected an Identity fusion slot at layer {index}")
        previous = layers[index - 1]
        if type(previous).__name__ != "Concat" or not isinstance(previous.f, list):
            raise ValueError(f"Layer {index} must follow an explicit Concat node")

    if validated["mode"] != "none":
        for index, (high_channels, low_channels, high_first) in FUSION_SPECS.items():
            gate = MSRFusionGate(
                high_channels=high_channels,
                low_channels=low_channels,
                high_first=high_first,
                mode=validated["mode"],
                dilations=validated["dilations"],
                eps=validated["eps"],
                score_clip=validated["score_clip"],
                hidden_ratio=validated["hidden_ratio"],
                scale_temperature=validated["scale_temperature"],
            )
            original = layers[index]
            gate.i = original.i
            gate.f = original.f
            gate.type = type(gate).__name__
            gate.np = sum(parameter.numel() for parameter in gate.parameters())
            layers[index] = gate

    model.yaml["tinydet_feature_module"] = validated
