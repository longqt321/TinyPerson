"""Local background evidence with optional tangent-space suppression.

The ring estimates an affine local background from eight neighbours. The
normalized contrast control and BTRM share the SAME background estimator;
they differ only in whether the center residual is ridge-projected away
from the local background tangent space.

Proof scope: the ridge residual is a non-expansive linear map when the
estimated tangent matrix is fixed. This does not imply an AP gain.
"""
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1))


def evidence_score(
    x: Tensor, mode: str = "btrm", radius: int = 1,
    ridge_relative: float = 0.05, eps: float = 1e-3,
) -> Tensor:
    """Return [B,1,H,W] normalized center evidence in full FP32."""
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("Expected floating-point BCHW features")
    if mode not in {"contrast", "btrm"}:
        raise ValueError("mode must be 'contrast' or 'btrm'")
    if type(radius) is not int or radius < 1:
        raise ValueError("radius must be a positive integer")
    if not (math.isfinite(ridge_relative) and ridge_relative > 0):
        raise ValueError("ridge_relative must be positive and finite")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError("eps must be positive and finite")
    b, c, h, w = x.shape
    if min(h, w) <= 2 * radius:
        raise ValueError("Feature map is too small for this ring radius")

    # FP32 prevents AMP overflow and stabilizes the 2x2 ridge solve.
    with torch.autocast(device_type=x.device.type, enabled=False):
        f = x.float()
        padded = F.pad(f, (radius, radius, radius, radius), mode="replicate")
        mean = torch.zeros_like(f)
        dx_sum = torch.zeros_like(f)
        dy_sum = torch.zeros_like(f)
        samples = []
        for dy, dx in OFFSETS:
            sy, sx = dy * radius, dx * radius
            neighbour = padded[:, :, radius + sy:radius + sy + h,
                               radius + sx:radius + sx + w]
            samples.append((dy, dx, neighbour))
            mean = mean + neighbour
            dx_sum = dx_sum + sx * neighbour
            dy_sum = dy_sum + sy * neighbour
        mean = mean / 8.0
        gx = dx_sum / (6.0 * radius * radius)
        gy = dy_sum / (6.0 * radius * radius)
        residual = f - mean

        xx = gx.square().sum(1, keepdim=True)
        yy = gy.square().sum(1, keepdim=True)
        xy = (gx * gy).sum(1, keepdim=True)
        lam = ridge_relative * (xx + yy) + eps * eps
        a, d = xx + lam, yy + lam
        determinant = (a * d - xy.square()).clamp_min(eps ** 4)

        if mode == "btrm":
            rhs_x = (gx * residual).sum(1, keepdim=True)
            rhs_y = (gy * residual).sum(1, keepdim=True)
            coef_x = (d * rhs_x - xy * rhs_y) / determinant
            coef_y = (a * rhs_y - xy * rhs_x) / determinant
            residual = residual - gx * coef_x - gy * coef_y

        sse = torch.zeros_like(xx)
        for dy, dx, neighbour in samples:
            fitted = mean + gx * (dx * radius) + gy * (dy * radius)
            sse = sse + (neighbour - fitted).square().sum(1, keepdim=True)
        sigma = (sse / (8.0 * c) + eps * eps).sqrt()

        magnitude = ((residual.square().sum(1, keepdim=True) + eps * eps).sqrt()
                     - eps) / math.sqrt(c)
        score = magnitude / sigma
        # Ignore padding-induced discrepancy near image boundaries.
        interior = torch.zeros_like(score)
        interior[..., radius:h-radius, radius:w-radius] = 1
        return score * interior


class LocalBackgroundEvidence(nn.Module):
    """Shape-preserving modulation of a single P3 feature map.

    Exact identity at initialization; learned gain is bounded in (-0.5, 0.5).
    """

    def __init__(self, mode: str = "btrm", radius: int = 1,
                 ridge_relative: float = 0.05, eps: float = 1e-3,
                 saturation: float = 3.0):
        super().__init__()
        if mode not in {"contrast", "btrm"}:
            raise ValueError("Unknown local background evidence mode")
        if not (math.isfinite(saturation) and saturation > 0):
            raise ValueError("saturation must be positive and finite")
        # Validate at creation rather than during a training step.
        if type(radius) is not int or radius < 1:
            raise ValueError("radius must be a positive integer")
        if not (math.isfinite(ridge_relative) and ridge_relative > 0
                and math.isfinite(eps) and eps > 0):
            raise ValueError("ridge_relative and eps must be positive and finite")
        self.mode = mode
        self.radius = radius
        self.ridge_relative = ridge_relative
        self.eps = eps
        self.saturation = saturation
        self.alpha_raw = nn.Parameter(torch.zeros(()))

    @property
    def effective_gain(self) -> Tensor:
        return 0.5 * torch.tanh(self.alpha_raw)

    def forward(self, x: Tensor) -> Tensor:
        score = evidence_score(x, self.mode, self.radius, self.ridge_relative, self.eps)
        gate = torch.tanh(score / self.saturation)
        # Cast factor back to input precision only at the end.
        return x * (1 + self.effective_gain * gate).to(x.dtype)


GATE_INDEX = 17
P3_SOURCE_INDEX = 16
DETECT_INDEX = 24


def validate_feature_module(options: dict | None) -> dict:
    if options is None:
        return {"mode": "none"}
    if not isinstance(options, dict) or set(options) - {
        "mode", "radius", "ridge_relative", "eps", "saturation"
    }:
        raise ValueError("feature_module must be a mapping with known keys")
    mode = options.get("mode", "none")
    if mode not in {"none", "contrast", "btrm"}:
        raise ValueError("feature_module.mode must be none, contrast or btrm")
    if mode == "none" and set(options) != {"mode"}:
        raise ValueError("feature_module 'none' does not take BTRM parameters")
    if mode != "none":
        LocalBackgroundEvidence(
            mode=mode, radius=options.get("radius", 1),
            ridge_relative=options.get("ridge_relative", 0.05),
            eps=options.get("eps", 1e-3),
            saturation=options.get("saturation", 3.0),
        )
    return dict(options)


def install_feature_module(model: nn.Module, options: dict | None) -> None:
    """Replace the explicit P3 Identity slot; never touch the P4/P5 pathways."""
    validated = validate_feature_module(options)
    layers = model.model
    if len(layers) != DETECT_INDEX + 1:
        raise ValueError("BTRM requires configs/model/yolo26n-btrm.yaml")
    gate = layers[GATE_INDEX]
    detect = layers[DETECT_INDEX]
    if not isinstance(gate, nn.Identity) or gate.f != -1 or gate.i != GATE_INDEX:
        raise ValueError("Expected a P3 nn.Identity gate at layer 17")
    if layers[GATE_INDEX + 1].f != P3_SOURCE_INDEX or detect.f != [
        GATE_INDEX, 20, 23
    ]:
        raise ValueError("P3/P4/P5 routing differs from the vetted BTRM architecture")
    if validated["mode"] == "none":
        return
    args = {key: value for key, value in validated.items() if key != "mode"}
    replacement = LocalBackgroundEvidence(mode=validated["mode"], **args)
    # Ultralytics' graph interpreter relies on these parser-populated attributes.
    for attr in ("i", "f", "type", "np"):
        setattr(replacement, attr, getattr(gate, attr))
    replacement.type = type(replacement).__name__
    replacement.np = sum(p.numel() for p in replacement.parameters())
    layers[GATE_INDEX] = replacement
    model.yaml["tinydet_feature_module"] = validated
