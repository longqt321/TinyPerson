"""Multi-scale residual refinement block used by the YOLO26n P2 YAML."""
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ultralytics.nn import tasks as ultralytics_tasks


class MSR(nn.Module):
    """Multi-scale residual refinement for one BCHW feature map.

    Local statistics use valid pixels only at image boundaries. Statistics
    are computed in FP32; learned projections follow the caller's AMP context.
    """

    def __init__(
        self,
        channels: int,
        kernels: tuple[int, ...] | list[int] = (3, 5, 7),
        eps: float = 1e-6,
        sigma_min: float = 1e-3,
        z_max: float = 10.0,
        reduction: int = 4,
        residual_scale: float = 1e-3,
    ):
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError("channels must be a positive integer")
        if not kernels or any(type(k) is not int or k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("kernels must be a nonempty sequence of positive odd integers")
        if len(set(kernels)) != len(kernels):
            raise ValueError("kernels must not contain duplicates")
        for name, value in (("eps", eps), ("sigma_min", sigma_min), ("z_max", z_max)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if type(reduction) is not int or reduction < 1:
            raise ValueError("reduction must be a positive integer")
        if not isinstance(residual_scale, (int, float)) or not math.isfinite(residual_scale):
            raise ValueError("residual_scale must be finite")

        self.channels = channels
        self.kernels = tuple(kernels)
        self.eps = float(eps)
        self.sigma_min = float(sigma_min)
        self.z_max = float(z_max)
        hidden = max(channels // reduction, 8)
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )
            for _ in self.kernels
        ])
        self.scale_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, channels),
                nn.Sigmoid(),
            )
            for _ in self.kernels
        ])
        self.feature_gate = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.Sigmoid())
        self.refine = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or not x.is_floating_point() or x.shape[1] != self.channels:
            raise ValueError(f"Expected floating BCHW input with C={self.channels}")

        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            normalized = []
            for kernel in self.kernels:
                pad = kernel // 2
                mean = F.avg_pool2d(x32, kernel, stride=1, padding=pad, count_include_pad=False)
                second = F.avg_pool2d(x32.square(), kernel, stride=1, padding=pad,
                                      count_include_pad=False)
                std = (second.sub(mean.square()).clamp_min(0).add(self.eps).sqrt()
                       .clamp_min(self.sigma_min))
                z = ((x32 - mean) / std).clamp(-self.z_max, self.z_max)
                normalized.append(torch.cat((F.relu(z), F.relu(-z)), dim=1).to(dtype))

        features = [projection(z) for projection, z in zip(self.projections, normalized)]
        gates = [gate(F.adaptive_avg_pool2d(feature, 1).flatten(1)).view(-1, self.channels, 1, 1)
                 for gate, feature in zip(self.scale_gates, features)]
        denominator = torch.stack(gates, dim=0).sum(dim=0).clamp_min(1e-6)
        evidence = sum(feature * gate for feature, gate in zip(features, gates)) / denominator
        residual = self.refine(x * self.feature_gate(evidence))
        return x + self.gamma.to(residual.dtype) * residual


# Ultralytics parse_model resolves YAML module names from tasks.py globals.
ultralytics_tasks.MSR = MSR


def load_p2_msr_weights(target: nn.Module, source: nn.Module) -> dict[str, int]:
    """Transfer matching weights, remapping legacy Detect layer 29 to 30."""
    from ultralytics.utils import LOGGER

    source_state = source.float().state_dict()
    target_state = target.state_dict()
    transferred, remapped, skipped = {}, 0, 0
    for key, value in source_state.items():
        candidates = [key]
        if key.startswith("model.29."):
            candidates.append("model.30." + key[len("model.29."):])
        for index, target_key in enumerate(candidates):
            if target_key in target_state and target_state[target_key].shape == value.shape:
                transferred[target_key] = value
                remapped += index > 0
                break
        else:
            skipped += 1
    if not any(key.startswith("model.30.") for key in transferred):
        raise RuntimeError("No P2 architecture detection-head weights matched")
    if len(transferred) < 0.8 * len(source_state):
        raise RuntimeError(
            f"Only {len(transferred)}/{len(source_state)} checkpoint tensors matched P2 MSR model"
        )
    target.load_state_dict(transferred, strict=False)
    missing = len(target_state) - len(transferred)
    LOGGER.info(
        "Transferred %d/%d checkpoint tensors (%d layer-index remapped, %d source skipped, "
        "%d target tensors left initialized)",
        len(transferred), len(source_state), remapped, skipped, missing,
    )
    return {
        "transferred": len(transferred), "remapped": remapped,
        "skipped": skipped, "missing": missing,
    }
