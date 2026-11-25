"""GauSS — Gaussian Scale-Space Feature Refinement for YOLO26n-P2.

Drop-in PyTorch module registered under the name ``GauSS``. It refines the
P2 detection feature with an explicit Gaussian scale-space decomposition,
band routing, content-aware upsampling and bounded residual gates.

Structure (all inside this file):

- ``LearnableGaussianScaleSpace``: per-channel separable Gaussian blurs at
  two learnable scales (sigma1 fine, sigma2 coarse). Sigmas are
  re-parameterized as ``sigma = center + half * tanh(raw)`` with
  sigma1 in [0.4, 0.9] (center 0.65, half 0.25) and sigma2 in [1.1, 1.8]
  (center 1.45, half 0.35); d(sigma)/d(raw) at init is 0.25 / 0.35.
- ``DySampleUp``: DySample-style point-sampling upsampler (arXiv:2308.15085)
  with zero-init offsets, equal to bilinear interpolation at init.
- ``BandBranch`` / ``DilatedContextBranch`` / ``RefinementBranch``: light
  per-band processors (dilated context rates default (1, 3, 5)).
- ``GauSS``: composition. Output
  ``y_p2 = x0 + alpha * z(sigma) + gamma_in * b_in + gamma_out * b_h
  + gamma_mid * b_m`` where the gates are ``g = 0.5 * tanh(raw)`` with
  warm-start values alpha=0.10, gamma_in=gamma_out=gamma_mid=0.05
  (constants ``GATE_INIT_*`` below).
- Hybrid band router: parameter-free log-energy prior measured on the raw
  bands at stride-4 (pre-BN physical magnitude, pooled to stride-8) plus a
  zero-init 1x1 learned correction inside the softmax
  (arXiv:2202.08906 router z-loss exposed via ``GauSS.router_z_loss()``).
- Ablation flags via env var ``TINYDET_ABLATION`` (frozen at construction):
  see ``ABLATION_FLAGS`` and ``mode_unused_param_names``. Band-fusion
  weights are renormalized over the active bands.
- fp32 routing math (energies, log-softmax, z-loss) for AMP safety.

Integration notes (everything stays inside this file):
- DDP: in default mode the parameters returned by
  ``module.unused_param_names()`` receive no gradient; either run
  DDP(find_unused_parameters=True) or use the list to build the DDP wrapper.
  In ``learned_router`` mode only router_corr.* is unused.
- Optimizer: ``configure_gauss_control_param_group(opt, model, lr_mult=...)``
  puts the control scalars (128 sigma + 4 gates) into a dedicated group
  excluded from weight decay, with an optional LR multiplier; idempotent,
  safe on resume.
- Legacy checkpoints: ``remap_v3_to_v31_state_dict`` and
  ``remap_v31_to_v32_state_dict`` convert older parameterizations; load
  with ``strict=False`` for params added after v3.
- Router z-loss (hybrid mode): optional; ``collect_gauss_router_zloss(model)``
  returns the scalar to add to the loss.
- Router flag precedence: learned_router > no_router > hybrid default.
- Export: PixelShuffle / PixelUnshuffle / grid_sample map to standard ONNX
  ops; verify the reflect-pad path on the target TensorRT version.

Contract summary (input 1280x1280, C2=64, P3=64, P4=128, D=32, B=1):
- Trainable parameters : 36,879
- Learned Conv2d layers: 27
- Control scalars      : 132 (128 sigma + alpha + gamma_in + gamma_out + gamma_mid)
- Exact MACs (B=1)     : 1,041,326,080 default
                         1,049,518,080 with learned_router
                         (use_saliency adds 819,200, not in contract)
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CONTROL_PARAM_SUFFIXES: tuple[str, ...] = (
    "sigma1x_raw",
    "sigma1y_raw",
    "sigma2x_raw",
    "sigma2y_raw",
    "alpha_raw",
    "gamma_in_raw",
    "gamma_out_raw",
    "gamma_mid_raw",
)

# Warm-started gate values: the output is a small bounded perturbation of
# the input feature at init, and every gate has a nonzero gradient.
GATE_INIT_ALPHA = 0.10
GATE_INIT_GAMMA_IN = 0.05
GATE_INIT_GAMMA_OUT = 0.05
GATE_INIT_GAMMA_MID = 0.05

# Sigma parameterization: sigma = center + half*tanh(raw).
SIGMA1_CENTER, SIGMA1_HALF = 0.65, 0.25  # range [0.4, 0.9]
SIGMA2_CENTER, SIGMA2_HALF = 1.45, 0.35  # range [1.1, 1.8]


def _gate_raw(gate: float) -> float:
    """Inverse of g = 0.5 * tanh(raw): the raw value that yields gate `gate`."""
    return math.atanh(2.0 * float(gate))

ABLATION_FLAGS: frozenset[str] = frozenset({
    "no_high_band",
    "no_mid_band",
    "no_low_band",
    "no_direct_anchor",
    "no_router",
    "no_router_corr",
    "no_saliency",        # kept for CLI compatibility; saliency is off by default
    "learned_router",
    "use_saliency",
    "energy_on_refined",  # legacy v3 behaviour: energies on post-BN refined bands
})


def active_ablation_flags() -> frozenset[str]:
    """Read the TINYDET_ABLATION env var (comma-separated flags).

    Unknown flags raise ValueError to catch typos early. Architecture and
    parameter layout stay identical regardless of flags; flags only change
    which contributions are active in forward().
    Set the env var BEFORE module construction (flags are frozen there);
    under multi-process DDP this means before the workers spawn.
    """
    raw = os.environ.get("TINYDET_ABLATION", "").strip()
    if not raw:
        return frozenset()
    flags = frozenset(part.strip() for part in raw.split(",") if part.strip())
    unknown = flags - ABLATION_FLAGS
    if unknown:
        raise ValueError(
            f"Unknown TINYDET_ABLATION flags: {sorted(unknown)}; "
            f"valid flags: {sorted(ABLATION_FLAGS)}"
        )
    return flags


def mode_unused_param_names(flags: Sequence[str] | frozenset[str]) -> tuple[str, ...]:
    """Parameter names that receive NO gradient for a given ablation mode.

    Use with DDP(find_unused_parameters=True) or to scope grad assertions.
    """
    flags = frozenset(flags)
    unused: list[str] = []
    if "learned_router" in flags:
        # hybrid correction is inactive in legacy learned-router mode
        unused += ["router_corr.weight", "router_corr.bias"]
    elif "no_router" in flags:
        unused += [
            "router_corr.weight",
            "router_corr.bias",
            "route_dw.weight",
            "route_bn.weight",
            "route_bn.bias",
            "route_out.weight",
            "route_out.bias",
        ]
    else:  # default hybrid mode
        unused += [
            "route_dw.weight",
            "route_bn.weight",
            "route_bn.bias",
            "route_out.weight",
            "route_out.bias",
        ]
    if "use_saliency" not in flags:
        unused += ["saliency_gate.weight", "saliency_gate.bias"]
    return tuple(unused)


class LearnableGaussianScaleSpace(nn.Module):
    """Separable 1D learnable Gaussian scale-space filter with single reflect pad.

    Two scales per channel and per axis (tanh-centered parameterization,
    d(sigma)/d(raw) = 0.25 / 0.35 at init):
      - Scale 1: support 5x5 (radius=2), sigma_1 = 0.65 + 0.25*tanh(raw),
                 range [0.4, 0.9], init 0.65
      - Scale 2: support 9x9 (radius=4), sigma_2 = 1.45 + 0.35*tanh(raw),
                 range [1.1, 1.8], init 1.45
    """

    coord1: Tensor
    coord2: Tensor

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.channels = int(channels)
        self.r1 = 2  # 5x5 support
        self.r2 = 4  # 9x9 support

        self.sigma1x_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.sigma1y_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.sigma2x_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.sigma2y_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))

        self.register_buffer(
            "coord1",
            torch.arange(-self.r1, self.r1 + 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "coord2",
            torch.arange(-self.r2, self.r2 + 1, dtype=torch.float32),
            persistent=False,
        )

    def compute_sigmas(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute bounded sigmas in FP32 (tanh-centered, slope 0.25/0.35)."""
        s1x = SIGMA1_CENTER + SIGMA1_HALF * torch.tanh(self.sigma1x_raw.float())
        s1y = SIGMA1_CENTER + SIGMA1_HALF * torch.tanh(self.sigma1y_raw.float())
        s2x = SIGMA2_CENTER + SIGMA2_HALF * torch.tanh(self.sigma2x_raw.float())
        s2y = SIGMA2_CENTER + SIGMA2_HALF * torch.tanh(self.sigma2y_raw.float())
        return s1x, s1y, s2x, s2y

    @staticmethod
    def _kernel_1d(sigma: Tensor, coords: Tensor) -> Tensor:
        """Build normalized 1D Gaussian kernel of shape (C, K) in FP32."""
        t2 = coords.unsqueeze(0).square()  # (1, K)
        denom = 2.0 * sigma.unsqueeze(1).square()  # (C, 1)
        logits = -t2 / denom  # (C, K)
        return torch.softmax(logits, dim=-1)

    def _apply_separable(
        self,
        x: Tensor,
        sigma_x: Tensor,
        sigma_y: Tensor,
        coords: Tensor,
        radius: int,
    ) -> Tensor:
        """Single reflect pad followed by horizontal and vertical DWConv2d."""
        c = self.channels
        gx = self._kernel_1d(sigma_x, coords).to(dtype=x.dtype).view(c, 1, 1, -1)
        gy = self._kernel_1d(sigma_y, coords).to(dtype=x.dtype).view(c, 1, -1, 1)

        x_pad = F.pad(x, (radius, radius, radius, radius), mode="reflect")
        x_h = F.conv2d(x_pad, gx, bias=None, stride=1, padding=0, groups=c)
        return F.conv2d(x_h, gy, bias=None, stride=1, padding=0, groups=c)

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Return low-pass features (L1, L2) and the 4 sigma vectors."""
        s1x, s1y, s2x, s2y = self.compute_sigmas()
        l1 = self._apply_separable(x, s1x, s1y, self.coord1, self.r1)
        l2 = self._apply_separable(x, s2x, s2y, self.coord2, self.r2)
        return l1, l2, (s1x, s1y, s2x, s2y)


class DySampleUp(nn.Module):
    """DySample-style point-sampling upsampler with zero-init offsets.

    Reference: "Learning to Upsample by Learning to Sample", ICCV 2023
    (arXiv:2308.15085). A 1x1 head predicts per-group, per-sub-pixel
    offsets; offsets are added to the bilinear base grid and applied with
    `F.grid_sample`.

    With zero-init offsets (enforced by GauSS._reset_parameters) the module
    is EXACTLY `F.interpolate(x, scale_factor=s, mode="bilinear",
    align_corners=False)`; the self-test asserts this equality, which makes
    the operator's correctness machine-checked instead of assumed.
    """

    def __init__(self, channels: int, scale: int = 2, groups: int = 4) -> None:
        super().__init__()
        channels = int(channels)
        scale = int(scale)
        groups = int(groups)
        if scale != 2:
            raise ValueError("DySampleUp is specialized for scale=2 (P3->P2 path)")
        if channels % groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")
        self.channels = channels
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(channels, groups * 2 * scale * scale, kernel_size=1, bias=True)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.last_diag: dict[str, float] = {}
        # Diagnostics need a GPU->CPU sync (.item()); see GauSS.arm_diagnostics.
        self._diag_armed = True
        self._diag_once = False

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        s, g = self.scale, self.groups
        if c != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {c}")

        off = self.offset(x)                      # (B, g*2*s^2, H, W)
        off = F.pixel_shuffle(off, s)             # (B, 2g, sH, sW)
        off = off.reshape(b, g, 2, s * h, s * w)  # [:, :, 0]=dy, [:, :, 1]=dx

        # Detached diagnostics: offset magnitude relative to the bilinear
        # baseline grid. Zero-init offsets -> 0.0 (pure bilinear).
        if self._diag_armed:
            with torch.no_grad():
                self.last_diag = {"offset_rms": float(off.float().square().mean().sqrt().item())}
            self._diag_armed = not self._diag_once

        # Base sampling grid in input-pixel units, align_corners=False convention:
        # output pixel J samples input at (J + 0.5)/s - 0.5. Computed in
        # fp32 regardless of input dtype (AMP-safe: avoids fp16 coordinate
        # drift from arange/division), cast to input dtype for grid_sample.
        ry = torch.arange(s * h, device=x.device, dtype=torch.float32)
        rx = torch.arange(s * w, device=x.device, dtype=torch.float32)
        base_y = ((ry + 0.5) / s - 0.5).view(1, 1, s * h, 1)
        base_x = ((rx + 0.5) / s - 0.5).view(1, 1, 1, s * w)
        grid_y = base_y + off[:, :, 0].float()
        grid_x = base_x + off[:, :, 1].float()

        # pixel units -> normalized coordinates for grid_sample
        gy_n = (2.0 * grid_y + 1.0) / h - 1.0
        gx_n = (2.0 * grid_x + 1.0) / w - 1.0
        grid = torch.stack((gx_n, gy_n), dim=-1).to(x.dtype)  # (B, g, sH, sW, 2)

        xg = x.reshape(b * g, c // g, h, w)
        out = F.grid_sample(
            xg,
            grid.reshape(b * g, s * h, s * w, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return out.reshape(b, c, s * h, s * w)


class BandBranch(nn.Module):
    """Single band encoder branch.

    PixelUnshuffle(2) -> Conv1x1(4D->D) + BN + SiLU -> DWConv3x3 + BN + SiLU
    -> PWConv1x1(D->D) + BN -> Residual Add + SiLU.
    """

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        d = int(hidden_dim)
        self.pu = nn.PixelUnshuffle(downscale_factor=2)
        self.proj = nn.Conv2d(4 * d, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_proj = nn.BatchNorm2d(d)
        self.dw = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1, groups=d, bias=False)
        self.bn_dw = nn.BatchNorm2d(d)
        self.pw = nn.Conv2d(d, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw = nn.BatchNorm2d(d)
        self.act = nn.SiLU(inplace=False)

    def forward(self, b_band: Tensor) -> Tensor:
        z = self.act(self.bn_proj(self.proj(self.pu(b_band))))
        d_feat = self.act(self.bn_dw(self.dw(z)))
        p_feat = self.bn_pw(self.pw(d_feat))
        return self.act(z + p_feat)


class DilatedContextBranch(nn.Module):
    """Single parallel dilated depthwise branch."""

    def __init__(self, hidden_dim: int = 32, dilation: int = 1) -> None:
        super().__init__()
        d = int(hidden_dim)
        dil = int(dilation)
        self.dw = nn.Conv2d(
            d,
            d,
            kernel_size=3,
            stride=1,
            padding=dil,
            dilation=dil,
            groups=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(d)
        self.act = nn.SiLU(inplace=False)

    def forward(self, c0: Tensor) -> Tensor:
        return self.act(self.bn(self.dw(c0)))


class RefinementBranch(nn.Module):
    """Residual depthwise-separable refinement branch.

    Used for Detail-Focused (kernel_size=3) and Structural-Context (kernel_size=5).
    """

    def __init__(self, hidden_dim: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        d = int(hidden_dim)
        k = int(kernel_size)
        pad = k // 2
        self.dw = nn.Conv2d(
            d, d, kernel_size=k, stride=1, padding=pad, groups=d, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(d)
        self.pw = nn.Conv2d(d, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw = nn.BatchNorm2d(d)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.bn_dw(self.dw(x)))
        h = self.bn_pw(self.pw(h))
        return self.act(x + h)


class GauSS(nn.Module):
    """GauSS — Gaussian Scale-Space Feature Refinement.

    Bounded residual gates (warm-started, see module docstring),
    energy-prior routing with bounded learned
    correction, content-aware upsampling, and contamination-free ablations.
    See module docstring for design details and integration notes.
    """

    def __init__(
        self,
        c1: Sequence[int] | int = (64, 64, 128),
        c2: int = 32,
        p4_ch: int | None = None,
        c_out: int | None = None,
        dilations: Sequence[int] = (1, 3, 5),
    ) -> None:
        super().__init__()
        if isinstance(c1, Sequence):
            if len(c1) != 3:
                raise ValueError(
                    f"GauSS expects 3 input channels [C2, P3, P4], got {list(c1)}"
                )
            c2_in, p3_in, p4_in = int(c1[0]), int(c1[1]), int(c1[2])
            d = int(c2)
        else:
            if p4_ch is None or c_out is None:
                raise ValueError(
                    "When c1 is an int, provide (c2_ch, p3_ch, p4_ch, c_out) or pass c1=[c2, p3, p4]."
                )
            c2_in, p3_in, p4_in = int(c1), int(c2), int(p4_ch)
            d = int(c_out)

        dil = tuple(int(v) for v in dilations)
        if len(dil) != 3 or any(v < 1 for v in dil):
            raise ValueError(f"dilations must be 3 positive ints, got {dil}")

        self.c2_channels = c2_in
        self.p3_channels = p3_in
        self.p4_channels = p4_in
        self.hidden_dim = d
        self.out_channels = d
        self.dilations = dil

        # ------------------------------------------------------------------ #
        # Block 1: Multi-Scale Band Feature Encoder
        # ------------------------------------------------------------------ #
        self.proj_c2 = nn.Conv2d(c2_in, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_p3 = nn.Conv2d(p3_in, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_p4 = nn.Conv2d(p4_in, d, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(d)

        self.gaussian = LearnableGaussianScaleSpace(channels=d)

        self.band_h = BandBranch(hidden_dim=d)
        self.band_m = BandBranch(hidden_dim=d)
        self.band_l = BandBranch(hidden_dim=d)

        # ------------------------------------------------------------------ #
        # Block 2: Gated Dilated Context Controller
        # ------------------------------------------------------------------ #
        self.bn3 = nn.BatchNorm2d(d)
        self.ctx_d1 = DilatedContextBranch(hidden_dim=d, dilation=dil[0])
        self.ctx_d2 = DilatedContextBranch(hidden_dim=d, dilation=dil[1])
        self.ctx_d3 = DilatedContextBranch(hidden_dim=d, dilation=dil[2])
        self.ctx_gate = nn.Conv2d(d, 3, kernel_size=1, stride=1, padding=0, bias=True)
        # Saliency gate (legacy v2/v3 path) — active only under use_saliency.
        self.saliency_gate = nn.Conv2d(d, 1, kernel_size=1, stride=1, padding=0, bias=True)

        # Legacy heavy learned router — computed ONLY under `learned_router`.
        self.route_dw = nn.Conv2d(
            d, d, kernel_size=3, stride=1, padding=1, groups=d, bias=False
        )
        self.route_bn = nn.BatchNorm2d(d)
        self.route_out = nn.Conv2d(d, 4, kernel_size=1, stride=1, padding=0, bias=True)

        # Hybrid-router correction: zero-init, added to the log-energy
        # prior inside the band softmax; behaviour-identical to pure energy
        # routing at init (ST-MoE-style z-loss available via router_z_loss()).
        self.router_corr = nn.Conv2d(d, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # Content-aware upsamplers (zero-init == bilinear at init).
        self.dy_up3 = DySampleUp(d, scale=2, groups=4)
        self.dy_up4 = DySampleUp(d, scale=2, groups=4)

        # ------------------------------------------------------------------ #
        # Block 3: Asymmetric Feature Refinement & Reconstruction
        # ------------------------------------------------------------------ #
        self.detail_branch = RefinementBranch(hidden_dim=d, kernel_size=3)
        self.struct_branch = RefinementBranch(hidden_dim=d, kernel_size=5)

        self.recon_proj = nn.Conv2d(
            d, 4 * d, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.recon_bn = nn.BatchNorm2d(4 * d)
        self.ps = nn.PixelShuffle(upscale_factor=2)

        # Bounded residual gates: g = 0.5 * tanh(raw), warm-started
        # (nonzero init); with alpha == gamma == 0 the output would be
        # sigma-independent and dL/dsigma == 0 exactly at step 1.
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        # gamma_in : refined high-band anchor at stride-8 (inside x_d)
        self.gamma_in_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        # gamma_out: DIRECT high-band anchor at native stride-4 (in y_p2)
        self.gamma_out_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        # gamma_mid: DIRECT mid-band anchor at native stride-4 (in y_p2);
        # b_m = l1 - l2 depends on BOTH sigma1 and sigma2, giving sigma2 a
        # one-hop gradient path from sigma1 and sigma2 to the output.
        self.gamma_mid_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.act = nn.SiLU(inplace=False)

        # Ablation flags (env var TINYDET_ABLATION); frozen at construction.
        self.ablation_flags: frozenset[str] = active_ablation_flags()

        # Detached diagnostic dictionary for training loggers.
        self.last_diag: dict[str, float] = {}
        # Non-detached router z-loss (ST-MoE); add to the training loss if
        # the hybrid correction is enabled: loss += 1e-3 * module.router_z_loss()
        self._z_loss: Tensor = torch.zeros((), dtype=torch.float32)
        # Diagnostics cost ~30 GPU->CPU syncs per forward. Default: every forward
        # (tests, probes). Training callbacks call arm_diagnostics(once=True) so
        # only the first forward after each arming pays for them.
        self._diag_armed = True
        self._diag_once = False

        self._reset_parameters()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _reset_parameters(self) -> None:
        """Kaiming init for convs, then explicit overrides for control heads."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Overrides AFTER the Kaiming sweep:
        with torch.no_grad():
            # zero-init learned router correction (hybrid == pure energy at init)
            self.router_corr.weight.zero_()
            self.router_corr.bias.zero_()
            # zero-init DySample offsets (== bilinear at init)
            for dy in (self.dy_up3, self.dy_up4):
                dy.offset.weight.zero_()
                dy.offset.bias.zero_()
            # Section 4.4 (legacy learned router): W_m = 0.01 * W_Kaiming, b = 0
            self.route_out.weight[3].mul_(0.01)
            if self.route_out.bias is not None:
                self.route_out.bias.zero_()
            # control scalars: sigmas centered at the configured init values,
            # gates warm-started so the sigma path is differentiable at step 1.
            self.gaussian.sigma1x_raw.zero_()
            self.gaussian.sigma1y_raw.zero_()
            self.gaussian.sigma2x_raw.zero_()
            self.gaussian.sigma2y_raw.zero_()
            self.alpha_raw.fill_(_gate_raw(GATE_INIT_ALPHA))
            self.gamma_in_raw.fill_(_gate_raw(GATE_INIT_GAMMA_IN))
            self.gamma_out_raw.fill_(_gate_raw(GATE_INIT_GAMMA_OUT))
            self.gamma_mid_raw.fill_(_gate_raw(GATE_INIT_GAMMA_MID))
            # saliency gate: small weights, zero bias (legacy path)
            self.saliency_gate.weight.mul_(0.01)
            if self.saliency_gate.bias is not None:
                self.saliency_gate.bias.zero_()

    # ------------------------------------------------------------------ #
    # Bounded control gates (ReZero-exact at raw = 0)
    # ------------------------------------------------------------------ #
    def arm_diagnostics(self, once: bool = True) -> None:
        """Compute diagnostics (here and in the DySample children) on the next forward.

        With ``once=True`` each module disarms itself after that forward, so
        ``last_diag`` is refreshed once per arming instead of every batch.
        """
        for module in (self, self.dy_up3, self.dy_up4):
            module._diag_once = once
            module._diag_armed = True

    def compute_alpha(self) -> Tensor:
        """Bounded residual gain alpha = 0.5 * tanh(alpha_raw) in (-0.5, 0.5)."""
        return 0.5 * torch.tanh(self.alpha_raw.float())

    def compute_gamma_in(self) -> Tensor:
        """Refined high-band anchor gain = 0.5 * tanh(gamma_in_raw)."""
        return 0.5 * torch.tanh(self.gamma_in_raw.float())

    def compute_gamma_out(self) -> Tensor:
        """Direct high-band anchor gain = 0.5 * tanh(gamma_out_raw)."""
        return 0.5 * torch.tanh(self.gamma_out_raw.float())

    def compute_gamma_mid(self) -> Tensor:
        """Direct mid-band anchor gain = 0.5 * tanh(gamma_mid_raw)."""
        return 0.5 * torch.tanh(self.gamma_mid_raw.float())

    def router_z_loss(self) -> Tensor:
        """ST-MoE router z-loss (Eq. 5, arXiv:2202.08906) of the correction.

        Squared log-sum-exp of the correction logits, averaged over
        positions: (logsumexp(corr))^2. Penalizes the overall logit
        magnitude (the numerical-stability role it plays in ST-MoE);
        softmax(log_e + corr) is shift-invariant in corr, so the loss
        does not bias routing direction. Equals (log 3)^2 at init
        (zero logits).
        Add to the training loss (suggested weight 1e-3):
            loss = loss + 1e-3 * module.router_z_loss()            # per module
            loss = loss + 1e-3 * collect_gauss_router_zloss(model)  # whole model
        """
        return self._z_loss

    def unused_param_names(self) -> tuple[str, ...]:
        """Parameters that receive no gradient under the current ablation mode."""
        return mode_unused_param_names(self.ablation_flags)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: Sequence[Tensor]) -> Tensor:
        """Forward pass accepting [C2, P3, P4] and returning refined P2."""
        if not isinstance(x, Sequence) or len(x) != 3:
            raise ValueError(
                "GauSS.forward expects a list/tuple of 3 tensors: [C2, P3, P4]."
            )
        c2, p3, p4 = x[0], x[1], x[2]
        flags = self.ablation_flags

        # ---- 3.1 P2 Feature Fusion (content-aware upsample, zero-init==bilinear) ----
        s2 = self.proj_c2(c2)
        s3 = self.proj_p3(p3)
        s4 = self.proj_p4(p4)

        s3_up = self.dy_up3(s3)
        x0 = self.act(self.bn2(s2 + s3_up))

        # ---- 3.2 & 3.3 Learnable Gaussian Scale-Space & Band Decomposition ----
        l1, l2, sigmas = self.gaussian(x0)
        b_h = x0 - l1
        b_m = l1 - l2
        b_l = l2

        # ---- 3.4 & 3.5 PixelUnshuffle & Band Refinement ----
        r_h = self.band_h(b_h)
        r_m = self.band_m(b_m)
        r_l = self.band_l(b_l)

        # ---- 4.1 Context Feature Construction ----
        p4_up = self.dy_up4(s4)
        c0 = self.act(self.bn3(s3 + p4_up))

        # ---- 4.2 Multi-Scale Dilated Context ----
        t1 = self.ctx_d1(c0)
        t2 = self.ctx_d2(c0)
        t3 = self.ctx_d3(c0)

        # ---- 4.3 Context Branch Gating ----
        g_c = self.ctx_gate(c0)
        pi = torch.softmax(g_c, dim=1)
        pi1, pi2, pi3 = pi[:, 0:1], pi[:, 1:2], pi[:, 2:3]
        c_ctx = self.act(c0 + pi1 * t1 + pi2 * t2 + pi3 * t3)

        # ---- 4.4 Band Routing (energy prior on RAW bands + zero-init correction) ----
        # Energies in FP32. Default: measured on the raw bands b_* at native
        # stride-4 (pre-BN physical magnitude), then avg-pooled to stride-8.
        if "energy_on_refined" in flags:
            e_h = r_h.detach().float().abs().mean(dim=1, keepdim=True)
            e_m = r_m.detach().float().abs().mean(dim=1, keepdim=True)
            e_l = r_l.detach().float().abs().mean(dim=1, keepdim=True)
        else:
            e_h4 = b_h.detach().float().abs().mean(dim=1, keepdim=True)
            e_m4 = b_m.detach().float().abs().mean(dim=1, keepdim=True)
            e_l4 = b_l.detach().float().abs().mean(dim=1, keepdim=True)
            e_h = F.avg_pool2d(e_h4, 2)
            e_m = F.avg_pool2d(e_m4, 2)
            e_l = F.avg_pool2d(e_l4, 2)

        # Band-ablation masks; softmax over masked log-energies renormalizes
        # the fusion weights over the ACTIVE bands automatically.
        mask = torch.tensor(
            (
                0.0 if "no_high_band" in flags else 1.0,
                0.0 if "no_mid_band" in flags else 1.0,
                0.0 if "no_low_band" in flags else 1.0,
            ),
            device=e_h.device,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        e = torch.cat((e_h, e_m, e_l), dim=1) * mask
        log_e = torch.log(e + 1e-12)

        u = self.act(r_h + r_m + r_l + c_ctx)

        if "learned_router" in flags:
            # Legacy v3 heavy learned router (computed only under this flag).
            v = self.act(self.route_bn(self.route_dw(u)))
            logits = self.route_out(v)
            band_logits = logits[:, :3].float()
            mix_logit = logits[:, 3:4].float()
            a = torch.softmax(band_logits, dim=1)
            m_gate = torch.sigmoid(mix_logit)
            self._z_loss = torch.zeros((), device=x0.device, dtype=torch.float32)
        elif "no_router" in flags:
            # Equal weights over the ACTIVE bands (renormalized, consistent
            # with the hybrid path). Legacy v3 used a fixed 1/3 per band,
            # which shrank the fused mass when a band was ablated.
            n_active = sum(
                1 for f in ("no_high_band", "no_mid_band", "no_low_band")
                if f not in flags
            )
            a = (mask.view(1, 3, 1, 1) * (1.0 / max(n_active, 1))).expand_as(log_e)
            m_gate = torch.full(
                (e.shape[0], 1, 1, 1), 0.5, dtype=r_h.dtype, device=r_h.device
            ).float()
            self._z_loss = torch.zeros((), device=x0.device, dtype=torch.float32)
        else:
            # Default hybrid: softmax(log-energy prior + zero-init correction).
            # At init corr == 0 -> a == e / sum(e) EXACTLY (pure v3 energy rule).
            corr = self.router_corr(u).float()
            a = torch.softmax(log_e + corr, dim=1)
            # ST-MoE router z-loss on the learned part only (shift-invariant
            # penalization; the energy prior is data, not a router logit).
            self._z_loss = (torch.logsumexp(corr, dim=1) ** 2).mean()
            # Detail branch priority = combined high+mid evidence share.
            m_gate = a[:, 0:1] + a[:, 1:2]

        a_h = a[:, 0:1].to(dtype=r_h.dtype)
        a_m = a[:, 1:2].to(dtype=r_h.dtype)
        a_l = a[:, 2:3].to(dtype=r_h.dtype)
        m_gate = m_gate.to(dtype=r_h.dtype)

        # ---- 5.1 Adaptive Band Fusion (masked = renormalized over active bands) ----
        r_h_eff = torch.zeros_like(r_h) if "no_high_band" in flags else r_h
        r_m_eff = torch.zeros_like(r_m) if "no_mid_band" in flags else r_m
        r_l_eff = torch.zeros_like(r_l) if "no_low_band" in flags else r_l

        f_band = a_h * r_h_eff + a_m * r_m_eff + a_l * r_l_eff
        gamma_in = self.compute_gamma_in().to(dtype=r_h.dtype)

        # ---- 5.2 & 5.3 Detail-Focused & Structural-Context Refinement ----
        x_d = f_band + c_ctx + gamma_in * r_h_eff
        f_d = self.detail_branch(x_d)

        # Background Saliency Gating (legacy; off by default — v3 evidence:
        # it degraded earth_person by -2.38 pp in the 2000-bootstrap study).
        if "use_saliency" in flags:
            saliency_mask = torch.sigmoid(self.saliency_gate(c_ctx))
            r_l_attenuated = r_l_eff * (1.0 - 0.5 * saliency_mask)
        else:
            r_l_attenuated = r_l_eff
        x_s = r_l_attenuated + c_ctx
        f_s = self.struct_branch(x_s)

        # ---- 5.4 Spatial Feature Mixing ----
        f_route = m_gate * f_d + (1.0 - m_gate) * f_s

        # ---- 5.5 P2 Reconstruction (linear output, direct high-band anchor) ----
        z0 = self.recon_bn(self.recon_proj(f_route))
        z = self.ps(z0)

        alpha = self.compute_alpha().to(dtype=z.dtype)
        gamma_out = self.compute_gamma_out().to(dtype=b_h.dtype)
        gamma_mid = self.compute_gamma_mid().to(dtype=b_h.dtype)
        if ("no_high_band" in flags) or ("no_direct_anchor" in flags):
            b_h_eff = torch.zeros_like(b_h)
        else:
            b_h_eff = b_h
        if ("no_mid_band" in flags) or ("no_direct_anchor" in flags):
            b_m_eff = torch.zeros_like(b_m)
        else:
            b_m_eff = b_m

        # Residual addition WITHOUT activation: pure linear combination.
        # Warm start: alpha/gamma != 0 at init -> y_p2 is a small bounded
        # perturbation of x0 (residual RMS ~0.1 * RMS(x0)) with nonzero
        # gate and sigma gradients from the first step.
        y_p2 = x0 + alpha * z + gamma_out * b_h_eff + gamma_mid * b_m_eff

        # ---- 11. Detached Diagnostics ----
        if self._diag_armed:
            with torch.no_grad():
                s1x, s1y, s2x, s2y = sigmas
                sigma1 = torch.cat((s1x, s1y), dim=0)
                sigma2 = torch.cat((s2x, s2y), dim=0)
                m_f = m_gate.detach().float()
                rms_res = (alpha * z).detach().float().square().mean().sqrt()
                rms_in = x0.detach().float().square().mean().sqrt().clamp_min(1e-12)
                rms_out = y_p2.detach().float().square().mean().sqrt().clamp_min(1e-12)
                rms_anchor = (gamma_out * b_h_eff).detach().float().square().mean().sqrt()
                rms_mid_anchor = (gamma_mid * b_m_eff).detach().float().square().mean().sqrt()
                a_f = a.detach().float()
                # torch.special.entr: -x*ln(x) per element, 0 at x==0 (no clamping bias).
                entropy = torch.special.entr(a_f).sum(dim=1).mean()
                energy_sum = (e_h.mean() + e_m.mean() + e_l.mean()).clamp_min(1e-12)
    
                diag = {
                    "routing/mixing_mean": float(m_f.mean().item()),
                    "routing/mixing_std": float(m_f.std(unbiased=False).item()),
                    "routing/high_band_mean": float(a_h.detach().float().mean().item()),
                    "routing/mid_band_mean": float(a_m.detach().float().mean().item()),
                    "routing/low_band_mean": float(a_l.detach().float().mean().item()),
                    "routing/entropy_mean": float(entropy.item()),
                    "routing/energy_high_mean": float(e_h.mean().item()),
                    "routing/energy_mid_mean": float(e_m.mean().item()),
                    "routing/energy_low_mean": float(e_l.mean().item()),
                    "routing/high_band_share": float((e_h.mean() / energy_sum).item()),
                    "routing/z_loss": float(self._z_loss.item()),
                    "context/dilation_1_mean": float(pi1.detach().float().mean().item()),
                    "context/dilation_2_mean": float(pi2.detach().float().mean().item()),
                    "context/dilation_3_mean": float(pi3.detach().float().mean().item()),
                    "refinement/residual_ratio": float((rms_res / rms_in).item()),
                    "contribution/alpha_z_share": float((rms_res / rms_out).item()),
                    "contribution/anchor_share": float((rms_anchor / rms_out).item()),
                    "contribution/mid_anchor_share": float((rms_mid_anchor / rms_out).item()),
                    "contribution/identity_share": float((rms_in / rms_out).item()),
                    "gates/alpha": float(self.compute_alpha().item()),
                    "gates/gamma_in": float(self.compute_gamma_in().item()),
                    "gates/gamma_out": float(self.compute_gamma_out().item()),
                    "gates/gamma_mid": float(self.compute_gamma_mid().item()),
                    "gaussian/sigma1_mean": float(sigma1.mean().item()),
                    "gaussian/sigma2_mean": float(sigma2.mean().item()),
                    "gaussian/sigma1_std": float(sigma1.std(unbiased=False).item()),
                    "gaussian/sigma2_std": float(sigma2.std(unbiased=False).item()),
                    "gaussian/sigma1_min": float(sigma1.min().item()),
                    "gaussian/sigma1_max": float(sigma1.max().item()),
                    "gaussian/sigma2_min": float(sigma2.min().item()),
                    "gaussian/sigma2_max": float(sigma2.max().item()),
                }
                # Merge DySample offset diagnostics (0.0 == pure bilinear at init).
                for dy_name in ("dy_up3", "dy_up4"):
                    dy_diag = getattr(getattr(self, dy_name, None), "last_diag", None)
                    if dy_diag:
                        diag.update({f"dysample/{dy_name}/{k}": v for k, v in dy_diag.items()})
                self.last_diag = diag
            self._diag_armed = not self._diag_once

        return y_p2

    def get_diagnostics(
        self, optimizer: torch.optim.Optimizer | None = None
    ) -> dict[str, float]:
        """Return all Section 11 logging metrics including optimizer control group stats."""
        out = dict(self.last_diag)
        ctrl_lr = float("nan")
        ctrl_wd = float("nan")
        if optimizer is not None:
            for pg in optimizer.param_groups:
                if pg.get("param_group") == "control":
                    ctrl_lr = float(pg.get("lr", float("nan")))
                    ctrl_wd = float(pg.get("weight_decay", float("nan")))
                    break
        out["optimizer/control_lr"] = ctrl_lr
        out["optimizer/control_weight_decay"] = ctrl_wd
        return out



# --------------------------------------------------------------------------- #
# Legacy checkpoint adaptation helpers
# --------------------------------------------------------------------------- #
def remap_v3_to_v31_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a legacy GauSS checkpoint to the tanh gate parametrization.

    v3 gates:  g = 0.5 * sigmoid(raw)          in [0, 0.5]
    v3.1:      g = 0.5 * tanh(raw)             in (-0.5, 0.5)
    Matching g requires raw_v31 = atanh(sigmoid(raw_v3)).

    `gamma_h_raw` is duplicated into `gamma_in_raw` and `gamma_out_raw`
    (v3 coupled both roles into one scalar). Works with module-prefixed
    keys (e.g. "model.12.gauss.alpha_raw"). Load the result with
    load_state_dict(..., strict=False): v3 checkpoints lack the v3.1-only
    params (router_corr.*, dy_up3/dy_up4 .offset.*) and those keep their
    constructor zero-init — exactly the intended state.
    """
    out: dict[str, Tensor] = {}

    def _sig_to_tanh(raw: Tensor) -> Tensor:
        s = torch.sigmoid(raw.float()).clamp(1e-6, 1.0 - 1e-6)
        return torch.atanh(s).to(raw.dtype)

    for k, v in state_dict.items():
        if k.endswith("alpha_raw"):
            out[k] = _sig_to_tanh(v)
        elif k.endswith("gamma_h_raw"):
            base = k[: -len("gamma_h_raw")]
            mapped = _sig_to_tanh(v)
            out[base + "gamma_in_raw"] = mapped.clone()
            out[base + "gamma_out_raw"] = mapped.clone()
        else:
            out[k] = v
    return out


def remap_v31_to_v32_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a previous-generation checkpoint to the tanh sigma parameterization.

    Only the sigma raws change. v3.1: sigma1 = 0.4 + 0.5*sigmoid(r),
    sigma2 = 1.1 + 0.7*sigmoid(r). v3.2: sigma1 = 0.65 + 0.25*tanh(r),
    sigma2 = 1.45 + 0.35*tanh(r). Both give (sigma - center)/half =
    2*sigmoid(r) - 1 = tanh(r/2), hence r_v32 = r_v31 / 2 exactly.

    Gate raws need no conversion: g = 0.5*tanh(raw) in both versions, so a
    v3.1 raw already denotes the same gate value. `gamma_mid_raw` is absent
    from v3.1 checkpoints and keeps its constructor warm-start value; load
    with strict=False.
    """
    out: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        if k.endswith(("sigma1x_raw", "sigma1y_raw", "sigma2x_raw", "sigma2y_raw")):
            out[k] = (v.float() * 0.5).to(v.dtype)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Section 9: Optimizer Parameter Group Utilities
# --------------------------------------------------------------------------- #
def is_gauss_control_param_name(name: str) -> bool:
    """Return True if parameter name belongs to the GauSS control group."""
    return name.endswith(CONTROL_PARAM_SUFFIXES)


def collect_gauss_control_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect GauSS control scalars (sigmas, alpha, gamma_in, gamma_out)."""
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if param.requires_grad and is_gauss_control_param_name(name):
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                params.append(param)
    return params


def configure_gauss_control_param_group(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    base_lr: float | None = None,
    lr_mult: float = 1.0,
) -> dict[str, Any] | None:
    """Move GauSS control parameters into a dedicated `control` group with weight_decay=0.0.

    Call AFTER the trainer builds the optimizer (idempotent; safe on resume).
    Under Ultralytics this must be hooked post `trainer.build_optimizer`,
    otherwise the raw scalars land in the default `weight` group and weight
    decay drags alpha_raw/gamma_*_raw toward 0, silently changing the
    warm-started gate values.

    `lr_mult` (v3.2) scales the control-group LR relative to the inferred
    base LR. The sigma gradient is intrinsically ~1e-4-scale; the trainer
    passes `gauss_control_lr_mult` from the train config (default 3.0).
    NOTE: this function's own default is 1.0 (= v3.1 optimizer behaviour);
    the trainer is the only caller that applies the 3x multiplier, so
    calling this function directly gives v3.1-equivalent optimization.
    """
    control_params = collect_gauss_control_params(model)
    if not control_params:
        return None
    ctrl_ids = {id(p) for p in control_params}

    inferred_lr: float | None = base_lr
    existing_ctrl_group: dict[str, Any] | None = None

    for pg in optimizer.param_groups:
        if pg.get("param_group") == "control":
            existing_ctrl_group = pg
            continue
        if inferred_lr is None and pg.get("param_group") in ("weight", None):
            inferred_lr = float(pg.get("lr", 1e-3))
        pg["params"] = [p for p in pg["params"] if id(p) not in ctrl_ids]

    if inferred_lr is None:
        inferred_lr = float(optimizer.param_groups[0].get("lr", 1e-3))
    ctrl_lr = inferred_lr * float(lr_mult)

    if existing_ctrl_group is not None:
        existing_ctrl_group["params"] = control_params
        existing_ctrl_group["weight_decay"] = 0.0
        existing_ctrl_group["lr"] = ctrl_lr
        existing_ctrl_group["initial_lr"] = ctrl_lr
        return existing_ctrl_group

    ctrl_group: dict[str, Any] = {
        "params": control_params,
        "lr": ctrl_lr,
        "initial_lr": ctrl_lr,
        "weight_decay": 0.0,
        "param_group": "control",
    }
    for k, v in optimizer.param_groups[0].items():
        if k not in ctrl_group and k != "params":
            ctrl_group[k] = v

    optimizer.param_groups.append(ctrl_group)
    return ctrl_group


def collect_gauss_router_zloss(model: nn.Module) -> Tensor:
    """Sum the router z-loss over every GauSS module found in `model`.

    Convenience for the Ultralytics loss hook (single-file constraint —
    the loss function lives outside this module):
        loss = loss + 1e-3 * collect_gauss_router_zloss(model)
    Returns a zero tensor when no GauSS module is present or the hybrid
    correction is inactive (learned_router / no_router modes).
    """
    total: Tensor | None = None
    for m in model.modules():
        if isinstance(m, GauSS):
            zl = m.router_z_loss()
            total = zl if total is None else total + zl
    return total if total is not None else torch.zeros(())


# --------------------------------------------------------------------------- #
# Section 8: Analytical MACs & Parameter Verification
# --------------------------------------------------------------------------- #
def compute_gauss_analytical_macs(
    c2_ch: int = 64,
    p3_ch: int = 64,
    p4_ch: int = 128,
    d: int = 32,
    h_p2: int = 320,
    w_p2: int = 320,
    learned_router: bool = False,
    router_correction: bool = True,
) -> dict[str, float]:
    """Compute exact analytical MACs (B=1) for the v3.1 forward graph.

    The heavy learned-router stack is only counted when `learned_router`
    is True (it is computed only under that flag); in that mode the
    router-correction conv is inactive and never counted. Otherwise the
    correction conv is counted unless `router_correction` is False
    (no_router mode). grid_sample / interpolate arithmetic is not
    counted (consistent with the v2/v3 contract, which never counted
    interpolation either).
    """
    h_p3, w_p3 = h_p2 // 2, w_p2 // 2
    h_p4, w_p4 = h_p2 // 4, w_p2 // 4
    n_p2 = d * h_p2 * w_p2

    m_c2 = h_p2 * w_p2 * d * c2_ch
    m_p3 = h_p3 * w_p3 * d * p3_ch
    m_p4 = h_p4 * w_p4 * d * p4_ch

    # Separable 1D Gaussian with single reflect pad:
    # Horizontal conv runs on (H + 2r) x W, vertical conv runs on H x W
    r1, k1 = 2, 5
    r2, k2 = 4, 9
    m_gauss_1 = d * (h_p2 + 2 * r1) * w_p2 * k1 + n_p2 * k1
    m_gauss_2 = d * (h_p2 + 2 * r2) * w_p2 * k2 + n_p2 * k2
    m_gauss = m_gauss_1 + m_gauss_2

    m_band_proj = 3 * (h_p3 * w_p3 * d * (4 * d))
    m_band_ref = 3 * (h_p3 * w_p3 * (d * 9 + d * d))
    m_dilated = 3 * (h_p3 * w_p3 * d * 9)
    m_ctx_gate = h_p3 * w_p3 * 3 * d
    # DySample offset heads: 1x1 conv with FIXED out-channels
    # groups*2*scale^2 = 4*2*4 = 32 (independent of d)
    m_upsample = (h_p3 * w_p3 + h_p4 * w_p4) * d * 32
    # Hybrid router correction: 1x1 conv d -> 3 at stride-8; inactive and
    # uncounted under learned_router (or when router_correction=False).
    m_router_corr = (
        h_p3 * w_p3 * 3 * d if (router_correction and not learned_router) else 0
    )
    # Legacy heavy router (only under learned_router flag)
    m_routing = h_p3 * w_p3 * (d * 9 + 4 * d) if learned_router else 0
    m_detail = h_p3 * w_p3 * (d * 9 + d * d)
    m_struct = h_p3 * w_p3 * (d * 25 + d * d)
    m_recon = h_p3 * w_p3 * (4 * d) * d

    total_macs = (
        m_c2
        + m_p3
        + m_p4
        + m_gauss
        + m_band_proj
        + m_band_ref
        + m_dilated
        + m_ctx_gate
        + m_upsample
        + m_router_corr
        + m_routing
        + m_detail
        + m_struct
        + m_recon
    )
    return {
        "c2_proj": float(m_c2),
        "p3_proj": float(m_p3),
        "p4_proj": float(m_p4),
        "gaussian_separable_single_pad": float(m_gauss),
        "band_projections": float(m_band_proj),
        "band_refinements": float(m_band_ref),
        "dilated_context": float(m_dilated),
        "context_gate": float(m_ctx_gate),
        "dynamic_upsamplers": float(m_upsample),
        "router_correction": float(m_router_corr),
        "routing_controller": float(m_routing),
        "detail_refinement": float(m_detail),
        "structural_refinement": float(m_struct),
        "reconstruction_proj": float(m_recon),
        "total_macs": float(total_macs),
        "total_gmacs": float(total_macs) / 1e9,
        "total_gflops": 2.0 * float(total_macs) / 1e9,
    }


__all__ = [
    "ABLATION_FLAGS",
    "CONTROL_PARAM_SUFFIXES",
    "BandBranch",
    "DilatedContextBranch",
    "DySampleUp",
    "GauSS",
    "LearnableGaussianScaleSpace",
    "RefinementBranch",
    "active_ablation_flags",
    "collect_gauss_control_params",
    "collect_gauss_router_zloss",
    "compute_gauss_analytical_macs",
    "configure_gauss_control_param_group",
    "is_gauss_control_param_name",
    "mode_unused_param_names",
    "remap_v3_to_v31_state_dict",
    "remap_v31_to_v32_state_dict",
]
