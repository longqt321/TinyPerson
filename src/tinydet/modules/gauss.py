"""GauSS v3.2 — Gaussian Scale-Space Feature Refinement for YOLO26n-P2.

Single-file revision of the GauSS v3 module. Drop-in replacement for
gauss.py: class name (GauSS), constructor signature and forward signature
are unchanged, so existing YAML registration keeps working.
Only this file is touched — no trainer / YAML changes required.

v3.2 changelog (mechanism-repair revision, driven by the 150e diagnosis):

 1. WARM-STARTED GATES (replaces the v3.1 "EXACT ReZero identity" item).
    Root-cause finding of the 150e campaign: with y = x0 + alpha*z(sigma)
    + gamma*b_h(sigma) and x0 independent of sigma, alpha == gamma == 0
    at init forces dL/dsigma == 0 EXACTLY at step 1 — and the measured
    sigma drift after 150 real epochs was only 3.5–5% of its allowed
    range (the scale-space mechanism was effectively frozen). Exact
    identity at init and a nonzero sigma gradient are mathematically
    incompatible, so the identity property is dropped BY DESIGN:
    - alpha_raw   init = atanh(2 * 0.10)  -> alpha   = 0.10
    - gamma_in    init = atanh(2 * 0.05)  -> gamma_in  = 0.05
    - gamma_out   init = atanh(2 * 0.05)  -> gamma_out = 0.05
    - gamma_mid   init = atanh(2 * 0.05)  -> gamma_mid = 0.05 (new)
    At init the output is a SMALL perturbation of x0 (residual RMS
    ~0.1 * RMS(x0)), not a bitwise identity; the self-test now asserts
    the sigma gradient and output sensitivity instead of the identity.

 2. MID-BAND DIRECT ANCHOR (new gate `gamma_mid_raw`). v3.1 gave sigma1
    a short gradient path (gamma_out * b_h, where b_h = x0 - l1 depends
    only on sigma1) but sigma2 only a deep indirect path (through the
    band branches, refinement stack and alpha). The new term
    gamma_mid * b_m (b_m = l1 - l2 depends on BOTH sigma1 and sigma2)
    gives sigma2 the same one-hop path to the output. Ablation hygiene:
    zeroed under `no_mid_band` and `no_direct_anchor` like b_h.

 3. Sigma re-parameterization: sigmoid-bounded -> tanh-centered.
    sigma1 = 0.65 + 0.25*tanh(raw) in [0.4, 0.9]; sigma2 = 1.45 +
    0.35*tanh(raw) in [1.1, 1.8] (same ranges and init values as v3.1,
    but d(sigma)/d(raw) doubles from 0.125/0.175 to 0.25/0.35 and the
    sigmoid saturation near the bounds is removed). v3.1 checkpoints
    convert with `remap_v31_to_v32_state_dict` (raw_v32 = raw_v31 / 2;
    gate raws are unchanged because both parameterizations map raw=0 to
    the same gate and remap preserves gate VALUES).

 4. Control-group learning-rate multiplier. `configure_gauss_control_
    param_group(..., lr_mult=...)` scales the dedicated `control` group
    (132 scalars, weight_decay=0) relative to the base LR. The sigma
    gradient is intrinsically ~1e-4-scale; with lr_mult=3 the control
    scalars can move on the same epoch timescale as the conv weights.
    The trainer passes `gauss_control_lr_mult` from the train config
    (default 3.0; set 1.0 to reproduce the v3.1 optimizer behaviour).

 5. Diagnostics: added `gaussian/sigma1_min/max`, `gaussian/sigma2_min/
    max` (per-channel spread visibility) and `contribution/mid_anchor_
    share`; `gates/gamma_mid` added to the gates block.

 6. Router correction becomes trainable in practice: with alpha != 0 at
    init, dL/da != 0, so the zero-init router correction now receives a
    real gradient from step 1 (in v3.1 it was gradient-dead at init for
    the same reason as sigma). The energy prior stays detached BY DESIGN
    (it is a fixed input-statistics prior; the learned correction is the
    trainable part) — see v3.1 item 3 for the ST-MoE z-loss rationale.

v3.1 changelog (each item traceable to review findings + arXiv evidence):

 1. EXACT ReZero identity at init. Gates re-parametrized from
    `0.5 * sigmoid(raw)` (v3: init -6 -> alpha ~ 0.0012) to
    `0.5 * tanh(raw)` with raw = 0, so alpha = gamma = 0 EXACTLY and
    y_p2 == baseline x0 bitwise at epoch 0 (verified in the self-test).
    Gate gradient at init rises from 0.0012 to 0.5 (~400x), removing the
    sigmoid saturation stall while keeping the bounded-range safety.
    [ReZero: arXiv:2003.04887]

 2. Content-aware upsampling. Both nearest-neighbour `F.interpolate`
    calls (P3->P2 and P4->P3 fusion) are replaced by a DySample-style
    point-sampling upsampler with zero-init offsets, i.e. EXACTLY equal to
    `F.interpolate(mode="bilinear", align_corners=False)` at init
    (verified in the self-test) and adaptively moved by gradients after.
    Rationale: nearest fusion at the high/low-band interface is the
    aliasing source identified by FreqFusion (arXiv:2408.12879);
    DySample is ultra-light (arXiv:2308.15085).

 3. Hybrid band router (default). Parameter-free log-energy prior measured
    on the RAW bands b_h/b_m/b_l at stride-4 (pre-BN physical magnitude,
    pooled to stride-8), plus a zero-init 1x1 learned correction inside
    the softmax. At init the correction is 0, so routing is EXACTLY the
    v3 energy-proportional rule; afterwards the model can adapt without
    the collapse risk, controlled by an ST-MoE router z-loss (Eq. 5 of
    arXiv:2202.08906 — squared log-sum-exp of the correction logits)
    exposed via `GauSS.router_z_loss()` (suggest loss weight 1e-3;
    model-wide helper: `collect_gauss_router_zloss(model)`).
    [ST-MoE: arXiv:2202.08906]
    - Pure-v3 energy routing:      TINYDET_ABLATION=no_router_corr
    - v3 heavy learned router:     TINYDET_ABLATION=learned_router
      (the route_dw/route_bn/route_out stack is now computed ONLY under
      this flag — removes ~10.65M dead MACs from the default path)

 4. Ablation hygiene.
    - `no_high_band` now gates BOTH high-band paths (refined r_h AND the
      direct stride-4 anchor gamma_out * b_h). In v3 the direct anchor
      leaked through and biased the ablation (~22% output change).
    - New flag `no_direct_anchor` isolates the stride-4 direct anchor
      from the refined anchor for clean contribution attribution.
    - Band-fusion weights are renormalized over the ACTIVE bands
      (masked log-energies before softmax in the hybrid router;
      uniform-over-active in `no_router` mode), so band ablations no
      longer shrink the fused feature mass.

 5. Split high-band gates: `gamma_in` (refined anchor at stride-8,
    inside x_d) and `gamma_out` (direct anchor at stride-4, in y_p2).
    v3 coupled both roles into one scalar.

 6. Dilated context rates {1, 3, 5} (was {1, 2, 3}) — module-owned
    context RF at stride-8 grows from 7x7 to 11x11 px, addressing the
    small effective-receptive-field regime for the 60%-large-object
    label distribution. [ERF: arXiv:1701.04128]
    Configurable via `GauSS(..., dilations=(1, 3, 5))`.

 7. fp32 routing math (energies, log-softmax, z-loss) for AMP safety.

 8. Review-pass corrections (v3.1.1):
    - `no_router` ablation: uniform weights renormalized over ACTIVE
      bands (was fixed 1/3 per band incl. ablated ones — shrank the
      fused feature mass, contradicting item 4).
    - Analytical MACs: `learned_router` mode no longer counts the
      inactive router-correction conv (1,051,975,680 -> 1,049,518,080);
      DySample offset-head MACs use the fixed 32 out-channels (correct
      for any hidden width d, not just d=32).
    - DySample base sampling grid built in fp32, cast to input dtype
      (removes fp16 coordinate drift under AMP).
    - New helper `collect_gauss_router_zloss(model)`; docstring notes for
      strict=False legacy loading and router-flag precedence.

Integration notes (everything stays inside this file):
 - DDP: in default mode the parameters returned by
   `module.unused_param_names()` receive no gradient; either run
   DDP(find_unused_parameters=True) or use the list to build the DDP
   wrapper. In `learned_router` mode only router_corr.* is unused.
 - Optimizer: call `configure_gauss_control_param_group(opt, model)`
   AFTER the trainer builds the optimizer so the control scalars
   (128 sigma + alpha + gamma_in + gamma_out) are excluded from weight
   decay (idempotent; safe on resume).
 - Legacy v3 checkpoints: convert with `remap_v3_to_v31_state_dict(sd)`
   before load_state_dict (sigmoid-space gates -> tanh-space, gamma_h
   -> gamma_in/gamma_out); load with strict=False — the v3.1-only params
   (router_corr.*, dy_up3/dy_up4 .offset.*) keep their constructor
   zero-init, which is exactly the intended state.
 - Router z-loss (hybrid mode): loss = loss + 1e-3 *
   collect_gauss_router_zloss(model). Optional — the correction also
   trains through the mixing weights without it.
 - Router flag precedence: learned_router > no_router > hybrid default.
 - Export: PixelShuffle / PixelUnshuffle / grid_sample map to standard
   ONNX ops (DepthToSpace / GatherND-style sampling); verify the
   reflect-pad path on the target TensorRT version.

Contract summary (input 1280x1280, C2=64, P3=64, P4=128, D=32, B=1):
 - Trainable parameters : 36,879
 - Learned Conv2d layers: 27
 - Control scalars      : 132 (128 sigma + alpha + gamma_in + gamma_out + gamma_mid)
 - Exact MACs (B=1)     : 1,041,326,080 default
                          1,049,518,080 with learned_router
                          (use_saliency adds 819,200, not in contract)
 - vs v3                : +24.58M MACs net (+2.42% module, ~ +0.14% of a
                          YOLO26n forward at 1280 — estimate)
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

# v3.2 warm-started gate values (see docstring item 1). Exact identity at
# init (all gates 0) forces dL/dsigma == 0 at step 1 — measured on the
# real module in the 150e diagnosis — so the gates start slightly open.
GATE_INIT_ALPHA = 0.10
GATE_INIT_GAMMA_IN = 0.05
GATE_INIT_GAMMA_OUT = 0.05
GATE_INIT_GAMMA_MID = 0.05

# v3.2 sigma parameterization (docstring item 3): sigma = center + half*tanh(raw).
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

    Two scales per channel and per axis (v3.2 tanh-centered parameterization,
    d(sigma)/d(raw) = 0.25 / 0.35 at init — twice the v3.1 sigmoid slope):
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
        """Compute bounded sigmas in FP32 (v3.2: tanh-centered, slope 0.25/0.35)."""
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
        with torch.no_grad():
            self.last_diag = {"offset_rms": float(off.float().square().mean().sqrt().item())}

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
    """GauSS v3.1 — Gaussian Scale-Space Feature Refinement.

    ReZero-inspired bounded gates (v3.2: warm-started, see docstring item 1),
    energy-prior routing with bounded learned
    correction, content-aware upsampling, and contamination-free ablations.
    See module docstring for the full v3.1 changelog and integration notes.
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

        # v3.1 hybrid-router correction: zero-init, added to the log-energy
        # prior inside the band softmax; behaviour-identical to pure energy
        # routing at init (ST-MoE-style z-loss available via router_z_loss()).
        self.router_corr = nn.Conv2d(d, 3, kernel_size=1, stride=1, padding=0, bias=True)

        # v3.1 content-aware upsamplers (zero-init == bilinear at init).
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

        # Bounded residual gates: g = 0.5 * tanh(raw). v3.2 warm start
        # (docstring item 1): nonzero init is REQUIRED for a nonzero sigma
        # gradient — with alpha == gamma == 0 the output is sigma-independent
        # and dL/dsigma == 0 exactly at step 1 (measured, 150e diagnosis).
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        # gamma_in : refined high-band anchor at stride-8 (inside x_d)
        self.gamma_in_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        # gamma_out: DIRECT high-band anchor at native stride-4 (in y_p2)
        self.gamma_out_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        # gamma_mid: DIRECT mid-band anchor at native stride-4 (in y_p2);
        # b_m = l1 - l2 depends on BOTH sigma1 and sigma2, giving sigma2 a
        # one-hop gradient path (v3.2 docstring item 2).
        self.gamma_mid_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.act = nn.SiLU(inplace=False)

        # Ablation flags (env var TINYDET_ABLATION); frozen at construction.
        self.ablation_flags: frozenset[str] = active_ablation_flags()

        # Detached diagnostic dictionary for training loggers.
        self.last_diag: dict[str, float] = {}
        # Non-detached router z-loss (ST-MoE); add to the training loss if
        # the hybrid correction is enabled: loss += 1e-3 * module.router_z_loss()
        self._z_loss: Tensor = torch.zeros((), dtype=torch.float32)

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
            # control scalars: sigmas centered (sigma == v3.1 init values),
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
        """Direct mid-band anchor gain = 0.5 * tanh(gamma_mid_raw) (v3.2)."""
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

        # ---- 4.4 Band Routing (v3.1: energy prior on RAW bands + zero-init correction) ----
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
            # v3.1 default hybrid: softmax(log-energy prior + zero-init correction).
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
        # v3.2 warm start: alpha/gamma != 0 at init -> y_p2 is a SMALL
        # perturbation of x0 (residual RMS ~0.1 * RMS(x0)), and dL/dsigma
        # is nonzero from step 1 (the v3.1 bitwise identity is dropped —
        # it mathematically forced a dead sigma gradient).
        y_p2 = x0 + alpha * z + gamma_out * b_h_eff + gamma_mid * b_m_eff

        # ---- 11. Detached Diagnostics ----
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
# Legacy checkpoint adaptation (v3 sigmoid-space gates -> v3.1 tanh-space)
# --------------------------------------------------------------------------- #
def remap_v3_to_v31_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert a GauSS v3 checkpoint to the v3.1 gate parametrization.

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
    """Convert a GauSS v3.1 checkpoint to the v3.2 parameterization.

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


# --------------------------------------------------------------------------- #
# Self-test (run `python src/tinydet/modules/gauss.py`)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(42)

    module = GauSS(c1=[64, 64, 128], c2=32)

    # 1. Static contract -------------------------------------------------
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    conv_count = sum(1 for m in module.modules() if isinstance(m, nn.Conv2d))
    ctrl_params = collect_gauss_control_params(module)
    ctrl_count = sum(p.numel() for p in ctrl_params)
    assert total_params == 36_879, f"Expected 36,879 params, got {total_params}"
    assert conv_count == 27, f"Expected 27 Conv2d modules, got {conv_count}"
    assert ctrl_count == 132, (
        f"Expected 132 control scalars (128 sigma + alpha + gamma_in + gamma_out + gamma_mid), got {ctrl_count}"
    )

    # 2. Analytical MACs -------------------------------------------------
    mac_info = compute_gauss_analytical_macs()
    assert int(mac_info["total_macs"]) == 1_041_326_080, (
        f"Expected 1,041,326,080 MACs (default), got {int(mac_info['total_macs'])}"
    )
    mac_lr = compute_gauss_analytical_macs(learned_router=True)
    assert int(mac_lr["total_macs"]) == 1_049_518_080, (
        f"Expected 1,049,518,080 MACs (learned_router), got {int(mac_lr['total_macs'])}"
    )
    mac_nr = compute_gauss_analytical_macs(router_correction=False)
    assert int(mac_nr["total_macs"]) == 1_038_868_480, (
        f"Expected 1,038,868,480 MACs (no_router), got {int(mac_nr['total_macs'])}"
    )

    # 3. DySample zero-init == bilinear reference (odd sizes on purpose) --
    xt = torch.randn(2, 32, 13, 17)
    ref_up = F.interpolate(xt, scale_factor=2, mode="bilinear", align_corners=False)
    got_up = module.dy_up3(xt)
    assert got_up.shape == (2, 32, 26, 34), f"Unexpected DySample shape {got_up.shape}"
    up_err = (got_up - ref_up).abs().max().item()
    assert up_err < 1e-5, f"DySample(zero-init) != bilinear: max|d|={up_err}"

    # 4. v3.2 warm start: near-identity output, LIVE sigma gradient --------
    c2_t = torch.randn(2, 64, 80, 80, requires_grad=True)
    p3_t = torch.randn(2, 64, 40, 40, requires_grad=True)
    p4_t = torch.randn(2, 128, 20, 20, requires_grad=True)

    with torch.no_grad():
        x0_ref = module.act(
            module.bn2(module.proj_c2(c2_t) + module.dy_up3(module.proj_p3(p3_t)))
        )
    out_p2 = module([c2_t, p3_t, p4_t])
    assert out_p2.shape == (2, 32, 80, 80), f"Unexpected output shape: {out_p2.shape}"
    # Warm start (v3.2): NOT bitwise identity, but a small perturbation.
    rel = (out_p2 - x0_ref).norm() / x0_ref.norm().clamp_min(1e-12)
    assert 1e-3 < float(rel.detach()) < 0.35, f"warm-start residual out of range: rel={float(rel)}"
    # The core v3.2 contract: sigma receives a gradient AT INIT.
    module.zero_grad(set_to_none=True)
    out_p2 = module([c2_t, p3_t, p4_t])
    out_p2.square().mean().backward()
    sigma_grads = [
        float(p.grad.abs().max())
        for n, p in module.named_parameters() if n.startswith("gaussian.sigma")
    ]
    assert max(sigma_grads) > 1e-8, (
        f"dL/dsigma == 0 at init (v3.1 regression): {sigma_grads}"
    )
    # Output sensitivity to a sigma perturbation (v3.1 measured 0.0 exactly).
    with torch.no_grad():
        base = module([c2_t, p3_t, p4_t]).clone()
        module.gaussian.sigma1x_raw.add_(0.4)
        module.gaussian.sigma1y_raw.add_(0.4)
        pert = module([c2_t, p3_t, p4_t])
        sens = float((pert - base).abs().max())
        module.gaussian.sigma1x_raw.sub_(0.4)
        module.gaussian.sigma1y_raw.sub_(0.4)
    assert sens > 1e-3, f"sigma perturbation has no output effect: {sens}"
    sigma_sens = sens

    # 5. Router correction is behaviour-identical to pure energy at init --
    # zero-init corr -> z-loss equals (log 3)^2 exactly (ST-MoE baseline value)
    zl = float(module._z_loss.item())
    assert abs(zl - (math.log(3.0) ** 2)) < 1e-6, f"z-loss at init: {zl}"
    diag0 = module.last_diag
    e_sum = (
        diag0["routing/energy_high_mean"]
        + diag0["routing/energy_mid_mean"]
        + diag0["routing/energy_low_mean"]
    )
    assert abs(
        diag0["routing/high_band_mean"]
        + diag0["routing/mid_band_mean"]
        + diag0["routing/low_band_mean"]
        - 1.0
    ) < 1e-5, "band weights must sum to 1"
    assert torch.allclose(
        collect_gauss_router_zloss(module), module.router_z_loss()
    ), "z-loss helper mismatch"

    # 6. Gradients: full coverage WITHOUT manual warming -------------------
    # (v3.1 required warming alpha_raw because the gates were zero-init; in
    # v3.2 the warm start makes every parameter's gradient live at init.)
    module.zero_grad(set_to_none=True)
    out_p2 = module([c2_t, p3_t, p4_t])
    out_p2.square().mean().backward()

    expected_unused = {
        "route_dw.weight",
        "route_bn.weight",
        "route_bn.bias",
        "route_out.weight",
        "route_out.bias",
        "saliency_gate.weight",
        "saliency_gate.bias",
    }
    for name, p in module.named_parameters():
        if name in expected_unused:
            assert p.grad is None, f"{name} should be graph-inactive in default mode"
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad on {name}"
            assert float(p.grad.abs().max().item()) > 0.0, f"Zero grad on {name}"
    assert set(module.unused_param_names()) == expected_unused

    # 7. Optimizer control group -----------------------------------------
    opt = torch.optim.AdamW(
        [{"params": list(module.parameters()), "weight_decay": 0.01, "param_group": "weight"}],
        lr=1e-3,
    )
    configure_gauss_control_param_group(opt, module, base_lr=1e-3)
    diag = module.get_diagnostics(optimizer=opt)
    assert diag["optimizer/control_weight_decay"] == 0.0
    assert diag["optimizer/control_lr"] == 1e-3
    assert diag["routing/mixing_std"] > 0.0, "Spatial mixing gate has zero spatial variance!"
    assert diag["gates/alpha"] > 0.0  # warm-started at init (v3.2)
    # lr_mult=3 scales the control group (v3.2 item 4)
    configure_gauss_control_param_group(opt, module, base_lr=1e-3, lr_mult=3.0)
    assert module.get_diagnostics(optimizer=opt)["optimizer/control_lr"] == 3e-3

    # 8. Ablation regression: no_high_band gates BOTH high-band paths -----
    os.environ["TINYDET_ABLATION"] = "no_high_band"
    m_abl = GauSS(c1=[64, 64, 128], c2=32).eval()
    with torch.no_grad():
        m_abl.gamma_out_raw.fill_(2.0)
        y_a = m_abl([c2_t.detach(), p3_t.detach(), p4_t.detach()])
        m_abl.gamma_out_raw.fill_(-100.0)
        y_b = m_abl([c2_t.detach(), p3_t.detach(), p4_t.detach()])
    leak = (y_a - y_b).abs().max().item()
    assert leak < 1e-6, f"no_high_band leaks the direct anchor: max|dy|={leak}"
    del os.environ["TINYDET_ABLATION"]

    # ...while in default mode the direct anchor MUST affect the output ---
    m_def = GauSS(c1=[64, 64, 128], c2=32).eval()
    with torch.no_grad():
        m_def.gamma_out_raw.fill_(2.0)
        y_a = m_def([c2_t.detach(), p3_t.detach(), p4_t.detach()])
        m_def.gamma_out_raw.fill_(-100.0)
        y_b = m_def([c2_t.detach(), p3_t.detach(), p4_t.detach()])
    assert (y_a - y_b).abs().max().item() > 1e-3, "direct anchor has no effect!"

    # 8b. no_router: uniform weights renormalized over ACTIVE bands -------
    os.environ["TINYDET_ABLATION"] = "no_router,no_high_band"
    m_nr = GauSS(c1=[64, 64, 128], c2=32).eval()
    with torch.no_grad():
        m_nr([c2_t.detach(), p3_t.detach(), p4_t.detach()])
    d_nr = m_nr.last_diag
    w_sum = (
        d_nr["routing/high_band_mean"]
        + d_nr["routing/mid_band_mean"]
        + d_nr["routing/low_band_mean"]
    )
    assert abs(w_sum - 1.0) < 1e-6, f"no_router weights sum to {w_sum}"
    assert d_nr["routing/high_band_mean"] == 0.0, "ablated band must get zero weight"
    del os.environ["TINYDET_ABLATION"]

    # 9. Legacy router + saliency mode: full grad coverage except corr ----
    os.environ["TINYDET_ABLATION"] = "learned_router,use_saliency"
    m_leg = GauSS(c1=[64, 64, 128], c2=32)
    with torch.no_grad():
        m_leg.alpha_raw.fill_(1.0)
    out_leg = m_leg([c2_t.detach(), p3_t.detach(), p4_t.detach()])
    out_leg.square().mean().backward()
    for name, p in m_leg.named_parameters():
        if name in ("router_corr.weight", "router_corr.bias"):
            assert p.grad is None, f"{name} should be inactive in learned_router mode"
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad on {name}"
            assert float(p.grad.abs().max().item()) > 0.0, f"Zero grad on {name}"
    assert m_leg.unused_param_names() == ("router_corr.weight", "router_corr.bias")
    del os.environ["TINYDET_ABLATION"]

    # 10. Legacy v3 checkpoint remap --------------------------------------
    legacy = {
        "alpha_raw": torch.tensor([-6.0]),
        "gamma_h_raw": torch.tensor([-5.0]),
        "model.12.gauss.gamma_h_raw": torch.tensor([-3.0]),
    }
    remapped = remap_v3_to_v31_state_dict(legacy)
    tgt_alpha = 0.5 * torch.sigmoid(torch.tensor(-6.0))
    assert torch.allclose(0.5 * torch.tanh(remapped["alpha_raw"]), tgt_alpha, atol=1e-7)
    tgt_g5 = 0.5 * torch.sigmoid(torch.tensor(-5.0))
    assert torch.allclose(0.5 * torch.tanh(remapped["gamma_in_raw"]), tgt_g5, atol=1e-7)
    assert torch.allclose(0.5 * torch.tanh(remapped["gamma_out_raw"]), tgt_g5, atol=1e-7)
    tgt_g3 = 0.5 * torch.sigmoid(torch.tensor(-3.0))
    assert torch.allclose(
        0.5 * torch.tanh(remapped["model.12.gauss.gamma_in_raw"]), tgt_g3, atol=1e-7
    )
    assert "gamma_h_raw" not in remapped

    # 10b. v3.1 -> v3.2 sigma remap: r_v32 == r_v31 / 2 (exact) -----------
    sd31 = {
        "gaussian.sigma1x_raw": torch.tensor([0.8, -0.4]),
        "gaussian.sigma2y_raw": torch.tensor([1.2]),
        "alpha_raw": torch.tensor([0.3616]),
        "conv.weight": torch.tensor([1.0]),
    }
    sd32 = remap_v31_to_v32_state_dict(sd31)
    assert torch.allclose(sd32["gaussian.sigma1x_raw"], sd31["gaussian.sigma1x_raw"] / 2)
    assert torch.allclose(sd32["gaussian.sigma2y_raw"], sd31["gaussian.sigma2y_raw"] / 2)
    # gate raws are version-invariant (same gate value -> same raw)
    assert torch.equal(sd32["alpha_raw"], sd31["alpha_raw"])
    assert torch.equal(sd32["conv.weight"], sd31["conv.weight"])
    # and the remap preserves the actual sigma VALUES
    m32 = LearnableGaussianScaleSpace(2)
    with torch.no_grad():
        m32.sigma1x_raw.copy_(sd32["gaussian.sigma1x_raw"])
        m32.sigma2y_raw.copy_(sd32["gaussian.sigma2y_raw"])
    s32 = m32.compute_sigmas()
    # v3.1 formula restated locally (the class now implements v3.2)
    v31_s1x = 0.4 + 0.5 * torch.sigmoid(sd31["gaussian.sigma1x_raw"])
    v31_s2y = 1.1 + 0.7 * torch.sigmoid(sd31["gaussian.sigma2y_raw"])
    assert torch.allclose(s32[0], v31_s1x, atol=1e-6), "sigma1 value drift in remap"
    assert torch.allclose(s32[3], v31_s2y, atol=1e-6), "sigma2 value drift in remap"

    # 10c. Control group is optimizable end-to-end: 200 Adam steps on a
    # SYNTHETIC objective (MSE to a random target). This only proves the
    # optimizer wiring moves the control scalars — it says nothing about
    # whether sigma is useful for detection (see scripts/gauss_probe.py for
    # the real-loss identifiability probes).
    torch.manual_seed(0)
    m_learn = GauSS(c1=[64, 64, 128], c2=32)
    m_learn.zero_grad(set_to_none=True)
    opt_l = torch.optim.Adam(collect_gauss_control_params(m_learn), lr=3e-3)
    tgt = torch.randn(2, 32, 80, 80)
    start = {
        n: p.detach().clone()
        for n, p in m_learn.named_parameters() if n.startswith("gaussian.sigma")
    }
    for _ in range(200):
        opt_l.zero_grad(set_to_none=True)
        loss_l = F.mse_loss(m_learn([c2_t.detach(), p3_t.detach(), p4_t.detach()]), tgt)
        loss_l.backward()
        opt_l.step()
    drift = max(
        float((p.detach() - start[n]).abs().max())
        for n, p in m_learn.named_parameters() if n.startswith("gaussian.sigma")
    )
    assert drift > 0.05, f"sigma barely moves under Adam: |dsigma_raw|max={drift:.4f}"

    print(
        f"[PASS] GauSS v3.2 verified: params={total_params:,} | convs={conv_count} | "
        f"ctrl={ctrl_count} | MACs={mac_info['total_gmacs']:.8f} GMAC "
        f"({mac_info['total_gflops']:.8f} GFLOPs) | warm start (rel={float(rel):.3f}) | "
        f"dL/dsigma@init={max(sigma_grads):.2e}, sens={sigma_sens:.2e} | "
        f"sigma drift/200 steps={drift:.3f} | DySample==bilinear (err={up_err:.2e}) | "
        f"ablations contamination-free"
    )
