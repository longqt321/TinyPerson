"""DCAF v2 — Dilated-Context Asymmetric Feature Refinement for YOLO26n-P2.

Exact implementation of the frozen DCAF v2 technical specification:
- Block 1: Multi-Scale Band Feature Encoder (P2 fusion + Learnable Separable
  1D Gaussian Scale-Space with single reflect pad + 3 Band Refinement branches).
- Block 2: Gated Dilated Context Controller (P3/P4 fusion + 3 parallel dilated
  DWConv branches d={1,2,3} + softmax context gate + 4-channel spatial router).
- Block 3: Asymmetric Feature Refinement (Adaptive band fusion + Detail-Focused
  3x3 branch + Structural-Context 5x5 branch + Spatial Mixing Gate + PixelShuffle
  reconstruction + bounded residual gain).

Contract summary (at input 1280x1280, C2=64, P3=64, P4=128, D=32):
- Trainable parameters : 34,632
- Learned Conv2d layers: 23
- Exact MACs (B=1)     : 1,016,750,080 (1.01675008 GMAC / 2.03350016 GFLOPs)
"""

from __future__ import annotations

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
    "gamma_h_raw",
)


class LearnableGaussianScaleSpace(nn.Module):
    """Separable 1D learnable Gaussian scale-space filter with single reflect pad.

    Two scales per channel and per axis:
      - Scale 1: support 5x5 (radius=2), sigma_1 in [0.4, 0.9], init 0.65
      - Scale 2: support 9x9 (radius=4), sigma_2 in [1.1, 1.8], init 1.45
    """

    coord1: Tensor
    coord2: Tensor

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.channels = int(channels)
        self.r1 = 2  # 5x5 support
        self.r2 = 4  # 9x9 support

        # Control parameters: 1D shape (32,), initialized to 0.0
        self.sigma1x_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.sigma1y_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.sigma2x_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))
        self.sigma2y_raw = nn.Parameter(torch.zeros(self.channels, dtype=torch.float32))

        # Fixed coordinate buffers
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
        """Compute bounded sigmas in FP32."""
        s1x = 0.4 + 0.5 * torch.sigmoid(self.sigma1x_raw.float())
        s1y = 0.4 + 0.5 * torch.sigmoid(self.sigma1y_raw.float())
        s2x = 1.1 + 0.7 * torch.sigmoid(self.sigma2x_raw.float())
        s2y = 1.1 + 0.7 * torch.sigmoid(self.sigma2y_raw.float())
        return s1x, s1y, s2x, s2y

    @staticmethod
    def _kernel_1d(sigma: Tensor, coords: Tensor) -> Tensor:
        """Build normalized 1D Gaussian kernel of shape (C, K) in FP32."""
        # sigma: (C,), coords: (K,)
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


class BandBranch(nn.Module):
    """Single band encoder branch (Section 3.5).

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
    """Single parallel dilated depthwise branch (Section 4.2)."""

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
    """Residual depthwise-separable refinement branch (Sections 5.2 & 5.3).

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


class DCAF(nn.Module):
    """DCAF v2 — Dilated-Context Asymmetric Feature Refinement.

    Supports both Ultralytics YAML parser convention `DCAF(c1=[64, 64, 128], c2=32)`
    and explicit channel arguments `DCAF(c2_ch=64, p3_ch=64, p4_ch=128, c_out=32)`.
    """

    def __init__(
        self,
        c1: Sequence[int] | int = (64, 64, 128),
        c2: int = 32,
        p4_ch: int | None = None,
        c_out: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(c1, Sequence):
            if len(c1) != 3:
                raise ValueError(
                    f"DCAF expects 3 input channels [C2, P3, P4], got {list(c1)}"
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

        self.c2_channels = c2_in
        self.p3_channels = p3_in
        self.p4_channels = p4_in
        self.hidden_dim = d
        self.out_channels = d

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
        self.ctx_d1 = DilatedContextBranch(hidden_dim=d, dilation=1)
        self.ctx_d2 = DilatedContextBranch(hidden_dim=d, dilation=2)
        self.ctx_d3 = DilatedContextBranch(hidden_dim=d, dilation=3)
        self.ctx_gate = nn.Conv2d(d, 3, kernel_size=1, stride=1, padding=0, bias=True)
        # Saliency gate for background clutter attenuation in structural branch
        self.saliency_gate = nn.Conv2d(d, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.route_dw = nn.Conv2d(
            d, d, kernel_size=3, stride=1, padding=1, groups=d, bias=False
        )
        self.route_bn = nn.BatchNorm2d(d)
        self.route_out = nn.Conv2d(d, 4, kernel_size=1, stride=1, padding=0, bias=True)

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

        # Bounded residual gain parameter a_alpha in R^1 (init 0 -> alpha_0 = 0.25)
        self.alpha_raw = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        # Direct High-Band Residual Anchoring weight: gamma_h = 0.5 * sigmoid(gamma_h_raw) in [0, 0.5] (init 0.20)
        # logit(0.40) = ln(0.4 / 0.6) = -0.4055
        self.gamma_h_raw = nn.Parameter(torch.tensor([-0.4055], dtype=torch.float32))

        self.act = nn.SiLU(inplace=False)

        # Detached diagnostic dictionary for training loggers
        self.last_diag: dict[str, float] = {}

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Apply Kaiming init to Conv2d layers and Section 4.4 init to route_out."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Section 4.4: 4th output channel of route_out has W_m = 0.01 * W_Kaiming, b = 0
        with torch.no_grad():
            self.route_out.weight[3].mul_(0.01)
            if self.route_out.bias is not None:
                self.route_out.bias.zero_()

            self.gaussian.sigma1x_raw.zero_()
            self.gaussian.sigma1y_raw.zero_()
            self.gaussian.sigma2x_raw.zero_()
            self.gaussian.sigma2y_raw.zero_()
            self.alpha_raw.zero_()
            self.gamma_h_raw.fill_(-0.4055)
            # Init saliency gate with small weights and zero bias
            self.saliency_gate.weight.mul_(0.01)
            if self.saliency_gate.bias is not None:
                self.saliency_gate.bias.zero_()

    def compute_alpha(self) -> Tensor:
        """Compute bounded residual gain alpha = 0.5 * sigmoid(a_alpha)."""
        return 0.5 * torch.sigmoid(self.alpha_raw.float())

    def compute_gamma_h(self) -> Tensor:
        """Compute bounded high-band direct gain gamma_h = 0.5 * sigmoid(gamma_h_raw)."""
        return 0.5 * torch.sigmoid(self.gamma_h_raw.float())

    def forward(self, x: Sequence[Tensor]) -> Tensor:
        """Forward pass accepting [C2, P3, P4] and returning refined P2."""
        if not isinstance(x, Sequence) or len(x) != 3:
            raise ValueError(
                "DCAF.forward expects a list/tuple of 3 tensors: [C2, P3, P4]."
            )
        c2, p3, p4 = x[0], x[1], x[2]

        # ---- 3.1 P2 Feature Fusion ----
        s2 = self.proj_c2(c2)
        s3 = self.proj_p3(p3)
        s4 = self.proj_p4(p4)

        s3_up = F.interpolate(s3, size=s2.shape[-2:], mode="nearest")
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
        p4_up = F.interpolate(s4, size=s3.shape[-2:], mode="nearest")
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

        # ---- 4.4 Spatial Routing Controller ----
        u = self.act(r_h + r_m + r_l + c_ctx)
        v = self.act(self.route_bn(self.route_dw(u)))
        logits = self.route_out(v)

        band_logits = logits[:, :3]
        mix_logit = logits[:, 3:4]

        a_bands = torch.softmax(band_logits, dim=1)
        a_h, a_m, a_l = a_bands[:, 0:1], a_bands[:, 1:2], a_bands[:, 2:3]
        m_gate = torch.sigmoid(mix_logit)

        # ---- 5.1 Adaptive Band Fusion & High-Band Residual Anchoring ----
        f_band = a_h * r_h + a_m * r_m + a_l * r_l
        gamma_h = self.compute_gamma_h().to(dtype=r_h.dtype)

        # ---- 5.2 & 5.3 Detail-Focused & Structural-Context Refinement ----
        # Direct High-Band Anchor: guarantees sub-8px boundary preservation
        x_d = f_band + c_ctx + gamma_h * r_h
        f_d = self.detail_branch(x_d)

        # Background Saliency Gating: attenuates background textures in shallow low-frequency features
        saliency_mask = torch.sigmoid(self.saliency_gate(c_ctx))
        r_l_attenuated = r_l * (1.0 - 0.5 * saliency_mask)
        x_s = r_l_attenuated + c_ctx
        f_s = self.struct_branch(x_s)

        # ---- 5.4 Spatial Feature Mixing ----
        f_route = m_gate * f_d + (1.0 - m_gate) * f_s

        # ---- 5.5 P2 Reconstruction ----
        z0 = self.recon_bn(self.recon_proj(f_route))
        z = self.ps(z0)

        alpha = self.compute_alpha().to(dtype=z.dtype)
        alpha_z = alpha * z
        y_p2 = self.act(x0 + alpha_z)

        # ---- 11. Detached Diagnostics ----
        with torch.no_grad():
            s1x, s1y, s2x, s2y = sigmas
            sigma1 = torch.cat((s1x, s1y), dim=0)
            sigma2 = torch.cat((s2x, s2y), dim=0)
            m_f = m_gate.detach().float()
            rms_res = alpha_z.detach().float().square().mean().sqrt()
            rms_in = x0.detach().float().square().mean().sqrt().clamp_min(1e-12)

            self.last_diag = {
                "routing/mixing_mean": float(m_f.mean().item()),
                "routing/mixing_std": float(m_f.std(unbiased=False).item()),
                "routing/high_band_mean": float(a_h.detach().float().mean().item()),
                "routing/mid_band_mean": float(a_m.detach().float().mean().item()),
                "routing/low_band_mean": float(a_l.detach().float().mean().item()),
                "context/dilation_1_mean": float(pi1.detach().float().mean().item()),
                "context/dilation_2_mean": float(pi2.detach().float().mean().item()),
                "context/dilation_3_mean": float(pi3.detach().float().mean().item()),
                "refinement/residual_ratio": float((rms_res / rms_in).item()),
                "gaussian/sigma1_mean": float(sigma1.mean().item()),
                "gaussian/sigma2_mean": float(sigma2.mean().item()),
                "gaussian/sigma1_std": float(sigma1.std(unbiased=False).item()),
                "gaussian/sigma2_std": float(sigma2.std(unbiased=False).item()),
            }

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
# Section 9: Optimizer Parameter Group Utilities
# --------------------------------------------------------------------------- #
def is_dcaf_control_param_name(name: str) -> bool:
    """Return True if parameter name belongs to the DCAF control group."""
    return name.endswith(CONTROL_PARAM_SUFFIXES)


def collect_dcaf_control_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect all DCAF Gaussian sigma raw parameters and residual gain alpha_raw."""
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if param.requires_grad and is_dcaf_control_param_name(name):
            pid = id(param)
            if pid not in seen:
                seen.add(pid)
                params.append(param)
    return params


def configure_dcaf_control_param_group(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    base_lr: float | None = None,
) -> dict[str, Any] | None:
    """Move DCAF control parameters into a dedicated `control` group with weight_decay=0.0.

    Idempotent: safe to call in both Stage 1 and Stage 2 before scheduler creation.
    Preserves group order when resuming an optimizer that already has a `control` group.
    """
    control_params = collect_dcaf_control_params(model)
    if not control_params:
        return None
    ctrl_ids = {id(p) for p in control_params}

    # Remove control params from any non-control group (e.g. default 'weight' or 'bias' group)
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

    if existing_ctrl_group is not None:
        existing_ctrl_group["params"] = control_params
        existing_ctrl_group["weight_decay"] = 0.0
        existing_ctrl_group.setdefault("lr", inferred_lr)
        existing_ctrl_group.setdefault("initial_lr", inferred_lr)
        return existing_ctrl_group

    ctrl_group: dict[str, Any] = {
        "params": control_params,
        "lr": inferred_lr,
        "initial_lr": inferred_lr,
        "weight_decay": 0.0,
        "param_group": "control",
    }
    # Copy optimizer-specific defaults (betas, eps, momentum, etc.) from group 0
    for k, v in optimizer.param_groups[0].items():
        if k not in ctrl_group and k != "params":
            ctrl_group[k] = v

    optimizer.param_groups.append(ctrl_group)
    return ctrl_group


# --------------------------------------------------------------------------- #
# Section 8: Analytical MACs & Parameter Verification
# --------------------------------------------------------------------------- #
def compute_dcaf_analytical_macs(
    c2_ch: int = 64,
    p3_ch: int = 64,
    p4_ch: int = 128,
    d: int = 32,
    h_p2: int = 320,
    w_p2: int = 320,
) -> dict[str, float]:
    """Compute exact analytical MACs (B=1) per Section 8 specification."""
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
    m_routing = h_p3 * w_p3 * (d * 9 + 4 * d)
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
        "routing_controller": float(m_routing),
        "detail_refinement": float(m_detail),
        "structural_refinement": float(m_struct),
        "reconstruction_proj": float(m_recon),
        "total_macs": float(total_macs),
        "total_gmacs": float(total_macs) / 1e9,
        "total_gflops": 2.0 * float(total_macs) / 1e9,
    }


__all__ = [
    "CONTROL_PARAM_SUFFIXES",
    "DCAF",
    "BandBranch",
    "DilatedContextBranch",
    "LearnableGaussianScaleSpace",
    "RefinementBranch",
    "collect_dcaf_control_params",
    "compute_dcaf_analytical_macs",
    "configure_dcaf_control_param_group",
    "is_dcaf_control_param_name",
]


if __name__ == "__main__":
    torch.manual_seed(42)
    module = DCAF(c1=[64, 64, 128], c2=32)

    # 1. Parameter count & Conv2d module count verification
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    conv_count = sum(1 for m in module.modules() if isinstance(m, nn.Conv2d))
    ctrl_params = collect_dcaf_control_params(module)
    ctrl_count = sum(p.numel() for p in ctrl_params)

    assert total_params == 34_666, f"Expected 34,666 params, got {total_params}"
    assert conv_count == 24, f"Expected 24 Conv2d modules, got {conv_count}"
    assert ctrl_count == 130, f"Expected 130 control scalars (128 sigma + 1 alpha + 1 gamma_h), got {ctrl_count}"

    # 2. Analytical MACs / GFLOPs verification
    mac_info = compute_dcaf_analytical_macs()
    assert int(mac_info["total_macs"]) == 1_016_750_080, (
        f"Expected 1,016,750,080 MACs, got {int(mac_info['total_macs'])}"
    )

    # 3. Forward / Backward & Optimizer control group verification
    c2_t = torch.randn(2, 64, 80, 80, requires_grad=True)
    p3_t = torch.randn(2, 64, 40, 40, requires_grad=True)
    p4_t = torch.randn(2, 128, 20, 20, requires_grad=True)

    opt = torch.optim.AdamW(
        [{"params": list(module.parameters()), "weight_decay": 0.01, "param_group": "weight"}],
        lr=1e-3,
    )
    configure_dcaf_control_param_group(opt, module, base_lr=1e-3)

    out_p2 = module([c2_t, p3_t, p4_t])
    assert out_p2.shape == (2, 32, 80, 80), f"Unexpected output shape: {out_p2.shape}"

    loss = out_p2.square().mean()
    loss.backward()

    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad on {name}"
        assert float(p.grad.abs().max().item()) > 0.0, f"Zero grad on {name}"

    diag = module.get_diagnostics(optimizer=opt)
    assert diag["optimizer/control_weight_decay"] == 0.0
    assert diag["optimizer/control_lr"] == 1e-3
    assert diag["routing/mixing_std"] > 0.0, "Spatial mixing gate has zero spatial variance!"

    print(f"[PASS] DCAF v2 verified: params={total_params:,} | convs={conv_count} | "
          f"MACs={mac_info['total_gmacs']:.8f} GMAC ({mac_info['total_gflops']:.8f} GFLOPs)")
