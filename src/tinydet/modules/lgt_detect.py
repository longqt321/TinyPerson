"""Local Gaussian Transport Detection Head (``LGTDetect``) for NMS-free tiny object detection.

Mathematical formulation
------------------------
In NMS-free detectors (YOLO26: ``reg_max=1``, YOLOv10: ``reg_max=16``, ``end2end=True``), each grid
cell ``u = (y, x)`` on FPN level ``s`` predicts decoded stride-normalized distances ``(l, t, r, b)``
and class logits ``s_cls``. Standard ``one2one`` heads suffer two failure modes on tiny crowded
objects:
1. Single-cell box regression variance collapses high-IoU precision (``AP75``).
2. Appearance-only ``one2one_cv3`` classification is geometry-blind, falsely suppressing the
   second object in close tiny pairs or cross-class neighbors in multi-class aerial scenes.

``LGTDetect`` models each cell's decoded box expectation as a diagonal 2D Gaussian
``G(u) = N(mu(u), diag(sigma(u)^2))`` with ``mu = ((r - l)/2, (b - t)/2)`` and
``sigma = ((l + r)/2, (t + b)/2)``, and performs:

1. **Coordinate-Frame Transport (Proposition 1):**
   Across the ``3x3`` spatial neighborhood ``delta_k in {-1, 0, +1}^2`` plus the co-located parent
   cell interpolated from FPN level ``s+1``, neighbor Gaussians are transported into cell ``u``'s
   local coordinate frame so identical physical boxes satisfy ``tilde_mu_k == mu`` and
   ``tilde_sigma_k == sigma`` exactly.

2. **Hyperparameter-Free Scale-Invariant Wasserstein Distance (Proposition 2):**
   ``D_k(u) = sum_{a in {x,y}} ((tilde_mu_ka - mu_a)^2 + (tilde_sigma_ka - sigma_a)^2) / (sigma_a^2 + tilde_sigma_ka^2 + eps)``
   Unlike fixed-constant NWD, ``D_k(lambda * B1, lambda * B2) == D_k(B1, B2)`` for all ``lambda > 0``.

3. **Class-Aware Local Wasserstein Barycenter & Differentiable Dominance Gate (Proposition 3):**
   - Multi-class safety: only neighbors sharing the same dominant class ``hat_c(u + delta_k) == hat_c(u)``
     participate in transport, preventing large vehicles from shifting or suppressing nearby pedestrians.
   - Box branch: refines ``(mu, sigma)`` for both DFL-free (``reg_max=1``) and DFL (``reg_max>1``)
     heads via ``w_k = softmax(gamma * log(sigmoid(s_k)) - tau_box * D_k)``.
   - Classification branch: computes differentiable NMS dominance ``M_dom`` and consensus ``M_sup``
     to modulate ``one2one`` class logits via a zero-initialized ``1x1`` residual bottleneck.

4. **Training-Only Dense Gaussian Field Distillation:**
   During training, ``one2many`` distills its dense Gaussian box field into ``one2one_cv2`` weighted
   by foreground confidence. At ``Detect.fuse()``, ``one2many`` is dropped with zero runtime cost.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors

OFFSETS_3X3: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)
NUM_SPATIAL = len(OFFSETS_3X3)  # 9
NUM_NEIGHBORS = (NUM_SPATIAL - 1) + 1  # 8 spatial + 1 parent = 9


def xyxy_to_gaussian(boxes: Tensor, eps_sigma: float = 1e-3) -> tuple[Tensor, Tensor]:
    """Convert ``(..., 4)`` ``[x1, y1, x2, y2]`` boxes into ``(mu, sigma)`` Gaussian parameters."""
    mu = 0.5 * (boxes[..., 2:] + boxes[..., :2])
    sigma = (0.5 * (boxes[..., 2:] - boxes[..., :2])).clamp(min=eps_sigma)
    return mu, sigma


def scale_invariant_wasserstein_xyxy(
    box1: Tensor,
    box2: Tensor,
    eps: float = 1e-6,
    eps_sigma: float = 1e-3,
) -> Tensor:
    """Self-normalized 2D Gaussian Wasserstein distance ``D(B1, B2) >= 0`` for ``(..., 4)`` xyxy boxes."""
    mu1, sig1 = xyxy_to_gaussian(box1.float(), eps_sigma=eps_sigma)
    mu2, sig2 = xyxy_to_gaussian(box2.float(), eps_sigma=eps_sigma)
    num = (mu1 - mu2).square() + (sig1 - sig2).square()
    den = sig1.square() + sig2.square() + eps
    return (num / den).sum(dim=-1)


def wasserstein_similarity_xyxy(box1: Tensor, box2: Tensor, kappa: float = 1.0) -> Tensor:
    """Bounded rational similarity ``1 / (1 + kappa * D)`` in ``(0, 1]`` with non-vanishing gradients."""
    dist = scale_invariant_wasserstein_xyxy(box1, box2)
    return 1.0 / (1.0 + kappa * dist)


def nwd_similarity_xyxy(
    box1: Tensor,
    box2: Tensor,
    constant_c: float = 12.8,
    eps: float = 1e-6,
    eps_sigma: float = 1e-3,
) -> Tensor:
    """Published Normalized Wasserstein Distance (Wang et al., 2021) ``exp(-sqrt(W_2^2) / C)`` in pixel units."""
    mu1, sig1 = xyxy_to_gaussian(box1.float(), eps_sigma=eps_sigma)
    mu2, sig2 = xyxy_to_gaussian(box2.float(), eps_sigma=eps_sigma)
    w2_sq = ((mu1 - mu2).square() + (sig1 - sig2).square()).sum(dim=-1)
    return torch.exp(-torch.sqrt(w2_sq + eps) / max(float(constant_c), 1e-3))


def ltrb_to_gaussian(ltrb: Tensor, eps_sigma: float = 0.05) -> tuple[Tensor, Tensor]:
    """Convert ``(B, 4, H, W)`` stride-normalized ``[l, t, r, b]`` into ``(mu, sigma)`` ``(B, 2, H, W)``."""
    left, top, right, bottom = ltrb.unbind(dim=1)
    mu = torch.stack((0.5 * (right - left), 0.5 * (bottom - top)), dim=1)
    sigma = torch.stack(
        ((0.5 * (left + right)).clamp(min=eps_sigma), (0.5 * (top + bottom)).clamp(min=eps_sigma)),
        dim=1,
    )
    return mu, sigma


def gaussian_to_ltrb(mu: Tensor, sigma: Tensor) -> Tensor:
    """Convert ``(mu, sigma)`` ``(B, 2, H, W)`` back to ``(B, 4, H, W)`` ``[l, t, r, b]``."""
    mu_x, mu_y = mu.unbind(dim=1)
    sig_x, sig_y = sigma.unbind(dim=1)
    return torch.stack((sig_x - mu_x, sig_y - mu_y, sig_x + mu_x, sig_y + mu_y), dim=1)


class SpatialShiftStencil(nn.Module):
    """Zero-FLOP ``3x3`` sliding-window ``unfold`` stencil extracting ``OFFSETS_3X3`` taps and valid masks."""

    tap_order: Tensor
    dx: Tensor
    dy: Tensor

    def __init__(self, channels: int = 5) -> None:
        super().__init__()
        self.channels = int(channels)
        order = torch.tensor([(dy + 1) * 3 + (dx + 1) for dx, dy in OFFSETS_3X3], dtype=torch.long)
        self.register_buffer("tap_order", order, persistent=False)

        dx_t = torch.tensor([dx for dx, _ in OFFSETS_3X3], dtype=torch.float32).view(1, NUM_SPATIAL, 1, 1)
        dy_t = torch.tensor([dy for _, dy in OFFSETS_3X3], dtype=torch.float32).view(1, NUM_SPATIAL, 1, 1)
        self.register_buffer("dx", dx_t, persistent=False)
        self.register_buffer("dy", dy_t, persistent=False)
        self._valid_cache: dict[tuple[int, int, torch.device], Tensor] = {}

    def __getstate__(self):
        state = dict(super().__getstate__())
        state["_valid_cache"] = {}
        return state

    def _get_valid_mask(self, height: int, width: int, device: torch.device) -> Tensor:
        key = (height, width, device)
        cached = self._valid_cache.get(key)
        if cached is not None:
            return cached
        valid = torch.ones((1, NUM_SPATIAL, height, width), dtype=torch.bool, device=device)
        for k, (dx, dy) in enumerate(OFFSETS_3X3):
            if dx == -1:
                valid[:, k, :, 0] = False
            elif dx == 1:
                valid[:, k, :, width - 1] = False
            if dy == -1:
                valid[:, k, 0, :] = False
            elif dy == 1:
                valid[:, k, height - 1, :] = False
        if len(self._valid_cache) >= 16:
            self._valid_cache.clear()
        self._valid_cache[key] = valid
        return valid

    def forward(self, field: Tensor) -> tuple[Tensor, Tensor]:
        """Return shifted ``(B, C, 9, H, W)`` field via zero-FLOP ``F.pad + unfold`` and in-bounds mask."""
        bsz, channels, height, width = field.shape
        # Legacy checkpoint fallback if older pickled instance lacks tap_order buffer
        tap_order = getattr(self, "tap_order", None)
        if tap_order is None or tap_order.device != field.device:
            tap_order = torch.tensor(
                [(dy + 1) * 3 + (dx + 1) for dx, dy in OFFSETS_3X3],
                dtype=torch.long,
                device=field.device,
            )
            self.register_buffer("tap_order", tap_order, persistent=False)
        windows = F.pad(field, (1, 1, 1, 1)).unfold(2, 3, 1).unfold(3, 3, 1)
        shifted = windows.reshape(bsz, channels, height, width, NUM_SPATIAL).index_select(-1, tap_order)
        shifted = shifted.permute(0, 1, 4, 2, 3).contiguous()
        valid = self._get_valid_mask(height, width, field.device)
        return shifted, valid


class LGTDetect(Detect):
    """YOLO End-to-End ``Detect`` head with Class-Aware Local Gaussian Transport and universal DFL support."""

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 1,
        end2end: bool = True,
        ch: list[int] | tuple[int, ...] = (),
        rel_hidden: int = 16,
        init_tau: float = 1.5,
        init_beta_box: float = 0.15,
        distill_gain: float = 0.5,
        cross_scale: bool = True,
        max_active_levels: int = 4,
        fg_gate: bool = False,
        enable_cls_gate: bool = True,
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        if end2end:
            self.end2end = True
        if not (0.0 < init_beta_box < 1.0):
            raise ValueError(f"init_beta_box must be in (0, 1), got {init_beta_box}")
        if init_tau <= 0.1:
            raise ValueError(f"init_tau must be > 0.1, got {init_tau}")

        self.rel_hidden = int(rel_hidden)
        self.distill_gain = float(distill_gain)
        self.cross_scale = bool(cross_scale)
        self.max_active_levels = max(1, int(max_active_levels))
        self.active_nl = min(self.nl, self.max_active_levels)
        self.fg_gate = bool(fg_gate)
        self.enable_cls_gate = bool(enable_cls_gate)
        self._distill_decay: float = 1.0

        stencil_ch = 6 if self.nc > 1 else 5
        self.lgt_stencil = SpatialShiftStencil(channels=stencil_ch)

        tau_raw = math.log(math.expm1(max(init_tau - 0.1, 1e-3)))
        gamma_raw = math.log(math.expm1(1.0))
        beta_logit = math.log(init_beta_box / (1.0 - init_beta_box))

        self.lgt_log_tau_box = nn.Parameter(torch.full((self.active_nl,), tau_raw, dtype=torch.float32))
        self.lgt_log_gamma_box = nn.Parameter(torch.full((self.active_nl,), gamma_raw, dtype=torch.float32))
        self.lgt_logit_beta_box = nn.Parameter(torch.full((self.active_nl,), beta_logit, dtype=torch.float32))
        self.lgt_log_tau_cls = nn.Parameter(torch.full((self.active_nl,), tau_raw, dtype=torch.float32))

        rel_in_channels = NUM_NEIGHBORS * 2 + 2  # 20 relational channels
        if self.enable_cls_gate:
            gate_builder = self._build_cls_gate_nobn if self.fg_gate else self._build_cls_gate
            self.lgt_cls_gate: nn.ModuleList | None = nn.ModuleList(
                gate_builder(rel_in_channels, self.rel_hidden, self.nc) for _ in range(self.active_nl)
            )
        else:
            self.lgt_cls_gate = None

        self._raw_o2o_gauss: list[tuple[Tensor, Tensor]] | None = None
        self._stride_ratios: list[float] | None = None
        self._parent_grid_cache: dict[
            tuple[int, int, int, int, int, torch.device, torch.dtype], tuple[Tensor, Tensor, Tensor]
        ] = {}
        self._level_mask_cache: dict[tuple[int, int, int, torch.device], tuple[Tensor, Tensor, Tensor]] = {}
        self._last_distill_tensor: Tensor | None = None
        self.last_stats: dict[str, float] = {}

    @staticmethod
    def _build_cls_gate(in_ch: int, hidden: int, out_ch: int) -> nn.Sequential:
        out_conv = nn.Conv2d(hidden, out_ch, kernel_size=1, bias=True)
        nn.init.zeros_(out_conv.weight)
        if out_conv.bias is not None:
            nn.init.zeros_(out_conv.bias)
        return nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            out_conv,
        )

    @staticmethod
    def _build_cls_gate_nobn(in_ch: int, hidden: int, out_ch: int) -> nn.Sequential:
        """Background-safe relational MLP without BatchNorm2d statistics drift across 98% background."""
        out_conv = nn.Conv2d(hidden, out_ch, kernel_size=1, bias=True)
        nn.init.zeros_(out_conv.weight)
        if out_conv.bias is not None:
            nn.init.zeros_(out_conv.bias)
        return nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            out_conv,
        )

    def __getstate__(self):
        state = dict(super().__getstate__())
        state["_raw_o2o_gauss"] = None
        state["_stride_ratios"] = None
        state["_parent_grid_cache"] = {}
        state["_level_mask_cache"] = {}
        state["_last_distill_tensor"] = None
        return state

    def _get_parent_ratio(self, level_idx: int, p_idx: int) -> float:
        cached = self._stride_ratios
        if cached is not None and level_idx < len(cached):
            return cached[level_idx]
        strides = self.stride.detach().cpu().tolist()
        if len(strides) >= self.nl and all(s > 0 for s in strides):
            ratios = [
                float(strides[i + 1]) / max(float(strides[i]), 1e-6)
                for i in range(len(strides) - 1)
            ]
            self._stride_ratios = ratios
            return ratios[level_idx]
        s_child = float(strides[level_idx]) if strides[level_idx] > 0 else float(2**level_idx)
        s_parent = float(strides[p_idx]) if strides[p_idx] > 0 else float(2**p_idx)
        return s_parent / max(s_child, 1e-6)

    def _get_parent_offsets(
        self,
        level_idx: int,
        height: int,
        width: int,
        hp: int,
        wp: int,
        ratio: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        key = (level_idx, height, width, hp, wp, device, dtype)
        cached = self._parent_grid_cache.get(key)
        if cached is not None:
            return cached
        x_idx = torch.arange(width, device=device, dtype=dtype)
        y_idx = torch.arange(height, device=device, dtype=dtype)
        xp = torch.div(x_idx * wp, width, rounding_mode="floor")
        yp = torch.div(y_idx * hp, height, rounding_mode="floor")
        dx_p = (ratio * (xp + 0.5) - (x_idx + 0.5)).view(1, 1, width)
        dy_p = (ratio * (yp + 0.5) - (y_idx + 0.5)).view(1, height, 1)
        valid_p = torch.ones((1, 1, height, width), device=device, dtype=torch.bool)
        if len(self._parent_grid_cache) >= 32:
            self._parent_grid_cache.clear()
        self._parent_grid_cache[key] = (dx_p, dy_p, valid_p)
        return dx_p, dy_p, valid_p

    def _decode_level_ltrb(self, raw_box: Tensor) -> Tensor:
        """Extract ``(B, 4, H, W)`` stride-normalized ``[l, t, r, b]`` regardless of ``reg_max``."""
        if self.reg_max <= 1:
            return raw_box
        bsz, _, height, width = raw_box.shape
        return self.dfl(raw_box.view(bsz, 4 * self.reg_max, -1)).view(bsz, 4, height, width)

    def _get_decode_boxes(self, x: dict[str, Tensor]) -> Tensor:
        """Decode ``x['ltrb']`` directly when refined post-DFL coordinates are provided (``reg_max > 1``)."""
        if "ltrb" not in x:
            return super()._get_decode_boxes(x)
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        return self.decode_bboxes(x["ltrb"], self.anchors.unsqueeze(0), x.get("angle")) * self.strides

    def _parent_candidate(
        self,
        level_idx: int,
        height: int,
        width: int,
        mu_list: list[Tensor],
        sigma_list: list[Tensor],
        smax_list: list[Tensor],
        cls_list: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        """Transport co-located parent cell ``level_idx + 1`` into child grid coordinates."""
        bsz = mu_list[level_idx].shape[0]
        device = mu_list[level_idx].device
        dtype = mu_list[level_idx].dtype

        if not self.cross_scale or level_idx + 1 >= self.nl:
            zero_2 = torch.zeros((bsz, 2, 1, height, width), device=device, dtype=dtype)
            sig_dummy = torch.full((bsz, 2, 1, height, width), 0.05, device=device, dtype=dtype)
            zero_1 = torch.zeros((bsz, 1, height, width), device=device, dtype=dtype)
            valid_p = torch.zeros((1, 1, height, width), device=device, dtype=torch.bool)
            cls_dummy = torch.full((bsz, 1, height, width), -1.0, device=device, dtype=dtype) if cls_list is not None else None
            return zero_2, sig_dummy, zero_1, valid_p, cls_dummy

        p_idx = level_idx + 1
        mu_p = mu_list[p_idx]
        sig_p = sigma_list[p_idx]
        smax_p = smax_list[p_idx]
        hp, wp = mu_p.shape[2], mu_p.shape[3]
        ratio = self._get_parent_ratio(level_idx, p_idx)

        tensors = [mu_p, sig_p, smax_p]
        if cls_list is not None:
            tensors.append(cls_list[p_idx])
        packed_p = torch.cat(tensors, dim=1)
        up = F.interpolate(packed_p, size=(height, width), mode="nearest")
        up_mu = up[:, 0:2]
        up_sig = up[:, 2:4]
        up_smax = up[:, 4:5]
        up_cls = up[:, 5:6] if cls_list is not None else None

        dx_p, dy_p, valid_p = self._get_parent_offsets(
            level_idx, height, width, hp, wp, ratio, device, dtype
        )
        tilde_mu_x = ratio * up_mu[:, 0] + dx_p
        tilde_mu_y = ratio * up_mu[:, 1] + dy_p
        tilde_mu_p = torch.stack((tilde_mu_x, tilde_mu_y), dim=1).unsqueeze(2)
        tilde_sig_p = (ratio * up_sig).clamp(min=0.05).unsqueeze(2)
        return tilde_mu_p, tilde_sig_p, up_smax, valid_p, up_cls

    def _get_level_masks(
        self,
        level_idx: int,
        height: int,
        width: int,
        valid_sp: Tensor,
        valid_p: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        key = (level_idx, height, width, valid_sp.device)
        cached = self._level_mask_cache.get(key)
        if cached is not None:
            return cached
        valid = torch.cat((valid_sp, valid_p), dim=1)  # (1, 10, H, W)
        valid_nb = valid[:, 1:].to(dtype=torch.float32)  # (1, 9, H, W)
        denom_nb = valid_nb.sum(dim=1, keepdim=True).clamp(min=1.0)  # (1, 1, H, W)
        if len(self._level_mask_cache) >= 32:
            self._level_mask_cache.clear()
        self._level_mask_cache[key] = (valid, valid_nb, denom_nb)
        return valid, valid_nb, denom_nb

    def _refine_level(
        self,
        level_idx: int,
        raw_box: Tensor,
        raw_cls: Tensor,
        mu_list: list[Tensor],
        sigma_list: list[Tensor],
        smax_list: list[Tensor],
        cls_list: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply Class-Aware Local Gaussian Transport to level ``level_idx`` in fp32 for AMP safety."""
        orig_dtype = raw_cls.dtype
        mu_0 = mu_list[level_idx]  # (B, 2, H, W) fp32 (retains grad through one2one_cv2 / DFL)
        sig_0 = sigma_list[level_idx]  # (B, 2, H, W) fp32
        s_0 = smax_list[level_idx]  # (B, 1, H, W) fp32
        _, _, height, width = mu_0.shape

        with torch.autocast(device_type=mu_0.device.type, enabled=False):
            with torch.no_grad():
                pack_items = [mu_0.detach(), sig_0.detach(), s_0.detach()]
                if cls_list is not None:
                    pack_items.append(cls_list[level_idx])
                packed = torch.cat(pack_items, dim=1)
                shifted, valid_sp = self.lgt_stencil(packed)
                del packed

                tilde_mu_sp_nb = torch.stack(
                    (
                        shifted[:, 0, 1:] + self.lgt_stencil.dx[:, 1:],
                        shifted[:, 1, 1:] + self.lgt_stencil.dy[:, 1:],
                    ),
                    dim=1,
                )  # (B, 2, 8, H, W)
                tilde_sig_sp_nb = shifted[:, 2:4, 1:].clamp(min=0.05)
                cand_s_sp = shifted[:, 4].clone()  # (B, 9, H, W)
                cand_cls_sp_nb = shifted[:, 5, 1:].clone() if cls_list is not None else None
                del shifted

                tilde_mu_p, tilde_sig_p, cand_s_p, valid_p, cand_cls_p = self._parent_candidate(
                    level_idx, height, width, mu_list, sigma_list, smax_list, cls_list
                )

                tilde_mu_nb = torch.cat((tilde_mu_sp_nb, tilde_mu_p), dim=2)  # (B, 2, 9, H, W)
                tilde_sig_nb = torch.cat((tilde_sig_sp_nb, tilde_sig_p), dim=2)  # (B, 2, 9, H, W)
                cand_s = torch.cat((cand_s_sp, cand_s_p), dim=1)  # (B, 10, H, W)
                del tilde_mu_sp_nb, tilde_sig_sp_nb, cand_s_sp, tilde_mu_p, tilde_sig_p, cand_s_p

                valid, valid_nb_spatial, denom_nb = self._get_level_masks(
                    level_idx, height, width, valid_sp, valid_p
                )

                if cls_list is not None and cand_cls_sp_nb is not None and cand_cls_p is not None:
                    cand_cls_nb = torch.cat((cand_cls_sp_nb, cand_cls_p), dim=1)  # (B, 9, H, W)
                    same_cls_mask = valid[:, 1:] & (cand_cls_nb == cls_list[level_idx])
                    valid_nb_eff = same_cls_mask.to(dtype=torch.float32)
                    del cand_cls_sp_nb, cand_cls_p, cand_cls_nb
                else:
                    same_cls_mask = valid[:, 1:]
                    valid_nb_eff = valid_nb_spatial

                mu_c = mu_0.detach().unsqueeze(2)
                sig_c = sig_0.detach().unsqueeze(2)
                nb_delta_mu = tilde_mu_nb - mu_c
                nb_delta_sig = tilde_sig_nb - sig_c
                num = nb_delta_mu.square() + nb_delta_sig.square()
                den = sig_c.square() + tilde_sig_nb.square() + 1e-6
                del tilde_mu_nb, tilde_sig_nb
                dist_nb = (num / den).sum(dim=1).masked_fill(~same_cls_mask, 1e4)  # (B, 9, H, W)
                del num, den

                log_q = F.logsigmoid(cand_s)  # (B, 10, H, W)
                score_diff = F.relu(cand_s[:, 1:] - s_0.detach()) * valid_nb_eff
                prob_nb = torch.sigmoid(cand_s[:, 1:]) * valid_nb_eff
                del cand_s

            # 1. Unified Local Wasserstein Barycenter box refinement (works for reg_max == 1 AND reg_max > 1)
            tau_box = F.softplus(self.lgt_log_tau_box[level_idx]) + 0.1
            gamma_box = F.softplus(self.lgt_log_gamma_box[level_idx])
            beta_box = torch.sigmoid(self.lgt_logit_beta_box[level_idx])

            logit_0 = gamma_box * log_q[:, :1]
            logit_nb = (gamma_box * log_q[:, 1:] - tau_box * dist_nb).masked_fill(
                ~same_cls_mask, -1e4
            )
            logits_w = torch.cat((logit_0, logit_nb), dim=1)
            weights_nb = torch.softmax(logits_w, dim=1)[:, 1:].unsqueeze(1)  # (B, 1, 9, H, W)

            delta_mu = (weights_nb * nb_delta_mu).sum(dim=2)
            delta_sig = (weights_nb * nb_delta_sig).sum(dim=2)
            mu_ref = mu_0 + beta_box * delta_mu
            sig_ref = (sig_0 + beta_box * delta_sig).clamp(min=0.05)
            refined_ltrb = gaussian_to_ltrb(mu_ref, sig_ref).to(raw_box.dtype)

            # 2. Differentiable Class-Aware NMS Dominance & Consensus Gate
            tau_cls = F.softplus(self.lgt_log_tau_cls[level_idx]) + 0.1
            aff_nb = torch.exp(-tau_cls * dist_nb) * valid_nb_eff

            dom_pairs = aff_nb * score_diff
            m_dom = dom_pairs.amax(dim=1, keepdim=True)
            m_sup = (aff_nb * prob_nb).sum(dim=1, keepdim=True) / denom_nb

            aff_feat = (aff_nb * prob_nb) if self.fg_gate else aff_nb
            rel_feat = torch.cat((aff_feat, dom_pairs, m_dom, m_sup), dim=1).to(orig_dtype)

        if not self.enable_cls_gate or self.lgt_cls_gate is None:
            refined_cls = raw_cls
        else:
            cls_delta = self.lgt_cls_gate[level_idx](rel_feat)
            if self.fg_gate:
                fg_prob = torch.sigmoid(s_0.detach()).to(orig_dtype)
                refined_cls = raw_cls + fg_prob * cls_delta
            else:
                refined_cls = raw_cls + cls_delta

        return refined_ltrb, refined_cls

    def forward_lgt_one2one(self, x: list[Tensor]) -> dict[str, Any]:
        """Run ``one2one`` branch with cross-scale Class-Aware Local Gaussian Transport."""
        if self.one2one_cv2 is None or self.one2one_cv3 is None:
            return {}
        bsz = x[0].shape[0]
        raw_boxes = [self.one2one_cv2[i](x[i]) for i in range(self.nl)]
        raw_clses = [self.one2one_cv3[i](x[i]) for i in range(self.nl)]

        device_type = x[0].device.type
        use_multiclass = self.nc > 1
        with torch.autocast(device_type=device_type, enabled=False):
            ltrb_list = [self._decode_level_ltrb(rb).float() for rb in raw_boxes]
            mu_list: list[Tensor] = []
            sigma_list: list[Tensor] = []
            smax_list: list[Tensor] = []
            cls_list: list[Tensor] | None = [] if use_multiclass else None
            # Only need up to active_nl + 1 (for parent of highest active level)
            needed_nl = min(self.nl, self.active_nl + (1 if self.cross_scale else 0))
            for i in range(needed_nl):
                mu_i, sig_i = ltrb_to_gaussian(ltrb_list[i])
                mu_list.append(mu_i)
                sigma_list.append(sig_i)
                cls_detached = raw_clses[i].detach().float()
                smax_list.append(cls_detached.amax(dim=1, keepdim=True))
                if cls_list is not None:
                    cls_list.append(cls_detached.argmax(dim=1, keepdim=True).to(dtype=torch.float32))

        if self.training and self.distill_gain > 0.0:
            self._raw_o2o_gauss = list(zip(mu_list[: self.active_nl], sigma_list[: self.active_nl]))
        else:
            self._raw_o2o_gauss = None

        out_ltrb: list[Tensor] = []
        out_scores: list[Tensor] = []
        for i in range(self.nl):
            if i < self.active_nl:
                ref_ltrb, ref_cls = self._refine_level(
                    i, raw_boxes[i], raw_clses[i], mu_list, sigma_list, smax_list, cls_list
                )
            else:
                ref_ltrb = ltrb_list[i].to(raw_boxes[i].dtype)
                ref_cls = raw_clses[i]
            out_ltrb.append(ref_ltrb.view(bsz, 4, -1))
            out_scores.append(ref_cls.view(bsz, self.nc, -1))

        if self.reg_max <= 1:
            return {
                "boxes": torch.cat(out_ltrb, dim=-1),
                "scores": torch.cat(out_scores, dim=-1),
                "feats": x,
            }

        out_raw_dfl = [rb.view(bsz, 4 * self.reg_max, -1) for rb in raw_boxes]
        return {
            "boxes": torch.cat(out_raw_dfl, dim=-1),
            "ltrb": torch.cat(out_ltrb, dim=-1),
            "scores": torch.cat(out_scores, dim=-1),
            "feats": x,
        }

    def forward(
        self, x: list[Tensor]
    ) -> dict[str, Tensor] | Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Run dual-head forward in training/eval and single-branch LGT forward after ``fuse()``."""
        preds: dict[str, Any] = self.forward_head(x, **self.one2many)
        if getattr(self, "one2one_cv2", None) is not None:
            x_detach = [xi.detach() for xi in x] if self.training else x
            one2one = self.forward_lgt_one2one(x_detach)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one" if self.end2end else "one2many"] if "one2one" in preds else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def get_lgt_stats(self) -> dict[str, float]:
        """Materialize scalar LGT diagnostics across active FPN levels at epoch end."""
        stats: dict[str, float] = {}
        beta_vals = torch.sigmoid(self.lgt_logit_beta_box).detach().cpu().tolist()
        tau_box_vals = (F.softplus(self.lgt_log_tau_box) + 0.1).detach().cpu().tolist()
        tau_cls_vals = (F.softplus(self.lgt_log_tau_cls) + 0.1).detach().cpu().tolist()
        gamma_vals = F.softplus(self.lgt_log_gamma_box).detach().cpu().tolist()
        for idx in range(self.active_nl):
            lvl = f"p{idx + 2}"
            stats[f"beta_box_{lvl}"] = float(beta_vals[idx])
            stats[f"tau_box_{lvl}"] = float(tau_box_vals[idx])
            stats[f"tau_cls_{lvl}"] = float(tau_cls_vals[idx])
            stats[f"gamma_box_{lvl}"] = float(gamma_vals[idx])
        if self._last_distill_tensor is not None:
            stats["distill_loss"] = float(self._last_distill_tensor.item())
        self.last_stats = stats
        return stats

    def aux_loss(self, preds: dict[str, dict[str, Tensor]], batch: dict[str, Tensor]) -> Tensor:
        """Training-only dense Gaussian box-field distillation from ``one2many`` to ``one2one_cv2``."""
        del batch
        raw_o2o_gauss = self._raw_o2o_gauss
        self._raw_o2o_gauss = None
        if self.distill_gain <= 0.0 or raw_o2o_gauss is None or "one2many" not in preds:
            return torch.zeros((), device=self.lgt_log_tau_box.device)

        o2m = preds["one2many"]
        if not o2m or "boxes" not in o2m or "scores" not in o2m:
            return torch.zeros((), device=self.lgt_log_tau_box.device)

        device_type = raw_o2o_gauss[0][0].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            o2m_boxes_flat = o2m["boxes"].detach().float()
            o2m_scores_flat = o2m["scores"].detach().float()
            bsz = o2m_boxes_flat.shape[0]

            offset = 0
            total_num = torch.zeros((), device=o2m_boxes_flat.device, dtype=torch.float32)
            total_den = torch.tensor(1e-6, device=o2m_boxes_flat.device, dtype=torch.float32)

            for mu_o2o, sig_o2o in raw_o2o_gauss:
                _, _, height, width = mu_o2o.shape
                count = height * width
                with torch.no_grad():
                    o2m_raw = o2m_boxes_flat[:, :, offset : offset + count].view(
                        bsz, 4 * self.reg_max, height, width
                    )
                    o2m_ltrb = self._decode_level_ltrb(o2m_raw).float()
                    o2m_prob = torch.sigmoid(o2m_scores_flat[:, :, offset : offset + count]).amax(dim=1)
                    if self.fg_gate:
                        o2m_prob = o2m_prob * (o2m_prob > 0.20).to(o2m_prob.dtype)
                    weight = o2m_prob.square().view(bsz, height, width)
                    mu_o2m, sig_o2m = ltrb_to_gaussian(o2m_ltrb)
                offset += count

                num = (mu_o2o - mu_o2m).square() + (sig_o2o - sig_o2m).square()
                den = sig_o2o.square() + sig_o2m.square() + 1e-6
                dist = (num / den).sum(dim=1)
                bounded_loss = dist / (1.0 + dist)

                total_num = total_num + (weight * bounded_loss).sum()
                total_den = total_den + weight.sum()

            mean_distill = total_num / total_den
            self._last_distill_tensor = mean_distill.detach()
            eff_gain = self.distill_gain * float(self._distill_decay)
            return mean_distill * (bsz * eff_gain)


def find_lgt_head(model: nn.Module) -> LGTDetect | None:
    """Return the ``LGTDetect`` head of a ``DetectionModel`` in O(1), or ``None`` if absent."""
    if isinstance(model, LGTDetect):
        return model
    seq = getattr(model, "model", None)
    if isinstance(seq, (nn.Sequential, nn.ModuleList)) and len(seq) > 0:
        last = seq[-1]
        if isinstance(last, LGTDetect):
            return last
        return None
    for module in model.modules():
        if isinstance(module, LGTDetect):
            return module
    return None
