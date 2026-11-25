# agy-allow-dup
"""Local Gaussian Transport Detection Head (``LGTDetect``) for NMS-free tiny object detection.

Mathematical formulation
------------------------
In DFL-free, NMS-free detectors (YOLO26: ``reg_max=1``, ``end2end=True``), each grid cell
``u = (y, x)`` on FPN level ``s`` predicts four scalar distances ``(l, t, r, b)`` in stride
units and class logits ``s_cls``. Standard ``one2one`` heads suffer two failure modes on tiny
crowded objects:
1. Single-cell box regression variance without DFL collapses high-IoU precision (``AP75``).
2. Appearance-only ``one2one_cv3`` classification is geometry-blind, falsely suppressing the
   second object in close tiny pairs.

``LGTDetect`` models each cell's box as a diagonal 2D Gaussian ``G(u) = N(mu(u), diag(sigma(u)^2))``
with ``mu = ((r - l)/2, (b - t)/2)`` and ``sigma = ((l + r)/2, (t + b)/2)``, and performs:

1. **Coordinate-Frame Transport (Proposition 1):**
   Across the ``3x3`` spatial neighborhood ``delta_k in {-1, 0, +1}^2`` (extracted via a frozen
   depthwise shift ``Conv2d``) plus the co-located parent cell interpolated from FPN level ``s+1``,
   neighbor Gaussians are transported into cell ``u``'s local coordinate frame:
   ``tilde_mu_k(u) = mu(u + delta_k) + delta_k``, ``tilde_sigma_k(u) = sigma(u + delta_k)``.
   Identical physical boxes across adjacent cells or adjacent FPN scales satisfy
   ``tilde_mu_k == mu`` and ``tilde_sigma_k == sigma`` exactly.

2. **Hyperparameter-Free Scale-Invariant Wasserstein Distance (Proposition 2):**
   ``D_k(u) = ((tilde_mu_kx - mu_x)^2 + (tilde_sigma_kx - sigma_x)^2) / (sigma_x^2 + tilde_sigma_kx^2 + eps)``
   ``       + ((tilde_mu_ky - mu_y)^2 + (tilde_sigma_ky - sigma_y)^2) / (sigma_y^2 + tilde_sigma_ky^2 + eps)``
   Unlike fixed-constant NWD, ``D_k(lambda * B1, lambda * B2) == D_k(B1, B2)`` for all ``lambda > 0``
   with zero manual scale constants, separating duplicate predictions on larger targets from
   distinct adjacent ``tiny1`` instances.

3. **Closed-Form Local Wasserstein Barycenter & Differentiable Dominance Gate (Proposition 3):**
   - Box branch (``one2one_cv2``): refines ``(mu, sigma)`` via confidence- and affinity-weighted
     Wasserstein barycenter ``w_k = softmax(gamma * log(sigmoid(s_k)) - tau_box * D_k)``.
   - Classification branch (``one2one_cv3``): computes differentiable NMS dominance pressure
     ``M_dom = max_{k != 0} A_k * ReLU(s_k - s_0)`` and neighborhood consensus support
     ``M_sup = mean_{k != 0} A_k * sigmoid(s_k)`` where ``A_k = exp(-tau_cls * D_k)``, modulating
     ``one2one`` class logits through a residual ``1x1`` bottleneck zero-initialized at its output.

4. **Training-Only Dense Gaussian Field Distillation:**
   During training, ``one2many`` (supervised with ``topk=10``) distills its dense Gaussian box field
   into ``one2one_cv2``'s pre-barycenter field weighted by ``one2many`` foreground confidence. At
   inference (``Detect.fuse()``), ``one2many`` is removed completely with zero runtime overhead.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ultralytics.nn.modules.head import Detect

# Center first (k=0), followed by the 8 surrounding (dx, dy) spatial taps.
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
NUM_CANDIDATES = NUM_SPATIAL + 1  # 9 spatial + 1 coarser-scale parent = 10
NUM_NEIGHBORS = NUM_CANDIDATES - 1  # 9 non-self candidates


def xyxy_to_gaussian(boxes: Tensor, eps_sigma: float = 1e-3) -> tuple[Tensor, Tensor]:
    """Convert ``(..., 4)`` ``[x1, y1, x2, y2]`` boxes into ``(mu, sigma)`` of shape ``(..., 2)``."""
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
    """Fixed depthwise ``3x3`` shift convolution extracting ``OFFSETS_3X3`` taps and valid masks.

    Stored as a non-parameter buffer (``register_buffer``) rather than an ``nn.Parameter`` so
    Ultralytics ``BaseTrainer`` never forces ``requires_grad=True`` on the Dirac shift stencil.
    """

    def __init__(self, channels: int = 5) -> None:
        super().__init__()
        self.channels = int(channels)
        weight = torch.zeros((self.channels * NUM_SPATIAL, 1, 3, 3), dtype=torch.float32)
        for c in range(self.channels):
            for k, (dx, dy) in enumerate(OFFSETS_3X3):
                weight[c * NUM_SPATIAL + k, 0, dy + 1, dx + 1] = 1.0
        self.register_buffer("shift_weight", weight, persistent=False)

        dx_t = torch.tensor([dx for dx, _ in OFFSETS_3X3], dtype=torch.float32).view(1, NUM_SPATIAL, 1, 1)
        dy_t = torch.tensor([dy for _, dy in OFFSETS_3X3], dtype=torch.float32).view(1, NUM_SPATIAL, 1, 1)
        self.register_buffer("dx", dx_t, persistent=False)
        self.register_buffer("dy", dy_t, persistent=False)

    def forward(self, field: Tensor) -> tuple[Tensor, Tensor]:  # agy-allow-dup
        """Return shifted ``(B, C, 9, H, W)`` field and ``(1, 9, H, W)`` in-bounds boolean mask."""
        bsz, channels, height, width = field.shape
        weight = self.shift_weight.to(dtype=field.dtype)
        shifted = F.conv2d(field, weight, padding=1, groups=self.channels).view(
            bsz, channels, NUM_SPATIAL, height, width
        )

        valid = torch.ones((1, NUM_SPATIAL, height, width), dtype=torch.bool, device=field.device)
        for k, (dx, dy) in enumerate(OFFSETS_3X3):
            if dx == -1:
                valid[:, k, :, 0] = False
            elif dx == 1:
                valid[:, k, :, width - 1] = False
            if dy == -1:
                valid[:, k, 0, :] = False
            elif dy == 1:
                valid[:, k, height - 1, :] = False
        return shifted, valid


class LGTDetect(Detect):
    """YOLO End-to-End ``Detect`` head with Local Gaussian Transport refinement and dominance gating.

    Args:
        nc: Number of object classes.
        reg_max: DFL bins (1 for YOLO26 DFL-free direct regression).
        end2end: Enable NMS-free one-to-one inference path.
        ch: Input feature channels per FPN level ``[P2, P3, ...]``.
        rel_hidden: Hidden channels of the ``1x1`` relational dominance gate MLP.
        init_tau: Initial sharpness ``tau`` for Wasserstein affinity and barycenter kernels.
        init_beta_box: Initial barycenter box blending factor in ``(0, 1)``.
        distill_gain: Weight of training-only ``one2many -> one2one`` dense Gaussian distillation.
        cross_scale: Include the co-located parent FPN level candidate ``P_{s+1} -> P_s``.
    """

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

        # Named without "one2one_" prefix so Detect.fuse() keeps them intact in end2end mode.
        self.lgt_stencil = SpatialShiftStencil(channels=5)

        tau_raw = math.log(math.expm1(max(init_tau - 0.1, 1e-3)))
        gamma_raw = math.log(math.expm1(1.0))
        beta_logit = math.log(init_beta_box / (1.0 - init_beta_box))

        self.lgt_log_tau_box = nn.Parameter(torch.full((self.nl,), tau_raw, dtype=torch.float32))
        self.lgt_log_gamma_box = nn.Parameter(torch.full((self.nl,), gamma_raw, dtype=torch.float32))
        self.lgt_logit_beta_box = nn.Parameter(torch.full((self.nl,), beta_logit, dtype=torch.float32))
        self.lgt_log_tau_cls = nn.Parameter(torch.full((self.nl,), tau_raw, dtype=torch.float32))

        rel_in_channels = NUM_NEIGHBORS * 2 + 2  # 9 affinities + 9 weighted score diffs + M_dom + M_sup = 20
        self.lgt_cls_gate = nn.ModuleList(
            self._build_cls_gate(rel_in_channels, self.rel_hidden, self.nc) for _ in range(self.nl)
        )

        self._raw_o2o_boxes: list[Tensor] | None = None
        self.last_stats: dict[str, float] = {}

    @staticmethod
    def _build_cls_gate(in_ch: int, hidden: int, out_ch: int) -> nn.Sequential:
        out_conv = nn.Conv2d(hidden, out_ch, kernel_size=1, bias=True)
        nn.init.zeros_(out_conv.weight)
        nn.init.zeros_(out_conv.bias)
        return nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            out_conv,
        )

    def __getstate__(self):
        state = dict(super().__getstate__())
        state["_raw_o2o_boxes"] = None
        return state

    def _decode_level_ltrb(self, raw_box: Tensor) -> Tensor:
        """Extract ``(B, 4, H, W)`` stride-normalized ``[l, t, r, b]`` regardless of ``reg_max``."""
        if self.reg_max <= 1:
            return raw_box
        bsz, _, height, width = raw_box.shape
        return self.dfl(raw_box.view(bsz, 4 * self.reg_max, -1)).view(bsz, 4, height, width)

    def _parent_candidate(
        self,
        level_idx: int,
        height: int,
        width: int,
        mu_list: list[Tensor],
        sigma_list: list[Tensor],
        smax_list: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Transport co-located parent cell ``level_idx + 1`` into child grid coordinates."""
        bsz = mu_list[level_idx].shape[0]
        device = mu_list[level_idx].device
        dtype = mu_list[level_idx].dtype

        if not self.cross_scale or level_idx + 1 >= self.nl:
            zero_2 = torch.zeros((bsz, 2, 1, height, width), device=device, dtype=dtype)
            zero_1 = torch.zeros((bsz, 1, height, width), device=device, dtype=dtype)
            valid_p = torch.zeros((1, 1, height, width), device=device, dtype=torch.bool)
            return zero_2, zero_2.clamp(min=0.05), zero_1, valid_p

        p_idx = level_idx + 1
        mu_p = mu_list[p_idx]
        sig_p = sigma_list[p_idx]
        smax_p = smax_list[p_idx]
        hp, wp = mu_p.shape[2], mu_p.shape[3]

        s_child = float(self.stride[level_idx].item()) if self.stride[level_idx] > 0 else float(2**level_idx)
        s_parent = float(self.stride[p_idx].item()) if self.stride[p_idx] > 0 else float(2**p_idx)
        ratio = s_parent / max(s_child, 1e-6)

        packed_p = torch.cat((mu_p, sig_p, smax_p), dim=1)
        up = F.interpolate(packed_p, size=(height, width), mode="nearest")
        up_mu = up[:, 0:2]
        up_sig = up[:, 2:4]
        up_smax = up[:, 4:5]

        x_idx = torch.arange(width, device=device, dtype=dtype)
        y_idx = torch.arange(height, device=device, dtype=dtype)
        xp = torch.div(x_idx * wp, width, rounding_mode="floor")
        yp = torch.div(y_idx * hp, height, rounding_mode="floor")
        dx_p = (ratio * (xp + 0.5) - (x_idx + 0.5)).view(1, 1, width)
        dy_p = (ratio * (yp + 0.5) - (y_idx + 0.5)).view(1, height, 1)

        tilde_mu_x = ratio * up_mu[:, 0] + dx_p
        tilde_mu_y = ratio * up_mu[:, 1] + dy_p
        tilde_mu_p = torch.stack((tilde_mu_x, tilde_mu_y), dim=1).unsqueeze(2)
        tilde_sig_p = (ratio * up_sig).clamp(min=0.05).unsqueeze(2)
        valid_p = torch.ones((1, 1, height, width), device=device, dtype=torch.bool)
        return tilde_mu_p, tilde_sig_p, up_smax, valid_p

    def _refine_level(
        self,
        level_idx: int,
        raw_box: Tensor,
        raw_cls: Tensor,
        mu_list: list[Tensor],
        sigma_list: list[Tensor],
        smax_list: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Apply Local Gaussian Transport to level ``level_idx`` in fp32 for AMP safety."""
        orig_dtype = raw_cls.dtype
        mu_0 = mu_list[level_idx]  # (B, 2, H, W) fp32
        sig_0 = sigma_list[level_idx]  # (B, 2, H, W) fp32
        s_0 = smax_list[level_idx]  # (B, 1, H, W) fp32
        _, _, height, width = mu_0.shape

        packed = torch.cat((mu_0, sig_0, s_0), dim=1)  # (B, 5, H, W)
        shifted, valid_sp = self.lgt_stencil(packed)  # (B, 5, 9, H, W), (1, 9, H, W)

        tilde_mu_sp = torch.stack(
            (shifted[:, 0] + self.lgt_stencil.dx, shifted[:, 1] + self.lgt_stencil.dy),
            dim=1,
        )  # (B, 2, 9, H, W)
        tilde_sig_sp = shifted[:, 2:4].clamp(min=0.05)  # (B, 2, 9, H, W)
        cand_s_sp = shifted[:, 4]  # (B, 9, H, W)

        tilde_mu_p, tilde_sig_p, cand_s_p, valid_p = self._parent_candidate(
            level_idx, height, width, mu_list, sigma_list, smax_list
        )

        tilde_mu = torch.cat((tilde_mu_sp, tilde_mu_p), dim=2)  # (B, 2, 10, H, W)
        tilde_sig = torch.cat((tilde_sig_sp, tilde_sig_p), dim=2)  # (B, 2, 10, H, W)
        cand_s = torch.cat((cand_s_sp, cand_s_p), dim=1)  # (B, 10, H, W)
        valid = torch.cat((valid_sp, valid_p), dim=1)  # (1, 10, H, W)

        mu_c = mu_0.unsqueeze(2)  # (B, 2, 1, H, W)
        sig_c = sig_0.unsqueeze(2)  # (B, 2, 1, H, W)
        num = (tilde_mu - mu_c).square() + (tilde_sig - sig_c).square()
        den = sig_c.square() + tilde_sig.square() + 1e-6
        dist_all = (num / den).sum(dim=1)  # (B, 10, H, W)
        dist_all[:, 0].zero_()
        dist_masked = dist_all.masked_fill(~valid, 1e4)

        # 1. Local Wasserstein Barycenter box refinement (for DFL-free reg_max == 1)
        if self.reg_max <= 1:
            tau_box = F.softplus(self.lgt_log_tau_box[level_idx]) + 0.1
            gamma_box = F.softplus(self.lgt_log_gamma_box[level_idx])
            beta_box = torch.sigmoid(self.lgt_logit_beta_box[level_idx])

            log_q = F.logsigmoid(cand_s.detach())
            logits_w = gamma_box * log_q - tau_box * dist_masked.detach()
            logits_w = logits_w.masked_fill(~valid, -1e4)
            weights_nb = torch.softmax(logits_w, dim=1)[:, 1:].unsqueeze(1)  # (B, 1, 9, H, W)

            delta_mu = (weights_nb * (tilde_mu[:, :, 1:].detach() - mu_c.detach())).sum(dim=2)
            delta_sig = (weights_nb * (tilde_sig[:, :, 1:].detach() - sig_c.detach())).sum(dim=2)
            mu_ref = mu_0 + beta_box * delta_mu
            sig_ref = (sig_0 + beta_box * delta_sig).clamp(min=0.05)
            refined_box = gaussian_to_ltrb(mu_ref, sig_ref).to(raw_box.dtype)
        else:
            refined_box = raw_box

        # 2. Differentiable NMS Dominance & Consensus Gate on classification logits
        tau_cls = F.softplus(self.lgt_log_tau_cls[level_idx]) + 0.1
        valid_nb = valid[:, 1:].to(dtype=torch.float32)  # (1, 9, H, W)
        aff_nb = torch.exp(-tau_cls * dist_masked[:, 1:].detach()) * valid_nb  # (B, 9, H, W)
        score_diff = F.relu(cand_s[:, 1:].detach() - s_0.detach()) * valid_nb  # (B, 9, H, W)
        prob_nb = torch.sigmoid(cand_s[:, 1:].detach()) * valid_nb  # (B, 9, H, W)

        dom_pairs = aff_nb * score_diff
        m_dom = dom_pairs.amax(dim=1, keepdim=True)  # (B, 1, H, W)
        denom_nb = valid_nb.sum(dim=1, keepdim=True).clamp(min=1.0)
        m_sup = (aff_nb * prob_nb).sum(dim=1, keepdim=True) / denom_nb  # (B, 1, H, W)

        rel_feat = torch.cat((aff_nb, dom_pairs, m_dom, m_sup), dim=1).to(orig_dtype)
        cls_delta = self.lgt_cls_gate[level_idx](rel_feat)
        refined_cls = raw_cls + cls_delta

        return refined_box, refined_cls

    def forward_lgt_one2one(self, x: list[Tensor]) -> dict[str, Tensor]:
        """Run ``one2one`` branch with cross-scale Local Gaussian Transport."""
        if self.one2one_cv2 is None or self.one2one_cv3 is None:
            return {}
        bsz = x[0].shape[0]
        raw_boxes = [self.one2one_cv2[i](x[i]) for i in range(self.nl)]
        raw_clses = [self.one2one_cv3[i](x[i]) for i in range(self.nl)]

        device_type = x[0].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            ltrb_list = [self._decode_level_ltrb(rb).float() for rb in raw_boxes]
            mu_list: list[Tensor] = []
            sigma_list: list[Tensor] = []
            smax_list: list[Tensor] = []
            for i in range(self.nl):
                mu_i, sig_i = ltrb_to_gaussian(ltrb_list[i])
                mu_list.append(mu_i)
                sigma_list.append(sig_i)
                smax_list.append(raw_clses[i].float().amax(dim=1, keepdim=True))

        if self.training and self.distill_gain > 0.0:
            self._raw_o2o_boxes = ltrb_list
        else:
            self._raw_o2o_boxes = None

        out_boxes: list[Tensor] = []
        out_scores: list[Tensor] = []
        for i in range(self.nl):
            ref_box, ref_cls = self._refine_level(
                i, raw_boxes[i], raw_clses[i], mu_list, sigma_list, smax_list
            )
            out_boxes.append(ref_box.view(bsz, 4 * self.reg_max, -1))
            out_scores.append(ref_cls.view(bsz, self.nc, -1))

        return {
            "boxes": torch.cat(out_boxes, dim=-1),
            "scores": torch.cat(out_scores, dim=-1),
            "feats": x,
        }

    def forward(  # agy-allow-dup
        self, x: list[Tensor]
    ) -> dict[str, Tensor] | Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Run dual-head forward in training/eval and single-branch LGT forward after ``fuse()``."""
        preds = self.forward_head(x, **self.one2many)
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

    def aux_loss(self, preds: dict[str, dict[str, Tensor]], batch: dict[str, Tensor]) -> Tensor:  # agy-allow-dup
        """Training-only dense Gaussian box-field distillation from ``one2many`` to ``one2one_cv2``."""
        del batch
        raw_o2o = self._raw_o2o_boxes
        self._raw_o2o_boxes = None
        if self.distill_gain <= 0.0 or raw_o2o is None or "one2many" not in preds:
            return torch.zeros((), device=next(self.parameters()).device)

        o2m = preds["one2many"]
        if not o2m or "boxes" not in o2m or "scores" not in o2m:
            return torch.zeros((), device=next(self.parameters()).device)

        device_type = raw_o2o[0].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            o2m_boxes_flat = o2m["boxes"].detach().float()
            o2m_scores_flat = o2m["scores"].detach().float()
            bsz = o2m_boxes_flat.shape[0]

            offset = 0
            total_num = torch.zeros((), device=o2m_boxes_flat.device, dtype=torch.float32)
            total_den = torch.tensor(1e-6, device=o2m_boxes_flat.device, dtype=torch.float32)

            for o2o_ltrb in raw_o2o:
                _, _, height, width = o2o_ltrb.shape
                count = height * width
                o2m_raw = o2m_boxes_flat[:, :, offset : offset + count].view(
                    bsz, 4 * self.reg_max, height, width
                )
                o2m_ltrb = self._decode_level_ltrb(o2m_raw).float()
                o2m_prob = torch.sigmoid(o2m_scores_flat[:, :, offset : offset + count]).amax(dim=1)
                weight = o2m_prob.square().view(bsz, height, width)
                offset += count

                mu_o2o, sig_o2o = ltrb_to_gaussian(o2o_ltrb.float())
                mu_o2m, sig_o2m = ltrb_to_gaussian(o2m_ltrb)
                num = (mu_o2o - mu_o2m).square() + (sig_o2o - sig_o2m).square()
                den = sig_o2o.square() + sig_o2m.square() + 1e-6
                dist = (num / den).sum(dim=1)
                bounded_loss = dist / (1.0 + dist)

                total_num = total_num + (weight * bounded_loss).sum()
                total_den = total_den + weight.sum()

            loss = (total_num / total_den) * bsz * self.distill_gain
            self.last_stats = {
                "distill_loss": float((total_num / total_den).detach().item()),
                "beta_box_p2": float(torch.sigmoid(self.lgt_logit_beta_box[0]).detach().item()),
                "tau_cls_p2": float((F.softplus(self.lgt_log_tau_cls[0]) + 0.1).detach().item()),
            }
            return loss


def find_lgt_head(model: nn.Module) -> LGTDetect | None:
    """Return the ``LGTDetect`` head of a ``DetectionModel``, or ``None`` if absent."""
    for module in model.modules():
        if isinstance(module, LGTDetect):
            return module
    return None
