"""GaussSplit: Gaussian scale-space instance splitting for crowded tiny objects.

Relationship to GauSS (``src/tinydet/modules/gauss.py``)
-------------------------------------------------------
GauSS applies a learnable Gaussian scale-space to the **whole P2 map**. GaussSplit
reuses that exact operator (``LearnableGaussianScaleSpace``, same tanh-bounded
sigma re-parameterisation, same separable depthwise kernels) but applies it at
**instance granularity**: a 7x7 neighbourhood around each existing detection.
At that granularity the operator has a specific job — deciding whether the local
response contains **one** blob or **two overlapping** blobs — which is the
question the detector cannot answer (one location emits one box).

Motivation (measured in this repo, TINY_RESEARCH_LOG.md R20-R22)
---------------------------------------------------------------
At matched effective size under 1:1 tiled inference, GTs whose nearest neighbour
is < 10 px are detected at 59.1% versus 76.0% for isolated GTs. The model is not
silent there: the nearest detection sits 0.37 GT-sizes away, overlaps the missed
GT at a median IoU of 0.319, and 53.5% of misses have a detection at IoU >= 0.3.
Only 3.7% of misses are NMS-credited to the neighbour. So the failure is
representational: one location, one box, two people.

Mechanism
---------
Model the detector's local response around a pair as a **mixture of two
Gaussians**. GaussSplit estimates (i) how many components there are (a Bernoulli
gate) and (ii) the parameters of the second component::

    l1, l2, sigmas = GaussianScaleSpace(P2)        # G1: two learnable scales
    fine   = sample(l1, 7x7 around detection)      # (C, 7, 7)
    dog    = sample(l1 - l2, 7x7)                  # difference of Gaussians
    patch  = compress([fine, dog])                 # 1x1 conv 2C -> C'
    x      = [patch.flatten(), log w, log h, score]
    out    = MLP(x) -> [logit_second, dx, dy, log_size_ratio]

Note: an earlier revision included a scalar response-width feature ``w_resp``
(soft second-moment radius of the fine patch). It was removed after
measurement: at patch 7 the statistic saturates at the window/blur width and
is not monotone in the blob size (see docs/LESSONS.md L-020 context), so it
carries no information. G4 = DoG channels + NWD loss.

Why a difference of Gaussians: classical scale-space theory (Witkin 1983;
Lindeberg 1994) states that a mixture of two Gaussians becomes separable at a
scale of order the gap between them, and that the sign pattern of the
Laplacian/DoG is what distinguishes one extremum from two. The DoG patch is
therefore a *learned-scale blob separator*, not a decorative feature.

Why the second box is decoded as a Gaussian (G2): for boxes of 2-8 px (41.7% of
TinyPerson GTs by sqrt(w*h), measured in this repo) IoU is discontinuous — a 1 px
shift can move it from 0.6 to 0.2 — while the Wasserstein distance between two
Gaussian boxes is smooth in size and position. ``nwd_loss`` implements the
normalised Wasserstein form of NWD (arXiv:2110.13389).

Design properties
-----------------
* **Post-hoc.** Does not touch the backbone, Detect head, assigner or detector
  loss. Attachable to any detector exposing a stride-4 map and a detection list.
* **Same output contract as PairSplit.** ``(N, 4) = [logit_second, dx, dy,
  log_size_ratio]``, decoded by ``PairSplit.decode``. This makes the ablation
  "PairSplit vs GaussSplit" isolate the *representation*, not the decode.
* **Smaller than PairSplit.** ~17.6 k parameters versus 23.7 k, so a gain cannot
  be attributed to extra capacity.
* **Conservative at init.** Output layer zero-initialised with
  ``bias[0] = INIT_SECOND_BIAS``: at step 0 ``p(second) = 0.1192``, offsets 0,
  ratio 1 — attaching the head changes nothing until it learns.

Provenance
----------
Self-tests in ``tests/test_gauss_split.py``: parameter count, init no-op, DoG
vanishing on a constant map, gather semantics, response-width monotonicity,
Wasserstein/NWD values against hand computation, loss masking, gradient flow to
every parameter, batch-order restoration, and decode parity with PairSplit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinydet.modules.gauss import LearnableGaussianScaleSpace
from tinydet.modules.pair_split import (
    INIT_SECOND_BIAS,
    PATCH_SIDE,
    gather_patches,
)
from tinydet.modules.pair_split import (
    count_parameters as _count_parameters,
)

# NWD's normalisation constant: the paper sets C to the average object size of
# the dataset. We default to the per-batch mean sqrt(area) of the matched GT,
# which is the same rule computed online, and expose it as an argument so a
# fixed value can be used for reproducibility.
DEFAULT_NWD_C: float | None = None


class GaussSplit(nn.Module):
    """Predict a second instance from the Gaussian scale-space of a detection.

    Args:
        c_in: channels of the stride-4 feature map (P2 is 32 in YOLO26n-P2).
        hidden: width of the two hidden layers.
        patch: odd side of the sampled neighbourhood (7 -> 7x7).
        stride: feature stride of ``feat`` relative to the box coordinate space.
        compress: channels after the 1x1 compression of [fine, dog].
        dropout: dropout on the hidden activations; 0 disables.

    Shapes:
        feat:  (B, c_in, H, W)
        boxes: (N, 4) xyxy in input pixels
        score: (N,)
        batch: (N,) index of the image each detection belongs to
        ->     (N, 4) = [logit_second, dx, dy, log_size_ratio]
    """

    def __init__(
        self,
        c_in: int = 32,
        hidden: int = 64,
        patch: int = PATCH_SIDE,
        stride: int = 4,
        compress: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if patch < 1 or patch % 2 == 0:
            raise ValueError(f"patch must be a positive odd integer, got {patch}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if compress < 1:
            raise ValueError(f"compress must be >= 1, got {compress}")
        self.c_in = int(c_in)
        self.hidden = int(hidden)
        self.patch = int(patch)
        self.stride = int(stride)
        self.compress = int(compress)

        # G1: the GauSS operator, reused verbatim (two learnable scales).
        self.scale_space = LearnableGaussianScaleSpace(self.c_in)
        self.proj = nn.Conv2d(2 * self.c_in, self.compress, kernel_size=1, bias=True)

        # patch + (log w, log h, score)
        in_dim = self.compress * self.patch * self.patch + 3
        self.fc1 = nn.Linear(in_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 4)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.proj.bias)
        for lin in (self.fc1, self.fc2):
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            nn.init.zeros_(lin.bias)
        # Zero output layer + conservative bias: at step 0 the module is a no-op
        # (p(second) = 0.119, offsets 0, ratio 1).
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        with torch.no_grad():
            self.fc3.bias[0] = INIT_SECOND_BIAS

    # ------------------------------------------------------------- scale space
    def scale_space_patches(
        self, feat: Tensor, boxes: Tensor, batch: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the fine patch and the DoG patch, both ``(N, C, p, p)``."""
        l1, l2, _ = self.scale_space(feat)
        fine = gather_patches(l1, boxes, batch, self.patch, self.stride)
        coarse = gather_patches(l2, boxes, batch, self.patch, self.stride)
        n, _cpp = fine.shape
        p = self.patch
        fine = fine.view(n, self.c_in, p, p)
        coarse = coarse.view(n, self.c_in, p, p)
        return fine, fine - coarse

    # ----------------------------------------------------------------- forward
    def forward(
        self, feat: Tensor, boxes: Tensor, score: Tensor, batch: Tensor | None = None
    ) -> Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))
        if batch is None:
            batch = torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
        batch = batch.to(torch.long)

        fine, dog = self.scale_space_patches(feat, boxes, batch)
        patch = self.proj(torch.cat([fine, dog], dim=1))  # (N, C', p, p)

        wf = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-3) / self.stride
        hf = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-3) / self.stride
        geom = torch.stack(
            [torch.log(wf), torch.log(hf), score.clamp(0.0, 1.0)],
            dim=1,
        )

        x = torch.cat([patch.flatten(1), geom], dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ------------------------------------------------------------------ geometry
def boxes_to_gaussians(boxes: Tensor) -> tuple[Tensor, Tensor]:
    """xyxy boxes -> (mean (N,2), std (N,2)) of the Gaussian box model."""
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
    sx = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-3) * 0.5
    sy = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-3) * 0.5
    return torch.stack([cx, cy], dim=1), torch.stack([sx, sy], dim=1)


def wasserstein2(box_a: Tensor, box_b: Tensor) -> Tensor:
    """Squared 2-Wasserstein distance between Gaussian box models, ``(N,)``.

    For diagonal covariances ``W2^2 = ||mu_a - mu_b||^2 + ||sigma_a - sigma_b||^2``
    (closed form of the Bures/Wasserstein distance between axis-aligned
    Gaussians).
    """
    ma, sa = boxes_to_gaussians(box_a)
    mb, sb = boxes_to_gaussians(box_b)
    return (ma - mb).square().sum(dim=1) + (sa - sb).square().sum(dim=1)


def nwd(box_a: Tensor, box_b: Tensor, c: float | Tensor | None = None) -> Tensor:
    """Normalised Wasserstein distance similarity in (0, 1]; 1 means identical.

    ``NWD = exp(-sqrt(W2^2) / C)`` following arXiv:2110.13389. When ``c`` is
    None the per-call mean ``sqrt(area)`` of the two box sets is used, which is
    the paper's rule (C = average object size) computed online.
    """
    d = wasserstein2(box_a, box_b).clamp(min=0.0).sqrt()
    if c is None:
        def area(b: Tensor) -> Tensor:
            return (b[:, 2] - b[:, 0]).clamp(min=1e-3) * (b[:, 3] - b[:, 1]).clamp(min=1e-3)
        c = 0.5 * (area(box_a).sqrt() + area(box_b).sqrt()).mean().clamp(min=1e-3)
    return torch.exp(-d / c)


def offsets_to_boxes(src: Tensor, off: Tensor) -> Tensor:
    """Apply ``[dx, dy, log_size_ratio]`` to source boxes -> xyxy."""
    cx = (src[:, 0] + src[:, 2]) * 0.5
    cy = (src[:, 1] + src[:, 3]) * 0.5
    w = (src[:, 2] - src[:, 0]).clamp(min=1e-3)
    h = (src[:, 3] - src[:, 1]).clamp(min=1e-3)
    ratio = torch.exp(off[:, 2])
    nw = w * ratio
    nh = h * ratio
    ncx = cx + off[:, 0] * w
    ncy = cy + off[:, 1] * h
    return torch.stack([ncx - nw * 0.5, ncy - nh * 0.5, ncx + nw * 0.5, ncy + nh * 0.5], dim=1)


# ---------------------------------------------------------------------- loss
def gauss_split_loss(
    out: Tensor,
    targets: Tensor,
    valid: Tensor,
    src_boxes: Tensor,
    geom: str = "nwd",
    cls_weight: float = 1.0,
    reg_weight: float = 1.0,
    nwd_c: float | None = DEFAULT_NWD_C,
) -> tuple[Tensor, dict[str, float]]:
    """BCE on the multiplicity gate plus a geometric loss on the second box.

    Args:
        out: ``(N, 4)`` head output.
        targets: ``(N, 4)`` from ``build_pair_targets``.
        valid: ``(N,)`` bool; rows without a GT match are excluded.
        src_boxes: ``(N, 4)`` the source detections the offsets are relative to.
        geom: ``"nwd"`` (Gaussian Wasserstein, default) or ``"l1"`` (ablation
            baseline matching PairSplit).
    """
    if out.numel() == 0 or not bool(valid.any()):
        zero = out.sum() * 0.0
        return zero, {"cls": 0.0, "reg": 0.0}

    o = out[valid]
    t = targets[valid]
    cls = F.binary_cross_entropy_with_logits(o[:, 0], t[:, 0])

    pos = t[:, 0] > 0.5
    if not bool(pos.any()):
        return cls_weight * cls, {"cls": float(cls.detach()), "reg": 0.0}

    src = src_boxes[valid][pos]
    if geom == "l1":
        reg = (o[pos, 1:] - t[pos, 1:]).abs().mean()
    elif geom == "nwd":
        pred = offsets_to_boxes(src, o[pos, 1:])
        gt = offsets_to_boxes(src, t[pos, 1:])
        reg = (1.0 - nwd(pred, gt, nwd_c)).mean()
    else:
        raise ValueError(f"geom must be 'nwd' or 'l1', got {geom!r}")
    return cls_weight * cls + reg_weight * reg, {"cls": float(cls.detach()), "reg": float(reg.detach())}


def count_parameters(module: nn.Module) -> int:
    """Re-exported from :mod:`tinydet.modules.pair_split` (single definition)."""
    return _count_parameters(module)
