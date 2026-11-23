"""PairSplit: a post-hoc instance-splitting head for crowded tiny objects.

Motivation (measured in this repo, see TINY_RESEARCH_LOG.md R20-R22)
-------------------------------------------------------------------
On TinyPerson, moving inference to 1:1 resolution (tiled protocol) fixes most of
the small-object failure, but a residual gap survives: at matched effective size,
GTs whose nearest neighbour is < 10 px away are detected at 59.1% versus 76.0%
for isolated GTs (tiled protocol; full-frame gap is +33.0 pp, tiled +17.0 pp).
The cleanest cell is s_eff 10-16 px at 1:1, where resolution is not the limiting
factor: 52.5% (d_nn < 6 px) versus 77.5% (d_nn >= 48 px).

That residual is NOT an NMS artefact. Among close-pair misses in the tiled
protocol, only 3.7% have a box that was credited to the neighbour; 96.3% have no
box reaching IoU 0.5 with the missed GT. But the model is not silent either: the
nearest detection sits 0.37 GT-sizes away, overlaps the missed GT at a median IoU
of 0.319, and 53.5% of misses have a detection at IoU >= 0.3.

So the failure is representational: one location emits one box, and two people
inside one response collapse to one detection. Every mainstream anchor-free head
(YOLO, FCOS, CenterNet) encodes exactly one object per location, so no decoding
rule can recover the second instance.

Mechanism
---------
PairSplit reads the local P2 neighbourhood around an existing detection and
predicts whether a SECOND person is present in that neighbourhood, plus where it
is relative to the first::

    patch  = grid_sample(P2, 7x7 around the detection centre)   # (C, 7, 7)
    x      = [patch.flatten(), log w, log h, score]             # C*49 + 3
    out    = MLP(x)                                             # 4
    out    = [logit_second, dx, dy, log_size_ratio]

``dx, dy`` are in units of the detection's own width/height, so the head is
scale-invariant by construction: a 5 px pair and a 50 px pair produce the same
targets. The second box is emitted at ``centre + (dx*w, dy*h)`` with side lengths
scaled by ``exp(log_size_ratio)``.

Design properties
-----------------
* **Post-hoc.** It does not touch the backbone, the Detect head, the assigner or
  the loss. It can be attached to any detector that exposes a stride-4 feature
  map and a detection list, which is what makes it comparable across papers.
* **Targets are derived from existing boxes**, no new annotation: for a detection
  matched to GT ``g`` at IoU >= 0.5, the target is ``1`` plus the offset/size of
  the nearest other GT within ``min(1.5 * max(w, h), 10 px)`` of ``g``'s centre,
  else ``0``. The 10 px cap keeps the supervision inside the head's receptive
  window (patch 7 at stride 4 = +-12 px) and restricts it to the close-pair
  regime the detector actually fails on.
* **Conservative at init.** The output layer is zero-initialised with a negative
  second-instance bias, so at step 0 the head predicts ``p(second) = 0.119`` and
  zero offsets: attaching it changes nothing until it has learned.
* **Cheap.** 23,690 trainable parameters at the default configuration
  (C=32, patch=7, hidden=64, 1x1 bottleneck 32->6), i.e. +0.9% over the baseline
  A2 model (2,515,976), versus the 34,666-parameter GauSS module (+1.38%) it
  replaces as the module of record.

Provenance
----------
Self-tests in ``tests/test_pair_split.py`` (11) and
``tests/test_pair_split_targets.py`` (11), all passing: shapes, parameter count, init
behaviour, gather semantics (a delta placed at a known feature cell must appear at the
expected patch offset), batch-order restoration, gradient flow to every parameter, a
numpy reference check of the MLP sub-module, the target builder and the loss. Run with
``pytest tests/``; locally pytest is absent, so they were executed with
``scratch/run_pair_split_tests.py`` (pytest-free harness, 22/22).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ultralytics.utils.metrics import box_iou

# Negative logit so that at initialisation the head is conservative:
# sigmoid(-2.0) = 0.1192.
INIT_SECOND_BIAS: float = -2.0

# Upper bound on the centre distance of a supervised pair, in input pixels.
# Measured on the train split (9,330 pairs): the close-pair gap that survives
# the 1:1 tiled protocol is concentrated below 10 px (median pair distance
# 14.1 px, 32% <= 10 px). Pairs wider than this are already detected, so
# supervising them would only teach the head to emit duplicates.
MAX_PAIR_DIST_PX: float = 10.0

# Patch side that guarantees every supervised pair is inside the receptive
# window: half-extent = (PATCH - 1) / 2 * STRIDE = 12 px at patch 7, stride 4.
# At patch 3 the half-extent is 6 px and only 14.1% of pairs were visible,
# which is why the first B0 run could not learn the regression (L-020).
PATCH_SIDE: int = 7


# --------------------------------------------------------------------- gather
def gather_patches(feat: Tensor, boxes: Tensor, batch: Tensor, patch: int, stride: int) -> Tensor:
    """Sample a ``patch x patch`` neighbourhood around each box centre.

    ``feat`` is ``(B, C, H, W)`` with stride ``stride`` relative to the box
    coordinate space; ``boxes`` are xyxy in input pixels. Returns
    ``(N, C * patch * patch)``. Uses ``grid_sample`` with
    ``padding_mode='zeros'`` so detections at the image border are handled
    without special cases.
    """
    b, c, h, w = feat.shape
    half = (patch - 1) // 2

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5 / stride
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5 / stride
    # align_corners=False: feature cell i maps to (2i + 1) / size - 1
    gx = (2.0 * cx + 1.0) / w - 1.0
    gy = (2.0 * cy + 1.0) / h - 1.0

    offs = torch.arange(-half, half + 1, dtype=feat.dtype, device=feat.device)
    gx = gx.view(-1, 1, 1) + offs.view(1, patch, 1) * (2.0 / w)  # (N, p, 1)
    gy = gy.view(-1, 1, 1) + offs.view(1, 1, patch) * (2.0 / h)  # (N, 1, p)
    grid = torch.stack(
        [gx.expand(-1, patch, patch), gy.expand(-1, patch, patch)], dim=-1
    )  # (N, p, p, 2)

    chunks = []
    for i in range(b):
        sel = batch == i
        if not bool(sel.any()):
            continue
        g = grid[sel]  # (n, p, p, 2)
        n = g.shape[0]
        out = F.grid_sample(
            feat[i : i + 1],
            g.reshape(1, n * patch * patch, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # (1, c, n*p*p, 1)
        chunks.append(out.reshape(c, n, patch * patch).permute(1, 0, 2).reshape(n, c * patch * patch))

    if not chunks:
        return feat.new_zeros((0, c * patch * patch))
    patches = torch.cat(chunks, dim=0)
    # restore the caller's ordering (the loop above groups by batch index)
    order = torch.argsort(batch, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return patches[inverse]


class PairSplit(nn.Module):
    """Predict a second instance from the P2 neighbourhood of a detection.

    Args:
        c_in: channels of the stride-4 feature map (P2 is 32 in YOLO26n-P2).
        hidden: width of the two hidden layers.
        patch: odd side of the sampled neighbourhood (7 -> 7x7).
        stride: feature stride of ``feat`` relative to the box coordinate space
            (4 for P2). Boxes are given in the *input pixel* space and divided by
            this value internally.
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
        dropout: float = 0.0,
        patch_channels: int = 6,
    ) -> None:
        super().__init__()
        if patch < 1 or patch % 2 == 0:
            raise ValueError(f"patch must be a positive odd integer, got {patch}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.c_in = int(c_in)
        self.hidden = int(hidden)
        self.patch = int(patch)
        self.stride = int(stride)
        self.patch_channels = int(patch_channels)

        # A 1x1 conv compresses the patch channels before flattening: without it
        # a patch-7 window would be 32*49 = 1568 features and the MLP alone would
        # cost ~100k parameters. With the bottleneck the whole head stays ~23k,
        # the same budget as the original patch-3 design.
        self.proj = (
            nn.Conv2d(self.c_in, self.patch_channels, 1)
            if self.patch_channels and self.patch_channels != self.c_in
            else nn.Identity()
        )
        in_dim = (self.patch_channels or self.c_in) * self.patch * self.patch + 3  # patch + (log w, log h, score)
        self.fc1 = nn.Linear(in_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)
        self.fc3 = nn.Linear(self.hidden, 4)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for lin in (self.fc1, self.fc2):
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            nn.init.zeros_(lin.bias)
        if isinstance(self.proj, nn.Conv2d):
            # Identity init: pass the patch through unchanged so the head starts
            # as the same function it was with the raw features.
            nn.init.eye_(self.proj.weight.view(self.proj.weight.shape[0], -1))
            nn.init.zeros_(self.proj.bias)
        # Zero output layer + conservative bias: at step 0 the module is a no-op
        # (p(second) = 0.119, offsets 0, ratio 1).
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        with torch.no_grad():
            self.fc3.bias[0] = INIT_SECOND_BIAS

    # ------------------------------------------------------------------ gather
    def _gather_patches(self, feat: Tensor, boxes: Tensor, batch: Tensor) -> Tensor:
        """Sample a ``patch x patch`` neighbourhood around each box centre."""
        return gather_patches(feat, boxes, batch, self.patch, self.stride)

    # ----------------------------------------------------------------- forward
    def forward(self, feat: Tensor, boxes: Tensor, score: Tensor, batch: Tensor | None = None) -> Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0, 4))
        if batch is None:
            batch = torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
        batch = batch.to(torch.long)

        patches = self._gather_patches(feat, boxes, batch)
        n = patches.shape[0]
        patches = patches.view(n, self.c_in, self.patch, self.patch)
        if isinstance(self.proj, nn.Conv2d):
            patches = self.proj(patches)
        patches = patches.flatten(1)

        wf = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-3) / self.stride
        hf = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-3) / self.stride
        geom = torch.stack([torch.log(wf), torch.log(hf), score.clamp(0.0, 1.0)], dim=1)

        x = torch.cat([patches, geom], dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    # ------------------------------------------------------------------ decode
    @staticmethod
    def decode(
        boxes: Tensor,
        score: Tensor,
        out: Tensor,
        threshold: float = 0.5,
        max_ratio: float = 3.0,
        cols: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Turn head outputs into an extra detection per accepted location.

        ``max_ratio`` bounds ``exp(log_size_ratio)`` so a mis-initialised or
        adversarial head cannot emit absurd boxes.

        ``cols`` (optional) is an ``(N, k)`` tensor of per-detection columns
        (e.g. the class index) that is filtered with the same mask and returned
        as a third element, so callers can build a full ``(N, 6)`` row without
        recomputing the acceptance mask.
        """
        if out.numel() == 0:
            empty = (boxes.new_zeros((0, 4)), score.new_zeros((0,)))
            return (*empty, cols.new_zeros((0, cols.shape[1]))) if cols is not None else empty
        p = torch.sigmoid(out[:, 0])
        keep = p > threshold
        if not bool(keep.any()):
            empty = (boxes.new_zeros((0, 4)), score.new_zeros((0,)))
            return (*empty, cols.new_zeros((0, cols.shape[1]))) if cols is not None else empty

        b = boxes[keep]
        o = out[keep]
        cx = (b[:, 0] + b[:, 2]) * 0.5
        cy = (b[:, 1] + b[:, 3]) * 0.5
        w = (b[:, 2] - b[:, 0]).clamp(min=1e-3)
        h = (b[:, 3] - b[:, 1]).clamp(min=1e-3)

        ratio = torch.exp(o[:, 3].clamp(-math.log(max_ratio), math.log(max_ratio)))
        nw = w * ratio
        nh = h * ratio
        ncx = cx + o[:, 1] * w
        ncy = cy + o[:, 2] * h

        extra = torch.stack(
            [ncx - nw * 0.5, ncy - nh * 0.5, ncx + nw * 0.5, ncy + nh * 0.5], dim=1
        )
        decoded = (extra, score[keep] * p[keep])
        return (*decoded, cols[keep]) if cols is not None else decoded


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# --------------------------------------------------------------------- targets


def build_pair_targets(
    boxes: Tensor,
    batch: Tensor,
    gt_boxes: Tensor,
    gt_batch: Tensor,
    iou_thr: float = 0.5,
    radius: float = 1.5,
    max_dist_px: float = MAX_PAIR_DIST_PX,
) -> tuple[Tensor, Tensor]:
    """Derive the second-instance target from existing GT, no new annotation.

    For each detection matched to a GT ``g`` at IoU >= ``iou_thr``:

    * if another GT ``h`` of the same image has its centre within
      ``min(radius * max(w_g, h_g), max_dist_px)`` of ``g``'s centre, the target is
      ``[1, (cx_h-cx_g)/w_g, (cy_h-cy_g)/h_g, log(sqrt(area_h/area_g))]``
      (the nearest such ``h`` wins);
    * otherwise the target is ``[0, 0, 0, 0]``.

    Returns ``(targets (N,4), valid (N,))``; ``valid`` is 0 for detections with
    no GT match, and those rows must be excluded from the loss.
    """
    n = boxes.shape[0]
    targets = boxes.new_zeros((n, 4))
    valid = boxes.new_zeros((n,), dtype=torch.bool)
    if n == 0 or gt_boxes.numel() == 0:
        return targets, valid

    for i in torch.unique(batch).tolist():
        det_idx = (batch == i).nonzero(as_tuple=True)[0]
        gt_idx = (gt_batch == i).nonzero(as_tuple=True)[0]
        if det_idx.numel() == 0 or gt_idx.numel() == 0:
            continue
        d = boxes[det_idx]
        g = gt_boxes[gt_idx]
        iou = box_iou(d, g)
        best_iou, best_g = iou.max(dim=1)
        matched = best_iou >= iou_thr
        valid[det_idx[matched]] = True

        gc = (g[:, :2] + g[:, 2:]) * 0.5
        gw = (g[:, 2] - g[:, 0]).clamp(min=1e-3)
        gh = (g[:, 3] - g[:, 1]).clamp(min=1e-3)
        ga = gw * gh

        for row, gi in zip(det_idx[matched].tolist(), best_g[matched].tolist(), strict=True):
            dist = torch.linalg.vector_norm(gc - gc[gi], dim=1)
            dist[gi] = float("inf")
            # The diagnosed failure mode is a pair closer than MAX_PAIR_DIST_PX.
            # Capping the radius there keeps the head on the region where the
            # detector actually collapses two people into one box; wider pairs
            # are already detected and would only add false positives at eval.
            limit = torch.minimum(
                radius * torch.maximum(gw[gi], gh[gi]),
                torch.full_like(gw[gi], max_dist_px),
            )
            nearest = int(dist.argmin())
            if not bool(dist[nearest] <= limit):
                continue  # isolated: target stays [0,0,0,0]
            targets[row, 0] = 1.0
            targets[row, 1] = (gc[nearest, 0] - gc[gi, 0]) / gw[gi]
            targets[row, 2] = (gc[nearest, 1] - gc[gi, 1]) / gh[gi]
            targets[row, 3] = 0.5 * torch.log(ga[nearest] / ga[gi])
    return targets, valid


def pair_split_loss(
    out: Tensor,
    targets: Tensor,
    valid: Tensor,
    cls_weight: float = 1.0,
    reg_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """BCE on ``logit_second`` plus masked L1 on the offsets.

    The regression term is masked to positives only: an isolated detection has no
    second instance to point at, so its offsets carry no supervision.
    """
    if out.numel() == 0 or not bool(valid.any()):
        zero = out.sum() * 0.0
        return zero, {"cls": 0.0, "reg": 0.0}
    o = out[valid]
    t = targets[valid]
    cls = F.binary_cross_entropy_with_logits(o[:, 0], t[:, 0])
    pos = t[:, 0] > 0.5
    reg = (o[pos, 1:] - t[pos, 1:]).abs().mean() if bool(pos.any()) else o.sum() * 0.0
    return cls_weight * cls + reg_weight * reg, {"cls": float(cls.detach()), "reg": float(reg.detach())}
