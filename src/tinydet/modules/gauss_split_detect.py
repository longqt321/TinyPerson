"""Pipeline-native GaussSplit head: a ``Detect`` subclass that hosts the module.

Why a ``Detect`` subclass instead of a separate layer
-----------------------------------------------------
Ultralytics assumes ``model.model[-1]`` **is** the detection head: ``init_criterion``,
``v8DetectionLoss.__init__``, ``bias_init`` and the stride probe all read it. Adding a
separate layer after ``Detect`` breaks those; adding one *before* ``Detect`` shifts its
index and invalidates every checkpoint key (``model.29.*`` becomes ``model.30.*``), so the
frozen baseline weights silently stop loading.

Subclassing keeps the graph and the state-dict keys identical to the baseline: the head's
parameters are extra keys (``model.29.head.*``) that simply do not exist in the baseline
checkpoint, so ``intersect_dicts`` loads the detector exactly as before and the head starts
from its own initialisation. The head's parameters are then

* counted by ``model.info()``, ``metadata.json`` and W&B,
* picked up by Ultralytics' optimizer (which is built before the criterion exists, so a
  head created inside a loss hook would never train),
* saved in ``best.pt`` / ``last.pt`` with the rest of the model, and
* preserved by ``AutoBackend``'s ``deepcopy`` at inference, because ``forward`` keeps
  caching the P2 map on the copy that actually runs.

The layer is a no-op for the detector: ``forward`` caches ``x[0]`` (P2, stride 4) and
delegates to ``Detect.forward``. The auxiliary loss and the extra boxes at inference are
wired in ``tinydet.models.ultralytics`` by patching Ultralytics' criterion and predictor —
the same pattern already used for the TAL tiny-floor.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.ops import nms
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.nms import non_max_suppression

from tinydet.modules.gauss_split import GaussSplit, gauss_split_loss
from tinydet.modules.pair_split import (
    MAX_PAIR_DIST_PX,
    PATCH_SIDE,
    PairSplit,
    build_pair_targets,
    pair_split_loss,
)

STRIDE = 4  # P2 stride of the A2-P2 model


class DetectGaussSplit(Detect):
    """``Detect`` head that also hosts the second-instance head on P2.

    Args:
        nc: number of classes.
        reg_max: DFL bins (1 disables DFL, as in the A2-P2 model).
        end2end: Ultralytics' end-to-end flag.
        ch: channels of the detection levels; ``ch[0]`` is P2.
        kind: ``"pair"`` (PairSplit head) or ``"gauss"`` (GaussSplit head).
        patch: odd side of the sampled neighbourhood (7 -> 7x7).
        hidden: width of the two hidden layers.
        patch_channels: 1x1 bottleneck channels for PairSplit (0 disables).
        compress: 1x1 compression channels for GaussSplit.
        geom: regression geometry, ``"l1"`` or ``"nwd"``.
        conf: confidence threshold used to select detections during training.
        max_dets: maximum detections per image used during training.
        iou_thr: IoU threshold for matching a detection to a GT.
        radius: GT-neighbour radius before the pixel cap.
        max_dist_px: pixel cap on the GT-neighbour radius.
        cls_weight / reg_weight: loss weights.
        gain: multiplier on the auxiliary loss before it is added to the detector loss.
        threshold: ``p(second)`` threshold used at inference.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple[int, ...] = (),
        kind: str = "pair",
        patch: int = PATCH_SIDE,
        hidden: int = 64,
        patch_channels: int = 6,
        compress: int = 4,
        geom: str = "l1",
        conf: float = 0.001,
        max_dets: int = 200,
        iou_thr: float = 0.5,
        radius: float = 1.5,
        max_dist_px: float = MAX_PAIR_DIST_PX,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        gain: float = 1.0,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        if kind not in {"pair", "gauss"}:
            raise ValueError(f"kind must be 'pair' or 'gauss', got {kind!r}")
        if patch % 2 != 1 or patch < 3:
            raise ValueError(f"patch must be an odd integer >= 3, got {patch}")
        c1 = int(ch[0])
        self.kind = str(kind)
        self.geom = str(geom)
        self.conf = float(conf)
        self.max_dets = int(max_dets)
        self.iou_thr = float(iou_thr)
        self.radius = float(radius)
        self.max_dist_px = float(max_dist_px)
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.gain = float(gain)
        self.threshold = float(threshold)
        self.head = (
            PairSplit(c_in=c1, patch=int(patch), hidden=int(hidden), patch_channels=int(patch_channels))
            if kind == "pair"
            else GaussSplit(c_in=c1, patch=int(patch), hidden=int(hidden), compress=int(compress))
        )
        self.feat: Tensor | None = None
        self.last_stats: dict[str, float] = {}

    def forward(self, x: list[Tensor]):
        """Cache the P2 map, then run the unmodified Detect head.

        The cache is detached: the head must not push gradients into the frozen
        detector, and a detached tensor is a graph leaf, so ultralytics'
        ``deepcopy`` in ``save_model``/``AutoBackend`` cannot fail on it.
        """
        self.feat = x[0].detach()
        return super().forward(x)

    def __getstate__(self):
        """Never persist the cached P2 map: it is transient (a stale tensor would
        bloat every checkpoint and could mismatch the next input's shape)."""
        state = dict(super().__getstate__())
        state["feat"] = None
        return state

    # ------------------------------------------------------------------ losses
    def aux_loss(self, preds: dict[str, dict[str, Tensor]], batch: dict[str, Tensor]) -> Tensor:
        """Second-instance loss, computed in fp32 even under AMP.

        The detections come from the *same* code path as inference
        (``Detect._inference`` + ``ultralytics.utils.nms.non_max_suppression`` on the
        branch the model uses at inference), so the head is supervised on the
        distribution it sees at eval time. fp32 is used because torchvision NMS and
        the BCE/Wasserstein terms are numerically fragile in fp16 while the head is
        small enough that the cast costs nothing.
        """
        device_type = next(self.parameters()).device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return self._aux_loss_fp32(preds, batch)

    def _aux_loss_fp32(self, preds: dict[str, dict[str, Tensor]], batch: dict[str, Tensor]) -> Tensor:
        branch = preds["one2one"] if self.end2end else preds["one2many"]
        with torch.no_grad():
            det = non_max_suppression(
                self._inference(branch).float(),
                conf_thres=self.conf,
                iou_thres=0.5,
                max_det=self.max_dets,
                nc=self.nc,
                end2end=self.end2end,
            )
        if self.feat is None:
            raise RuntimeError("DetectGaussSplit has not seen a forward pass yet")
        feat = self.feat.detach().float()
        h, w = feat.shape[2:]
        device = feat.device
        # Detections are in letterboxed-image pixels; the feature map is stride-4.
        img_w, img_h = w * STRIDE, h * STRIDE

        # batch["bboxes"] is normalised xywh on the letterboxed input.
        g = torch.cat([batch["batch_idx"].view(-1, 1), batch["bboxes"]], 1).to(device)
        cx, cy, bw, bh = (g[:, k] for k in (1, 2, 3, 4))
        gt_boxes = torch.stack(
            [
                (cx - bw / 2) * img_w,
                (cy - bh / 2) * img_h,
                (cx + bw / 2) * img_w,
                (cy + bh / 2) * img_h,
            ],
            dim=1,
        )
        gt_batch = g[:, 0].long()

        boxes_list, scores_list, batch_list = [], [], []
        for i, rows in enumerate(det):
            boxes_list.append(rows[:, :4])
            scores_list.append(rows[:, 4])
            batch_list.append(torch.full((rows.shape[0],), i, dtype=torch.long, device=device))
        boxes = torch.cat(boxes_list, 0)
        scores = torch.cat(scores_list, 0)
        batch_idx = torch.cat(batch_list, 0)

        targets, valid = build_pair_targets(
            boxes,
            batch_idx,
            gt_boxes,
            gt_batch,
            iou_thr=self.iou_thr,
            radius=self.radius,
            max_dist_px=self.max_dist_px,
        )
        out = self.head(feat, boxes, scores, batch_idx)
        if self.kind == "gauss":
            loss, parts = gauss_split_loss(
                out,
                targets,
                valid,
                boxes,
                geom=self.geom,
                cls_weight=self.cls_weight,
                reg_weight=self.reg_weight,
            )
        else:
            loss, parts = pair_split_loss(
                out, targets, valid, cls_weight=self.cls_weight, reg_weight=self.reg_weight
            )
        n_valid = float(valid.sum())
        self.last_stats = {
            **parts,
            "pos_rate": float((targets[:, 0] > 0.5).sum()) / n_valid if n_valid else 0.0,
            "n_det": float(boxes.shape[0]),
            # mean() of an empty tensor is NaN and would poison the epoch average.
            "p_second_mean": float(torch.sigmoid(out[:, 0]).mean().detach()) if out.shape[0] else 0.0,
        }
        # Per-detection mean, NOT multiplied by the batch size: the head's effective
        # learning rate then matches the pre-registered recipe (AdamW, lr 1e-3,
        # batch 64 patches) instead of drifting with `batch`.
        return loss * self.gain

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def add_second_boxes(self, preds: list[Tensor]) -> list[Tensor]:
        """Append the head's second boxes to each image's ``(N, 6)`` prediction."""
        if self.feat is None:
            raise RuntimeError("DetectGaussSplit has not seen a forward pass yet")
        feat = self.feat.float()
        out: list[Tensor] = []
        for i, pred in enumerate(preds):
            if pred.numel() == 0:
                out.append(pred)
                continue
            boxes, scores = pred[:, :4].float(), pred[:, 4].float()
            batch = torch.full((boxes.shape[0],), i, dtype=torch.long, device=boxes.device)
            o = self.head(feat, boxes, scores, batch)
            extra, extra_score, extra_cls = PairSplit.decode(
                boxes, scores, o, threshold=self.threshold, cols=pred[:, 5:6]
            )
            if extra.shape[0] == 0:
                out.append(pred)
                continue
            extra_full = torch.cat([extra, extra_score.unsqueeze(1), extra_cls], dim=1)
            n_src = pred.shape[0]
            merged = torch.cat([pred, extra_full], dim=0)
            # Suppress only the *new* boxes: the detector's own detections are kept
            # verbatim, so a head that emits nothing leaves the output bit-identical
            # (and the detector's NMS threshold is never silently tightened).
            keep = nms(merged[:, :4], merged[:, 4], 0.5)
            keep = torch.cat([torch.arange(n_src, device=keep.device), keep[keep >= n_src]])
            out.append(merged[keep])
        return out


def find_gauss_split_head(model: nn.Module) -> DetectGaussSplit | None:
    """Return the DetectGaussSplit head of a DetectionModel, or None if absent."""
    for module in model.modules():
        if isinstance(module, DetectGaussSplit):
            return module
    return None
