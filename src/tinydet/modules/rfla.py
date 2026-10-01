import math

import torch
import torch.nn.functional as F
from torch import nn


def receptive_field_similarity(
    points: torch.Tensor,
    sigma: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    half = ((boxes[:, 2:] - boxes[:, :2]).float() / 2).clamp_min(1e-6)
    centers = (boxes[:, 2:] + boxes[:, :2]).float() / 2
    delta = points.float()[None] - centers[:, None]
    variance_ratio = (sigma.float()[None, :, None] / half[:, None]).square()
    kl = 0.5 * (variance_ratio + (delta / half[:, None]).square() - 1 - variance_ratio.log()).sum(
        -1
    )
    return 1 / (1 + kl.clamp_min(0))


class RFLAAssigner(nn.Module):
    def __init__(
        self,
        num_classes: int,
        topk: int = 3,
        extra_topk: int = 1,
        shrink: float = 0.9,
        chunk_size: int = 64,
        one_to_one: bool = False,
        inside_only: bool = False,
    ):
        super().__init__()

        if (
            type(topk) is not int
            or topk < 1
            or type(extra_topk) is not int
            or extra_topk < 0
            or type(chunk_size) is not int
            or chunk_size < 1
            or not math.isfinite(shrink)
            or not 0 < shrink <= 1
        ):
            raise ValueError("RFLA requires topk>=1, extra_topk>=0, chunk_size>=1, 0<shrink<=1")

        self.num_classes = num_classes
        self.topk = 1 if one_to_one else topk
        self.extra_topk = 0 if one_to_one else extra_topk
        self.shrink = shrink
        self.chunk_size = chunk_size
        self.inside_only = inside_only
        self.sigma = None

    def _candidate_mask(
        self,
        points: torch.Tensor,
        sigma: torch.Tensor,
        boxes: torch.Tensor,
        k: int,
        blocked: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_gt = boxes.shape[0]
        num_anchors = points.shape[0]

        candidates = torch.zeros(
            (num_gt, num_anchors),
            dtype=torch.bool,
            device=points.device,
        )

        if k == 0 or num_gt == 0 or num_anchors == 0:
            return candidates

        selectable = min(k, num_anchors)

        for start in range(0, num_gt, self.chunk_size):
            end = min(start + self.chunk_size, num_gt)
            chunk = boxes[start:end]

            similarity = receptive_field_similarity(points, sigma, chunk)

            if self.inside_only:
                inside = (
                    (points[None] > chunk[:, None, :2]) & (points[None] < chunk[:, None, 2:])
                ).all(-1)
                similarity = similarity.masked_fill(~inside, -torch.inf)

            if blocked is not None:
                similarity = similarity.masked_fill(
                    blocked[None],
                    -torch.inf,
                )

            values, indices = similarity.topk(
                selectable,
                dim=-1,
                largest=True,
            )

            valid = torch.isfinite(values)

            rows = torch.arange(
                end - start,
                device=points.device,
            )[:, None].expand_as(indices)

            candidates[
                rows[valid] + start,
                indices[valid],
            ] = True

        return candidates

    @staticmethod
    def _resolve(
        candidates: torch.Tensor,
        areas: torch.Tensor,
    ) -> torch.Tensor:
        num_gt, num_anchors = candidates.shape

        selected = torch.full(
            (num_anchors,),
            num_gt,
            dtype=torch.long,
            device=candidates.device,
        )

        if not candidates.any():
            return selected

        priority = areas.argsort(stable=True)
        rank = torch.empty_like(priority)
        rank[priority] = torch.arange(
            num_gt,
            device=candidates.device,
        )

        pair_priority = torch.where(
            candidates,
            rank[:, None],
            num_gt,
        )

        winning_priority, winning_gt = pair_priority.min(dim=0)
        valid = winning_priority < num_gt

        selected[valid] = winning_gt[valid]

        return selected

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ):
        batch, anchors, _ = pd_scores.shape

        labels = torch.full(
            (batch, anchors),
            self.num_classes,
            device=pd_scores.device,
            dtype=torch.long,
        )
        boxes = torch.zeros_like(pd_bboxes)
        scores = torch.zeros_like(pd_scores)
        foreground = torch.zeros(
            (batch, anchors),
            device=pd_scores.device,
            dtype=torch.bool,
        )
        indices = torch.zeros(
            (batch, anchors),
            device=pd_scores.device,
            dtype=torch.long,
        )

        if self.sigma is None or self.sigma.shape != (anchors,):
            raise ValueError("Set assigner.sigma to the per-point RF standard deviation in pixels")

        for image in range(batch):
            valid = mask_gt[image, :, 0].bool().nonzero().flatten()

            if valid.numel() == 0:
                continue

            gt = gt_bboxes[image, valid].float()
            areas = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

            stage1_candidates = self._candidate_mask(
                anc_points,
                self.sigma,
                gt,
                self.topk,
            )

            stage1 = self._resolve(
                stage1_candidates,
                areas,
            )

            stage1_fg = stage1 < gt.shape[0]

            selected = stage1.clone()

            if self.extra_topk > 0:
                stage2_candidates = self._candidate_mask(
                    anc_points,
                    self.sigma * self.shrink,
                    gt,
                    self.extra_topk,
                    blocked=stage1_fg,
                )

                stage2 = self._resolve(
                    stage2_candidates,
                    areas,
                )

                stage2_fg = stage2 < gt.shape[0]
                add = ~stage1_fg & stage2_fg
                selected[add] = stage2[add]

            fg = selected < gt.shape[0]

            if not fg.any():
                continue

            original = valid[selected[fg]]

            foreground[image] = fg
            indices[image, fg] = original

            assigned_labels = gt_labels[
                image,
                original,
                0,
            ].long()

            labels[image, fg] = assigned_labels
            boxes[image, fg] = gt_bboxes[image, original]
            scores[image, fg] = F.one_hot(
                assigned_labels,
                self.num_classes,
            ).to(scores.dtype)

        return labels, boxes, scores, foreground, indices
