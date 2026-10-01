# NWD placement ablation on P3-only YOLO26n

Normalized Wasserstein Distance (NWD) is a bounding-box similarity metric. For boxes `(cx, cy, w, h)`, squared distance is `Δcx² + Δcy² + (Δw/2)² + (Δh/2)²`; similarity is `exp(-sqrt(distance)/C)`. We use `C=12.8` input-image pixels. The implementation shares one formula for assignment and regression. This constant is a starting point, not tuned for TinyPerson. See the [original NWD paper](https://arxiv.org/abs/2110.13389).

Ultralytics 8.4.164 calls `TaskAlignedAssigner.iou_calculation` from `get_box_metrics`. Its original overlap is nonnegative CIoU. NWD-TAL substitutes NWD there; TAL still computes `classification_score^0.5 * similarity^6`, filters centers inside GT, selects top-k, resolves conflicts, and normalizes quality-aware target scores. YOLO26 end-to-end loss uses one-to-many top-k 10 and one-to-one top-k 7 then top-k 1. Both branches receive the same metric. TAL receives predicted and GT `xyxy` boxes in input-image pixels, because `v8DetectionLoss.get_assigned_targets_and_loss` multiplies decoded boxes and anchor points by stride before assignment.

NWD regression replaces only `BboxLoss`'s weighted `(1 - CIoU)` box term with weighted `(1 - NWD)`. The native signed L1 term of YOLO26 (`reg_max=1`), classification loss, gains, branch weights and schedule remain. The regression wrapper multiplies its feature-grid boxes by per-anchor stride before NWD. Assignment and regression use `C=12.8` pixels. This is an adaptation of NWD to YOLOv26, not an exact reproduction of the original NWD detector implementation. NWD-RKA is not implemented.

| Variant | TAL similarity | Regression box term |
| --- | --- | --- |
| `baseline` | clamped CIoU | 1 - CIoU |
| `nwd_loss` | clamped CIoU | 1 - NWD |
| `nwd_tal` | NWD | 1 - CIoU |
| `nwd_full` | NWD | 1 - NWD |

All four variants use shared training settings, dataset, architecture, random initialization, seed, evaluation, and hardware from `configs/experiment/ablation_nwd.yaml`. Compare `nwd_loss - baseline`, `nwd_tal - baseline`, and `nwd_full - baseline`; multiple seeds would also permit an interaction estimate. Existing AP-at-IoU-threshold logging remains active.

Run on Modal only when ready to spend GPU time:

```bash
uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/ablation_nwd.yaml --parallel
```
