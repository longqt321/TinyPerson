# NWD placement ablation on P3-only YOLO26n

## Experimental scope and compute constraint

The NWD placement ablation and normalization-constant tuning in this research phase were intentionally performed on `yolo26n-p3-only.yaml`, not on the standard YOLO26n or YOLO26n-P2 architecture. This was a screening decision driven by limited GPU availability, training time, and compute budget: the P3-only detector is cheaper to iterate on, allowing the placement ablation and coarse `C` sensitivity study to be completed before spending substantially more resources on the stronger P2 model.

Consequently, every NWD result reported in this document is architecture-specific. A negative result on P3-only YOLO26n must not be stated as evidence that NWD is ineffective on YOLO26n generally, and the selected constant must not be assumed to be globally optimal across detection architectures. The P3-only experiments are used to select a plausible NWD formulation and candidate constant under a constrained screening budget. Transfer to the stronger YOLO26n-P2 architecture is evaluated separately with a controlled P2 baseline-versus-NWD experiment.

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

## Seed-42 observation

The 100-epoch seed-42 ablation does not support replacing native CIoU with NWD at `C=12.8` in the current P3-only YOLO26n setting.

| Variant | Precision | Recall | AP50 | mAP50-95 | Relative mAP50-95 vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.362869 | 0.316968 | 0.229974 | 0.065681 | — |
| `nwd_loss` | 0.332829 | 0.309145 | 0.219407 | 0.061454 | -6.44% |
| `nwd_tal` | 0.336769 | 0.323042 | 0.214735 | 0.056518 | -13.95% |
| `nwd_full` | 0.325143 | 0.301157 | 0.191910 | 0.049489 | -24.65% |

The ordering is `baseline > nwd_loss > nwd_tal > nwd_full` by mAP50-95. NWD regression alone is the closest variant to the baseline, while NWD in TAL produces the larger degradation. `nwd_tal` is the only variant with higher recall than baseline (+1.92% relative), but this comes with lower precision, AP50, and mAP50-95.

The TAL result is also consistent with the validation box objective: `baseline` and `nwd_tal` retain the same CIoU regression term, yet their final validation box losses are approximately 2.88 and 3.48 respectively. This suggests that the NWD-based assignment selects positives that are less favorable for precise CIoU localization under the current formulation. Box-loss magnitudes must not be compared directly between CIoU-regression and NWD-regression variants because the objectives have different scales.

Combining NWD assignment and NWD regression does not recover the degradation. `nwd_full` is the weakest variant, so the current result provides no evidence of a beneficial interaction between the two NWD placements.

This is a single-seed, P3-only result and `C=12.8` was not tuned for TinyPerson. The result therefore rejects the tested configuration on this screening architecture, not NWD in general and not NWD on YOLO26n-P2. The next controlled experiment keeps native TAL and screens the NWD regression constant on the same P3-only architecture before spending compute on a stronger detector. The coarse sweep uses `C ∈ {1.6, 3.2, 6.4, 12.8, 25.6}` with architecture, initialization protocol, augmentation, optimizer, L1 term, and evaluation fixed.

Run on Modal only when ready to spend GPU time:

```bash
uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/ablation_nwd.yaml --parallel
```
