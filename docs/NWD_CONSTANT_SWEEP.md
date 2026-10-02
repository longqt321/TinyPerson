# Phase 1: NWD regression constant sensitivity

## Objective and scope

Test whether the negative result of the seed-42 NWD regression ablation is sensitive to the normalization constant `C`, without changing assignment, architecture, optimization, augmentation, or auxiliary L1 supervision.

This constant sweep is performed exclusively on `yolo26n-p3-only.yaml`. P3-only is used as a compute-efficient screening architecture because GPU availability, training time, and experiment budget are limited. Running the full placement study and five-point constant sweep directly on YOLO26n-P2 would substantially increase the cost of exploratory tuning. The purpose of this phase is therefore to identify a reasonable NWD formulation and candidate `C` cheaply before testing transfer on the stronger P2 detector.

The selected `C` is consequently a P3-only screening result, not a claim of a universal optimum for YOLO26n, YOLO26n-P2, or TinyPerson detection in general. Likewise, failure to outperform the P3-only CIoU baseline cannot by itself establish that NWD is ineffective on a different detection architecture.

The complete coarse factor-of-two sweep covers:

| Variant | C (input-image pixels) | Horizon | TAL | Regression |
| --- | ---: | ---: | --- | --- |
| `nwd_loss_c1_6` | 1.6 | 50 epochs | native CIoU TAL | NWD |
| `nwd_loss_c3_2` | 3.2 | 50 epochs | native CIoU TAL | NWD |
| `nwd_loss_c6_4` | 6.4 | 50 epochs | native CIoU TAL | NWD |
| existing reference | 12.8 | 100 epochs; compare at epoch 50 | native CIoU TAL | NWD |
| `nwd_loss_c25_6` | 25.6 | 50 epochs | native CIoU TAL | NWD |

## Protocol

All screening comparisons use seed 42, the P3-only architecture, and the same 50-epoch horizon. All non-`C` settings remain fixed. `torch.compile` is disabled because compilation is not part of the research variable and a previous compiled Modal run encountered a TorchInductor/Triton CUDA illegal-instruction failure.

The primary metric is validation `mAP50-95`. Secondary diagnostics are precision, recall, AP50, and AP at IoU thresholds 0.60, 0.70, 0.75, and 0.80. NWD box-loss magnitudes across different constants are not directly comparable because changing `C` changes the scale and sensitivity of `1 - NWD` itself.

The existing 100-epoch `C=12.8` run is interpreted at epoch 50 for screening; its epoch-100 endpoint is not used to rank it against 50-epoch runs.

## Observation

The sweep brackets the useful region sufficiently for the P3-only screening experiment. Reducing `C` from 25.6 toward 6.4 improves the early-training curves, with `C=6.4` generally stronger than `C=25.6` and the matched-horizon `C=12.8` reference across the main detection metrics. Extending the sweep downward to `C=3.2` and `C=1.6` does not produce a consistent additional improvement: their precision, recall, AP50, and mAP50-95 curves largely overlap the other small-`C` runs through the 50-epoch screening horizon.

High-IoU metrics such as AP0.80-AP0.95 remain low and noisy for all screened constants. Isolated spikes at these thresholds are therefore not treated as evidence for selecting `C=1.6` or `C=3.2`.

The result does not support continuing the P3-only constant sweep below `C=1.6`. Within the tested range, `C=6.4` is retained as the candidate setting because it reaches the strongest overall screening behavior without evidence that further reducing the constant gives a systematic gain.

This screening result does not establish that NWD outperforms native CIoU, nor that `C=6.4` is optimal on YOLO26n-P2. The previous 100-epoch seed-42 P3-only experiment favored its baseline over NWD with `C=12.8`. The next question is whether the candidate NWD formulation transfers to the stronger P2 architecture when both P2 variants are trained under the same controlled protocol.

## Decision

- Stop tuning the NWD normalization constant on the compute-efficient P3-only screening architecture after `C ∈ {1.6, 3.2, 6.4, 12.8, 25.6}`.
- Retain `C=6.4` as the candidate value selected by P3-only screening, not as a universal optimum.
- Do not tune localization weight, augmentation, learning rate, or reintroduce NWD-TAL as part of this selection step.
- Validate transfer with a controlled YOLO26n-P2 baseline-versus-NWD `C=6.4` experiment before deciding whether full-length or multi-seed NWD experiments are justified.
- Interpret all current NWD tuning conclusions as conditional on the P3-only screening architecture and the available compute budget.

## Run

```bash
uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/nwd_constant_sweep.yaml --parallel
```
