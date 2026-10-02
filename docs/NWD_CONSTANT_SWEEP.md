# Phase 1: NWD regression constant sensitivity

## Objective

Test whether the negative result of the seed-42 NWD regression ablation is sensitive to the normalization constant `C`, without changing assignment, architecture, optimization, augmentation, or auxiliary L1 supervision.

The completed `C=12.8` run is the center reference. This screening adds only the two missing factor-of-two settings:

| Variant | C (input-image pixels) | TAL | Regression |
| --- | ---: | --- | --- |
| existing reference | 12.8 | native CIoU TAL | NWD |
| `nwd_loss_c6_4` | 6.4 | native CIoU TAL | NWD |
| `nwd_loss_c25_6` | 25.6 | native CIoU TAL | NWD |

## Protocol

The two new runs use seed 42 and 50 epochs for screening. All non-`C` settings are shared in `configs/experiment/nwd_constant_sweep.yaml`. `torch.compile` is disabled because the previous compiled Modal run encountered a TorchInductor/Triton CUDA illegal-instruction failure; compilation is not part of the research variable.

The primary metric is validation `mAP50-95`. Secondary diagnostics are precision, recall, AP50, and AP at IoU thresholds 0.60, 0.70, 0.75, and 0.80. Do not compare absolute NWD box-loss magnitude against the CIoU baseline because the objectives have different scales.

The existing 100-epoch `C=12.8` history should be compared at the same 50-epoch horizon for screening. Final endpoint comparisons against its 100-epoch value are not controlled comparisons.

## Decision rule

- If neither `C=6.4` nor `C=25.6` improves meaningfully over `C=12.8` at the matched horizon and all remain below the CIoU baseline, close the NWD branch as a negative result under this setup.
- If one setting materially closes the gap or exceeds the baseline, promote that setting to a full-length run and then compare baseline versus the selected NWD regression configuration across multiple seeds.
- Do not reintroduce NWD-TAL or tune localization weights during this phase.

## Run

```bash
uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/nwd_constant_sweep.yaml --parallel
```
