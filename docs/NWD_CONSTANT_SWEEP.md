# Phase 1: NWD regression constant sensitivity

## Objective

Test whether the negative result of the seed-42 NWD regression ablation is sensitive to the normalization constant `C`, without changing assignment, architecture, optimization, augmentation, or auxiliary L1 supervision.

The initial screening covered `C=6.4`, the existing `C=12.8` reference, and `C=25.6`. At the matched 50-epoch horizon, the curves indicate that `C=6.4` is generally stronger than `C=12.8`, while `C=25.6` is weaker. This motivates extending the coarse factor-of-two sweep downward before selecting a constant for full-length or multi-seed training.

The complete screened range is therefore:

| Variant | C (input-image pixels) | Status | TAL | Regression |
| --- | ---: | --- | --- | --- |
| `nwd_loss_c1_6` | 1.6 | new screening run | native CIoU TAL | NWD |
| `nwd_loss_c3_2` | 3.2 | new screening run | native CIoU TAL | NWD |
| `nwd_loss_c6_4` | 6.4 | completed | native CIoU TAL | NWD |
| existing reference | 12.8 | completed | native CIoU TAL | NWD |
| `nwd_loss_c25_6` | 25.6 | completed | native CIoU TAL | NWD |

## Protocol

The new `C=1.6` and `C=3.2` runs use seed 42 and 50 epochs for screening. All non-`C` settings remain fixed in `configs/experiment/nwd_constant_sweep.yaml`. `torch.compile` remains disabled because compilation is not part of the research variable and a previous compiled Modal run encountered a TorchInductor/Triton CUDA illegal-instruction failure.

The primary metric is validation `mAP50-95`. Secondary diagnostics are precision, recall, AP50, and AP at IoU thresholds 0.60, 0.70, 0.75, and 0.80. NWD box-loss magnitudes across different constants are not directly comparable because changing `C` changes the scale and sensitivity of `1 - NWD` itself.

All constant comparisons must use the same 50-epoch horizon. The existing 100-epoch `C=12.8` run must therefore be read at epoch 50 for this screening rather than compared by its final endpoint.

## Decision rule

- If performance peaks around `C=6.4` and decreases again at `C=3.2` or `C=1.6`, stop the constant sweep and promote the best setting to full-length evaluation.
- If performance continues improving monotonically toward smaller `C`, the optimum has not yet been bracketed; only then consider one additional lower factor-of-two setting.
- If no NWD constant materially closes the gap to the CIoU baseline, close the NWD regression branch as a negative result under this setup.
- Do not reintroduce NWD-TAL or tune localization weights during this phase.

## Run

```bash
uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/nwd_constant_sweep.yaml --parallel
```
