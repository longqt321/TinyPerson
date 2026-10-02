# NWD regression validation on YOLO26n-P2

## Question

The existing NWD placement and constant experiments were performed on the custom P3-only YOLO26n architecture. They therefore do not establish whether the same result transfers to the stronger P2 detector used for TinyPerson.

This experiment asks one controlled question:

> Does NWD regression with the selected `C=6.4` improve YOLO26n-P2 when compared with the same P2 architecture using its native localization objective?

## Design

| Variant | Architecture | TAL | Regression |
| --- | --- | --- | --- |
| `p2_baseline` | `yolo26n_p2.yaml` | native | native |
| `p2_nwd_c6_4` | `yolo26n_p2.yaml` | native | NWD, `C=6.4` |

Both models train from scratch. Architecture, initialization protocol, seed, dataset, optimizer, augmentation, L1 supervision, training horizon, evaluation, and hardware are held fixed. NWD-TAL is not enabled.

The initial comparison uses seed 42 and 50 epochs. This is a screening experiment, not a multi-seed confirmation. The primary metric is validation mAP50-95; AP50, precision, recall, and AP at individual IoU thresholds are secondary diagnostics.

## Interpretation

- If `p2_nwd_c6_4` clearly remains below `p2_baseline`, the previous P3-only negative result has transferred to the stronger P2 architecture and there is little justification for further NWD tuning before moving to another hypothesis.
- If `p2_nwd_c6_4` matches or exceeds `p2_baseline`, NWD appears architecture-dependent. The next step should test that interaction explicitly rather than generalizing the P3-only conclusion.
- Do not compare NWD and native box-loss magnitudes directly because the objectives have different scales.
- Do not retune `C`, loss weight, augmentation, or learning rate during this experiment.

## Run

```bash
uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/ablation_nwd_p2.yaml --parallel
```
