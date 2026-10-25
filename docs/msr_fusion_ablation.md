# MSR fusion ablation

`configs/experiment/ablation_msr_fusion.yaml` compares four MSR fusion modes on the same A2 P2 graph:

- `adaptive`: current default; per-scale sigmoid gates normalized by their sum.
- `mean`: fixed equal weight for every scale.
- `max`: elementwise maximum across scales.
- `sk_softmax`: channel-wise softmax across scales.

The model graph, P2 attachment, dataset, seed, optimizer, epochs, image size, batch size, and augmentation stay fixed. Only the final `fusion_mode` constructor argument changes. Diagnostics collect forward statistics and sampled gradients every 100 training steps. Existing A2 no-MSR result remains the external baseline; this experiment does not train it again.

This experiment isolates fusion behavior within MSR. It does not ablate local statistics, signed evidence projection, feature modulation, or residual refinement.

Run from the repository root in the active Modal environment:

```fish
modal run --detach modal/app.py::compare --config configs/experiment/ablation_msr_fusion.yaml --parallel
```

Each arm gets a new run directory. Diagnostics are written to `<run>/diagnostics/msr_metrics.jsonl` and the existing W&B run under `msr/...` keys. Training has not been started by creating this config.
