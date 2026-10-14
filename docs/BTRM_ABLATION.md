# BTRM P3 ablation (A0 / A2 / A3)

## Research question

Does removing local affine background-tangent variation improve extremely tiny
person detection **beyond** ordinary background-normalized center contrast?

- **A0**: YOLO26n P3/P4/P5, identity P3 gate (true baseline).
- **A2**: identical architecture, normalized center-minus-ring evidence.
- **A3**: identical architecture, BTRM ridge residual, same ring statistics.

All three share `configs/model/yolo26n-btrm.yaml`. Layer **17** is an explicit
`nn.Identity` gate between P3 layer 16 and the Detect head. P4's next
downsampling layer **18** reads from original layer **16**, so only the P3 input
to Detect is modified. The Detect references are `[17, 20, 23]`.

Do **not** compare this A0 to a historical truncated-P3/P4 baseline.

## Run steps (CachyOS / Fish, from repository root)

```fish
uv sync --extra cpu --group dev
uv run --extra cpu python scripts/btrm_smoke.py
uv run --extra cpu pytest tests/test_btrm.py -q
uv run --extra cpu modal run modal/app.py::compare \
    --config configs/experiment/ablation_btrm.yaml --parallel
```

The experiment YAML has `execution.parallel: true`, so parallel training is
also the default when calling `compare` without `--parallel`.

Parallel mode spawns **three independent L4 containers** simultaneously. It is
not three training processes competing for one GPU. No training jobs are started
by this PR. Three L4s can cost roughly triple the per-minute single-GPU price.
Each container copies the same TinyPerson data to its own temporary disk.

The YAML fixes 50 epochs, patience 50, input 1280, batch 16,
AdamW 0.001, seed 42, AMP, deterministic settings, augmentation, and dataset.
No loss, TAL, NWD, RFLA, P2/P4/P5 architecture or inference threshold
changes are part of the comparison.

If the W&B secret does not exist, remove
`output.wandb_project` from YAML before running.
Only explicitly add `--reuse` if code and dataset are unchanged; reuse
validates config and model YAML, **not** Python source changes.

## Stored outputs

Each model gets its own directory in the `tiny-person-runs` Modal Volume:

```text
/mnt/runs/btrm_a0/<RUN_ID>/
  config.yaml
  metadata.json
  checkpoints/{best,last}.pt
  metrics/train.csv
  metrics/train.png
  metrics/val.json
  metrics/benchmark.json
  metrics/size_val.json
```

A2 and A3 store the same paths under `btrm_a2` and `btrm_a3`.
The custom PyTorch module is serialized inside the checkpoint, so the
`tinydet` package must be available when loading `best.pt`.

Once all three runs and evaluations finish successfully, CPU-only report
generation writes:

```text
/mnt/runs/comparisons/btrm/compare-<GROUP_ID>/
  summary.json  # config identity, metrics, run IDs, deltas
  summary.csv   # one row per arm
  deltas.csv    # A2-A0, A3-A2, A3-A0
  report.md     # readable table, interpretation
```

Prints the output path in the Modal console. W&B uses the same comparison
group. Model metadata records git commit and dirty status.

## Measures and caveats

- Native Ultralytics `AP50`, `AP` (AP50:95), precision, recall.
- Tiny bins (max GT bbox side in **1280-letterboxed input pixels**):
  `0-8`, `8-16`, `16-32`, `>32`. Outputs include AP50,
  AP50:95, GT count and maximum Recall@IoU=0.5 at prediction
  `conf=0.001`. The polygon labels are converted to GT boxes correctly.
- GPU latency, VRAM, trainable parameter count and final learned scalar
  gate strength (effective gain).

The custom size-AP protocol ignores detections matching GT outside the
target bin, whereas the native overall Ultralytics AP is a separate protocol.
Do not assume the resulting AP figures are numerically identical.

**Key comparison:** A3−A2, because both branches use identical background
mean, affine nuisance variance and trainable gain; only A3 performs the ridge
projection. A3−A0 measures net utility, but cannot isolate the projection.

One seed gives *descriptive* effects, not statistical significance.
If the first screen is positive, verify on an untouched test set and multiple
seeds. A mathematical contraction property does not guarantee object AP.

## Core mathematics

Given center feature `f`, eight neighboring features `b_j`,
neighbor mean `mu`, and locally fitted affine gradients
`G = [g_x, g_y]`:

```text
d = f - mu
d_perp = [I - G (G^T G + lambda I)^(-1) G^T] d
sigma_bg^2 = SSE_neighbors / (8*C) + eps^2
evidence = (sqrt(||d_perp||^2 + eps^2) - eps) / (sqrt(C)*sigma_bg)
F_out = F * (1 + 0.5*tanh(alpha_raw)*tanh(evidence/saturation))
```

A2 replaces `d_perp` by `d` but keeps the **same variance estimator**.
For fixed `G` and strictly positive `lambda`, the residual projection's
singular values lie in `(0,1]`; it is non-expansive. This says nothing
about whether the estimated affine background is semantically correct.
