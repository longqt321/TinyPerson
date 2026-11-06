# Code guidance

Audience: AI coding agents.

## Project shape

Code lives under `src/tinydet`: model construction, the GauSS/DCAF module, training,
diagnostics, evaluation, analysis, and research result aggregation. Files in `scripts/` are thin CLI
entrypoints; put reusable logic in the package.

The model is configured through Ultralytics YAML under `configs/model/`.
Experiment settings under `configs/experiment/` define seed, data, training,
and the model list for comparisons.

## GauSS / DCAF invariants

- DCAF sits on P2 in the nano graph and must preserve `[B, 32, H, W]`.
- P2 is the stride-4 neck output; P3/P4/P5 and Detect order must stay unchanged
  in controlled comparisons.
- `src/tinydet/modules/dcaf.py` reads ablation flags from the `TINYDET_ABLATION`
  environment variable at construction time. Flags only disable contributions in
  `forward()`; parameter count and checkpoint layout stay identical.
- Diagnostics are optional and must never break training. Collector or sink
  failures should disable only diagnostics.

## Working rules

- Find the real caller and fix the root cause. Keep the diff narrow.
- Reuse existing helpers before adding code. Do not add dependencies or
  abstractions for speculative needs.
- Keep metric names, split names, and W&B namespaces stable unless the task
  explicitly changes them.
- Follow `ruff`, line length 100, Python 3.12.
- Do not guess Ultralytics APIs; check the installed source or existing usage.

## Verification

For model or config changes:

```bash
uv run --extra cpu python scripts/validate_tiny_ablation.py \
  configs/model/yolo26n_tiny_A2_p2.yaml --imgsz 640
uv run --extra cpu python -c "import tinydet.modules.dcaf"
```

Then run the lint checks in the root `AGENTS.md`. CI runs only a CPU lint
subset and does not prove CUDA AMP or real-data training works.
