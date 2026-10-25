# Training and artifact operations

Audience: AI coding agents.

## Training

Full training and benchmark require Modal with an NVIDIA L4. The local trainer
intentionally fails without CUDA.

```bash
modal run --detach modal/app.py::compare \
  --config configs/experiment/ablation_msr_p2.yaml \
  --parallel
```

Use `--smoke` only to check the pipeline. Five-epoch metrics do not rank models.

## Artifacts

Runs live in the Modal Volume `tiny-person-runs` under
`<experiment>/<run-id>/`. A run can contain config, metadata, environment,
training output, checkpoints, metrics, evaluation output, and diagnostics.

- `TRAINED` means training finished but finalize/benchmark may still be pending.
- `COMPLETED` is required for `--reuse`.
- `--reuse` also requires `checkpoints/best.pt` and an exactly matching config.
- A run may expose `best.pt` under `checkpoints/` or `training/weights/`.
  If two distinct checkpoints exist, specify which one to evaluate.
- W&B is supplementary logging, not the source of truth. Use saved artifacts.
- W&B init failure is non-fatal: training logs `W&B disabled after init failure`
  and continues with artifacts in the runs Volume.
- Aggregate downloaded artifacts with `tools/list_runs.py --config ... --markdown ...`.
  The report selects one primary run per experiment/seed and computes paired deltas.

## Reproducibility

- Check `metadata.json` for commit, dirty flag, config hash, environment, and
  dataset signature.
- Results recorded for commit `8326468` used `git_dirty=true`; the commit alone
  does not reproduce the code. Keep the run config and model YAML.
- Check `git status` before changing or comparing results, and do not revert
  another agent's uncommitted work.
