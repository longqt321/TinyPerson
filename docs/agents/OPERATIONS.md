# Training and artifact operations

Audience: AI coding agents.

## Training

Full training requires a CUDA GPU. CPU training is intentionally disabled: both
`train.py` and `tinydet.training.trainer.train` refuse to run without CUDA.

### Modal (primary)

```bash
modal run --detach modal/app.py::compare \
  --config configs/experiment/ablation_msr_p2.yaml \
  --parallel
```

Use `--smoke` only to check the pipeline. Five-epoch metrics do not rank models.

### 60-epoch campaign (detached by default)

```bash
modal run -d launch_train_and_eval_60e.py          # audit -> 2x train -> finalize -> tiled + full eval
modal app logs <app-id>                            # monitor
modal run launch_train_and_eval_60e.py::fetch      # pull eval + orchestration summaries into reports/
```

The whole chain runs inside `modal/app.py::_orchestrate_60e`, so a dropped local
connection cannot kill the campaign. The `-d` flag is still required: without it
Modal stops the app when the local process exits. `_train` commits the runs
Volume at the end of every epoch, so `results.csv`, `weights/last.pt` and the
`metrics/*.csv` files survive any disconnect.

Per-epoch metric files written under `<run>/metrics/`:

| File | Content |
| --- | --- |
| `training/results.csv` (Ultralytics) | train/val box, cls, dfl losses; precision, recall, mAP50, mAP50-95; lr per group; epoch time |
| `metrics/ap_iou.csv` | AP at IoU 0.50:0.05:0.95 |
| `metrics/p2refine_diagnostics.csv` | P2Refine rho/cos/w_out_norm (only when P2Refine is in the model) |
| `metrics/gauss_diagnostics.csv` | GauSS router weights, entropy, band energies, z-loss, gates, sigmas, DySample offset RMS, control-group lr/weight-decay |
| `metrics/system_metrics.csv` | epoch wall time, gradient norm, AMP scaler scale, VRAM allocated/peak, Ultralytics fitness and best_fitness |
| `metrics/size_stratified.csv` | official TinyPerson benchmark (class_mode=merge) on the per-epoch val predictions: AP and MR at IoU 0.25/0.50/0.75 per size bucket (tiny1/tiny2/tiny3/small/reasonable/all) |
| `metrics/val_predictions_latest.json` | per-image val predictions used by the size-stratified scoring (overwritten each epoch) |
| `diagnostics/annotations_val.json` | official COCO annotations subset to the val split (created once at train start) |

### Modal resource tiers

`modal/app.py` defines four tiers so that plumbing containers do not reserve
training-sized resources:

| Tier | cpu (request, limit) | memory MB (request, limit) | Used by |
| --- | --- | --- | --- |
| `CPU_LIMIT` / `MEMORY_LIMIT_MB` | (4, 8) | (12288, 24576) | `_train`, `_evaluate`, `_benchmark`, `_analyze_geometry`, `_run_comparison_report`, `_run_checkpoint_evaluation`, `_run_tiled_checkpoint_evaluation`, `_run_visdrone_val` |
| `AUDIT_CPU` / `AUDIT_MEMORY_MB` | (4, 8) | (8192, 16384) | `_audit_original_dataset` (24 decode threads) |
| `POSTPROCESS_CPU` / `POSTPROCESS_MEMORY_MB` | (2, 4) | (4096, 8192) | `_finish_evaluation` (loads prediction artifacts, computes AP on CPU) |
| `LIGHT_CPU` / `LIGHT_MEMORY_MB` | (1, 2) | (1024, 2048) | `_finalize`, `_orchestrate_60e`, `run_comparison`, `_preflight` |

Billing follows actual usage, so the request value is what matters most: a
`LIGHT` container costs roughly a quarter of the CPU and a twelfth of the RAM
of a training container while it waits on handles or copies files.

### Any other CUDA host (Colab, Kaggle, RunPod, ...)

`train.py` is the environment-agnostic entry point. It resolves every path to an
absolute path, so the working directory does not matter.

The project is managed with `uv`. `uv sync --extra gpu` provisions a Python 3.14
environment and installs the CUDA build of torch pinned in `uv.lock`,
matching the Modal image. Do not `pip install` torch
separately: a CPU wheel would shadow the CUDA one.

```bash
# One-time per session: install uv, then the pinned environment.
pip install -q uv
uv sync --extra gpu --group dev

uv run --extra gpu python train.py \
  --config configs/experiment/p2refine_tinyperson_wacv.yaml \
  --data /content/data_local/tinyperson-yolo-v2/data.yaml \
  --output-root /content/drive/MyDrive/tinydet_runs \
  --device 0 --batch 8 --imgsz 1280 --workers 4 \
  --name p2refine_colab_seed42
```

- `--data` overrides `data.config`; `--data-root` overrides the `path` key inside
  the dataset YAML (use it when the data lives somewhere other than the tracked
  YAML expects).
- `--output-root`, `--name`, `--seed`, `--device`, `--epochs`, `--batch`,
  `--imgsz`, `--workers` override the experiment YAML. `--no-wandb` disables W&B.
- Resume requires `--run-id`. Without it every invocation creates a fresh
  timestamped run folder (`YYYYMMDD-HHMMSS-<commit>-seed<seed>`), so
  `training/weights/last.pt` is never found and training restarts from epoch 0.
  Pass the same `--run-id` on the rerun to pick up `last.pt` with
  `resume=True`. Modal does this for you via a stable run id; `train.py` does
  not guess.
- Always pass `--extra gpu` to `uv run`. A bare `uv run python ...` re-syncs the
  environment without the extra and pulls a different torch build (the PyPI
  wheel, e.g. `2.14.0+cu130`) over the pinned `2.11.0+cu128`. Verified locally:
  bare `uv run` swapped the wheel; `uv run --extra gpu [--group dev]` kept it.
- `--dry-run` resolves the config, dataset and model and prints parameters,
  GFLOPs and strides without training. It runs on CPU, so it is the way to
  validate a setup before spending GPU time.

Copy the dataset from Drive to local disk before training; reading images over a
FUSE mount starves the GPU. On Kaggle, `/kaggle/working` persists between
sessions, so it can hold checkpoints without mounting Drive.

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
