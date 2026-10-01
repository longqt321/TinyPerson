# Tiny Person Research

Research runs use Modal GPUs. Local work uses CPU PyTorch.

## Local setup

```bash
uv sync --extra cpu --group dev
uv run --extra cpu pytest
uv run --extra cpu ruff check .
uv run --extra cpu python scripts/sanity.py --config configs/experiment/baseline.yaml
```

## Modal

Create Modal credentials, then upload TinyPerson data to `tiny-person-data` under `/tinyperson`. Dataset YAML must exist at `/mnt/data/tinyperson/dataset.yaml`. Place pretrained weights in `tiny-person-models` if experiment config names them. Run artifacts persist in `tiny-person-runs`.

```bash
uv run --extra cpu modal run modal/app.py::train --config configs/experiment/baseline.yaml
uv run --extra cpu modal run modal/app.py::evaluate --run baseline/RUN_ID --split val
uv run --extra cpu modal run modal/app.py::benchmark --run baseline/RUN_ID
```

Use `tools/list_runs.py --root outputs --csv outputs/experiments.csv` for downloaded results. Each run holds config, metadata, environment, checkpoints, and metrics. Git commit and dirty state are recorded; commit code before final runs.

For ERF, provide an image already in the data volume:

```bash
uv run --extra cpu modal run modal/app.py::erf --run baseline/RUN_ID --image-path /mnt/data/tinyperson/images/example.jpg
```

Download selected artifacts with `uv run --extra cpu python tools/download_results.py baseline/RUN_ID`. Upload explicitly with `uv run --extra cpu python tools/upload_data.py data LOCAL_DATASET --remote /tinyperson`.

Activation maps and frequency spectra use the same input image:

```bash
uv run --extra cpu modal run modal/app.py::diagnose --run baseline/RUN_ID --image-path /mnt/data/tinyperson/images/example.jpg
```

## Compare models

Edit `configs/experiment/comparison.yaml` to set the shared dataset and training parameters, then list model YAML files under `models`. Each model uses the same seed, dataset, training parameters, GPU and output root. Runs execute sequentially by default; each completed run gets validation metrics and a GPU benchmark. Add `--parallel` to train models in waves of up to four separate L4 containers. Evaluation and benchmarking follow each wave, so comparison uses at most four GPUs at a time. A failed run stops the sequence.

```bash
uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/comparison.yaml
```

The dataset must exist in the Modal data volume at `/mnt/data/tinyperson/data.yaml`. Results are saved under `/mnt/runs/<model-name>/<run-id>/`.

## Modal resource limits

Each remote function uses one L4 GPU, a 2-core CPU request with a 4-core cap, and an 8 GiB RAM request with a 24 GiB hard cap. Training can scale to at most four containers when parallel comparison is selected; other functions stay at one container each. Training has a six-hour timeout; evaluation and diagnostics have one-hour timeouts. `train` and `compare` reject other GPU types in experiment YAML. These are per-run limits, not a total monetary cap. Set a Workspace usage budget and net spend limit in Modal Usage & Billing to cap total charges across runs.

If a comparison stops after a completed run, evaluate and benchmark that run separately, then resume at the next model with `--start-at MODEL_NAME`. This avoids retraining completed models.

## Reuse completed comparison runs

Add `--reuse` to `compare` to search the Modal run volume before starting GPUs. A run is reusable only when its saved experiment config exactly equals the requested config, metadata says `COMPLETED`, and `checkpoints/best.pt` exists. New runs also record a SHA-256 hash of the model YAML; a changed model YAML prevents reuse. Existing older runs lack this hash and print a warning when reused. Missing validation or benchmark results are computed without retraining. The dataset path is compared, but dataset bytes are not hashed: use a new versioned dataset path or omit `--reuse` after changing dataset contents. Partial/subset YAML matches are never reused.

```bash
uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/comparison.yaml --reuse --parallel
```

## Context ablation models

`configs/model/yolo26n-no-p5.yaml` now stops its backbone at P4 and detects from P3/P4; no P5 features are computed. `configs/model/yolo26n-p3-only.yaml` stops at P3 and detects only from P3. The previous detection-only P5 ablation is preserved as `configs/model/yolo26n-detect-no-p5.yaml` to reproduce the older run. All four models are listed in `configs/experiment/comparison.yaml`.

Removing deep stages also reduces parameter count: standard 2,504,190; truncated P4 590,452; P3-only 144,106. An accuracy difference therefore cannot isolate receptive-field context from model capacity. The old detection-only ablation keeps the full backbone and offers a useful control. Its prior checkpoint remains under the old `yolo26n_no_p5` run name; that run does not represent the new truncated model.

## Explicit augmentation settings

`configs/experiment/comparison.yaml` records the detection augmentation defaults of the installed Ultralytics 8.4.164 release. Training uses 50 epochs with patience 15. Mosaic runs at probability 1.0 until the last 10 epochs; HSV jitter, translation, scaling and horizontal flip are enabled. Rotation, shear, perspective, vertical flip, BGR swap, mixup, CutMix, copy-paste and multi-scale are explicitly disabled. `copy_paste_mode` is recorded but has no effect while `copy_paste: 0.0`; `auto_augment` and `erasing` are classification-only and are not part of this detection configuration. `compile: false` makes the torch.compile choice explicit; a full GPU compile run has not been verified. Ultralytics validates all listed options before training.

## Live W&B training logs

Comparison runs specify `output.wandb_project: tiny-person-research`. The Modal training image installs W&B. Create a Modal Secret named `wandb-secret` containing `WANDB_API_KEY` in the same Modal environment as the run (Modal dashboard → Secrets → Create Secret). Do not place the key in YAML or Git. Then run `modal/app.py::compare` normally. Each trained model gets a separate W&B run; runs started by one comparison share a W&B group. The Modal console prints each W&B URL. Ultralytics logs training losses, validation metrics, learning rates and plots after each epoch; its native integration also uploads the best model as an artifact at training end. Existing finished runs cannot gain historical live logs. Omitting `wandb_project` disables W&B and does not require the Secret for that experiment.

GPU workers use min_containers=0, buffer_containers=0 and scaledown_window=2 to release idle GPU reservations promptly. Validation and benchmarking still require GPU. Training disables Ultralytics plots; a separate CPU worker publishes checkpoints and CSV metrics and generates metrics/results.png. Live W&B metrics and its final sync remain inside training. Runs pass through TRAINED before CPU finalization marks them COMPLETED. These settings apply to new Modal runs, not already-running apps.

W&B runs use `latest` for the newest execution of each `config.name` and `historical` for earlier executions, regardless of completion state or seed. Filter W&B by Tags contains `latest` to show one run per model; remove the filter to inspect history. Logs and Modal checkpoints are preserved. To refresh existing runs or repair tags after an API failure, run `uv run --extra cpu modal run modal/app.py::organize_wandb --project hoangngan/tiny-person-research` (CPU only). This labels runs; it does not configure the browser workspace filter or resume training.

Training now records mean box AP at every IoU threshold from 0.50 through 0.95 in steps of 0.05. Each new run saves `metrics/ap_iou.csv` per epoch and sends the same `metrics/AP0.50(B)` through `metrics/AP0.95(B)` series to W&B. Separate `evaluate` calls add `AP0.50` through `AP0.95` to `metrics/val.json` or `metrics/test.json`. These values are averaged over classes from the same Ultralytics validation pass; older CSV files cannot recover the intermediate IoU thresholds without rerunning validation on their checkpoints.

### NWD placement ablation

Use `configs/experiment/ablation_nwd.yaml` for the four-way P3-only baseline/NWD regression/NWD TAL/NWD full comparison. Optional `model.modules` settings select training-only plugins. See [method and commands](docs/NWD_ABLATION.md). The Modal image pins Ultralytics 8.4.164 to match the tested loss API and local lockfile.

### Heavy-Bottom architecture study

[Study design, checkpoint choices, diagnostics, four architectures, and commands](docs/HEAVY_BOTTOM.md). Phase 1 uses an explicit Modal run `best.pt` or pretrained YOLO26n checkpoint without retraining. Phase 2 is a separate four-model, 50-epoch screening configuration at `configs/experiment/heavy_bottom.yaml`; it is not launched automatically.
