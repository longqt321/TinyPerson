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
