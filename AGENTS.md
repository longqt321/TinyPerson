# AGENTS.md

This is a research codebase for tiny-person detection with YOLO26n on TinyPerson,
testing P2 features and Distributional Quality-Aware Calibration (D-QAC / QAC v2).

Use `uv` with Python 3.12. Local work is CPU-only; full training and benchmark
require a Modal NVIDIA A10 or L4.

```bash
uv sync --extra cpu --group dev
uv run --extra cpu --group dev ruff check .
```

Before changing experiments or results, read:

- `README.md` and `docs/QAC_V2_REPORT.md` for research questions, DUGC and SATAR mechanisms.
- `docs/agents/DATA_EVALUATION.md` for dataset and metric rules.
- `docs/agents/CODE.md` for code, lint, and checkpoint compatibility.
- `docs/agents/OPERATIONS.md` for Modal training and artifact lifecycle.

Run `git status` before editing. Do not commit datasets, checkpoints, run artifacts,
or generated predictions.
