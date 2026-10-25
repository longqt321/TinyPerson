# AGENTS.md

This is a research codebase for tiny-person detection with YOLO26n on TinyPerson,
testing P2 features and Multi-Scale Residual (MSR) on P2.

Use `uv` with Python 3.12. Local work is CPU-only; full training and benchmark
require a Modal NVIDIA L4.

```bash
uv sync --extra cpu --group dev
uv run --extra cpu --group dev ruff check .
```

Before changing experiments or results, read:

- `README.md` and `docs/MSR_P2.md` for research questions and the paired E0-E3 protocol.
- `docs/agents/DATA_EVALUATION.md` for dataset and metric rules.
- `docs/agents/CODE.md` for code, lint, and checkpoint compatibility.
- `docs/agents/OPERATIONS.md` for Modal training and artifact lifecycle.
- `docs/modal_results_2026-10-20.md` for current evidence and limits.

Run `git status` before editing. Do not commit datasets, checkpoints, run artifacts,
or generated predictions.
