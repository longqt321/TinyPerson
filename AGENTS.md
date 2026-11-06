# AGENTS.md

This is a research codebase for tiny-person detection with YOLO26n on TinyPerson,
testing P2 features and the GauSS module (Gaussian Scale-Space Alignment and
Refinement, implemented as DCAF v2.1).

Use `uv` with Python 3.12. Local work is CPU-only; full training and benchmark
require a Modal NVIDIA A10 or L4.

```bash
uv sync --extra cpu --group dev
uv run --extra cpu --group dev ruff check .
```

Before changing experiments or results, read:

- `README.md` and `reports/gauss_final_report.md` for research questions and the GauSS mechanisms.
- `docs/agents/DATA_EVALUATION.md` for dataset and metric rules.
- `docs/agents/CODE.md` for code, lint, and checkpoint compatibility.
- `docs/agents/OPERATIONS.md` for Modal training and artifact lifecycle.

Run `git status` before editing. Do not commit datasets, checkpoints, run artifacts,
or generated predictions.

## Research reporting

- For requests to write or revise a paper, report, related work, benchmark analysis, or submission draft, use `.claude/skills/ai-research-report/SKILL.md`.
- Read `docs/report/00_state.md` first when it exists; continue from its recorded phase.
- Ground every result in repository evidence. Verify citations against primary sources; never invent values, metadata, or references.
- Follow all six literature layers in the skill's `references/search-strategy.md`.
