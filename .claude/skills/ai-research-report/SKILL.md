---
name: ai-research-report
description: Use when writing or revising an AI/ML research paper, technical report, benchmark analysis, related work, or submission draft from results in a repository.
---

# AI Research Report

Create research reports from repository evidence. Never invent results, metadata, citations, or experimental details.

## Start

1. Read repository instructions, README, research reports, metric/data rules, and `docs/report/00_state.md` if present.
2. Identify venue, page limit, language, and required format. Ask only for missing decisions that block accurate writing; use sensible defaults for reversible formatting.
3. Inspect repository and git status before editing. Do not overwrite unrelated work.
4. Create `docs/report/` workspace and maintain `00_state.md` after each completed phase.

## Phases

Proceed in order; stop on unresolved contradictions.

1. **Inventory:** Find result files, configs, logs, seeds, commits, hardware, and run metadata. Record every reported value with exact file and row/key/line in `01_inventory.md`. State whether baselines were rerun or sourced from published work.
2. **Analysis:** Check split and metric consistency. Summarize repeated runs with mean/std or confidence intervals; disclose single-run limits. Use paired tests for paired comparisons and account for multiple comparisons when relevant. List missing ablations, uneven gains, regressions, and failures in `03_analysis.md`. Map each claim to evidence and confidence.
3. **Literature:** Search six layers: direct work, core techniques, mathematical foundations, original/seminal work, adjacent fields, and critiques/failures. Use at least two distinct English queries per layer. Verify each included work's title, authors, year, venue, and claims against a primary source. Record why each source matters in `02_literature.md`; mark unresolved details `[CẦN XÁC MINH]`, never cite them as established fact. Stop after all layers covered and two consecutive search rounds add no core/foundation work.
4. **Outline:** In `04_outline.md`, map sections, paragraph/sentence targets, claims, evidence, citations, figures, and tables.
5. **Figures/tables:** Generate from actual result files using scripts; never hand-enter result values. Prefer vector figures and booktabs LaTeX tables. Keep regeneration scripts beside outputs.
6. **Draft:** Write Method, Experiments, Related Work, Introduction, Limitations, Conclusion, then Abstract. Save sections separately under `draft/`. Every comparative claim points to a specific table/figure. Update state after each section.
7. **Review:** Apply `references/checklist.md`; regenerate figures/tables and compare every number. Verify citation keys and references. Record findings and fixes in `99_review.md`.
8. **Finalize:** Compile requested format, check references and layout, inspect final PDF when produced. Report draft path, unresolved decisions, and weak or unsupported claims.

## Stop conditions

- Conflicting values for same setup: record conflict in inventory and ask which source/run is authoritative.
- Missing seed, hardware, venue, or other metadata: recover from repository where possible; otherwise mark unavailable and ask if it materially affects claims. Never guess.
- Missing experiment or ablation: document gap; do not launch new experiment unless user asks.
- Weak evidence: do not state as strong conclusion in Abstract or Conclusion.
- Policies, venue rules, or templates may change: verify against official venue source.

## Workspace

```text
docs/report/
├── 00_state.md
├── 01_inventory.md
├── 02_literature.md
├── 03_analysis.md
├── 04_outline.md
├── figures/       # vector outputs and generation scripts
├── tables/        # .tex/.md outputs and generation scripts
├── references.bib
├── draft/         # section drafts
└── 99_review.md
```

Detailed guidance: [search-strategy.md](references/search-strategy.md), [results-analysis.md](references/results-analysis.md), [imrad-writing.md](references/imrad-writing.md), [figures-tables.md](references/figures-tables.md), [checklist.md](references/checklist.md).
