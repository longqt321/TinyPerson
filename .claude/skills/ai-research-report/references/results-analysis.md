# Benchmark inventory and analysis

## Inventory

Search repository for CSV, JSON/JSONL, YAML, logs, TSV, and TensorBoard event files. For each usable result, record:

| Result | Method | Dataset | Metric | Seed/runs | Source file | Row/key/line | Configuration |
|---|---|---|---|---|---|---|---|

Also record config, split, commit/version, hardware, runtime when available. State whether baseline was rerun under same conditions or copied from a paper; cite exact source for published values.

## Consistency checks

Before analysis, verify:

- Same method/dataset/metric values agree, or differences have explained provenance (seed, split, code version).
- Proposed method and baseline use compatible data splits and evaluation code.
- Run counts and conditions are comparable.
- Metric implementation matches repository definition.

If same-setup values conflict, stop, document each source, and ask which run is authoritative. Do not choose silently.

## Statistical analysis

- Report mean and standard deviation or 95% confidence interval across seeds.
- Use paired tests for paired runs. For multiple comparisons, consider Holm or Bonferroni correction.
- With fewer than five runs, disclose low power and avoid strong claims. With one run, state single-run limitation.
- Put test name and p-value/interval in table or caption when used.
- Example bootstrap for paired differences:

```python
import numpy as np

def bootstrap_ci(diff, n_boot=10_000, alpha=0.05, seed=0):
    diff = np.asarray(diff)
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return diff.mean(), lo, hi
```

## Ablation and failures

- List each proposed component and whether repository contains an ablation removing it independently.
- Missing ablation is a documented gap, not evidence of component effect. Do not run new experiments unless asked.
- Record every dataset/metric where method ties or regresses. Preserve negative results in main tables; discuss plausible explanations as hypotheses, not established causes.

## Claim ledger

In `03_analysis.md`, maintain:

| Claim | Evidence (source/table/test) | Confidence | Caveat |
|---|---|---|---|

Confidence: strong (multiple seeds, tests, consistent datasets); medium (limited seeds or one dataset); weak (single run or effect within variability). Weak claims cannot be asserted as established findings in Abstract or Conclusion. Avoid “significantly” without statistical testing and direct comparison.
