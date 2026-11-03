# Figures and tables

## Rules

- Every figure/table must be self-contained with informative caption and referenced in text.
- Place near first reference. Figure caption below; table caption above.
- Generate numeric content from repository result files using checked-in scripts. Never hand-enter benchmark values.
- Prefer PDF/EPS vector figures. Ensure labels remain legible at final column width and in grayscale.
- Use colorblind-safe palettes such as Okabe–Ito.

## Choosing format

| Content | Suitable tool/output |
|---|---|
| Architecture/pipeline | TikZ, draw.io, Figma, Mermaid, Graphviz; export vector |
| Training curves/comparisons/heatmaps | Matplotlib or Seaborn; export PDF |
| Qualitative examples | Scripted Python extraction with documented source IDs |
| Results/ablation | Generated LaTeX `booktabs` or Markdown tables |

## Matplotlib defaults

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "figure.figsize": (3.3, 2.4),
})
```

One-column width is often about 3.3 inches, two-column about 6.8 inches; check venue template.

## LaTeX table style

Use `\toprule`, `\midrule`, `\bottomrule`; no vertical rules. Caption states metric direction, units, and run count. Bold best result only when comparison is valid. Explain significance marker and test in caption.

Generate table from source data, for example:

```python
summary = df.groupby(["method", "dataset"])["score"].agg(["mean", "std"])
```

Adapt field names and source paths to actual repository. Store scripts beside generated outputs under `tables/` or `figures/`.

## Typical allocation

Method: one architecture figure. Experiments: main results, ablation, and one or two analysis figures when supported. Appendix holds detailed results. Keep body to roughly 4–7 figures/tables unless venue or evidence warrants more.
