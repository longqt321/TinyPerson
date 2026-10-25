# Visualization plan

Visualizations consume `samples.jsonl` and `errors.jsonl`; they do not repeat
matching or metric calculations.

## Implemented

| Figure | Research question | Data and cost | Interaction |
|---|---|---|---|
| `error_categories.png` | Which primary error categories dominate? | Error counts from validated records; negligible. | Static bar chart. |
| `recall_by_size_bin.png` | Does recall collapse for very small people? | Size-bin aggregation with denominators; negligible. | Static bar chart. |
| `confidence_distribution.png` | Do FPs separate from TPs by confidence? | Prediction scores and outcomes; negligible. | Static TP/FP histogram. |

They are generated only with `--plots` and add value beyond tables by making
scale concentration and confidence separation immediately visible.

## Proposed

| Figure | Research question | Data and cost | Why not now |
|---|---|---|---|
| Error gallery | What do representative duplicate, localization, and miss cases look like? | Box overlays from records plus source rasters; low compute, but image access and layout work. | Needs a reviewed sampling policy to avoid cherry-picking. |
| Density-size heatmap | Do tiny objects fail disproportionately in crowded scenes? | Existing records; negligible. | Useful after more than one checkpoint or seed is analyzed. |
| Paired corrected/regressed gallery | What changed between control and treatment? | Two `samples.jsonl` files and source rasters. | Requires paired checkpoints on the same split. |
| PR curves per size bin | How does the full threshold tradeoff differ by scale? | Requires re-running matching at every confidence threshold rather than cumulative primary outcomes. | Current operating-point table is sufficient for triage. |
| Spatial error heatmap | Are misses concentrated at borders? | Existing boxes and image dimensions; low compute. | Requires care because image sizes vary; aggregate in normalized coordinates. |

All future figures must load the validated records and preserve the same
provenance fields. No dashboard or external tracking service is required.
