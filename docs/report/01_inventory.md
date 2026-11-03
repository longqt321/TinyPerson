# Inventory DCAF v2.1

## Scope and protocol

Target: DCAF v2.1 against YOLO26n-P2 baseline only. This inventory excludes QAC-specific sources and prior `report-bundle/` manuscript. Existing DCAF report is an index to claims, not sufficient provenance by itself.

| Result | Method | Dataset/split | Metric | Seed/runs | Source | Exact location |
|---|---|---|---|---|---|---|
| 150-epoch tiled benchmark | Baseline | TinyPerson official test, 786 images, 13,687 positives | AP/MR at IoU .25/.50/.75, size bins | seed42, one run | `reports/eval/baseline_tinyperson_150e_tiled.json` | `checkpoint`, `summary.benchmark`, `summary.protocol` |
| 150-epoch tiled benchmark | DCAF v2.1 | Same declared protocol | Same metrics | seed42, one run | `reports/eval/dcaf_v2_1_tinyperson_150e_tiled.json` | `checkpoint`, `summary.benchmark`, `summary.protocol` |
| 150-epoch full-frame benchmark | Baseline | TinyPerson official test | AP/MR, size bins | seed42 implied by paired run; verify metadata | `reports/eval/baseline_tinyperson_150e_full.json` | Pending extraction |
| 150-epoch full-frame benchmark | DCAF v2.1 | TinyPerson official test | AP/MR, size bins | seed42 implied by paired run; verify metadata | `reports/eval/dcaf_v2_1_tinyperson_150e_full.json` | Pending extraction |
| 50-epoch and validation results | DCAF and baseline | TinyPerson | AP/MR | Verify per artifact | `reports/eval/*.json`, `reports/dcaf_autonomous/cycle_*/` | Pending; not yet treated as primary result |
| Geometry diagnostics | DCAF cycles 1/2 | TinyPerson validation | Geometry/error/size summaries | Runs named with `seed42` | `reports/geometry/dcaf_cycle_*/*` | CSV and report JSON; diagnostic, not benchmark headline |

## Primary 150e tiled raw values

Values below read from JSON and shown as percentages to two decimal places for readability; raw fractional values remain in source.

| Metric | Baseline | DCAF v2.1 | Delta (DCAF − baseline) |
|---|---:|---:|---:|
| all AP@0.25 | 21.28% | 20.68% | −0.61 pp |
| all AP@0.50 | 11.95% | 12.94% | +0.98 pp |
| all AP@0.75 | 1.28% | 1.40% | +0.11 pp |
| all MR@0.50 | 96.93% | 96.73% | −0.21 pp |
| tiny AP@0.50 | 9.07% | 9.76% | +0.69 pp |
| tiny1 AP@0.50 | 4.72% | 5.18% | +0.46 pp |
| tiny2 AP@0.50 | 6.90% | 9.50% | +2.59 pp |
| tiny3 AP@0.50 | 13.51% | 13.37% | −0.14 pp |
| small AP@0.50 | 17.43% | 19.23% | +1.80 pp |
| reasonable AP@0.50 | 20.49% | 21.20% | +0.72 pp |

Sources: baseline JSON keys `summary.benchmark.{all,tiny,tiny1,tiny2,tiny3,small,reasonable}` lines 8–75; DCAF JSON same keys, lines 8–75. Delta calculated from raw fractions, rounded to two decimal percentage points. Results are single-seed; no inferential statistics supported.

## Provenance and unresolved consistency issues

- Both JSON files declare 786 images evaluated, 0 dense images excluded, max 200 detections/image, tiled crop 640×512 with NMS merge 0.50, and standalone implementation of documented TinyPerson protocol.
- Checkpoint field says `training/weights/last.pt`; project data rules require best checkpoint reporting. This conflicts with `docs/agents/DATA_EVALUATION.md` rule to report `best.pt`. Confirm evaluation artifact/checkpoint before using as final result.
- `reports/DCAF_FINAL_REPORT.md` states 150e tiled values matching AP@0.50 broadly, but some values/deltas differ from raw JSON (including AP@0.25 and MR@0.25). Raw JSON are primary evidence; discrepancy needs full audit before final manuscript.
- Report claims “TOV Protocol” and zero dense images excluded, while WACV protocol description reportedly excludes dense images. Do not call it official benchmark protocol until implementation/protocol difference resolved.
- Model size and latency claims in report are not yet independently verified against raw profiling artifacts.
- Baseline and DCAF appear matched by run timestamp and seed in checkpoint paths; confirm configs and metadata before declaring controlled comparison.
- Hardware, training parameters, and exact evaluation command require metadata/config verification.

## Baseline provenance

Baseline appears rerun in repository, not copied from paper. External FCOS comparison mentioned in old report is out of scope pending primary-source verification and protocol comparability.
