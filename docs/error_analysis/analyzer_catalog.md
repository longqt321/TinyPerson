# Analyzer catalog

Status values: **implemented**, **partially implemented**, **proposed**.

## General analyzers

| Analyzer | Research question | Procedure and output | Assumptions and limits | Status |
|---|---|---|---|---|
| Data/evaluation integrity | Can the result be trusted? | Check duplicate images, missing/extra predictions, finite scores, valid boxes, class mapping, manifest dimensions, dense exclusion, and count reconciliation. Emit `integrity.issues` and invariant booleans. | Structural format errors raise; per-record errors are reported and excluded explicitly. | implemented |
| Matching and error decomposition | How does each prediction fail? | Greedy score order within each image. At the primary IoU, classify each prediction as TP, duplicate FP, localization FP, background FP, or neutral (ignore/IoD/out-of-size). Each positive GT is matched or missed. Emit `samples.jsonl` and `errors.jsonl`. | Person class only. The diagnostic taxonomy is not the official AP definition, although TP/FP outcomes reproduce it. | implemented |
| AP metric cross-check | Does the diagnostic matcher reproduce the existing evaluator? | Recompute bin AP from matched records and compare with `evaluate_tinyperson` on the same filtered annotations/predictions. Emit per-bin absolute differences. | Requires complete predictions for the selected annotations; reference uses its own dense and ignore rules. | implemented |
| Aggregation with denominators | Which groups contain errors? | Aggregate by size, density, boundary position, class, confidence, and FP subtype. Report counts, denominators, rates, and explicit `null` for empty groups. | Descriptive only; groups are not causal. | implemented |
| Error prioritization | What should be investigated first? | Rank observed error counts, then rate, with the affected metric named. | Transparent triage, not a root-cause score. | implemented |
| Confidence operating points | What is the FP burden at operational thresholds? | Cumulative TP/FP/recall/precision for fixed score thresholds using primary outcomes. | Not a full PR curve; NMS was already applied before export. | partially implemented |
| Paired comparison | Which samples were corrected or regressed? | Require the same annotation/manifest/split hashes; compare per-image miss + FP counts and emit corrected/regressed examples and count deltas. | Image-level status, not object-identity tracking. | implemented |

## Task-specific analyzers for TinyPerson

| Analyzer | Failure mode | Evidence produced | Why it fits this repository | Status |
|---|---|---|---|---|
| Size-bin recall | Misses concentrate on extremely small people. | Ground truths, matches, misses, recall, and miss rate for half-open sqrt-area bins `tiny1`, `tiny2`, `tiny3`, `small`, `reasonable`; separate evaluator-compatible AP bins. | TinyPerson's central difficulty is object scale, and the benchmark already defines scale ranges. | implemented |
| Duplicate detections | Multiple boxes compete for one person. | Duplicate FP count and share of FP. | NMS still leaves duplicates in crowded scenes; this tests a known detector failure mode. | implemented |
| Localization versus background FP | Remaining FPs are badly placed or spurious. | Localization FP count/share and background FP count/share at a documented low-IoU boundary. | Separates box-quality errors from scene-level false alarms. | implemented |
| Density analysis | Errors increase with crowding. | Per-image positive-object bins `single`, `sparse`, `crowded`, `very_crowded`; recall and FP/image. | TinyPerson contains dense scenes, and density is computable from annotations. | implemented |
| Border analysis | People near image borders are harder. | Recall for objects whose center lies within 2% of a border. | Small objects near borders lose context and may be clipped; only association is measured. | implemented |
| Annotation candidates | Some misses may be annotation problems. | Invalid boxes, duplicate annotations, and missed positives listed in integrity records/error JSONL for manual review. | The pipeline must distinguish model failure from annotation candidates. | partially implemented |
| Confidence behavior | FP burden depends on score threshold. | Confidence histogram inputs and operating-point table. | The model's post-NMS confidence is available in predictions. | implemented |
| Resolution/inference policy | Different imgsz, tiling, or postprocessing change errors. | Run the same analyzer on prediction files produced under controlled settings and use paired comparison. | Directly relevant to tiny objects, but requires multiple prediction artifacts. | partially implemented |
| Occlusion/truncation | Partially visible people fail more often. | No reliable annotation fields. | Would require additional metadata or manual review. | proposed |
| Internal feature diagnostics | P2/MSR feature quality explains size-bin failures. | Requires hooks and intermediate activations. | Not prediction-level; add only after the current evidence is stable. | proposed |

## Out of scope

Classification confusion, segmentation boundaries, tracking identity switches,
and keypoint errors are not applicable to this single-class detection task.
