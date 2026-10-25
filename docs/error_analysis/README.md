# Error analysis workflow

The analyzer loads source-coordinate predictions and original TinyPerson COCO
annotations, decomposes each prediction and ground truth, writes machine-readable
records, and produces a local report. It does not require W&B, Modal, or an
external dashboard.

## Validation split with a local checkpoint

The derivative split needs `manifest.json` because YOLO labels alone lost
TinyPerson ignore/dense metadata.

```bash
uv run --extra cpu python scripts/analyze_errors.py \
  --run outputs/tiny_A2_p2/20261019-194809-8326468-seed42 \
  --data datasets/derived/tinyperson-yolo/data.yaml \
  --annotations datasets/TinyPerson_official/tiny/tiny_set/erase_with_uncertain_dataset/annotations/tiny_set_train.json \
  --manifest datasets/derived/tinyperson-yolo/manifest.json \
  --split val --limit 20 \
  --output outputs/error_analysis/tiny_A2_p2-val-smoke --plots
```

`--limit` is applied both to inference and to the expected image set. Remove it
for the full validation split. Without `--predictions`, inference is explicit and
requires `--run` or `--checkpoint` plus `--data`.

## Reuse saved predictions

```bash
uv run --extra cpu python scripts/analyze_errors.py \
  --predictions outputs/MODEL/RUN/evaluation-val/predictions.pt \
  --annotations data/tiny_set_train.json \
  --manifest data/manifest.json --split val \
  --output outputs/error_analysis/MODEL-RUN
```

Prediction JSON in the repository export format is also accepted.

## Official test predictions

```bash
uv run --extra cpu python scripts/predict_tinyperson_test.py \
  --checkpoint /path/to/best.pt \
  --annotations /path/to/tiny_set_test_with_dense.json \
  --image-root /path/to/test \
  --output predictions/test.json

uv run --extra cpu python scripts/analyze_errors.py \
  --predictions predictions/test.json \
  --annotations /path/to/tiny_set_test_with_dense.json \
  --split test --output outputs/error_analysis/official-test
```

The analysis report is diagnostic. The official metric remains the output of
`scripts/evaluate_tinyperson_benchmark.py`.

## Paired comparison

Analyze both checkpoints on the same annotations, manifest, split, score
threshold, and IoU. Then pass the control records to the treatment run:

```bash
uv run --extra cpu python scripts/analyze_errors.py \
  ... --output outputs/error_analysis/treatment \
  --baseline-samples outputs/error_analysis/control/samples.jsonl
```

This reports images that became clean, images that regressed, and miss/FP count
deltas. It refuses a baseline with different annotation, manifest, or split
hashes.

## Outputs

- `summary.json`: provenance, integrity findings, counts, rates, aggregates,
  diagnostic AP, metric cross-check, comparison, limitations.
- `samples.jsonl`: one canonical record per analyzed image, including all
  positive/ignored ground truths, all retained predictions, primary outcomes, and
  per-image counts.
- `errors.jsonl`: flattened misses and false positives with subtype, IoU, score,
  size, density, and boundary context.
- `report.md`: non-visual report with tables, denominators, caveats, and
  traceable follow-up priorities.
- `plots/*.png`: optional figures built from the validated records.

## Interpretation rules

- A miss is a positive ground truth without a matching prediction under the
  configured rules. It is not evidence that the annotation is wrong.
- A duplicate or localization error is a false positive subtype. Background FPs
  are false positives without a nearby positive target.
- Ignore/uncertain regions and out-of-size detections are neutral, matching the
  TinyPerson evaluator's intent.
- Diagnostic AP is cross-checked against the repository evaluator where possible.
  A mismatch invalidates the analysis run until investigated.
- Associations by size, density, boundary, or confidence are not causal. Test a
  hypothesis with a controlled paired comparison.
