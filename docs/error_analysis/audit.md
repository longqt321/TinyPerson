# Error analysis audit

## Research task and current evidence

The repository studies tiny-person detection with YOLO26n on TinyPerson. The
current question is whether P2 features and the Multi-Scale Residual (MSR)
module on P2 improve detection over the E0-E3 paired controls. Aggregate
validation metrics and one-seed results are in `docs/modal_results_2026-10-20.md`.
Those metrics do not explain which objects, scales, or image conditions fail.

## Existing data flow

Two evaluation paths exist and must not be mixed:

1. **Derivative YOLO validation.** `scripts/prepare_tinyperson_yolo.py` reads the
   original COCO train annotations, merges `sea_person` and `earth_person` to
   class 0, writes YOLO labels, and creates a deterministic group split. The
   current manifest has 746 images: 662 train and 84 validation. Checkpoints are
   selected on this split. YOLO labels do not retain TinyPerson `ignore`,
   `uncertain`, or `in_dense_image` metadata.
2. **Official TinyPerson test.** `scripts/predict_tinyperson_test.py` predicts on
   original-resolution test rasters and exports source-coordinate boxes.
   `scripts/evaluate_tinyperson_benchmark.py` applies the repository TinyPerson
   protocol: dense-image exclusion, ignore/uncertain neutral regions, IoU/IoD
   matching, 200 detections per image, and size-binned AP.

Training-time `PredictionCollector` keeps the native Ultralytics preprocessing,
inference, and NMS, then stores postprocessed predictions and prepared targets in
`evaluation-val/predictions.pt`. `finalize_predictions` scales boxes back to
source coordinates, calls the Ultralytics metric accumulator, and exports
`metrics/val.json`. Only aggregates survive that path.

## Reusable infrastructure

- `tinydet.evaluation.tinyperson_benchmark`: size ranges, IoU/IoD, greedy
  matching, ignore handling, and 101-point AP. This remains the metric source of
  truth.
- `tinydet.evaluation.evaluator`: checkpoint discovery, native preprocessing and
  NMS, `predictions.pt` layout, and source-coordinate scaling.
- `tinydet.data.image_audit`: original raster, label geometry, letterbox, and
  COCO-dimension integrity checks.
- `tinydet.utils.experiment` and run `metadata.json`: commit, dirty flag, config,
  checkpoint, environment, and dataset provenance.
- `tinydet.research.results`: existing aggregate run selection and paired deltas.
- `tests/test_tinyperson_benchmark.py` and `tests/test_data_preflight.py`: the
  protocol and data-integrity reference tests.

## Verified limitations

- No existing component emits sample-level error records or per-size recall from
  saved predictions.
- Derivative YOLO validation cannot support ignore-aware or size-specific
  TinyPerson diagnostics because its labels discarded the required metadata.
- The official evaluator is standalone, not a vendored TinyBenchmark reference;
  official miss rate still requires the reference evaluator.
- Saved aggregate metrics do not embed checkpoint or prediction hashes. Run
  metadata and artifact hashes must be joined externally.
- The local A2 checkpoint exists, but the corresponding remote validation
  predictions and official test predictions are not part of the repository.
- TinyPerson annotations do not expose reliable occlusion or truncation flags, so
  those failure modes cannot be measured without additional data.

## Integration design

`src/tinydet/error_analysis/` consumes source-coordinate prediction JSON or an
existing `predictions.pt`. It reconstructs annotations from the original COCO
file and selects a split through the derivative `manifest.json`. Matching mirrors
the repository TinyPerson protocol and is cross-checked against
`evaluate_tinyperson` on the selected image subset.

The outputs separate raw predictions, canonical sample records, matched
outcomes, error records, aggregates, and reports. The core runs locally without
W&B, Modal, or a dashboard. Inference is explicit and only runs when a checkpoint
and dataset YAML are supplied without saved predictions.

## Implementation priorities

1. Data and evaluation integrity, provenance, and count reconciliation.
2. Sample-level matching and error decomposition at a documented primary IoU.
3. Task-specific size, density, boundary, confidence, and false-positive
   subtype analyses.
4. Paired sample-level comparison and transparent prioritization.
5. Non-visual report and metric cross-check before optional plots.

The implemented inventory and its limitations are in
[`analyzer_catalog.md`](analyzer_catalog.md). Execution is documented in
[`README.md`](README.md).
