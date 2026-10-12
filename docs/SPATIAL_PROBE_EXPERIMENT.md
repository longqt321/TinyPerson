# Spatial Probe Sensitivity — frozen YOLO26n

**Research question:** Does low C5 center-feature separability reflect degraded
accessible information or spatial sampling misalignment?

This is an *oracle diagnostic*, not a new detector and not proof of causality.

## Procedure

1. Start from the corrected segmentation parser in PR #9. Use a frozen trained
   YOLO26n checkpoint; no backward pass or YOLO weight updates.
2. For every annotated person in the selected split, decode the polygon bounds.
   Build one equally sized non-overlapping background ROI (rejection sampling).
   Both share the original positive's size bin. Dropped pairs are logged by omission.
3. Resolve C2–C5 and execute one forward pass for each image.
4. Extract feature vectors with: original nearest-cell center (historical baseline),
   bilinear center (`torch.nn.functional.grid_sample`), local 3×3 average
   (`avg_pool2d` then bilinear sample), and GT ROI (`torchvision.ops.roi_align`).
5. Train one train-only standardized scikit-learn logistic-regression probe per
   stage and sampler. Train/test split is group-disjoint by source video, or
   explicit image grouping when video IDs do not exist. **No sample leakage.**
6. Evaluate AP on identical test sample IDs. For each size bin use positives
   and their matched negatives; report random AP baseline and paired AP changes.
   Bootstrap group IDs, retaining the same sampled groups across methods.

`group_by: video` deliberately fails when a name has no identifiable video
token. Do not automatically fall back to image grouping. If there are too few
distinct source videos, choose another independent dataset split or report the
limitation.

## Run

```fish
uv sync --extra cpu --group dev
uv run --extra cpu --group dev pytest tests/test_spatial_probe.py
uv run --extra cpu --group dev ruff check src/tinydet/analysis/spatial_probe.py tests/test_spatial_probe.py
uv run --extra cpu python scripts/spatial_probe.py \\
  --checkpoint outputs/yolo26n/20260929-214018-9b91e49-seed42/checkpoints/best.pt \\
  --data datasets/TinyTiny_person/data.yaml \\
  --config configs/analysis/spatial_probe.yaml \\
  --output outputs/spatial_probe_v1
```

Artifacts: `samples.csv`, `predictions.csv`, `metrics.csv`,
`paired_deltas.csv`, `metadata.json`.

## Interpretation cautions

- A larger GT ROIAlign AP does not establish that an inference-time detector
  can find or align such regions without ground truth.
- C5 separability changes might reflect normalization, sampling, downsampling,
  feature semantics, contextual mixing, or model capacity; not necessarily
  literal information destruction.
- The baseline YOLO model was selected with validation measurements: this
  probe split is group-held-out *for the probe only*, not an independent
  detector benchmark.
- Small grouped test cohorts create wide uncertainty intervals.
- Feature pooling enlarges context; better AP is evidence of recoverable
  signal under a changed sampling operator, not proof of a unique mechanism.
