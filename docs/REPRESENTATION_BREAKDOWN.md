# Representation-breakdown study

This is an **exploratory diagnosis**, not the previous Heavy-Bottom architecture ablation.
It does not modify YOLO training, model weights or the default experiments.

## Goals

1. Measure frozen feature separability of tiny ground-truth centers vs sampled background.
2. Measure feature sensitivity to replacing an annotated object with mean nearby color.
3. Measure detection-state flips after translating an input by 0–3 pixels.

Stage keys `C2`–`C5` refer to the four backbone C3k2 stage outputs resolved
from the actual YOLO26 architecture. The experiment intentionally does **not**
assume these are all detection-head outputs.

## Run in Modal

Use a completed YOLO26n run with `checkpoints/best.pt` in `tiny-person-runs`.
The existing `configs/experiment/baseline.yaml` is a five-epoch, no-P5
configuration and is **not** the full YOLO26n baseline. Prefer a properly
trained YOLO26n run from the standard comparison.

```bash
uv run --extra cpu modal run modal/app.py::representation_diagnose \
  --run yolo26n/ACTUAL_RUN_ID \
  --config configs/analysis/representation.yaml
```

The remote study uses the existing L4 GPU worker and all three mounted Volumes.
Artifacts are written below:
`/mnt/runs/yolo26n/ACTUAL_RUN_ID/diagnostics/representation/`

Local (CPU, potentially slow):

```bash
uv run --extra cpu python scripts/representation.py \
  --checkpoint /path/to/best.pt \
  --data /path/to/data.yaml \
  --config configs/analysis/representation.yaml \
  --output outputs/representation
```

## Outputs

- `samples.csv`: GT instance inventory in letterboxed input coordinates
- `separability/metrics.csv`: ranking AP by backbone stage / size bin
- `counterfactual/feature_delta.csv`: normalized feature perturbation proxy
- `shift_stability/predictions.csv`: GT detection status at fixed IoU 0.5 and conf 0.25
- `shift_stability/miss_flip_rate.csv`: detection changes versus phase (0,0)
- PNG figures, `metadata.json`, `report/findings.md`

## Interpretation limits

- The separability split is by **source image**, never instance. This is a
  lightweight linear probe, not a strong upper bound on recoverable information.
- The size bins use the maximum GT side length after input letterboxing,
  consistent with the existing `heavy_bottom.py` convention.
- Background negatives are sampled from nearby and distant locations and
  excluded from GT boxes. They are *not* texture-matched negatives.
- The mean-color fill is **not** a physically valid counterfactual: it introduces
  an edit artifact and changes local texture. It is a diagnostic sensitivity
  test only; causal claims require source-background pairs or controlled
  synthetic compositing, including matched no-object controls.
- The shift test translates the whole input, *not* the object alone. Confounds
  include padding, border effects and matching thresholds; targets near borders
  are skipped.
- Results are grouped by image but images from the same scene/camera may be
  correlated. Final scientific claims require a stricter group split and
  uncertainty estimates.
- The three studies run on frozen checkpoints; finding a difference does not
  establish novelty. Review anti-aliasing, scale-aware downsampling, shift
  equivariance and spectral preservation before proposing a module.

## Compute budget

Start with 50–200 instances to validate the pipeline; counterfactuals and
shift experiments each trigger additional forward passes. A large full split
can exceed the one-hour Modal diagnostics limit. Disable unneeded experiments
in YAML when running larger studies.

## Filesystem security

The local CLI `--output` must resolve to a strict subdirectory of the
repository's trusted `outputs/` directory. Absolute paths outside that root,
`..` traversal, and symlink escapes are rejected. Modal diagnostics are limited
to a subdirectory of the selected run's root in `tiny-person-runs`; YAML
output paths cannot escape that run. The study validates the output root again
before creating directories and writing CSV/plots/metadata. These boundaries
address SonarCloud's CLI-to-filesystem path-traversal flow rather than ignoring
its warning.

## Diagnostic schema v2 (follow-up PR)

- `dataset_audit/size_distribution.csv` counts all valid GT instances by maximum side
  after 1280-pixel letterboxing. This is an inventory, not COCO size bins.
- Sampling remains deterministic and round-robin across nonempty bins. A missing
  tiny bin is explicitly reported rather than synthesized; selected bin counts
  are written to metadata.
- `shift_stability/predictions.csv` retains the legacy `detected` and
  `best_confidence` columns for compatibility. The latter is a thresholded
  matched score, **not** the unconditional maximum prediction confidence.
- New fields include `num_predictions`, `max_confidence`, `best_iou`,
  `confidence_at_best_iou`, `best_iou_at_conf_001/01/05/25`, and
  `detected_conf_001/01/05/25` (all with IoU >= 0.5).
- Shift cohort eligibility is per GT, preserving a common set of valid targets
  for all offsets. Zero flips with zero baseline recall do not imply stability.
- This PR does not implement texture-matched negatives, causal interventions,
  or subpixel feature alignment; those require separate methodological validation.
