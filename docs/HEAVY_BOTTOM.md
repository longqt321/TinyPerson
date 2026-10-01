# Heavy-Bottom: YOLO26n backbone allocation study

**Hypothesis.** More backbone capacity at shallow, high-resolution stages may retain tiny-person detail. Deep semantic context may still be needed to separate people from clutter. “Heavy-Bottom” is this project's working name. Negative results are informative.

## Phase 1: fixed baseline probes, no training

Use an explicit trained run's `checkpoints/best.pt` from Modal Volume `tiny-person-runs`, or an explicit pretrained YOLO26n checkpoint in `tiny-person-models`. Never select a run automatically. The shared probe file saves image paths, GT line indices, original boxes, letterboxed boxes, and size bins. Sampling is deterministic (`seed=42`, default 200 instances), round-robin over bins. Size is **max(width, height) after letterboxing to model input**, in pixels; bins are `[0,8)`, `[8,16)`, `[16,32)`, and `[32,∞)`.

Installed Ultralytics 8.4.164 and the repository's YOLO26n YAML resolve backbone stage outputs C2/C3/C4/C5 to layers **2/4/6/8**. These are principal C3k2 blocks after P2/P3/P4/P5 downsampling, not arbitrary convolutions. The same stage names resolve separately per checkpoint. The baseline is unchanged.

Activation maps use `mean(abs(feature), channels)`. Each panel independently normalizes its map for display; do not compare panel intensity quantitatively. The project-specific **object/background activation ratio** divides mean activation in the rasterized GT region by mean activation in a local 2× box ring plus `1e-9`. The ring excludes the object. Regions clip to feature boundaries; sub-cell GTs cover at least one cell and use a one-cell ring when needed. CSV summaries give count, mean, median, and standard deviation by stage and size bin.

Grad-CAM uses the **person score of the decoded one-to-one candidate whose predicted center is nearest the GT center**. It backpropagates that scalar to each stage, applies channel-mean gradient weights, then ReLU. This is detection-targeted attribution, not causal proof; nearest candidate can be low confidence. Panels include fixed early probes plus first available success, difficult, and false-negative cases. Categories use NMS detections: success requires confidence ≥0.25 and IoU ≥0.5, difficult has IoU ≥0.1 at confidence ≥0.05 without a success, and false negative has neither. ERF instead targets mean absolute stage feature magnitude at the GT-center cell. It sums absolute input gradients across RGB channels. ERF50 and ERF90 are the smallest squares centered at the GT center containing at least 50% or 90% of total gradient energy. The report records side length, clipped area, and input-area fraction. This **effective** receptive field differs from theoretical receptive field. Zero-gradient probes have no ERF value.

Outputs live under the selected run's `diagnostics/heavy_bottom/` by default, with shared `heavy_bottom/probe_instances.json` on the run volume. A pretrained checkpoint writes to `heavy_bottom/<checkpoint-stem>/`.

## Phase 2: capacity screening

Four models share the original YOLO26n head, downsampling sequence, dataset, 1280 px image size, batch 16, AdamW, learning rate 0.001, augmentations, 50 epochs, patience 15, seed 42, deterministic mode, AMP, and L4 device class. All initialize randomly; no variant receives a pretrained advantage. Only backbone C3k2 repeat counts and output channels change. The official repository baseline YAML remains intact. FLOPs are measured at 640 px; ratios remain comparable at the common train size.

| Model | C2/C3/C4/C5 internal blocks | C2/C3/C4/C5 output channels | Parameters | GFLOPs | vs baseline |
| --- | --- | --- | ---: | ---: | ---: |
| deep_heavy | 1/1/2/2 | 56/112/128/256 | 2,740,872 | 5.960 | +1.2% |
| baseline | 1/1/1/1 | 64/128/128/256 | 2,504,190 | 5.892 | 0% |
| moderate_heavy_bottom | 2/2/1/1 | 64/128/128/224 | 2,423,150 | 6.162 | +4.6% |
| strong_heavy_bottom | 3/3/1/1 | 64/112/128/224 | 2,408,602 | 6.285 | +6.7% |

Depth scaling is discrete. Deep-heavy has +9.5% parameters; moderate and strong have about −3–4%. All variants are within ±10% baseline GFLOPs. Width reductions compensate for extra blocks while preserving C4/C5 context paths and the unchanged detection head. Compute differences remain a confound and are reported.

Validation retains repository precision, recall, AP50, AP50:95, and AP at each IoU threshold. `evaluate-size` computes AP50 and AP50:95 per input-pixel size bin with 101-point interpolation (`conf=0.001`, `max_det=1000`). It matches target-bin GT first and **ignores** detections matched to excluded-bin GT, so excluded objects do not become false positives. YOLO text annotations have no crowd/ignore flags; this wrapper cannot recover annotations absent from those files. Run the same saved Phase 1 probe on every trained architecture. The report combines size AP, activation ratio, ERF, training curves, and parameter/GFLOP comparisons. Panels come from each diagnostic output.

One seed and 50 epochs are screening only. Select candidates for later 100-epoch, 3–5-seed confirmation. This implementation makes no accuracy claim.

## Commands

```bash
# Phase 1 on an explicit full YOLO26n run; use --checkpoint /mnt/models/yolo26n.pt instead if desired.
# List actual Volume paths: uv run --extra cpu modal volume ls tiny-person-runs /yolo26n
uv run --extra cpu modal run modal/app.py::heavy_bottom_diagnose --run yolo26n/20260929-144813-9b91e49-seed42

# Inspect all four architectures before training.
PYTHONPATH=src uv run --extra cpu python scripts/heavy_bottom.py stats

# Phase 2 training; launch only after reviewing the configuration and GPU cost.
uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/heavy_bottom.yaml --parallel

# Evaluate each trained run using existing metrics, then size-conditioned AP.
uv run --extra cpu modal run modal/app.py::evaluate --run baseline/<actual-run-id>
uv run --extra cpu modal run modal/app.py::heavy_bottom_size_evaluate --run baseline/<actual-run-id>

# Reuse identical probes for each variant.
uv run --extra cpu modal run modal/app.py::heavy_bottom_diagnose --run baseline/<actual-run-id>

# After downloading completed runs, generate comparison plots locally.
PYTHONPATH=src uv run --extra cpu python scripts/heavy_bottom.py report \
  --run baseline=outputs/baseline/RUN_ID --run deep_heavy=outputs/deep_heavy/RUN_ID \
  --run moderate_heavy_bottom=outputs/moderate_heavy_bottom/RUN_ID \
  --run strong_heavy_bottom=outputs/strong_heavy_bottom/RUN_ID \
  --output outputs/heavy_bottom/report
```

`RUN_ID` and `<actual-run-id>` in other examples are placeholders. Select a completed run with `best.pt` from the Volume. The older `baseline/20260929-001154-unknown-seed42` uses the no-P5 model and cannot supply C5 for this study. The `probe`, `diagnose`, `evaluate-size`, and `report` subcommands also accept local paths for offline work. Phase 1 does not train. Full Phase 2 training is not launched by this implementation.
