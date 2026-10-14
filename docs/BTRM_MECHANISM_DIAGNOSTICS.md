# BTRM: Mathematical Audit → Mechanism Diagnostics

## Fixed experimental subjects

This protocol inspects the **existing** 50-epoch checkpoints only:

| Arm | Frozen run |
|---|---|
| A0 | `btrm_a0/20261014-063504-0a3e661-seed42` |
| A2 | `btrm_a2/20261014-063507-0a3e661-seed42` |
| A3 | `btrm_a3/20261014-063508-0a3e661-seed42` |

**No training or parameter updates.** The BTRM module source and historical
weights are not modified by these diagnostics.

## Stage 1 (blocking): proof and independent computational audit

Let `r` be the ring radius in feature-map pixels. For
`u_j = (r*dx_j, r*dy_j)` with eight neighboring offsets, symmetry implies

```text
sum_j u_j = 0
sum_j u_j u_j^T = diag(6*r^2, 6*r^2)
mu = sum_j b_j/8
g_x = sum_j (r*dx_j) b_j / (6*r^2)
g_y = sum_j (r*dy_j) b_j / (6*r^2)
```

Thus the implementation's `gx` and `gy` are exactly the
least-squares minimizers for the **unregularized affine ring fit**.

For center residual `d = f_center - mu`, with `G=[g_x,g_y]`:

```text
lambda = ridge_relative * trace(G^T G) + eps^2 > 0
q* = solve(G^T G + lambda I_2, G^T d)
d_perp = d - G q*
sigma = sqrt(sum_j ||b_j - mu - G u_j||^2/(8*C) + eps^2)
z = (sqrt(||d_perp||^2 + eps^2) - eps)/(sqrt(C)*sigma)
g = tanh(z/saturation)
F' = F*(1 + 0.5*tanh(alpha_raw)*g)
```

A2 uses the same background model and `sigma`, replacing `d_perp`
with `d`. The two-dimensional inverse is mathematically equivalent to
the reference `numpy.linalg.solve`.

If `G` is held fixed and `lambda > 0`, SVD gives eigenvalues
`lambda/(s_i^2+lambda)` along the column space of `G`, and 1 in
orthogonal directions. Therefore `||d_perp|| <= ||d||`. This is a
**linear-algebra property**, not proof of object saliency or mAP.

Run offline without Modal:

```fish
uv run --extra cpu python scripts/btrm_math_audit.py --output /tmp/btrm_math_audit.json
uv run --extra cpu pytest -q tests/test_btrm_math_audit.py tests/test_btrm_diagnostics.py
```

The oracle uses independent NumPy **float64** ring least squares and linear
solve, compares 72 scores across 6 different feature regimes, checks exact
initial identity, bounded modulation and 4 autograd-vs-central-difference
gradients. It raises on mismatches. The original module performs its
internal arithmetic in **float32** even if input is float64, intentionally
for AMP stability. This is a measurable precision limitation.

GitHub CI initial result (CPU, commit of PR #16):
- Independent formula comparisons: 72/72 pass.
- Max score *absolute* error: 0.0002620807.
- Max score *relative* error, after excluding oracle values <=1e-5:
  2.5096 (near-zero signals can have large relative errors).
- Gate factor reconstruction from float64: <1e-7 maximum absolute error.
- Four selected finite-difference/autograd gradient checks: pass.

**Do not interpret a maximum 251% relative score error as a 251% detection
error.** It is a denominator effect at near-zero scores. Nonetheless,
this finding prevents claiming exact floating-point equality. Real
activation scales and GPU parity must also be tested.

## Stage 2: saved predictions and real checkpoint gate parity

This stage does not run inference for PR analysis: it reads
`evaluation-val-size/predictions.pt` from the existing runs, and computes:

- TP, FP and ignored detections at seven fixed confidence thresholds
  (`0.001, .005, .01, .05, .1, .25, .5`) in all four size bins;
- precision, recall, FP per image, confidence of TP/FP;
- exact gained and lost annotated 0–8px GT IDs (A3 vs A0).

The matching uses the *same* target-bin-first / excluded-bin-ignore
interpretation as the existing custom size AP evaluation. This is an
observational breakdown, **not** a new ground-truth metric definition.

Separately, each frozen model is forwarded on the same sampled
validation images, one image per step on an L4. P3 layer-16 input and
layer-17 output are captured with PyTorch hooks; output must numerically
match the original module's mathematical gate. Any mismatch raises and
blocks the final report.

For *each* checkpoint, both contrast and BTRM `z` and `g` are
recalculated from the **same** P3 features. This controls for
different backbone weights between separately trained A2 and A3.
Measure full-map gate saturation `fraction(g>=0.95)`, score medians
and quantiles, learned signed `effective_gain`, and GT-center versus
one background control near that GT on the same image.

An extra check counts eight **neighbor cell centers inside the GT box**.
This is a coarse geometric indicator of possible foreground
contamination in the "background ring", **not** an effective receptive-field
measurement. Background sampling excludes every annotated object,
including annotations not included in the per-image diagnostic subset.

## Run the combined staged diagnostic on Modal

From the new diagnostics branch:

```fish
git fetch origin
git switch research/btrm-math-and-mechanism-diagnostics
uv sync --extra cpu --group dev

# No GPU required for the independent proof.
uv run --extra cpu python scripts/btrm_math_audit.py

# This runs CPU math gate -> CPU saved PR -> L4 frozen feature analysis -> CPU report.
# There is NO training code in this entrypoint.
uv run --extra cpu modal run modal/app.py::btrm_diagnose --max-images 128
```

Use `--max-images 322` for exhaustive validation feature diagnostics.
128/322 images is a reproducible random subset, **not** a full-dataset
feature conclusion. The original 322-image saved detection predictions
are always analyzed exhaustively. Three L4 inference passes are run
sequentially, not three parallel training jobs.

The CPU mathematical gate is a prerequisite for the GPU stage; if it
fails, no GPU diagnostic will run.

### Artifacts

```text
tiny-person-runs:
  comparisons/btrm_diagnostics/audit-<ID>/
    math_audit.json
    prediction_summary.json
    precision_recall_thresholds.csv
    matched_deltas.csv
    tiny_gt_changes.csv
    mechanism_summary.json
    mechanism_report.md
    btrm_a2_gate_hist_0_8.png
    btrm_a3_gate_hist_0_8.png
    pr_thresholds_0_8.png
    pr_thresholds_8_16.png
  btrm_a0/<RUN_ID>/diagnostics/btrm_mechanism/val/
    samples.csv
    per_image.csv
    summary.json
  # Same three files for A2 and A3
```

Download the resulting Markdown report and CSVs using
`modal volume get tiny-person-runs comparisons/btrm_diagnostics/audit-<ID>/mechanism_report.md ./`
after the entrypoint prints the actual `audit-<ID>`.

## Interpretation / falsification

The analysis order is mandatory:

1. **Formula correctness**: ring least squares, ridge solution, normalization,
   gate. If independent oracle or gradient tests fail, do not proceed.
2. **Implementation correctness**: correct P3 connection, no NaN/Inf,
   checkpoint/output gate parity under real inputs. If this fails, fix
   the numerical or wiring bug before discussing AP.
3. **Mechanism**: only if (1) and (2) pass, check whether the gate is
   saturated, whether it separates GT from nearby background, whether
   the assumed background ring crosses GT, and whether TP gains are
   offset by FP and confidence-ranking losses.
4. **Research claim**: do not claim BTRM improves mAP based on the current
   seed-42 ablation, which shows a lower overall AP50 for A3.

The study cannot prove that a background ring is semantically correct:
feature-map receptive fields overlap, and a tiny object may influence
both center and neighborhood. Valid algebra does not imply that
`d_perp` is a better object-presence signal.
