# QAC v3 Adaptive: Self-Learning Mechanisms (Proposal)

> Status: proposal (not yet implemented). Companion to `QAC_V3_IMPROVEMENT.md` (hyperparameter-focused, rejected for being param-tuning). This document proposes **learned, input-dependent control** of the existing QAC v3 mechanisms instead of fixed scalars.

## 1. Why QAC v3 lost to DFL16 (controlled, 3.24M params)

| Metric | QAC v3 | DFL16 | Δ |
|---|---|---|---|
| mAP | 7.04 | 7.38 | −0.34 |
| Precision | 37.28 | 38.54 | −1.26 |
| Recall | 28.71 | 27.41 | +1.30 |
| FP | 817 | 982 | −16.8% |
| Miss | 1249 | 1115 | +12% |

QAC v3 trades FP for miss: it is **over-protective** (suppresses too many true positives) and **under-confident** on small objects. Three over-aggressive fixed scalars are the cause:

1. `veto_theta = −2.2` → `v_sem = 0.30·Q_geo` at `c=−2.2`, leaking background boost into FP-prone regions.
2. `quality_gamma = 0.65` clamped to `0.70` → over-lifts low-quality tiny boxes, distorting ranking.
3. `area_ref = 1024`, `beta_min = 3.0` → too little gradient protection for 8–16px objects (recall 30.02% vs QAC v2's 52.87% at `area_ref=576`, `beta_min=1.5`).

The deeper issue: these are **global constants**. The image content, object density, and size distribution vary per image and per level. A single scalar cannot be optimal for both a sparse image of 4px pedestrians and a dense crowd of 32px ones.

## 2. Design principle: learned functions of (level, image, object size)

Every control parameter becomes a **small learned function** of observable signals, with a **monotonic, bounded output** so the model cannot collapse to a degenerate regime. Concretely:

- `beta` (TAL exponent) — function of GT object area, already partially is. Make `area_ref` and the shape of the curve **learned per level**, and add an image-level residual.
- `gamma` (peak-lift exponent) — function of object area **and** current quality, so lifting only happens when the predicted quality is reliable.
- `veto_theta` / `veto_slope` — function of **image density** (global feature stats), so dense images get a stricter veto.
- `nwd_weight` — function of IoU **and** area, so high-IoU large objects use pure IoU.
- `calib_scale` — per-level **and** per-image via a tiny MLP on pooled features.
- `qual_loss_weight` — adaptive by foreground ratio, so sparse batches are not drowned.

Crucially, all adaptations are **bounded and residual**: the base value is the current QAC v3 default, and the network learns a **delta** (via `tanh`/`sigmoid` with a small scale). This guarantees graceful degradation: if adaptation is useless, the model reverts to v3 behavior. This is what prevents over-specialization to tiny objects — large/medium objects keep the IoU-dominated path because `s_scale → 1` already routes them there, and the learned deltas are regularized to be small.

## 3. Proposed mechanisms (M1–M6)

### M1. Size-conditioned `beta` via learned `area_ref_l` (per level)

Current: `beta = beta_min + (beta_max - beta_min) * s_scale`, `s_scale = sqrt(area_gt)/sqrt(area_ref)`.

Change: replace fixed `area_ref` with a **learned per-level reference** `area_ref_l = softplus(area_ref_raw_l)` initialized to current values (1024 for P2, 512 for P3, 256 for P4), plus a **per-image residual** `δ_img = MLP_pooled(features)` (scalar, bounded ±0.3 in sqrt-space). The curve shape becomes learnable by parameterizing `beta` as a 2-parameter monotonic function:

```
beta(area) = beta_min + (beta_max - beta_min) * sigmoid(k_l * (log area - log area_ref_l) + δ_img)
```

- `k_l` (steepness) learned per level, initialized to 1.0.
- Effect: P2 (tiny-heavy) gets a sharper transition so 8–16px objects receive `beta ≈ beta_min` (strong gradient protection), while medium objects on P3/P4 get `beta ≈ beta_max` (IoU-dominated). The model **learns** where the transition sits instead of us fixing 32px.

Why not over-specialize: the residual `δ_img` is image-conditioned, so a batch with only medium objects pushes `δ_img` up (less protection needed), and the loss regularizer (L2 on `δ_img` mean) keeps it near zero by default.

### M2. Quality-gated peak lifting (`gamma` becomes a function of predicted quality)

Current: `gamma_eff = gamma_min + (1 - gamma_min) * s_scale`, applied as `q_hybrid ^ gamma_eff`.

Problem: lifting a low-quality (uncertain) tiny box amplifies noise.

Change: gate the lift by the quality head's own confidence:

```
gamma_eff = gamma_min + (1 - gamma_min) * s_scale * sigmoid((q_pred - 0.5) / τ)
```

- `τ` (temperature) learned, init 0.1.
- For tiny objects with low predicted quality → `gamma_eff ≈ gamma_min` (no lift, no harm). For high-quality tiny objects → full lift.
- This directly targets the miss/FP tradeoff: only confident tiny detections get boosted.

### M3. Image-density-conditioned soft-veto

Current: `v_sem = σ(slope·c + shift)` with `shift = -theta·slope`, `theta = −2.2` (fixed per level).

Change: make `theta` and `slope` **functions of global image statistics**:

```
θ_img = θ_l + MLP_density(pool(features))   # bounded ±0.5
slope_img = softplus(slope_raw_l + MLP_density2(pool(features)))
```

- `MLP_density`: 2-layer, ~1k params, input = GAP of the (concatenated) level features.
- Dense images (high feature activation variance) → lower θ → stricter veto → fewer FP. Sparse images → higher θ → more recall.
- This addresses QAC v3's FP reduction (good) but miss increase (bad): the veto becomes **context-aware** instead of always-on.

### M4. IoU-aware NWD blending weight

Current: `w_nwd = nwd_w_max - (nwd_w_max - 0.15) * s_scale` (area only).

Change: also condition on the **current IoU** between pred and GT:

```
w_nwd = base_w_nwd(s_scale) * sigmoid(-k·(iou - 0.5)) 
```

- High IoU → weight → 0 (pure IoU, no NWD smoothing needed). Low IoU tiny → weight → max.
- Prevents NWD from smoothing away well-localized large boxes (a source of the precision drop).

### M5. Adaptive `calib_scale` per level + per image

Current: `calib_scale` is a learned `(nl,1,1,1)` parameter, init 0.65.

Change: keep the learned base, add a per-image residual:

```
scale_img = calib_scale_l * (1 + α * tanh(MLP(pool(features))))
```

- `α = 0.2` (small), so the image effect is a modulation, not an override.
- Images where calibration hurts (e.g., high-contrast) get reduced scale automatically.

### M6. Size-aware quality head (conditioning, not separate heads)

Current: quality head `pw_qual_head` is a 1×1 conv on `h_joint` (no size info).

Change: concatenate a **size embedding** derived from the DFL moment descriptor's `box_w, box_h` (already computed) to `h_joint` before the quality head. Implement as an extra 1×1 conv channel (cheap) rather than a separate MLP:

```
qual_input = cat(h_joint, size_embed(box_w, box_h))   # +4 channels
```

- Lets the quality head learn different quality criteria for 4px vs 32px objects **without** separate per-size heads (which would over-specialize).
- The size embedding is a continuous function of area → smooth generalization across sizes.

## 4. Why this is not over-specialization

Three safeguards:

1. **Bounded residuals**: every adaptation is `base * (1 + small_delta)` with `small_delta ∈ [-0.3, 0.3]`. Large/medium objects keep their IoU-dominated path.
2. **Monotonic size routing**: `s_scale` already routes large objects to `beta_max`, `gamma ≈ 1`, `w_nwd ≈ 0.15`. Adaptations only *modulate* within the route; they cannot flip a large object into the tiny regime.
3. **Regularization**: L2 on per-image residuals' batch-mean (encourages zero-mean adaptation) + entropy bonus on the size distribution of lifted boxes (prevents collapse to "always lift tiny").

Additionally, M6's continuous size embedding ensures the quality head generalizes to unseen sizes (interpolation), unlike discrete size bins.

## 5. Ablation plan (minimal, ordered)

Run on TinyPerson val, same 3.24M params, 100 epochs, seed 42:

| Stage | Mechanisms | Expected |
|---|---|---|
| A0 | QAC v3 baseline (current) | mAP 7.04 |
| A1 | + M1 (adaptive beta) | recall 8–16px ↑, mAP +0.2–0.4 |
| A2 | + M2 (quality-gated lift) | FP ↓, precision ↑ |
| A3 | + M3 (density veto) | miss ↓, recall ↑ |
| A4 | + M4 (IoU-aware NWD) | precision ↑, large-object AP stable |
| A5 | + M5, M6 | quality head calibration ↑, AP75 ↑ |

Key success metric beyond mAP: **AP75** (ranking bottleneck — oracle score-only gave ×5.7). Target: AP75 ≥ 0.06 (from 0.0237) without regressing AP50.

Generalization check: evaluate on **COCO person** (medium/large) to confirm no regression on non-tiny objects. Acceptance: COCO person AP unchanged within ±0.5.

## 6. Implementation scope

- Modify `src/tinydet/modules/qac_v3.py` only. Add ~6 small `nn.Parameter` + 3 tiny MLPs (total <50k params, <2% of model).
- Backward-compatible: new params default to current behavior (residuals init 0), so old checkpoints load via `strict=False`.
- Loss changes in `QualityAwareDetectionLossV3`: replace scalar reads with function calls; keep `base_target_scores_sum` normalization.
- Config: add `adaptive: true` flag to `configs/model/yolo26n_tiny_A2_p2_qac_v3.yaml` (default false for reproducibility).

## 7. Open risks

- Image-level MLPs add compute (~3% slower training) — acceptable.
- Density estimation from GAP may be noisy early in training — use EMA smoothing (momentum 0.9) on the pooled stats.
- Risk of the model learning to "always veto" → monitor veto_theta trajectory; add a floor at −3.0.

## 8. What we are NOT doing

- No new version (stays QAC v3).
- No hyperparameter-only changes (all params become learned).
- No separate per-size experts (avoids over-specialization; continuous conditioning instead).
