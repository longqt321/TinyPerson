# DCAF Autonomous Research Decision Memo — Cycle 2

**Cycle**: 2  
**Target Architecture**: DCAF v2.1 (Anti-Background Suppression & High-Band Residual Anchoring)  
**Baseline**: DCAF Cycle 1 Best (`mAP50-95 = 0.07226`, `mAP50 = 0.23865`, `Recall = 0.30238`, `Latency = 14.43 ms`)  
**Target Epochs**: 100 epochs (50 ep Stage 1 + 50 ep Stage 2)  
**Budget Accounting**: Productive consumed = 100, Wasted = 1, Budget remaining = 200 productive epochs.  
**Date**: 2026-11-02  

---

## 1. Quantitative Findings & Bottleneck Analysis from Cycle 1

In Cycle 1, DCAF v2 achieved significant empirical gains over Baseline A2:
- **mAP50-95 increased by +7.48%** ($0.06723 \to 0.07226$).
- **mAP50 increased by +9.69%** ($0.21757 \to 0.23865$).
- **High-IoU AP (AP0.80–0.90) increased by 5× to 8×**.
- **Tiny1 Recall reached 33.17%** (up from ~15% in previous architectures).

However, deep diagnostic analysis on the serialized checkpoint and predictions revealed three specific bottlenecks:
1. **High-Band Under-Utilization**:
   - The learned router output bias was $[-1.403, +1.021, +0.020, -0.037]$. The negative bias on High-Band ($-1.403$) suppressed $B_H$ relative to Mid-Band $B_M$.
   - While $B_M$ ($8–16$ px) performed well, tiny1 objects ($1–8$ px) still suffered 538 misses out of 805 GT instances (66.8% miss rate).
2. **Background False Positive Dominance (60.53% of all FPs)**:
   - Out of 14,318 total FPs at $conf=0.001$, 8,667 were pure background noise (water ripples, textures).
   - In Block 3, the Structural-Context branch receives $X_s = R_L + C_{\text{ctx}}$, which carries low-frequency context but lacks spatial saliency suppression. Uninformative background features are reconstructed into P2 without gating.
3. **AP75 Plateau & Tiny1 Vertical Bounding Bias ($dh = +4.61$ px)**:
   - While mAP50 surged, AP75 remained at $0.01867$, bounded by vertical aspect-ratio overestimation in micro-bounding boxes.

---

## 2. Architectural Hypotheses for DCAF v2.1

To address these bottlenecks without sacrificing inference speed or parameter compactness:

### Hypothesis 1: High-Band Direct Residual Pathway (H-BRP)
- **Mechanism**: In Block 3, instead of solely routing $F_{\text{band}} = a_H R_H + a_M R_M + a_L R_L$, add a direct, learnable high-band shortcut into the Detail-Focused branch:
  $$X_d = F_{\text{band}} + C_{\text{ctx}} + \gamma_H \cdot R_H$$
  where $\gamma_H$ is a learnable scalar initialized to $0.20$.
- **Hypothesis**: This prevents the router from completely extinguishing sub-8px high-frequency boundary gradients during early epochs, directly boosting Tiny1 recall and box boundary sharpness.

### Hypothesis 2: Background Saliency Gating (BSG) in Structural Context
- **Mechanism**: The Structural-Context branch is meant to anchor background discrimination. Instead of plain $X_s = R_L + C_{\text{ctx}}$, compute a foreground spatial attention mask from context:
  $$S_{\text{sal}} = \text{Sigmoid}(\text{Conv}_{1\times 1}(C_{\text{ctx}}))$$
  Refine $X_s$ through foreground-focused background attenuation:
  $$X_s = R_L \odot (1.0 - 0.5 \cdot S_{\text{sal}}) + C_{\text{ctx}}$$
  where background regions retain context smoothing while suppressing high-energy clutter.
- **Hypothesis**: Reduces Background FP by suppressing non-salient shallow feature activations in P2.

### Hypothesis 3: Dynamic Anisotropic Gaussian Initialization
- Scale 1 Gaussian $\sigma_1$ is re-centered around $0.60$ (closer to 1–6 px scale footprint), giving a tighter support for sub-8px person extraction.

---

## 3. Complexity & Parameter Impact

- Parameter delta: +1 Conv1x1 (32 $\to$ 1) in Context Controller = **33 parameters**.
- 1 scalar $\gamma_H$ in control group = **1 parameter**.
- Total module parameters: **34,666** (was 34,632, delta = +34 parameters, +0.001%).
- GFLOPs delta: $\approx 0.001$ GFLOPs.
- Theoretical Latency: Exactly identical (14.4 ms on A10).

---

## 4. Transfer Learning Strategy

- The backbone and neck weights from **Cycle 1 Best Checkpoint** (`dcaf_c1_best.pt`) are fully compatible and transferred.
- DCAF layers that share exact shapes (Gaussian scale space, band encoders, dilated context, detail/structural refinement, pixel-shuffle) will inherit Cycle 1 converged representations, allowing Cycle 2 to train as a high-precision refinement stage.
- Schedule: 50 epochs Stage 1 + 50 epochs Stage 2 (100 epochs total).
