# GauSS Autonomous Research Decision Memo — Cycle 1

**Cycle**: 1  
**Target Model**: YOLO26n-P2 + GauSS v2 (c1=[64, 64, 128], c2=32)  
**Baseline Reference**: YOLO26n-tiny_A2_p2_cont (100 epochs, seed 42, best val AP50-95=0.0672 / AP50=0.2176 / Recall=0.2943)  
**Date**: 2026-11-02  
**Target Training Epochs**: 100 epochs (50 ep Stage 1 + 50 ep Stage 2)  
**Budget Allocation**: Productive = 100, Wasted = 0, Remaining Productive = 200  

---

## 1. Problem Statement & Historical Evidence

Analysis of past experimental records on TinyPerson (`reports/experiment_results_report.md`):
1. **P2 Necessity**: Architectural ablation established that adding P2 (stride 4) produced the single largest gain in the project (+1.10 AP, +1.72 Recall). However, P2 shallow features contain high spatial resolution but poor semantic discrimination, leading to high Background False Positives (over 78,000–84,000 on official test).
2. **Tiny1/Tiny2 Bottleneck**: Objects spanning 1–8 px (tiny1) have extremely low recall (<16–29%). They are easily drowned out by background noise and high-frequency clutter.
3. **MSR and BTRM Pitfalls**:
   - MSR attempted local residual modulation, but $\gamma$ remained small (RMS update only 2–3% of input), causing 0.36–0.89 AP degradation.
   - BTRM suffered from scale conflicts: improving tiny1/tiny2 slightly while severely degrading tiny3 and small objects because it lacked scale-decoupled routing.
4. **Frequency & Scale-Space Insight (SET CVPR 2024 / SSFF)**: Tiny objects require anisotropic Gaussian low-pass filtering to separate pure high-frequency edge noise from genuine object structures, followed by multi-scale dilated context to guide spatial routing.

---

## 2. GauSS v2 Architectural Hypothesis

GauSS (Gaussian Scale-Space Feature Alignment and Refinement) addresses these bottlenecks through three tightly coupled macro-blocks:

1. **Multi-Scale Gaussian Scale-Space Feature Decomposition (Block 1)**:
   - Uses separable 1D learnable Gaussian filters at two distinct supports ($5\times 5$ with $\sigma_1 \in [0.4, 0.9]$, init 0.65; $9\times 9$ with $\sigma_2 \in [1.1, 1.8]$, init 1.45).
   - Decomposes P2 into three explicit frequency bands:
     - High band ($B_H = X_0 - L_1$): captures sub-8px high-frequency details.
     - Mid band ($B_M = L_1 - L_2$): captures 8–20px structural boundaries.
     - Low band ($B_L = L_2$): captures smooth background/context.
   - PixelUnshuffle($2\times$) downsamples bands to P3 spatial dimensions ($160\times 160$) so depthwise refinement operates efficiently.

2. **Gated Dilated Context Controller (Block 2)**:
   - Fuses semantic context from P3 ($64$ ch) and upsampled P4 ($128$ ch).
   - Employs 3 parallel dilated depthwise branches ($d \in \{1, 2, 3\}$) with softmax gating $\pi$ to dynamically aggregate local and large-field context without resolution loss.
   - Generates dynamic 4-channel spatial routing logits: 3 channels for adaptive band weighting $(a_H, a_M, a_L)$ and 1 channel for the spatial mixing gate $M$.

3. **Asymmetric Feature Refinement & Bounded Reconstruction (Block 3)**:
   - Detail-focused branch ($3\times 3$ DWConv) refines $F_{\text{band}} + C_{\text{ctx}}$.
   - Structural-context branch ($5\times 5$ DWConv) refines $R_L + C_{\text{ctx}}$ to stabilize background discrimination.
   - Asymmetric routing: $F_{\text{route}} = M \odot F_d + (1 - M) \odot F_s$.
   - Reconstruction via PixelShuffle back to $320\times 320$, scaled by learnable bounded residual gain $\alpha = 0.5 \cdot \text{sigmoid}(a_\alpha) \in (0, 0.5)$ (init 0.25).

---

## 3. Mathematical & Complexity Analysis

- **Trainable Parameters**: 34,632 (exact match with contract).
- **Control Parameters**: 129 scalars (128 Gaussian $\sigma_{\text{raw}}$ + 1 $\alpha_{\text{raw}}$).
  - All assigned to optimizer group `control` with `weight_decay = 0.0`.
- **Analytical MACs**: 1,016,750,080 MACs ($1.0168$ GMAC / $2.0335$ GFLOPs at $1280\times 1280$).
- **Total Model Parameters**: ~2,550,608 (an increase of only +1.38% over baseline A2's 2,515,976 params).
- **Latency Impact**: Theoretical overhead < 1.0 ms on NVIDIA A10.

---

## 4. Initialization & Gradient Flow Guarantees

1. **Mixing Gate Initialization**: The 4th channel of `route_out` is scaled by $0.01 \times W_{\text{Kaiming}}$ with zero bias. This guarantees $M \approx 0.5 \pm 0.005$ at initialization, preventing early branch collapse towards either purely detail or purely structural pathways.
2. **Residual Gain $\alpha$**: Initialized at $a_\alpha = 0 \implies \alpha = 0.25$, allowing the newly introduced module to inject meaningful residual gradients from epoch 1 while avoiding shock to pretrained backbone representations.
3. **Control Group Isolation**: Verified that `sigma_raw` and `alpha_raw` are exempted from weight decay in both Stage 1 and Stage 2, preventing shrinkage towards 0.

---

## 5. Decision & Execution Plan

- **Action**: Train YOLO26n-P2 with integrated GauSS v2 from scratch (pretrain backbone from baseline/YOLO26 weights where applicable, initialize GauSS with Kaiming + calibrated control init).
- **Schedule**:
  - Stage 1: 50 epochs, lr0=0.001, AdamW, batch 8, imgsz 1280.
  - Stage 2: 50 epochs, lr0=0.0006, lrf=0.02, cosine decay, warmup 3 epochs.
  - Total: 100 actual epochs.
- **Monitoring**: Real-time tracking of box loss, cls loss, DFL loss, Gaussian $\sigma$ drift, mixing gate mean/std, band routing weights $(a_H, a_M, a_L)$, and residual ratio.
