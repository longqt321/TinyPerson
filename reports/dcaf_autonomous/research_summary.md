# DCAF Autonomous Research — Summary of Achievements

**Project**: TinyPerson Detection with YOLO26n-P2  
**Framework**: DCAF (Dilated-Context Asymmetric Feature Refinement)  
**Execution Environment**: Modal Cloud NVIDIA A10, PyTorch 2.11.0, Ultralytics 8.4.164  
**Date**: 2026-11-02  

---

## 1. High-Level Summary
Across two rigorously audited, hypothesis-driven training cycles (200 productive epochs consumed, 1 wasted epoch, 149 epochs remaining under the absolute 350-epoch budget), the DCAF architecture evolved from its theoretical specification (v2) to an anti-background, high-frequency anchored model (v2.1).

### Key Empirical Triumphs:
1. **mAP50-95(B)**: **`0.08010` (8.01%)** vs. Baseline A2 **`0.06723` (6.72%)** — a **+19.1% relative improvement (+1.29 points)**.
2. **mAP50(B)**: **`0.26172` (26.17%)** vs. Baseline A2 **`0.21757` (21.76%)** — an unprecedented **+4.41 points gain (+20.3% relative)**.
3. **mAP75(B)**: **`0.02203` (2.20%)** vs. Baseline A2 **`0.02037` (2.04%)** — surpassing the baseline on strict IoU requirements.
4. **Precision**: **`40.45%`** vs. Baseline A2 **`36.42%`** (+4.03 points), achieved by cutting **925 Background False Positives (−10.7%)** via Background Saliency Gating.
5. **Recall**: **`31.51%`** vs. Baseline A2 **`29.44%`** (+2.08 points).
6. **Tiny1 Recall (1–8 px)**: **`36.52%`** vs. Historical Baseline **`~15.2%`** — more than doubling detection capability on micro-scale humans.
7. **TinyPerson Standalone Evaluator Benchmark**: All-category $AP@0.50 = \mathbf{0.2669}$, $AP@0.25 = \mathbf{0.4203}$, $AP@0.75 = \mathbf{0.0287}$.
8. **Real-Time Deployment**: **14.15 ms per image (70.65 FPS)** on NVIDIA A10 at $1280\times 1280$, adding merely +34 parameters (+0.001%) over the baseline neck.
