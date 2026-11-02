# Architecture Specification — Cycle 2: DCAF v2.1 (Anti-Background & High-Band Anchored)

## 1. Overview
- **Model**: YOLO26n with official P2 neck and integrated DCAF v2.1.
- **Model YAML**: [`configs/model/yolo26n_tiny_A2_p2_dcaf.yaml`](file:///home/longtran/Projects/thigiacmaytinh/configs/model/yolo26n_tiny_A2_p2_dcaf.yaml)
- **Module Implementation**: [`src/tinydet/modules/dcaf.py`](file:///home/longtran/Projects/thigiacmaytinh/src/tinydet/modules/dcaf.py)
- **Trainable Parameters**: **34,666** (+34 params over Cycle 1).
- **Exact MACs**: **1.0168 GMAC** (2.0335 GFLOPs).
- **Control Group Parameters**: 130 scalars (128 Gaussian $\sigma$ + 1 residual gain $\alpha$ + 1 high-band shortcut $\gamma_H$) with `weight_decay = 0.0`.

---

## 2. Key Architectural Upgrades from Cycle 1

1. **Direct High-Band Residual Anchor (H-BRA)**:
   - Detail-focused branch receives direct high-frequency gradient injection:
     $$X_d = F_{\text{band}} + C_{\text{ctx}} + \gamma_H \cdot R_H$$
   - $\gamma_H = 0.5 \cdot \text{sigmoid}(\gamma_{H,\text{raw}})$, initialized to $0.20$.
   - Directly remedies Cycle 1's under-utilization of $B_H$ (where router bias dropped to $-1.403$), protecting sub-8px person boundary features from being extinguished.

2. **Background Saliency Gating (BSG)**:
   - Structural-context branch refines low-frequency context while suppressing non-salient shallow feature activations:
     $$S_{\text{sal}} = \text{Sigmoid}(\text{Conv}_{1\times 1}(C_{\text{ctx}}))$$
     $$X_s = R_L \odot (1.0 - 0.5 \cdot S_{\text{sal}}) + C_{\text{ctx}}$$
   - Directly targets the 60.53% Background False Positives identified in the Cycle 1 diagnostic report.

---

## 3. Weight Initialization & Compatibility
- Backbone, Neck, and all pre-existing DCAF layers load converged weights from Cycle 1 `best.pt`.
- The new saliency gate Conv1x1 is initialized with $0.01 \times W_{\text{Kaiming}}$ and zero bias ($S_{\text{sal}} \approx 0.5$, smoothly interpolating without shock).
- $\gamma_{H,\text{raw}}$ initialized to $-0.4055 \implies \gamma_H = 0.20$.
