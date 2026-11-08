# Architecture Specification — Cycle 1: GauSS v2 (YOLO26n-P2)

## 1. Overview
- **Model**: YOLO26n with official P2 neck and integrated GauSS v2 (Gaussian Scale-Space Feature Alignment and Refinement).
- **Model YAML**: [`configs/model/yolo26n_tiny_A2_p2_gauss.yaml`](file:///home/longtran/Projects/thigiacmaytinh/configs/model/yolo26n_tiny_A2_p2_gauss.yaml)
- **Module Implementation**: [`src/tinydet/modules/gauss.py`](file:///home/longtran/Projects/thigiacmaytinh/src/tinydet/modules/gauss.py)
- **Configuration Contract**:
  - Input: `[C2, P3, P4]` corresponding to shapes `[B, 64, 320, 320]`, `[B, 64, 160, 160]`, `[B, 128, 80, 80]` at input $1280\times 1280$.
  - Output: Refined P2 feature map of shape `[B, 32, 320, 320]`.
  - Trainable parameters in GauSS: **34,632**.
  - Exact MACs (B=1): **1,016,750,080** (1.0168 GMAC / 2.0335 GFLOPs).
  - Total model parameters: **2,550,608** (only +34.6k params over baseline A2).

---

## 2. Block-by-Block Architecture

### Block 1: Multi-Scale Band Feature Encoder
1. **Projection**: $1\times 1$ convolutions map inputs to hidden dimension $D=32$:
   - $S_2 = \text{Conv}_{1\times 1}(C_2) \in \mathbb{R}^{B\times 32\times 320\times 320}$
   - $S_3 = \text{Conv}_{1\times 1}(P_3) \in \mathbb{R}^{B\times 32\times 160\times 160}$
   - $S_4 = \text{Conv}_{1\times 1}(P_4) \in \mathbb{R}^{B\times 32\times 80\times 80}$
   - $X_0 = \text{SiLU}(\text{BN}(S_2 + \text{NearestUp}(S_3)))$
2. **Separable 1D Gaussian Scale-Space Filtering**:
   - Reflect pad $(r, r, r, r)$ followed by horizontal and vertical depthwise 1D convs.
   - Scale 1 ($5\times 5$ support, radius 2): $\sigma_1 = 0.4 + 0.5 \cdot \text{sigmoid}(\sigma_{1,\text{raw}}) \in [0.4, 0.9]$, init $0.65$. Output: $L_1$.
   - Scale 2 ($9\times 9$ support, radius 4): $\sigma_2 = 1.1 + 0.7 \cdot \text{sigmoid}(\sigma_{2,\text{raw}}) \in [1.1, 1.8]$, init $1.45$. Output: $L_2$.
3. **Explicit Band Decomposition**:
   - High Band: $B_H = X_0 - L_1$
   - Mid Band: $B_M = L_1 - L_2$
   - Low Band: $B_L = L_2$
4. **Band Refinement**:
   - Each band passes through $\text{PixelUnshuffle}(2)$ ($32\times 320\times 320 \to 128\times 160\times 160$).
   - Projected via $\text{Conv}_{1\times 1}(128\to 32) + \text{BN} + \text{SiLU}$.
   - Depthwise $3\times 3$ conv + Pointwise $1\times 1$ conv + residual add + SiLU $\to R_H, R_M, R_L \in \mathbb{R}^{B\times 32\times 160\times 160}$.

### Block 2: Gated Dilated Context Controller
1. **Context Feature Construction**:
   - $C_0 = \text{SiLU}(\text{BN}(S_3 + \text{NearestUp}(S_4))) \in \mathbb{R}^{B\times 32\times 160\times 160}$.
2. **Parallel Dilated Depthwise Convolutions**:
   - $T_1 = \text{DWConv}_{3\times 3, d=1}(C_0)$
   - $T_2 = \text{DWConv}_{3\times 3, d=2}(C_0)$
   - $T_3 = \text{DWConv}_{3\times 3, d=3}(C_0)$
3. **Context Softmax Gating**:
   - Gate logits $G_c = \text{Conv}_{1\times 1}(C_0) \in \mathbb{R}^{B\times 3\times 160\times 160}$.
   - Softmax over 3 channels $\to (\pi_1, \pi_2, \pi_3)$.
   - $C_{\text{ctx}} = \text{SiLU}(C_0 + \sum_{k=1}^3 \pi_k T_k)$.
4. **Spatial Routing Controller**:
   - $U = \text{SiLU}(R_H + R_M + R_L + C_{\text{ctx}})$.
   - $V = \text{SiLU}(\text{BN}(\text{DWConv}_{3\times 3}(U)))$.
   - Logits $L_{\text{route}} = \text{Conv}_{1\times 1}(V) \in \mathbb{R}^{B\times 4\times 160\times 160}$.
   - Channels 0..2: Softmax $\to (a_H, a_M, a_L)$ (adaptive band weights).
   - Channel 3: Sigmoid $\to M \in (0, 1)$ (spatial mixing gate). Initialized with weight scaled by $0.01$ and zero bias $\implies M \approx 0.5$.

### Block 3: Asymmetric Feature Refinement & Reconstruction
1. **Adaptive Band Fusion**:
   - $F_{\text{band}} = a_H R_H + a_M R_M + a_L R_L$.
2. **Detail-Focused Branch**:
   - Input: $X_d = F_{\text{band}} + C_{\text{ctx}}$.
   - Refinement: DWConv $3\times 3$ + PWConv $1\times 1$ + residual add $\to F_d$.
3. **Structural-Context Branch**:
   - Input: $X_s = R_L + C_{\text{ctx}}$.
   - Refinement: DWConv $5\times 5$ + PWConv $1\times 1$ + residual add $\to F_s$.
4. **Spatial Mixing**:
   - $F_{\text{route}} = M \odot F_d + (1 - M) \odot F_s \in \mathbb{R}^{B\times 32\times 160\times 160}$.
5. **Reconstruction & Bounded Residual Projection**:
   - $Z_0 = \text{BN}(\text{Conv}_{1\times 1}(F_{\text{route}})) \in \mathbb{R}^{B\times 128\times 160\times 160}$.
   - $Z = \text{PixelShuffle}(2)(Z_0) \in \mathbb{R}^{B\times 32\times 320\times 320}$.
   - Learnable residual gain $\alpha = 0.5 \cdot \text{sigmoid}(a_\alpha) \in (0, 0.5)$, init $a_\alpha = 0 \implies \alpha = 0.25$.
   - Output: $Y_{P2} = \text{SiLU}(X_0 + \alpha Z)$.

---

## 3. Optimizer Configuration & Parameter Isolation
- **Control Group Parameters** (129 scalars):
  - `gaussian.sigma1x_raw` (32)
  - `gaussian.sigma1y_raw` (32)
  - `gaussian.sigma2x_raw` (32)
  - `gaussian.sigma2y_raw` (32)
  - `alpha_raw` (1)
- **Rules Verified**:
  - `weight_decay = 0.0`.
  - Base learning rate and warmup identical to weight group.
  - Dynamically configured via `_setup_gauss_optimizer` callback at `on_pretrain_routine_end`.
