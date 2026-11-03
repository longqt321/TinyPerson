# ĐỀ XUẤT NGHIÊN CỨU: QAC v4 — Distributional Quality Score + Frequency-Gated Background Suppression + Density-Adaptive Ranking

> **Mục tiêu**: Một hướng đi cải tiến tiềm năng cho phiên bản mới nhất của detector TinyPerson, dựa trên bằng chứng thực nghiệm của repo (oracle AP75, phân tích lỗi, chẩn đoán representation) và survey văn liệu 2025–2026.
>
> **Trạng thái**: Đề xuất nghiên cứu — chưa có code/experiment. Cần 1–2 vòng ablation để xác thực.
>
> **Forward-compat**: Thiết kế để chạy được trên cả YOLO26n-P2 (`reg_max=16`, có DFL) và YOLO26 (không DFL, end-to-end).

---

## 0. Tại sao QAC v3 thua, và bài học

### 0.1. Bằng chứng thực nghiệm

| Chỉ số | QAC v3 (ep50) | DFL16 (ep50) | Δ |
| :--- | ---: | ---: | ---: |
| mAP50-95 | 7.04% | **7.38%** | −0.34% |
| mAP50 | 22.86% | **23.48%** | −0.62% |
| Precision | 37.28% | **38.54%** | −1.26% |
| Recall | **28.71%** | 27.41% | +1.30% |
| Tổng FP | **817** | 982 | −165 (−16.8%) |
| AP@0.90 | **0.071%** | 0.010% | +0.061% (7×) |

QAC v3 thắng về **khử FP và AP@IoU cao**, nhưng thua về **mAP** và **recall tại ngưỡng confidence cứng 0.25**. Nguyên nhân gốc: QAC v2/v3 **điều biến logit phân loại in-place** (`c' = c + scale·(Q_geo ⊙ σ(c))`). Điều này conflates hai mục tiêu khác nhau — học phân loại (cần target TAL) và xếp hạng NMS (cần phản ánh chất lượng localization). Kết quả: Precision tăng nhưng Recall giảm, và mAP (trung bình trên IoU) không cải thiện.

### 0.2. Trần Oracle AP75 — phát hiện then chốt

| Variant | AP@0.75 | Diễn giải |
| :--- | ---: | :--- |
| baseline | 0.0237 | Model thật |
| **box** (thay hộp bằng GT) | 0.0237 | **Δ ≈ 0 — hồi quy hộp không phải nút thắt** |
| **score** (thay điểm bằng IoU) | **0.1355** | **×5.7 — nút thắt là XẾP HẠNG** |

> **Kết luận:** Toàn bộ headroom của AP75 nằm ở **thứ hạng độ tin cậy**, không ở độ chính xác hộp. Module hiệu quả phải tác động vào **score dùng cho NMS ranking**, không phải vào logit phân loại.

### 0.3. Bài học thiết kế

1. **Không điều biến logit in-place.** Tách biệt: classification head học phân loại; quality head học chất lượng localization; ranking score = f(cls, quality).
2. **Không cạnh tranh với DFL head về tham số.** DFL16 (reg_max 1→16) đóng góp +1.81 điểm AP. Giữ DFL (SADR8 cho thấy reg_max=8 cải thiện recall +3.6 điểm) và thêm module **additive**.
3. **Module phải giải quyết cả 2 nút thắt:** (a) ranking/score, (b) background FP (63% tổng FP).
4. **Density-aware**: very_crowded (81 ảnh = 51.5% GT) chỉ đạt recall 40.07%.

---

## 1. Hướng đi đề xuất: QAC v4 — "DQS-FGB-DAR"

**Tên đầy đủ**: **Distributional Quality Score + Frequency-Gated Background Suppression + Density-Adaptive Ranking**

Ba đóng góp cốt lõi, mỗi đóng góp giải quyết một nút thắt đã đo được:

| # | Đóng góp | Nút thắt giải quyết | Bằng chứng đo được |
| :--- | :--- | :--- | :--- |
| 1 | **Quality Score Head (QSH)** — dự đoán IoU quality, dùng cho ranking | AP75 (oracle score ×5.7) | Oracle: 0.0237 → 0.1355 |
| 2 | **Frequency-Gated Background Suppression (FGS)** | 63% FP là background | Error analysis: bg_FP 84,359 / 133,941 |
| 3 | **Density-Adaptive Ranking (DAR)** | very_crowded recall 40.07% | Error analysis: 81 ảnh = 51.5% GT |
| + | **Giữ SADR (reg_max=8)** | Recall +3.6 điểm | SADR8: miss 6,544 → 6,049 |

---

## 2. Đóng góp 1: Quality Score Head (QSH)

### 2.1. Cơ sở: GFLv2 DGQP (Distribution-Guided Quality Predictor)

GFLv2 (Li et al., NeurIPS 2020) nhận ra: **hình dạng phân phối hồi quy hộp nội tại tương quan với chất lượng localization**. Phân phối nhọn → chất lượng cao; phân phối phẳng/đa đỉnh → chất lượng thấp. DGQP trích xuất top-k xác suất từ phân phối DFL và qua MLP nhỏ dự đoán IoU quality, rồi dùng `J = C × I` cho cả QFL training và NMS ranking.

**Điểm khác biệt với QAC v2/v3**: QAC v2 dùng DFL sharpness làm **gate điều biến logit** (conflates). QSH dùng nó làm **score riêng cho ranking** (decoupled). Đây chính là thay đổi then chốt theo oracle.

### 2.2. Thiết kế cho TinyPerson

**Khi có DFL (reg_max > 1, codebase hiện tại):**

```
DFL logits (B, 4*reg_max, H, W)
        │
        ▼
  _extract_topk (k=5, như QAC v2)
        │
        ▼
  Top-5 prob per edge × 4 = 20 channels
        │
        ▼
  MLP(20 → 16 → 1) + Sigmoid  →  I ∈ (0,1)   [predicted IoU quality]
        │
        ▼
  Ranking score J = σ(cls_logit) × I   (hoặc I nếu calibrate ranking only)
```

**Khi không có DFL (YOLO26, reg_max=1):** thay top-k DFL bằng **thống kê feature** (mean, std, max của F_cls trên patch) — cùng MLP. Forward-compat bằng cách inject `quality_extractor` callable.

### 2.3. Target huấn luyện: NWD-based soft target (không phải IoU cứng)

Vấn đề: IoU cứng có "cliff" cho tiny objects — 1 pixel lệch → IoU sụt mạnh → gradient không ổn định. QAC v2 SATAR đã dùng NWD. QSH dùng NWD làm target:

$$I_{\text{target}} = \text{NWD}(box_{\text{pred}}, box_{\text{gt}}) = \exp\left(-\frac{\sqrt{\mathcal{W}_2^2}}{C}\right)$$

với $C = \max(12.8, 0.8\sqrt{w_t \cdot h_t})$ (adaptive theo scale, như QAC v3 đề xuất). Loss: QFL (Quality Focal Loss) với target $I_{\text{target}}$.

**Tại sao NWD target tốt hơn IoU target cho tiny:** NWD cung cấp gradient liên tục ngay cả khi box không overlap (IoU=0), và bão hòa chậm hơn ở IoU cao (giúp AP75). Đây là lý do oracle score-only cho gain lớn.

### 2.4. Tham số bổ sung

| Thành phần | Tham số |
| :--- | ---: |
| Top-k extractor | 0 (deterministic) |
| MLP (20 → 16 → 1) | 20×16 + 16 + 16×1 + 1 = **353** |
| Per-level (4 levels) | 4 × 353 = 1,412 |
| **Tổng QSH** | **~1.4k** (so với 2.064 của QAC v3, 88 của QAC v2) |

### 2.5. Khác biệt quyết định so với QAC v2/v3

| | QAC v2/v3 | QSH (v4) |
| :--- | :--- | :--- |
| Mục tiêu | Điều biến logit phân loại | Dự đoán chất lượng localization |
| Ảnh hưởng đến cls loss | Có (thay đổi target TAL) | **Không** — cls head học độc lập |
| Ảnh hưởng đến ranking/NMS | Gián tiếp (qua logit) | **Trực tiếp** (J = C × I) |
| Target | IoU^γ (cứng) | NWD (mềm, liên tục) |
| Phù hợp oracle | Không (oracle cho thấy score là lever) | **Có** |

---

## 3. Đóng góp 2: Frequency-Gated Background Suppression (FGS)

### 3.1. Cơ sở: Frequency-domain cho tiny object

63% FP là background (biển, cát, sóng). Các nghiên cứu 2025 (SET, FSDETR, FDSI-RTDETR, FD-YOLO) cho thấy: macro-structure và smooth background (sóng biển chậm) concentrate ở **low-frequency**; biên person và texture person ở **high-frequency / mid-high**. Learnable filter mask trên spectrum có thể suppress background clutter mà giữ edge person.

### 3.2. Thiết kế

```
P2 features F (B, C, H, W)
        │
        ├──► Spatial branch (identity, residual)
        │
        └──► Frequency branch:
              2D DCT/FFT → |spectrum|
                   │
                   ▼
              Learnable radial mask M (H, W)  [1 tham số mỗi tần số radial]
                   │
                   ▼
              Multiply spectrum by M
                   │
                   ▼
              Inverse FFT → F_freq
                   │
                   ▼
              Conv 1×1 (C → 1) + Sigmoid → G_freq ∈ (0,1)
        │
        ▼
  F_out = F + γ · (F ⊙ G_freq)    [residual, γ init 0.001]
```

**Tham số:** radial mask ~H/2 ≈ 160 (cho 320×320 P2) + Conv 1×1 (C×1) ≈ 33. Tổng ~200 tham số/level × 4 = 800. Rất nhẹ.

**Lý do chọn frequency thay vì 1×1 conv (QAC v2/v3):** QAC v2 gate là projection điểm cô lập — không thấy context 3×3, không phân biệt bọt biển đơn lẻ với person. Frequency branch nhìn toàn cục spectrum → phân biệt texture periodic (sóng) với edge aperiodic (person).

### 3.3. Khởi tạo và an toàn

- Mask khởi tạo = 1 (identity) cho mọi tần số → tại epoch 0, `F_out = F` (không thay đổi).
- `γ` khởi tạo 0.001 → residual nhỏ.
- Dùng FFT real (`rfft2`) để tiết kiệm; spectrum magnitude (không phase) để bất biến dịch chuyển.

---

## 4. Đóng góp 3: Density-Adaptive Ranking (DAR)

### 4.1. Vấn đề

very_crowded: 81 ảnh chứa 7,048 GT (51.5% tổng), recall chỉ 40.07%. Standard NMS với IoU threshold 0.5 suppress quá nhiều box trong đám đông. QAC v2/v3 không giải quyết vấn đề này.

### 4.2. Thiết kế

Ba thành phần, tất cả là **post-processing / inference-time**, không thay đổi huấn luyện:

1. **Density estimation từ QSH**: tổng số detection có score > 0.05 trên ảnh → nếu > 100, classify "dense".
2. **Adaptive NMS IoU threshold**: dense → 0.45 (thay vì 0.50, cho phép box chồng lấn hơn); sparse → 0.50.
3. **Soft-NMS (Gaussian penalty)** thay vì hard suppression: score *= exp(−iou²/σ²) với σ=0.5. Giữ box nếu score còn ≥ 0.001.
4. **WBF (Weighted Boxes Fusion)** khi có TTA hoặc ensemble: blend box thay vì suppress. Research cho thấy WBF đạt miss rate thấp nhất trong dense crowd.

### 4.3. Chỉ số mục tiêu

| Chỉ số | Baseline | Mục tiêu |
| :--- | ---: | ---: |
| Recall very_crowded | 40.07% | ≥ 50% |
| Tổng recall | 52.19% | ≥ 56% |

---

## 5. Giữ SADR (reg_max=8) — đóng góp thứ 4 (kế thừa)

SADR8 đã chứng minh: AP@0.50 +2.87 điểm, Recall +3.62 điểm, Miss −495, bg_FP −6,151 trên official test. Đây là đóng góp **regression** (tìm thêm box), bổ sung cho QSH (ranking) và FGS (khử FP). Không cạnh tranh.

**Config**: `reg_max: 8` (không phải 16, không phải 1). Lý do: DFL clamp target `[0, 6.99]` trên mỗi level; reg_max=16 bóp cụt nhánh DFL ở P2/P3/P4/P5 cho box rộng. reg_max=8 cân bằng coverage và chi phí (+8.0% params).

---

## 6. Tổng hợp kiến trúc v4

```
Input (1280×1280)
   │
   ▼
Backbone YOLO26n + P2 (stride 4) + SPPF
   │
   ├──► P2 (stride 4) ──► [FGS: frequency-gated suppression] ──► P2' 
   │                                                              │
   │                                                              ▼
   │                                                          Detect (SADR, reg_max=8)
   │                                                              │
   │                                                              ├──► cls head (normal training, unchanged)
   │                                                              ├──► reg head (DFL, reg_max=8)
   │                                                              └──► QSH (quality score head)  ← MỚI
   │                                                                   │
   │                                                                   ▼
   │                                                              I ∈ (0,1) per detection
   │                                                                   │
   │                                                                   ▼
   │                                                              Ranking score J = σ(cls) × I
   │                                                                   │
   │                                                                   ▼
   │                                                              [DAR: density-adaptive NMS]
   │                                                                   │
   └──────────────────────────────────────────────────────────────────▼
                                                              Final detections
```

### 6.1. Tham số bổ sung so với baseline A2 (2,515,976)

| Module | Tham số | Ghi chú |
| :--- | ---: | :--- |
| SADR (reg_max 8 vs 1) | +201,960 | Đã đo |
| FGS (frequency gate, 4 levels) | ~800 | Mask radial + Conv 1×1 |
| QSH (quality score head, 4 levels) | ~1,412 | MLP 20→16→1 |
| **Tổng** | **~2,719,000** | +8.1% so với baseline |

### 6.2. Latency ước tính

| Thành phần | Ước tính |
| :--- | ---: |
| Baseline A2 | 16.37 ms |
| SADR (đã đo) | 16.06 ms (không tăng) |
| FGS (FFT 320×320 × 4, MLP nhẹ) | +0.5–1.0 ms |
| QSH (MLP 20→16→1 × 4 levels) | +0.1 ms |
| **Tổng v4** | **~17–18 ms** (~55–58 FPS) |

Vẫn trong ngưỡng real-time.

---

## 7. Thiết kế ablation (để xác thực)

### 7.1. Các biến thể cần chạy

| Biến thể | SADR | FGS | QSH | DAR | Mục đích |
| :--- | :---: | :---: | :---: | :---: | :--- |
| v4-full | ✓ | ✓ | ✓ | ✓ | Proposed |
| v4-no-QSH | ✓ | ✓ | ✗ | ✓ | Đo đóng góp QSH (ranking) |
| v4-no-FGS | ✓ | ✗ | ✓ | ✓ | Đo đóng góp FGS (bg FP) |
| v4-no-DAR | ✓ | ✓ | ✓ | ✗ | Đo đóng góp DAR (crowded) |
| v4-QSH-only | ✗ | ✗ | ✓ | ✗ | QSH đơn lẻ vs baseline |
| SADR8 (existing) | ✓ | ✗ | ✗ | ✗ | Baseline cải tiến hiện tại |

### 7.2. Chỉ tiêu mục tiêu (so với SADR8 hiện tại)

| Chỉ số | SADR8 (hiện tại) | Mục tiêu v4-full |
| :--- | ---: | ---: |
| AP@0.50 | 0.2589 | **≥ 0.275** |
| AP@0.75 | 0.0233 | **≥ 0.035** (đạt ≥ 50% headroom oracle) |
| Recall tổng | 0.5580 | **≥ 0.58** |
| Recall very_crowded | 0.4352 | **≥ 0.50** |
| Background FP | 78,208 | **≤ 65,000** (−17%) |
| tiny1 Recall | 0.2392 | **≥ 0.28** |

### 7.3. Giao thức

- Dataset: `/mnt/data/tinyperson_official_v1/data.yaml`, imgsz 1280, seed 42 (rồi 43, 44 nếu dương).
- Huấn luyện: 50 epoch chặng 1 (lr 0.001), +50 epoch nối tiếp (lr 0.0006, cosine) nếu chặng 1 dương.
- Đánh giá: validation + official test + phân tích hình học + phân tích lỗi, giữ nguyên protocol.
- **Tiêu chí dừng**: nếu AP@0.75 không tăng so với SADR8 sau 1 seed → dừng biến thể đó (theo invariant #7).

---

## 8. Tại sao đây là hướng đột phá, không phải lặp lại

| Hướng đã thử | Kết quả | Tại sao v4 khác |
| :--- | :--- | :--- |
| QAC v2/v3 (điều biến logit in-place) | mAP thua baseline | v4 **decouple**: cls head độc lập, quality head riêng cho ranking |
| MSR (feature normalization) | AP giảm, latency +6ms | v4 **không** normalize feature; chỉ gate frequency, residual nhỏ |
| BTRM (contrast enhancement) | AP@0.75 giảm | v4 **không** thay đổi feature; chỉ thêm score head |
| SADR8 (regression) | AP@0.50 +2.87 | v4 **giữ** SADR, thêm ranking + bg suppression |
| QAC v3 (Spatial-Relational Gate 1×1) | FP giảm nhưng mAP giảm | v4 **frequency-domain** gate, nhìn toàn cục spectrum |

**Khác biệt then chốt**: v4 tấn công trực tiếp vào **score dùng cho ranking** (oracle đã chứng minh đây là lever), thay vì điều biến logit (QAC) hay feature (MSR/BTRM). Đây là hướng mà không có biến thể nào trong repo đã thử.

---

## 9. Rủi ro và phương án dự phòng

| Rủi ro | Mức độ | Phương án |
| :--- | :--- | :--- |
| QSH target (NWD) không đủ discriminative cho tiny | Trung bình | Thử target = IoU + NWD hybrid; hoặc dùng GFLv2 gốc (top-k DFL sharpness) |
| FGS làm mất edge person (high-pass filter person) | Trung bình | Khởi tạo mask = 1 (identity); monitor gate_mean trong diagnostics |
| DAR làm tăng FP trong dense | Thấp | Soft-NMS penalty σ=0.5; theo dõi FP/image |
| Latency vượt ngưỡng real-time | Thấp | FGS dùng rfft2 (real FFT) + radial mask (160 param); QSH MLP 20→16→1 |
| Không tương thích YOLO26 (không DFL) | Thấp | QSH có 2 extractor: DFL-topk (nếu reg_max>1) hoặc feature-stats (nếu reg_max=1) |

---

## 10. Nguồn văn liệu tham khảo

1. **GFLv2** (Li et al., NeurIPS 2020) — Distribution-Guided Quality Predictor, top-k DFL sharpness → IoU quality → ranking score.
2. **NWD** (Wang et al., 2021) — Normalized Wasserstein Distance for tiny object detection, smooth gradient khi IoU=0.
3. **SET** (CVPR) — Spectral Enhancement for Tiny Object Detection, frequency-domain background smoothing.
4. **FSDETR / FD-YOLO** (2025) — Frequency-spatial feature pyramid, learnable filter mask cho background clutter.
5. **ScaleBridge-Det** (arXiv 2025) — Density-guided query allocation cho dense aerial scenes.
6. **DQ-DETR** (2024) — Dynamic query cho tiny object, count-guided feature enhancement.
7. **SO-RTDETR** (2025) — NWD loss + P2 enhancement cho aerial tiny objects.
8. **YOLO26** (Ultralytics, 2025) — STAL (small-target-aware label assignment), ProgLoss, loại bỏ DFL, end-to-end NMS-free.
9. **Soft-NMS** (Bodla et al., 2017) và **WBF** (Solovyev et al., 2021) — post-processing cho dense crowd.
10. **Uncertainty-Aware Gradient Stabilization** (2025) — treat localization as interval classification cho tiny objects.
11. **Oracle AP75 analysis** (repo này) — score-only oracle nâng AP75 ×5.7.

---

## 11. Kết luận

Hướng đi đề xuất **QAC v4 (DQS-FGB-DAR)** dựa trên 3 bằng chứng thực nghiệm mạnh nhất của repo:

1. **Oracle AP75** chứng minh ranking/score là nút thắt → QSH attacks trực tiếp.
2. **63% FP là background** → FGS frequency-domain attacks trực tiếp.
3. **very_crowded recall 40%** → DAR attacks trực tiếp.

Kết hợp với SADR (đã chứng minh hiệu quả) và keeping DFL (để không mất +1.81 điểm AP), đây là hướng **additive, không cạnh tranh** với bất kỳ thành phần nào đã có. Forward-compat với YOLO26 (không DFL) bằng cách swap quality extractor.

**Bước tiếp theo**: implement QSH + FGS trên codebase hiện tại (ước tính 2–3 ngày), chạy ablation §7.1 (6 biến thể), đánh giá theo §7.2. Nếu AP@0.75 ≥ 0.035 (đạt ≥ 50% headroom oracle) → mở rộng seed 43/44 và báo cáo. Nếu không → dừng, phân tích nguyên nhân, thử target hybrid.
