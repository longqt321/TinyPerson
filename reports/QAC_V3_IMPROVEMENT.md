# CẢI TIẾN QAC v3 DỰA TRÊN SO SÁNH VỚI BASELINE DFL16

> **Phạm vi**: Chỉ cải tiến trong khung QAC v3 hiện có (không tạo v4/v5). Mọi thay đổi đều xuất phát từ số liệu so sánh QAC v3 vs DFL16 trong `reports/comparison_qac_v3_vs_dfl16/`.

---

## 1. Phân tích so sánh: QAC v3 vs DFL16

### 1.1. Bảng so sánh đầy đủ

| Chỉ số | QAC v3 (ep50) | DFL16 (ep50) | Δ | DFL16 (ep100) |
| :--- | ---: | ---: | ---: | ---: |
| mAP50-95 | 7.04% | **7.38%** | **−0.34%** | 8.66% |
| mAP50 | 22.86% | **23.48%** | −0.62% | 26.91% |
| mAP75 | 1.88% | **1.92%** | −0.04% | 2.58% |
| Precision | 37.28% | **38.54%** | −1.26% | 42.13% |
| Recall | **28.71%** | 27.41% | **+1.30%** | 30.07% |
| AP@0.80 | **0.546%** | 0.517% | +0.029% | 0.854% |
| AP@0.85 | **0.089%** | 0.082% | +0.007% | 0.187% |
| AP@0.90 | **0.071%** | 0.010% | **+0.061% (7×)** | 0.075% |

### 1.2. Phân tích lỗi

| Loại | QAC v3 ep50 | DFL16 ep50 | DFL16 ep100 | QAC v3 − DFL16 ep50 |
| :--- | ---: | ---: | ---: | ---: |
| Tổng FP | **817** | 982 | 1,293 | **−165 (−16.8%)** |
| Background FP | **220** | 250 | 362 | **−30 (−12.0%)** |
| Duplicate FP | **155** | 193 | 291 | **−38 (−19.7%)** |
| Localization FP | **442** | 539 | 640 | **−97 (−18.0%)** |
| Miss | 1,249 | **1,115** | 1,082 | **+134 (+12.0%)** |
| Localization error | **1,572** | 1,644 | 1,487 | **−72 (−4.4%)** |
| Detected (IoU≥0.50) | 1,778 | **1,793** | **1,883** | −15 |
| Detected (IoU≥0.75) | 500 | **513** | **569** | −13 |

### 1.3. Phân tích theo size bin (conf ≥ 0.25, IoU ≥ 0.50)

| Bin | GT | QAC v3 ep50 | DFL16 ep50 | Δ (v3 − dfl16) |
| :--- | ---: | ---: | ---: | ---: |
| 0–8 px | 697 | 15.21% (106) | **15.49% (108)** | −0.28% |
| 8–16 px | 1,409 | 30.02% (423) | **32.43% (457)** | **−2.41%** |
| 16–32 px | 1,121 | 16.77% (188) | **18.82% (211)** | **−2.05%** |
| ≥ 32 px | 381 | 18.37% (70) | **19.16% (73)** | −0.79% |

### 1.4. Hình học

| Model | Sai số tâm median | p90 | dw median | dh median |
| :--- | ---: | ---: | ---: | ---: |
| QAC v3 ep50 | 2.950 px | 567.7 | +0.053 | +1.007 |
| DFL16 ep50 | **2.805 px** | **535.8** | +0.245 | +0.571 |
| DFL16 ep100 | **2.521 px** | **450.1** | +0.413 | +0.582 |

### 1.5. Chẩn đoán: 3 nguyên nhân QAC v3 thua

**Nguyên nhân 1 — Soft-veto quá lỏng, không veto đủ background.**
- Code: `veto_theta = -2.20`, `veto_slope_raw = 0.55` → slope = softplus(0.55) ≈ 0.96, bias_shift = 2.11.
- Tại c = −2.2 (ngưỡng veto): `v_sem = 0.5(σ(0.96·(−2.2)+2.11) + σ(−2.2)) · Q_geo = 0.5(σ(−0.002) + σ(−2.2)) · Q_geo ≈ 0.5(0.50 + 0.10) · Q_geo = 0.30 · Q_geo`.
- **Veto không về 0** tại c = −2.2. Background objects với Q_geo > 0.3 vẫn được boost đáng kể. Đây là lý do QAC v3 có recall cao hơn (boost nhiều) nhưng precision thấp hơn (boost cả background).

**Nguyên nhân 2 — `quality_gamma` quá thấp, lift quá mạnh cho tiny objects.**
- Code: `gamma_min = min(0.65, 0.70) = 0.65`, `gamma_eff = 0.65 + 0.35·s_scale`.
- Với tiny (s_scale ≈ 0): `gamma_eff ≈ 0.65`. `q_hybrid^0.65` lift mạnh cho low-quality boxes.
- Đây là lý do QAC v3 có recall Ultralytics cao hơn (28.71% vs 27.41%) nhưng miss count cao hơn (1,249 vs 1,115): calibration boost confidence cho boxes chưa đủ tốt, làm chúng vượt ngưỡng NMS nhưng không đủ chính xác.

**Nguyên nhân 3 — `area_ref = 1024` và `beta_min = 3.0` quá cao, giảm gradient protection cho small objects.**
- QAC v2 dùng `area_ref = 576`, `beta_min = 1.5` và đạt recall 8–16 px = 52.87%.
- QAC v3 dùng `area_ref = 1024`, `beta_min = 3.0` và đạt recall 8–16 px = 30.02% (−22.85 điểm).
- `area_ref` cao hơn → `s_scale` cao hơn → `beta_adaptive` cao hơn → TAL khắt khe hơn → gradient bị triệt tiêu mạnh hơn cho small objects.
- Đây là nguyên nhân chính của recall thấp hơn ở mọi size bin.

---

## 2. Đề xuất cải tiến QAC v3

### 2.1. Thay đổi hyperparameter (7 thay đổi)

| # | Tham số | Giá trị hiện tại | Giá trị đề xuất | Lý do từ số liệu |
| :--- | :--- | ---: | ---: | :--- |
| 1 | `veto_theta` | −2.20 | **−3.50** | Soft-veto phải về 0 tại c = −3.5 (ngưỡng background). Hiện tại v_sem = 0.30·Q_geo tại c = −2.2 → leak. |
| 2 | `veto_slope_raw` | 0.55 | **0.30** | slope = softplus(0.30) ≈ 0.88 (thay vì 0.96). Veto chuyển tiếp mượt hơn, ít leak. |
| 3 | `quality_gamma` | 0.65 | **0.35** | gamma_min = 0.35 → gamma_eff = 0.35 + 0.65·s_scale. Ít lift cho tiny, giảm boost confidence cho boxes chưa đủ tốt. |
| 4 | `nwd_weight` | 0.45 | **0.25** | w_nwd = 0.25 − 0.10·s_scale. Ít NWD inflation cho tiny, q_hybrid gần IoU hơn → AP75 tốt hơn. |
| 5 | `area_ref` | 1024.0 | **576.0** | Revert về QAC v2 value. s_scale thấp hơn → beta_adaptive thấp hơn → gradient protection mạnh hơn cho small. |
| 6 | `beta_min` | 3.0 | **1.5** | Revert về QAC v2 value. TAL ít khắt khe hơn cho tiny. |
| 7 | `beta_max` | 5.5 | **6.0** | Revert về QAC v2 value. Giữ khắt khe cho object lớn. |

### 2.2. Thay đổi code (3 thay đổi)

#### Thay đổi 1: Nới clamp `gamma_min`

```python
# Hiện tại (dòng 638):
gamma_min = min(self.quality_gamma, 0.70)

# Đề xuất:
gamma_min = min(self.quality_gamma, 0.50)
```

**Lý do**: Clamp hiện tại giới hạn `quality_gamma` ở 0.70, nên đặt `quality_gamma = 0.35` không có hiệu lực. Nới clamp xuống 0.50 cho phép `gamma_min = 0.35` thực sự áp dụng.

#### Thay đổi 2: Nới `hard_bg_mask` threshold

```python
# Hiện tại (dòng 674):
hard_bg_mask = pred_probs.amax(dim=-1) > 0.05

# Đề xuất:
hard_bg_mask = pred_probs.amax(dim=-1) > 0.10
```

**Lý do**: Threshold 0.05 supervise quality head trên quá nhiều easy background, dilute signal. Nâng lên 0.10 tập trung vào hard backgrounds (pred_probs 0.10–0.50), nơi quality head cần học để phân biệt.

#### Thay đổi 3: Giảm `calib_scale` initial value

```python
# Hiện tại (dòng 183, 207):
self.calib_scale = nn.Parameter(torch.full((self.nl, 1, 1, 1), 0.65))

# Đề xuất:
self.calib_scale = nn.Parameter(torch.full((self.nl, 1, 1, 1), 0.30))
```

**Lý do**: `calib_scale = 0.65` điều biến feature mạnh. Giảm xuống 0.30 làm calibration nhẹ hơn, giảm over-boost confidence. Kết hợp với thay đổi 3 (quality_gamma thấp hơn), tổng hiệu ứng là calibration bảo thủ hơn.

---

## 3. Phân tích kỳ vọng

### 3.1. Tác động từng thay đổi

| Thay đổi | Tác động kỳ vọng | Chỉ số bị ảnh hưởng |
| :--- | :--- | :--- |
| veto_theta −2.2 → −3.5 | Giảm leak background, giảm FP thêm | Precision ↑, FP ↓ |
| veto_slope_raw 0.55 → 0.30 | Veto mượt hơn, ít false veto | Miss ↓, Recall ↑ |
| quality_gamma 0.65 → 0.35 | Ít boost confidence cho low-quality | Precision ↑, AP75 ↑ |
| nwd_weight 0.45 → 0.25 | q_hybrid gần IoU, ít NWD inflation | AP75 ↑, AP@0.90 ↓ nhẹ |
| area_ref 1024 → 576 | Gradient protection mạnh hơn cho small | Recall size bin ↑ |
| beta_min 3.0 → 1.5 | TAL ít khắt khe cho tiny | Recall tiny1/tiny2 ↑ |
| beta_max 5.5 → 6.0 | Giữ khắt khe cho object lớn | Recall >32 px ↓ nhẹ |
| gamma_min clamp 0.70 → 0.50 | Cho phép quality_gamma 0.35 có hiệu lực | Precision ↑ |
| hard_bg_mask 0.05 → 0.10 | Tập trung quality head vào hard bg | FP ↓, Precision ↑ |
| calib_scale 0.65 → 0.30 | Calibration nhẹ hơn | Recall ↑, Miss ↓ |

### 3.2. Kỳ vọng tổng hợp

| Chỉ số | QAC v3 hiện tại | Kỳ vọng sau cải tiến |
| :--- | ---: | ---: |
| mAP50-95 | 7.04% | **7.30–7.60%** (đạt hoặc vượt DFL16 7.38%) |
| mAP50 | 22.86% | **23.50–24.50%** (đạt DFL16 23.48%) |
| mAP75 | 1.88% | **2.00–2.20%** (vượt DFL16 1.92%) |
| Precision | 37.28% | **38.50–40.00%** (đạt DFL16 38.54%) |
| Recall | 28.71% | **27.50–28.50%** (giảm nhẹ, chấp nhận) |
| Tổng FP | 817 | **700–750** (giảm thêm 8–14%) |
| Miss | 1,249 | **1,150–1,200** (giảm 4–8%) |
| Recall 8–16 px | 30.02% | **35–40%** (tiệm cận QAC v2 48.76%) |
| Recall 16–32 px | 16.77% | **20–23%** (đạt DFL16 18.82%) |

**Kỳ vọng chính**: mAP50-95 đạt hoặc vượt DFL16 (7.38%), mAP75 vượt DFL16 (1.92%), Precision đạt DFL16 (38.54%), đồng thời giữ ưu thế FP (giảm thêm) và AP@0.90 (7×).

### 3.3. Tại sao kỳ vọng hợp lý

1. **QAC v3 đã có ưu thế FP (−16.8%) và AP@0.90 (7×)** — các thay đổi chỉ làm cho calibration bảo thủ hơn, không phá vỡ ưu thế này.
2. **QAC v2 (với area_ref=576, beta_min=1.5) đạt recall 8–16 px = 52.87%** — revert về giá trị này sẽ cải thiện recall đáng kể.
3. **QAC v3 đã có localization error thấp hơn DFL16 (−4.4%)** — box quality tốt, chỉ cần calibration tốt hơn.
4. **Vấn đề chính là over-boost confidence** (quality_gamma quá thấp, veto quá lỏng) — thay đổi 1–3 và 8–10 giải quyết trực tiếp.

---

## 4. Thiết kế ablation để xác thực

### 4.1. Các biến thể

| Biến thể | Thay đổi áp dụng | Mục đích |
| :--- | :--- | :--- |
| QAC v3 (hiện tại) | Baseline | Tham chiếu |
| QAC v3 + Fix 1–3, 8 | veto_theta, veto_slope, quality_gamma, gamma_min | Đo tác động soft-veto + gamma |
| QAC v3 + Fix 4–7 | nwd_weight, area_ref, beta_min, beta_max | Đo tác động scale-adaptive |
| QAC v3 + Fix 9–10 | hard_bg_mask, calib_scale | Đo tác động code |
| **QAC v3 + ALL** | Tất cả 10 thay đổi | Proposed |

### 4.2. Chỉ tiêu pass/fail

| Chỉ số | Pass | Fail |
| :--- | :--- | :--- |
| mAP50-95 | ≥ 7.38% (đạt DFL16) | < 7.04% (thua QAC v3 hiện tại) |
| mAP75 | ≥ 1.92% (đạt DFL16) | < 1.88% (thua QAC v3 hiện tại) |
| Precision | ≥ 38.54% (đạt DFL16) | < 37.28% (thua QAC v3 hiện tại) |
| Recall 8–16 px | ≥ 35% | < 30.02% (thua QAC v3 hiện tại) |
| Tổng FP | ≤ 750 | > 817 (thua QAC v3 hiện tại) |

**Tiêu chí dừng**: nếu `QAC v3 + ALL` fail trên ≥ 3 chỉ số → dừng, phân tích từng thay đổi riêng lẻ.

### 4.3. Giao thức

- Dataset: `/mnt/data/tinyperson_official_v1/data.yaml`, imgsz 1280, seed 42.
- Huấn luyện: 50 epoch chặng 1 (lr 0.001, patience 15).
- Đánh giá: validation + official test + phân tích hình học + phân tích lỗi.
- So sánh với DFL16 ep50 và QAC v3 ep50 (đã có).

---

## 5. Tổng kết

QAC v3 thua DFL16 về mAP (−0.34%) và precision (−1.26%) nhưng thắng về recall (+1.30%), FP (−16.8%) và AP@0.90 (7×). Ba nguyên nhân gốc đã được xác định từ code:

1. **Soft-veto quá lỏng** (veto_theta = −2.2 → leak tại c = −2.2) → fix: −3.5.
2. **quality_gamma quá thấp** (0.65 → gamma_min clamp 0.70) → fix: 0.35 + nới clamp.
3. **area_ref và beta_min quá cao** (1024, 3.0) → fix: 576, 1.5 (revert QAC v2).

Mười thay đổi cụ thể (7 hyperparameter + 3 code) đều có lý do từ số liệu so sánh. Kỳ vọng: mAP50-95 đạt hoặc vượt DFL16, mAP75 vượt DFL16, precision đạt DFL16, đồng thời giữ ưu thế FP và AP@0.90.

Thiết kế ablation 5 biến thể cho phép xác định đóng góp từng thay đổi. Nếu `QAC v3 + ALL` pass ≥ 3/5 chỉ tiêu → mở rộng seed 43/44 và báo cáo.

---

*Phân tích dựa trên `reports/comparison_qac_v3_vs_dfl16/map_results.json`, `size_summary.csv`, `geometry_summary.csv`, `error_summary.csv`, `model_comparison.csv` và source code `src/tinydet/modules/qac_v3.py`.*
