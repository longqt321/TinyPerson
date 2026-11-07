# BÁO CÁO KẾT QUẢ THỰC NGHIỆM

> [!NOTE]
> **Hồ sơ lịch sử.** Tài liệu này tổng hợp các hướng nghiên cứu đã dừng (A0–A3, MSR, BTRM, SADR, QAC v1–v3, oracle AP75). Kết quả hiện hành của dự án là GauSS/DCAF v2.1, xem [`gauss_final_report.md`](gauss_final_report.md).

## Phát hiện người tí hon trên TinyPerson với YOLO26n-P2 và các cơ chế hiệu chuẩn chất lượng

**Dự án**: TinyDet — Nghiên cứu Tiny Person Detection
**Tập dữ liệu**: TinyPerson Official v1
**Môi trường**: Modal Cloud — NVIDIA L4 / A10, CUDA 12.8, PyTorch 2.11.0, Ultralytics 8.4.164, Python 3.12
**Ngày tổng hợp**: 01/11/2026

---

## 1. Tóm tắt kết quả

Báo cáo trình bày kết quả của 6 nhóm thực nghiệm trên bài toán phát hiện người tí hon: (i) ablation kiến trúc A0–A3, (ii) module MSR đa tỷ lệ, (iii) module BTRM, (iv) phân phối hóa hồi quy hộp SADR, (v) chuỗi hiệu chuẩn chất lượng QAC v1/v2/v3, và (vi) các phân tích chẩn đoán độc lập (trần Oracle AP75, phân tích lỗi, chẩn đoán representation).

**Kết quả dương tính rõ ràng nhất:**

| Kết quả | Mức cải thiện |
| :--- | :--- |
| Kiến trúc A2 (P2 + SPPF) so với A0 baseline | AP +1.10 điểm, Recall +1.72 điểm |
| SADR8 (DFL `reg_max=8`) trên official test | AP@0.50 +2.87 điểm, Recall +3.62 điểm, Miss −495 |
| QAC v3 khử báo động giả | FP 817 so với 982 của baseline DFL16 (−16.8%) |

**Kết quả âm tính:**

| Thí nghiệm | Kết quả |
| :--- | :--- |
| MSR (cả 3 chế độ fusion) | AP thấp hơn baseline 0.36–0.89 điểm, độ trễ +4.6–6.2 ms |
| BTRM (scratch và warm-start) | AP@0.75 giảm 0.35–0.52 điểm |
| QAC v3 (đối chứng cùng dung lượng) | mAP50-95 thấp hơn baseline DFL16 0.34 điểm |

**Phát hiện chẩn đoán then chốt:** trần Oracle AP75 cho thấy nút thắt của chỉ số AP75 là **xếp hạng độ tin cậy**, không phải độ chính xác hồi quy hộp. Thay thế điểm số bằng IoU với ground-truth nâng AP75 từ 2.37% lên 13.55% (×5.7), trong khi thay thế hộp bằng ground-truth gần như không thay đổi kết quả.

---

## 2. Thiết lập thực nghiệm

### 2.1. Dữ liệu

| Hạng mục | Giá trị |
| :--- | :--- |
| Dataset huấn luyện | derivative 2 lớp cũ trên Modal Volume (đã ngừng dùng — xem ghi chú bên dưới) |
| Số ảnh derivative | 746 (662 train / 84 validation) |
| Tập kiểm tra chính thức | 816 ảnh, loại 30 ảnh dense theo protocol → **786 ảnh được chấm** |
| Ground-truth dương trên test | **13,687** |
| Độ phân giải đầu vào | 1280 × 1280 |
| Dataset signature | `c306a33f71f4628908c598749be1cef99a64a25ae5da178587bb6e6e2715579d` |

> **Ghi chú provenance (2026-11-07).** Báo cáo này là hồ sơ lịch sử của các run đã chạy;
> số liệu bên dưới giữ nguyên, không chỉnh sửa. Derivative 2 lớp mô tả trong báo cáo đã bị
> loại bỏ khỏi repo và không còn là nguồn dữ liệu chuẩn. Kiểm tra lại artifact sau này cho
> thấy các run thực tế trỏ `/mnt/data/tinyperson-yolo` với `val: images/train` và 2 lớp
> `sea_person`/`earth_person`; vì vậy cột "Validation" trong mục 3–4 được đo trên chính tập
> train và không phải validation độc lập. Derivative hiện hành là `tinyperson-yolo-v2`
> (một lớp `person`, val 84 ảnh tách theo nhóm video, test 786 ảnh riêng) — xem
> `docs/TINYPERSON_BENCHMARK.md`.

Phân bố kích thước đối tượng (derivative validation, 322 ảnh):

| Dải kích thước | Số instance | Số ảnh |
| :--- | ---: | ---: |
| 0–8 px | 798 | 84 |
| 8–16 px | 3,772 | 195 |
| 16–32 px | 4,682 | 228 |
| > 32 px | 2,765 | 209 |

Trên official test, phân bố theo protocol TinyPerson (`sqrt(area)`):

| Bin | Số GT |
| :--- | ---: |
| tiny1 (1–8 px) | 2,333 |
| tiny2 (8–12 px) | 2,778 |
| tiny3 (12–20 px) | 3,767 |
| small (20–32 px) | 2,485 |
| reasonable (> 32 px) | 2,324 |

### 2.2. Cấu hình huấn luyện

| Tham số | Giá trị |
| :--- | :--- |
| Optimizer | AdamW |
| Learning rate | 0.001 (chặng 1), 0.0006 (chặng nối tiếp) |
| Epochs | 50 (chặng 1), +50 (chặng nối tiếp) |
| Batch size | 8 |
| Patience | 15 (chặng 1), 20 (chặng nối tiếp) |
| AMP | Bật |
| Augmentation | translate 0.05, scale 0.15, fliplr 0.5, hsv nhẹ; mosaic 0, mixup 0, cutmix 0 |
| Seed | 42 |

### 2.3. Giao thức đánh giá

Ba tầng đánh giá được giữ tách biệt:

1. **Validation Ultralytics** trên derivative (84 ảnh, 3,608 GT) — dùng chọn checkpoint.
2. **Official TinyPerson test** (786 ảnh, 13,687 GT) — evaluator nội bộ repo, `conf=0.001`, `max_det=200`, AP tại IoU 0.25/0.50/0.75 theo size bin, hiệu chuẩn 101 điểm.
3. **Phân tích hình học và lỗi** — ghép cặp tham lam giảm dần confidence, ngưỡng `conf ≥ 0.25`, `IoU ≥ 0.50`.

Mọi metric báo cáo lấy từ checkpoint `best.pt`.

### 2.4. Kiến trúc các biến thể

| Ký hiệu | Đặc điểm |
| :--- | :--- |
| A0 | YOLO26n baseline, không P2 |
| A1 | A0 bỏ SPPF |
| A2 | Thêm nhánh P2 (stride 4), giữ SPPF |
| A3 | Thêm P2, bỏ SPPF |
| DFL16 | A2 với `reg_max=16` (bật hồi quy phân phối DFL) |
| SADR8 | A2 với `reg_max=8` |
| MSR | Module chuẩn hóa residual cục bộ đa tỷ lệ trên P2 (kernel 3/5/7) |
| BTRM | Module tương phản biên đa tỷ lệ trên P2, khởi tạo identity |
| QAC v1 | Cổng chất lượng 1×1 + hiệu chuẩn logit, 260 tham số |
| QAC v2 (D-QAC) | DUGC (top-5 DFL sharpness + Semantic-Veto) + SATAR (NWD + β(A)), 88 tham số |
| QAC v3 | Trích xuất moment DFL đầy đủ + Spatial-Relational Gate + Soft-Veto + NWD động, 2,064 tham số |

---

## 3. Kết quả ablation kiến trúc (A0–A3)

Validation, 50 epoch, seed 42, dataset official_v1 (xem ghi chú provenance ở mục 2.1 — cột này đo trên tập train):

| Model | AP@[.50:.95] | AP@0.50 | AP@0.75 | Precision | Recall | Params | Latency L4 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 baseline | 0.04810 | 0.17132 | 0.00845 | 0.3249 | 0.2353 | 2,504,190 | 13.85 ms |
| A1 (bỏ SPPF) | 0.05029 | 0.17459 | 0.01424 | 0.3223 | 0.2445 | 2,339,582 | 14.09 ms |
| **A2 (P2 + SPPF)** | **0.05912** | **0.19689** | **0.01585** | **0.3582** | **0.2525** | 2,515,976 | 16.37 ms |
| A3 (P2, bỏ SPPF) | 0.05591 | 0.17992 | — | 0.3333 | 0.2605 | — | 16.52 ms |

**Kết luận:** A2 đạt AP cao nhất. So với A0, A2 tăng AP +1.10 điểm, Recall +1.72 điểm, Precision +3.33 điểm, đổi lại độ trễ tăng 2.52 ms. A3 tăng Recall +2.52 điểm so với A0 nhưng Precision thấp hơn A2. Nhánh P2 là đóng góp kiến trúc có hiệu quả đo được lớn nhất trong toàn bộ nghiên cứu.

---

## 4. Kết quả module MSR

### 4.1. Ablation chế độ fusion (validation, 50 epoch)

| Run | AP@[.50:.95] | AP@0.50 | AP@0.75 | Precision | Recall | Params | Latency | VRAM infer |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 baseline | **0.05912** | **0.19689** | **0.01585** | **0.3582** | 0.2525 | 2,515,976 | 16.37 ms | 214.8 MB |
| MSR mean | 0.05549 | 0.18443 | 0.01400 | 0.3281 | 0.2657 | 2,523,657 | 20.93 ms | 292.8 MB |
| MSR adaptive | 0.05385 | 0.17457 | 0.01572 | 0.3051 | 0.2389 | 2,525,313 | 22.52 ms | 266.7 MB |
| MSR max | 0.05023 | 0.17275 | 0.01023 | 0.3172 | 0.2395 | 2,523,657 | 22.17 ms | 295.3 MB |

Cả ba chế độ đều thấp hơn baseline từ 0.36 đến 0.89 điểm AP, với độ trễ tăng 4.6–6.2 ms và VRAM tăng 52–80 MB. Chỉ số Recall của chế độ `mean` cao hơn baseline 1.32 điểm nhưng Precision giảm 3.01 điểm.

### 4.2. MSR trên nhánh official (4 biến thể, 50 epoch)

| Run | Số epoch hoàn tất | Best AP@[.50:.95] | AP@0.50 tại best | Peak VRAM |
| :--- | ---: | ---: | ---: | ---: |
| A0 identity | 50 | **0.0640** | 0.2085 | 12.21 GB |
| A1 local_max | 39 | 0.0607 | 0.1945 | — |
| A3 full_adaptive | 44 | 0.0561 | 0.1840 | — |
| A2 full_max | 39 | 0.0553 | 0.1883 | — |

Chế độ identity (không can thiệp) đạt AP cao nhất, cao hơn mọi biến thể MSR từ 0.33 đến 0.87 điểm.

### 4.3. Chẩn đoán nội bộ module MSR

Từ `msr_metrics.jsonl` (84–91 bản ghi mỗi run, thu thập mỗi 100 bước):

| Chỉ số | adaptive | max | mean |
| :--- | ---: | ---: | ---: |
| `modulation/gate_mean` (đầu → cuối) | 0.441 → 0.468 | 0.531 → 0.566 | 0.501 → 0.492 |
| `modulation/gate_std` | 0.248 → 0.314 | 0.276 → 0.307 | 0.237 → 0.299 |
| `modulation/near_zero_frac` | 0.006 → 0.084 | 0.053 → 0.059 | 0.002 → 0.055 |
| `modulation/near_one_frac` | 0.000 → 0.007 | 0.021 → 0.100 | 0.008 → 0.034 |
| `residual/gamma` | 0.0029 → 0.121 | 0.0040 → 0.106 | 0.0033 → 0.101 |
| `residual/update_input_ratio` | 0.0007 → 0.031 | 0.0014 → 0.033 | 0.0008 → 0.033 |
| `evidence/scale_3/mean` | 0.434 → 0.605 | 0.275 → 0.486 | 0.446 → 0.622 |
| `evidence/scale_7/mean` | 0.305 → 0.674 | 0.359 → 0.537 | 0.411 → 1.006 |
| `grad/egm_parameter_norm` | 3.30 → 3.93 | 3.29 → 3.69 | 3.29 → 3.98 |

Hệ số `gamma` học được tăng từ ~0.003 lên ~0.10–0.12, nhưng tỉ lệ đóng góp residual chỉ đạt 2–3% RMS của input — module duy trì gần trạng thái identity suốt quá trình huấn luyện. Chế độ `max` có `near_one_frac` tăng lên 10%, tức cổng bão hòa, tương ứng với kết quả AP thấp nhất. Toàn bộ `finite_frac = 1.0`, không có giá trị không hữu hạn.

---

## 5. Kết quả module BTRM

Official test, 786 ảnh, 13,687 GT:

| Metric | A2 | A2 + BTRM scratch | A2 adapt | A2 + BTRM adapt | BTRM γ=0 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| AP@0.25 | 0.4013 | 0.3768 | 0.4026 | 0.4027 | 0.3798 |
| AP@0.50 | 0.2301 | 0.2120 | 0.2343 | 0.2335 | 0.2056 |
| AP@0.75 | **0.0237** | 0.0202 | **0.0247** | 0.0195 | 0.0209 |
| AP mean | 0.2184 | 0.2030 | 0.2205 | 0.2186 | 0.2021 |

AP@0.75 theo size bin:

| Bin | A2 | BTRM scratch | A2 adapt | BTRM adapt | BTRM γ=0 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| all | 0.0237 | 0.0202 | 0.0247 | 0.0195 | 0.0209 |
| tiny | 0.0174 | 0.0102 | 0.0181 | 0.0181 | 0.0093 |
| tiny1 | 0.00345 | 0.00506 | 0.00552 | 0.00585 | 0.00224 |
| tiny2 | 0.0116 | 0.0157 | 0.0179 | 0.0177 | 0.0138 |
| tiny3 | 0.0229 | 0.0156 | 0.0230 | 0.0230 | 0.0141 |
| small | 0.0329 | 0.0240 | 0.0337 | 0.0339 | 0.0263 |
| reasonable | 0.0416 | 0.0392 | 0.0424 | 0.0369 | 0.0436 |

BTRM scratch tăng AP@0.75 ở tiny1 (+0.0016) và tiny2 (+0.0041) nhưng giảm mạnh ở tiny3 (−0.0073) và small (−0.0089). BTRM warm-start giữ nguyên tiny1/tiny2/tiny3 nhưng AP@0.75 tổng giảm do bin `reasonable` giảm 0.0055.

Validation:

| Run | AP | AP@0.50 | AP@0.75 | Precision | Recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A2 scratch | 0.05912 | 0.19689 | 0.01585 | 0.3582 | 0.2525 |
| BTRM scratch | 0.05899 | 0.19649 | 0.01227 | 0.3404 | 0.2658 |
| A2 adapt | 0.05977 | 0.19780 | 0.01340 | 0.3354 | 0.2583 |
| BTRM adapt | 0.05916 | 0.19694 | 0.01337 | 0.3309 | 0.2641 |

---

## 6. Kết quả SADR (phân phối hóa hồi quy hộp)

SADR8 thay `reg_max=1` bằng `reg_max=8`, cho phép mỗi biên dự đoán phân phối 8 bin thay vì một khoảng cách vô hướng. Thêm 201,960 tham số (2,515,976 → 2,717,936, +8.0%).

### 6.1. Validation (84 ảnh)

| Metric | A2 control | SADR8 | Δ |
| :--- | ---: | ---: | ---: |
| AP@[.50:.95] | 0.05912 | **0.06852** | **+0.0094** |
| AP@0.50 | 0.19689 | **0.21318** | **+0.0163** |
| AP@0.75 | 0.01585 | **0.01941** | **+0.0036** |
| Precision | 0.35815 | 0.35415 | −0.0040 |
| Recall | 0.25249 | **0.27993** | **+0.0274** |

### 6.2. Official test (786 ảnh)

| Metric | A2 control | SADR8 | Δ |
| :--- | ---: | ---: | ---: |
| AP@0.25 | 0.40127 | **0.42592** | **+0.0247** |
| AP@0.50 | 0.23014 | **0.25887** | **+0.0287** |
| AP@0.75 | 0.02375 | 0.02330 | −0.0004 |
| AP mean | 0.21838 | **0.23603** | **+0.0176** |
| tiny1 AP@0.75 | 0.00345 | **0.00579** | **+0.0023** |
| tiny2 AP@0.75 | 0.01163 | **0.01834** | **+0.0067** |
| tiny3 AP@0.75 | 0.02293 | 0.02343 | +0.0005 |
| small AP@0.75 | 0.03288 | **0.03517** | **+0.0023** |
| reasonable AP@0.75 | 0.04158 | **0.04563** | **+0.0040** |

AP@0.50 theo size bin:

| Bin | A2 | SADR8 | Δ |
| :--- | ---: | ---: | ---: |
| tiny1 | 0.1153 | 0.1214 | +0.0061 |
| tiny2 | 0.2223 | 0.2502 | +0.0279 |
| tiny3 | 0.2405 | 0.2934 | +0.0529 |
| small | 0.2767 | 0.3033 | +0.0266 |
| reasonable | 0.3085 | 0.3241 | +0.0156 |

### 6.3. Phân tích lỗi (IoU 0.50)

| Đại lượng | A2 | SADR8 | Δ |
| :--- | ---: | ---: | ---: |
| True positive | 7,143 | **7,638** | **+495** |
| Miss | 6,544 | **6,049** | **−495** |
| Tổng FP | 133,941 | 127,312 | −6,629 |
| Background FP | 84,359 | 78,208 | −6,151 |
| Localization FP | 39,121 | 38,549 | −572 |
| Duplicate FP | 10,461 | 10,555 | +94 |
| Recall | 0.5219 | **0.5580** | **+3.62 điểm** |
| Precision | 0.0506 | **0.0566** | **+0.60 điểm** |

Số GT đạt IoU ≥ 0.75 đếm trực tiếp từ samples: 1,877 → **2,177** (+300).

AP@0.75 tổng giảm nhẹ 0.0004 không do hộp kém chính xác hơn — số hộp đạt IoU ≥ 0.75 tăng 300 — mà do thay đổi thứ hạng confidence giữa các mức IoU. AP@0.75 của nhóm "tiny" gộp giảm nhẹ (0.0174 → 0.0161) trong khi tiny1/tiny2/tiny3 riêng lẻ đều tăng.

Độ trễ L4 batch 1: 16.06 ms (A2 control 16.44 ms), 260.9 MB.

---

## 7. Kết quả chuỗi QAC (hiệu chuẩn chất lượng)

### 7.1. QAC v1 và QAC v2 — validation và official test

| Chỉ số | Baseline A2 (50ep) | QAC v1 | QAC v2 (50ep) | QAC v2 cont (100ep) |
| :--- | ---: | ---: | ---: | ---: |
| mAP50-95 | 5.57% | 6.57% | 7.16% | **8.22%** |
| mAP50 | 18.40% | 20.78% | 23.07% | **26.09%** |
| AP75 | 1.14% | 2.03% | 2.10% | 2.07% |
| Precision | 34.27% | 35.56% | 39.59% | **42.47%** |
| Recall | 24.56% | 27.41% | 28.24% | 31.76% |
| Params | 2,515,976 | 3,241,464 | 3,241,464 | 3,241,464 |
| Latency A10 | 9.8 ms | 10.6 ms | 10.6 ms | 8.3 ms |

Phân rã 10 ngưỡng IoU của QAC v2 (50 epoch):

| IoU | Baseline A2 | QAC v1 | QAC v2 |
| :---: | ---: | ---: | ---: |
| 0.50 | 0.18397 | 0.20781 | 0.23069 |
| 0.55 | 0.14941 | 0.17001 | 0.18384 |
| 0.60 | 0.10673 | 0.12507 | 0.13275 |
| 0.65 | 0.06889 | 0.08127 | 0.08756 |
| 0.70 | 0.03252 | 0.04631 | 0.04920 |
| 0.75 | 0.01142 | 0.02025 | 0.02099 |
| 0.80 | 0.00382 | 0.00414 | 0.00962 |
| 0.85 | 0.00042 | 0.00136 | 0.00092 |
| 0.90 | 0.00001 | 0.00036 | 0.00013 |
| 0.95 | 0.00000 | 0.00000 | 0.00000 |

### 7.2. QAC v2 — phân tích hình học (3,608 GT, validation)

| Chỉ số | 50 epoch | 100 epoch |
| :--- | ---: | ---: |
| Detected | 1,276 | 1,377 |
| Miss | 998 | 948 |
| Localization error | 1,334 | 1,283 |
| Tổng FP | 3,248 | 2,996 |
| — localization FP | 1,633 | 1,383 |
| — background FP | 1,047 | 1,055 |
| — duplicate FP | 568 | 558 |
| Recall (conf 0.25, IoU 0.50) | 35.37% | **38.17%** |

Recall theo dải kích thước:

| Dải | GT | Detected 50ep → 100ep | Recall 50ep → 100ep | Sai số tâm median 50ep → 100ep |
| :--- | ---: | ---: | ---: | ---: |
| 0–8 px | 697 | 187 → 203 | 26.83% → 29.12% | 2.014 → 1.758 px |
| 8–16 px | 1,409 | 687 → 745 | 48.76% → **52.87%** | 1.720 → **1.566 px** |
| 16–32 px | 1,121 | 309 → 336 | 27.56% → 29.97% | 4.203 → 4.123 px |
| > 32 px | 381 | 93 → 93 | 24.41% → 24.41% | 8.482 → 8.476 px |
| Toàn bộ | 3,608 | 1,276 → 1,377 | 35.37% → 38.17% | 2.641 → 2.524 px |

Độ lệch kích thước hộp (median, 100 epoch): dải 0–8 px có `dh = +3.69 px` (phình chiều cao), dải > 32 px có `dh = −9.50 px` (co rút chiều cao).

Số đối tượng có sai số tâm ≤ 2 px: 1,575 / 3,608 (43.7%).

### 7.3. QAC v3 — đối chứng công bằng với Baseline DFL16

Cả hai mô hình dùng `reg_max=16`, `end2end=True`, kiến trúc dual-head YOLO26 đầy đủ. QAC v3 thêm 2,064 tham số (3,241,464 → 3,243,528, +0.06%).

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

Recall theo size bin (conf ≥ 0.25, IoU ≥ 0.50):

| Bin | GT | QAC v3 ep50 | DFL16 ep50 | DFL16 ep100 |
| :--- | ---: | ---: | ---: | ---: |
| 0–8 px | 697 | 15.21% (106) | **15.49% (108)** | 21.81% (152) |
| 8–16 px | 1,409 | 30.02% (423) | **32.43% (457)** | 39.03% (550) |
| 16–32 px | 1,121 | 16.77% (188) | **18.82% (211)** | 21.86% (245) |
| ≥ 32 px | 381 | 18.37% (70) | **19.16% (73)** | 24.15% (92) |
| Tổng phát hiện | 3,608 | 787 | **849** | 1,039 |

Phân tích lỗi (IoU 0.50, conf 0.001):

| Loại | QAC v3 ep50 | DFL16 ep50 | DFL16 ep100 |
| :--- | ---: | ---: | ---: |
| Tổng False Positive | **817** | 982 | 1,293 |
| Background FP | **220** | 250 | 362 |
| Duplicate FP | **155** | 193 | 291 |
| Localization FP | **442** | 539 | 640 |
| Miss | 1,249 | **1,115** | 1,082 |
| Localization error | 1,572 | 1,644 | 1,487 |

QAC v3 giảm 16.8% tổng FP so với baseline cùng dung lượng, trong đó background FP giảm 12.0% và duplicate FP giảm 19.7%. Đổi lại số miss tăng 134 và recall theo size bin thấp hơn ở cả 4 dải khi đo tại ngưỡng confidence cứng 0.25.

Hình học (3,608 GT):

| Model | IoU ≥ 0.50 | IoU ≥ 0.75 | Sai số tâm median | p90 | dw median | dh median |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| QAC v3 ep50 | 1,778 | 500 | 2.950 px | 567.7 | +0.053 | +1.007 |
| DFL16 ep50 | 1,793 | 513 | 2.805 px | 535.8 | +0.245 | +0.571 |
| DFL16 ep100 | **1,883** | **569** | **2.521 px** | **450.1** | +0.413 | +0.582 |

Sai số tâm median của QAC v3 tốt hơn baseline ở bin 0–8 px (2.162 so với 2.298 px) và ≥ 32 px (7.473 so với 7.803 px), kém hơn ở bin 8–16 px (1.759 so với 1.754 px) và 16–32 px (4.393 so với 4.273 px).

### 7.4. Tốc độ hội tụ giai đoạn đầu (QAC v3)

| Epoch | QAC v3 mAP50 | Baseline DFL16 mAP50 |
| :---: | ---: | ---: |
| 5 | **7.68%** | 1.52% |
| 13 | **16.07%** | 13.94% |

QAC v3 đạt mAP50 gấp 5 lần baseline ở epoch 5 và vượt trước ở epoch 13.

---

## 8. Phân tích trần Oracle AP75

Thí nghiệm thay thế có kiểm soát các thành phần dự đoán bằng ground-truth để đo headroom, giữ nguyên evaluator và protocol. A2 baseline, official test.

| Variant | AP@0.25 | AP@0.50 | AP@0.75 | Diễn giải |
| :--- | ---: | ---: | ---: | :--- |
| baseline | 0.4013 | 0.2301 | 0.0237 | Model thật |
| **box** — thay hộp bằng GT khi IoU ≥ 0.75 | 0.4012 | 0.2301 | 0.0237 | **Không thay đổi** |
| **score** — thay điểm bằng IoU lớn nhất với GT | 0.4455 | 0.3857 | **0.1355** | **AP75 ×5.7** |
| both — hộp GT + điểm nhị phân | 0.1823 | 0.1547 | 0.1192 | Sụp do điểm 0/1 cực đoan |

Chi tiết theo size bin của variant `score`:

| Bin | baseline AP@0.50 | score AP@0.50 | baseline AP@0.75 | score AP@0.75 |
| :--- | ---: | ---: | ---: | ---: |
| all | 0.2301 | 0.3857 | 0.0237 | **0.1355** |
| tiny1 | 0.1153 | 0.1734 | 0.00345 | **0.0289** |
| tiny2 | 0.2223 | 0.3815 | 0.0116 | **0.0977** |
| tiny3 | 0.2405 | 0.4133 | 0.0229 | **0.1354** |
| small | 0.2767 | 0.4812 | 0.0329 | **0.1927** |
| reasonable | 0.3085 | 0.5195 | 0.0416 | **0.2413** |

**Kết luận định lượng:** với mô hình hiện tại, độ chính xác hồi quy hộp không phải là yếu tố giới hạn AP75 (variant `box` cho Δ ≈ 0). Toàn bộ headroom nằm ở thứ hạng độ tin cậy (variant `score` nâng AP75 lên 13.55%). Headroom lớn nhất ở các bin lớn: reasonable 0.0416 → 0.2413 (×5.8), small 0.0329 → 0.1927 (×5.9); bin tiny1 chỉ 0.00345 → 0.0289 (×8.4 nhưng giá trị tuyệt đối vẫn rất thấp).

---

## 9. Phân tích lỗi trên official test

### 9.1. A2 baseline

| Đại lượng | Count | Tỉ lệ |
| :--- | ---: | ---: |
| Positive ground truth | 13,687 | — |
| Matched (true positive) | 7,143 | 52.19% |
| Missed | 6,544 | 47.81% |
| Predictions | 145,299 | — |
| False positive | 133,941 | 92.2% của prediction |
| — background FP | 84,359 | **63.0%** của FP |
| — localization FP | 39,121 | 29.2% của FP |
| — duplicate FP | 10,461 | 7.8% của FP |
| Neutral predictions | 4,215 | — |

### 9.2. Recall theo size bin (A2 baseline)

| Bin | GT | Matched | Missed | Recall | Miss rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| reasonable | 2,324 | 1,619 | 705 | 69.66% | 30.34% |
| small | 2,485 | 1,594 | 891 | 64.14% | 35.86% |
| tiny3 | 3,767 | 2,123 | 1,644 | 56.36% | 43.64% |
| tiny2 | 2,778 | 1,316 | 1,462 | 47.37% | 52.63% |
| **tiny1** | 2,333 | 491 | 1,842 | **21.05%** | **78.95%** |

### 9.3. Recall theo mật độ ảnh (A2 baseline)

| Density bin | Số ảnh | GT | Matched | Recall | FP/ảnh |
| :--- | ---: | ---: | ---: | ---: | ---: |
| sparse | 398 | 1,941 | 1,359 | 70.02% | 176.5 |
| single | 101 | 101 | 72 | 71.29% | 159.1 |
| crowded | 186 | 4,597 | 2,888 | 62.82% | 171.0 |
| **very_crowded** | 81 | **7,048** | 2,824 | **40.07%** | 157.6 |
| none | 20 | 0 | 0 | — | 152.8 |

81 ảnh `very_crowded` chứa 51.5% tổng ground-truth nhưng chỉ đạt recall 40.07%.

### 9.4. Vị trí đối tượng

| Vị trí | GT | Matched | Recall |
| :--- | ---: | ---: | ---: |
| interior | 13,302 | 6,973 | 52.42% |
| border | 385 | 170 | 44.16% |

### 9.5. Điểm vận hành confidence (A2 baseline)

| Ngưỡng | Predictions | TP | FP | Recall | Precision |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 145,299 | 7,143 | 133,941 | 52.19% | 5.06% |
| 0.005 | 102,236 | 7,051 | 91,304 | 51.52% | 7.17% |
| 0.010 | 82,375 | 6,948 | 71,787 | 50.76% | 8.82% |
| 0.020 | 63,090 | 6,744 | 53,194 | 49.27% | 11.25% |
| 0.050 | 40,591 | 6,233 | 32,101 | 45.54% | 16.26% |
| 0.100 | 24,573 | 5,397 | 17,852 | 39.43% | 23.21% |
| 0.200 | 9,661 | 3,713 | 5,495 | 27.13% | 40.32% |
| 0.500 | 445 | 345 | 91 | 2.52% | 79.13% |

Không tồn tại ngưỡng nào đạt đồng thời precision và recall cao: precision chỉ vượt 40% khi recall đã giảm dưới 28%.

### 9.6. So sánh lỗi giữa các biến thể

| Run | TP | Miss | FP | Loc FP | Bg FP | Dup FP | Recall | Precision |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 | 7,143 | 6,544 | 133,941 | 39,121 | 84,359 | 10,461 | 0.5219 | 0.0506 |
| A2 adapt | 7,208 | 6,479 | 128,487 | 39,910 | 77,917 | 10,660 | 0.5266 | 0.0531 |
| A2 + BTRM | 6,997 | 6,690 | 136,956 | 37,684 | 89,045 | 10,227 | 0.5112 | 0.0486 |
| A2 + BTRM adapt | 7,209 | 6,478 | 128,321 | 39,950 | 77,720 | 10,651 | 0.5267 | 0.0532 |
| A2 + BTRM γ=0 | 6,811 | 6,876 | 130,048 | 36,739 | 83,239 | 10,070 | 0.4976 | 0.0498 |
| A2 + SADR8 | **7,638** | **6,049** | **127,312** | 38,549 | **78,208** | 10,555 | **0.5580** | **0.0566** |

---

## 10. Benchmark tốc độ

Đo trên NVIDIA L4, batch 1, 1280×1280, warmup 10, 100 iteration:

| Run | Params | Latency | FPS | VRAM inference |
| :--- | ---: | ---: | ---: | ---: |
| A0 baseline | 2,504,190 | 13.85 ms | 72.2 | 148.6 MB |
| A1 (bỏ SPPF) | 2,339,582 | 14.09 ms | 71.0 | 147.9 MB |
| A2 (P2 + SPPF) | 2,515,976 | 16.37 ms | 61.1 | 214.8 MB |
| MSR mean | 2,523,657 | 20.93 ms | 47.8 | 292.8 MB |
| MSR max | 2,523,657 | 22.17 ms | 45.1 | 295.3 MB |
| MSR adaptive | 2,525,313 | 22.52 ms | 44.4 | 266.7 MB |

Trên A10 (từ report QAC v2): QAC v2 50 epoch đạt 10.6 ms (94.3 FPS), chặng nối tiếp 100 epoch đạt 8.3 ms (120.5 FPS), VRAM < 1.5 GB. Phân rã: tiền xử lý 1.0 ms, suy luận 8.3 ms, hậu xử lý NMS 1.9 ms.

Giá trị `gflops` trong toàn bộ `benchmark.json` là `null` — không có số FLOPs đo được trong artifact.

---

## 11. Chẩn đoán representation

Linear probe trên đặc trưng các tầng C2–C5, validation, 1280 px, AUPRC:

| Stage | v1/v2 (chỉ mẫu > 32 px, 51 pos) | v3 (4 bin, 61 pos / 153 neg) all | v3 0–8 px | v3 8–16 px | v3 16–32 px | v3 > 32 px |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| C2 | 0.8845 | 0.8702 | **0.4354** | 0.6861 | 0.8441 | 0.6837 |
| C3 | 0.9327 | 0.8795 | 0.5246 | 0.7298 | 0.8870 | 0.8199 |
| C4 | 0.9939 | 0.8838 | 0.5338 | 0.6462 | 0.8640 | 0.7444 |
| C5 | 0.9762 | **0.6461** | **0.2582** | 0.2285 | 0.4388 | 0.4891 |

Phiên bản v1/v2 chỉ lấy mẫu bin > 32 px (dataset audit cho thấy 12,018/12,019 instance thuộc bin này), nên không đo được bin nhỏ. Phiên bản v3 lấy 50 mẫu mỗi bin và cho kết quả: khả năng phân biệt ở bin 0–8 px rất thấp ngay từ tầng C2 (AUPRC 0.435) và suy giảm hoàn toàn ở C5 (0.258).

Độ ổn định dịch chuyển (v3): `shift_flip_rate_nonzero = 0.0800`; recall baseline dưới dịch chuyển tại conf 0.25 là 0.2704 (196 mẫu), tại conf 0.001 là 0.5357.

---

## 12. Provenance các run

| Experiment | Run ID | Commit | Status | Thời gian train | Peak VRAM |
| :--- | :--- | :--- | :--- | ---: | ---: |
| A0 baseline | 20261019-194809-8326468 | 8326468 | COMPLETED | 940 s | 5.47 GB |
| A1 no_sppf | 20261019-194809-8326468 | 8326468 | COMPLETED | 939 s | 5.39 GB |
| A2 p2 | 20261019-194809-8326468 | 8326468 | COMPLETED | 1,781 s | 10.15 GB |
| MSR mean | 20261020-142918-8efefb4 | 8efefb4 | COMPLETED | 2,853 s | 13.72 GB |
| MSR max | 20261020-142918-8efefb4 | 8efefb4 | COMPLETED | 2,668 s | 13.86 GB |
| MSR adaptive | 20261020-142917-8efefb4 | 8efefb4 | COMPLETED | 2,898 s | 13.91 GB |
| MSR official a0_identity | 20261019-142706-6a15122 | 6a15122 | TRAINED | 1,818 s | 12.21 GB |
| MSR official a1_local_max | 20261019-142801-6a15122 | 6a15122 | RUNNING (39 ep) | — | — |
| MSR official a2_full_max | 20261019-142828-6a15122 | 6a15122 | RUNNING (39 ep) | — | — |
| MSR official a3_full_adaptive | 20261019-142858-6a15122 | 6a15122 | RUNNING (44 ep) | — | — |

Toàn bộ run có `git_dirty = true`. Commit ID một mình không tái lập được code; cần kèm `config.yaml` và model YAML đã lưu trong từng thư mục run.

Các run QAC v1/v2/v3, DFL16, SADR8, BTRM không có thư mục artifact cục bộ — chỉ tồn tại trên Modal Volume `tiny-person-runs`. Số liệu của các run này được trích từ report và từ `reports/comparison_qac_v3_vs_dfl16/`, `reports/geometry/`, `outputs/metrics/`, `outputs/predictions/`.

---

## 13. Kết luận

**1. Nhánh P2 là đóng góp kiến trúc có hiệu quả lớn nhất.** A2 (P2 + SPPF) tăng AP +1.10 điểm và Recall +1.72 điểm so với A0, với chi phí 2.52 ms độ trễ. Đây là kết quả ổn định và có artifact đầy đủ.

**2. Phân phối hóa hồi quy hộp (SADR8) là cải tiến có hiệu quả đo được rõ nhất trên official test.** AP@0.50 +2.87 điểm, Recall +3.62 điểm, Miss giảm 495 đối tượng, Background FP giảm 6,151. Chi phí +8.0% tham số, độ trễ không tăng.

**3. Nút thắt của AP75 là xếp hạng độ tin cậy, không phải hồi quy hộp.** Trần Oracle cho thấy thay hộp bằng ground-truth không cải thiện AP75 (Δ ≈ 0), trong khi thay điểm bằng IoU nâng AP75 từ 2.37% lên 13.55%. Ba nguồn bằng chứng độc lập xác nhận: (i) trần Oracle, (ii) 63% FP là background chứ không phải hộp lệch, (iii) 41.05% hộp đạt IoU ≥ 0.75 có confidence < 0.25 và bị NMS loại.

**4. Hiệu chuẩn chất lượng (QAC) cải thiện biên nhưng không vượt baseline cùng dung lượng về mAP.** QAC v3 giảm 16.8% tổng FP, giảm 12.0% background FP, giảm 19.7% duplicate FP, tăng Recall +1.30 điểm và đạt AP@0.90 gấp 7 lần baseline. Tuy nhiên mAP50-95 thấp hơn baseline DFL16 0.34 điểm và Recall tại ngưỡng confidence cứng 0.25 thấp hơn ở cả 4 dải kích thước. QAC v3 hội tụ nhanh hơn rõ rệt ở giai đoạn đầu (epoch 5: mAP50 7.68% so với 1.52%).

**5. Mức tăng từ baseline gốc lên QAC v2 chủ yếu đến từ việc mở rộng head DFL, không phải từ module hiệu chuẩn.** Baseline DFL16 (`reg_max=16`, không QAC) đo được 7.38% mAP50-95, cao hơn QAC v2 (7.16%) và QAC v3 (7.04%). Việc chuyển `reg_max` từ 1 lên 16 đóng góp phần lớn nhất trong mức tăng 5.57% → 7.38%.

**6. Hai hướng tiếp cận feature-level không hiệu quả.** MSR (cả 3 chế độ fusion) giảm AP 0.36–0.89 điểm và tăng độ trễ 4.6–6.2 ms; chẩn đoán nội bộ cho thấy module duy trì gần identity (residual chỉ 2–3% RMS input). BTRM giảm AP@0.75 0.35–0.52 điểm. Cả hai đều không cải thiện phát hiện người tí hon.

**7. Khó khăn tập trung ở hai vùng.** Bin tiny1 (1–8 px) đạt recall 21.05% (miss 78.95%); nhóm ảnh `very_crowded` (81 ảnh, 51.5% tổng GT) đạt recall 40.07%. Chẩn đoán representation cho thấy đặc trưng ở bin 0–8 px có AUPRC chỉ 0.435 ngay từ tầng C2 và giảm còn 0.258 ở C5 — giới hạn nằm ở khả năng biểu diễn đặc trưng, không chỉ ở đầu dự đoán.

**8. Độ tin cậy của số liệu.** Mọi so sánh dùng một seed duy nhất (42). Evaluator official test là implementation nội bộ của repo, chưa đối chiếu với reference evaluator TinyBenchmark. Các run QAC/DFL16/SADR/BTRM không có artifact cục bộ. Số FLOPs không đo được. Các kết luận nêu trên có giá trị trong phạm vi protocol nhất quán của dự án.

---

## Phụ lục: Bảng tổng hợp toàn bộ kết quả official test

| Biến thể | AP@0.25 | AP@0.50 | AP@0.75 | AP mean | Recall | Precision |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 baseline | 0.4013 | 0.2301 | 0.0237 | 0.2184 | 0.5219 | 0.0506 |
| A2 adapt (warm-start) | 0.4026 | 0.2343 | **0.0247** | 0.2205 | 0.5266 | 0.0531 |
| A2 + BTRM scratch | 0.3768 | 0.2120 | 0.0202 | 0.2030 | 0.5112 | 0.0486 |
| A2 + BTRM adapt | 0.4027 | 0.2335 | 0.0195 | 0.2186 | 0.5267 | 0.0532 |
| A2 + BTRM γ=0 | 0.3798 | 0.2056 | 0.0209 | 0.2021 | 0.4976 | 0.0498 |
| A2 + MSR adapt | 0.3972 | 0.2323 | 0.0199 | — | — | — |
| **A2 + SADR8** | **0.4259** | **0.2589** | 0.0233 | **0.2360** | **0.5580** | **0.0566** |

AP@0.75 theo size bin (official test):

| Biến thể | all | tiny | tiny1 | tiny2 | tiny3 | small | reasonable |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 baseline | 0.0237 | 0.0174 | 0.00345 | 0.0116 | 0.0229 | 0.0329 | 0.0416 |
| A2 adapt | **0.0247** | **0.0181** | 0.00552 | 0.0179 | 0.0230 | 0.0337 | **0.0424** |
| A2 + BTRM | 0.0202 | 0.0102 | 0.00506 | 0.0157 | 0.0156 | 0.0240 | 0.0392 |
| A2 + BTRM adapt | 0.0195 | 0.0181 | **0.00585** | 0.0177 | 0.0230 | 0.0339 | 0.0369 |
| A2 + BTRM γ=0 | 0.0209 | 0.0093 | 0.00224 | 0.0138 | 0.0141 | 0.0263 | 0.0436 |
| A2 + MSR adapt | 0.0199 | 0.0184 | 0.00616 | 0.0179 | 0.0242 | 0.0269 | 0.0403 |
| A2 + SADR8 | 0.0233 | 0.0161 | 0.00579 | **0.0183** | **0.0234** | **0.0352** | 0.0456 |

---

*Báo cáo tổng hợp từ 199 file artifact (802 MB), 831k dòng log JSONL và 970 dòng report/docs trong repository. Mọi con số dẫn được về file nguồn trong `outputs/`, `reports/` hoặc `docs/`.*
