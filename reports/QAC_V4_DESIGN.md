# QAC v4 — Thiết kế v2 (sau review kỹ thuật)

> **Trạng thái**: Thiết kế nghiên cứu v2. Supersede `QAC_V4_PROPOSAL.md` (v1) nhưng **v1 vẫn giữ nguyên** trong repo để đối chiếu. Mọi điểm bị review đánh dấu trong §0.
> **Nguyên tắc**: Không tạo version mới về cơ chế — vẫn là QAC v3 + ba đóng góp additive (QSH, FGS, DAR) + SADR8. Chỉ sửa **thiết kế** của ba đóng góp theo bằng chứng và review.
> **Thứ tự ưu tiên triển khai**: QSH (đầu tiên) → FGS (sau frequency attribution) → DAR (sau khi đo NMS-suppression).

---

## 0. Changelog v1 → v2 (theo review)

| # | Điểm review | Quyết định v2 |
|---|---|---|
| 3.1 | QSH decoupling khỏi cls logit | **Giữ nguyên** — đây là điểm đúng nhất của v4 |
| 3.2 | Top-k DFL nên thêm mean-of-top-k (DGQP-style) | **Chấp nhận**: descriptor = top-k per side + mean-of-top-k. Chưa thêm entropy/variance/skewness |
| 3.3 | NWD thuần túy ≠ IoU quality; AP75 đo IoU | **Chấp nhận**: Direction A — IoU là target chính (QFL), NWD làm auxiliary loss cho tiny |
| 3.4 | Dual-head one2many/one2one cần QSH riêng | **Chấp nhận bắt buộc**: QSH riêng cho mỗi branch, eval đúng branch sẽ deploy |
| 3.5 | Non-DFL "thay thế tương đương" là không đúng | **Chấp nhận**: coi là variant riêng với hypothesis/ablation riêng |
| 4.1 | Giả định low-freq=bg, high-freq=person chưa có bằng chứng | **Chấp nhận**: làm frequency attribution experiment trước khi thiết kế FGS |
| 4.2 | Công thức `F + γ(F⊙G)` chỉ khuếch đại, không suppress | **Chấp nhận**: sửa thành attenuating gate hoặc signed residual, identity chính xác tại epoch 0 |
| 4.3 | Magnitude-only FFT → IFFT là sai toán học (mất phase) | **Chấp nhận**: giữ complex spectrum để reconstruct; magnitude-only chỉ dùng như descriptor branch |
| 4.4 | Diagram P2-only nhưng param table 4 levels — không nhất quán | **Chấp nhận**: P2-only ban đầu |
| 5.1 | Chưa chứng minh NMS là nguyên nhân crowded miss | **Chấp nhận**: đo 3 metric pre/post-NMS trước |
| 5.2 | Giảm NMS IoU 0.50→0.45 làm suppress *mạnh hơn* | **Chấp nhận**: sửa chiều — crowded cần threshold cao hơn hoặc Soft-NMS |
| 5.3 | Soft-NMS ngưỡng 0.001 + density 100 chưa có bằng chứng | **Chấp nhận**: cap max pred, theo dõi FP/image; hiệu chuẩn lại density sau khi QSH xong |
| 5.4 | One-to-many vs one-to-one inference path phải thống nhất | **Chấp nhận**: Option 1 (one-to-many + custom NMS) cho ablation; Option 2 (NMS-free) cho deploy |

---

## 1. Bằng chứng nền (giữ nguyên từ v1)

| Chỉ số | QAC v3 (ep50) | DFL16 (ep50) | Δ |
|---|---:|---:|---:|
| mAP50-95 | 7.04% | 7.38% | −0.34% |
| Precision | 37.28% | 38.54% | −1.26% |
| Recall | 28.71% | 27.41% | +1.30% |
| FP tổng | 817 | 982 | −16.8% |
| AP@0.90 | 0.071% | 0.010% | 7× |

**Oracle AP75** (đo trong repo): score-only oracle nâng AP75 từ 0.0237 → 0.1355 (×5.7); box-only oracle Δ≈0. **Nút thắt là ranking, không phải regression.** Đây là cơ sở duy nhất cho QSH.

**SADR8** (đã chứng minh): reg_max=8 → AP@0.50 +2.87, Recall +3.62, Miss −495, bg_FP −6,151. Giữ làm nền regression.

**Error profile (A2, 84 ảnh val, 3,608 GT)**: `detected=787, misses=1249, localization_errors=1572, FP=817` trong đó `background_fp=220, duplicate=155, localization_fp=442`. Background FP chiếm 26.9% FP trong val nhỏ này; trên toàn tập A2 là 63% (84,359 / 133,941). Đây là cơ sở cho FGS — nhưng **chưa** là bằng chứng rằng frequency-domain phân biệt được chúng.

### 1.1. Sửa sai: YOLO dùng CIoU, không phải IoU thuần túy

Một tiền đề tôi dùng trước đây ("IoU = 0 khi không overlap → không có gradient") là **sai** cho YOLO:

- **Bbox regression loss** (`BboxLoss`, `ultralytics/utils/loss.py:134`) dùng `bbox_iou(..., CIoU=True)` → có center-distance penalty, gradient tồn tại ngay cả khi không overlap.
- **Assigner** (`TaskAlignedAssigner.iou_calculation`, `ultralytics/utils/tal.py:246`) cũng dùng `bbox_iou(..., CIoU=True)`, clamp ≥ 0.

**Hệ quả:**

1. **NWD không "giải quyết vấn đề gradient"** — vì CIoU đã giải quyết. Mọi biện minh NWD dựa trên "gradient continuity khi IoU=0" đều không hợp lệ.
2. **QAC v3 patch NWD vào assigner** (`_patch_assigner_with_nwd`) yếu hơn tuyên bố, vì CIoU base đã xử lý non-overlap. Đây có thể là một phần lý do QAC v3 không cải thiện mAP — đáng kiểm tra bằng ablation "bỏ NWD assigner patch".
3. **QSH target**: IoU vẫn là target đúng (vì AP evaluator đo IoU), và **không** có vấn đề gradient. NWD auxiliary được giữ lại với lý do đúng: **làm mượt target cho tiny object** (IoU dốc → target high-variance; NWD mượt hơn → ổn định hơn), tức bài toán variance/label smoothing, không phải gradient.

---

## 2. QSH — Quality Score Head (triển khai đầu tiên)

### 2.1. Ý tưởng (giữ từ v1, theo GFLv2 DGQP)

```
DFL logits (B, 4*reg_max, H, W)
        │
        ▼
  top-k per side (k=5) + mean-of-top-k   ← v2: thêm mean-of-top-k
        │
        ▼
  4*k + 1 = 21 channels (với k=5)
        │
        ▼
  MLP(21 → 16 → 1) + Sigmoid  →  I ∈ (0,1)   [predicted IoU quality]
        │
        ▼
  Ranking score J = σ(cls_logit) × I
```

**Vì sao thêm mean-of-top-k**: GFLv2 DGQP dùng top-k cùng thống kê mean của các xác suất được chọn để phản ánh độ sắc của phân phối. Top-k đơn thuần cho biết "đỉnh ở đâu", mean-of-top-k cho biết "đỉnh sắc đến mức nào". Không thêm entropy/variance/skewness ở vòng này — kiểm chứng DGQP-style statistics trước (invariant: một thay đổi một lúc).

**reg_max=8 kiểm tra ổn định**: với 8 bins, top-5 vẫn lấy được nhưng descriptor ít ổn định hơn reg_max=16. Cần ablation riêng: reg_max=8 vs reg_max=16 cho QSH, đo AP75 và calibration của `I` (Pearson corr với IoU thực).

### 2.2. Target — Direction A (IoU là target chính)

```
L_quality = QFL(I_pred, IoU_gt)          # primary
          + λ_nwd · BCE(I_pred, NWD)     # auxiliary, λ_nwd = 0.3
```

- **Lý do**: AP75 đo IoU. Nếu QSH học NWD thuần túy, nó có thể xếp hạng tốt theo NWD nhưng không tốt theo IoU threshold mà evaluator dùng. IoU là target trả lời trực tiếp "quality head có dự đoán được localization quality mà AP evaluator quan tâm không?"
- **Sửa sai thêm (theo review)**: biện minh cũ "NWD cung cấp gradient liên tục khi IoU=0" là **sai** — YOLO dùng CIoU ở cả bbox loss lẫn assigner, gradient đã tồn tại khi không overlap (xem §1.1). NWD auxiliary được giữ lại với lý do đúng: **làm mượt target cho tiny object** (IoU dốc → target high-variance cho 0–8px; NWD mượt hơn). Đây là bài toán variance, không phải gradient.
- **Hệ quả cho QAC v3**: nếu NWD assigner patch là gần-redundant với CIoU, cần ablation "bỏ patch" để xác nhận trước khi giữ NWD auxiliary ở QSH.
- QFL (Quality Focal Loss) với target `IoU ∈ [0,1]` — theo GFLv2.

**Hybrid (Direction B) là fallback**: nếu IoU supervision cho tiny có target variance quá cao (đo bằng calibration/corr của `I` với IoU thực trên bin 0–8px), thử `I_target = α(A)·IoU + (1−α(A))·NWD` với `α` phụ thuộc kích thước GT (nhỏ → NWD nặng hơn). Chỉ sau khi Direction A thất bại về variance, không phải về gradient.

### 2.3. Dual-head — bắt buộc, không phải chi tiết

YOLO26 có one-to-many (dùng NMS mặc định) và one-to-one (NMS-free, ≤300 pred). QAC v3 đã phải có calibrator riêng cho hai branch (`one2one_quality_gates`, `one2one_calib_scale`, …). QSH phải làm tương tự:

- `QSH_o2m` và `QSH_o2o`, **không chia sẻ weight** ở vòng đầu (có thể thử chia sẻ sau nếu có lý do thực nghiệm).
- Mỗi branch nhận quality target theo assignment tương ứng (o2m: top-10 candidates; o2o: top-1).
- **Validation phải đánh giá đúng branch sẽ sử dụng khi triển khai.** Nếu deploy one-to-one NMS-free thì metric QSH phải đo trên o2o branch.

### 2.4. Forward-compat với YOLO26 (reg_max=1)

**Không** coi feature-stats (`mean/std/max` của `F_cls`) là thay thế tương đương cho DFL top-k. DFL top-k mang ngữ nghĩa geometric quality của phân phối regression; feature-stats không. Vì vậy:

- `quality_extractor` là một interface với hai implementation: `DFLTopKExtractor` (reg_max>1) và `FeatureStatsExtractor` (reg_max=1).
- Implementation thứ hai là **một variant riêng** với hypothesis và ablation riêng, không phải "drop-in replacement". Đặt tên `QSH-FS` và đánh giá độc lập.

### 2.5. Tham số

| Thành phần | Tham số |
|---|---:|
| Top-k extractor | 0 (deterministic) |
| MLP (21 → 16 → 1) | 21×16 + 16 + 16×1 + 1 = 369 |
| Per-level (4 levels) × 2 branches | 2 × 4 × 369 ≈ 2,952 |
| **Tổng QSH (v2)** | **~3.0k** (so với 2.064k của QAC v3) |

---

## 3. FGS — Frequency-Gated Suppression (chỉ sau frequency attribution)

### 3.1. Gate trước: frequency attribution experiment (bắt buộc trước khi thiết kế)

Lấy feature map P2 của SADR8. So sánh các vùng:

1. Person TP
2. Background FP
3. Localization FP
4. Person bị miss

Thử mask low-, mid-, high-frequency band **riêng biệt**, đo biến động score và recall/AP. Chỉ khi có bằng chứng một dải tần chứa tín hiệu phân biệt ổn định giữa TP và background FP mới thiết kế learnable radial mask dựa trên kết quả đó.

**Lý do**: SET (CVPR 2025) quan sát tiny object bị ảnh hưởng bởi high-frequency noise, và loại bỏ một phần high-frequency có thể *cải thiện* detection. High-frequency không mặc nhiên là person; low-frequency không mặc nhiên là background. Person tiny sau feature encoding có thể mất nhiều đặc trưng high-frequency; background clutter có thể chứa rất nhiều high-frequency components. Giả định cứng của v1 (low-freq=bg, high-freq=person) là không có bằng chứng.

### 3.2. Sửa công thức gate (lỗi logic v1)

v1: `F_out = F + γ(F⊙G_freq)` với `G∈(0,1)`, `γ>0` → chỉ có thể **khuếch đại**, không suppress. Module mang tên "Suppression" nhưng công thức chỉ cho phép cộng feature cùng dấu.

**v2 — hai lựa chọn, chọn một trong ablation**:

- **Attenuating gate**: `F_out = F ⊙ (1 − γ·G_freq)`, `γ ∈ [0,1]`. Trực tiếp suppress.
- **Signed residual modulation**: `F_out = F + γ·F⊙tanh(G_freq)`. Cho phép cả tăng và giảm.

Cả hai: khởi tạo sao cho `F_out = F` **chính xác** tại epoch 0 (không chỉ "gần identity" như γ=0.001). γ init = 0, G init = 0 (hoặc mask init = 0).

### 3.3. Phase handling (lỗi toán học v1)

v1: `FFT → |spectrum| → mask → IFFT` — sai. FFT tạo phổ phức `X = |X|e^{jφ}`; magnitude chỉ chứa biên độ, phase chứa thông tin vị trí/cấu trúc không gian. Không thể khôi phục feature map bằng IFFT nếu chỉ giữ magnitude.

**v2**:

- Để reconstruct feature map: giữ complex spectrum:
  `F_freq = IFFT(M ⊙ FFT(F))` với `M` complex mask (hoặc real mask nhân lên complex spectrum, giữ phase). Dùng `rfft2`/`irfft2`.
- Nếu muốn dùng magnitude làm descriptor bất biến dịch chuyển: dùng làm **đầu vào của một nhánh dự đoán gate riêng**, không dùng magnitude đơn độc để reconstruct.

### 3.4. Scope: P2-only ban đầu

Diagram v1 đặt FGS chỉ trên P2 nhưng bảng tham số tính 4 levels — không nhất quán. v2: **P2-only** ban đầu.

Lý do: tiny person cần feature resolution cao; P2 là nơi phù hợp kiểm tra giả thuyết tần số; dễ đo tác động; giảm chi phí FFT; tránh làm nhiễu feature level phục vụ object lớn hơn. Mở rộng P3/P4 chỉ nếu P2-only có lợi và các level khác thực sự cần điều chỉnh (ablation riêng).

### 3.5. Tham số (P2-only)

| Thành phần | Tham số |
|---|---:|
| Radial mask (P2, ~H/2 radial bins) | ~160 |
| Conv 1×1 (C → 1) | C + 1 |
| **Tổng FGS** | **~200** (P2-only) |

---

## 4. DAR — Density-Adaptive Ranking (chỉ sau khi đo NMS-suppression)

### 4.1. Ba metric bắt buộc trước khi làm gì

| Metric | Ý nghĩa |
|---|---|
| Pre-NMS recall | Detector có tìm ra person trước suppression không? |
| Post-NMS recall | Còn bao nhiêu person sau suppression? |
| NMS-suppressed GT matches | Bao nhiêu GT có candidate phù hợp nhưng bị NMS loại bỏ? |

**Nếu pre-NMS recall đã thấp, Soft-NMS không giải quyết được phần lớn miss.** DAR chỉ trực tiếp giúp một phần trong số: candidate không được sinh, candidate confidence thấp, NMS suppress box đúng, top-k pre-filter loại candidate, localization không đạt IoU threshold.

Hiện tại eval pipeline chỉ lưu post-NMS (`detected/misses/localization_errors` trong `error_summary.csv`). Cần thêm logging pre-NMS. Đây là công việc nhỏ nhưng phải làm trước.

### 4.2. Sửa chiều NMS IoU threshold

v1 nói giảm threshold 0.50 → 0.45 để "giữ nhiều hơn trong crowds". **Sai chiều**: với hard NMS, IoU vượt threshold → suppress; threshold thấp hơn làm điều kiện suppress dễ đạt hơn → suppress *mạnh hơn*. Trong ảnh crowded, person gần nhau có box overlap cao. Để giữ nhiều prediction hơn cần **tăng** threshold (hoặc Soft-NMS), không giảm.

### 4.3. Soft-NMS — đánh giá riêng, có giới hạn

Gaussian penalty `s_i' = s_i·exp(−IoU_i²/σ²)` với σ=0.5 có thể giữ một số detection overlap cao thay vì xóa hoàn toàn. Nhưng:

- Có thể giữ duplicate detections hoặc background FP mà hard NMS đã loại.
- Cần **cap số prediction tối đa** và theo dõi FP/image.
- Ngưỡng score ≥ 0.001 không mặc định an toàn — theo dõi.

**Density estimator**: v1 dùng `N_candidate(score>0.05) > 100`. Ngưỡng 100 chưa có bằng chứng. Khi chuyển sang `J = C·I`, phân phối score thay đổi → số candidate vượt 0.05 thay đổi theo. **Hiệu chuẩn lại density estimator sau khi QSH xong.**

### 4.4. Thống nhất inference path với YOLO26

- **Option 1 (nghiên cứu)**: one-to-many + custom Soft-NMS/density-adaptive NMS. Dễ đo pre-NMS/post-NMS recall và tác động suppression. **Chọn cho ablation ban đầu.**
- **Option 2 (deploy)**: one-to-one NMS-free (nms=False, ≤300 pred). Giữ native end-to-end detection; cần cơ chế ranking/candidate selection phù hợp với one-to-one output, không ghép Soft-NMS máy móc.

Phải xác định rõ v4 chạy theo hướng nào trước khi viết code. Không trộn hai hướng trong cùng một ablation.

---

## 5. SADR8 — đóng góp thứ 4 (kế thừa, không đổi)

reg_max=8 (không phải 16, không phải 1). Config: `reg_max: 8`. DFL clamp target `[0, 6.99]` trên mỗi level; reg_max=16 bóp cụt nhánh DFL ở P2/P3/P4/P5 cho box rộng. reg_max=8 cân bằng coverage và chi phí (+8.0% params).

---

## 6. Kiến trúc tổng hợp (v2)

```
Input (1280×1280)
   │
   ▼
Backbone YOLO26n + P2 (stride 4) + SPPF
   │
   ├──► P2 (stride 4) ──► [FGS: P2-only, sau attribution] ──► P2'
   │                                                              │
   │                                                              ▼
   │                                                          Detect (SADR8, reg_max=8)
   │                                                              │
   │                                                              ├──► cls head (normal training, unchanged)
   │                                                              ├──► reg head (DFL, reg_max=8)
   │                                                              └──► QSH_o2m / QSH_o2o  ← MỚI (decoupled)
   │                                                                   │
   │                                                                   ▼
   │                                                              I ∈ (0,1) per detection
   │                                                                   │
   │                                                                   ▼
   │                                                              Ranking score J = σ(cls) × I
   │                                                                   │
   │                                                                   ▼
   │                                                              [DAR: Option 1, sau khi đo NMS]
   │                                                                   │
   └──────────────────────────────────────────────────────────────────▼
                                                               Final detections
```

**Tham số bổ sung (so với baseline A2, 2,515,976)**:

| Module | Tham số | Ghi chú |
|---|---:|---|
| SADR8 (reg_max 8 vs 1) | +201,960 | Đã đo |
| FGS (P2-only) | ~200 | Sau attribution |
| QSH (2 branches × 4 levels) | ~2,952 | MLP 21→16→1 |
| **Tổng v4** | **~2,721,000** | +8.2% |

---

## 7. Ablation (v2, có thứ tự gating)

### 7.1. Biến thể

| Biến thể | SADR | FGS | QSH | DAR | Mục đích |
|---|---:|---:|---:|---:|---|
| **QSH-only** | ✗ | ✗ | ✓ (o2m) | ✗ | Đo đóng góp QSH đơn lẻ vs baseline (oracle lever) |
| QSH + SADR8 | ✓ | ✗ | ✓ | ✗ | QSH trên nền regression đã chứng minh |
| + FGS (P2) | ✓ | ✓ | ✓ | ✗ | Sau khi attribution pass |
| + DAR (Option 1) | ✓ | ✓ | ✓ | ✓ | Sau khi đo NMS-suppression |
| v4-full | ✓ | ✓ | ✓ (o2m+o2o) | ✓ | Proposed |
| SADR8 (existing) | ✓ | ✗ | ✗ | ✗ | Baseline cải tiến hiện tại |

**Gating rules** (theo invariant #7 của repo):
- Nếu AP@0.75 của QSH-only không tăng so với SADR8 sau 1 seed → dừng biến thể đó, phân tích nguyên nhân, thử Direction B (hybrid target) trước khi thêm FGS/DAR.
- FGS chỉ được thêm nếu frequency attribution cho thấy ≥1 band có tín hiệu phân biệt ổn định (TP vs bg FP) với Δ AP ≥ +0.3.
- DAR chỉ được thêm nếu `NMS-suppressed GT matches` ≥ 15% tổng miss.

### 7.2. Chỉ tiêu mục tiêu (so với SADR8 hiện tại)

| Chỉ số | SADR8 | Mục tiêu v4-full |
|---|---:|---:|
| AP@0.50 | 0.2589 | ≥ 0.275 |
| AP@0.75 | 0.0233 | ≥ 0.035 (≥50% headroom oracle) |
| Recall tổng | 0.5580 | ≥ 0.58 |
| Recall very_crowded | 0.4352 | ≥ 0.50 |
| Background FP | 78,208 | ≤ 65,000 (−17%) |
| tiny1 (0–8px) Recall | 0.2392 | ≥ 0.28 |

### 7.3. Giao thức

- Dataset: `/mnt/data/tinyperson_official_v1/data.yaml`, imgsz 1280, seed 42 (rồi 43, 44 nếu dương).
- Huấn luyện: 50 epoch chặng 1 (lr 0.001), +50 epoch nối tiếp (lr 0.0006, cosine) nếu chặng 1 dương.
- Đánh giá: validation + official test + phân tích hình học + phân tích lỗi, giữ nguyên protocol + 3 metric pre/post-NMS mới.
- **Tiêu chí dừng**: AP@0.75 không tăng so với SADR8 sau 1 seed → dừng biến thể đó.

---

## 8. Tại sao đây là hướng đột phá (v2)

| Hướng đã thử | Kết quả | Tại sao v4 khác |
|---|---|---|
| QAC v2/v3 (điều biến logit in-place) | mAP thua baseline | v4 **decouple**: cls head độc lập, quality head riêng cho ranking |
| MSR (feature normalization) | AP giảm, latency +6ms | v4 **không** normalize feature; chỉ gate frequency, residual nhỏ |
| BTRM (contrast enhancement) | AP@0.75 giảm | v4 **không** thay đổi feature; chỉ thêm score head |
| SADR8 (regression) | AP@0.50 +2.87 | v4 **giữ** SADR, thêm ranking + bg suppression |
| QAC v3 (Spatial-Relational Gate 1×1) | FP giảm nhưng mAP giảm | v4 **frequency-domain** gate, nhìn toàn cục spectrum (nếu attribution pass) |

**Khác biệt then chốt**: v4 tấn công trực tiếp vào **score dùng cho ranking** (oracle đã chứng minh đây là lever), thay vì điều biến logit (QAC) hay feature (MSR/BTRM). Đây là hướng mà không có biến thể nào trong repo đã thử.

---

## 9. Rủi ro và phương án dự phòng (v2)

| Rủi ro | Mức độ | Phương án |
|---|---|---|
| QSH target (IoU) không đủ ổn định cho tiny | Trung bình | Direction B hybrid target; hoặc GFLv2 gốc (top-k sharpness) |
| QSH descriptor không ổn định với reg_max=8 | Trung bình | Ablation reg_max=8 vs 16 cho QSH; đo calibration của I |
| FGS làm mất edge person (high-pass filter) | Trung bình | Attribution trước; mask init = identity; monitor gate_mean |
| FGS reconstruction sai phase | Cao (nếu không sửa) | Giữ complex spectrum (§3.3) |
| DAR tăng FP trong dense | Thấp–Trung bình | Cap max pred; theo dõi FP/image; Option 1 cho ablation |
| NMS threshold chiều sai | Cao (nếu không sửa) | Đã sửa §4.2 |
| Latency vượt ngưỡng real-time | Thấp | FGS rfft2 + radial mask; QSH MLP 21→16→1 |
| Không tương thích YOLO26 (không DFL) | Thấp | QSH-FS là variant riêng, ablation độc lập |
| Dual-head không được eval đúng branch | Cao (nếu không sửa) | QSH riêng + eval branch sẽ deploy (§2.3) |

---

## 10. Nguồn văn liệu tham khảo

1. **GFLv2** (Li et al., NeurIPS 2020) — Distribution-Guided Quality Predictor, top-k DFL sharpness + mean → IoU quality → ranking score.
2. **NWD** (Wang et al., 2021) — Normalized Wasserstein Distance for tiny object detection.
3. **SET** (CVPR 2025) — Spectral Enhancement for Tiny Object Detection; high-freq noise có thể hại tiny object.
4. **FSDETR / FD-YOLO** (2025) — Frequency-spatial feature pyramid, learnable filter mask.
5. **ScaleBridge-Det** (arXiv 2025) — Density-guided query allocation.
6. **DQ-DETR** (2024) — Dynamic query cho tiny object.
7. **SO-RTDETR** (2025) — NWD loss + P2 enhancement.
8. **YOLO26** (Ultralytics, 2025) — STAL, ProgLoss, loại bỏ DFL, end-to-end NMS-free, dual-head one2many/one2one.
9. **Soft-NMS** (Bodla et al., 2017) và **WBF** (Solovyev et al., 2021) — post-processing cho dense crowd.
10. **Uncertainty-Aware Gradient Stabilization** (2025) — interval classification cho tiny objects.
11. **Oracle AP75 analysis** (repo này) — score-only oracle nâng AP75 ×5.7.

---

## 11. Kết luận

v2 giữ nguyên ba đóng góp của v1 (QSH, FGS, DAR) + SADR8, nhưng **sửa thiết kế** theo review: QSH decoupled với IoU primary target + mean-of-top-k + dual-head; FGS gate đúng toán học (complex spectrum, signed/attenuating, P2-only) sau attribution; DAR có bằng chứng (3 metric pre/post-NMS) và sửa chiều threshold. QSH là thành phần triển khai đầu tiên; FGS và DAR bị gated bởi bằng chứng. Forward-compat với YOLO26 (non-DFL) là variant riêng, không phải drop-in.

**Bước tiếp theo**: (1) implement QSH-only trên o2m branch; (2) thêm logging pre-NMS vào eval pipeline; (3) chạy frequency attribution trên P2 features của SADR8. Sau đó mới quyết định thêm FGS/DAR.
