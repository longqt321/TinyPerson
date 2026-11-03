# QAR-NF v2: Quality-Aware Ranking for NMS-Free Detectors (sau phản biện)

> **Trạng thái**: Thiết kế nghiên cứu v2. Supersede `QAR_NF_PROPOSAL.md` (v1, giữ nguyên để đối chiếu). v2 **chấp nhận phần lớn phản biện** về khả năng cải thiện metrics, thêm Phase 0 (frozen-detector probe) và Phase A (đo trần ranking) trước khi implement, sửa công thức ranking loss (intra + inter + FG-BG + threshold-aware weighting), và thống nhất mục tiêu metrics.
> **Nguyên tắc**: Không tạo version mới về cơ chế — vẫn là SADR8 + QSH-NF + ranking loss + scale conditioning. Chỉ sửa **thiết kế, kỳ vọng và giao thức đo** theo phản biện.

---

## 0. Phản hồi phản biện (điểm nào chấp nhận / tinh chỉnh / bảo lưu)

| # | Điểm phản biện | Quyết định v2 | Lý do |
|---|---|---|---|
| 1 | QSH-NF có thể không dự đoán được IoU cho tiny (1-8px) | **Chấp nhận**: thêm ablation B1-B4 và đo correlation `I` vs IoU theo size bin | Phản biện đúng: DFL sharpness ≠ localization correctness; cần bằng chứng |
| 2 | AP75 chỉ tăng nếu có prediction đủ localization bị mis-rank | **Chấp nhận**: thêm Phase A (oracle quality score trên SADR8) để đo trần | Phản biện đúng: oracle hiện tại là A2, không phải SADR8 |
| 3 | Pairwise intra-GT không xử lý inter-object ranking | **Chấp nhận**: thêm inter-GT pairs vào loss | Thiếu sót thật: AP tính trên danh sách toàn ảnh |
| 4 | Trọng số `w_ij ∝ |IoU_i-IoU_j|` không ưu tiên cặp gần ngưỡng 0.75 | **Chấp nhận**: thêm threshold-aware weighting | Hợp lý cho AP75 |
| 5 | Candidate matching/duplicate chưa rõ | **Chấp nhận**: định nghĩa rõ candidate set và duplicate labels | Rủi ro thật |
| 6 | Scale temperature τ>1 làm mềm, không boost tiny; monotone → không đổi thứ tự trong cùng scale | **Chấp nhận tinh chỉnh**: scale conditioning là **optional gated component**, chỉ giữ nếu có bằng chứng calibration bias theo size | Phản biện toán học đúng |
| 7 | Recall bị giới hạn bởi candidate coverage (raw vs top-300 vs max_det=200) | **Chấp nhận**: thêm đo `R_raw`, `R_top300`, `R_top200` | Cần biết headroom trước khi kỳ vọng |
| 8 | FP ranking không giảm raw FP count, chỉ ảnh hưởng Top-K | **Chấp nhận**: phân biệt raw/top300/top200/conf=0.25 | Phản biện đúng |
| 9 | Tiny1 recall rủi ro cao (representation collapse không được xử lý) | **Chấp nhận tinh chỉnh**: tiny1 không phải kỳ vọng mặc định; chỉ giữ nếu Phase 0 dương | Phản biện đúng |
| 10 | Crowded recall không có cơ chế density; pairwise intra-GT không giải quyết inter-GT | **Chấp nhận**: crowded recall là metric phụ; không kỳ vọng tăng mạnh | Phản biện đúng |
| 11 | `J=C×I ≤ C` → tại threshold cố định recall không tăng từ re-ranking | **Chấp nhận phân tích**: tách lợi ích từ quality re-ranking vs ranking loss backprop vào cls | Phản biện toán học đúng |
| 12 | IoU + NWD dual target có thể xung đột | **Chấp nhận**: IoU-only trước (B1), NWD auxiliary sau (B3), gated | Phản biện đúng |
| 13 | DFL stats có thể backprop vào regression nếu không detach | **Chấp nhận**: hai thí nghiệm (detached vs not) | Phản biện đúng |
| 14 | Mục tiêu metrics không thống nhất (abstract mAP≥0.085 vs §5.3) | **Chấp nhận**: thống nhất theo §4 | Lỗi của v1 |
| 15 | Oracle 0.1355 đo trên A2, không phải SADR8 | **Chấp nhận**: không gọi "50% headroom"; Phase A đo trên SADR8 | Phản biện đúng |
| 16 | Evaluator path: default one-to-many+NMS; one-to-one chỉ với nms=False | **Chấp nhận**: verify path trước; đảm bảo QSH trên nhánh sẽ deploy | Phản biện đúng |
| 17 | Ablation thiếu Phase A (đo trần ranking trên SADR8) | **Chấp nhận**: thêm Phase A | Phản biện đúng |
| 18 | Frozen-detector probe trước khi implement đầy đủ | **Chấp nhận**: Phase 0 | Thí nghiệm 저비용, high-value |
| 19 | λ_q=0.25, λ_r=0.5, λ_bg=0.1 chưa chứng minh cân bằng | **Chấp nhận**: thêm logging gradient norm per loss | Phản biện đúng |
| 20 | QAR-NF-2 chưa chắc tốt hơn QAR-NF-0 | **Bảo lưu + gating**: nếu QAR-NF-1 không đánh bại QAR-NF-0, không tiến QAR-NF-2 | Đây là giả thuyết ablation, không phải kết luận |

**Tổng kết**: phản biện **không bác bỏ** QAR-NF, mà **hạ kỳ vọng** và **thêm các bước đo lường bắt buộc**. v2 giữ nguyên cơ chế nhưng làm rõ: (a) gain chính có khả năng ở AP75/mAP/precision; (b) recall/tiny1/crowded là secondary; (c) scale conditioning là optional; (d) ranking loss là giả thuyết cần kiểm chứng, không phải bước nâng cấp đương nhiên.

---

## 1. Bằng chứng nền (giữ nguyên từ v1, đã rà soát)

| Chỉ số | QAC v3 (ep50) | DFL16 (ep50) | DFL16 (ep100) |
|---|---:|---:|---:|
| mAP50-95 | 7.04% | 7.38% | 8.66% |
| AP@0.50 | 22.86% | 23.48% | 26.91% |
| AP@0.75 | 1.88% | 1.92% | 2.58% |
| Precision | 37.28% | 38.54% | 42.13% |
| Recall | 28.71% | 27.41% | 30.07% |

**Oracle AP75 (đo trên A2, không phải SADR8)**: score-only 0.0237 → 0.1355 (×5.7); box-only Δ≈0. Nút thắt là ranking.

**SADR8** (đã chứng minh): AP@0.50 +2.87, Recall +3.62, Miss −495, bg_FP −6,151 trên official test. Giữ làm nền regression (AP75 SADR8 = 0.0233 theo comparison).

**Error profile (A2, 84 ảnh val, 3,608 GT)**: detected=787, misses=1249, loc_err=1572, FP=817 (bg_FP=220, dup=155, loc_FP=442). Background FP 63% (84,359/133,941) trên toàn tập.

**Representation collapse (không phải nút thắt cho AP75)**: AUPRC 0-8px 0.435 (C2) → 0.258 (C5). Nhưng oracle box-only Δ≈0 → localization đã đủ tốt khi được chấm điểm đúng. Representation collapse **không** là binding constraint cho AP75.

**Tiny handling có sẵn trong YOLO**: `select_candidates_in_gts` phóng to GT < `stride_val`. QAR-NF không duplicate; phải đo xem QAR-NF thêm gì ngoài cái có sẵn.

**Evaluator**: `MAX_DETECTIONS = 200` (`src/tinydet/evaluation/tinyperson_benchmark.py:16`). One-to-one head xuất tối đa 300. Cần tách tác động Top-300 vs Top-200.

---

## 2. Phát biểu bài toán hình thức (giữ từ v1)

Gọi `D = {(x_k, y_k)}`. Mỗi ảnh sinh `N` candidate `{(b_i, c_i)}`. Detector chọn top-M theo `σ(c_i)`. Gọi `u_i = IoU(b_i, gt(i))`.

**Mục tiêu thực tế** (AP@τ): sắp xếp candidate sao cho candidate có `u_i ≥ τ` đứng trước. Đây là **ranking theo `u`**, không phải regression `u`.

**Gap**: `σ(c_i)` huấn luyện bởi BCE với target TAL `t_i = cls^α·u^β` (soft, không phải `u`). Thứ tự của `σ(c_i)` **không nhất thiết** trùng thứ tự của `u_i`. Oracle score-only chứng minh: thay `σ(c_i)` bằng `u_i` → AP75 ×5.7.

**Bài toán QAR-NF**: học `J_i = σ(c_i)·I_i` và objective `L` sao cho thứ tự của `J` xấp xỉ thứ tự của `u`, với `I` là quality score từ phân phối DFL + scale. `L` là pairwise surrogate của AP (RankNet + margin).

**Tiêu chí đo được**: (1) Spearman correlation `(J, u)` theo size bin; (2) ECE theo size bin; (3) AP@0.50/@0.75 khi xếp hạng theo J vs C.

---

## 3. Phương pháp: QAR-NF (v2 revised)

### 3.1. Kiến trúc (giữ)

```
Nhánh one-to-one (inference path, end2end):
  F_cls (64ch) ──► cls_proj ──► c_base (logit) ──► σ(c_base) = C
  F_box (64ch) ──► box_proj ──► b (DFL logits)
  g_dfl (24ch moments) ──┐
                          ▼
              ┌─────────────────────┐
              │  QSH-NF (MỚI)       │
              │  top-k sharpness    │
              │  + mean-of-top-k    │
              │  + scale embedding  │
              │  → MLP → I_raw      │
              │  → τ(scale) → I     │
              └─────────────────────┘
                          │
                          ▼
              J = C × I   ← điểm chọn lọc cuối (top-300 selection)
```

Khác biệt với QAC v3: QAC v3 `c_final = c_base + q_pred·v_sem` (in-place). QAR-NF `J = σ(c_base) × I` (decoupled, chỉ dùng cho selection, **không** chạm cls loss target).

### 3.2. QSH-NF: quality head decoupled + scale conditioning (optional)

**Input features** (theo GFLv2 DGQP, đã hiệu chỉnh):
- Top-k (k=5) probability per side (l,t,r,b): `4×5 = 20` channels.
- Mean-of-top-k per side: `4` channels.
- Scale embedding: sin/cos của `log(area)` từ `box_w, box_h`: `4` channels (continuous, không discrete bins).
- Tổng: `28` channels → MLP (28 → 16 → 1) + sigmoid → `I_raw`.

**Target** (v2: IoU primary, NWD auxiliary gated):
```
L_quality (Phase B1) = QFL(I_pred, IoU_gt)                    # IoU-only trước
L_quality (Phase B3) = QFL(I, IoU) + λ_nwd·BCE(I, NWD)        # NWD auxiliary sau, gated
```
- IoU là target chính (AP đo IoU; YOLO dùng CIoU → gradient tồn tại).
- NWD auxiliary = variance reduction (smoothing) cho tiny, **không** phải gradient. Chỉ thêm nếu B1 dương và B3 không làm xấu AP@0.75.

**Scale-conditioned temperature** (v2: optional, gated):
```
I = sigmoid( logit(I_raw) / τ(scale) )
τ(scale) = 1.0 + 0.5 · tanh( MLP_scale(scale_embed) )   # bounded [0.5, 1.5]
```
- Init τ=1.0 (identity). Residual bounded ±0.5.
- **Lưu ý toán học** (theo phản biện): τ>1 làm mềm distribution (score về gần 0.5), không boost tiny. Transform monotone → không đổi thứ tự trong cùng scale; chỉ thay đổi thứ tự cross-scale.
- **Quyết định**: scale conditioning chỉ giữ nếu Phase B4 cho thấy calibration bias theo size (ECE theo size bin khác nhau đáng kể). Nếu không, bỏ để giảm phức tạp.

**DFL stats detach** (v2: hai thí nghiệm):
- B2-detach: `I = f_θ(DFL_stats.detach(), scale)` — pure ranking, không ảnh hưởng regression.
- B2-grad: `I = f_θ(DFL_stats, scale)` — joint quality-localization learning.
- So sánh AP và regression metrics giữa hai. Nếu B2-grad cải thiện AP, phân tích gain đến từ ranking hay box regression.

### 3.3. Ranking-aware objective (v2 revised)

**Vấn đề v1**: chỉ intra-GT pairwise → không xử lý inter-object ranking (AP tính trên danh sách toàn ảnh).

**v2: ba thành phần ranking**:

**(a) Intra-GT pairwise** (giữ, nhưng threshold-aware weighting):
```
L_rank^intra = (1/|GT|) Σ_gt  Σ_{(i,j)∈C_gt×C_gt, u_i>u_j}  w_ij · softplus(-(J_i - J_j))
```
- `C_gt` = top-10 candidate cho GT đó (theo align metric).
- `w_ij` = threshold-aware: ưu tiên cặp mà một candidate đạt và candidate còn lại không đạt các ngưỡng IoU quan trọng (0.50, 0.75, 0.90):
```
w_ij = 1 + λ_thr · Σ_{τ∈{0.50,0.75,0.90}}  [ (u_i≥τ) XOR (u_j≥τ) ]
```
  với `λ_thr = 0.5`. Cặp "khó" (chênh lệch nhỏ quanh ngưỡng) nhận trọng số cao hơn cặp "dễ" (chênh lệch lớn).
- Chuẩn hóa `w_ij` theo cặp trong mỗi GT.

**(b) Inter-GT pairwise** (MỚI):
```
L_rank^inter = Σ_{(a,b): gt_a ≠ gt_b, u_a > u_b}  softplus( m_inter - (J_a - J_b) )
```
- Chỉ giữa **best candidate** của mỗi GT (top-1 theo align metric) và các GT khác trong cùng ảnh.
- `m_inter = 0.1` (margin nhỏ, vì inter-GT chênh lệch thường nhỏ).
- Mục đích: đảm bảo detection chất lượng tốt hơn giữa các GT được ưu tiên trong danh sách toàn ảnh.

**(c) Foreground-background ranking** (giữ, hard negative mining):
```
L_rank^bg = Σ_{i∈FG, j∈BG_top20}  softplus( m_bg - (J_i - J_j) ),   m_bg = 0.3
```
- Chỉ BG có cls score cao (top-20) → tập trung hard negatives.

**Tổng**:
```
L_rank = λ_intra·L_rank^intra + λ_inter·L_rank^inter + λ_bg·L_rank^bg
```
với `λ_intra=0.5, λ_inter=0.2, λ_bg=0.1` (init), tune trong ablation.

**Candidate matching & duplicate labels** (v2: định nghĩa rõ):
- Candidate set `C_gt` = top-10 theo align metric (TAL), bao gồm cả candidate đã được assign positive và candidate chưa assign.
- Duplicate handling: nếu ≥2 candidate của cùng GT có `u ≥ 0.50`, chỉ candidate tốt nhất được coi là "positive alternative" cho ranking; các candidate còn lại được coi là "duplicate FP" và bị đẩy xuống qua `L_rank^bg` (coi là BG hard negative).
- Candidate thuộc nhiều GT: theo TAL assignment (top-1 per GT), một candidate chỉ thuộc GT có align metric cao nhất; không duplicate across GTs trong candidate set.

**Tổng loss one-to-one branch**:
```
L_o2o = L_cls (BCE, target TAL) + L_box (CIoU) + L_dfl
      + λ_q·L_quality + λ_r·L_rank
```
với `λ_q=0.25, λ_r=0.5`.

**Gradient norm logging**: log `||∇L_cls||`, `||∇L_box||`, `||∇L_quality||`, `||∇L_rank||` mỗi step để kiểm tra cân bằng.

### 3.4. Dual-head handling (giữ)

- QSH-NF trên **cả** one-to-many và one-to-one, **không chia sẻ weight**.
- Quality target theo assignment (o2m: top-10; o2o: top-1).
- Eval đúng branch sẽ deploy.
- `fuse()` handling: QSH là sub-module của one2one head, giữ nguyên khi fuse.

### 3.5. Forward-compat (giữ)

reg_max=1 → `QAR-NF-FS` variant riêng với hypothesis/ablation riêng.

### 3.6. Tham số bổ sung

| Thành phần | Tham số |
|---|---:|
| QSH-NF MLP (28 → 16 → 1) | 481 |
| Scale embedding + τ MLP | ~120 |
| Per level × 2 branches | ~4,808 |
| **Tổng QAR-NF** | **~4.8k** (+8.3% vs baseline) |

---

## 4. Kỳ vọng metrics (v2: thống nhất và thực tế)

### 4.1. Bảng kỳ vọng (theo phản biện)

| Metric | SADR8 | Kỳ vọng QAR-NF | Độ tin cậy | Ghi chú |
|---|---:|---:|---|---|
| AP@0.75 | 0.0233 | ≥ 0.030 (thử nghiệm) / ≥0.035 (mục tiêu nghiên cứu) | Trung bình | Gain chính có khả năng nhất |
| mAP50-95 | 0.0866 (DFL16 ep100) / SADR8 chưa đo | ≥ 0.090 | Trung bình | Phụ thuộc ranking nhất quán trên nhiều ngưỡng |
| AP@0.50 | 0.2589 | 0.265–0.275 | Thấp–trung bình | Ranking theo IoU không nhất thiết giúp AP50 |
| Precision | 0.4213 (DFL16 ep100) | tăng nhẹ | Trung bình | FP trong Top-K có thể giảm |
| Recall tổng | 0.5580 (SADR8) | 0.56–0.57 | Thấp–trung bình | Bị giới hạn bởi candidate coverage |
| Tiny1 recall | 0.2392 | không kỳ vọng mặc định | Thấp | Representation collapse không được xử lý |
| Very-crowded recall | 0.4352 | metric phụ | Thấp | Không có cơ chế density |
| Background FP (raw) | 78,208 | không kỳ vọng giảm raw | — | Chỉ ảnh hưởng Top-K |

**Lưu ý**: mAP trong abstract v1 (≥0.085) và §5.3 (AP50≥0.275) không thống nhất → v2 dùng AP@0.50, AP@0.75, mAP50-95 riêng biệt, đo trên val TinyPerson. Oracle 0.1355 đo trên A2 → **không** gọi "50% headroom".

### 4.2. Tại sao recall khó tăng (phân tích phản biện)

Ranking head không tạo prediction mới. Recall tăng chỉ nếu:
- True detection bị đẩy ra khỏi Top-300 vì cls score thấp.
- Background FP chiếm vị trí trong Top-300.
- Quality score đưa true detection lên trên background FP.
- Candidate đó vẫn vượt conf threshold và max_det=200.

Cần đo `R_raw` (recall trên toàn candidate trước Top-300) vs `R_top300` vs `R_top200`. Nếu `R_raw ≈ R_top300`, ranking gần như không còn headroom recall.

**Lưu ý toán học** (theo phản biện): `J = C×I ≤ C` → tại threshold cố định, re-ranking không thể tạo thêm prediction vượt ngưỡng. Nhưng khi ranking loss backprop vào cls, cls thay đổi → quan hệ tập con không còn đúng tuyệt đối. Cần tách hai cơ chế.

### 4.3. Tại sao scale conditioning là optional (phân tích phản biện)

- τ>1 làm mềm distribution (score về gần 0.5), không boost tiny.
- Transform monotone → không đổi thứ tự trong cùng scale; chỉ thay đổi thứ tự cross-scale.
- Gain (nếu có) nhỏ. Chỉ giữ nếu Phase B4 cho thấy calibration bias theo size (ECE theo size bin khác nhau đáng kể).

---

## 5. Giao thức đo lường (v2: Phase 0 → A → B → C → D)

### Phase 0 — Frozen-detector probe (低비용, high-value)

**Mục đích**: kiểm tra xem QSH có cải thiện ranking ngay cả khi detector đóng băng.

**Cách làm**:
1. Lấy checkpoint SADR8 (đóng băng toàn bộ).
2. Trên tập train, extract DFL stats (top-k sharpness + mean + scale) cho mọi candidate.
3. Huấn luyện quality predictor nhỏ (MLP 28→16→1) với target IoU (và NWD auxiliary).
4. Trên val: so sánh xếp hạng theo C-only vs C×I: AP@0.50, AP@0.75, precision, recall@Top-200, FP/duplicate trong Top-200.

**Tiêu chí**: nếu AP@0.75 cải thiện ≥ +0.003 với p<0.05 → có bằng chứng ban đầu. Nếu không → dừng, phân tích nguyên nhân trước khi thêm pairwise/scale.

### Phase A — Đo trần ranking trên SADR8 (bắt buộc trước implement)

| Thí nghiệm | Câu hỏi |
|---|---|
| Original score | SADR8 đạt bao nhiêu AP@0.50/@0.75? |
| Oracle quality score (IoU thay score) | AP có thể tăng tới đâu? |
| Oracle trên toàn candidate set | Detector có đủ candidate trước Top-300 không? |
| Oracle sau Top-300 | Bao nhiêu headroom mất do selection? |
| Oracle sau max_det=200 | Giới hạn cuối của evaluator? |
| `R_raw` vs `R_top300` vs `R_top200` | Candidate coverage headroom cho recall? |

**Nếu oracle quality score không cải thiện AP@0.75 trên SADR8 → dừng QAR-NF.**

### Phase B — QSH pointwise (gated)

| Variant | Quality target | Mục đích |
|---|---|---|
| B0 | Không có QSH | Baseline SADR8 |
| B1 | IoU + QFL | Kiểm tra dự đoán IoU |
| B2-detach | IoU + QFL, DFL stats detach | Pure ranking, không ảnh hưởng regression |
| B2-grad | IoU + QFL, DFL stats có gradient | Joint quality-localization |
| B3 | B2 + NWD auxiliary | Kiểm tra NWD smoothing |
| B4 | B2 + scale temperature | Đo lợi ích scale calibration |

**Đo**: correlation `I` vs IoU theo size bin; AP@0.50/@0.75 khi xếp hạng theo C vs J; recall@Top-200; FP/duplicate. Nếu B2 không cải thiện AP → chưa có lý do thêm pairwise.

### Phase C — Pairwise ranking (gated)

| Variant | Objective |
|---|---|
| C0 | QSH pointwise |
| C1 | QSH + intra-GT pairwise (threshold-aware) |
| C2 | QSH + inter-GT pairwise |
| C3 | QSH + FG-BG ranking |
| C4 | QSH + cả ba |

**Đo**: gradient norm per loss; AP@0.50/@0.75; duplicate FP count; recall@Top-200.

### Phase D — Đánh giá đầy đủ (chỉ nếu C có gain)

- 3 seeds (42, 43, 44), paired bootstrap p<0.05.
- Official test + size bins + geometry + error analysis.
- Verify evaluator path: đảm bảo QSH trên nhánh sẽ deploy (one-to-one với `nms=False` nếu deploy NMS-free).

**Tiêu chí dừng** (invariant #7): AP@0.75 không tăng so với SADR8 sau 1 seed → dừng biến thể.

---

## 6. Rủi ro (v2, mở rộng theo phản biện)

| Rủi ro | Mức độ | Phương án |
|---|---|---|
| QSH không dự đoán được IoU cho tiny | Trung bình | Phase B correlation check; nếu fail → dừng |
| Không có prediction đủ localization bị mis-rank | Cao (nếu Phase A fail) | Phase A oracle trước |
| Pairwise intra-GT không xử lý inter-object | Trung bình | Thêm inter-GT pairs (§3.3b) |
| Threshold-aware weighting phức tạp | Thấp | Ablation C1 vs C4 |
| Duplicate FP tăng | Trung bình | Duplicate labels rõ ràng; theo dõi dup count |
| Pairwise xung đột one-to-one assignment | Trung bình | Định nghĩa candidate matching rõ ràng |
| DFL stats backprop vào regression | Trung bình | B2-detach vs B2-grad |
| IoU+NWD dual target xung đột | Trung bình | IoU-only trước (B1), NWD gated (B3) |
| Scale conditioning không giúp | Thấp | Optional, gated by B4 |
| Evaluator path mismatch (one-to-many vs one-to-one) | Cao (nếu không verify) | Verify trước Phase D |
| λ không cân bằng | Trung bình | Gradient norm logging |
| Tiny1/crowded recall không tăng | Cao (đã hạ kỳ vọng) | Secondary metrics, không phải tiêu chí thành công |

---

## 7. Implementation scope (v2)

- `src/tinydet/modules/qac_rank.py` (hoặc sửa `qac_v3.py`): QSH-NF head + ranking losses (intra/inter/FG-BG) + scale conditioning (gated).
- `configs/model/yolo26n_tiny_A2_p2_qac_rank.yaml`.
- `src/tinydet/models/registry.py`: `QAC_RANK_KEYS` + hook.
- Eval: thêm logging pre/post-NMS recall, raw/top300/top200 recall gap.
- Phase 0 script: frozen-detector quality predictor (low-cost).
- Ước tính: 4-5 ngày (Phase 0: 1 ngày, Phase A: 1 ngày, B-D: 2-3 ngày).

---

## 8. Tại sao đây là đóng góp khoa học (v2)

| Hướng đã thử | Kết quả | Tại sao QAR-NF khác |
|---|---|---|
| QAC v2/v3 (điều biến logit in-place) | mAP thua baseline | QAR-NF **decouple** + ranking objective |
| QSH v4 (pointwise QFL) | Chưa chạy | QAR-NF thêm **pairwise ranking** (intra+inter+FG-BG) |
| GFLv2 (NMS-based) | COCO AP +2.6 | QAR-NF thích ứng **NMS-free** + scale conditioning |
| Ranking losses (AP/aLRP/RS) | AP cải thiện | QAR-NF kết hợp **quality head + scale conditioning** trên YOLO26 |
| SADR8 (regression) | Recall +3.62 | QAR-NF **giữ** SADR, thêm ranking |

**Khác biệt then chốt**: QAR-NF tấn công trực tiếp vào **score dùng để chọn lọc trong chế độ NMS-free** (oracle lever), với **objective là ranking** (không phải pointwise regression), và **scale conditioning optional**. Đây là hướng mà không có biến thể nào trong repo hay văn liệu đã thử trên YOLO26 NMS-free — nhưng v2 thừa nhận gain chính có khả năng ở AP75/mAP/precision, không phải recall/tiny1/crowded.

---

## 9. Kết luận (v2)

QAR-NF v2 **không thay đổi cơ chế** so với v1, mà **hạ kỳ vọng và thêm các bước đo lường bắt buộc** theo phản biện. Ba thay đổi chính:

1. **Phase 0 (frozen-detector probe) + Phase A (đo trần ranking)** trước khi implement — tránh đầu tư vào cơ chế khi không có headroom.
2. **Ranking loss revised**: intra-GT (threshold-aware) + inter-GT + FG-BG, với duplicate labels rõ ràng.
3. **Scale conditioning optional**, IoU primary + NWD auxiliary gated, detached vs non-detached experiment.

**Phán quyết** (theo phản biện, đã đồng ý): QAR-NF đáng đầu tư một vòng thực nghiệm nghiêm túc, với kỳ vọng gain có ý nghĩa nhất ở AP75, sau đó mAP50-95 và precision. Không kỳ vọng tăng lớn ở AP50, recall, tiny1 và crowded recall. QSH pointwise là thành phần có khả năng đóng góp chính; pairwise ranking là giả thuyết cần kiểm chứng; scale conditioning là optional.

**Bước tiếp theo**: (1) Phase 0 frozen-detector probe; (2) Phase A ranking ceiling; (3) chỉ nếu dương, Phase B QSH pointwise; (4) chỉ nếu B dương, Phase C pairwise; (5) chỉ nếu C dương, Phase D đầy đủ.
