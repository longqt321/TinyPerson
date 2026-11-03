# QAR-NF: Quality-Aware Ranking for NMS-Free Detectors

> **Báo cáo khoa học — đề xuất cải tiến đơn lẻ cho YOLO26n (TinyPerson)**
> Phiên bản: 1.0 · Ngày: 2026-11-01 · Trạng thái: đề xuất nghiên cứu (chưa có code/experiment)
> Phạm vi: giữ nguyên backbone YOLO26n + P2 của repo; can thiệp vào **điểm số dùng để chọn lọc** của nhánh one-to-one (nhánh inference trong chế độ end-to-end NMS-free).

---

## 0. Tóm tắt (Abstract)

Bằng chứng thực nghiệm của repo cho thấy **nút thắt của AP@0.75 là thứ hạng điểm số (ranking), không phải độ chính xác hộp** (oracle score-only nâng AP75 từ 0.0237 → 0.1355, ×5.7; oracle box-only Δ≈0). Tuy nhiên, mọi cơ chế hiện có (QAC v2/v3) đều **điều biến logit phân loại in-place**, conflating phân loại với chất lượng localization, và đều nhắm vào nhánh one-to-many (dùng NMS). Trong khi đó, YOLO26 trong repo chạy ở chế độ **end-to-end NMS-free**: inference dùng nhánh one-to-one, chọn top-300 theo **điểm thô cls**, không có NMS để "sửa" thứ hạng sau đó.

**Đóng góp đề xuất — QAR-NF (Quality-Aware Ranking for NMS-Free detectors):** một cơ chế đơn lẻ, gồm ba thành phần ghép chặt, tấn công trực tiếp vào **điểm số dùng để chọn lọc** của nhánh one-to-one, với mục tiêu huấn luyện là **ranking** (không phải regression điểm chất lượng pointwise), và **hiệu chuẩn theo scale** để không over-specialize vào tiny.

Ba thành phần:
1. **QSH-NF**: quality head decoupled trên nhánh one-to-one, dự đoán chất lượng localization `I` từ thống kê phân phối DFL (top-k sharpness + mean) + **conditioning theo scale** (learned temperature per scale). Điểm chọn lọc `J = σ(cls) × I`.
2. **Ranking-aware objective**: thêm **pairwise ranking loss** (RankNet-style margin) trên các candidate của cùng một GT, với thứ tự ground-truth là IoU. Đây là khác biệt chính so với GFLv2 (dùng QFL pointwise): tối ưu trực tiếp thứ tự mà AP phụ thuộc.
3. **Scale-calibrated temperature**: hiệu chuẩn `I` theo scale box (continuous embedding), để tiny và large objects có calibration khác nhau mà không dùng discrete bins (tránh over-specialization).

**Kỳ vọng**: AP@0.75 ≥ 0.035 (≥50% headroom oracle), mAP ≥ 0.085, recall ≥ 0.32, FP không tăng, crowded recall ≥ 0.50, và **không regression** trên COCO person (medium/large).

---

## 1. Vấn đề: bằng chứng thực nghiệm (Evidence Dossier)

### 1.1. Oracle AP75 — nút thắt là ranking

| Variant | AP@0.75 | Diễn giải |
|---|---:|---|
| baseline | 0.0237 | Model thật |
| box (thay hộp bằng GT) | 0.0237 | Δ≈0 — hồi quy hộp **không** phải nút thắt |
| score (thay điểm bằng IoU) | 0.1355 | ×5.7 — nút thắt là **XẾP HẠNG** |

→ Toàn bộ headroom của AP75 nằm ở **thứ hạng độ tin cậy**, không ở độ chính xác hộp. Module hiệu quả phải tác động vào **score dùng để chọn lọc**, không phải logit phân loại.

### 1.2. QAC v3 thua DFL16 (controlled, 3.24M params, 50ep)

| Metric | QAC v3 | DFL16 | Δ |
|---|---:|---:|---:|
| mAP50-95 | 7.04% | 7.38% | −0.34% |
| Precision | 37.28% | 38.54% | −1.26% |
| Recall | 28.71% | 27.41% | +1.30% |
| FP tổng | 817 | 982 | −16.8% |
| Miss | 1249 | 1115 | +12% |

QAC v3 **conflate** hai mục tiêu: điều biến logit in-place `c_final = c_base + q_pred·v_sem` vừa thay đổi target TAL vừa sửa ranking gián tiếp. Kết quả: FP giảm nhưng miss tăng, mAP không cải thiện.

### 1.3. Phân tích lỗi & density

- Background FP chiếm **63%** FP (84,359 / 133,941) trên toàn tập; trên val nhỏ: bg_FP=220, dup=155, loc_FP=442.
- very_crowded: 81 ảnh = 51.5% GT, recall chỉ **40.07%**.
- Size-bin recall (QAC v3 ep50, conf 0.25): 0-8px 15.2%, 8-16px 30.0%, 16-32px 16.8%, 32+ 18.4%.
- SADR8 (reg_max=8): AP@0.50 +2.87, Recall +3.62, Miss −495, bg_FP −6,151 → regression improvement có bằng chứng. Giữ làm nền.

### 1.4. Representation collapse (không phải nút thắt cho AP75)

AUPRC cho 0-8px: 0.435 (C2) → 0.258 (C5). Feature info suy giảm theo độ sâu. **Nhưng** oracle box-only cho Δ≈0 → localization đã đủ tốt khi được chấm điểm đúng. Vì vậy representation collapse **không** là binding constraint cho AP75; **ranking** mới là. (Representation vẫn đáng theo dõi nhưng không phải mục tiêu chính.)

### 1.5. Điều YOLO26 trong repo thực sự có (đã rà soát)

- Ultralytics 8.4.164. Vanilla YOLO26: `reg_max=1` (không DFL), end2end dual head, STAL/ProgLoss **không có trong code**.
- Repo: backbone YOLO26 + P2 (4 levels), `reg_max=16` (DFL re-enabled), `end2end=True`, inference dùng nhánh one-to-one, chọn top-300 theo điểm cls thô, **không NMS**.
- Bbox loss dùng **CIoU** (gradient tồn tại khi không overlap) — "IoU=0 → no gradient" là sai.
- Assigner dùng CIoU + `select_candidates_in_gts` phóng to GT < `stride_val` (xử lý tiny có sẵn).
- one-to-one branch **detach** khỏi backbone trong training.
- `fuse()` xóa one-to-many branch khi inference end2end.

**Hệ quả then chốt**: trong chế độ end2end, **không có NMS để sửa thứ hạng sau inference**. Thứ hạng phải được học **bên trong điểm số của nhánh one-to-one**. Đây chính là chỗ QAR-NF can thiệp.

### 1.6. Phát biểu bài toán hình thức

Gọi `D = {(x_k, y_k)}` là tập ảnh và GT. Mỗi ảnh sinh ra `N` candidate `{(b_i, c_i)}` với `b_i` là box, `c_i` là cls logit. Detector hiện tại chọn top-M theo `σ(c_i)`. Gọi `u_i = IoU(b_i, gt(i))` (0 nếu không match).

**Mục tiêu thực tế** (AP@τ): sắp xếp candidate sao cho các candidate có `u_i ≥ τ` đứng trước các candidate còn lại. Đây là bài toán **ranking theo `u`**, không phải bài toán regression `u`.

**Khoảng cách hiện tại**: `σ(c_i)` được huấn luyện bởi BCE với target TAL `t_i = cls^α·u^β` (soft, không phải `u`). Vì vậy thứ tự của `σ(c_i)` **không nhất thiết** trùng thứ tự của `u_i`. Oracle score-only chứng minh: nếu thay `σ(c_i)` bằng `u_i`, AP75 tăng ×5.7. Đây là **gap định lượng** mà QAR-NF lấp.

**Bài toán QAR-NF**: học hàm `J_i = σ(c_i) · I_i` và một objective `L` sao cho thứ tự của `J` xấp xỉ thứ tự của `u`, với `I` là quality score dự đoán từ phân phối DFL và scale. Cụ thể:
```
minimize  E[ L_AP-approx(J, u) ]  subject to  J = σ(c)·I,  I = f_θ(DFL_stats, scale)
```
trong đó `L_AP-approx` là pairwise surrogate của AP (RankNet + margin), và `f_θ` là MLP nhỏ với scale-conditioned temperature.

**Tính chất mong muốn**: `J` là **monotone** theo `u` (ranking consistency), và **calibrated** theo scale (tiny không overconfident). Đây là hai tiêu chí đo được: Spearman correlation `(J, u)` và ECE theo size bin.

---

## 2. Liên quan công trình (Related Work)

| Công trình | Ý tưởng | Tại sao liên quan / khác biệt |
|---|---|---|
| **GFLv2** (Li et al., NeurIPS 2020) | DGQP: top-k sharpness → MLP → IoU quality → `J = C×I` | Nền tảng decoupling. Nhưng thiết kế cho **NMS-based** detector; QAR-NF thích ứng cho **NMS-free** và thêm ranking loss + scale conditioning. |
| **TOOD** (Feng et al., ICCV 2021) | TAL: `align = cls^α × IoU^β`, soft target | Repo đang dùng TAL (alpha=0.5, beta=6.0). QAR-NF giữ TAL, thêm quality head + ranking loss ở tầng score. |
| **NWD** (Wang et al., 2021) | Gaussian Wasserstein cho tiny | Dùng làm **auxiliary target** cho quality head (smoothing), không phải target chính (AP đo IoU). |
| **SET** (CVPR 2025) | Frequency-domain background smoothing, high-freq noise hại tiny | Bổ sung cho FGS (nếu cần), nhưng không phải nút thắt chính (ranking). |
| **AP Loss / aLRP / RS Loss** | Ranking-based detection losses | Chứng minh ranking loss cải thiện AP. QAR-NF dùng pairwise ranking (RankNet-style) thay vì pointwise QFL. |
| **RankDet** | IoU-guided ranking loss | Cùng hướng; QAR-NF kết hợp với quality head và scale conditioning. |
| **YOLOv10** | NMS-free dual assignment | Tiền thân; YOLO26 refine. QAR-NF là lớp quality-aware ranking trên cùng kiến trúc. |
| **Temperature scaling / multivariate calibration** | Scale-aware calibration | Cảm hứng cho scale-conditioned temperature. |

**Khoảng trống (gap) mà QAR-NF lấp**: không có công trình nào (theo survey) kết hợp (a) quality-aware ranking score, (b) pairwise ranking loss thay cho QFL pointwise, (c) scale conditioning, (d) trên nhánh **one-to-one NMS-free** của YOLO26. GFLv2 là NMS-based; YOLOv10/YOLO26 NMS-free nhưng không có quality-aware ranking score (chỉ TAL trong assignment).

---

## 3. Phương pháp: QAR-NF

### 3.1. Kiến trúc

```
Nhánh one-to-one (inference path, end2end):
  F_cls (64ch) ──► cls_proj ──► c_base (logit)
  F_box (64ch) ──► box_proj ──► b (DFL logits)
  g_dfl (24ch moments) ──┐
                          ▼
              ┌─────────────────────┐
              │  BoxGuided... (giữ) │  ← feature refinement (giữ từ v3, optional)
              └─────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │  QSH-NF (MỚI)       │  ← quality head decoupled
              │  top-k sharpness    │
              │  + mean-of-top-k    │
              │  + scale embedding  │
              │  → MLP → I ∈ (0,1) │
              └─────────────────────┘
                          │
                          ▼
              J = σ(c_base) × I_scale   ← điểm chọn lọc cuối (top-300 selection)
```

**Khác biệt với QAC v3**: QAC v3 làm `c_final = c_base + q_pred·v_sem` (in-place, conflates). QAR-NF làm `J = σ(c_base) × I` (decoupled, chỉ dùng cho ranking/selection, **không** chạm cls loss target).

**Khác biệt với GFLv2**: GFLv2 dùng QFL pointwise và scale-agnostic. QAR-NF dùng (i) **pairwise ranking loss** zusätzlich, (ii) **scale conditioning** trong quality head.

### 3.2. QSH-NF: quality head decoupled + scale conditioning

**Input features** (theo GFLv2 DGQP, đã hiệu chỉnh theo review):
- Top-k (k=5) probability per side (l,t,r,b) từ phân phối DFL: `4×5 = 20` channels.
- Mean-of-top-k per side: `4` channels (phản ánh độ sắc của phân phối).
- **Scale embedding** (MỚI): từ `box_w, box_h` (đã có trong moment descriptor) → `sin/cos` encoding của `log(area)` → `4` channels. Đây là conditioning liên tục, không discrete bins → smooth generalization.
- Tổng: `28` channels → MLP (28 → 16 → 1) + sigmoid → `I_raw ∈ (0,1)`.

**Scale-conditioned temperature** (MỚI):
```
I = sigmoid( (logit(I_raw)) / τ(scale) )
τ(scale) = 1.0 + 0.5 · tanh( MLP_scale(scale_embed) )   # bounded [0.5, 1.5]
```
- `τ` lớn hơn cho tiny (làm I "mềm" hơn, tránh overconfident) — **learned**, không fix.
- Init `τ=1.0` (identity). Residual bounded ±0.5.
- Lý do: IoU của tiny object rất dốc → target high-variance → cần calibration mềm hơn. Large object IoU ổn định → τ≈1.0.

**Target** (Direction A, đã sửa theo review):
```
L_quality = QFL(I_pred, IoU_gt)            # primary (AP đo IoU)
          + λ_nwd · BCE(I_pred, NWD)       # auxiliary smoothing, λ_nwd=0.3
```
- IoU là target chính (vì AP evaluator đo IoU; YOLO dùng CIoU nên gradient tồn tại → không có vấn đề gradient).
- NWD auxiliary = variance reduction (smoothing) cho tiny, không phải gradient.

### 3.3. Ranking-aware objective (điểm mới so với GFLv2)

GFLv2 dùng QFL (pointwise). QAR-NF thêm **pairwise ranking loss** trên nhánh one-to-one.

**Ký hiệu**: với mỗi ảnh, gọi `C` là tập candidate trong top-k (k=10) của mỗi GT; `J_i = σ(c_i)·I_i` là điểm chọn lọc; `u_i = IoU(box_i, gt)` là chất lượng thật.

**Pairwise loss** (RankNet-style, weighted by chênh lệch chất lượng):
```
L_rank^gt = Σ_{(i,j) ∈ C×C, u_i > u_j}  w_ij · softplus( -(J_i - J_j) )
            với w_ij = |u_i - u_j| / Σ|u· - u·|     (chuẩn hóa theo cặp)
```
- Chỉ những cặp có thứ tự chất lượng rõ ràng (u_i > u_j) mới tạo gradient.
- Trọng số `w_ij` tỉ lệ chênh lệch IoU → cặp "gần nhau" (khó) đóng góp ít hơn cặp "xa nhau" (dễ), ổn định gradient.

**Foreground-vs-background margin** (hard negative mining):
```
L_rank^bg = Σ_{i∈FG, j∈BG_top20}  softplus( m - (J_i - J_j) ),   m = 0.3
```
- Chỉ BG có cls score cao (top-20) mới tham gia → tập trung vào hard negatives, tránh làm loãng gradient bởi BG dễ.

**Tổng**:
```
L_rank = (1/|GT|) Σ_gt L_rank^gt  +  λ_bg · L_rank^bg
```
với `λ_bg = 0.1` (init).

- Chỉ áp dụng trên **nhánh one-to-one** (nhánh inference) để không làm pha loãng one-to-many.
- λ_rank = 0.5 (init), tune trong ablation.
- **Tại sao pairwise thay vì listwise**: pairwise đơn giản hơn, gradient ổn định, đã chứng minh hiệu quả (RankNet, RankDet). Listwise (ListMLE) là fallback nếu pairwise không đủ.

**Tổng loss one-to-one branch**:
```
L_o2o = L_cls (BCE, target TAL) + L_box (CIoU) + L_dfl + λ_q·L_quality + λ_r·L_rank
```
với λ_q=0.25, λ_r=0.5.

### 3.4. Dual-head handling (bắt buộc)

- QSH-NF đặt trên **cả** one-to-many và one-to-one, **không chia sẻ weight** (vì one-to-one detach → backbone gradient không chảy; weight sharing sẽ làm quality head one-to-one phụ thuộc backbone quality head one-to-many một cách không mong muốn).
- Quality target theo assignment tương ứng (o2m: top-10; o2o: top-1).
- **Eval đúng branch sẽ deploy**: nếu deploy NMS-free (one-to-one), metric QSH phải đo trên o2o branch.
- `fuse()` handling: đảm bảo QSH trên one-to-one branch không bị xóa khi fuse (QSH là sub-module của one2one head, giữ nguyên).

### 3.5. Forward-compat với YOLO26 vanilla (reg_max=1)

Nếu `reg_max=1` (không DFL), không có phân phối → QSH-NF dùng **FeatureStatsExtractor** (mean/std/max của F_cls trên patch) làm variant riêng (`QAR-NF-FS`) với hypothesis/ablation riêng. Không coi là drop-in tương đương (theo review §3.5).

### 3.6. Tham số bổ sung

| Thành phần | Tham số |
|---|---:|
| QSH-NF MLP (28 → 16 → 1) | 28×16 + 16 + 16×1 + 1 = 481 |
| Scale embedding + τ MLP | ~120 |
| Per level × 2 branches | 2 × 4 × (481+120) ≈ 4,808 |
| **Tổng QAR-NF** | **~4.8k** (so với 2.064k QAC v3, 1.412k QSH v4) |
| Pairwise ranking loss | 0 (loss, không tham số) |
| **Tổng model** | baseline + SADR8 + ~4.8k ≈ **2.724M** (+8.3%) |

---

## 4. Tại sao không over-specialize vào tiny

Bốn safeguard:

1. **Ranking loss là scale-agnostic**: pairwise ranking chỉ quan tâm thứ tự tương đối (IoU), không quan tâm kích thước. Model không bị "ép" chỉ detect tiny.
2. **Scale conditioning liên tục**: scale embedding dùng sin/cos của log(area) → smooth interpolation giữa tiny và large. Không discrete bins → không overfit vào bin cụ thể.
3. **Bounded temperature residual**: τ ∈ [0.5, 1.5], init 1.0. Large object → τ≈1.0 (không thay đổi). Tiny object → τ learned trong [0.5,1.5]. Không thể "phá" large-object path.
4. **Generalization check**: eval trên COCO person (medium/large). Acceptance: COCO person AP unchanged within ±0.5. Nếu regression → dừng.

Ngoài ra, QAR-NF **không** thay đổi feature representation hay regression → localization path cho large object không bị ảnh hưởng (oracle box-only Δ≈0 cho thấy localization đã đủ).

---

## 5. Ablation (gated, theo invariant #7 của repo)

### 5.1. Biến thể

| Biến thể | QSH-NF | Ranking loss | Scale cond. | Mục đích |
|---|---:|---:|---:|---|
| **QAR-NF-0** (QSH pointwise) | ✓ (o2o) | ✗ | ✗ | Đo đóng góp quality head decoupled (GFLv2-style) |
| **QAR-NF-1** | ✓ | ✓ (pairwise) | ✗ | Đo đóng góp ranking loss |
| **QAR-NF-2** | ✓ | ✓ | ✓ | Full proposal |
| QAR-NF-2 + SADR8 | ✓ | ✓ | ✓ | Trên nền regression đã chứng minh |
| SADR8 (existing) | ✗ | ✗ | ✗ | Baseline cải tiến |

### 5.2. Gating rules

- Nếu AP@0.75 của QAR-NF-0 không tăng so với SADR8 sau 1 seed → dừng, thử scale conditioning trước ranking loss.
- Nếu pairwise ranking loss không cải thiện AP@0.75 → thử listwise (ListMLE) hoặc bỏ, giữ pointwise QFL.
- Nếu scale conditioning làm regression trên COCO person → bỏ scale conditioning, giữ QSH + ranking loss.

### 5.3. Chỉ tiêu mục tiêu (so với SADR8 hiện tại)

| Chỉ số | SADR8 | Mục tiêu QAR-NF-2 |
|---|---:|---:|
| AP@0.50 | 0.2589 | ≥ 0.275 |
| AP@0.75 | 0.0233 | ≥ 0.035 (≥50% headroom oracle) |
| Recall tổng | 0.5580 | ≥ 0.58 |
| Recall very_crowded | 0.4352 | ≥ 0.50 |
| Background FP | 78,208 | ≤ 70,000 |
| tiny1 (0-8px) Recall | 0.2392 | ≥ 0.28 |
| COCO person AP | baseline | unchanged ±0.5 |

### 5.4. Giao thức

- Dataset: `/mnt/data/tinyperson_official_v1/data.yaml`, imgsz 1280, seed 42 (rồi 43, 44).
- 50 epoch chặng 1 (lr 0.001) + 50 epoch nối tiếp (lr 0.0006, cosine) nếu chặng 1 dương.
- Đánh giá: val + official test + phân tích hình học + lỗi + 3 metric pre/post-NMS (nếu đo được).
- **Tiêu chí dừng**: AP@0.75 không tăng so với SADR8 sau 1 seed → dừng biến thể.
- **Thống kê**: báo cáo mean ± std trên 3 seed; dùng paired bootstrap (1000 resample) để kiểm tra ý nghĩa ΔAP@0.75 (p < 0.05). Nếu p ≥ 0.05 → coi là không có cải thiện.

---

## 6. Rủi ro và phương án dự phòng

| Rủi ro | Mức độ | Phương án |
|---|---|---|
| Pairwise ranking loss không cải thiện AP@0.75 | Trung bình | Thử listwise (ListMLE); hoặc bỏ, giữ QFL pointwise |
| Quality head (IoU) không calibrate được cho tiny | Trung bình | Scale conditioning (τ); hoặc hybrid target (Direction B) |
| Ranking loss làm tăng FP (rank BG cao hơn) | Trung bình | Margin m=0.3; hard negative mining top-20 BG; theo dõi FP/image |
| one-to-one detach → gradient yếu cho quality head | Trung bình | QFL + ranking loss đủ signal; theo dõi calibration corr(I, IoU) |
| fuse() xóa quality head trên one-to-one | Cao (nếu không handle) | Đảm bảo QSH là sub-module của one2one head, fuse giữ lại |
| reg_max=8 vs 16 cho QSH descriptor | Trung bình | Ablation reg_max=8 vs 16 cho QSH; đo calibration |
| Over-specialization vào tiny | Thấp (đã safeguard) | COCO person check; pairwise ranking scale-agnostic |

---

## 7. Implementation scope

- Sửa `src/tinydet/modules/qac_v3.py` (hoặc tạo `qac_rank.py` mới):
  - Thêm `QSH-NF` head trên one-to-one branch (MLP 28→16→1 + scale conditioning).
  - Thêm pairwise ranking loss vào `QualityAwareDetectionLossV3` (chỉ branch one-to-one).
  - Giữ backward compat: params mới init identity (residual 0), checkpoint cũ load qua `strict=False`.
- `configs/model/yolo26n_tiny_A2_p2_qac_rank.yaml`: `qac_rank: true` + keys.
- `src/tinydet/models/registry.py`: thêm `QAC_RANK_KEYS` và hook.
- Eval: thêm logging pre/post-NMS recall (để DAR có bằng chứng nếu cần sau).
- Ước tính: 3-4 ngày implement + ablation.

---

## 8. Tại sao đây là đóng góp khoa học (không phải lặp lại)

| Hướng đã thử | Kết quả | Tại sao QAR-NF khác |
|---|---|---|
| QAC v2/v3 (điều biến logit in-place) | mAP thua baseline | QAR-NF **decouple**: cls head độc lập, quality head riêng cho ranking |
| QSH v4 (pointwise QFL) | Chưa chạy | QAR-NF thêm **pairwise ranking loss** (tối ưu thứ tự, không chỉ điểm) |
| GFLv2 (NMS-based) | COCO AP +2.6 | QAR-NF thích ứng cho **NMS-free** one-to-one + scale conditioning |
| Ranking losses (AP/aLRP/RS) | AP cải thiện | QAR-NF kết hợp với **quality head + scale conditioning** trên YOLO26 |
| SADR8 (regression) | Recall +3.62 | QAR-NF **giữ** SADR, thêm ranking (không cạnh tranh) |

**Khác biệt then chốt**: QAR-NF tấn công trực tiếp vào **score dùng để chọn lọc trong chế độ NMS-free** (oracle lever), với **objective là ranking** (không phải pointwise regression), và **scale conditioning** để generalize. Đây là hướng mà không có biến thể nào trong repo hay văn liệu (theo survey) đã thử trên YOLO26 NMS-free.

---

## 9. Kết luận

QAR-NF là một cải tiến đơn lẻ, có căn cứ thực nghiệm (oracle AP75 ×5.7), có cơ sở lý thuyết (GFLv2 decoupling + ranking losses + scale-aware calibration), có ablation sạch (3 biến thể gated), có safeguard chống over-specialization (scale-agnostic ranking loss, continuous scale embedding, bounded residual, COCO person check), và implementable trên codebase hiện tại. Đây là đề xuất để viết thành báo cáo khoa học.

**Bước tiếp theo**: implement QAR-NF-0 (QSH pointwise) trên nhánh one-to-one, đo AP@0.75 sau 1 seed. Nếu dương, thêm ranking loss (QAR-NF-1), rồi scale conditioning (QAR-NF-2).
