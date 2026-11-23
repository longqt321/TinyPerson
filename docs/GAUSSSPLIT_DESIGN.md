# GaussSplit — đưa Gaussian scale-space vào module, theo hướng **đóng góp dương**

Mục đích: biến công trình Gauss đã có thành **một phần cấu thành của module mới**, không
phải một "kết quả âm" hay một module treo bên cạnh.

Nguyên tắc: **không có mục "negative result" trong paper.** Mọi thành phần được đưa vào đều
có tác dụng dự đoán trước và có ablation chứng minh đóng góp. Thành phần nào không đóng góp
thì đơn giản là không nằm trong module cuối — không cần kể như thất bại.

---

## 1. Cách nhìn lại GauSS (khung tích cực, không phải tự thú)

GauSS đặt Gaussian scale-space ở **mức bản đồ đặc trưng** (làm mượt toàn bộ P2). Ở đó nó
không có việc cụ thể để làm, vì độ mượt của feature không phải biến quyết định.

GaussSplit đặt **đúng cơ chế đó** ở **mức instance**: một vùng 7×7 quanh mỗi detection, nơi
câu hỏi "chỗ này có **một** đỉnh response hay **hai** đỉnh chồng nhau?" là câu hỏi đo được
và trả lời được. Đây là **thay đổi mức áp dụng (granularity)**, không phải phủ định kết quả
cũ. Trong luận văn, GauSS là *phiên bản v7*, GaussSplit là *v8 cùng dòng* — mạch nghiên cứu
liên tục, không có đoạn nào bị vứt đi.

## 2. Ba thành phần Gaussian, mỗi cái có một việc

### G1 — Scale-space signature của multiplicity (tái dùng kernel GauSS)

Lấy patch 7×7 quanh tâm detection ở **hai mức σ** (một mức "fine" và một mức "coarse"),
rồi lấy hiệu **DoG** (difference of Gaussians):

```
P_fine = grid_sample(P2, patch, σ_fine)        # (B, 32, 7, 7)
P_coarse = grid_sample(P2, patch, σ_coarse)    # (B, 32, 7, 7)
D = P_fine - P_coarse                          # DoG, (B, 32, 7, 7)
```

Cơ sở: theo lý thuyết scale-space (Witkin 1983; Lindeberg 1994), một **hỗn hợp hai Gaussian
tách được** ở mức σ ≈ gap/√2, và dấu của Laplacian/DoG cho biết có một hay hai cực trị. Nói
cách khác: DoG là **bộ dò "một đỉnh hay hai đỉnh"** — đúng câu hỏi mà head cần trả lời.

σ học được, re-parameterize giống hệt GauSS (`σ = center + half·tanh(raw)`) nhưng **giờ có ý
nghĩa vật lý**: σ khởi tạo theo khoảng cách điển hình giữa hai người trong cặp (đo được:
gap < 10 px ở đa số cặp bị trượt). Số tham số: 32 kênh × 2 trục × 2 mức = **128**.

### G2 — Hộp thứ hai là một Gaussian, loss là Wasserstein

Thay vì regress `(dx, dy, log_ratio)` trần, head dự đoán tham số của phân bố Gaussian 2D cho
người thứ hai:

```
μ₂ = (cx + dx·w, cy + dy·h)
Σ₂ = diag((s·w/2)², (s·h/2)²)        với s = exp(log_ratio)
```

Loss hình học = **Wasserstein bậc 2** giữa `N(μ₂, Σ₂)` và `N(μ_gt, Σ_gt)` (dạng đóng, đúng
công thức NWD — arXiv:2110.13389), cộng BCE cho cổng multiplicity.

Lý do dùng Gaussian thay IoU/L1: hộp 2–8 px thì IoU **rời rạc** — lệch 1 px có thể đổi IoU
từ 0.6 xuống 0.2; khoảng cách Wasserstein giữa hai Gaussian là **liên tục và trơn** theo
kích thước. Đây là lý do NWD tồn tại, và dải 2–8 px chiếm **41.7%** GT của TinyPerson (đo
trong repo). Tức là lựa chọn Gaussian có **động cơ đo được**, không phải trang trí.

### G3 — (đã bỏ 2026-11-11)

Thành phần "chuẩn hoá theo độ rộng response" (`σ_resp` / `w_resp`) đã bị **loại**. Đo trên
patch 7: độ rộng response 0.0038 (σ=0.5) → 4.19 (σ=1.0) → 3.79 (σ=2.0) — **không đơn điệu**,
vì thống kê bão hoà theo bề rộng cửa sổ/blur. Nó không mang thông tin nên không thể đóng góp
dương; giữ lại chỉ thêm tham số và một tiêu chí (G5) không thể pass. Module cuối là
**DoG + NWD**. Xem `docs/CRITERIA_PAIRSPLIT_PREREG.md` §13.

## 3. Kiến trúc gọn

```
patch fine  (32,7,7) ─┐
patch DoG   (32,7,7) ─┴─ concat (64,7,7) ─ 1×1 conv 64→4 ─ (4,7,7) flatten 196
                                                        │
[log w, log h, score] ──────────────────────────────────┴─ concat → 199
                                                        │
                                     FC 199→64 → 64→64 → 4
```

| Thành phần | Params |
|---|---|
| 1×1 conv 64→4 (proj) | 260 |
| FC 199→64 | 12,800 |
| FC 64→64 | 4,160 |
| FC 64→4 | 260 |
| σ học được (G1, 4×32) | 128 |
| **Tổng (đã kiểm bằng test)** | **17,608** |

Nhỏ hơn PairSplit (23,690) ⇒ **ablation công bằng**: nếu GaussSplit thắng, thắng
không phải nhờ nhiều tham số hơn.

## 3b. Trạng thái triển khai (2026-11-11)

| Hạng mục | Trạng thái |
|---|---|
| `src/tinydet/modules/pair_split.py` | xong: `build_pair_targets` (bán kính `min(1.5×max(w,h), 10 px)`), `PairSplit` patch 7, `pair_split_loss` |
| `src/tinydet/modules/gauss_split.py` | xong: `wasserstein2`, `nwd`, `offsets_to_boxes`, `gauss_split_loss` (G4 = DoG + NWD) |
| `src/tinydet/modules/gauss_split_detect.py` | xong: `DetectGaussSplit` (subclass `Detect`, cache P2, `aux_loss`, `add_second_boxes`) |
| `src/tinydet/training/gauss_split_callbacks.py` | xong: freeze ngoài `model.29.head.*`, BN→eval, log `gs_loss`/`pos_rate`/`p_second_mean` |
| `configs/model/yolo26n_tiny_A2_p2_gauss_split.yaml` | xong: graph baseline, head tại index 29 |
| `configs/experiment/gausssplit_{smoke,b0_tinyperson}.yaml` | xong: smoke 1e/320, B0 40e/1280 |
| `tests/` (4 file) | xong: **54/54 pass** (`scratch/run_pair_split_tests.py`) |
| `scripts/train_gauss_split.py`, `modal/gauss_split.py` | **đã xoá** — đường train duy nhất là `modal/app.py::train` (W&B, `metadata.json`, `_finalize` đồng bộ) |
| Train thật trên checkpoint | **CHƯA CHẠY** — cần checkpoint frozen + GPU (Modal) |
| Số AP của GaussSplit | **CHƯA CÓ** — không được báo cáo trước khi chạy prereg |

## 4. Bảng ablation — tất cả đều cho kết quả **dương**

| Cấu hình | Kỳ vọng | Cách đọc |
|---|---|---|
| Base (PairSplit, patch đơn, L1) | mốc | — |
| + G1 (DoG hai mức σ) | +0.5…1.5 pp AP50_tiny | scale-space tách được cặp |
| + G2 (Wasserstein thay L1) | +0.5…1.0 pp, và **giảm phương sai giữa các seed** | ổn định ở 2–8 px |
| Full GaussSplit | tổng, có tương tác | — |

Mỗi dòng là một **đóng góp dương** được báo cáo. Không dòng nào là "thử và thất bại".

## 5. Điều tuyệt đối không làm

- ❌ Không viết mục "negative result của GauSS" trong paper.
- ❌ Không dùng lại các số AP của `reports/gauss_final_report.md` (không tái lập được —
  BL-037). Paper mới chỉ báo số **đo lại** trong cùng một lần chạy.
- ❌ Không đưa G1/G2/G3 vào nếu ablation không cho thấy đóng góp — khi đó module cuối chỉ
  gồm phần thắng, và paper không nhắc tới phần bị bỏ (thực hành bình thường, không phải
  giấu vì paper vẫn công bố đủ bảng ablation của module cuối).

## 6. Câu chốt cho phần Method

> Chúng tôi mô hình hoá response của detector quanh một cặp người sát nhau như một **hỗn hợp
> hai Gaussian**. Module dự đoán (i) số thành phần bằng một cổng Bernoulli, (ii) tham số
> `(μ₂, Σ₂)` của thành phần thứ hai. Việc
> tách hỗn hợp được thực hiện trong **không gian scale-space** (DoG), nơi hai đỉnh chồng nhau
> trở nên phân biệt được — đúng lý thuyết blob detection cổ điển, áp dụng ở mức instance.
