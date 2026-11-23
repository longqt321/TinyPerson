# GaussSplit runbook — lượt train cuối

Viết 2026-11-11. Đây là tài liệu vận hành cho lượt train cuối; mọi con số tiêu chí đã
định trước ở `docs/CRITERIA_PAIRSPLIT_PREREG.md` (Phần 1 áp cho cả hai, Phần 2 riêng cho
họ Gaussian). Tài liệu này trả lời: checkpoint nào, train bao nhiêu, đánh giá thế nào.

---

## 1. Checkpoint / baseline để gắn GaussSplit

**Checkpoint: `best.pt` của run `baseline_tinyperson_wacv_150e`.**

- Config: `configs/experiment/baseline_tinyperson_wacv_150e.yaml`
  (model `yolo26n_tiny_A2_p2_tinyperson.yaml`, 150 epoch, imgsz 1280, seed 42, A10).
- **Lý do bắt buộc phải là run này:** toàn bộ bằng chứng R20–R22 (gap cặp sát +17pp ở
  tiled, 53.5% miss có detection ở IoU≥0.3, v.v.) được đo trên **chính checkpoint này**.
  Gắn head lên checkpoint khác = mất cơ sở động cơ, reviewer sẽ bắt ngay.
- **Không dùng** checkpoint `gauss_v3_tinyperson_wacv_150e` — đó là model có GauSS trong
  backbone; gắn head lên đó thì không tách được "head đóng góp" khỏi "GauSS đóng góp".
- File không có trong repo (repo không commit checkpoint). Tìm trên Modal:

  ```bash
  uv run python tools/list_runs.py --config configs/experiment/baseline_tinyperson_wacv_150e.yaml
  ```

  rồi lấy `checkpoints/best.pt` (hoặc `training/weights/best.pt`) của run mới nhất khớp
  config. Đường dẫn volume trên Modal thường là `/mnt/runs/<run_id>/checkpoints/best.pt`.

## 2. Train bao nhiêu

**40 epoch × 41 bước/epoch = 1.640 bước** (config `gausssplit_b0_tinyperson.yaml`, imgsz 1280,
batch 16 — đúng theo pipeline chuẩn, không còn script riêng).

- "Bước" = 1 batch 16 ảnh full-frame 1280. Pool train thật (đo 2026-11-11): 662 ảnh ⇒
  **3.195 tile có GT** (30.798 GT). 1.640 bước × 16 ảnh ≈ 26.240 lượt ảnh full-frame —
  nhiều gấp nhiều lần 20.000 tile của thiết kế cũ, nên không phải tăng epoch.
- Số positive trong pool: **6.055 GT (19.7%)** có láng giềng trong bán kính
  `min(1.5×max(w,h), 10 px)` (luật cũ 1.5× cho 18.949 = 61.5%, nhưng phần lớn nằm ngoài cửa
  sổ receptive của head). Với 26k lượt ảnh, mỗi positive được thấy ~50 lần; dư cho head nhỏ.
- Chi phí dự kiến: ~40 phút GPU trên A10 (freeze toàn bộ detector, chỉ head 23.690 tham số
  học + forward aux mỗi batch). Rẻ.
- Dấu hiệu hội tụ: `gs_loss` giảm rõ trong vài trăm bước đầu; `pos_rate` > 0 và ổn định.
  **Không** kỳ vọng đúng 19.7%: con số đó đo trên **tile 640×512 (1:1)**, còn train chạy
  full-frame letterbox 1280 (hệ số ~0.667 với ảnh 1920 rộng) ⇒ cùng một cặp người có
  khoảng cách pixel nhỏ hơn ~1.5× trong không gian train. Cửa sổ ±12 px ở không gian 1280
  tương đương ±18 px ở không gian tile, tức **vẫn phủ trùm** dải ≤10 px cần cho eval —
  không có điểm mù, chỉ là target được lấy từ dải vật lý rộng hơn một chút (≤15 px tile).
  Nếu `gs_loss` phẳng sau ~5 epoch ⇒ khả năng K3 (head không tìm được signal) — dừng, báo
  cáo, đừng tune. Kill criteria K1–K6 xem `docs/CRITERIA_PAIRSPLIT_PREREG.md`.

## 3. Tiêu chí đánh giá (đo gì, bằng gì)

Giao thức: **tiled 640×512, overlap 100, 1:1, global NMS 0.5** — đúng giao thức của
baseline tiled. Toàn bộ đường eval nằm trong `modal/app.py::evaluate_tiled`
(`_run_tiled_checkpoint_evaluation`), dùng `tinydet.evaluation.tiled_inference.predict_tiled_image`
+ `tinydet.evaluation.tinyperson_benchmark.evaluate_tinyperson`; không còn script eval riêng.

> [!IMPORTANT]
> **Mốc AP đúng của checkpoint 150e là 0.3757 (tiled) / 0.3002 (full-frame)** — đã tái lập
> ngày 2026-11-11 bằng chính pipeline này: head-off trên 786 ảnh test cho AP50_tiny =
> **0.37556** (chênh 0.0001 so với report đã lưu, do `max_det` 200 vs 300 — không đáng kể).
> Cặp số **0.1886 / 0.3540** trong log R6.a là legacy: 0.1886 là model GauSS (không phải
> baseline), 0.3540 là baseline tiled của **checkpoint cũ trước chiến dịch 150e**. Hai bộ số
> này không cùng một model — tuyệt đối không trộn lẫn. Mọi so sánh trong paper dùng
> checkpoint `20261109-182052-a20f7b6-seed42` ⇒ mốc = **0.3757**.

**Padding đã xác minh bằng forward pass thật** (không phải suy luận): predict một tile
640×512 qua model A2-P2 cho P2 shape **(32, 128, 160)** = 512/4 × 640/4 ⇒ **không có
padding**, tọa độ feature = tọa độ tile / 4 chính xác. (LetterBox `auto=True` tính padding
là `dw mod stride` = 128 mod 32 = 0.)

Evaluator: `evaluate_tinyperson(..., class_mode="merge")` (model 1 class; sai cờ này = số rác).
Chỉ số chính: **AP50_tiny** (bin theo `sqrt(w*h)`, đúng chuẩn TinyBenchmark).

**Annotations:** `evaluate_tiled` tự lấy `tiny_set_test.json` (test) hoặc `tiny_set_train.json`
(val) rồi lọc theo đúng danh sách ảnh của split bằng `subset_annotations_to_split` — cùng một
luật cho mọi run, không cần file annotation chuẩn bị tay. Không dùng
`tiny_set_test_with_dense.json` trực tiếp (816 ảnh, thừa 30 ảnh ⇒ evaluator từ chối prediction
thiếu).

Số đối chứng bắt buộc phải có trong cùng một lần chạy (cùng checkpoint, cùng giao thức):

| Run | Lệnh | Ý nghĩa |
|---|---|---|
| head-OFF | `evaluate_tiled --checkpoint <baseline best.pt> --name base_test` | mốc (kỳ vọng ≈ **0.3757**; đo được 0.37556) |
| `B0` | train `gausssplit_b0_tinyperson` rồi `evaluate_tiled --head-threshold <val-selected>` | baseline PairSplit |
| `G4` | cần config riêng (chưa tạo — xem `docs/BACKLOG.md`) | GaussSplit đầy đủ |

**Chỉ tiêu chính P1/P2:** AP50_tiny của head-ON − head-OFF > 0 với CI bootstrap 95%
(cluster theo video prefix, B=2000) cận dưới > 0. Trần lý thuyết **+3–4 pp**; kết quả
+10 pp ⇒ nghi leakage, audit trước khi tin.

**Tiêu chí phụ (guards, vi phạm = vô hiệu P1):** xem Phần 1 mục 5 của prereg — G1 (TP
của GT đơn lẻ không giảm quá 1pp), G2 (precision hộp thứ hai ≥ 0.5), G3 (số detection/GT
không tăng quá 20%), G4 (gain phải tập trung ở dải d_nn<10px).

**Tiêu chí riêng GaussSplit (Phần 2):** P3 — mỗi thành phần còn lại (DoG, NWD) phải có
đóng góp dương qua bảng biến thể G1/G2/G3; G6 — σ học được phải dịch chuyển khỏi giá trị
khởi tạo. (G5/`w_resp` đã huỷ — xem prereg §13.)

## 4. Tiêu chí kiểm nghiệm (khi nào dừng, khi nào chết)

Kill criteria — gặp thì **dừng, không tune để cứu**:

| Mã | Điều kiện | Hành động |
|---|---|---|
| K1 | CI cận dưới ≤ 0 sau 1 lần train ở hyper cố định | báo fail, không sweep |
| K2 | precision hộp thứ hai < 0.5 | head phát nhiễu — kết luận âm cho claim representation |
| K3 | `p(second)` đứng ở 0.119 ± 0.02 và `reg` không giảm | R22 bị bác — viết lại là bác bỏ giả thuyết |
| K4 | số detection/GT tăng > 40% | head đang nhân bản, không NMS-cứu |
| K5 | G4 − B0 < 0.5pp với CI chứa 0 | Gaussian không tạo khác biệt — module của paper là B0 |
| K6 | G4 < B0 − 1pp | dừng nhánh Gaussian |

Nguyên tắc: **hyper-parameter đã khoá trước khi nhìn kết quả** (lr 1e-3, AdamW, wd 1e-4,
clip 10, max_ratio 3.0, iou_thr 0.5, radius `min(1.5×max(w,h), 10 px)`, patch 7).
Ngoại lệ duy nhất: **ngưỡng `p(second)` được chọn trên val** (84 ảnh,
`scratch/ann_val_official.json`) rồi test chấm **một lần** — xem prereg §13. Đổi bất kỳ
hyper-parameter nào khác sau khi đã nhìn AP = tự phá prereg.

## 5. Kiểm toán code gắn chặt với GauSS cũ — kết luận: KHÔNG XOÁ

Đã rà toàn bộ tham chiếu `gauss` trong repo. Phân loại:

| Thành phần | Vai trò với GaussSplit | Quyết định |
|---|---|---|
| `src/tinydet/modules/gauss.py` | **Load-bearing**: `gauss_split.py` import nguyên `LearnableGaussianScaleSpace` (kernel + re-parameterize σ) | GIỮ — là nền của G1 |
| `src/tinydet/training/gauss_gate.py` | trainer import vô điều kiện (`trainer.py:13`), chỉ kích hoạt khi config có `gauss_gate` | GIỮ — vô hại với baseline, xoá là vỡ trainer |
| `configs/model/yolo26n_tiny_A2_p2_gauss*.yaml`, `configs/experiment/gauss_*.yaml` | công trình cũ (paper trước) | GIỮ — tái lập được công trình cũ |
| `scripts/gauss_probe.py`, `scripts/test_gauss_gate.py`, `scripts/test_resume_and_gate.py`, `scripts/eval_ablation_local.py` | tooling của công trình cũ | GIỮ — bằng chứng tái lập |
| `register_ultralytics_modules()` | đăng ký class GauSS cho YAML cũ + `add_safe_globals` để load checkpoint cũ | GIỮ |

**Lý do không xoá gì:** (1) GaussSplit dùng lại operator của GauSS — xoá `gauss.py` là vỡ
module mới; (2) các config/tooling cũ là tài sản tái lập của paper trước — user đã nói rõ
không muốn lãng phí công trình Gauss; (3) coupling còn lại (trainer→gauss_gate) là
import vô điều kiện nhưng runtime-no-op với baseline, rủi ro xoá lớn hơn lợi ích.

Điều duy nhất nên làm khi dọn: **không đưa config GauSS vào paper mới**; paper mới chỉ
cần `baseline_tinyperson_wacv_150e` + head. GauSS cũ xuất hiện duy nhất ở vai trò tiền
thân kiến trúc của G1 (một câu trong Method là đủ).

## 6. Checklist chạy (theo thứ tự) — lệnh đã kiểm chứng trên Modal (2026-11-11)

Checkpoint thật (đã xác minh bằng volume listing + load thử; đúng checkpoint của 0.3757):

```
/mnt/runs/baseline_tinyperson_wacv_150e/20261109-182052-a20f7b6-seed42/training/weights/best.pt
```

```bash
# 0) local: kiểm tra code trước khi tốn GPU (2 giây)
.venv/bin/python scratch/run_pair_split_tests.py     # 54/54
.venv/bin/ruff check src/ tests/ scripts/ modal/     # All checks passed

# Lưu ý về lệnh modal: `modal` phải là bản trong venv của project.
#   - Nếu venv đang activate: `modal run ...` là đủ.
#   - Nếu không: dùng `.venv/bin/python -m modal run ...` (đã kiểm chứng bằng --help).
#   - KHÔNG dùng `~/.local/bin/modal` trực tiếp: Python của uv-tool đó thiếu `yaml`
#     ⇒ ModuleNotFoundError ngay khi import modal/app.py. `PYTHONPATH=$PWD/src`
#     không sửa được lỗi này.

# 1) smoke trên GPU (~3 phút): 1 epoch, imgsz 320, batch 4 — kiểm tra gs_loss > 0,
#    freeze đúng (2,515,976 frozen / 23,690 trainable), BN eval, best.pt có head
.venv/bin/python -m modal run modal/app.py::train \
  --config configs/experiment/gausssplit_smoke.yaml

# 2) train thật B0 (PairSplit head, L1) — 40 epoch x 41 step, ~40 phút trên A10
#    DÙNG ORCHESTRATOR DETACHED: toàn bộ chuỗi chạy bên trong Modal
#    (_orchestrate_campaign), local chết không ảnh hưởng. `modal/app.py::train`
#    KHÔNG dùng cho run dài: orchestration của nó chạy ở máy local (L-026).
#    --no-evaluate: dừng sau train+finalize, không chấm test trước khi chọn ngưỡng.
.venv/bin/python -m modal run -d launch_campaign.py::main \
  --configs configs/experiment/gausssplit_b0_tinyperson.yaml --no-evaluate
#    kiểm chứng ngay: `modal app list` phải thấy Tasks >= 1, `modal container list`
#    phải có container; nếu Tasks 0 thì run đã chết, đừng chờ.
#    theo dõi: `.venv/bin/modal app logs <APP_ID>`

# 3) chọn ngưỡng p(second) trên VAL (84 ảnh) — sweep, KHÔNG nhìn test
#    mỗi lệnh ~1 phút; chạy 0.30 / 0.40 / 0.50 / 0.60
.venv/bin/python -m modal run modal/app.py::evaluate_tiled \
  --checkpoint gausssplit_b0_tinyperson/<RUN_ID>/training/weights/best.pt \
  --name b0_val_t050 --head-threshold 0.50 --split val
#    -> đọc AP50_tiny trong reports/eval/b0_val_t050.json, chọn ngưỡng tốt nhất
#    (--split val dùng annotations tiny_set_train.json lọc theo split val, images/val)

# 4) chấm TEST một lần duy nhất với ngưỡng đã chọn (786 ảnh, ~10 phút)
.venv/bin/python -m modal run modal/app.py::evaluate_tiled \
  --checkpoint gausssplit_b0_tinyperson/<RUN_ID>/training/weights/best.pt \
  --name b0_test --head-threshold 0.50

# 5) đối chứng head-off trong cùng giao thức (kỳ vọng 0.3757)
.venv/bin/python -m modal run modal/app.py::evaluate_tiled \
  --checkpoint baseline_tinyperson_wacv_150e/20261109-182052-a20f7b6-seed42/training/weights/best.pt \
  --name base_test

# 6) tải summary về và ghi vào bảng kết quả
modal volume get tiny-person-runs /eval/b0_test/summary.json reports/eval/b0_test.json
```

**Không còn script train riêng.** Head là một lớp trong model YAML
(`DetectGaussSplit`), loss đi qua `E2ELoss` của Ultralytics, nên mọi thứ chuẩn
(`metadata.json`, `environment.txt`, `config.yaml`, `training/results.csv`,
`checkpoints/best.pt`, `metrics/`, W&B, `tools/list_runs.py`) đều được sinh như
mọi run khác. `gs_loss` nằm ngay trong bảng loss của Ultralytics, và callback
in thêm `p_second_mean` / `pos_rate` mỗi epoch.

Mỗi run sinh đúng cấu trúc chuẩn (xem `docs/agents/OPERATIONS.md`): `metadata.json`,
`environment.txt`, `config.yaml`, `training/results.csv`, `training/weights/best.pt`,
`checkpoints/best.pt`, `metrics/`. Cột `gs_loss` nằm trong `results.csv`. Tiled eval
ghi `eval/<name>/summary.json` + `predictions_tiled.json` trên volume.

Ghi lại `run_id` Modal vào `docs/DECISIONS.md` sau khi chạy. Evaluator từ chối chấm
file dự đoán thiếu ảnh (guard chống báo nhầm split thiếu hụt).

### 6b. Tín hiệu phải thấy ngay trong log (thiếu = dừng, gửi log)

| Thời điểm | Dòng log / giá trị | Ý nghĩa nếu sai |
|---|---|---|
| smoke, đầu train | `frozen detector params: 2,515,976 \| trainable: 23,690 \| head: 23,690` | freeze sai ⇒ detector bị train lại, mất tiền đề "frozen" |
| smoke, epoch 0 | `BN modules forced to eval: 145` | BN trôi ⇒ detector khác baseline |
| smoke, bảng loss | có cột `gs_loss` và **> 0** (config smoke có pretrained) | `gs_loss = 0` ⇒ checkpoint không load được / không có positive |
| smoke, cuối | `best.pt` có 8 key `model.29.head.*`, và **không** chứa `feat` | checkpoint sai hoặc còn cache rác |
| B0, mỗi epoch | `p_second_init=0.1192`, `p_second_mean` **rời khỏi** 0.1192, `pos_rate` > 0 và ổn định (giá trị tuyệt đối không kỳ vọng = 19.7% — đo ở không gian tile, train ở không gian 1280) | K3: head không tìm được signal ⇒ dừng, không tune |
| B0, `results.csv` | header có `train/gs_loss` + `val/gs_loss`; giá trị giảm dần | cột lệch ⇒ cơ chế log hỏng |
| eval | `--> GaussSplit head threshold = <x>` | thiếu dòng này ⇒ `--head-threshold` không tới model |

Lưu ý: `gs_loss` có thể in `-0` khi batch không có positive nào (loss = 0 đúng nghĩa; dấu âm
chỉ là chi tiết số học dấu phẩy động). Với checkpoint B0 đã train, `pos_rate` phải > 0 ở phần
lớn epoch.

## 7. Rà soát lại (2026-11-11, sau câu hỏi "chắc chắn chưa?") — 6 phát hiện

| # | Phát hiện | Xử lý |
|---|---|---|
| 1 | **Nghi ngờ padding**: suy luận ban đầu cho rằng `rect=False` ⇒ pad 64px top ⇒ head lấy sai chỗ 16 cells | **Kiểm bằng forward pass thật**: P2 = (32,128,160) ⇒ KHÔNG padding, tọa độ = tile/4 đúng. Không phải bug. Nguyên nhân: `auto=True` ⇒ padding = `dw mod stride` = 0 |
| 2 | **BUG THẬT — thiếu NMS**: `predict_tiled` cũ không áp NMS, trong khi baseline tiled có NMS 0.5 ⇒ so sánh không công bằng | Đã sửa: port `batched_nms` từ script eval gốc, áp lên (raw + extra) sau khi ghép tile; đối chiếu bit-for-bit với bản gốc trên 5 bộ box ngẫu nhiên — identical |
| 3 | **BUG THẬT — sai split**: script dùng `images/val` (84 ảnh) trong khi baseline trên `images/test` (786 ảnh) | Đã sửa: `--split test` mặc định; xác nhận train/val/test **rời nhau hoàn toàn** (0 giao nhau) ⇒ không leakage |
| 4 | Head-off baseline không tạo được bằng script cũ | Đã thêm `--head none` cho `--eval-only` |
| 5 | **BUG THẬT — ghi đè file dự đoán**: cả 3 run dùng chung `val_predictions.json` ⇒ run sau ghi đè run trước, mất baseline | Đã sửa: tên file suy từ `--out` (`G4_predictions.json`, `B0_predictions.json`, `off_predictions.json`) |
| 6 | **MỐC AP SAI — 0.3540 là checkpoint cũ**: tái lập pipeline ngày 2026-11-11 cho head-off 786 ảnh = **0.37556**, khớp `reports/eval/baseline_tinyperson_wacv_150e_tiled.json` (0.37568). Cặp 0.1886/0.3540 (log R6.a) gồm model GauSS (0.1886) và baseline checkpoint cũ (0.3540) — không cùng model với 150e | Đã sửa: mốc so sánh = **0.3757**; ghi cảnh báo ở §3; sửa PAPER_OUTLINE và PREREG |

Kiểm tra thêm đã làm: ở init, hộp "thứ hai" là **bản sao chính xác** của hộp nguồn
(offset 0, ratio 1) — đã verify bằng forward. Nghĩa là nếu head học mở gate nhưng không
học dịch chuyển, ta sẽ có trùng lặp thuần ⇒ guard G3 (detection/GT ≤ +20%) là lưới bắt
đúng cho mode thất bại này; ngoài ra nên báo cáo thêm tỉ lệ hộp-thứ-hai có IoU>0.9 với hộp
nguồn như một diagnostic.

Không phát hiện leakage: target sinh từ GT của **train split**, eval trên **test split**,
hai tập rời nhau 100%.
