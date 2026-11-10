
## ADR-005: Thêm thư viện Seaborn cho trực quan hóa báo cáo học thuật

- **Trạng thái**: Chấp thuận
- **Lý do**: Cần tạo các biểu đồ phân tích thực nghiệm học thuật (AP theo size-bin, đa ngưỡng IoU, so sánh quốc tế, ma trận per-class VisDrone) ở định dạng vector SVG sắc nét cho báo cáo tổng kết.
- **Phương án đã xem**: Matplotlib thuần túy, Plotly, Seaborn.
- **Lựa chọn**: Seaborn 0.13.2 (dựa trên Matplotlib, cú pháp thống kê rõ ràng, style chuẩn báo cáo paper).
- **License**: BSD-3-Clause (hoàn toàn tự do sử dụng).

## ADR-006: Cờ ablation GauSS qua biến môi trường TINYDET_ABLATION

- **Trạng thái**: Chấp thuận
- **Lý do**: Cần 5 ablation (bỏ band cao/giữa/thấp, bỏ router, bỏ saliency gate) mà không viết lại module, không đổi số tham số, giữ checkpoint tương thích. Env var cho phép bật cờ per-run qua Modal không cần sửa model YAML.
- **Phương án đã xem**: (1) env var đọc trong GauSS.__init__/forward; (2) 5 file YAML riêng với 5 class GauSS biến thể; (3) YAML flag truyền qua parse_model.
- **Lựa chọn**: (1) env var — ít xâm nhập nhất, không đổi kiến trúc, mọi run dùng chung model YAML gốc.
- **Rủi ro**: env var phải set trên Modal function; quên set = chạy full model (an toàn: mặc định là full).

## ADR-007: Dọn dẹp cấu trúc repo và chuẩn hóa tên file (2026-11-06)

- **Trạng thái**: Chấp thuận
- **Lý do**: Repo trộn tên tiếng Việt không dấu và tiếng Anh; nhiều tài liệu trỏ tới file đã bị xóa ở commit `5eeedde` (module QAC); CI lint một file không tồn tại (`src/tinydet/modules/msr.py`); hai cặp báo cáo trùng byte-identical; 133 MB JSON prediction bị track trong git.
- **Phương án đã xem**: (1) giữ nguyên và chỉ thêm banner "lỗi thời"; (2) chuyển tài liệu cũ vào `docs/archive/`; (3) xóa hẳn và chuẩn hóa tên.
- **Lựa chọn**: (3) theo quyết định của người dùng — xóa hẳn hướng QAC/QAR, chuyển toàn bộ tên tài liệu sang tiếng Anh snake_case, cập nhật mọi link tham chiếu.
- **Phạm vi**: chỉ tài liệu, cấu hình git và CI. Không sửa mã nguồn (`src/`, `scripts/`, `modal/`).
- **Xóa kèm**: `.superpowers/` (ledger phiên trước), `.history/` (backup VS Code, untracked), `reports/geometry/tiny_A2_p2_qac_v2*/` (artifact QAC v2) — theo xác nhận của người dùng.
- **Rủi ro**: lịch sử QAC chỉ còn trong git history và `reports/experiment_results_report.md`; nếu cần tái lập QAC phải checkout commit cũ.

## ADR-008: Sửa lỗi lọc lớp trong evaluator TinyPerson (`class_mode`)

- **Trạng thái**: Chấp thuận
- **Ngày**: 2026-11-06
- **Lý do**: `evaluate_tinyperson` bỏ toàn bộ dự đoán có `class_id != 0`
  (`tinyperson_benchmark.py`, vòng lặp dựng `detections`), trong khi `_load_gt`
  vẫn giữ ground truth của **cả hai** category (`sea_person` = 1,
  `earth_person` = 2). Hệ quả: mẫu số recall chứa cả 8.748 hộp `earth_person`
  nhưng không có dự đoán nào cho chúng, nên AP tuyệt đối bị nén xuống khoảng
  19 pp. Đây là lỗi pipeline, không phải "protocol khắt khe hơn" như bản thảo cũ
  lập luận.
- **Phương án đã xem**:
  1. Gộp hai lớp thành một lớp duy nhất trước khi khớp (`merge`).
  2. Giữ nguyên hành vi cũ và chỉ ghi chú trong bài (`primary`).
  3. Khớp theo từng category đúng chuẩn COCO/TOV, đồng thời trả về cả bản gộp
     (`per_class`).
- **Lựa chọn**: (3) làm mặc định, giữ (1) và (2) làm tuỳ chọn để tái lập số cũ.
  `merge` che mất cả phần thắng lẫn phần thua; `per_class` vừa đúng chuẩn vừa
  kể được câu chuyện có phạm vi trung thực.
- **API**: thêm tham số `class_mode: str = "per_class"` vào
  `evaluate_tinyperson()`; giá trị hợp lệ nằm trong `CLASS_MODES`. Mọi caller
  hiện có đều truyền tham số vị trí nên không phải sửa.
- **Kiểm chứng hồi quy**: `class_mode="primary"` tái lập chính xác số đã công bố
  — GauSS 12.9360 / 5.1776 / 9.4962, Baseline 11.9538 / 4.7164 / 6.9036
  (all / tiny1 / tiny2). Đây là cổng chặn: nếu lệch, phép sửa đã làm hỏng logic
  cũ.
- **Rủi ro**: số liệu trong `paper/main.tex` thay đổi ở mọi bảng dùng AP tuyệt
  đối. Bảng legacy được giữ lại nhưng gắn nhãn rõ là sai và không nên trích dẫn.
- **Ảnh hưởng**: `reports/eval/*_per_class.json` (4 file mới),
  `reports/eval/ablation/*/summary_per_class.json`.

## ADR-009: Một lớp `person` và derivative `tinyperson-yolo-v2` là nguồn train chuẩn

- **Trạng thái**: Chấp thuận
- **Ngày**: 2026-11-07
- **Lý do**: benchmark TinyPerson chính thức gộp `sea_person` + `earth_person` khi chấm
  (evaluator dùng `class_mode="merge"`), và mọi con số so sánh được với tài liệu
  (BPD-YOLO 38–39 %) đều ở chế độ một lớp. Derivative cũ trên Modal có
  `val: images/train` nên `best.pt` không có ý nghĩa, còn derivative parquet local lại
  đặt `val` = chính tập test.
- **Phương án đã xem**:
  1. Giữ 2 lớp và chấm `per_class` — trung thực về mặt phân loại nhưng không so sánh được
     với protocol công bố, và làm mẫu số recall khác đi.
  2. Giữ 2 lớp nhưng chấm `merge` — lãng phí một đầu ra, vẫn cần val thật.
  3. Một lớp `person` + eval `merge` (chọn).
- **Lựa chọn**: (3). Nguồn ảnh là `erase_with_uncertain_dataset` vì tác giả TinyPerson
  khuyến nghị tô vùng ignore để chúng không trở thành negative ngầm. `val` (84 ảnh) tách
  theo nhóm video từ train; `test` (786 ảnh) nằm trong split riêng và không bao giờ dùng
  để chọn checkpoint.
- **API**: `scripts/prepare_tinyperson_yolo.py` thêm `--test-annotations` /
  `--test-image-root`; `scripts/convert_parquet_to_yolo.py` thêm `--val-fraction` /
  `--seed` và sinh 3 split. `evaluate_tinyperson` từ chối `per_class` khi model chỉ phát
  một lớp trên annotation 2 category.
- **Kiểm chứng**: derivative mới có 662/84/786 ảnh, 14 825/3 608/13 687 box, 100 % label
  là class `0`, giao theo basename giữa 3 split bằng 0.
- **Rủi ro**: mọi số liệu cũ trong `paper/main.tex` phải được đo lại; không được trộn
  kết quả từ derivative cũ với derivative mới.

## ADR-010: Thay thế GauSS v2.1 bằng P2Refine và dọn dẹp chuỗi layer chết

- **Trạng thái**: Chấp thuận
- **Ngày**: 2026-11-08
- **Lý do**:
  1. Phân tích thực nghiệm và lý thuyết cho thấy GauSS v2.1 gặp hiện tượng classification-head desync
     ngay từ step 0 (đầu phân loại khởi tạo cho $x_0$, nhưng nhận $y = x_0 + \alpha z$ với năng lượng lệch lớn).
  2. GauSS v2.1 trong YAML cũ có chuỗi layers 17→18→19 (Upsample từ 16 → Concat với 2 → C3k2) tạo ra output
     hoàn toàn không được layer nào tiêu thụ (dead branch lãng phí ~0.87 GMAC và 8.7k tham số).
  3. Cả 5 giả thiết cơ chế của GauSS v2.1 (saliency router, scale-space band, dilated context)
     đều đã bị bác bỏ bởi ablation bootstrap trước đó.
- **Phương án đã xem**:
  1. Sửa GauSS v2.1 với đề xuất N4' (zero-init output projection, bỏ double-SiLU) + giữ nguyên kiến trúc.
  2. Bỏ hoàn toàn module tinh chế, dùng baseline A2 (không có refine ở P2).
  3. Tích hợp `P2Refine`: module tinh chế tối giản, ReZero-style (zero-init exit projection không BN),
     chỉ lấy input [C2, P3], đảm bảo đồng nhất bit-exact tại step 0, đồng thời dọn sạch chuỗi layer chết (chọn).
- **Lựa chọn**: (3).
  - Kiến trúc `P2Refine`: Stem kết hợp C2 + Up(P3) qua Conv 1x1 + BN + SiLU. Nhánh residual gồm DW3x3 → PW (chuyển sang mid=16) → DW5x5 → W_out (zero-init, không BN). Output = stem + W_out(h).
  - Tổng tham số module: 6,000 (so với 34,700 của GauSS).
  - MACs module: 0.4375 GMAC (so với 1.0176 GMAC của GauSS).
  - Mô hình đầy đủ: 2,513,272 tham số, 31.067 GFLOPs ở 1280x1280 (giảm so với 32.2 GFLOPs của GauSS).
- **Hệ thống theo dõi & Diagnostics**:
  - Tạo `P2RefineDiagnosticsCallback` trong `src/tinydet/training/module_diagnostics.py` tự động theo dõi
    và ghi lại các chỉ số `rho_mean`, `rho_max`, `cos_x0_delta`, `w_out_norm` vào `metrics/p2refine_diagnostics.csv`
    và đồng bộ lên W&B mỗi epoch (khắc phục điểm thiếu sót của GauSS khi 21 run trước đó không có log module).
- **Kiểm chứng**:
  - Module self-test trên CPU: `rel=0.000000 exact=True` ở cả train và eval mode.
  - Tích hợp model YAML: `scripts/validate_tiny_ablation.py` pass với strides [4, 8, 16, 32], shape detection heads chuẩn.
  - Tự động kiểm tra bit-exact identity at step 0 trong mock trainer lifecycle.
  - Ruff lint clean 100%.

## ADR-011: `train.py` làm cổng vào chung cho mọi GPU host, vẫn giữ cổng chặn CPU

- **Trạng thái**: Chấp thuận
- **Ngày**: 2026-11-08
- **Lý do**: Trước đây đường vào huấn luyện duy nhất là `modal run modal/app.py::...`.
  Repo không có `train.py`, nên không thể dùng Colab/Kaggle/RunPod dù phần lõi
  (`src/tinydet/**`) hoàn toàn không hardcode `/mnt`. Yêu cầu của người dùng: train
  được trên mọi môi trường có GPU, nhưng tuyệt đối không train trên CPU (local/laptop).
- **Phương án đã xem**:
  1. Chuyển project sang `.ipynb` — bị loại: mất version control, khó debug, sửa module
     phải sửa trong cell.
  2. Sửa `configs/experiment/*.yaml` bỏ `/mnt/*` — bị loại: phá các run Modal đang dùng.
  3. Thêm `train.py` ở root, override bằng CLI, giữ nguyên config gốc (chọn).
- **Lựa chọn**: (3).
  - `train.py` là thin wrapper: `argparse` → `dataclasses.replace()` → ghi config tạm
    → `tinydet.training.trainer.train()`. Không nhân đôi logic huấn luyện.
  - Cờ: `--data`, `--data-root`, `--output-root`, `--name`, `--device`, `--epochs`,
    `--batch`, `--imgsz`, `--workers`, `--seed`, `--no-wandb`, `--dry-run`.
  - `--dry-run` chạy được trên CPU: resolve config + dataset + model, in params/GFLOPs/
    strides, không train. Dùng để kiểm tra trước khi tốn GPU.
- **Cổng chặn CPU giữ nguyên** (đúng yêu cầu):
  - `train.py` trả `exit 1` khi `torch.cuda.is_available()` là False.
  - `trainer.train()` vẫn raise `RuntimeError`; thông điệp đổi từ "requires Modal GPU"
    thành "requires a CUDA GPU" cho trung thực.
  - `torch.cuda.reset_peak_memory_stats()` và `max_memory_allocated()` được guard để
    không crash nếu cổng bị nới trong tương lai.
- **API phụ**: `normalize_dataset_yaml(path, root=None)` nhận `root` để trỏ `path` trong
  `data.yaml` sang bản copy dữ liệu trên host khác, không cần sửa YAML được track.
  Cache key dùng `path` đã resolve nên đường dẫn tương đối và tuyệt đối dùng chung cache.
- **Môi trường trên host khác**: dùng `uv sync --extra gpu`, không dùng `pip install`.
  `uv` tự tải CPython 3.12.13 (thỏa `requires-python = ">=3.12,<3.13"`, trong khi Colab
  và Kaggle mặc định 3.11) và cài `torch==2.11.0+cu128` từ index `pytorch-cu128` khai báo
  trong `pyproject.toml`, khớp `uv.lock`. `pip install -e .` vừa fail vì Python 3.11, vừa
  có nguy cơ cài torch CPU đè lên torch CUDA có sẵn.
- **Sửa kèm theo**: `wandb>=0.21,<1` được thêm vào `[project.dependencies]`. Trước đó
  `wandb` không có trong `pyproject.toml` lẫn `uv.lock`, nên mọi run dùng `uv sync` đều
  im lặng tắt W&B (`_init_wandb` bắt `ImportError`). Trên Modal lỗi này bị che vì
  `modal/image.py:8` cài `wandb` riêng.
- **Kiểm chứng**:
  - `train.py --dry-run` trên CPU: 320 layers, 2,513,272 params, 31.067 GFLOPs,
    strides `[4, 8, 16, 32]`, 84 ảnh val — khớp số đo của `validate_tiny_ablation.py`.
  - Chạy từ `/tmp` với đường dẫn tuyệt đối: pass (hết phụ thuộc CWD).
  - `train.py` không có `--dry-run`: `exit 1` kèm thông điệp hướng dẫn.
  - `trainer.train()` gọi trực tiếp trên CPU: raise `RuntimeError` đúng.
  - `normalize_dataset_yaml`: `root=None` giữ hành vi cũ; `root` tương đối == tuyệt đối;
    `root` không tồn tại → `FileNotFoundError`.
  - `uv sync --extra gpu` (thử trong thư mục sạch): `Using CPython 3.12.13`, tải
    `torch 782.3MiB` + CUDA libs, resolve 198 packages.
  - `uv run --extra cpu python -c "import wandb"` → `0.31.0` sau khi thêm vào lock.
  - `scripts/validate_tiny_ablation.py` không tham số: pass cả 3 model YAML còn tồn tại.
  - `ruff check` sạch trên mọi file đã sửa.
- **Chưa xác minh**: hành vi `resume=True` của Ultralytics trên checkpoint thật, và việc
  train thực sự chạy trên Colab/Kaggle. Cần một run GPU thật để xác nhận.
- **Rủi ro**: T4 15 GB có thể OOM ở `imgsz 1280, batch 8`; dùng `--batch 4`. Kết quả lệch
  theo GPU khác nhau; `metadata.json` đã log `gpu` và phải ghi GPU nào trong paper.

## ADR-012: Thay GauSS v3 bằng GauSS v3.1 (2026-11-09)

- **Bối cảnh**: Rà soát cuối cho thấy GauSS v3 còn ba điểm yếu kỹ thuật: (1) cổng ReZero
  dùng `0.5 * sigmoid(raw)` với `raw = -6.0` nên `alpha ≈ 0.0012` chứ không bằng 0, đồng
  thời đạo hàm tại init bị bão hòa (~0.0012) làm module học rất chậm trong các epoch đầu;
  (2) hai phép `F.interpolate(mode="nearest")` (P3→P2, P4→P3) gây aliasing đúng tại chỗ
  giao dải cao/thấp — nơi thông tin của người 2–8 px nằm; (3) cờ `no_high_band` chỉ triệt
  tiêu `r_h` mà vẫn để lọt nhánh neo trực tiếp `gamma_h * b_h`, làm nhiễm bẩn ablation.
- **Quyết định**: Thay toàn bộ `src/tinydet/modules/gauss.py` bằng bản v3.1. Giữ nguyên
  tên class (`GauSS`/`GauSS`), chữ ký constructor `(c1, c2, ...)` và forward `[C2, P3, P4]`,
  nên YAML, `parse_model` patch và trainer không phải sửa.
- **Thay đổi chính**:
  - Cổng ReZero: `0.5 * tanh(raw)`, `raw = 0` → `alpha = gamma_in = gamma_out = 0` chính xác
    bitwise, đạo hàm tại init 0.5 (gấp ~400 lần bản sigmoid).
  - `DySampleUp` (offset zero-init) thay `nearest`; tại init bằng đúng
    `F.interpolate(mode="bilinear", align_corners=False)` (sai số < 1e-5, có test).
  - Router hybrid: `softmax(log(energy) + corr)` với `corr` zero-init, năng lượng đo trên
    band thô `b_h/b_m/b_l` ở stride-4 rồi avg-pool về stride-8; kèm `router_z_loss()`
    (ST-MoE, Eq. 5) và helper `collect_gauss_router_zloss(model)`.
  - Tách `gamma_in` (neo tinh chế stride-8) và `gamma_out` (neo trực tiếp stride-4).
  - Ablation sạch: `no_high_band` khóa cả hai nhánh dải cao, thêm cờ `no_direct_anchor`,
    trọng số band được chuẩn hóa lại trên các band còn hoạt động.
  - Dilation context `{1, 3, 5}` (ERF stride-8 từ 7×7 lên 11×11).
  - Toán định tuyến chạy FP32 để an toàn AMP.
- **Phương án đã xem**: (1) giữ v3 và vá từng điểm — để lại kiến trúc nửa vời, ablation vẫn
  nhiễm; (2) viết module mới thay thế — trùng lặp nghiệp vụ và tốn công đổi YAML/checkpoint;
  (3) chọn v3.1 vì là drop-in replacement, không thêm dependency.
- **Hệ quả**: 36,878 tham số (+2,212), 27 Conv2d (+3), 131 control scalar (+1),
  1.0413 GMAC (+2.42% module, ~+0.14% forward YOLO26n ở 1280). Checkpoint v3 phải chuyển
  bằng `remap_v3_to_v31_state_dict()` và load `strict=False`.
- **Kiểm chứng**: `uv run --extra cpu python src/tinydet/modules/gauss.py` → PASS
  (params, convs, control scalars, MACs, ReZero identity bitwise, DySample == bilinear,
  ablation không rò rỉ, remap checkpoint). Load + forward YOLO26n-P2 qua
  `configs/model/yolo26n_tiny_A2_p2_gauss_tinyperson.yaml`: 2,552,854 tham số, output
  `one2many`/`one2one` đúng. `ruff check src/` sạch.
- **Chưa xác minh**: chưa train GPU thật với v3.1; chưa nối `router_z_loss` vào hàm loss của
  trainer (tùy chọn); DDP nhiều GPU cần `find_unused_parameters=True` hoặc dùng
  `module.unused_param_names()` vì `route_*`/`saliency_gate` không nhận gradient ở chế độ
  mặc định.

## ADR-013: Thống nhất tên GauSS, xóa toàn bộ định danh DCAF (2026-11-09)

- **Bối cảnh**: Module trải qua nhiều tên gọi (DCAF v2.1, DCAF/GauSS v3, GauSS v3.1) nằm rải rác
  trong code, config, docs, script và artifact, gây nhầm lẫn khi viết báo cáo và khi đọc log.
- **Quyết định**: `GauSS` là tên duy nhất. Xóa class alias `DCAF`, đổi toàn bộ helper
  (`is_gauss_control_param_name`, `collect_gauss_control_params`,
  `configure_gauss_control_param_group`, `compute_gauss_analytical_macs`,
  `collect_gauss_router_zloss`), đổi tên file artifact trong repo (`reports/gauss_*`,
  `docs/report/GAUSS_BASELINE_HANDOFF.md`, `reports/gauss_autonomous/`), và thay chuỗi
  `DCAF`/`dcaf` trong toàn bộ văn bản tracked. Tên viết đầy đủ thống nhất là
  "Gaussian Scale-Space Feature Alignment and Refinement".
- **Phương án đã xem**: (1) giữ alias `DCAF = GauSS` để load checkpoint cũ — để lại định danh cũ
  trong code, trái yêu cầu; (2) đổi tên nhưng giữ nguyên tên artifact lịch sử — còn chuỗi `dcaf`
  trong repo. Chọn phương án đổi toàn bộ, ghi hệ quả vào `docs/BACKLOG.md` (BL-018, BL-019).
- **Hệ quả**: checkpoint cũ pickle class `DCAF` không unpickle trực tiếp được (BL-018); các
  đường dẫn trỏ volume Modal cũ (`dcaf_v2_1_*`) không còn khớp cho tới khi có run mới (BL-019).
- **Kiểm chứng**: `git grep -I -i dcaf` trên file tracked trả về rỗng (trừ `uv.lock`, là hash
  package không liên quan và không được sửa tay). Self-test `gauss.py` PASS, forward YOLO26n-P2
  với `configs/model/yolo26n_tiny_A2_p2_gauss_tinyperson.yaml` OK, `ruff check src/` sạch.
- **Chưa xác minh**: chưa load thử một checkpoint v2.1/v3 thật sau khi đổi tên.

## ADR-014: Đồng bộ TAL assigner cho track VisDrone (2026-11-09)

- **Bối cảnh**: Track TinyPerson đã đặt `tal_beta: 4.0` và bật `tal_tiny_floor: 0.5`
  (ưu tiên anchor cho vật thể < 8 px). Track VisDrone vẫn giữ `tal_beta: 6.0` mặc định và
  không có floor, dù VisDrone cũng có mật độ vật thể nhỏ rất cao (pedestrian/people/motor).
  Nếu hai track dùng assigner khác nhau, mọi so sánh chéo dataset trong báo cáo đều bị nhiễu.
- **Quyết định**: Đồng bộ `tal_beta: 4.0` + `tal_tiny_floor: 0.5` cho cả hai model VisDrone
  (`yolo26n_tiny_A2_p2_visdrone.yaml`, `yolo26n_tiny_A2_p2_gauss_visdrone.yaml`).
- **Phương án đã xem**: (1) giữ 6.0 cho VisDrone và ghi rõ mỗi track tự tune — trung thực
  nhưng làm yếu claim "module cải thiện cả hai domain"; (2) đồng bộ theo TinyPerson (chọn).
- **Hệ quả**: Track VisDrone phải train lại từ đầu (chưa có run VisDrone nào với cấu hình này
  trong repo). Baseline VisDrone và GauSS VisDrone dùng chung assigner, nên ΔAP quy được cho
  module. Hai track vẫn tách biệt: không trộn số liệu TinyPerson với VisDrone.
- **Kiểm chứng**: `yaml.safe_load` xác nhận cả hai model YAML có `tal_beta: 4.0`,
  `tal_tiny_floor: 0.5`, `nc: 10`; build + forward YOLO thành công (baseline 2,518,352 params;
  GauSS 2,555,230 params; 4 feature levels mỗi head). So khớp experiment config:
  `baseline_visdrone_tpami` và `gauss_v3_visdrone_tpami` chỉ khác `name` và `model.config`.
  `ruff check src/ configs/` sạch.
- **Chưa xác minh**: chưa có run VisDrone thật với assigner mới; ảnh hưởng của floor 0.5 lên
  10 lớp VisDrone (nhiều lớp trung bình/lớn) chưa được đo.

## ADR-015: Khai báo `polars` là direct dependency (2026-11-09)

- **Bối cảnh**: `scripts/convert_parquet_to_yolo.py` import trực tiếp `polars` để đọc parquet
  Hugging Face và ghi YOLO labels, nhưng `pyproject.toml` không khai báo. `uv tree` xác nhận
  `polars 1.44.2` chỉ tồn tại trong lock nhờ là dependency bậc ba của `ultralytics`. Nếu
  ultralytics bỏ polars ở bản sau, script vỡ mà không có cảnh báo nào.
- **Quyết định**: khai báo `polars>=1.44,<2` trong `[project].dependencies`.
- **Phương án đã xem**: (1) chuyển script về pandas (đã khai báo) — mất engine Rust đa nhân,
  chậm hơn trên parquet lớn và không cần thiết; (2) giữ import ngầm qua ultralytics — phụ thuộc
  bậc ba không được bảo đảm; (3) khai báo tường minh (chọn).
- **Hệ quả**: thêm một direct dependency nhưng **không tăng kích thước cài đặt** vì polars đã
  có sẵn trong lock/environment. Không thay đổi bất kỳ số liệu nào.
- **Kiểm chứng**: `uv tree` cho thấy polars 1.44.2 nằm dưới ultralytics; `uv add polars` cập nhật
  `pyproject.toml` + lockfile bằng công cụ (không sửa tay); `python -c "import polars"` OK.
- **Chưa xác minh**: chưa chạy lại `convert_parquet_to_yolo.py` trên parquet thật sau khi khai báo
  (hành vi import không đổi).

## ADR-016: Nâng cấp dependency lên phiên bản mới nhất (2026-11-09)

- **Bối cảnh**: audit phiên bản cho thấy Modal image dùng index `cu128` chỉ có torch tới
  **2.11.0** (lệch 3 minor so với local 2.14.0), và một số constraint chặn phiên bản mới:
  `pandas<3` (chặn 3.1.1), `polars<2` (chặn 2.1.0), `Pillow<13` (chặn 13.0.0),
  `mypy<2` (chặn 2.4.1), `ultralytics==8.4.164` (mới nhất 8.4.208).
- **Quyết định**:
  - Local (CPU): giữ `torch 2.14.0+cpu + torchvision 0.29.0+cpu` — đã xác minh
    `torchvision 0.29.0+cpu` **pin chặt `torch==2.14.0`** và torch 2.15 chưa có bản
    torchvision `+cpu`, nên đây là cặp mới nhất khả dĩ trên index pytorch-cpu.
  - Modal (GPU): đổi index `cu128` → **`cu132`** (Modal host driver 610.57.04, CUDA driver
    API 13.3; A10 = Ampere sm_86 vẫn được CUDA 13 hỗ trợ) → resolve **torch 2.15.0+cu132 +
    torchvision 0.29.2+cu132** (uv resolver xác minh tương thích). Pin `ultralytics==8.4.208`,
    numpy>=2.5, pandas>=3.1, Pillow>=13, polars>=2.1, thêm `polars` vào image (script
    convert được mount vào container).
  - Local: nâng ultralytics 8.4.208, numpy 2.5.4, pandas 3.1.1, polars 2.1.0, modal 1.6.3,
    pillow 13.0.0, ruff 0.17.4, mypy 2.4.1.
  - **Giữ opencv-python-headless 4.14** (không lên 5.0): cv2 5.0 rất mới, ultralytics chỉ yêu
    cầu >=4.7 nhưng chưa được upstream kiểm chứng với cv2 5; lợi ích không đáng rủi ro.
- **Phương án đã xem**: (1) giữ cu128 — torch cũ 3 minor, lệch local; (2) cu130 — torch
  2.14.1, gần local nhưng vẫn không phải mới nhất; (3) cu132 (chọn) — khớp torch 2.15 mới nhất.
- **Hệ quả**: lần build image Modal tiếp theo sẽ cài bộ mới; các checkpoint cũ vẫn load được
  vì ultralytics giữ API trainer/validator/callbacks (đã grep surface). Local torch không đổi
  nên checkpoint local không bị ảnh hưởng.
- **Kiểm chứng**: sau nâng cấp — forward YOLO26n-P2+GauSS PASS; ruff sạch; gauss self-test
  PASS; toàn bộ 7 stress suite + 4 equivalence test PASS; benchmark eval không đổi (12.65 s
  vs 12.52 s, trong nhiễu); pandas 3 + seaborn plot PASS; polars 2 parquet PASS; pillow 13
  exif PASS; mypy 2.4.1 trên 3 module mới sạch; `modal profile current` OK.
- **Chưa xác minh**: chưa build/run image Modal mới thật (cần `modal run` thật); chưa train
  trên torch 2.15 GPU (AMP/BN hành vi có thể khác nhẹ — theo dõi epoch đầu).

## ADR-017: Nâng Python 3.12 → 3.14 (local + Modal image) (2026-11-09)

- **Bối cảnh**: `requires-python = ">=3.12,<3.13"` và `debian_slim(python_version="3.12")`
  ghim cứng 3.12, trong khi máy local đã có CPython 3.14.9 mặc định. Yêu cầu: dùng bản mới nhất
  mà không xung đột package.
- **Kiểm chứng trước khi đổi** (uv resolver thật, không suy đoán):
  - 3.13 và 3.14 đều resolve sạch: `torch 2.15.0+cu132 + torchvision 0.29.2+cu132` (GPU),
    `torch 2.15.0+cpu + torchvision 0.29.2+cpu` (CPU), ultralytics 8.4.208, numpy 2.5.4,
    pandas 3.1.1, polars 2.1.0.
  - Modal client 1.6.3: `SUPPORTED_PYTHON_SERIES` của builder mặc định `2026.10` có 3.14
    (và 3.13/3.15); builder `2024.10` trở về trước thì không.
- **Quyết định**: nâng lên **3.14** ở cả 6 chỗ ghim:
  `pyproject.toml` (`requires-python`, `ruff target-version = py314`), `modal/image.py`,
  `.github/workflows/cpu-checks.yml`, `docs/agents/CODE.md`, `docs/agents/OPERATIONS.md`,
  `AGENTS.md`. Thêm `.python-version` = `3.14` (do `uv python pin`).
- **Không sửa**: `outputs/*/environment.txt`, `backups/*/environment.txt`,
  `reports/*.md` — đây là bản ghi lịch sử của các run đã chạy trên 3.12, sửa sẽ làm sai
  provenance.
- **Hệ quả**: lockfile re-resolve cho marker 3.14 (216 package). Local giữ
  `torch 2.14.0+cpu` (uv giữ phiên bản đã lock vì vẫn tương thích 3.14; wheel cp314 có sẵn),
  không phải 2.15 như GPU.
- **Kiểm chứng sau khi đổi** (đã chạy, output thật): `python -V` = 3.14.9; import
  torch/torchvision/numpy/pandas/ultralytics/polars OK; `ruff check src scripts modal
  launch_train_and_eval_60e.py` → "All checks passed!"; GauSS self-test PASS
  (params=36,878, MACs=1.0413 GMAC, ReZero identity exact); 7/7 stress suite
  "FAILURES: none"; fork(7) == in-process PASS; benchmark eval vectorised 5.34 s vs
  scalar 28.08 s (5.3x).
- **Chưa xác minh**: chưa build/run image Modal với Python 3.14 thật (cần `modal run`);
  chưa train GPU trên 3.14; chưa chạy CI GitHub với 3.14.

## ADR-018: Thêm albumentations 2.0.8 + preset augmentation `tinyperson_b` (2026-11-09)

- **Bối cảnh**: run 60e seed42 cho thấy MR50 ~94% (recall là nút thắt chính) và recipe
  augmentation gần như trống (mosaic/mixup/copy_paste/multi_scale = 0). Cần tăng đa dạng
  dữ liệu mà không phá scale tiny object.
- **Phương án đã xem**:
  1. `CopyPaste` của Ultralytics — loại: paste bằng `cv2.drawContours(instances2.segments)`
     (data/augment.py:2049); dataset chỉ có box → segments rỗng → no-op âm thầm.
  2. `A.CopyAndPaste` của Albumentations — loại: KHÔNG tồn tại trong release 2.0.8
     (chỉ ở master; `dir(A)` filter copy/paste = [], đã chạy thật). Thông tin web về
     bbox-only donor là docs bản chưa phát hành.
  3. `A.OverlayElements` — loại: `NotImplementedError: BBoxes not implemented` (test thật).
  4. `A.Mosaic` — loại: phá scale tiny (mosaic đã chủ động tắt từ trước).
  5. ✅ **Albumentations 2.0.8 với preset transform tự chọn** — mọi transform đã test
     với bbox format yolo + đo tốc độ tại 1280×1280.
- **Kích thước / license**: albumentations 2.0.8 wheel 369 KB (Apache-2.0), kéo thêm
  `simsimd` + `stringzilla` (cùng Apache-2.0); `opencv-python 5.0.0.93` đã có sẵn trong
  lock (giới hạn `opencv-python-headless>=4.10,<5` của ta không đổi, headless vẫn 4.14).
- **Quyết định**:
  - Thêm `albumentations>=2.0.8,<3` vào `pyproject.toml`.
  - Module mới `src/tinydet/augment/albumentations_stack.py` với preset
    `tinyperson_b`: Downscale 0.25 (scale_range 0.4–0.7), MotionBlur 0.15,
    ImageCompression 0.10 (q 30–70), RandomBrightnessContrast 0.20, CLAHE 0.10,
    RandomSizedBBoxSafeCrop 0.20 (1280×1280, erosion 0), CoarseDropout 0.10.
  - Trainer mở rộng `train.augmentations_preset` → `augmentations` trước khi gọi
    `model.train()` (key preset bị pop, không rò vào Ultralytics cfg).
  - Bật cho **cả baseline và GauSS** (cùng recipe để so sánh công bằng, không confound).
  - Không làm ablation "không augment" (quyết định của user, 09/11/2026).
- **Tại sao các transform này**: mô phỏng đúng domain thu nhận TinyPerson (mất độ phân
  giải, rung camera, nén JPEG, chiếu sáng) + crop-safe zoom tăng hiệu dụng kích thước
  object 2–8px + occlusion. Tổng overhead ~19 ms/ảnh @1280 CPU (đo thật), không đáng kể
  so với ~25 s/epoch GPU.
- **Lưu ý tích hợp đã xác minh trong source**:
  - Hook chính thức: `build_transforms` → `Albumentations(p=1.0, transforms=getattr(hyp,
    "augmentations", None))` (ultralytics/data/augment.py:2911); spatial transforms được
    wrap với `BboxParams(format="yolo", label_fields=["class_labels","idx"])`.
  - Vị trí: sau RandomPerspective(1280), trước RandomHSV/RandomFlip → bbox tự cập nhật;
    LetterBox KHÔNG nằm trong pipeline train (dataset.py:342) nên crop phải giữ shape —
    vì vậy dùng `RandomSizedBBoxSafeCrop(1280,1280)` (giữ 1280×1280, đã test) thay vì
    `BBoxSafeRandomCrop` (thay đổi shape → vỡ batch, đã test thấy 1077×1014).
  - Serialize: `A.from_dict` yêu cầu `__class_fullname__` KHÔNG có tiền tố `A.` và
    tuple viết dạng list (dict {min,max} bị pydantic từ chối) — đã test từng transform.
- **Kiểm chứng**: preset `from_dict` 7/7 OK; Compose chạy với bbox yolo giữ shape 1280;
  smoke test dataloader thật (`test_aug_dataloader.py`, CPU): YOLODataset 662 ảnh train +
  `hyp.augmentations` → pipeline chứa Albumentations với 7 transform, `contains_spatial=True`,
  3 sample đều giữ shape (3,1280,1280), labels còn nguyên, ảnh finite;
  `load_config` cả 2 YAML → preset resolve 7 transform; `ruff check` sạch.
- **Chưa xác minh**: tác động thực lên AP/MR (chưa train); hành vi trên GPU dataloader
  8 workers; tương tác với `mosaic=0` + `RandomPerspective` chưa được chạy qua 1 epoch thật.

## ADR-019: Campaign 150e (config mới) + tune siêu tham số sau rà soát 60e (2026-11-09)

- **Bối cảnh**: run 60e seed42 (`20261109-123602-4d5bf2f`) là **nội bộ**, KHÔNG đưa vào
  báo cáo khoa học cuối; chỉ kết quả nổi bật của vòng chính thức mới được báo cáo. Cần
  vòng train mới với config riêng, không sửa config 60e đã đóng băng, và không hardcode
  danh sách config trong script launch.
- **Rà soát 60e (số thật từ `metrics/` của cả 2 run)**:
  - `val/box_loss` đạt min ở e55 (baseline) và e54 (GauSS); `val/cls_loss` min ở e44/e57 →
    **chưa hội tụ**, kéo dài schedule là hợp lý hơn tăng LR.
  - `optim/amp_scale` = 256 không đổi suốt 60e, không NaN → AMP ổn định, không cần hạ lr0.
  - `optim/grad_norm` median 154–158 (max 660) nhưng loss hữu hạn → không có dấu hiệu nổ.
  - `system/vram_peak_gb` 10.08 (baseline) / 10.81 (GauSS) trên A10 24 GB → còn dư lớn.
  - `system/epoch_time_s` median 22.5 s (baseline) / 28.1 s (GauSS).
  - AP50(all) tốt nhất ở e56 (baseline 0.2069) / e58 (GauSS 0.2101) → dao động ±0.005
    giữa các epoch cuối ⇒ patience 15 quá ngắn cho schedule dài.
- **Quyết định**:
  1. Tạo config mới `configs/experiment/{baseline_tinyperson_wacv_150e,gauss_v3_tinyperson_wacv_150e}.yaml`;
     config 60e giữ nguyên (đóng băng).
  2. Tune (chỉ những gì có bằng chứng từ trên):
     - `epochs` 60 → **150** (val loss còn giảm).
     - `patience` 15 → **40** (dao động cuối schedule ±0.005 AP).
     - `batch` 8 → **16** (VRAM peak 10.8/24 GB; batch lớn cho assigner nhiều positive hơn
       với tiny object). Giữ `lr0 = 0.002` — không áp linear scaling để tránh bất ổn,
       vì AMP/grad_norm đang ổn định.
     - `warmup_epochs` 2.0 → **3.0** (schedule dài hơn).
     - Giữ nguyên: `imgsz 1280`, `AdamW`, `lrf 0.1`, `cos_lr`, `scale 0.15`,
       `translate 0.05`, `mosaic/mixup/cutmix/copy_paste = 0`, `multi_scale 0`.
  3. `augmentations_preset: tinyperson_b` (ADR-018) cho **cả hai** config.
  4. Bỏ hardcode: `launch_train_and_eval_60e.py` → **`launch_campaign.py`**, nhận
     `--configs a.yaml,b.yaml` (mặc định là cặp 150e); `_orchestrate_60e` →
     `_orchestrate_campaign`, tên eval suy ra từ config (`<run>_tiled`, `<run>_full`),
     tag epoch in ra lấy từ `train.epochs`.
- **Hệ quả**: run dir mới `*_150e/2026...-seed42`; eval dir `/eval/<run>_tiled|_full`
  (không còn hậu tố `_60e`). Số 60e chỉ dùng nội bộ để chọn siêu tham số.
- **Kiểm chứng**: `load_config` cả 2 config mới OK (epochs 150, batch 16, patience 40,
  preset tinyperson_b); `ruff check launch_campaign.py modal/app.py src/tinydet/augment
  src/tinydet/training/trainer.py` → All checks passed.
- **Chưa xác minh**: VRAM thực tế ở batch 16 (ước tính ~18–20 GB từ 10.8 GB @ batch 8);
  thời gian/epoch ở batch 16; tác động của batch 16 + augmentation mới lên AP; chưa chạy
  campaign 150e thật.

## ADR-020: Nâng RAM ceiling của container GPU lên 48 GB (2026-11-10)

- **Bối cảnh:** campaign 150e (app `ap-sTJpvV9GSDzzFYKOEq0PAW`) — cả baseline và gauss bị
  `Runner killed (SIGKILL), exit code: 137` ở ~epoch 131/150 với
  `MEMORY_LIMIT_MB = (12288, 24576)`. Modal retry `_train` hai lần, tạo run dir mới
  (`20261109-170302`, `20261109-171442`) và train lại từ epoch 1 — tức nếu không sửa thì
  crash lặp lại vô hạn. 60e batch 8 với cùng limit không gặp vấn đề.
- **Nguyên nhân:** RAM container tăng dần theo epoch (per-epoch val metrics + W&B media);
  batch 16 làm baseline cao hơn. Ước lượng từ điểm chết: ~185 MB/epoch.
- **Phương án đã xem:**
  1. Giảm `workers` 8 → 4: giảm RAM dataloader nhưng tăng thời gian/epoch, và không giải
     quyết được phần tăng theo epoch (vẫn chết nếu schedule dài hơn).
  2. Tắt W&B media / giảm tần suất log ảnh: có thể là nguồn tăng, nhưng chưa có bằng chứng
     (chưa đo RSS) — sửa mù sẽ mất tín hiệu debug.
  3. **Nâng ceiling 24576 → 49152 MB** (đã chọn): sửa trực tiếp nguyên nhân chết, chi phí
     thêm ~$0.19/h RAM (24 GB × $0.00000222/GiB/s), tức ~$0.5 cho cả campaign. Giữ nguyên
     mọi siêu tham số đã chốt ở ADR-019 nên kết quả vẫn so sánh được.
- **Hệ quả:** `MEMORY_LIMIT_MB = (24576, 49152)` cho mọi container GPU trong `modal/app.py`.
  Request 24 GB cũng tăng theo (Modal tính theo mức dùng thực, không theo request).
- **Kiểm chứng:** log 3h của app chứa `SIGKILL, exit code: 137` cho cả 2 run; run dir gốc
  không có `training/completed.flag` dù `metrics/system_metrics.csv` có đủ epoch;
  `ruff check modal/app.py` → All checks passed.
- **Chưa xác minh:** liệu 48 GB có đủ cho trọn 150 epoch (chưa chạy lại); nguồn tăng RAM
  thật sự (BL-028); chi phí thực tế trên dashboard Modal.

## ADR-021: Hoãn ablation 2×2 (augment × model); ghi nhận gap GauSS thu hẹp (2026-11-10)

- **Bối cảnh:** kết quả chính thức campaign 150e (`ap-yJXm5NjIoH73hanG7cjlBP`, commit
  `a20f7b6`) cho thấy GauSS vẫn thắng baseline nhưng cách biệt nhỏ hơn nhiều so với vòng
  60e nội bộ: AP@0.50 tiled all +0.0046 (+1.1%) vs +0.0221 (+7.3%); bucket tiny −0.0016
  (60e: +0.0337). Chi tiết: `GAP_SHRINK_ANALYSIS` (artifact) và BL-030.
- **Giả thuyết chính:** preset augmentation B bật cho cả hai nhánh đã cung cấp scale
  invariance bằng dữ liệu, trùng chức năng GauSS. Bằng chứng gián tiếp: baseline tăng
  +40.7% ở tiny (GauSS +24.4%); GauSS hội tụ nhanh hơn (90% fitness ở e61 vs e76) nhưng
  trần tương đương; gap theo epoch dao động ±0.01.
- **Phương án đã xem:**
  1. **Chạy ngay ablation 2×2** (2 run 150e không augment, ~2 h GPU ≈ $4): khoa học nhất
     nhưng user chọn hoãn để chốt hướng phân tích trước.
  2. **Bản rẻ 60e có augment** (~45 phút ≈ $1.5): nhanh nhưng confound do under-trained.
  3. **Không chạy gì, chỉ phân tích dữ liệu đã có** (đã chọn): ghi nhận hiện tượng, ghi
     giả thuyết + dự đoán bác bỏ được vào BACKLOG, chưa đưa kết luận nhân quả vào báo cáo.
- **Quyết định:** chọn (3) — chưa chạy thí nghiệm mới; mọi phát biểu về nguyên nhân phải
  giữ ở dạng giả thuyết. Không sửa kết quả 150e đã ghi; không đụng config 60e.
- **Hệ quả:** báo cáo khoa học hiện **không được** claim "GauSS tăng AP đáng kể" cho recipe
  150e có augment. Claim còn đứng vững trên dữ liệu hiện có: GauSS thắng AP@0.25 tiled ở
  mọi bucket, MR@0.25 thấp hơn ở mọi bucket, hội tụ nhanh hơn ~15 epoch, và oracle `score`
  cho thấy score calibration là nút thắt (+0.21 AP50).
- **Kiểm chứng:** các bảng trong artifact `GAP_SHRINK_ANALYSIS` tính trực tiếp từ
  `reports/eval/*_150e_*.json` và `/tmp/deepbench/final/*_60e_*/summary.json`;
  đường cong hội tụ từ `metrics/system_metrics.csv` của 2 run 150e.
- **Chưa xác minh:** nguyên nhân nhân quả của gap thu hẹp (cần BL-030); độ lớn nhiễu seed
  (chỉ seed 42, dao động quan sát ±0.01 AP); định nghĩa oracle variant `both`.
## ADR-022: Sửa cơ chế GauSS (v3.2) — bỏ "exact identity at init", warm-start cổng (2026-11-10)

- **Bối cảnh:** phân tích truy nguyên 150e (`ROOT_CAUSE_150E`, artifact) đo trực tiếp trên
  module thật: `max|∂L/∂σ| = 0.000e+00` tại init và độ nhạy output với σ cũng `0.000e+00`;
  sau 150 epoch thật σ chỉ trôi 3.5–5% dải cho phép; `high_band_share` đứng yên
  0.102→0.107. Gap AP50 tiled +0.0048 với 95% CI [−0.0031, +0.0145] (paired bootstrap,
  B=120) ⇒ không có hiệu ứng. Nguyên nhân không phải "module kém" mà là **cơ chế lõi
  không hoạt động**.
- **Chẩn đoán toán học:** `y = x0 + alpha·z(σ) + gamma_out·b_h(σ)` với `x0` không phụ
  thuộc σ. Nếu `alpha = gamma = 0` (đúng trạng thái ReZero-exact của v3.1) thì `y ≡ x0`
  ⇒ `∂L/∂σ ≡ 0` **chính xác**, không phải "nhỏ". Exact identity tại init và gradient σ
  khác 0 là **hai điều kiện loại trừ nhau**. Đo thực nghiệm xác nhận: alpha=0 → 0.000e+00;
  alpha=0.10 → 8.1e-05.
- **Phương án đã xem:**
  1. **Giữ identity, thêm loss phụ cho σ** (ví dụ ép σ khớp thống kê band): giữ được tính
     chất "không đổi baseline ở epoch 0" nhưng phải bịa supervision ngoài task — không có
     cơ sở lý thuyết, và vẫn không cho σ tác động lên output.
  2. **Tăng LR cho σ** (giữ identity): vô hiệu về mặt toán học — gradient bằng 0 thì mọi
     LR đều bằng 0.
  3. **Warm-start cổng (đã chọn):** bỏ identity bitwise, init `alpha=0.10`,
     `gamma_in/out/mid=0.05`; output tại init là nhiễu động nhỏ (residual norm 17% của x0)
     thay vì identity. Đây là cách duy nhất để `∂L/∂σ ≠ 0` ngay bước 1.
- **Quyết định (v3.2, `src/tinydet/modules/gauss.py`):**
  1. Warm-start cổng: `alpha=0.10`, `gamma_in=0.05`, `gamma_out=0.05`, `gamma_mid=0.05`
     (raw = atanh(2g)).
  2. Thêm **mid-band direct anchor** `gamma_mid · b_m` với `b_m = l1 − l2` phụ thuộc **cả**
     σ1 và σ2 ⇒ σ2 có đường gradient 1-hop ra output (v3.1 chỉ có σ1 có đường này).
  3. Đổi tham số hoá σ từ sigmoid sang **tanh-centered**: `σ1 = 0.65 + 0.25·tanh(raw)`,
     `σ2 = 1.45 + 0.35·tanh(raw)` — cùng dải và cùng giá trị init, nhưng đạo hàm
     `dσ/draw` tăng gấp đôi (0.25/0.35 so với 0.125/0.175) và hết bão hoà sigmoid.
  4. `configure_gauss_control_param_group(..., lr_mult=...)`; trainer đọc
     `gauss_control_lr_mult` từ config train, **mặc định 3.0** (1.0 = hành vi v3.1).
  5. Chẩn đoán mới: `gaussian/sigma{1,2}_{min,max}`, `gates/gamma_mid`,
     `contribution/mid_anchor_share`.
  6. Router correction giờ học được (trước đây `∂L/∂a = 0` tại init vì alpha=0). Energy
     prior vẫn `detach()` **có chủ ý**: nó là prior thống kê input cố định, phần học được
     là correction (ST-MoE z-loss giữ ổn định).
- **Kiểm chứng (số thật, chạy local CPU):**
  - Self-test `python src/tinydet/modules/gauss.py`: **PASS** — params 36,879 (+1 so với
    v3.1), ctrl 132, MACs không đổi 1.04132608 GMAC, `max|∂L/∂σ|@init = 5.01e-05`,
    `sens = 3.89e-02`, `σ drift/200 Adam steps = 1.074`, remap v3.1→v3.2 chính xác
    (r/2, giá trị σ bảo toàn), ablation `no_high_band`/`no_direct_anchor` vẫn sạch.
  - Script đo gradient: σ grad 1.2e-05…4.4e-05 so với **median conv grad 7.4e-05** ⇒ cùng
    bậc; `router_corr.weight` grad 6.4e-04 (trước: 0).
  - Smoke toàn model (`build_model` + forward/backward 320×320): 4/4 tham số σ có
    gradient, 132 control params được thu thập.
  - `ruff check src/tinydet/modules/gauss.py src/tinydet/training/trainer.py`: pass.
- **Hệ quả:**
  - Checkpoint v3.1 **không load trực tiếp** được ý nghĩa như cũ: dùng
    `remap_v31_to_v32_state_dict` (chỉ chia đôi sigma raw; gate raw giữ nguyên vì cùng
    `0.5·tanh`), load `strict=False` (thiếu `gamma_mid_raw`).
  - Config mới `configs/experiment/gauss_v32_tinyperson_wacv_150e.yaml` (giữ nguyên recipe
    150e, thêm `gauss_control_lr_mult: 3.0`). Config 150e cũ và 60e **không sửa**.
  - Mọi kết quả 150e đã báo vẫn là của v3.1; không được trộn số giữa hai phiên bản.
- **Chưa xác minh:** chưa train lại v3.2 trên GPU ⇒ chưa biết σ có thực sự dịch chuyển
  trong train thật và AP có cải thiện; chưa kiểm tra tính ổn định AMP/fp16 với warm-start;
  chưa chạy multi-seed; giá trị `lr_mult=3.0` chưa được tune.

## ADR-023: Giao thức kiểm chứng 3 tầng cho GauSS v3.2 — không train 150e nếu không qua cổng G1 (2026-11-10)

- **Bối cảnh:** sau khi sửa cơ chế (ADR-022), phân tích sâu hơn với loss thật trên CPU
  cho thấy hai tầng vấn đề tách biệt: (a) gradient chết ở init (đã sửa bằng warm-start)
  và (b) **tính định danh của σ** — paired delta trên 8 batch ảnh thật: σ1 full-range
  dịch chuyển đổi loss ~0 (t ≤ 0.66) cả ở init lẫn ở checkpoint đã train; σ2 có tín hiệu
  đơn điệu ở điểm đã train (t = −3.4, model muốn σ2 nhỏ hơn); xóa toàn bộ residual GauSS
  (gates→0) đổi loss +0.014 ± 0.038 (bằng không thống kê).
- **Quyết định:** áp dụng giao thức 3 tầng (artifact `CRITERIA_V32_PREREG`):
  1. **C0 (đã xong, $0):** T1 gradient tồn tại (PASS); T2 sign consistency 0.33 = random
     (FAIL — nhiễu chi phối tại init); T3 identifiability (σ1 FAIL, σ2 có ở điểm trained);
     T4 mini-train 150 bước: σ mean đứng yên, σ per-channel std tăng nhanh với lr_mult=3
     (0.036 đơn vị σ vs 0.0244 của v3.1 sau 150 epoch).
  2. **G1 (GPU ~$0.35):** launch đúng config 150e, kill tại e20 nếu fail cổng đã đăng ký:
     G1.1 |σ2_mean−1.45| ≥ 0.02; G1.2 σ2_std ≥ 0.02; G1.3 alpha ≥ 0.15; G1.4 val
     mAP50@e20 ≥ 0.09. Điểm dừng e10: σ2_std < 0.01 ∧ alpha < 0.12 ⇒ kill.
  3. **G2 (150e, ~$2.2):** chỉ chạy nếu G1 đạt; kết quả phân xử bằng paired bootstrap
     (L-011), không bằng gap thô.
- **Dự đoán đã đăng ký (trước khi nhìn kết quả GPU):** P1 σ1_mean đứng yên (85%);
  P2 σ2_mean dịch ≥ 0.02 hướng giảm (55%); P3 alpha ≥ 0.15 @e20 (80%); P4 σ2_std ≥ 0.05
  cuối (70%); **P5 CI bootstrap chứa 0 (45%)** — mức trung thực cao nhất, vì residual
  GauSS đóng góp ~0 vào train loss.
- **Nhánh sửa cấu trúc nếu G1 fail:** S1 mở rộng dải σ2 (ưu tiên, test local được trước);
  S3 σ cố định + routing học được (trung thực nhất về claim); S2 bỏ bn_proj; S4 auxiliary
  supervision (cuối).
- **Hệ quả:** không launch 150e v3.2 nếu không qua G1; mọi claim AP tiếp tục bị cấm theo
  ADR-021 cho đến khi bootstrap nói khác.
- **Chưa xác minh:** probe ở 320px/batch 4 có thể đánh giá thấp SNR của train thật
  (1280px/batch 16); động học cosine+warmup chưa được mô phỏng; P5 là ước lượng chủ quan.
