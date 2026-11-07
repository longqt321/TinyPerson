
## ADR-005: Thêm thư viện Seaborn cho trực quan hóa báo cáo học thuật

- **Trạng thái**: Chấp thuận
- **Lý do**: Cần tạo các biểu đồ phân tích thực nghiệm học thuật (AP theo size-bin, đa ngưỡng IoU, so sánh quốc tế, ma trận per-class VisDrone) ở định dạng vector SVG sắc nét cho báo cáo tổng kết.
- **Phương án đã xem**: Matplotlib thuần túy, Plotly, Seaborn.
- **Lựa chọn**: Seaborn 0.13.2 (dựa trên Matplotlib, cú pháp thống kê rõ ràng, style chuẩn báo cáo paper).
- **License**: BSD-3-Clause (hoàn toàn tự do sử dụng).

## ADR-006: Cờ ablation DCAF qua biến môi trường TINYDET_ABLATION

- **Trạng thái**: Chấp thuận
- **Lý do**: Cần 5 ablation (bỏ band cao/giữa/thấp, bỏ router, bỏ saliency gate) mà không viết lại module, không đổi số tham số, giữ checkpoint tương thích. Env var cho phép bật cờ per-run qua Modal không cần sửa model YAML.
- **Phương án đã xem**: (1) env var đọc trong DCAF.__init__/forward; (2) 5 file YAML riêng với 5 class DCAF biến thể; (3) YAML flag truyền qua parse_model.
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

## ADR-010: Thay thế DCAF v2.1 bằng P2Refine và dọn dẹp chuỗi layer chết

- **Trạng thái**: Chấp thuận
- **Ngày**: 2026-11-08
- **Lý do**:
  1. Phân tích thực nghiệm và lý thuyết cho thấy DCAF v2.1 gặp hiện tượng classification-head desync
     ngay từ step 0 (đầu phân loại khởi tạo cho $x_0$, nhưng nhận $y = x_0 + \alpha z$ với năng lượng lệch lớn).
  2. DCAF v2.1 trong YAML cũ có chuỗi layers 17→18→19 (Upsample từ 16 → Concat với 2 → C3k2) tạo ra output
     hoàn toàn không được layer nào tiêu thụ (dead branch lãng phí ~0.87 GMAC và 8.7k tham số).
  3. Cả 5 giả thiết cơ chế của DCAF v2.1 (saliency router, scale-space band, dilated context)
     đều đã bị bác bỏ bởi ablation bootstrap trước đó.
- **Phương án đã xem**:
  1. Sửa DCAF v2.1 với đề xuất N4' (zero-init output projection, bỏ double-SiLU) + giữ nguyên kiến trúc.
  2. Bỏ hoàn toàn module tinh chế, dùng baseline A2 (không có refine ở P2).
  3. Tích hợp `P2Refine`: module tinh chế tối giản, ReZero-style (zero-init exit projection không BN),
     chỉ lấy input [C2, P3], đảm bảo đồng nhất bit-exact tại step 0, đồng thời dọn sạch chuỗi layer chết (chọn).
- **Lựa chọn**: (3).
  - Kiến trúc `P2Refine`: Stem kết hợp C2 + Up(P3) qua Conv 1x1 + BN + SiLU. Nhánh residual gồm DW3x3 → PW (chuyển sang mid=16) → DW5x5 → W_out (zero-init, không BN). Output = stem + W_out(h).
  - Tổng tham số module: 6,000 (so với 34,700 của DCAF).
  - MACs module: 0.4375 GMAC (so với 1.0176 GMAC của DCAF).
  - Mô hình đầy đủ: 2,513,272 tham số, 31.067 GFLOPs ở 1280x1280 (giảm so với 32.2 GFLOPs của DCAF).
- **Hệ thống theo dõi & Diagnostics**:
  - Tạo `P2RefineDiagnosticsCallback` trong `src/tinydet/training/module_diagnostics.py` tự động theo dõi
    và ghi lại các chỉ số `rho_mean`, `rho_max`, `cos_x0_delta`, `w_out_norm` vào `metrics/p2refine_diagnostics.csv`
    và đồng bộ lên W&B mỗi epoch (khắc phục điểm thiếu sót của DCAF khi 21 run trước đó không có log module).
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
