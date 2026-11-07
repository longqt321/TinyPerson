
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

## ADR-010: Thay thế DCAF v2.1 bằng P2Refine và dọn dẹp chuỗi layer chết

> Ghi chú: ADR-008 và ADR-009 thuộc một thay đổi khác (evaluator `class_mode`) chưa
> được commit trong nhánh này, nên số hiệu được giữ nguyên và để trống ở đây.

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
