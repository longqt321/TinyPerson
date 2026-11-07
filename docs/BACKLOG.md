# Backlog

Việc phát hiện thêm trong lúc làm task, chưa nằm trong phạm vi đã thống nhất.

## BL-001: Mâu thuẫn chính sách GPU giữa code và tài liệu

- **Phát hiện:** 2026-11-05
- **Hiện trạng:** `modal/app.py` chặn mọi GPU khác A10 ở hai chỗ:
  `train` (dòng 317-319) và `compare` (dòng 448), với thông điệp
  "Only A10 GPU is allowed by the resource cap". Nhưng
  `docs/agents/OPERATIONS.md` dòng 7 lại ghi "Full training and benchmark
  require Modal with an NVIDIA L4".
- **Ảnh hưởng:** không rõ GPU nào là chuẩn để tái lập kết quả. Cũng chưa rõ
  `_train_ablation` (gọi thẳng `base_app._train.with_options(gpu=...)`, bỏ qua
  local entrypoint) có vượt được guard hay không.
- **Đề xuất:** chọn một nguồn sự thật duy nhất, sửa bên còn lại, và ghi lý do
  vào `docs/DECISIONS.md`.

## BL-002: Chưa có số đo throughput của L4 và T4 cho workload này

- **Phát hiện:** 2026-11-05
- **Hiện trạng:** mọi so sánh chi phí giữa A10G / L4 / T4 trong repo đều phải
  giả định hệ số chậm (1.75× cho L4, 2.30× cho T4) suy từ bandwidth lý thuyết.
  Không có `results.csv` nào của L4 hay T4 trong repo.
- **Ảnh hưởng:** không thể kết luận chắc chắn GPU nào rẻ hơn **trên mỗi run**,
  chỉ kết luận được trên \$/giờ.
- **Đề xuất:** chạy 1 pilot 3 epoch trên L4 và T4 với đúng config, ghi
  s/epoch vào `reports/`. Chi phí ước tính dưới \$0.20.

## BL-003: `save_period` không ghi checkpoint định kỳ

- **Phát hiện:** 2026-11-05
- **Hiện trạng:** config đặt `save_period: -1` nên Ultralytics không ghi
  checkpoint theo chu kỳ; chỉ có `last.pt` và `best.pt` được ghi mỗi epoch.
  Yêu cầu ban đầu là "lưu checkpoint mỗi 10 epoch vào thư mục riêng theo tên run".
- **Ảnh hưởng:** không có snapshot trung gian để quay lại; chỉ resume được từ
  epoch gần nhất.
- **Đề xuất:** nếu cần snapshot định kỳ, đặt `save_period: 10` cho các run tiếp
  theo — nhưng phải ghi vào `docs/DECISIONS.md` vì nó làm tăng dung lượng volume
  và có thể ảnh hưởng thời gian epoch.

## BL-004: `scripts/validate_tiny_ablation.py` trỏ tới model YAML không tồn tại (đã đóng — xem BL-015)

- **Phát hiện:** 2026-11-06
- **Hiện trạng:** `DEFAULT_MODELS` trong `scripts/validate_tiny_ablation.py`
  liệt kê `yolo26n_tiny_A0_baseline.yaml`, `yolo26n_tiny_A1_no_sppf.yaml`,
  `yolo26n_tiny_A3_p2_no_sppf.yaml`. Cả ba file đã bị xóa khỏi `configs/model/`;
  chỉ còn `yolo26n_tiny_A2_p2.yaml`. Script cũng còn nhánh `_msr`/`_btrm`.
- **Ảnh hưởng:** chạy script không tham số sẽ `FileNotFoundError`. `docs/agents/CODE.md`
  đã được sửa để gọi với `configs/model/yolo26n_tiny_A2_p2.yaml`, nhưng bản thân
  script vẫn sai.
- **Đề xuất:** cập nhật `DEFAULT_MODELS` và bỏ nhánh `_msr`/`_btrm` khi có dịp
  sửa code (ngoài phạm vi task dọn dẹp này, vì yêu cầu là không đổi code).

## BL-005: `reports/experiment_results_report.md` là hồ sơ của hướng nghiên cứu đã dừng

- **Phát hiện:** 2026-11-06
- **Hiện trạng:** file này tổng hợp A0–A3, MSR, BTRM, SADR, QAC v1–v3, oracle AP75.
  Đây là hồ sơ lịch sử của các hướng đã dừng, không phải kết quả GauSS hiện tại.
  Các tài liệu QAC/QAR riêng lẻ đã được xóa; file này được giữ vì nó là nguồn
  duy nhất ghi lại số liệu A0–A3/MSR/BTRM/SADR.
- **Ảnh hưởng:** người đọc mới dễ nhầm đây là kết quả hiện hành.
- **Đề xuất:** thêm banner "lịch sử — đã ngừng" ở đầu file, hoặc chuyển vào
  `docs/archive/` khi không còn ai cần đối chiếu.


## BL-012: `compute_dcaf_analytical_macs` chưa có phiên bản chuẩn hóa cho P2Refine

> Ghi chú: BL-006 đến BL-011 thuộc một thay đổi khác (evaluator `class_mode`) chưa được
> commit trong nhánh này, nên số hiệu được giữ nguyên và để trống ở đây.

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** Trong `src/tinydet/modules/dcaf.py` có hàm giải tích tính MACs cho DCAF.
  Đối với `P2Refine`, MACs hiện được đo thực nghiệm độc lập bằng script `measure_p2refine.py`.
- **Ảnh hưởng:** Các script báo cáo hoặc profile tự động cần một hàm giải tích thống nhất
  để xuất bảng thống kê FLOPs chi tiết cho từng block mà không cần instantiate tensor.
- **Đề xuất:** Thêm hàm `compute_p2refine_analytical_macs` vào `src/tinydet/modules/p2_refine.py`
  và re-export trong `tinydet.modules`.

## BL-013: DCAF diagnostics chưa từng được lưu trong 21 lượt train trước

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** 21 thư mục run cũ trên Modal volume (`metadata.json`) không có bất kỳ log
  nào về router gate, band weight hay dynamic scale-space.
- **Đã xử lý:** Đã triển khai `P2RefineDiagnosticsCallback` gắn tự động qua trainer lifecycle
  để ghi lại `rho_mean`, `rho_max`, `cos_x0_delta`, `w_out_norm` cho P2Refine.
- **Chưa xử lý:** Không thể phục hồi dữ liệu động học (training dynamics) của 21 run DCAF cũ
  ngoại trừ các checkpoint `best.pt`/`last.pt` còn lưu.

## BL-014: `wandb` thiếu khỏi `pyproject.toml` nên `uv sync` không cài

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** `src/tinydet/training/trainer.py`, `iou_logging.py`, `module_diagnostics.py`
  và `wandb_media.py` đều `import wandb` trong `try/except` hoặc có guard, nhưng `wandb`
  không nằm trong `[project.dependencies]` và không có trong `uv.lock`. Trên Modal thì
  `modal/image.py:8` cài riêng nên không lộ; trên máy local hoặc Colab/Kaggle dùng
  `uv sync` thì W&B **im lặng tắt** — `_init_wandb` bắt `ImportError` và in
  "W&B disabled after init failure".
- **Ảnh hưởng:** mọi run chạy bằng `uv sync` đều không log W&B dù config có
  `wandb_project`. Không mất dữ liệu (CSV và metadata.json vẫn ghi), nhưng mất khả năng
  theo dõi trực tuyến.
- **Đã xử lý:** thêm `"wandb>=0.21,<1"` vào `[project.dependencies]`, chạy `uv lock`
  (thêm `wandb v0.31.0`). Kiểm chứng: `uv run --extra cpu python -c "import wandb"` →
  `0.31.0`.
- **Chưa xử lý:** `modal` vẫn là dependency bắt buộc (xem BL-016).

## BL-015: `scripts/validate_tiny_ablation.py` trỏ tới 3 model YAML đã xóa

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** `DEFAULT_MODELS` liệt kê `yolo26n_tiny_A0_baseline.yaml`,
  `yolo26n_tiny_A1_no_sppf.yaml`, `yolo26n_tiny_A3_p2_no_sppf.yaml`. Cả ba không còn
  trong `configs/model/`, nên chạy script không tham số sẽ `FileNotFoundError`.
- **Đã xử lý:** rút `DEFAULT_MODELS` còn 3 file thực có:
  `yolo26n_tiny_A2_p2.yaml`, `yolo26n_tiny_A2_p2_tinyperson.yaml`,
  `yolo26n_tiny_A2_p2_p2refine_tinyperson.yaml`. Chạy không tham số giờ pass cả 3.
- **Ghi chú:** mục này thay thế BL-004 (đã đóng).

## BL-016: `modal` vẫn là dependency bắt buộc của package

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** `pyproject.toml:17` khai báo `modal>=1,<2` trong `[project.dependencies]`.
  Trên Colab/Kaggle, `pip install -e .` kéo theo `modal` dù không dùng.
- **Ảnh hưởng:** tăng thời gian cài và dung lượng, nhưng không chặn gì. `src/tinydet/**`
  không import `modal` ở đâu.
- **Đề xuất:** chuyển `modal` sang optional-dependency (ví dụ `[project.optional-dependencies]
  modal = [...]`) và cập nhật `uv sync` trong tài liệu. Chưa làm vì sẽ đổi lệnh cài mặc
  định của mọi người và ảnh hưởng workflow Modal hiện tại.

## BL-017: Chưa có smoke test tự động cho `train.py`

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** `train.py` được kiểm chứng thủ công bằng `--dry-run` trên CPU và bằng
  cách gọi `trainer.train()` để xác nhận cổng chặn CUDA. Repo chưa có test suite
  (`pytest` không nằm trong dependency).
- **Ảnh hưởng:** nếu ai sửa `_resolve_model_config` hoặc `_resolve_paths`, lỗi chỉ lộ ra
  khi chạy thật.
- **Đề xuất:** thêm `tests/test_train_cli.py` kiểm tra `--dry-run` exit 0 và không có
  `--dry-run` exit 1 trên CPU. Cần thêm `pytest` vào `[dependency-groups].dev` trước.
