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

## BL-004: `scripts/validate_tiny_ablation.py` trỏ tới model YAML không tồn tại

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

