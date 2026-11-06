
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
