# Trạng thái báo cáo DCAF

## Phạm vi
- Handoff dossier DCAF v2.1 và baseline YOLO26n-P2 cho model bên ngoài viết báo cáo.
- Tiếng Việt; không viết thêm báo cáo nghiên cứu.
- Không mở hoặc trích xuất nội dung QAC hay `report-bundle/`.

## Tiến độ
- Giai đoạn 0: hoàn tất; phạm vi DCAF-only, tiếng Việt, technical report Markdown.
- Giai đoạn 1: inventory tại `01_inventory.md`; kết quả ưu tiên lấy từ JSON DCAF và baseline paired.
- Giai đoạn 2: phân tích DCAF-only tại `03_analysis.md`; một seed, kết luận mô tả.
- Giai đoạn 3: chưa tìm nguồn ngoài để tránh citation chưa kiểm chứng; related work trong draft là khung chủ đề, chưa phải tổng quan hoàn chỉnh.
- Giai đoạn 4: outline tại `04_outline.md`.
- Giai đoạn 5: bảng kết quả trong draft đối chiếu thủ công với JSON; chưa có script sinh tự động hoặc hình.
- Giai đoạn 6: draft tiếng Việt tại `draft/dcaf_report.md`.
- Giai đoạn 7: review ghi tại `99_review.md`; chưa biên dịch định dạng submission.

## Cảnh báo provenance
- Artifact eval 150e trỏ `training/weights/last.pt`; ghi đúng checkpoint này, không gọi là `best.pt`.
- Giao thức artifact khai báo 786 ảnh, dense excluded = 0; mô tả là standalone implementation, không khẳng định tuân thủ hoàn toàn official benchmark.
- Kết quả đơn seed; không kiểm định ý nghĩa thống kê.
- Draft là báo cáo nội bộ có giới hạn bằng chứng, chưa phải bản nộp học thuật do related work chưa xác minh và venue chưa biết.
