# Archived: YOLO26n ablation trước thí nghiệm MSR-P2

Đây là báo cáo validation từ tháng 9/2026 cho các kiến trúc trước nhánh MSR-P2 hiện tại. Giữ để tham khảo lịch sử; không gộp các số này với run MSR hoặc xem đây là kết luận mới nhất.

---

# So sánh kết quả train trên Modal Volume

Nguồn: `results.csv`, `config.yaml`, `metadata.json` của bốn run ngày 2026-09-29. Chỉ tải file kết quả; không tải checkpoint `.pt`.

Các run dùng cùng TinyPerson dataset, seed 42, ảnh 1280, batch 16, AdamW, 50 epoch và cùng cấu hình augmentation. Chỉ `model.config` khác nhau. Chỉ số dưới đây là validation ghi trong quá trình train, chưa phải đánh giá độc lập trên test set.

| Mô hình | Epoch có dữ liệu | mAP50-95 tốt nhất (epoch) | mAP50-95 cuối | mAP50 cuối | Precision cuối | Recall cuối | Trạng thái |
|---|---:|---:|---:|---:|---:|---:|---|
| YOLO26n + P2 | 49/50 | 0.07540 (48) | 0.07500 | 0.24737 | 0.34158 | 0.34052 | RUNNING, bị ngắt |
| YOLO26n gốc | 50/50 | 0.06716 (45) | 0.06443 | 0.22827 | 0.34730 | 0.32313 | RUNNING, bị ngắt sau train |
| YOLO26n không P5 | 50/50 | 0.06443 (48) | 0.06192 | 0.21784 | 0.32203 | 0.30424 | COMPLETED |
| YOLO26n chỉ P3 | 50/50 | 0.05302 (46) | 0.05207 | 0.19045 | 0.30170 | 0.29017 | RUNNING, bị ngắt sau train |

## Nhận xét

- P2 dẫn đầu: mAP50-95 tốt nhất cao hơn mô hình gốc 0.00824, tức 12.3% tương đối. P2 mới có 49 epoch nên chưa phải kết quả train trọn vẹn.
- Bỏ P5 giảm mAP50-95 tốt nhất 0.00273 (4.1% tương đối) so với mô hình gốc; bỏ thêm P4 giảm 0.01414 (21.1% tương đối). Kết quả hiện tại không ủng hộ giả thuyết rằng bỏ context rộng giúp tăng độ chính xác trên bộ validation này.
- Cả ba run có trạng thái RUNNING vẫn giữ CSV và checkpoint trên Volume, nhưng chưa hoàn tất hậu xử lý. Vì vậy bảng dựa trên metrics từng epoch, không phải lần chạy `evaluate` độc lập từ `best.pt`.
- Chỉ một seed, không có phương sai giữa các lần chạy. Chênh lệch nhỏ giữa gốc và không P5 cần lặp thêm seed để kết luận chắc.
