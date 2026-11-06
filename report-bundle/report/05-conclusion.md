# Chương 5. Kết luận và hướng nghiên cứu tiếp theo

## 5.1 Kết luận

Báo cáo trình bày thiết kế GauSS cho tinh chỉnh đặc trưng P2 trên YOLO26n-P2. Cơ chế kết hợp phân rã Gaussian đa mức, định tuyến theo vị trí và hai nhánh refinement. Đây là mô tả kiến trúc, chưa phải kết luận về hiệu quả thực nghiệm: bản tổng hợp hiện có không kèm run artifact GauSS để kiểm tra AP, Miss Rate hay chi phí suy luận. Vì vậy, câu hỏi liệu module có cải thiện detector trong điều kiện TinyPerson đã nêu vẫn chưa được giải đáp bằng bằng chứng có thể tái lập.

## 5.2 Hướng nghiên cứu tiếp theo

Bước tiếp theo là hoàn thiện provenance thực nghiệm: cung cấp cấu hình, checkpoint, log huấn luyện, evaluator output và hồ sơ đo latency/VRAM, sau đó chạy lại hoặc đối chiếu toàn bộ chỉ số với cùng giao thức. Khi có kết quả nền tảng được xác minh, ablation có thể tách riêng ảnh hưởng của σ học được, tính bất đẳng hướng, router, saliency mask và hai nhánh refinement. Mở rộng sang dữ liệu khác chỉ có ý nghĩa sau khi quy trình trên TinyPerson được tái lập và giới hạn evaluator được làm rõ.
