# IDEA-03 — Đầu tinh chỉnh box nhỏ

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-03
- trạng thái: đề xuất
- ưu tiên: P1
- nút thắt: sai lệch vị trí gần đúng
- chi phí ước tính: 2 lần chạy × 50 epoch trên A10; bắt buộc đo latency
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Một head tinh chỉnh nhẹ, class-agnostic, trên P2/P3 sẽ sửa các box sai lệch gần
đúng và tăng AP75 mà không làm thay đổi classification.

## Bằng chứng
- 866 GT có best IoU `0.10–0.50`; sai số tâm trung vị `3.81 px`.
- 1.157 GT có best IoU `0.50–0.75`; chỉ cần cải thiện nhẹ để vượt AP75.
- Box đã đạt `IoU>=0.75` có sai số tâm trung vị `0.94 px`, nên không cần tinh
  chỉnh toàn bộ detection.
- NMS hiện tại đã có 39.121 localization FP; tinh chỉnh phải không làm tăng duplicate.

## Cơ chế
RoIAlign hoặc deformable sampling tại P2/P3 lấy đặc trưng quanh box đề xuất.
Một MLP dự đoán residual `dx, dy, dw, dh`; box cuối = box đề xuất + residual.
Chỉ tinh chỉnh các candidate có score/quality trung bình, không đổi class score.

## Tính mới và công trình liên quan
Đây là module mới ở tầng refinement, khác MSR (feature contrast) và khác DFL
(regression representation). Cần so sánh với box voting để chứng minh học được
thông tin đặc trưng, không chỉ là trung bình box.

## Thay đổi tối thiểu
- Thêm `TinyBoxRefine` trong `src/tinydet/modules/` và hook vào `Detect`.
- Chỉ tinh chỉnh P2/P3; P4/P5 giữ nguyên.
- Cấu hình: số lớp MLP, số điểm sample, ngưỡng score để tinh chỉnh.

## Giao thức thực nghiệm
- Control: A2 baseline.
- Treatment: A2 + refinement head.
- 50 epoch, imgsz 1280, batch 8, AdamW, cùng augmentation và A10.
- Đo latency A10 batch 1 trước khi mở rộng seed.

## Chỉ số và ngưỡng
- Chính: AP75.
- Ngưỡng: AP75 tăng `+0.002`; AP50 không giảm quá `0.002`; latency tăng <=10%;
  VRAM tăng <=10%.
- Kiểm tra duplicate FP và localization FP.

## Loại bỏ thành phần và bác bỏ
- Tinh chỉnh P2 only so với P2/P3.
- Residual identity (không học) để kiểm tra implementation.
- So sánh refinement với box voting.
- Bác bỏ: nếu refinement làm AP75 giảm hoặc latency vượt 10%, dừng.
- Tiêu chí dừng: dừng sau 1 seed nếu AP75 không tăng `+0.001` hoặc latency/VRAM
  vượt 10%.

## Rủi ro
- Box refine có thể học label noise.
- NMS có thể loại box tốt sau refine.
- Thêm tham số và độ phức tạp; cần checkpoint compatibility.

## Tuyên bố trong bài báo
- Nếu dương: “một head tinh chỉnh nhẹ, class-agnostic, cải thiện localization
  IoU cao cho đối tượng nhỏ”.
- Nếu âm: refinement không hơn box voting, tức lỗi nằm ở ranking/NMS.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, W&B, benchmark latency/VRAM, CSV phân tích lỗi.

## Quyết định
- GO nếu AP75 tăng và latency trong budget.
- NO-GO nếu chỉ tăng AP50 hoặc duplicate FP.

## Điểm
tác động 4, bằng chứng 3, tính mới 4, chi phí 2, rủi ro 4, khả thi 3; tổng 20/30.
