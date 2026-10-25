# IDEA-01 — Gán nhãn TAL có nhận biết tâm

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-01
- trạng thái: đề xuất
- ưu tiên: P0
- nút thắt: sai lệch vị trí gần đúng
- chi phí ước tính: 2 lần chạy × 50 epoch trên A10; chỉ mở rộng 3 seed khi qua ngưỡng
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Thêm prior khoảng cách tâm đã chuẩn hóa vào TAL sẽ giữ được positive gần GT tốt
hơn beta=4, giúp tăng AP75 của nhóm GT có best IoU `0.10–0.50`.

## Bằng chứng
- 866/3.608 GT có best IoU `0.10–0.50`; sai số tâm trung vị `3.81 px`.
- Nhóm `IoU>=0.75` chỉ lệch tâm `0.94 px`, nên lỗi không nằm ở các box đã tốt.
- TAL beta=4 làm AP50 tăng `0.002288`, precision tăng `0.009645`, recall tăng
  `0.010456`, nhưng AP75 gần như không đổi (`-0.000172`).
- Kết luận: cách gán nhãn hiện tại có thể đã chọn đúng hướng recall, nhưng chưa
  chuyển thành localization IoU cao.

## Cơ chế
TAL hiện tại xếp hạng candidate bằng `cls^alpha * CIoU^beta`. Với box nhỏ, CIoU
thay đổi rất mạnh chỉ vì vài pixel. Thêm một prior tâm theo kích thước
`exp(-0.5 * (d / (tau * sqrt(area)))²)` sẽ ưu tiên anchor có tâm gần GT, kể cả
khi CIoU còn thấp. Positive được giữ lại sẽ có gradient regression ổn định hơn.

## Tính mới và công trình liên quan
Đây là thay đổi cách gán nhãn, không phải module đặc trưng. Khác TAL beta=4 ở
chỗ beta chỉ đổi trọng số CIoU; center prior thêm một tín hiệu hình học độc lập.
Không tuyên bố là thuật toán mới nếu chưa có ablation và so sánh với TAL.

## Thay đổi tối thiểu
- Thêm `CenterAwareAssigner` trong `src/tinydet/training/` hoặc subclass
  `TaskAlignedAssigner`.
- Chỉ thay assigner của `v8DetectionLoss`; không đổi CIoU, L1/DFL, head.
- Cấu hình: `tal_center_tau: 0.5`.

## Giao thức thực nghiệm
- Control: E0 A2, TAL beta=6.
- T1: TAL beta=4.
- T2: center-aware, beta=6.
- T3: center-aware, beta=4.
- 50 epoch, imgsz 1280, batch 8, AdamW, cùng augmentation và GPU A10.

## Chỉ số và ngưỡng
- Chính: AP75; AP75 tiny1/tiny2/tiny3.
- Ngưỡng: AP75 tăng ít nhất `+0.0015`; AP50 không giảm quá `0.002`; recall
  không giảm quá `0.005`.
- Chỉ tuyên bố sau 3 seed paired.

## Loại bỏ thành phần và bác bỏ
- Ablation tau: `0.25`, `0.5`, `1.0`.
- So center prior với TAL beta=4 và TAL beta=6.
- Bác bỏ: nếu AP75 không tăng nhưng recall tăng, đây chỉ là assignment recall,
  không phải localization.
- Tiêu chí dừng: dừng sau 2 seed nếu AP75 không vượt control.

## Rủi ro
- Có thể nhận positive chất lượng thấp, làm precision và AP50 giảm.
- Prior tâm có thể xung đột với NMS và xếp hạng confidence.
- Cần sửa loss path, không được sửa site-packages.

## Tuyên bố trong bài báo
- Nếu dương: “gán nhãn có nhận biết tâm cải thiện localization IoU cao cho đối
  tượng nhỏ mà không đổi feature extractor”.
- Nếu âm: gán nhãn chỉ cải thiện recall/AP50, không phải AP75.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, W&B `hoangngan/tiny-person-research`, `val.json`,
  `per_object_errors.csv`.

## Quyết định
- GO nếu AP75 tăng và tiny1 không sụt.
- NO-GO nếu chỉ AP50/recall tăng.

## Điểm
tác động 4, bằng chứng 3, tính mới 3, chi phí 3, rủi ro 3, khả thi 4; tổng 20/30.
