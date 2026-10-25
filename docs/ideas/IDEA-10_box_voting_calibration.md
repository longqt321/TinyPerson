# IDEA-10 — Box voting và hiệu chỉnh điểm theo kích thước

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-10
- trạng thái: đề xuất
- ưu tiên: P1
- nút thắt: xếp hạng và FP trùng lặp
- chi phí ước tính: 0 train; chỉ inference trên checkpoint có sẵn
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Box voting/score calibration theo kích thước sẽ tăng AP75/AP50 trên checkpoint
có sẵn mà không cần train lại.

## Bằng chứng
- 39.121 localization FP và 10.461 duplicate FP ở ngưỡng IoU 0.50.
- 179/436 box `IoU>=0.75` có confidence `<0.25`.
- Thử nghiệm NMS trước đó tăng mAP/AP50 nhưng giảm AP75; cần cách khác.

## Cơ chế
Cluster các box gần nhau, hợp nhất bằng weighted box fusion; calibrate score theo
size bin để box chất lượng cao không bị đè dưới FP. Không đổi trọng số model.

## Tính mới và công trình liên quan
Đây là hậu xử lý, không phải module. Có thể dùng WBF/soft-NMS có sẵn. Không được
tuyên bố là cải thiện kiến trúc.

## Thay đổi tối thiểu
- Thêm script inference-only trong `scripts/`.
- So sánh NMS chuẩn, soft-NMS, WBF, size-aware calibration.
- Giữ nguyên checkpoint, conf, max_det.

## Giao thức thực nghiệm
- Control: NMS hiện tại, checkpoint A2/E2.
- Treatment: từng post-processing, cùng checkpoint và threshold.
- Đánh giá trên validation; test chỉ sau freeze.

## Chỉ số và ngưỡng
- Chính: AP75.
- Ngưỡng: AP75 tăng `+0.001`; AP50 không giảm; latency tăng <=20%.
- Theo dõi localization FP/duplicate FP.

## Loại bỏ thành phần và bác bỏ
- NMS IoU `0.4/0.5/0.6`.
- WBF weight; calibration theo size.
- Bác bỏ: nếu chỉ AP50 tăng, không dùng cho tuyên bố AP75.
- Tiêu chí dừng: dừng sau 2 biến thể post-processing nếu validation AP75 không
  tăng `+0.001`; không tune threshold trên test.

## Rủi ro
- Không phải đóng góp model.
- Có thể overfit validation threshold.
- Latency có thể vượt budget.

## Tuyên bố trong bài báo
- Nếu dương: hậu xử lý giải quyết một phần xếp hạng/duplicate.
- Nếu âm: model cần cải thiện localization, không phải NMS.

## Sản phẩm lưu trữ
- Prediction JSON, script, `val.json`, benchmark latency.

## Quyết định
- GO nếu AP75 tăng trên validation.
- NO-GO nếu chỉ AP50 tăng.

## Điểm
tác động 3, bằng chứng 3, tính mới 2, chi phí 5, rủi ro 2, khả thi 5; tổng 20/30.
