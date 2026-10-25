# IDEA-08 — Nhánh tần số/biên trên P2

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-08
- trạng thái: đề xuất
- ưu tiên: P2
- nút thắt: bỏ sót
- chi phí ước tính: 2 lần chạy × 50 epoch trên A10
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Một nhánh high-frequency/edge trên P2 sẽ giữ biên của đối tượng nhỏ, giúp recall
và AP75 cho đối tượng có độ tương phản thấp.

## Bằng chứng
- 1.149 GT có best IoU `<0.10`; nhiều đối tượng nhỏ mất contrast.
- MSR (local contrast) không cải thiện AP75; cần kiểm tra tín hiệu khác.
- P2 stride 4 là level duy nhất đủ độ phân giải cho đối tượng 4–8 px.

## Cơ chế
High-pass/edge filter tạo nhánh đặc trưng nhạy với biên và texture tần số cao.
Fuse với P2 bằng residual gate để không phá đặc trưng gốc. Box head có thêm tín
hiệu biên, giảm nhầm lẫn background.

## Tính mới và công trình liên quan
Frequency/edge feature đã có trong detection; đây là ablation cho đối tượng nhỏ.
Phải so sánh với MSR để tránh trùng lặp khái niệm “đặc trưng rõ hơn”.

## Thay đổi tối thiểu
- Thêm module `EdgeBranch` trong `src/tinydet/modules/`, chỉ P2.
- Cấu hình: kernel, high-pass type, residual scale.
- Không đổi assignment/loss.

## Giao thức thực nghiệm
- Control: A2 baseline.
- Treatment: A2 + EdgeBranch.
- 50 epoch, cùng data/augmentation/optimizer/GPU A10.

## Chỉ số và ngưỡng
- Chính: AP75; recall tiny1.
- Ngưỡng: AP75 tăng `+0.0015`; AP50 không giảm; tiny1 recall không giảm.

## Loại bỏ thành phần và bác bỏ
- Fixed high-pass so với learnable filter.
- Edge only so với edge + P2 residual.
- Bác bỏ: nếu AP75 không tăng, frequency không phải tín hiệu thiếu.
- Tiêu chí dừng: dừng sau 1 seed nếu AP75 không tăng `+0.0015` hoặc precision
  giảm quá `0.01`.

## Rủi ro
- Khuếch đại noise và background.
- Có thể làm tăng false positive.
- Không có tính mới mạnh nếu chỉ là filter cố định.

## Tuyên bố trong bài báo
- Nếu dương: edge/high-frequency cues cải thiện detection đối tượng nhỏ.
- Nếu âm: contrast/frequency không phải nút thắt.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, W&B, visualization đặc trưng, `val.json`.

## Quyết định
- GO nếu AP75 tăng và precision không giảm mạnh.
- NO-GO nếu chỉ tăng recall.

## Điểm
tác động 2, bằng chứng 1, tính mới 3, chi phí 3, rủi ro 4, khả thi 3; tổng 16/30.
