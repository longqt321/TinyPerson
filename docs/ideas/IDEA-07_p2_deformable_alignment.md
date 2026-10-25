# IDEA-07 — Căn chỉnh đặc trưng biến dạng trên P2

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-07
- trạng thái: đề xuất
- ưu tiên: P2
- nút thắt: sai lệch vị trí gần đúng
- chi phí ước tính: 2 lần chạy × 50 epoch trên A10; bắt buộc benchmark latency
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Deformable/offset sampling trên P2 sẽ căn chỉnh đặc trưng theo biên đối tượng,
giúp box regression chính xác hơn MSR và tăng AP75.

## Bằng chứng
- 866 GT có best IoU `0.10–0.50`; sai số tâm trung vị `3.81 px`.
- MSR (contrast/evidence) không cải thiện AP75; đặc trưng rõ hơn không đủ.
- P2 có stride 4, nhưng box 4–8 px vẫn rất nhạy với lệch 1–2 px.

## Cơ chế
Học offset cho từng vị trí sample trên P2, lấy đặc trưng tại biên/tâm đối tượng
thay vì grid cố định. Đặc trưng được căn chỉnh theo hình học, không chỉ theo
contrast. Box head nhận đặc trưng đã align nên gradient regression ổn định hơn.

## Tính mới và công trình liên quan
Deformable convolution đã có; đóng góp là áp dụng cho P2 đối tượng nhỏ và đo
được headroom. Khác MSR: MSR chuẩn hóa local evidence, ý tưởng này học sample offset.

## Thay đổi tối thiểu
- Thêm module `P2DeformAlign` trong `src/tinydet/modules/`, chèn trước Detect.
- Chỉ P2; P3/P4/P5 giữ nguyên.
- Cấu hình: số offset, kernel, residual scale.

## Giao thức thực nghiệm
- Control: A2 baseline.
- Treatment: A2 + P2DeformAlign.
- 50 epoch, cùng data/augmentation/optimizer/GPU A10.
- Benchmark latency/VRAM A10 batch 1.

## Chỉ số và ngưỡng
- Chính: AP75; tiny1 AP75.
- Ngưỡng: AP75 tăng `+0.0015`; latency <=+10%; AP50 không giảm.
- Theo dõi localization FP.

## Loại bỏ thành phần và bác bỏ
- Offset only so với offset + gate.
- P2 only so với P2/P3.
- Identity initialization để kiểm tra implementation.
- Bác bỏ: nếu AP75 không tăng, feature alignment không phải nút thắt.
- Tiêu chí dừng: dừng sau 1 seed nếu AP75 không tăng `+0.0015` hoặc latency vượt 10%.

## Rủi ro
- Thêm tham số và chi phí.
- Deformable operation có thể không ổn định với AMP.
- Có thể lặp lại thất bại của MSR nếu chỉ học contrast.

## Tuyên bố trong bài báo
- Nếu dương: “căn chỉnh P2 theo hình học cải thiện localization đối tượng nhỏ”.
- Nếu âm: thất bại của MSR/contrast nằm ở đặc trưng, không phải alignment.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, W&B, latency/VRAM, phân tích lỗi.

## Quyết định
- GO nếu AP75 tăng và latency trong budget.
- NO-GO nếu chỉ tăng AP50.

## Điểm
tác động 3, bằng chứng 1, tính mới 4, chi phí 2, rủi ro 4, khả thi 2; tổng 16/30.
