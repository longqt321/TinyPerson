# IDEA-04 — Gán nhãn mềm Gaussian

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-04
- trạng thái: đề xuất
- ưu tiên: P2
- nút thắt: bỏ sót và thiếu positive
- chi phí ước tính: 2 lần chạy × 50 epoch trên A10
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Trọng số positive mềm theo khoảng cách tâm sẽ giảm tình trạng thiếu positive cho
đối tượng nhỏ và tăng recall/AP75 so với hard inside-box assignment.

## Bằng chứng
- 1.149 GT có best IoU `<0.10`; gần như không có detection gần.
- 866 GT có best IoU `0.10–0.50`; nhiều box có tâm gần nhưng không đủ overlap.
- Hard assignment chỉ lấy anchor nằm trong GT; với box 4–8 px, số anchor rất ít.

## Cơ chế
Thay hard mask bằng trọng số Gaussian theo khoảng cách từ anchor đến tâm GT,
chuẩn hóa theo kích thước GT. Classification target là trọng số mềm; box loss
vẫn dùng CIoU + regression. Mục tiêu là giữ candidate gần GT thay vì loại hẳn.

## Tính mới và công trình liên quan
Soft label/assignment đã có trong nhiều detector; đây là ablation cho đối tượng
nhỏ, không phải module mới. Phải so sánh với TAL và center-aware assignment.

## Thay đổi tối thiểu
- Custom assigner/loss trong `src/tinydet/training/`.
- Cấu hình: `gaussian_sigma`.
- Chỉ đổi assignment/label target; không đổi head hay feature.

## Giao thức thực nghiệm
- Control: E0 A2.
- Treatment: Gaussian soft assignment.
- 50 epoch, cùng data/augmentation/optimizer/GPU A10.

## Chỉ số và ngưỡng
- Chính: AP75; recall tiny1.
- Ngưỡng: AP75 tăng `+0.001`; recall tiny1 tăng `+0.01`; precision không giảm
  quá `0.01`.
- Theo dõi số positive/GT và tỷ lệ anchor được gán.

## Loại bỏ thành phần và bác bỏ
- Sigma `0.3`, `0.5`, `0.7`.
- Soft classification only so với soft classification + box weight.
- Bác bỏ: nếu recall tăng nhưng AP75 không tăng, đây chỉ là detection, không
  phải localization.
- Tiêu chí dừng: dừng sau 1 seed nếu recall tiny1 không tăng `+0.01` hoặc AP75
  không tăng `+0.001`.

## Rủi ro
- Nhận positive chất lượng thấp.
- Nhiễu giữa các đối tượng gần nhau.
- Sai lệch annotation có thể bị khuếch đại.

## Tuyên bố trong bài báo
- Nếu dương: soft assignment cải thiện recall tiny1 và AP75.
- Nếu âm: hard assignment không phải nút thắt.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, W&B, chẩn đoán số positive, `val.json`.

## Quyết định
- GO nếu tiny1 và AP75 cùng tăng.
- NO-GO nếu chỉ recall tăng.

## Điểm
tác động 3, bằng chứng 2, tính mới 3, chi phí 3, rủi ro 4, khả thi 4; tổng 19/30.
