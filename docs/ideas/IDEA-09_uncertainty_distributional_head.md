# IDEA-09 — Đầu phân phối box có nhận biết bất định

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-09
- trạng thái: đề xuất
- ưu tiên: P1
- nút thắt: sai lệch vị trí gần đúng
- chi phí ước tính: 2–3 lần chạy × 50 epoch trên A10; 3 seed nếu qua ngưỡng
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Một head dự đoán phân phối khoảng cách có phương sai, với bin theo stride, sẽ
tăng AP75 so với scalar regression và DFL8 hiện tại.

## Bằng chứng
- A2 hiện tại dùng `reg_max=1`; scalar regression rất nhạy với 1–2 px.
- SADR8 tăng validation AP75 `+0.003554`, nhưng official-test AP75 giảm
  `-0.000443`; chỉ thêm bin chưa đủ.
- 866 GT near-miss có sai số tâm trung vị `3.81 px`; cần tín hiệu độ bất định.

## Cơ chế
Head dự đoán phân phối `p(d)` cho mỗi cạnh và cả phương sai; loss NLL hoặc
uncertainty-weighted regression. Bin được scale theo stride/level để box lớn
không bị clamp 7 bin. Box cuối là kỳ vọng; uncertainty được dùng để weighting.

## Tính mới và công trình liên quan
DFL là thành phần có sẵn; đóng góp phải là scale-adaptive bin + uncertainty,
không phải “thêm DFL”. Phải so sánh trực tiếp với SADR4/8/16 và scalar.

## Thay đổi tối thiểu
- Sửa head `Detect` để output distribution + variance.
- Thêm loss trong `src/tinydet/training/`.
- Cấu hình: `reg_max`, `bin_mode`, `uncertainty_weight`.

## Giao thức thực nghiệm
- Control: A2 `reg_max=1`.
- T1: SADR8 (dial).
- T2: uncertainty head.
- 50 epoch, cùng data/augmentation/optimizer/GPU A10.

## Chỉ số và ngưỡng
- Chính: AP75; official-test AP75 chỉ sau freeze.
- Ngưỡng: AP75 validation tăng `+0.002`; AP50 không giảm; calibration/NLL tốt hơn.
- Kiểm tra tiny1/tiny2/tiny3.

## Loại bỏ thành phần và bác bỏ
- Bin `4/8/16`; scale-adaptive so với fixed.
- Variance on/off.
- Bác bỏ: nếu AP75 không hơn SADR8, uncertainty không có ích.
- Tiêu chí dừng: dừng nếu validation AP75 không vượt SADR8 sau 2 seed.

## Rủi ro
- Độ phức tạp head/loss cao.
- Có thể overfit uncertainty trên tập nhỏ.
- DFL đã được nghiên cứu; tính mới phải được định nghĩa chặt.

## Tuyên bố trong bài báo
- Nếu dương: “distributional regression thích nghi scale và nhận biết bất định
  cải thiện localization IoU cao cho đối tượng nhỏ”.
- Nếu âm: distributional regression không giải quyết được AP75.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, W&B, calibration plot, `val.json`, test metrics sau freeze.

## Quyết định
- GO nếu AP75 > SADR8 và scalar.
- NO-GO nếu chỉ tăng AP50.

## Điểm
tác động 4, bằng chứng 3, tính mới 3, chi phí 2, rủi ro 4, khả thi 3; tổng 19/30.
