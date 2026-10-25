# IDEA-11 — Hàm loss cân bằng theo kích thước

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-11
- trạng thái: đề xuất
- ưu tiên: P1
- nút thắt: tiny1
- chi phí ước tính: 1–2 lần chạy × 50 epoch trên A10
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Đánh trọng số loss theo size bin sẽ giảm sự thống trị của box lớn và tăng AP75
cho tiny1/tiny2 mà không làm giảm AP75 tổng.

## Bằng chứng
- Tiny1 AP75 khoảng `0.00345` trong A2 baseline cũ; đây là nhóm yếu nhất.
- 1.149 GT không có detection gần; tiny1 recall thấp.
- Loss hiện tại cộng theo từng positive, không chuẩn hóa theo size bin.

## Cơ chế
Weight positive theo nghịch đảo tần suất size bin hoặc chuẩn hóa loss theo bin.
Gradient của tiny1 được tăng tương đối, giảm bias về box lớn. Không đổi head.

## Tính mới và công trình liên quan
Size-balanced loss/focal theo size đã có; đây là ablation cho TinyPerson.
Phải so sánh với class weighting và TAL assignment.

## Thay đổi tối thiểu
- Sửa loss weighting trong `v8DetectionLoss`.
- Cấu hình: `size_loss_weights`.
- Chỉ đổi weight; không đổi assigner/head.

## Giao thức thực nghiệm
- Control: A2 baseline.
- Treatment: size-balanced loss.
- 50 epoch, cùng data/augmentation/optimizer/GPU A10.

## Chỉ số và ngưỡng
- Chính: AP75 tiny1; AP75 tổng.
- Ngưỡng: tiny1 AP75 tăng `+0.002`; AP75 tổng tăng `+0.001`; AP50 không giảm.
- Theo dõi gradient norm theo size bin.

## Loại bỏ thành phần và bác bỏ
- Weight `2/5/10`.
- Per-bin so với continuous inverse-size.
- Bác bỏ: nếu tiny1 tăng nhưng AP75 tổng giảm, đây chỉ là trade-off.
- Tiêu chí dừng: dừng sau 1 seed nếu AP75 tiny1 không tăng `+0.002` hoặc AP75
  tổng giảm.

## Rủi ro
- Overfit tiny1, giảm reasonable/small.
- Mất cân bằng precision/recall.
- Weight quá lớn gây NaN.

## Tuyên bố trong bài báo
- Nếu dương: tối ưu cân bằng theo kích thước cải thiện tiny1 mà không mất AP75 tổng.
- Nếu âm: loss weighting không đủ để thay đổi AP75.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, W&B, chẩn đoán gradient, `val.json` theo bin.

## Quyết định
- GO nếu tiny1 và AP75 tổng cùng tăng.
- NO-GO nếu chỉ dịch chuyển lợi ích giữa các bin.

## Điểm
tác động 3, bằng chứng 2, tính mới 3, chi phí 5, rủi ro 3, khả thi 5; tổng 21/30.
