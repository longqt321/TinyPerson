# IDEA-06 — Copy-paste và crop cho đối tượng nhỏ

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-06
- trạng thái: đề xuất
- ưu tiên: P2
- nút thắt: bỏ sót và tiny1
- chi phí ước tính: 2 lần chạy × 50 epoch trên A10
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Copy-paste có kiểm soát và crop sampling sẽ tăng số positive tiny1, giúp recall
và AP75 tiny1 tăng mà không cần đổi kiến trúc.

## Bằng chứng
- 1.149 GT có best IoU `<0.10`; 31,8% không có detection gần.
- Tiny1 là nhóm yếu nhất; AP75 tiny1 khoảng `0.00345` trong A2 baseline cũ.
- Ảnh đông chiếm phần lớn miss; số positive tiny1 trên mỗi batch thấp.

## Cơ chế
Copy-paste tạo thêm instance nhỏ ở tỷ lệ hợp lý; crop sampling tăng tỷ lệ pixel
thuộc đối tượng nhỏ. P2 nhận nhiều positive hơn, giảm class imbalance và tăng recall.

## Tính mới và công trình liên quan
Copy-paste là augmentation đã biết; đây là ablation cho đối tượng nhỏ, không phải
module. Phải kiểm tra không tạo artifact phi thực tế và không rò rỉ giữa split.

## Thay đổi tối thiểu
- Sửa pipeline augmentation trong `trainer.py` hoặc dataset wrapper.
- Cấu hình: `tiny_copy_paste`, `tiny_crop_prob`.
- Không đổi model, loss, optimizer.

## Giao thức thực nghiệm
- Control: A2 baseline.
- Treatment: copy-paste/crop sampling.
- 50 epoch, cùng seed, imgsz, batch, augmentation cơ sở và GPU A10.
- Kiểm tra visual sample trước khi train.

## Chỉ số và ngưỡng
- Chính: AP75 tiny1; AP75 tổng.
- Ngưỡng: AP75 tiny1 tăng `+0.002`; AP75 tổng tăng `+0.001`; AP50 không giảm.
- Theo dõi recall/precision theo size bin.

## Loại bỏ thành phần và bác bỏ
- Crop only so với copy-paste.
- Tỷ lệ copy-paste `0.1`, `0.3`.
- Bác bỏ: nếu tiny1 tăng nhưng AP75 tổng giảm, đây chỉ là trade-off.
- Tiêu chí dừng: dừng sau 1 seed nếu AP75 tiny1 không tăng `+0.002` hoặc AP75
  tổng giảm.

## Rủi ro
- Composite không tự nhiên, model học artifact.
- Có thể làm tăng label noise và duplicate.
- Không được trộn ảnh giữa train/validation.

## Tuyên bố trong bài báo
- Nếu dương: sampling ở mức dữ liệu cải thiện AP75 tiny1.
- Nếu âm: kiến trúc/loss mới là cần thiết, augmentation không đủ.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, ảnh mẫu, W&B, `val.json` theo size bin.

## Quyết định
- GO nếu AP75 tiny1 tăng và AP75 tổng không giảm.
- NO-GO nếu chỉ recall tăng.

## Điểm
tác động 3, bằng chứng 2, tính mới 2, chi phí 4, rủi ro 3, khả thi 5; tổng 19/30.
