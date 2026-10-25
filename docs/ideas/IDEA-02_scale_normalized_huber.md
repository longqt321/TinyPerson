# IDEA-02 — Hồi quy Huber chuẩn hóa theo kích thước

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-02
- trạng thái: đề xuất
- ưu tiên: P0
- nút thắt: sai lệch vị trí gần đúng
- chi phí ước tính: 1 lần chạy × 50 epoch trên A10; 2 lần nếu cần ablation
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Thay nhánh L1 chuẩn hóa theo ảnh bằng Huber trên `ltrb` chuẩn hóa theo kích
thước box sẽ ổn định gradient cho đối tượng nhỏ và tăng AP75, không dùng dead-zone.

## Bằng chứng
- `reg_max=1`: nhánh L1 hiện tại chia theo imgsz 1280, nên 1 px chỉ đóng góp
  `1/1280`; gradient tọa độ rất nhỏ so với CIoU.
- Các box được phát hiện có sai số tâm trung vị `1.16 px`; nhóm `0.50–0.75` là
  `1.51 px`.
- 866 GT `0.10–0.50` có sai số tâm trung vị `3.81 px`.
- CIoU tiếp tục được giữ; chỉ thay cách đo sai số tọa độ.

## Cơ chế
Huber trên `ltrb` chuẩn hóa theo cạnh box cho mỗi box một gradient có scale
tương đương, kể cả box 4–16 px. Với sai số lớn, Huber tuyến tính thay vì bình
phương; với sai số nhỏ, nó vẫn đẩy về target. Không có vùng loss=0 nên không
khuyến khích model dừng sớm.

## Tính mới và công trình liên quan
Đây là loss engineering, không phải module mới. Khác L1/1280 hiện tại và khác
DFL/SADR: không thêm bin, không đổi head. Nếu kết quả tốt, đóng góp là một
protocol regression cho đối tượng nhỏ chứ không phải thuật toán mới.

## Thay đổi tối thiểu
- Subclass `BboxLoss` hoặc custom criterion trong `src/tinydet/training/`.
- Giữ CIoU; thay nhánh L1 bằng Huber trên `ltrb`.
- Cấu hình: `huber_delta: 0.1`; chuẩn hóa theo `sqrt(area)` hoặc stride.

## Giao thức thực nghiệm
- Control: A2 baseline, `reg_max=1`, L1/1280.
- Treatment: cùng A2, chỉ thay nhánh L1 bằng Huber.
- 50 epoch, imgsz 1280, batch 8, AdamW, cùng augmentation và GPU A10.

## Chỉ số và ngưỡng
- Chính: AP75.
- Ngưỡng: AP75 tăng `+0.0015`; AP50 không giảm quá `0.002`; recall không giảm
  quá `0.005`.
- Theo dõi `box_loss`, `l1_loss`, gradient norm và số positive/GT.

## Loại bỏ thành phần và bác bỏ
- Chuẩn hóa theo stride so với theo `sqrt(area)`.
- Delta `0.05`, `0.1`, `0.2`.
- Bác bỏ: nếu AP75 không tăng, loss không phải nút thắt.
- Tiêu chí dừng: dừng sau 1 seed nếu loss phân kỳ hoặc AP75 giảm.

## Rủi ro
- Sai lệch annotation 2–3 px có thể bị fit quá mức.
- Huber có thể làm gradient quá lớn ở box rất nhỏ; cần clamp.
- AMP và custom loss phải được kiểm tra trên CUDA thật.

## Tuyên bố trong bài báo
- Nếu dương: “loss localization bền vững, chuẩn hóa theo kích thước, cải thiện
  AP75 mà không cần module đặc trưng mới”.
- Nếu âm: reparameterization của loss không đủ; nút thắt nằm ở assignment/detection.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, W&B, `val.json`, `results.csv`, chẩn đoán gradient.

## Quyết định
- GO nếu AP75 tăng và AP50 ổn định.
- NO-GO nếu chỉ cải thiện train loss mà không cải thiện validation.

## Điểm
tác động 4, bằng chứng 4, tính mới 3, chi phí 5, rủi ro 2, khả thi 5; tổng 23/30.
