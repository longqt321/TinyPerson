# Tóm tắt: 3 hệ thống tổ hợp + 12 primitive

Đọc [TECHNICAL_DESIGN_GUIDE.md](TECHNICAL_DESIGN_GUIDE.md) và ba file
`COMPOSITE-*.md` trước. Các file `IDEA-*.md` là primitive/ablation, không phải
module nghiên cứu độc lập. Mỗi composite chỉ có một treatment setting; không có
ma trận ablation. Không có kết quả thực nghiệm nào trong tài liệu này.

## Hệ thống tổ hợp

| Mã | Hệ thống | Module 1 | Module 2 | Module 3 | Nút thắt chính |
|---|---|---|---|---|---|
| COMPOSITE-01 | CGLR | Context-Evidence Gate | Candidate Refinement | Quality Ranking | background FP → near-miss → mis-ranking |
| COMPOSITE-02 | CASS | Density Routing | Crowd-Aware Assignment | Duplicate-Aware Set Scoring | crowded miss → positive starvation → duplicate |
| COMPOSITE-03 | SURE | Scale-Conditioned Alignment | Uncertainty Distributional Regression | IoU-Calibrated Confidence | geometric misalignment → pixel sensitivity → score miscalibration |

Mỗi composite có mục 11 khóa trước: shape tensor, RF/hình học sau resize, số
mẫu dương/âm, khởi tạo, gradient budget, optimizer và dự đoán hành vi. Không có
ablation ẩn; mọi con số phải được giữ nguyên trong lần chạy duy nhất.

## Primitive tham chiếu (không train riêng)

| Mã | Ý tưởng | Nút thắt | Cơ chế chính | Thay đổi | Chi phí | Kết luận nhanh |
|---|---|---|---|---|---|---|
| 01 | Gán nhãn TAL có nhận biết tâm | Sai lệch gần đúng | Thêm prior khoảng cách tâm vào TAL | Sửa assigner | 2 lần chạy | Nối tiếp TAL4, nhưng phải chứng minh không chỉ tăng recall |
| 02 | Hồi quy Huber chuẩn hóa theo kích thước | Sai lệch gần đúng | Huber trên `ltrb` chuẩn hóa theo cạnh box | Sửa box loss | 1 lần chạy | Rẻ nhất, trực tiếp nhắm AP75, nên chạy sớm |
| 03 | Đầu tinh chỉnh box nhỏ | Sai lệch gần đúng | RoIAlign/deformable sampling + MLP residual | Thêm module | 2 lần chạy | Module rõ cho bài báo, nhưng latency và NMS là rủi ro |
| 04 | Gán nhãn mềm Gaussian | Bỏ sót | Trọng số mềm theo khoảng cách tâm | Sửa assigner/label | 2 lần chạy | Có thể tăng recall nhưng dễ chỉ là detection |
| 05 | Xếp hạng độ tin cậy theo chất lượng box | Xếp hạng | Target `cls * IoU^gamma` hoặc ranking loss | Sửa classification loss | 1 lần chạy | Nhắm trực tiếp 179 box tốt nhưng confidence thấp |
| 06 | Copy-paste và crop cho đối tượng nhỏ | Bỏ sót/tiny1 | Tăng positive tiny1 bằng augmentation | Sửa pipeline dữ liệu | 2 lần chạy | Ít tính mới, nhưng rẻ và dễ kiểm soát |
| 07 | Căn chỉnh đặc trưng biến dạng trên P2 | Sai lệch gần đúng | Học offset sample trên P2 | Thêm module | 2 lần chạy | Tính mới cao nhưng rủi ro và chi phí cũng cao |
| 08 | Nhánh tần số/biên trên P2 | Bỏ sót | High-pass/edge + residual gate | Thêm module | 2 lần chạy | Giả thuyết yếu, dễ trùng ý nghĩa với MSR |
| 09 | Đầu phân phối box có nhận biết bất định | Sai lệch gần đúng | Phân phối + phương sai, bin theo stride | Sửa head/loss | 2–3 lần chạy | Hấp dẫn nhưng phải thắng SADR8 rõ ràng |
| 10 | Box voting và hiệu chỉnh điểm theo kích thước | Xếp hạng/trùng lặp | WBF/soft-NMS + calibration | Hậu xử lý | 0 train | Rẻ, nhưng không phải đóng góp kiến trúc |
| 11 | Hàm loss cân bằng theo kích thước | Tiny1 | Weight theo nghịch đảo tần suất size bin | Sửa loss weighting | 1–2 lần chạy | Tốt cho tiny1, cần bảo vệ AP75 tổng |
| 12 | Trần Oracle AP75 | Phân tích | Thay box/score bằng oracle để đo headroom | Script phân tích | 0 train | Nên làm trước để biết nên đầu tư vào đâu |

## Thứ tự đề xuất

1. `IDEA-12` — đo trần AP75, không train.
2. `COMPOSITE-01` CGLR — một treatment duy nhất: gate, refinement và ranking
   cùng học; kiểm tra gate có làm sạch evidence và refinement có chuyển near-miss
   thành IoU cao hay không.
3. `COMPOSITE-02` CASS — nếu kết quả chẩn đoán cho thấy miss/duplicate ở ảnh đông
   là phần lớn headroom.
4. `COMPOSITE-03` SURE — chỉ sau khi hai hướng trên xác định regression/calibration
   là nút thắt còn lại.

## Tiêu chí dừng chung

- AP75 không tăng sau 1 seed: dừng biến thể đó.
- Chỉ AP50/recall tăng: không dùng để tuyên bố cải thiện localization.
- Muốn viết bài báo: cần 3 seed paired, cùng GPU SKU, cùng split và checkpoint policy.
- Không chạy ablation cho từng module; nếu treatment đầy đủ thất bại, bác bỏ
  giả thuyết hệ thống.
- Không dùng official test để chọn ý tưởng, threshold hay NMS.
