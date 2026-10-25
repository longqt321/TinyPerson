# Sprint ý tưởng — cải thiện AP75 cho TinyPerson

Mục đích: tạo một backlog ý tưởng có thể bác bỏ được, trong 30 phút, **không train**.
Mỗi ý tưởng là một file riêng theo [TEMPLATE.md](TEMPLATE.md). Đây là tài liệu
đề xuất, không phải kết quả thực nghiệm.

Đọc [TECHNICAL_DESIGN_GUIDE.md](TECHNICAL_DESIGN_GUIDE.md) trước để biết khung
phân tích: hợp đồng tensor, initialization, gradient, receptive field, tỷ lệ
dương/âm, optimizer, gating và cách dự đoán hành vi trước khi train.

Quy ước ngôn ngữ: phần giải thích viết bằng tiếng Việt có dấu; các thuật ngữ
chuẩn như AP75, TAL, DFL, NMS, WBF, P2, GT, FP được giữ nguyên để tránh dịch
sai nghĩa.

## Bằng chứng chung

Nguồn: lần chạy `tal_e0_a2/20261024-115214-8efefb4-seed42` (A10, 50 epoch) và
phân tích lỗi trên validation 84 ảnh.

- AP `0.059396`; AP50 `0.190028`; AP75 `0.016198`; precision `0.326866`;
  recall `0.258945`.
- 3.608 GT; best IoU: `>=0.75`: 436; `0.50–0.75`: 1.157; `0.10–0.50`: 866;
  `<0.10`: 1.149.
- Sai số tâm trung vị: nhóm `>=0.75` là `0.94 px`; `0.50–0.75` là `1.51 px`;
  `0.10–0.50` là `3.81 px`; `<0.10` là `403.87 px`.
- Trong 436 box có `IoU>=0.75`, 179 box có confidence `<0.25`.
- MSR ở beta=6: AP75 giảm `0.001381`; AP giảm `0.004469`; recall tăng `0.006853`.
- TAL beta=4: AP75 đổi `-0.000172`; AP50 tăng `0.002288`; precision tăng
  `0.009645`; recall tăng `0.010456`.
- SADR8: validation AP75 tăng `0.003554`, official-test AP75 giảm `0.000443`.

Diễn giải: các box đã đạt IoU cao thường đã lệch tâm dưới 1–2 px. Nút thắt lớn
hơn là 55,8% GT có best IoU `<0.50`, trong đó 31,8% gần như không có detection
gần.

## Luật chơi

1. Mỗi experiment chỉ đổi một biến so với control.
2. Cùng dataset, split, seed, imgsz, optimizer, augmentation và GPU SKU.
3. Không trộn A10 với A10G trong một paired comparison.
4. Chọn `best.pt` trên validation; official test chỉ chạy sau khi freeze.
5. AP75 là chỉ số chính; AP50/AP trung bình chỉ là guardrail.
6. Một seed là tín hiệu; muốn tuyên bố phải có ít nhất 3 seed paired.
7. Kết quả âm được giữ; không đổi metric hoặc xóa lần chạy.
8. Không dùng test để tune kiến trúc, loss, threshold hay NMS.
9. Mỗi composite chỉ train **một treatment setting**; control là A2 có sẵn.
   Không train ma trận ablation cho từng module.

## Thiết kế tổ hợp — hướng chính

Các `IDEA-01..12` là primitive/ablation, không phải sản phẩm nghiên cứu độc lập.
Ba thiết kế dưới đây ghép 2–3 module theo chuỗi nhân quả; mỗi module xử lý một
lỗi còn lại của module trước.

| Mã | Hệ thống | Chuỗi module | Vấn đề nối tiếp |
|---|---|---|---|
| COMPOSITE-01 | CGLR | Context-Evidence Gate → Candidate Refinement → Quality Ranking | BTRM/MSR tăng background FP → candidate near-miss → box tốt bị xếp hạng thấp |
| COMPOSITE-02 | CASS | Density Routing → Crowd-Aware Assignment → Duplicate-Aware Set Scoring | Ảnh đông gây miss → positive starvation → duplicate và `max_det` bão hòa |
| COMPOSITE-03 | SURE | Scale-Conditioned Alignment → Uncertainty Distributional Regression → IoU-Calibrated Confidence | Feature lệch hình học → tọa độ nhạy pixel → score không phản ánh IoU |

Điểm mới phải được mô tả ở **cách các module ràng buộc lẫn nhau**, không phải ở
việc tái sử dụng MSR, DFL, IoU-aware ranking hay soft assignment.

Mỗi file composite có thêm **mục 11 — Đặc tả kỹ thuật chi tiết và pre-registration**:
hợp đồng tensor, hình học sau resize, số mẫu positive/negative, khởi tạo bằng
con số, gradient budget, optimizer, chẩn đoán bắt buộc và dự đoán hành vi trước
khi train. Đây là phần dùng để bác bỏ giả thuyết, không phải danh sách tính năng.

## Chấm điểm

| Mã | Ý tưởng | Nút thắt | Tác động | Bằng chứng | Tính mới | Chi phí | Rủi ro | Khả thi | Tổng |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | Gán nhãn TAL có nhận biết tâm | sai lệch gần đúng | 4 | 3 | 3 | 3 | 3 | 4 | 20 |
| 02 | Hồi quy Huber chuẩn hóa theo kích thước | sai lệch gần đúng | 4 | 4 | 3 | 5 | 2 | 5 | 23 |
| 03 | Đầu tinh chỉnh box nhỏ | sai lệch gần đúng | 4 | 3 | 4 | 2 | 4 | 3 | 20 |
| 04 | Gán nhãn mềm Gaussian | bỏ sót | 3 | 2 | 3 | 3 | 4 | 4 | 19 |
| 05 | Xếp hạng độ tin cậy theo chất lượng box | xếp hạng | 4 | 3 | 4 | 4 | 3 | 4 | 22 |
| 06 | Copy-paste và crop cho đối tượng nhỏ | bỏ sót/tiny1 | 3 | 2 | 2 | 4 | 3 | 5 | 19 |
| 07 | Căn chỉnh đặc trưng biến dạng trên P2 | sai lệch gần đúng | 3 | 1 | 4 | 2 | 4 | 2 | 16 |
| 08 | Nhánh tần số/biên trên P2 | bỏ sót | 2 | 1 | 3 | 3 | 4 | 3 | 16 |
| 09 | Đầu phân phối box có nhận biết bất định | sai lệch gần đúng | 4 | 3 | 3 | 2 | 4 | 3 | 19 |
| 10 | Box voting và hiệu chỉnh điểm theo kích thước | xếp hạng/trùng lặp | 3 | 3 | 2 | 5 | 2 | 5 | 20 |
| 11 | Hàm loss cân bằng theo kích thước | tiny1 | 3 | 2 | 3 | 5 | 3 | 5 | 21 |
| 12 | Trần Oracle AP75 | phân tích | 5 | 5 | 2 | 5 | 1 | 5 | 23 |

## Danh sách ưu tiên

**Bước 0 — chẩn đoán, không train**

- `IDEA-12` Trần Oracle AP75: đo bao nhiêu headroom đến từ box, bao nhiêu từ xếp hạng.
- Dùng kết quả này để chọn giữa CGLR (localization/ranking), CASS (crowded/miss)
  và SURE (regression/calibration).

**Hướng chính — hệ thống tổ hợp**

1. `COMPOSITE-01` CGLR: gần nhất với chuỗi lỗi đã đo (BTRM background FP →
   near-miss → mis-ranking) và có thiết kế kỹ thuật rõ nhất.
2. `COMPOSITE-02` CASS: nhắm 90,7% miss nằm trong ảnh đông và `max_det` bão hòa.
3. `COMPOSITE-03` SURE: đắt nhất, chỉ làm sau khi CGLR/CASS cho thấy regression
   hoặc calibration là nút thắt còn lại.

**Primitive tham chiếu**

- `IDEA-01..11` chỉ dùng như thành phần hoặc control tham chiếu. Không train
  riêng từng primitive và không chạy ablation cho composite.

## Kế hoạch 30 phút

- 0–5 phút: chốt template, chấm điểm và danh sách ưu tiên.
- 5–20 phút: hoàn thiện 12 primitive + 3 thiết kế tổ hợp, mỗi file một trang.
- 20–30 phút: kiểm tra tính nhất quán, chọn top 3, ghi rõ tiêu chí dừng.

## Trạng thái thực thi

- Không train, không chạy GPU, không sửa dataset.
- Các file trong thư mục này là giả thuyết và giao thức đề xuất.

## Kết quả audit 30 phút

- 12/12 file primitive có đủ các mục của template.
- 12/12 file primitive có ngưỡng, điều kiện bác bỏ và tiêu chí dừng tường minh.
- Điểm tổng trong từng file primitive khớp với bảng chấm điểm.
- 3/3 file `COMPOSITE-*` có 2–3 module, hợp đồng tensor, initialization, gradient,
  receptive field, tỷ lệ dương/âm, optimizer, gating, dự đoán hành vi và tiêu chí
  bác bỏ.
- 3/3 file `COMPOSITE-*` chỉ định đúng một treatment setting, không có bảng
  ablation.
- Ba thiết kế tổ hợp đều nêu rõ module sau xử lý lỗi còn lại của module trước,
  không chỉ liệt kê kỹ thuật đã biết.
- Không còn placeholder ngoài `TEMPLATE.md`.
- Lần chạy baseline `tal_e0_a2/20261024-115214-8efefb4-seed42` chưa thấy trong
  `outputs/` local; số liệu đang dựa trên artifact/lần chạy trên Modal. Trước khi
  viết bài báo, cần tải `metadata.json`, `results.csv` và validation artifact về
  local để kiểm chứng lại nguồn gốc.
