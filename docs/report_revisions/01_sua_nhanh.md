# Sửa nhanh toàn báo cáo

File này gom các lỗi lặp lại ở nhiều chương. Làm trước để tránh sửa lại nhiều
lần.

## 1. Chọn phiên bản BTRM

**Vị trí:** toàn báo cáo, đặc biệt 2.3, 2.4, 2.5, 3.6 và kết luận.

**Hiện trạng:** báo cáo mô tả BTRM trên P3, tám điểm vòng bán kính r, phép
chiếu Tikhonov, hệ số điều chế bị chặn trong [0,5; 1,5]. Mã nguồn hiện tại là
BTRM v2 trên P2, dùng hai mức trung bình 3 × 3 và 5 × 5, gradient ngang và dọc,
phép chiếu 1 × 1, cổng sigmoid và gamma khởi tạo bằng 0.

**Cách sửa:** chọn BTRM v2 trên P2. Xóa toàn bộ mô tả tám điểm vòng, Tikhonov,
CFAR, chuẩn hóa phương sai và hệ số điều chế bị chặn. Thay bằng nội dung trong
`02_chuong_2.md`. Xóa các câu trong chương 1 nhắc đến P3 như vị trí tích hợp.

**Lý do:** không được để chương 2 mô tả một kiến trúc khác với mã đã chạy. Nếu
giữ BTRM P3, toàn bộ kết quả chương 3 sẽ không khớp.

## 2. Sửa số liệu dữ liệu

**Vị trí:** 3.2, câu bắt đầu `Nghiên cứu sử dụng bộ dữ liệu TinyPerson`.

**Hiện trạng:** báo cáo ghi tập xác thực 322 ảnh và 12.017 khung. Con số này
không khớp protocol hiện tại.

**Cách sửa:** thay bằng:

> Bản derivative dùng để huấn luyện có 746 ảnh, chia thành 662 ảnh huấn luyện
> và 84 ảnh validation. Tập kiểm tra chính thức có 816 ảnh. Theo protocol
> TinyPerson, 30 ảnh dense bị loại khỏi phép chấm, còn 786 ảnh và 13.687 khung
> tham chiếu dương tính.

**Lý do:** `docs/agents/DATA_EVALUATION.md` ghi 746 ảnh, 662 train, 84
validation. `docs/BTRM_REPORT.md` ghi 786 ảnh test sau khi loại 30 ảnh dense.

## 3. Sửa tên chỉ số

**Vị trí:** toàn báo cáo.

**Cách sửa:**

- `AP50` thành `AP@0.50`.
- `AP50–95` hoặc `AP50-95` trong bảng cũ phải xóa cùng các giá trị cũ. Không
  đổi nhãn cột AP50–95 thành APmean vì hai metric khác nhau.
- Trên official test, cột trung bình ba ngưỡng ghi là `APmean`.
- Trên validation, cột mAP chuẩn của Ultralytics ghi là `AP@[0.50:0.95]`.
- Thêm `AP@0.25` và `AP@0.75` vào mọi bảng so sánh chính.

APmean trong báo cáo này là trung bình cộng của AP@0.25, AP@0.50 và AP@0.75,
khớp khóa `AP_mean_IoU_0.25_0.50_0.75` trong artifact. Không dùng các giá trị
AP50–95 cũ của COCO cho cột này. Hai metric khác nhau và không được trộn.

**Lý do:** AP@0.25 và AP@0.75 là bằng chứng trung tâm cho câu hỏi localization.
Chỉ báo AP@0.50 làm mất đúng phần kết quả cần giải thích.

## 4. Sửa cách ghi chênh lệch

**Vị trí:** 3.4 đến 3.9, kết luận.

**Hiện trạng:** câu `thấp hơn A0 khoảng 1,085 điểm phần trăm` dùng đúng đơn vị
nhưng các câu khác có thể ghi `giảm 1,085%`. Cách ghi không thống nhất.

**Cách sửa:** mọi chênh lệch tuyệt đối giữa hai giá trị AP ghi là `điểm phần
trăm`. Chênh lệch tương đối mới ghi `%`. Ví dụ:

> AP@0.75 giảm từ 0,023745 xuống 0,020200, tức giảm 0,3545 điểm phần trăm.

Không viết `giảm 0,35%` nếu người đọc có thể hiểu là giảm 0,35 điểm AP.

## 5. Xóa đoạn lặp

**Vị trí:** 2.4, hai đoạn liên tiếp bắt đầu bằng `Tám điểm lấy mẫu này cho phép
xấp xỉ không gian lân cận ngay sát đối tượng.`

**Cách sửa:** xóa một trong hai đoạn. Nếu đã thay toàn bộ 2.4 bằng BTRM v2 thì
không cần xử lý riêng.

**Vị trí:** 2.2, hai đoạn gần nhau cùng nói `nhóm đã hoàn tất huấn luyện và ghi
nhận kết quả của ba cấu hình A0, A2 và A3`.

**Cách sửa:** giữ một đoạn, gộp thông tin về A1, A4, A5 vào một câu.

## 6. Sửa caption lỗi

**Vị trí:** 2.3, caption dưới hình kiến trúc.

**Hiện trạng:** `Hình 2.1Vịuy í tícìnhợpủa BTRM trên nhánh P3`.

**Cách sửa:** thay bằng:

> Hình 2.1. Vị trí tích hợp BTRM trên nhánh P2 của YOLO26n. BTRM nhận đầu vào
> từ layer 19 và truyền sang Detect cùng P3, P4, P5.

## 7. Sửa cách gọi tên mô-đun

**Vị trí:** 2.1, 2.3, 3.5, kết luận.

**Cách sửa:** dùng thống nhất `BTRM v2`. Lần đầu xuất hiện, ghi tên mô tả lấy
từ docstring của mã nguồn:

> BTRM v2 (Boundary-aware Multi-scale Trend Removal).

Tên này mô tả mô-đun loại xu thế đa tỉ lệ có nhận biết biên. Không tự đặt một
expansion khác nếu nhóm nghiên cứu chưa chốt tên chính thức.

Không dùng `BTRM đầy đủ`, `BTRM cải tiến`, `BTRM mới` vì các tên này không nói
được khác biệt gì so với phiên bản cũ.

## 8. Bỏ khẳng định chưa có bằng chứng

Các câu sau phải xóa hoặc hạ mức:

- `BTRM tạo thêm ứng viên cho nhóm đối tượng cực nhỏ`: chỉ giữ nếu kèm số
  recall tiny1 và nói rõ đó là recall tại conf 0,001.
- `phù hợp với khả năng giả định nền tuyến tính bậc nhất còn hạn chế`: xóa vì
  BTRM v2 không còn giả định nền tuyến tính.
- `chứng minh`, `chứng thực`, `xác nhận H01`: thay bằng `cho thấy`, `phù hợp
  với`, `chưa đủ để kết luận`.
- `BTRM thất bại`: thay bằng `BTRM không cải thiện AP@0.75 trong protocol hiện
  tại`.

## 9. Sửa quy ước viết tắt

Thêm bảng viết tắt ngay sau mục lục, trước chương 1. Bảng gồm:

| Viết tắt | Tiếng Anh | Tiếng Việt |
|---|---|---|
| AP | Average Precision | Độ chính xác trung bình |
| TP | True Positive | Phát hiện đúng |
| FP | False Positive | Phát hiện nhầm |
| FN | False Negative | Bỏ sót |
| GT | Ground Truth | Khung tham chiếu |
| IoU | Intersection over Union | Tỷ số giao trên hợp |
| P2 | Pyramid level 2 | Bản đồ đặc trưng mức 2 |
| BTRM | Boundary-aware Multi-scale Trend Removal | Loại xu thế đa tỉ lệ nhận biết biên |

## 10. Kiểm tra lại mục lục và danh mục

Sau khi sửa, cập nhật lại:

- Số chương và số mục.
- Số trang trong mục lục.
- Danh mục hình.
- Danh mục bảng.

Không để mục lục trỏ tới heading đã xóa. Không để hình trong danh mục nhưng
không có trong thân bài.
