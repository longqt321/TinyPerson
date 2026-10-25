# Tóm tắt, kết luận và tài liệu tham khảo

## Tóm tắt

**Vị trí:** hai đoạn dưới heading `TÓM TẮT`.

**Vấn đề:** tóm tắt hiện tại mô tả BTRM P3, tám điểm vòng và Tikhonov. Kết quả
được tóm tắt bằng AP@0.50 và APmean, không có AP@0.75. Con số
`AP50 0,2344` và `AP50-95 0,0682` thuộc protocol cũ.

**Cách sửa:** thay toàn bộ hai đoạn bằng bản dưới. Bản này giữ độ dài khoảng
300 từ, đủ để đứng độc lập.

**Nội dung thay thế:**

> Phát hiện người có kích thước cực nhỏ trong ảnh chụp khoảng cách xa gặp khó
> khăn vì thông tin hình thái suy giảm sau các tầng tích chập, trong khi nền
> cảnh rộng và chứa nhiều cấu trúc giống người. Báo cáo dùng TinyPerson [1] và
> YOLO26n [7] để kiểm tra hai câu hỏi. Câu hỏi thứ nhất là liệu tín hiệu tương
> phản cục bộ trên bản đồ P2 có bổ sung thông tin phân biệt cho dải tiny,
> tức là căn bậc hai diện tích hộp từ 1 đến 20 điểm ảnh, hay không. Câu hỏi thứ
> hai là nếu không, lỗi còn lại nằm ở biểu
> diễn đặc trưng hay ở hồi quy hộp.
>
> Nhóm hiện thực BTRM v2 trên P2. Mô-đun tính hai mức trung bình 3 × 3 và 5 ×
> 5, hai đạo hàm riêng theo phương ngang và phương dọc, cùng độ lớn gradient.
> Năm tín hiệu được chiếu bằng tích chập 1 × 1, đi qua BatchNorm và SiLU, rồi
> được điều chế bằng cổng sigmoid. Gamma khởi tạo bằng 0, nên mô-đun là ánh xạ
> đồng nhất tại thời điểm bắt đầu huấn luyện.
>
> Thực nghiệm paired trên TinyPerson official_v1 dùng 786 ảnh test sau khi loại
> 30 ảnh dense, seed 42 và evaluator nội bộ. BTRM không cải thiện AP@0.75.
> Trên test, AP@0.75 giảm từ 0,023745 xuống 0,020200 khi huấn luyện scratch và
> từ 0,024719 xuống 0,019529 khi warm-start từ cùng checkpoint A2. Phân tích
> lỗi cho thấy BTRM giảm 1.437 localization FP nhưng tăng 4.686 background FP
> và mất 146 TP. Hộp lỗi trung vị vẫn hẹp hơn GT và tâm lệch khoảng 0,24 đến
> 0,26 cạnh GT.
>
> Kết quả không ủng hộ H01 ở cấp mô hình. Báo cáo không kết luận H01 sai ở cấp
> đặc trưng vì chưa có AUPRC theo dải kích thước. Kết quả đến từ một seed và
> một evaluator chưa đối chiếu với bản tham chiếu, nên chỉ dùng để so sánh nội bộ.

**Lưu ý:** nếu chỉ còn 250 từ, bỏ đoạn mô tả kiến trúc và giữ ba đoạn còn lại.
Không bỏ câu giới hạn evaluator và một seed.

## Kết luận

**Vị trí:** thay toàn bộ chương 4 hiện tại, 92 từ.

**Vấn đề:** kết luận hiện tại chỉ nói A3 không cải thiện AP@0.50. Không trả lời
Q1, Q2, Q3. Không nêu đóng góp, giới hạn và hướng tiếp theo.

**Cấu trúc thay thế:** bốn mục nhỏ, tổng 700 đến 900 từ.

### 4.1. Kết quả chính

> Q1. YOLO26n A2 đạt AP@0.25 0,401268, AP@0.50 0,230141 và AP@0.75 0,023745
> trên official test. Khoảng cách giữa AP@0.50 và AP@0.75 cho thấy mô hình
> phát hiện được vùng chứa người nhưng hộp chưa đủ chính xác ở IoU cao.
>
> Q2. BTRM v2 không cải thiện AP@0.75. Mức giảm là 0,3545 điểm phần trăm ở
> cặp scratch và 0,5190 điểm phần trăm ở cặp warm-start. Recall tiny1 không
> tăng ổn định trên cả hai cặp. Kết quả không ủng hộ H01 ở cấp mô hình.
>
> Q3. Trong cặp scratch, BTRM giảm localization FP nhưng tăng background FP
> nhiều hơn và làm mất TP. Hộp lỗi trung vị hẹp hơn GT, center offset không
> giảm. Các quan sát này phù hợp với giả thuyết nút thắt nằm ở hồi quy hoặc
> gán nhãn hộp, nhưng chưa chứng minh được nguyên nhân.

### 4.2. Đóng góp

> Báo cáo có ba đóng góp. Thứ nhất, báo cáo xây dựng phép so sánh paired giữa
> A2 và BTRM v2, gồm cả cặp scratch và cặp warm-start. Thứ hai, báo cáo phân
> rã lỗi thành TP, miss, localization FP, background FP và duplicate FP. Thứ
> ba, báo cáo công bố chi phí tham số, độ trễ, bộ nhớ và thời gian huấn luyện.
>
> Báo cáo không công bố một kiến trúc mới. BTRM v2 là công cụ kiểm tra giả
> thuyết. Kết quả âm là kết quả hợp lệ vì nó cho thấy hướng can thiệp này
> không cải thiện AP@0.75 trong protocol đã chạy.

### 4.3. Giới hạn

> Kết quả chỉ có một seed, một dataset và một evaluator nội bộ. Chưa có
> AUPRC theo dải kích thước, chưa có nhiều seed và chưa có VisDrone. Các kết
> luận chỉ có hiệu lực trong protocol đã mô tả.

### 4.4. Hướng tiếp theo

> Hướng thứ nhất là chạy thêm seed 43 và 44 cho cả hai cặp paired, báo mean và
> độ lệch chuẩn. Hướng thứ hai là đo AUPRC ở cấp đặc trưng để kiểm tra H01
> trước khi kết luận về giá trị của tín hiệu cục bộ. Hướng thứ ba là chuyển
> sang hồi quy và gán nhãn hộp, ví dụ RFLA, NWD hoặc một đầu hồi quy phân bố.
> Hướng thứ tư là kiểm tra BTRM trên VisDrone để xem kết quả có lặp lại ngoài
> TinyPerson hay không.

**Bảng cần chèn:** Bảng 4.1, tóm tắt giả thuyết và bằng chứng. Vị trí sau 4.1,
trước 4.2.

> Bảng 4.1. Giả thuyết, phép đo và kết luận.

| Giả thuyết | Phép đo | Kết quả | Kết luận |
|---|---|---|---|
| H01 cấp mô hình | AP@0.75, recall tiny1 | AP@0.75 giảm ở cả hai cặp, một seed | Không ủng hộ trong protocol |
| H01 cấp đặc trưng | AUPRC theo dải | Chưa có | Chưa kết luận |
| Lỗi hộp | Localization FP | Giảm 1.437 ở cặp scratch | Chưa phải nút thắt chính trong cặp này |
| Nhận nhầm nền | Background FP | Tăng 4.686 ở cặp scratch | Cần phân tích riêng |
| Hình học hộp | Width ratio, center offset | Hộp hẹp, tâm lệch | Phù hợp với giả thuyết hồi quy/gán nhãn, chưa kiểm định |

## Tài liệu tham khảo

**Vị trí:** toàn bộ mục `TÀI LIỆU THAM KHẢO`.

**Vấn đề:** có 41 mục nhưng một số mục có thể không được trích trong thân bài.
Metadata của [7] là bản thảo arXiv 2026, cần kiểm tra lại. Cách ghi tên hội
nghị không thống nhất.

**Cách sửa:**

1. Chạy kiểm tra mỗi số `[n]` xuất hiện ít nhất một lần trong thân bài. Bỏ các
   mục không được trích.
2. Giữ thứ tự xuất hiện đầu tiên. Nếu thêm tài liệu mới, đánh lại số.
3. Dùng một kiểu thống nhất: tác giả, tên bài, tên hội nghị hoặc tạp chí, tập,
   số, trang, năm. Không trộn `Kỷ yếu Hội nghị` với `Proceedings`.
4. Với [7], kiểm tra arXiv ID hoặc dùng tài liệu chính thức của Ultralytics.
   Nếu không xác minh được, ghi nguồn là tài liệu kỹ thuật kèm URL và ngày
   truy cập, không ghi `Bản thảo khoa học arXiv` khi chưa chắc.
5. Với [1], dùng đúng tên bài của TinyPerson. Nếu báo cáo dùng cả bài Scale
   Match và benchmark TinyPerson, tách thành hai tài liệu.
6. Thêm tài liệu cho RFLA, NWD và YOLO26 nếu chương 1 và chương 4 nhắc tới.

**Mẫu ghi thống nhất:**

> [1] X. Yu, Y. Gong, N. Jiang, Q. Ye, Z. Han. Scale Match for Tiny Person
> Detection. WACV, 2020, tr. 1257-1265.

**Không làm:** không giữ tài liệu chỉ vì đã có trong danh sách cũ. Không thêm
tài liệu chưa đọc để làm dày danh mục.
