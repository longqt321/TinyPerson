# Chương 2. Cơ sở lý thuyết

## 2.1 Bài toán phát hiện vật thể cực nhỏ

### 2.1.1 Đặc điểm của bài toán

Phát hiện vật thể cực nhỏ có ít thông tin ảnh trong vùng đối tượng; TinyPerson định nghĩa các nhóm kích thước theo $AS=\sqrt{w\cdot h}$ [1]. Tác giả TinyPerson sử dụng Miss Rate như một độ đo bổ sung cho bài toán phát hiện người [1].

### 2.1.2 Kim tự tháp đặc trưng và tầng P2

Kiến trúc Feature Pyramid Network (FPN) xây dựng biểu diễn đặc trưng đa tỷ lệ [2]. Các mức P2–P5 với stride 4, 8, 16, 32 là cách bố trí trong cấu hình detector dùng ở nghiên cứu này, không phải định nghĩa bắt buộc của FPN. Với ảnh vào 1280 px, đặc trưng P2 có độ phân giải 320×320 trong cấu hình này.

## 2.2 Gaussian scale-space và đặc trưng đa tỷ lệ

Scale-space mô tả ảnh qua một họ biểu diễn được làm trơn ở các mức tỷ lệ; phép Gaussian là nền tảng thường dùng cho biểu diễn này [CẦN NGUỒN: Lindeberg chưa truy cập được trong phiên]. SIFT cũng xây dựng biểu diễn theo tỷ lệ Gaussian để phát hiện đặc trưng ổn định qua thay đổi tỷ lệ [3]. Hai ý tưởng này tạo bối cảnh cho phép làm trơn đa mức, nhưng không tự thân chứng minh rằng tham số Gaussian học được sẽ cải thiện phát hiện người tí hon.

FcaNet tiếp cận thông tin tần số trong bài toán attention kênh bằng phân rã phổ đa thành phần [4]. GauSS khác về mục tiêu và vị trí xử lý: dùng phép làm trơn Gaussian trên đặc trưng không gian ở neck, rồi điều phối các dải sinh ra theo vị trí. Đây là phân biệt giữa hai thiết kế, không phải so sánh hiệu năng.

## 2.3 Detector nền và thiết kế thí nghiệm

YOLO26 là họ detector của Ultralytics với suy luận end-to-end tùy chọn và hồi quy hộp không dùng DFL [5]. Cấu hình P2 được dùng trong nghiên cứu là lựa chọn kiến trúc cụ thể, còn số tham số và các nhánh đầu dò phải lấy từ cấu hình thực nghiệm. TinyPerson phân nhóm đối tượng theo kích thước ảnh; benchmark cũng ghi nhận sự khác nhau giữa các phiên bản quy tắc đánh giá, nên kết quả cần nêu rõ evaluator và giao thức [1]. Báo cáo sử dụng quy ước nội bộ mô tả ở Mục 4.2, chưa xác nhận tương đương với evaluator tham chiếu.
