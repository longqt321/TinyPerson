# Chương 1. Mở đầu

## 1.1 Động cơ và bài toán

Phát hiện người tí hon đặt yêu cầu bảo toàn tín hiệu hình ảnh yếu: theo giao thức TinyPerson, kích thước nhóm tiny nằm trong $AS=\sqrt{w\cdot h}\in[2,20]$ px, còn tiny1 là khoảng 2–8 px [1]. Ở mức này, thông tin hình thái trong vùng đối tượng hạn chế, trong khi đặc trưng ảnh còn chịu ảnh hưởng của nền. Kim tự tháp đặc trưng biểu diễn nhiều tỷ lệ và tạo cơ sở cho các đầu dò ở độ phân giải khác nhau [2]. Với đầu dò P2, vấn đề nghiên cứu là cách kết hợp chi tiết không gian độ phân giải cao với ngữ cảnh sâu hơn mà không làm tăng quá nhiều chi phí tính toán.

## 1.2 Mục tiêu và câu hỏi nghiên cứu

Nghiên cứu khảo sát liệu module GauSS, đặt tại tầng P2 của YOLO26n-P2, có thể cải thiện phát hiện người tí hon trên TinyPerson hay không. Đánh giá tập trung vào ba khía cạnh: chất lượng phát hiện theo các dải kích thước, độ nhạy với chế độ suy luận tiled/full-frame, và chi phí tham số cùng suy luận. Các câu hỏi này được kiểm tra bằng so sánh với cấu hình nền trong điều kiện huấn luyện và đánh giá được mô tả ở Chương 4.

## 1.3 Phạm vi và đóng góp

Báo cáo tập trung vào một phương pháp: GauSS trên YOLO26n-P2 [5]. Đóng góp của nghiên cứu là mô tả cơ chế phân rã Gaussian có tham số học được, định tuyến theo nội dung và tinh chỉnh bất đối xứng; đồng thời thiết lập giao thức so sánh trên TinyPerson. Mức hiệu quả định lượng không được khẳng định ở phần mở đầu; kết luận phụ thuộc vào việc xác minh artifact huấn luyện, đánh giá và đo chi phí của run GauSS.

## 1.4 Bố cục báo cáo

Chương 2 tổng hợp nền tảng về phát hiện người tí hon, biểu diễn đa tỷ lệ và detector sử dụng. Chương 3 trình bày kiến trúc GauSS theo luồng dữ liệu. Chương 4 mô tả thiết lập đánh giá, kết quả và các giới hạn của bằng chứng thực nghiệm. Chương 5 tổng kết kết luận trong phạm vi dữ liệu có thể xác minh.
