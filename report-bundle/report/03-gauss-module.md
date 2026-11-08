# Chương 3. Module GauSS

**Thiết kế module.** GauSS thay thế phép ghép đặc trưng tại tầng P2 trong cấu hình nghiên cứu. Backbone Gaussian phân rã hai tensor đầu vào C2 và P3 thành ba thành phần theo độ mịn: $B_h = X-L_1$, $B_m=L_1-L_2$, $B_l=L_2$, trong đó $L_i=G(X;\sigma_i)$. Hai mức làm trơn được tạo bằng tích chập Gaussian tách biệt theo hai trục, với tham số σ học được theo kênh và trục, chặn trong các khoảng cấu hình đã nêu trong mã. Cách phân rã này là định nghĩa thao tác của module; việc tham số σ thay đổi thế nào sau huấn luyện cần kiểm tra trực tiếp trên checkpoint.

**Điều phối và tái tạo.** Nhánh ngữ cảnh kết hợp P3 với P4 đã lấy mẫu lên, rồi áp dụng depthwise convolution 3×3 với dilation 1, 2 và 3; softmax theo vị trí kết hợp các đầu ra. Router nhận thông tin từ các dải và ngữ cảnh, sinh trọng số cho ba dải cùng cổng trộn hai nhánh. Nhánh detail dùng kernel 3×3, nhánh structure dùng kernel 5×5 và giảm đóng góp dải thấp bằng saliency mask. Sau trộn, biểu diễn 32 kênh được chiếu lên 128 kênh và PixelShuffle ×2 để khớp độ phân giải P2. Đầu ra được điều chỉnh bởi hệ số residual bị chặn rồi cộng với đặc trưng fuse ban đầu.

Cấu hình YAML và mã triển khai trong checkout dùng tên GauSS, không phải GauSS. Vì vậy phần mô tả này ghi lại cấu trúc của implementation được đối chiếu, nhưng chưa xác lập rằng tên GauSS là tên công bố chính thức của cùng phiên bản. Cần đối chiếu tên phương pháp, checkpoint và cấu hình thí nghiệm trước khi nộp báo cáo.

[HÌNH CẦN BỔ SUNG: fig-3-1, sơ đồ kiến trúc GauSS từ cấu hình hoặc nguồn cấu trúc được pipeline hỗ trợ. Hiện pipeline chỉ dựng biểu đồ/bảng từ dữ liệu; cần người dùng cung cấp sơ đồ nguồn hoặc định dạng cấu trúc được hỗ trợ.]
