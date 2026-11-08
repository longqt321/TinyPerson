# Chương 4. Thiết lập và kết quả thực nghiệm

## 4.1 Thiết kế đánh giá

Đánh giá được tổ chức theo ba câu hỏi: liệu GauSS cải thiện AP và Miss Rate theo kích thước đối tượng trên TinyPerson; liệu kết quả thay đổi giữa tiled và full-frame inference; và mức chi phí tính toán của module khi đặt trong YOLO26n-P2. So sánh dự kiến dùng cùng dữ liệu, huấn luyện và evaluator cho baseline và GauSS.

## 4.2 Dữ liệu, mô hình và phép đo

Bản tổng hợp mô tả TinyPerson Official v1 với 662 ảnh train, 84 validation và 786 ảnh test sau khi loại 30 ảnh dense khỏi 816 ảnh ban đầu. Test được ghi nhận có 13.687 ground-truth dương, 1.989 ignore và 2.832 uncertain. Các chi tiết này chưa đối chiếu được với manifest dữ liệu hoặc run artifact trong bundle; chúng được giữ ở đây như giao thức được báo cáo, không phải xác minh độc lập.

Thiết lập tiled được mô tả với tile 640×512, overlap 100 px, ánh xạ hộp về ảnh gốc và global NMS tại IoU 0,50 (`nms_pre=2000`, `max_per_img=1000`). Chế độ đối chiếu thứ hai là single-shot full-frame ở 1280×1280. Kích thước được tính bằng $AS=\sqrt{w\cdot h}$; nhóm tiny1 [2,8), tiny2 [8,12), tiny3 [12,20], tiny [2,20], small (20,32] và reasonable >32 px. Các ngưỡng này theo bản tổng hợp đánh giá; evaluator và quy tắc biên cần được cung cấp để tái lập chính xác.

Bản tổng hợp cũng ghi baseline YOLO26n-P2 và GauSS ở 150 epoch, seed 42, batch 8, AdamW, AMP, `lr0=0.002`, `lrf=0.1`, cosine schedule, warmup 5 epoch, `val=false`. Tuy nhiên, config, log, checkpoint và lệnh đo của run GauSS chưa có trong bundle. Bởi vậy, không thể xác nhận tính đồng nhất của các điều kiện hoặc khôi phục các số liệu thực nghiệm chỉ từ bản tổng hợp.

## 4.3 Kết quả được báo cáo

Bản tổng hợp nêu mức tăng AP50 lớn nhất ở tiny2, từ 6,90% lên 9,50%, cùng các kết quả AP theo nhóm kích thước, Miss Rate, full-frame, mật độ cảnh và chi phí GPU. Các con số này không có CSV evaluator, log profiler hoặc output model-info tương ứng trong bundle. Vì vậy chúng không được diễn giải như phát hiện đã kiểm chứng và chưa thể trả lời ba câu hỏi nghiên cứu. Cần artifact gốc trước khi có thể trình bày bảng/biểu đồ hoặc kết luận định lượng.

[HÌNH CẦN BỔ SUNG: fig-4-1, biểu đồ AP theo nhóm kích thước và chế độ suy luận. Cần CSV evaluator nguồn cùng metadata trong data/ và khai báo trong figures/specs.json.]

## 4.4 Diễn giải và giới hạn

Tài liệu tổng hợp ban đầu gợi ý kết quả thay đổi theo bin kích thước và chế độ inference, nhưng do thiếu artifact gốc, không thể phân biệt hiệu ứng của module với sai khác evaluator, cấu hình, checkpoint hoặc quy trình đo. Cũng chưa có nhiều seed, ablation hoặc đối chiếu độc lập với evaluator tham chiếu. Kết luận hiện tại vì thế giới hạn ở mô tả kiến trúc; hiệu quả của GauSS trên TinyPerson vẫn là câu hỏi thực nghiệm chưa được xác lập trong bộ hồ sơ này.

## 4.5 Tái lập và tài liệu cần bổ sung

Để hoàn tất đánh giá, cần bổ sung config model/train/eval, checkpoint baseline và GauSS, log huấn luyện, kết quả evaluator đầy đủ, script hoặc lệnh inference, cùng profiler output và thông tin GPU cho phép tái lập latency, FLOPs và VRAM. Cần xác nhận thêm quan hệ giữa tên GauSS trong báo cáo và implementation mang tên GauSS trong checkout. Khi dữ liệu nguồn có mặt, có thể dựng biểu đồ và bảng bằng pipeline sẵn có; hiện chưa khai báo hoặc sinh hình/bảng định lượng.
