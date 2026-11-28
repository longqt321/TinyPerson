# BÁO CÁO ĐỒ ÁN MÔN HỌC

## THỊ GIÁC MÁY TÍNH

**Đề tài:** GauCon: Đầu phát hiện đồng thuận Gauss cục bộ và gán nhãn Wasserstein tách rời cho phát hiện người cực nhỏ trên ảnh hàng hải

**Giảng viên hướng dẫn:** TS. Huỳnh Hữu Hùng

**Nhóm sinh viên thực hiện:**

- Trần Đức Long
- Phan Thị Thảo Tiên

**Đơn vị:** Trường Đại học Bách khoa, Đại học Đà Nẵng

**Năm học:** 2026–2027

---

## MỤC LỤC

## TÓM TẮT

TinyPerson là bộ dữ liệu phát hiện người ở bối cảnh ven biển, trong đó nhiều đối tượng chỉ chiếm 2–20 pixel và thường xuất hiện trên nền sóng, bọt nước hoặc vùng phản xạ. Các bộ phát hiện một giai đoạn sử dụng nhánh gán nhãn đơn mục tiêu có thể suy luận không cần Non-Maximum Suppression (NMS). Tuy nhiên, hồi quy hộp tại từng ô độc lập chưa khai thác đầy đủ thông tin hình học từ các ô lân cận. Việc thay trực tiếp độ chồng lấn bằng độ tương đồng Wasserstein trong bộ gán nhãn căn chỉnh tác vụ cũng có thể tạo mục tiêu phân loại cao cho các dự đoán có IoU thấp.

Báo cáo đề xuất GauCon, đầu phát hiện đồng thuận Gauss cục bộ, kết hợp với DW-TAL trên YOLO26n-P2. GauCon biểu diễn hộp dự đoán bằng phân phối Gauss hai chiều, đưa các ô lân cận và ô cha ở tầng đặc trưng kế tiếp về cùng hệ tọa độ, sau đó tính trọng số dựa trên khoảng cách Wasserstein chuẩn hóa theo phương sai cục bộ. Hộp được hiệu chỉnh bằng một bước nội suy hướng về trọng tâm Wasserstein. Nhánh phân loại sử dụng thông tin quan hệ giữa các ô để điều chỉnh logit. Trong huấn luyện, DW-TAL dùng độ tương đồng Wasserstein để hỗ trợ chọn ứng viên dương, nhưng hiệu chuẩn mục tiêu phân loại theo IoU thực tế. Nhánh giám sát đa mẫu cung cấp thêm tín hiệu chưng cất trong huấn luyện và được loại bỏ khi chuyển sang suy luận.

Các mô hình được huấn luyện trong 80 epoch, khởi tạo từ trọng số COCO, với batch size 16 và seed 42 trên NVIDIA L4. Trên tập kiểm tra TinyPerson gồm 786 ảnh và 13.687 hộp người hợp lệ, GauCon có thêm 714 tham số, nâng tổng số tham số sau hợp nhất lên 2.402.374. Với giao thức lát cắt $640 \times 512$ có NMS khi ghép kết quả giữa các lát, $AP_{50}^{all}$ tăng từ 45,58% lên 48,53% (+2,95 điểm phần trăm) so với mô hình cơ sở; $AP_{75}^{all}$ đạt 7,37%, cao hơn cấu hình NWD 1,28 điểm. Với suy luận toàn khung $1280 \times 1280$ không dùng NMS, $AP_{50}^{all}$ tăng từ 43,08% lên 44,13%. Chênh lệch AP50 giữa GauCon và NWD ở cả hai giao thức chưa có bằng chứng bootstrap đủ để kết luận là khác biệt có ý nghĩa. Kết quả cũng cho thấy GauCon còn hạn chế ở nhóm tiny1 và vẫn làm tăng báo nhầm nền trong giao thức lát cắt.

**Abstract.** TinyPerson contains distant people in maritime scenes, many of whom occupy only 2–20 pixels against waves, foam, and reflections. Although one-to-one detectors can operate without Non-Maximum Suppression (NMS), independent box regression at each feature-map cell does not use geometric evidence from neighboring cells. Directly replacing overlap with Wasserstein similarity in task-aligned assignment can also assign high classification targets to low-IoU predictions. This report presents GauCon, a local Gaussian consensus head, and DW-TAL on YOLO26n-P2. GauCon maps predicted boxes to diagonal 2D Gaussians, aligns spatial neighbors and a parent cell in a common coordinate frame, and applies damped interpolation toward a local Wasserstein barycenter. DW-TAL uses Wasserstein similarity during positive-sample selection and calibrates classification targets against true IoU. The models were trained for 80 epochs from COCO-pretrained weights with batch size 16 and seed 42. On the 786-image test set, GauCon adds 714 parameters. Under tiled $640\times512$ evaluation, which uses NMS to merge overlapping crops, $AP_{50}^{all}$ increases from 45.58% to 48.53% (+2.95 percentage points), while $AP_{75}^{all}$ reaches 7.37%. Under single-pass full-frame $1280\times1280$ inference without NMS, $AP_{50}^{all}$ increases from 43.08% to 44.13%. Bootstrap intervals do not establish a clear $AP_{50}$ advantage over NWD in either protocol. GauCon also trails NWD on the smallest size group and increases background false positives in tiled evaluation.

**Từ khóa:** phát hiện người cực nhỏ; bộ phát hiện đầu-cuối không dùng NMS; trọng tâm Wasserstein cục bộ; gán nhãn tách rời DW-TAL; TinyPerson; YOLO26n.

---

## 1. TỔNG QUAN NGHIÊN CỨU LIÊN QUAN

### 1.1. Bài toán phát hiện người cực nhỏ trên ảnh hàng hải

Trong các chiến dịch tìm kiếm cứu nạn trên biển và giám sát bờ biển bằng UAV, camera góc rộng đặt ở độ cao hàng chục đến hàng trăm mét thu nhận các khung hình kích thước lớn (lên tới $3840 \times 2160$ pixel), trong đó mỗi người đang bơi hoặc trôi nổi chỉ chiếm một vùng rất nhỏ [1]. Khác với người đi bộ trên đường phố có kích thước hàng trăm pixel và rõ nét cấu trúc đầu, thân, tay chân, một người ở xa trên mặt biển mất hầu hết chi tiết hình thái và có cường độ sáng gần trùng với dải bọt sóng bạc đầu hoặc vùng phản chiếu ánh nắng.

![](docs/report/figures/r_gt_distribution.png)

Hình 1.1: Phân bố kích thước *ground truth* theo cạnh trung bình hình học trong tập kiểm tra TinyPerson [1]

Hình 1.1 thống kê phân bố cạnh trung bình hình học $s = \sqrt{w \cdot h}$ của 13.687 hộp Ground Truth người trên 786 khung hình kiểm tra của bộ dữ liệu TinyPerson [1]. Có tới 64,9% số đối tượng thật (8.878 cá thể) nhỏ hơn hoặc bằng 20 pixel (*tiny*), trong đó 17,0% (2.333 cá thể) thuộc nhóm cực nhỏ 2–8 pixel (*tiny1*) và 20,3% (2.778 cá thể) nằm trong dải 8–12 pixel (*tiny2*). Ngoài kích thước nhỏ, các cảnh ven bờ thường chứa từng nhóm người tập trung sát nhau với khoảng cách giữa hai tâm hộp gần nhất dưới 10 pixel (2.583 cá thể trong tập kiểm tra).

Hai đặc điểm này tạo ra ràng buộc trái chiều khi thiết kế mạng phát hiện cho UAV. Về độ chính xác, các mạng phát hiện huấn luyện trên MS COCO [20] thực hiện hạ mẫu xuống bước lưới 8, 16 và 32, khiến một đối tượng rộng 5 pixel bị thu về 0,63 đến 0,16 ô lưới và hòa lẫn vào nền biển. Về phần cứng, thiết bị tính toán trên UAV bị giới hạn chặt về công suất và độ trễ, khiến các mạng hai giai đoạn nặng hàng chục triệu tham số hoặc phụ thuộc hậu xử lý tuần tự trên CPU khó đáp ứng tốc độ xử lý trực tuyến.

### 1.2. Các hướng tiếp cận đã có

#### 1.2.1. Bộ phát hiện hai giai đoạn và căn chỉnh phân bố tỉ lệ tiền huấn luyện

Các nghiên cứu mở đầu trên TinyPerson sử dụng bộ phát hiện hai giai đoạn Faster R-CNN [21] và Cascade R-CNN [26] với mạng trích chọn đặc trưng (Backbone) ResNet-50 tiền huấn luyện (*pretrained*) trên ImageNet. Yu và cộng sự [1] chỉ ra sự chênh lệch lớn giữa phân bố kích thước vật thể trong ImageNet/COCO (trung bình trên 100 pixel) với TinyPerson (trung bình dưới 18 pixel), qua đó đề xuất phép biến đổi histogram kích thước Scale Match giúp đưa $AP_{50}^{tiny}$ của Faster R-CNN-FPN lên 50,89%. Jiang và cộng sự mở rộng thành SM+ [2] bằng tinh chỉnh ở mức cá thể và giả nhãn (52,61% $AP_{50}^{tiny}$). Trên khối tổng hợp đặc trưng (Neck) FPN [22], Gong và cộng sự [3] thay tỷ lệ ghép mặc định 1:1 bằng hệ số ghép thích nghi để hạn chế tầng sâu lấn át tầng nông, trong khi Hong và cộng sự xây dựng SSPNet [4] lọc đặc trưng theo dải tỉ lệ để đạt 59,13% $AP_{50}^{tiny}$.

Nhóm mô hình hai giai đoạn giữ độ ổn định định vị cao nhờ RoIAlign và hai bước hồi quy nối tiếp, nhưng đòi hỏi từ 32 triệu đến hơn 60 triệu tham số và buộc phải suy luận trên các mảnh cắt nhỏ, dẫn đến tốc độ chậm khi triển khai trên thiết bị biên.

#### 1.2.2. Mở rộng tầng bước lưới 4 ($P_2$) trên bộ phát hiện một giai đoạn cỡ nano

Để giữ kích thước mạng dưới 3 triệu tham số mà vẫn bảo toàn độ phân giải không gian cho đối tượng 4–12 pixel, các biến thể YOLO gọn nhẹ bổ sung nhánh đầu ra $P_2$ ở bước lưới 4 vào tháp đặc trưng FPN–PANet [22, 24]. Trên TinyPerson theo thiết lập một lớp *person*, Chao và cộng sự [5] ghi nhận YOLOv8n+P2 (2,93 triệu tham số) đạt 38,0% $mAP_{50}$ và kiến trúc BPD-YOLO (1,50 triệu tham số) đạt 39,1% $mAP_{50}$ sau 300 epoch huấn luyện từ đầu (*scratch*).

Nhánh $P_2$ tăng đáng kể mật độ lấy mẫu không gian với lượng tham số nhỏ. Tuy vậy, các mạng như YOLOv8 và YOLO11 [33] vẫn dùng cơ chế gán nhãn đa mẫu (*one-to-many*) khi huấn luyện và phụ thuộc thuật toán NMS tham lam [30] khi suy luận. Với ngưỡng cố định $\mathrm{IoU} = 0{,}50$ trên đám đông ven biển, NMS dễ xóa nhầm một trong hai người đứng sát nhau khi hộp dự đoán giao nhau quá ngưỡng, hoặc giữ lại nhiều dự đoán lệch 1–2 pixel cho cùng một người nhỏ (khi $\mathrm{IoU} < 0{,}50$) làm tăng lỗi hộp trùng lặp ($E_{\mathrm{dupe}}$).

#### 1.2.3. Bộ phát hiện đầu-cuối không dùng NMS và bài toán đồng thuận hộp bao

DETR [12] và bản tăng tốc thời gian thực RT-DETR [11] loại bỏ bước NMS bằng cơ chế tự chú ý toàn cục giữa các truy vấn vật thể, cho phép các truy vấn trao đổi thông tin vị trí và độ tin cậy để tránh dự đoán lặp. Tuy nhiên, chi phí tính toán bậc hai của phép tự chú ý trên lưới $P_2$ ($320 \times 320 = 102.400$ ô) vượt quá ngân sách phần cứng của mạng nano.

Ở hướng tích chập thuần túy, Sun và cộng sự (*What Makes for End-to-End Object Detection?* [10], đề xuất khuôn khổ OneNet) chứng minh rằng một bộ phát hiện gán nhãn đơn mục tiêu bắt buộc phải kết hợp cả điểm phân loại lẫn sai lệch hộp trong hàm chi phí ghép cặp để tạo khoảng cách điểm rõ ràng giữa ô được chọn và các ô xung quanh. Kế thừa nguyên lý đó, YOLOv10 [8] và YOLO26 [9] sử dụng cặp nhánh song song khi huấn luyện: nhánh giám sát đa mẫu phụ trợ $\mathcal{H}_{1:M}$ (*one-to-many*) cung cấp tín hiệu giám sát dày đặc nuôi Backbone và Neck, còn nhánh gán nhãn đơn mục tiêu $\mathcal{H}_{1:1}$ (*one-to-one*) xuất trực tiếp một dự đoán cho mỗi đối tượng. Khi hợp nhất mô hình để suy luận, nhánh phụ $\mathcal{H}_{1:M}$ bị tháo bỏ hoàn toàn, giúp thời gian xử lý ổn định và không phụ thuộc vào số lượng vật thể trong khung hình.

Khi kiểm tra thực tế nhánh $\mathcal{H}_{1:1}$ của YOLO26n-P2 trên TinyPerson, chúng tôi xác định hai hạn chế cấu trúc xuất phát từ việc thiếu tương tác cục bộ sau giải mã:
1. Không có bước đồng thuận giảm phương sai cho hộp đơn ô: Các phương pháp hậu xử lý như Soft-NMS [31] và Weighted Boxes Fusion (WBF) [34] tận dụng các hộp kề nhau cùng chỉ vào một đối tượng để trung bình hóa tọa độ theo trọng số độ tin cậy. Trong nhánh $\mathcal{H}_{1:1}$ cô lập, ô được chọn tự hồi quy 4 cạnh độc lập; với vật 4–8 pixel ở bước lưới 4, chỉ cần sai số nửa ô lưới là chỉ số IoU đã rớt xuống dưới $0{,}75$.
2. Nhánh phân loại thiếu thông tin hình học lân cận: Khác với các phương pháp học triệt tiêu hộp như Learning NMS [35] và tầng giải mã của RT-DETR [11], toán tử phân loại $\Phi_{\mathrm{cls}}^{(s)}$ của $\mathcal{H}_{1:1}$ chỉ quan sát đặc trưng thị giác cục bộ mà không biết mức độ chồng lấn giữa hộp của ô hiện tại với hộp các ô kế bên. Do đó, tại ranh giới hai người đứng sát nhau hoặc một mục tiêu mờ nằm giữa hai ô lưới, nhánh phân loại khó phân biệt đâu là khung bao trùng lặp cần giảm điểm và đâu là cá thể lân cận cần giữ.

#### 1.2.4. Khoảng cách phân phối Gauss Wasserstein cho vật thể cực nhỏ

Tính nhạy cảm quá mức của chỉ số $\mathrm{IoU}$ với sai số dịch chuyển vài pixel trên vật thể nhỏ đã được chứng minh trên bộ dữ liệu AI-TOD [6]. Với hộp chuẩn $4 \times 4$ pixel, lệch tâm 1 pixel làm $\mathrm{IoU}$ giảm từ $1{,}00$ xuống $0{,}39$, còn lệch 4 pixel khiến $\mathrm{IoU} = 0$ làm triệt tiêu gradient hồi quy. Để làm trơn hàm đo, Yang và cộng sự đề xuất Gaussian Wasserstein Distance (GWD) [15] cho hộp xoay, sau đó Wang và Xu phát triển Normalized Wasserstein Distance (NWD) [13, 14] và cơ chế gán nhãn trường tiếp nhận Gauss RFLA [16] cho vật nhỏ trên không ảnh. NWD biểu diễn mỗi hộp $(c_x, c_y, w, h)$ bằng phân phối chuẩn hai chiều $\mathcal{G} = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ rồi quy đổi khoảng cách $2$-Wasserstein [19] qua hàm mũ:
$$\mathrm{NWD}(\mathcal{G}_1, \mathcal{G}_2) = \exp\!\left(-\frac{\sqrt{W_2^2(\mathcal{G}_1, \mathcal{G}_2)}}{C}\right)$$ (1.1)
Trong đó: $\boldsymbol{\mu} = (c_x, c_y)^\top$ là tọa độ tâm hộp (tính theo pixel), $\boldsymbol{\Sigma} = \mathrm{diag}(w^2/4,\, h^2/4)$ là ma trận hiệp phương sai đường chéo, và $C > 0$ là hằng số quy mô cố định tính bằng pixel (giá trị thường chọn $C = 12{,}8$ theo kích thước trung bình bộ dữ liệu [13]).

Đưa trực tiếp phương trình (1.1) vào một bộ phát hiện đa tỉ lệ không dùng NMS gây ra hai bất cập:
- Không gần bất biến theo tỉ lệ vì hằng số $C$ cố định: Xét hai cặp hộp có cùng sai lệch tâm 25% chiều rộng theo mỗi trục ($\Delta c_x = \Delta c_y = 0{,}25 s$ với $w = h = s$), khoảng cách gốc là $\sqrt{W_2^2} = s\sqrt{0{,}125}$. Ở vật 4 pixel, $\mathrm{NWD} = 0{,}895$; nhưng ở người lớn 64 pixel gần camera, cùng mức lệch tương đối 25% cho $\mathrm{NWD} = 0{,}171$. Sự trôi tỉ lệ này làm lệch trọng số ghép cặp giữa các tầng $P_2\text{–}P_5$.
- Chỉ tác động lên hàm mất mát khi huấn luyện: GWD [15] và NWD [13, 14] chỉ dùng khoảng cách Wasserstein cho từng cặp dự đoán–GT lúc huấn luyện, không áp dụng định lý trọng tâm Wasserstein (*Wasserstein barycenter*) [18] để hợp nhất phân phối lân cận trong quá trình suy luận.

#### 1.2.5. Gán nhãn căn chỉnh tác vụ và xung đột hiệu chuẩn mục tiêu mềm

Trong YOLO26 [9] và TOOD [17], bộ gán nhãn động căn chỉnh tác vụ (TAL) chấm điểm từng vị trí neo qua chỉ số căn chỉnh $t = s^\alpha u^\beta$ (với các tham số $\alpha, \beta$ do cấu hình quy định), giữ lại top-$k$ ứng viên nằm trong Ground Truth (GT). Trong các thực nghiệm của đồ án, cả Base, NWD và GauCon đều dùng $\alpha = 0{,}5$ và $\beta = 4{,}0$, rồi chuẩn hóa để gán mục tiêu phân loại mềm (*soft target*) $\mathbf{t}_{\mathrm{cls}}$ cho hàm mất mát BCE:
$$t_{\mathrm{cls}}(u) = \hat{t}(u) = \frac{t(u)}{\max_{v \in \Omega_{\mathrm{pos}}} t(v) + \epsilon} \cdot \max_{v \in \Omega_{\mathrm{pos}}} u(v)$$ (1.2)
Trong đó: $\Omega_{\mathrm{pos}}$ là tập mẫu dương được gán cho cùng một đối tượng GT, $s(u)$ là xác suất dự đoán và $u(u)$ là độ chồng lấn hình học tại ô $u$.

Generalized Focal Loss [28] và VarifocalNet [36] chứng minh rằng độ tin cậy chỉ xếp hạng chính xác khi mục tiêu phân loại phản ánh đúng $\mathrm{IoU}$ thực tế của hộp dự đoán. Với $u = \mathrm{CIoU}$ trong (1.2), thừa số $\max u(v)$ giới hạn mức trần của mục tiêu mềm đúng bằng $\mathrm{IoU}$ tốt nhất hiện có. Trái lại, nếu thay $u$ bằng độ tương đồng Gauss mở rộng (như NWD), một dự đoán cho vật 4 pixel bị lệch sang bên cạnh ($\mathrm{IoU} = 0{,}02$) vẫn nhận giá trị $u \approx 0{,}55\text{–}0{,}75$. Hệ quả là mạng bị huấn luyện để gán độ tin cậy $0{,}6\text{–}0{,}7$ cho các ô lệch tâm và vùng nền biển gần đó, làm tăng lỗi báo nhầm nền ($E_{\mathrm{bkg}}$) và lỗi định vị ($E_{\mathrm{loc}}$) theo phân rã TIDE [37].

#### 1.2.6. Suy luận theo lát cắt chồng lấn (*Tiled Slicing*)

SAHI [38] và quy trình đánh giá gốc của TinyPerson [1] chia ảnh lớn thành các lát cắt nhỏ có phần chồng lấn, sau đó gộp dự đoán bằng NMS toàn cục. Cách làm này giữ độ phân giải của đối tượng nhỏ nhưng làm tăng số lượt suy luận và có thể tạo hộp bị cắt tại biên lát. Vì vậy, báo cáo đánh giá cả giao thức lát cắt ở độ phân giải gốc và giao thức suy luận toàn khung $1280 \times 1280$ trong một lượt.

### 1.3. Khoảng trống nghiên cứu

Bảng 1.1 đối chiếu các nhóm giải pháp hiện có theo năm tiêu chí kỹ thuật cốt lõi của bài toán UAV hàng hải, và Bảng 1.2 tổng hợp các mốc kết quả công bố trước đây trên TinyPerson để đối sánh quy mô mô hình.

Bảng 1.1: Đối chiếu đặc trưng kỹ thuật giữa các hướng tiếp cận hiện có và giải pháp đề xuất

| Hướng tiếp cận / Mô hình | Quy mô tham số | Suy luận không cần NMS | Đồng thuận hộp cục bộ trong Head | Gần bất biến theo tỉ lệ, không cần hằng số $C$ | Hiệu chuẩn mục tiêu mềm theo IoU thực |
|---|---|---|---|---|---|
| Hai giai đoạn + Scale Match / SSPNet [1, 4] | 32–60+ M | Không | Không (RoI 2 tầng) | Không | Không |
| YOLO cỡ nano có $P_2$ + NMS (BPD-YOLO [5]) | 1,5–2,9 M | Không | Không (NMS ngoài) | Không | Có (TAL gốc) |
| Hậu xử lý gộp hộp (Soft-NMS [31], WBF [34]) | +0 | Không (chạy sau mạng) | Có (ngoài đồ thị) | Không | Không can thiệp |
| Bộ phát hiện đầu-cuối YOLO26n-P2 gốc [9] | 2,40 M | Có | Không (đơn ô cô lập) | Không | Có (thiếu mẫu dương <6 px) |
| YOLO26n-P2 ghép trực tiếp NWD [13, 14] | 2,40 M | Có | Không | Không (cố định $C$) | Không (khuếch đại mục tiêu mềm) |
| YOLO26n-P2 + GauCon & DW-TAL (Đề xuất) | 2,40 M (+714) | Có | Có (Trọng tâm $W_2$ + FG-Gate) | Có (Chuẩn hóa theo phương sai cục bộ) | Có (Tách rời chọn mẫu & hiệu chuẩn mục tiêu mềm) |

Bảng 1.2: Tổng hợp kết quả tham khảo đã công bố trên tập kiểm tra TinyPerson chia theo nhóm giao thức

| Mô hình | Khởi tạo trọng số | Tham số | Hậu xử lý NMS | $AP_{50}^{tiny}$ (%) | $AP_{50}^{tiny1}$ (%) |
|---|---|---|---|---|---|
| (a) Chuẩn gốc WACV 2020 [1] (2 lớp, Tiled Slicing) | | | | | |
| FCOS [25] | ImageNet | ~32,0 M | Có | 16,90 | 3,39 |
| RetinaNet [23] | ImageNet | ~34,0 M | Có | 30,82 | 11,47 |
| Faster R-CNN [21] | ImageNet | ~41,0 M | Có | 43,55 | 29,21 |
| (b) Họ hai giai đoạn cải tiến [1, 2, 4] (2 lớp, Tiled Slicing) | | | | | |
| Faster R-CNN-FPN [21, 22] | ImageNet | ~41,0 M | Có | 47,35 | 30,25 |
| Faster R-CNN-FPN + MSM [1] | ImageNet | ~41,0 M | Có | 50,89 | 33,79 |
| Faster R-CNN-FPN + SM+ [2] | ImageNet | ~41,0 M | Có | 52,61 | 34,20 |
| Cascade R-CNN [26] | ImageNet | >60,0 M | Có | 57,19 | 45,21 |
| Faster R-CNN + SSPNet [4] | ImageNet | >41,0 M | Có | 59,13 | 47,56 |
| (c) Nhóm một giai đoạn cỡ nano [5] (1 lớp *person*, 1024 px, 300 epoch) | | | | | |
| YOLOv8n+P2 [5] | Từ đầu (*Scratch*) | 2,93 M | Có | 38,00* | – |
| BPD-YOLO [5] | Từ đầu (*Scratch*) | 1,50 M | Có | 39,10* | – |

*Ghi chú: Bảng 1.2 chỉ mang tính tham chiếu bối cảnh từ các công bố trước vì mỗi nhóm khác nhau về số lớp (2 lớp so với 1 lớp), độ phân giải ảnh, bộ dữ liệu khởi tạo (ImageNet so với COCO hoặc huấn luyện từ đầu 300 epoch) và bộ tiêu chí chấm điểm. Dấu (*) ở nhóm (c) thể hiện bài báo gốc [5] báo cáo $mAP_{50}$ tổng hợp trên một lớp person ở 300 epoch từ đầu.*

Từ các công trình được khảo sát, đồ án tập trung đánh giá hai ý tưởng trong cùng một đầu phát hiện gọn nhẹ. Thứ nhất, thông tin hình học từ các ô lân cận được đưa vào nhánh dự đoán đơn mục tiêu tại $P_2$ và $P_3$. Thứ hai, độ tương đồng Wasserstein được dùng để hỗ trợ chọn ứng viên dương, trong khi mục tiêu phân loại cuối cùng được hiệu chuẩn theo IoU thực tế. Đây là phạm vi kết hợp và đánh giá của đồ án, không phải tuyên bố rằng từng thành phần riêng lẻ chưa từng được nghiên cứu.

### 1.4. Mục tiêu, đối tượng và phạm vi nghiên cứu

Đồ án xác định bốn mục tiêu kỹ thuật kiểm chứng xuyên suốt qua Chương 2, Chương 3 và Kết luận:
1. Thiết lập mô hình cơ sở (*baseline*) và quy trình đánh giá: Tách 746 ảnh huấn luyện theo tiền tố chuỗi video thành 662 ảnh *train* và 84 ảnh *val*; sử dụng tập kiểm tra chính thức gồm 786 ảnh để đánh giá cuối cùng. Hai giao thức đánh giá gồm lát cắt $640 \times 512$ và toàn khung $1280 \times 1280$.
2. Thiết kế đầu phát hiện GauCon chi phí cực thấp: Bổ sung cơ chế đồng thuận hộp trọng tâm Wasserstein và cổng điều biến FG-Gate không dùng chuẩn hóa theo batch chọn lọc trên $P_2, P_3$, giới hạn mức tăng tham số dưới 0,05% (< 1.000 tham số) và giữ chi phí tham số và tính toán bổ sung ở mức thấp.
3. Thiết kế cơ chế gán nhãn tách rời DW-TAL và chưng cất phân phối hộp: Kết hợp cổng kích thước logistic ($\tau_{\mathrm{px}} = 20$ px) vào khâu chọn mẫu dương, tái hiệu chuẩn mục tiêu mềm phân loại theo $\mathrm{IoU}$ thực để hạn chế báo nhầm nền, và chưng cất phân phối Gauss từ nhánh $\mathcal{H}_{1:M}$ không tốn chi phí khi suy luận.
4. Kiểm định thực nghiệm đa chiều: So sánh trực tiếp YOLO26n-P2 Base, biến thể ghép NWD [13, 14], hậu xử lý đối chứng Base + Soft-NMS và mô hình đề xuất dưới cùng điều kiện huấn luyện (80 epoch, khởi tạo từ trọng số COCO, batch size 16, seed 42 trên GPU NVIDIA L4), phân rã chi tiết qua TIDE [37], $d_{\mathrm{NN}}$, các dải con *reasonable*, 1.000 lần Paired Image Bootstrap, đường cong IoU và chi phí phần cứng.

Phạm vi nghiên cứu tập trung vào bài toán phát hiện người trên ảnh hàng hải theo thiết lập một lớp *person* trên TinyPerson với một lần huấn luyện (seed = 42); các hạn chế về đơn seed và kiểm chứng đa lớp được phân tích tại Mục 3.8.

### 1.5. Bố cục báo cáo

Chương 2 trình bày cơ sở lý thuyết 2-Wasserstein, kiến trúc tổng thể YOLO26n-P2 tích hợp GauCon, chi tiết bốn khối toán học của GauCon, thuật toán gán nhãn tách rời DW-TAL, luồng biến đổi tensor và phân tích chi tiết 714 tham số. Chương 3 báo cáo giao thức đánh giá, kết quả đối chứng ở hai chế độ lát cắt và toàn khung so với Base, Base + Soft-NMS và NWD, phân tích cân bằng ưu/nhược điểm qua TIDE, phân rã dải con *reasonable*, kiểm định Bootstrap 95% CI, diễn biến tham số nội tại, ví dụ định tính, hiệu năng phần cứng L4 và các hạn chế phương pháp luận. Phần Kết luận đối chiếu từng mục tiêu đặt ra và nêu hướng mở rộng.

---

## 2. GIẢI PHÁP ĐỀ XUẤT

### 2.1. Cơ sở lý thuyết và động lực toán học

#### 2.1.1. Tỷ lệ stride và giới hạn thông tin đơn ô trên vật thể cực nhỏ

Trong mô hình *anchor-free*, bản đồ đặc trưng mức $s$ có bước nhảy lưới (stride) $r_s \in \{4, 8, 16, 32\}$ chia ảnh đầu vào $H \times W$ thành lưới $(H/r_s) \times (W/r_s)$. Một người có cạnh trung bình $s_{\mathrm{in}}$ pixel tại đầu vào phủ trung bình $(s_{\mathrm{in}}/r_s) \times (s_{\mathrm{in}}/r_s)$ ô trên mức $s$. Bảng 2.1 thống kê độ phủ ô lưới trên từng tầng ở chế độ lát cắt 1:1 và khi co toàn khung hình về $1280 \times 1280$.

Bảng 2.1: Độ phủ ô lưới của đối tượng theo stride $r_s$ ở chế độ lát cắt gốc và chế độ thu nhỏ toàn khung $1280 \times 1280$

| Tầng đặc trưng | Stride $r_s$ | Kích thước lưới (@1280) | Số kênh đầu vào Head | Số ô phủ bởi vật 8 px (Lát cắt 1:1) | Số ô phủ bởi vật 8 px gốc (Toàn khung co ~3×, còn ~2,7 px) | Vai trò trong kiến trúc đề xuất |
|---|---|---|---|---|---|---|
| $P_2$ | 4 | $320 \times 320$ | 32 | $2{,}00$ ô | $0{,}67$ ô | Tầng chủ lực cho *tiny1*–*tiny2*; kích hoạt GauCon |
| $P_3$ | 8 | $160 \times 160$ | 64 | $1{,}00$ ô | $0{,}33$ ô | Tầng chủ lực cho *tiny2*–*small*; kích hoạt GauCon |
| $P_4$ | 16 | $80 \times 80$ | 128 | $0{,}50$ ô | $0{,}17$ ô | Phục vụ vật >28 px + làm ô cha cho $P_3$; giữ nguyên $\mathcal{H}_{1:1}$ gốc |
| $P_5$ | 32 | $40 \times 40$ | 256 | $0{,}25$ ô | $0{,}08$ ô | Phục vụ vật lớn và ngữ cảnh rộng; giữ nguyên $\mathcal{H}_{1:1}$ gốc |

Bảng 2.1 giải thích hai quyết định thiết kế nền tảng:
1. Ở chế độ lát cắt 1:1, một người rộng 6–16 pixel phủ 1,5–4 ô trên lưới $P_2$ và 1–2 ô trên lưới $P_3$. Các ô trong cửa sổ $3 \times 3$ quanh tâm vật cùng tiếp nhận phần thân đối tượng và đưa ra các hộp dự đoán độc lập có sai số biên ngẫu nhiên. Hợp nhất có trọng số các ước lượng lân cận này ngay trong đầu mạng giúp giảm phương sai biên hộp so với hồi quy đơn ô cô lập.
2. Trái lại, trên hai tầng $P_4$ ($r_4 = 16$) và $P_5$ ($r_5 = 32$), khoảng cách hai tâm ô kế tiếp đạt 16–32 pixel đầu vào (tương ứng 48–96 pixel trên ảnh gốc 4K). Việc trộn hộp giữa các ô cách xa như vậy làm nhòe biên của các đối tượng lớn vốn đã được định vị chính xác. Vì thế, GauCon chỉ tác động trên hai mức $P_2$ và $P_3$.

#### 2.1.2. Không gian metric 2-Wasserstein và định lý trọng tâm phân phối Gauss

Với hai phân phối xác suất có mô-men bậc hai hữu hạn $\mathcal{P}, \mathcal{Q} \in \mathcal{P}_2(\mathbb{R}^2)$, bình phương khoảng cách $2$-Wasserstein [19] bằng chi phí vận chuyển cực tiểu giữa chúng:
$$W_2^2(\mathcal{P}, \mathcal{Q}) = \inf_{\pi \in \Pi(\mathcal{P}, \mathcal{Q})} \int_{\mathbb{R}^2 \times \mathbb{R}^2} \|\mathbf{z}_1 - \mathbf{z}_2\|_2^2\,\mathrm{d}\pi(\mathbf{z}_1, \mathbf{z}_2)$$ (2.1)
Trong đó: $\Pi(\mathcal{P}, \mathcal{Q})$ là tập hợp các phân phối đồng thời có phân phối biên là $\mathcal{P}$ và $\mathcal{Q}$.

Khi $\mathcal{P} = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ và $\mathcal{Q} = \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$ là hai phân phối chuẩn có hiệp phương sai đường chéo giao hoán $\boldsymbol{\Sigma}_i = \mathrm{diag}(\sigma_{ix}^2, \sigma_{iy}^2)$ (ứng với hộp bao song song trục tọa độ ảnh), tích phân (2.1) rút gọn thành dạng giải tích đóng [15, 19]:
$$W_2^2(\mathcal{P}, \mathcal{Q}) = \sum_{a \in \{x, y\}} \Big((\mu_{1a} - \mu_{2a})^2 + (\sigma_{1a} - \sigma_{2a})^2\Big)$$ (2.2)
Trong đó: $\boldsymbol{\mu}_i = (\mu_{ix}, \mu_{iy})^\top$ là tâm hộp và $\boldsymbol{\sigma}_i = (\sigma_{ix}, \sigma_{iy})^\top$ là bán kính hai chiều của hộp bao.

Để hợp nhất tập $M$ phân phối lân cận $\{\tilde{\mathcal{G}}_k = \mathcal{N}(\tilde{\boldsymbol{\mu}}_k, \mathrm{diag}(\tilde{\boldsymbol{\sigma}}_k^2))\}_{k=0}^{M-1}$ theo bộ trọng số lồi không âm $\{w_k\}_{k=0}^{M-1}$ ($\sum_{k=0}^{M-1} w_k = 1$), chúng tôi áp dụng định lý trọng tâm trong không gian Wasserstein của Agueh và Carlier [18]. Nghiệm $\mathcal{G}^\star$ cực tiểu hóa tổng khoảng cách vận chuyển có trọng số:
$$\mathcal{G}^\star = \arg\min_{\mathcal{G} \in \mathcal{P}_2(\mathbb{R}^2)} \sum_{k=0}^{M-1} w_k\,W_2^2(\mathcal{G}, \tilde{\mathcal{G}}_k)$$ (2.3)
tồn tại duy nhất trong chính họ phân phối chuẩn đường chéo $\mathcal{G}^\star = \mathcal{N}(\boldsymbol{\mu}^\star, \mathrm{diag}({\boldsymbol{\sigma}^\star}^2))$ với biểu thức tường minh:
$$\boldsymbol{\mu}^\star = \sum_{k=0}^{M-1} w_k\,\tilde{\boldsymbol{\mu}}_k, \qquad \boldsymbol{\sigma}^\star = \sum_{k=0}^{M-1} w_k\,\tilde{\boldsymbol{\sigma}}_k$$ (2.4)
Trong đó: $\boldsymbol{\mu}^\star$ và $\boldsymbol{\sigma}^\star$ là tâm và bán kính của hộp đồng thuận tối ưu. Khác với phép trộn tuyến tính hai mật độ xác suất (làm giãn phương sai $\sigma$ khi hai tâm lệch nhau, minh họa ở Hình 2.3b), phép nội suy trọng tâm trong không gian Wasserstein giữ nguyên độ rộng thực $\boldsymbol{\sigma}^\star$ mà không làm phình kích thước hộp.

### 2.2. Kiến trúc tổng thể mạng YOLO26n-P2 tích hợp GauCon và DW-TAL

![](docs/report/figures/fig_gaucon_pipeline_vi.png)

Hình 2.1: Kiến trúc tổng thể mạng phát hiện YOLO26n-P2 tích hợp GauCon chọn lọc trên $P_2, P_3$ và các nhánh chỉ phục vụ huấn luyện

Hình 2.1 trình bày cấu trúc đầy đủ của hệ thống theo bốn khối nối tiếp từ trái sang phải: mạng Backbone CSPDarknet trích xuất đặc trưng đa tỉ lệ $C_2\text{–}C_5$, khối Neck FPN–PANet kết hợp đặc trưng hai chiều từ trên xuống (*Top-Down*) và từ dưới lên (*Bottom-Up*) để tạo bốn bản đồ đặc trưng $\{P_2, P_3, P_4, P_5\}$ với số kênh $\{32, 64, 128, 256\}$, đầu phát hiện suy luận không NMS $\mathcal{H}_{1:1}$, và các nhánh phụ chỉ dùng khi huấn luyện (khung nét đứt màu tím phía dưới).

Duy trì thiết kế hai nhánh của YOLO26 [9], mô hình tách biệt vai trò từng nhánh như sau:
- Nhánh gán nhãn đơn mục tiêu $\mathcal{H}_{1:1}$: Tại mỗi mức $s \in \{P_2, P_3, P_4, P_5\}$, cặp nhánh tách rời gồm nhánh hộp $\Phi_{\mathrm{box}}^{(s)}$ ($\mathrm{Conv}_{3\times 3} \to \mathrm{Conv}_{3\times 3} \to \mathrm{PWConv}_{1\times 1}$) và nhánh phân loại $\Phi_{\mathrm{cls}}^{(s)}$ ($[\mathrm{DWConv}_{3\times 3} \to \mathrm{PWConv}_{1\times 1}]_{\times 2} \to \mathrm{PWConv}_{1\times 1}$) xuất bản đồ khoảng cách biên thô $\mathbf{D}_0^{(s)} \in \mathbb{R}^{B \times 4 \times H_s \times W_s}$ (biểu diễn biên trực tiếp không phân rã phân phối) và bản đồ logit phân loại thô $\mathbf{S}_{\mathrm{raw}}^{(s)} \in \mathbb{R}^{B \times C \times H_s \times W_s}$. Hai mức độ phân giải cao $P_2, P_3$ đi qua GauCon (có sử dụng ô cha từ mức $s+1$ với tỉ lệ nội suy $2\times$), trong khi $P_4, P_5$ giữ nguyên đầu ra gốc (ánh xạ đồng nhất) trước khi tổng hợp toàn bộ $136.000$ vị trí trên cả 4 mức ($320^2 + 160^2 + 80^2 + 40^2 = 102.400 + 25.600 + 6.400 + 1.600$) để chọn Top-$K$ dự đoán không cần NMS.
- Nhánh giám sát đa mẫu phụ trợ $\mathcal{H}_{1:M}$, DW-TAL và chưng cất phân phối Gauss (chỉ dùng khi huấn luyện): Nhánh $\mathcal{H}_{1:M}$ cung cấp tín hiệu giám sát dày đặc (tối đa $k = 10$ ứng viên dương cho mỗi GT) và truyền phân phối hộp Gauss sang các mức $(P_2, P_3)$ của nhánh $\mathcal{H}_{1:1}$ qua mất mát $\mathcal{L}_{\mathrm{distill}}$. Khi hợp nhất mô hình để xuất bản suy luận, toàn bộ khối huấn luyện khung tím phía dưới bị gỡ bỏ khỏi đồ thị tính toán.

Về khởi tạo trọng số (Base, + NWD, + GauCon cùng nạp từ trọng số tiền huấn luyện MS COCO của YOLO26n): vì mô hình COCO gốc chỉ có 3 mức $P_3\text{–}P_5$ và cấu hình số kênh khác, các tầng Backbone khớp kích thước kế thừa trọng số tiền huấn luyện từ COCO, trong khi nhánh bổ sung tầng $P_2$ và các khối Neck/Head thay đổi số kênh được khởi tạo ngẫu nhiên đồng nhất giữa ba cấu hình.

![](docs/report/figures/fig_gaucon_head_vi.png)

Hình 2.2: Vi cấu trúc xử lý bên trong đầu phát hiện GauCon tại một ô lưới hoạt động $u = (y, x)$

Hình 2.2 minh họa các bước xử lý của GauCon tại mỗi tầng $s \in \{P_2, P_3\}$: quy đổi tọa độ các ô lân cận, tính khoảng cách chuẩn hóa theo phương sai cục bộ $D_k$, tinh chỉnh hộp bằng trọng tâm Wasserstein và hiệu chỉnh logit bằng cổng quan hệ FG-Gate.

### 2.3. Chi tiết toán học các thành phần của đầu phát hiện GauCon

#### 2.3.1. Biểu diễn hộp bằng phân phối Gauss và quy đổi hệ tọa độ lân cận

Xét ô lưới $u = (y, x)$ ở mức đặc trưng $s \in \{P_2, P_3\}$ có tâm neo tại $(x + 0{,}5,\, y + 0{,}5)$ trong hệ đơn vị bước lưới $r_s$. Gọi $\mathbf{d}_0(u) = (l, t, r, b) \in \mathbb{R}_{\ge 0}^4$ là bốn khoảng cách tới biên do toán tử $\Phi_{\mathrm{box}}^{(s)}$ dự đoán trực tiếp mỗi cạnh (nếu áp dụng lên kiến trúc có biểu diễn phân phối rời rạc DFL trên $R > 1$ khoảng chia cạnh, $\mathbf{d}_0$ được lấy ngay sau phép chiếu kỳ vọng softmax). Hộp dự đoán tại ô $u$ được ánh xạ thành phân phối Gauss 2D có hiệp phương sai đường chéo $\mathcal{G}_0(u) = \mathcal{N}(\boldsymbol{\mu}_0(u), \mathrm{diag}(\boldsymbol{\sigma}_0(u)^2))$ lấy tâm ô $u$ làm gốc:
$$\boldsymbol{\mu}_0(u) = \frac{1}{2}\begin{bmatrix} r - l \\ b - t \end{bmatrix}, \qquad \boldsymbol{\sigma}_0(u) = \max\!\left(\frac{1}{2}\begin{bmatrix} l + r \\ t + b \end{bmatrix},\, \epsilon_\sigma\right)$$ (2.5)
Trong đó: $\boldsymbol{\mu}_0(u) = (\mu_{0x}, \mu_{0y})^\top$ là độ lệch tâm hộp so với tâm ô $u$ (đơn vị: bước lưới $r_s$), $\boldsymbol{\sigma}_0(u) = (\sigma_{0x}, \sigma_{0y})^\top$ là bán kính hộp hai chiều (với $l,t,r,b$ lấy tại ô $u$), và $\epsilon_\sigma = 0{,}05$ bước lưới là ngưỡng chặn dưới chống suy biến. Phép ánh xạ ngược $\mathcal{G}^{-1}$ khôi phục chính xác bốn khoảng cách biên tinh chỉnh:
$$\mathbf{d}_{\mathrm{ref}} = \big(\sigma_{\mathrm{ref},x} - \mu_{\mathrm{ref},x},\; \sigma_{\mathrm{ref},y} - \mu_{\mathrm{ref},y},\; \sigma_{\mathrm{ref},x} + \mu_{\mathrm{ref},x},\; \sigma_{\mathrm{ref},y} + \mu_{\mathrm{ref},y}\big)$$ (2.6)
Trong đó: $\mathbf{d}_{\mathrm{ref}}$ là bộ khoảng cách biên tại ô $u$ sau khi cập nhật bởi trọng tâm Wasserstein.

Quanh ô $u$, GauCon thu thập 9 ứng viên lân cận: 8 ô xung quanh trong cửa sổ $3 \times 3$ tương ứng với $\delta_k = (\Delta x_k, \Delta y_k) \in \{-1, 0, +1\}^2,\; \delta_k \neq (0, 0)$ ($k = 1, \dots, 8$) và 1 ô cha đồng vị trí $u_p = (\lfloor y/2 \rfloor, \lfloor x/2 \rfloor)$ nội suy láng giềng gần nhất từ mức $s + 1$ ($k = 9$, tỉ số bước lưới $\rho = r_{s+1}/r_s = 2$: $P_2$ lấy cha từ $P_3$, $P_3$ lấy cha từ $P_4$). GauCon trích xuất tám dự đoán tại các ô lân cận trong cửa sổ $3 \times 3$ và loại các vị trí nằm ngoài biên bản đồ đặc trưng. Cơ chế này bổ sung thông tin hình học lân cận mà không thêm một lớp tích chập học được.

Vì tâm hộp $\boldsymbol{\mu}_0(u + \delta_k)$ của hàng xóm ban đầu quy chiếu về tâm ô $u + \delta_k$, ta phải cộng bù độ dời tâm ô để đưa toàn bộ 9 ứng viên về chung hệ tọa độ cục bộ của ô $u$:
$$\tilde{\boldsymbol{\mu}}_k(u) = \boldsymbol{\mu}_0(u + \delta_k) + \delta_k, \qquad \tilde{\boldsymbol{\sigma}}_k(u) = \boldsymbol{\sigma}_0(u + \delta_k), \qquad k \in \{1, \dots, 8\}$$ (2.7)
$$\tilde{\boldsymbol{\mu}}_9(u) = \rho\,\boldsymbol{\mu}_0^{(s+1)}(u_p) + \boldsymbol{\Delta}_p(u), \qquad \tilde{\boldsymbol{\sigma}}_9(u) = \max\!\big(\rho\,\boldsymbol{\sigma}_0^{(s+1)}(u_p),\, \epsilon_\sigma\big)$$ (2.8)
Trong đó: $\boldsymbol{\Delta}_p(u) = \rho(u_p + 0{,}5) - (u + 0{,}5) \in \{-0{,}5, +0{,}5\}^2$ là sai lệch tọa độ từ tâm ô cha trên mức $s+1$ về tâm ô con $u$ đo theo đơn vị bước lưới $r_s$. Nhờ (2.7)–(2.8), nếu hai ô dự đoán cùng một hộp bao vật lý trên ảnh gốc, các vector $(\tilde{\boldsymbol{\mu}}_k, \tilde{\boldsymbol{\sigma}}_k)$ và $(\boldsymbol{\mu}_0, \boldsymbol{\sigma}_0)$ bằng nhau chính xác bất kể bước lưới.

![](docs/report/figures/fig_gaucon_geometry.png)

Hình 2.3: Minh họa hình học cơ chế hoạt động của GauCon và DW-TAL: *(a)* Quy đổi tọa độ lân cận $3\times 3$ và ô cha $P_3$ trên lưới $P_2$ phân tách người đứng sát nhau ($D_k \gg 1 \implies w_k \approx 0$); *(b)* Trọng tâm Wasserstein giữ nguyên bề rộng hộp $\sigma^\star$ so với phép trộn mật độ gây giãn hộp; *(c)* Tác dụng giới hạn mục tiêu mềm của DW-TAL theo $\mathrm{IoU}$ thực tại ô lệch tâm trên vật 4 px ($\mathrm{IoU} \approx 0$)

#### 2.3.2. Khoảng cách Wasserstein chuẩn hóa theo phương sai cục bộ

Từ phương trình (2.2), để loại bỏ hằng số $C$ phụ thuộc bộ dữ liệu của NWD [13], GauCon chuẩn hóa sai lệch trên từng trục $a \in \{x, y\}$ bằng tổng phương sai cục bộ của chính cặp phân phối đang xét:
$$D_k(u) = \sum_{a \in \{x, y\}} \frac{\big(\tilde{\mu}_{ka}(u) - \mu_{0a}(u)\big)^2 + \big(\tilde{\sigma}_{ka}(u) - \sigma_{0a}(u)\big)^2}{\sigma_{0a}(u)^2 + \tilde{\sigma}_{ka}(u)^2 + \epsilon}$$ (2.9)
Trong đó: $D_k(u) \ge 0$ là khoảng cách Wasserstein chuẩn hóa không đơn vị giữa ô $u$ và ứng viên $k$, $\epsilon = 10^{-6}$ tránh chia cho 0. Với các hướng trượt vượt biên bản đồ đặc trưng hoặc khác lớp dự đoán chiếm ưu thế ($\arg\max_c \mathbf{s}_{\mathrm{raw}}(u_k) \neq \arg\max_c \mathbf{s}_{\mathrm{raw}}(u)$ khi $C > 1$), khoảng cách bị gán $D_k(u) \leftarrow 10^4$ để loại bỏ hoàn toàn đóng góp của ứng viên đó.

![](docs/report/figures/fig_gaucon_theory_curves.png)

Hình 2.4: Đặc tính của cổng kích thước logistic và phép đo chuẩn hóa theo phương sai: *(a)* Cổng logistic $g(s_{\mathrm{gt}})$ ($\tau_{\mathrm{px}} = 20$ px) so với hàm suy giảm đuôi nặng Cauchy; *(b)* Độ tương đồng $S_W = (1 + D)^{-1}$ gần như giữ nguyên giá trị $0{,}80$ qua các tỉ lệ hộp $s$ khi độ lệch tâm bằng 25% chiều rộng theo mỗi trục, trong khi NWD ($C = 12{,}8$ px) suy giảm từ 0,895 xuống 0,171

Hình 2.4(b) minh họa tính ổn định theo tỉ lệ của khoảng cách (2.9). Khi $\epsilon$ nhỏ so với các thành phần phương sai, việc co giãn đồng thời hai hộp làm tử số và mẫu số thay đổi gần như cùng tỷ lệ. Do đó, $D_k$ gần như không đổi trong miền kích thước được khảo sát. Ở các thang cực nhỏ, hằng số $\epsilon$ có thể làm sai khác tính chất này.

#### 2.3.3. Tinh chỉnh hộp bằng Trọng tâm Wasserstein cục bộ

Áp dụng phương trình (2.4) lên tập 10 phân phối $\{\mathcal{G}_0(u), \tilde{\mathcal{G}}_1(u), \dots, \tilde{\mathcal{G}}_9(u)\}$, trọng số đóng góp của từng ứng viên được tính từ độ tin cậy cực đại $s_k(u) = \max_c \mathbf{s}_{\mathrm{raw}}(u_k)$ và khoảng cách hình học $D_k(u)$:
$$\ell_0 = \gamma_{\mathrm{box}}^{(s)}\log\sigma(s_0), \qquad \ell_k = \gamma_{\mathrm{box}}^{(s)}\log\sigma(s_k) - \tau_{\mathrm{box}}^{(s)} D_k \;\; (k=1..9)$$ (2.10)
$$w_k(u) = \frac{\exp\big(\ell_k(u)\big)}{\sum_{j=0}^{9} \exp\big(\ell_j(u)\big)}, \qquad k \in \{0, \dots, 9\}$$ (2.11)
Trong đó: $\sigma(\cdot)$ là hàm sigmoid, $s_k = s_k(u)$ và $D_k = D_k(u)$ viết gọn tại ô $u$, $\gamma_{\mathrm{box}}^{(s)} = \mathrm{softplus}(\hat{\gamma}_{\mathrm{box}}^{(s)})$ (khởi tạo 1,0) điều khiển độ nhạy theo độ tin cậy, và $\tau_{\mathrm{box}}^{(s)} = \mathrm{softplus}(\hat{\tau}_{\mathrm{box}}^{(s)}) + 0{,}1$ (khởi tạo 1,5) kiểm soát bán kính gom cụm trong không gian Wasserstein trên mức $s$. Các đại lượng từ các ô láng giềng $(\tilde{\boldsymbol{\mu}}_k, \tilde{\boldsymbol{\sigma}}_k, s_k)$ được ngắt lan truyền ngược gradient khi tính $w_k$, đồng thời phép tính hình học thực hiện ở độ chính xác đơn (FP32) nhằm bảo toàn ổn định số học khi huấn luyện hỗn hợp FP16.

Thay vì gán trực tiếp bằng trọng tâm đầy đủ (2.4) vốn tương ứng $\beta_{\mathrm{box}}^{(s)} = 1$, mô hình thực hiện một bước cập nhật nội suy có hệ số bước khả học $\beta_{\mathrm{box}}^{(s)} = \sigma(\hat{\beta}_{\mathrm{box}}^{(s)}) \in (0, 1)$ (khởi tạo 0,06) hướng về trọng tâm Wasserstein (khi $\beta_{\mathrm{box}} \to 0$ sẽ giữ nguyên dự đoán thô gốc):
$$\boldsymbol{\mu}_{\mathrm{ref}}(u) = \boldsymbol{\mu}_0(u) + \beta_{\mathrm{box}}^{(s)} \sum_{k=1}^{9} w_k(u)\,\big(\tilde{\boldsymbol{\mu}}_k(u) - \boldsymbol{\mu}_0(u)\big)$$ (2.12)
$$\boldsymbol{\sigma}_{\mathrm{ref}}(u) = \max\!\left(\boldsymbol{\sigma}_0(u) + \beta_{\mathrm{box}}^{(s)} \sum_{k=1}^{9} w_k(u)\,\big(\tilde{\boldsymbol{\sigma}}_k(u) - \boldsymbol{\sigma}_0(u)\big),\, \epsilon_\sigma\right)$$ (2.13)
Trong đó: phần tổng có trọng số $\sum_{k=1}^{9} w_k(\tilde{\boldsymbol{\mu}}_k - \boldsymbol{\mu}_0) = \boldsymbol{\mu}^\star - \boldsymbol{\mu}_0$ chính là dịch chuyển hướng về trọng tâm Wasserstein ở (2.4). Khi hai người đứng sát nhau ở hai ô kề nhau (Hình 2.3a), khoảng cách hình học lớn làm giảm trọng số của các hộp không tương ứng, hạn chế việc hai hộp bị kéo về cùng một vị trí. Khi nhiều ô cùng dự đoán một người, nội suy trọng tâm có thể giảm biến thiên của biên hộp.

#### 2.3.4. Cổng điều biến quan hệ lân cận không dùng BatchNorm (FG-Gate)

Để bổ sung nhận thức hình học lân cận cho toán tử phân loại $\Phi_{\mathrm{cls}}^{(s)}$ của $\mathcal{H}_{1:1}$, tại mỗi ô $u$ trên $s \in \{P_2, P_3\}$, ta định nghĩa hệ số gắn kết hình học $A_k(u) = \exp(-\tau_{\mathrm{cls}}^{(s)} D_k(u)) \in (0, 1]$ (với $\tau_{\mathrm{cls}}^{(s)} = \mathrm{softplus}(\hat{\tau}_{\mathrm{cls}}^{(s)}) + 0{,}1$, khởi tạo 1,5) và xác suất tiền cảnh $p_k(u) = \sigma(s_k(u))$ cho $k \in \{1, \dots, 9\}$. Từ đó xây dựng vector quan hệ 20 kênh $\mathbf{r}_u \in \mathbb{R}^{20}$ gồm bốn thành phần:
$$F_k(u) = A_k(u)\,p_k(u), \qquad M_k(u) = A_k(u)\,\max\!\big(0,\, s_k(u) - s_0(u)\big)$$ (2.14)
$$M_{\mathrm{dom}}(u) = \max_{1 \le k \le 9} M_k(u), \qquad M_{\mathrm{sup}}(u) = \frac{1}{|\mathcal{V}_u|} \sum_{k \in \mathcal{V}_u} A_k(u)\,p_k(u)$$ (2.15)
Trong đó: $k \in \{1, \dots, 9\}$ đánh số 9 láng giềng hợp lệ, $\mathbf{F}(u) = [F_1, \dots, F_9]^\top \in \mathbb{R}^9$ đo mức hỗ trợ của láng giềng có độ tin cậy cao; $\mathbf{M}(u) = [M_1, \dots, M_9]^\top \in \mathbb{R}^9$ đo mức ô $u$ bị một láng giềng trùng hộp có logit cao hơn lấn át (tạo tín hiệu ức chế trùng lặp khả vi); $M_{\mathrm{dom}}(u) \in \mathbb{R}$ là mức bị lấn át cực đại; $M_{\mathrm{sup}}(u) \in \mathbb{R}$ là đồng thuận trung bình trên tập láng giềng hợp lệ $\mathcal{V}_u$; và $\mathbf{r}_u = [\mathbf{F}(u)^\top, \mathbf{M}(u)^\top, M_{\mathrm{dom}}(u), M_{\mathrm{sup}}(u)]^\top \in \mathbb{R}^{20}$.

Trên các lưới $P_2$ và $P_3$, phần lớn ô không chứa đối tượng và có xác suất tiền cảnh thấp. Nếu đặt chuẩn hóa theo batch (BN) [32] trong mạng ánh xạ $\mathcal{M}_s: \mathbb{R}^{20} \to \mathbb{R}^C$, thống kê trung bình và phương sai trượt bị kéo lệch về phân bố nền và tạo độ lệch cộng cố định lên hàng loạt ô sóng. Do đó, $\mathcal{M}_s$ được thiết kế bằng hai lớp tích chập điểm $1 \times 1$ không dùng $\mathrm{BN}$ với chiều ẩn $d_{\mathrm{rel}} = 16$: $\mathcal{M}_s(\mathbf{r}_u) = \mathrm{PWConv}_{16\to C}\big(\mathrm{SiLU}(\mathrm{PWConv}_{20\to 16}(\mathbf{r}_u))\big)$ (khởi tạo trọng số và bias lớp cuối bằng $0$), kết hợp nhân với cổng xác suất tiền cảnh $\sigma(s_0(u))$ của chính ô $u$:
$$\mathbf{s}_{\mathrm{ref}}(u) = \mathbf{s}_{\mathrm{raw}}(u) + \sigma\!\big(s_0(u)\big)\,\mathcal{M}_s(\mathbf{r}_u)$$ (2.16)
Trong đó: $\mathbf{s}_{\mathrm{ref}}(u) \in \mathbb{R}^C$ là logit sau điều biến. Khi khởi tạo, $\mathbf{s}_{\mathrm{ref}}(u) \equiv \mathbf{s}_{\mathrm{raw}}(u)$ nên trọng số tiền huấn luyện không bị xáo trộn; trong suốt quá trình học, nhân tử $\sigma(s_0(u))$ giữ logit tại các ô nền ($\sigma(s_0) < 10^{-2}$) đứng yên, chỉ điều chỉnh tại các ô ứng viên tiền cảnh để hạ điểm dự đoán trùng ($M_{\mathrm{dom}}$ lớn) hoặc tăng điểm mục tiêu nhỏ có nhiều láng giềng đồng thuận ($M_{\mathrm{sup}}$ lớn).

### 2.4. Gán nhãn Wasserstein tách rời (DW-TAL) và chưng cất phân phối hộp giảm dần

![](docs/report/figures/fig_dwtal_flow_vi.png)

Hình 2.5: Sơ đồ luồng hai nhánh tách rời của DW-TAL: nhánh trên chọn mẫu dương bằng kết hợp $\mathrm{CIoU}$–$S_W$ qua cổng kích thước logistic; nhánh dưới giới hạn ngưỡng trên mục tiêu mềm phân loại theo $\mathrm{IoU}$ thực tế để giảm báo nhầm nền

#### 2.4.1. Cổng điều biến kích thước logistic bảo vệ đối tượng vừa và lớn

Gọi $s_{\mathrm{gt}} = \sqrt{w_{\mathrm{gt}}\,h_{\mathrm{gt}}}$ là kích thước hình học của hộp GT ở đầu vào. Để tín hiệu Wasserstein chỉ hỗ trợ vật nhỏ dễ bị mất gradient $\mathrm{IoU}$ mà không làm nhiễu các đối tượng vừa và lớn vốn hội tụ tốt dưới $\mathrm{CIoU}$ [27], trọng số kích hoạt được điều biến bởi cổng logistic:
$$g(s_{\mathrm{gt}}) = \sigma\!\left(\frac{\tau_{\mathrm{px}} - s_{\mathrm{gt}}}{\beta_{\mathrm{px}}}\right)$$ (2.17)
Trong đó: $\tau_{\mathrm{px}} = 20{,}0$ pixel đặt tại ranh giới nhóm *tiny* và độ dốc $\beta_{\mathrm{px}} = 3{,}5$ pixel. Như minh họa trên Hình 2.4(a), hàm suy giảm đuôi nặng $(1 + s_{\mathrm{gt}}^2/32^2)^{-1}$ vẫn rò rỉ 25–40% trọng số Wasserstein sang người 40–55 pixel, còn cổng logistic (2.17) đạt $g(5) = 0{,}986$ ở vật 5 pixel nhưng tắt về $g(35) = 0{,}014$ ngay khi vượt ngưỡng 32 pixel.

#### 2.4.2. Thuật toán tách rời khâu chọn mẫu và hiệu chuẩn mục tiêu mềm (DW-TAL)

Trong bước chọn mẫu dương của cả hai nhánh $\mathcal{H}_{1:M}$ và $\mathcal{H}_{1:1}$, chỉ số căn chỉnh $t = s^\alpha \tilde{u}^\beta$ dùng độ chồng lấn pha trộn giữa $\mathrm{CIoU}$ và độ tương đồng Wasserstein hữu tỷ $S_W = (1 + D)^{-1} \in (0, 1]$:
$$\tilde{u}(B_p, B_g) = \big(1 - \alpha_{\mathrm{tal}}(s_{\mathrm{gt}})\big)\,\mathrm{CIoU}(B_p, B_g) + \alpha_{\mathrm{tal}}(s_{\mathrm{gt}})\,S_W(B_p, B_g)$$ (2.18)
Trong đó: $(B_p, B_g) = (B_{\mathrm{pred}}, B_{\mathrm{gt}})$ là cặp hộp dự đoán và nhãn thật, $\alpha_{\mathrm{tal}}(s_{\mathrm{gt}}) = \lambda_{\mathrm{tal}}\,g(s_{\mathrm{gt}})$ với $\lambda_{\mathrm{tal}} = 0{,}45$. Nhờ $S_W > 0$ ngay khi dự đoán chưa chạm vào GT 3–4 pixel, các đối tượng cực nhỏ vẫn thu được tập mẫu dương ban đầu.

Ngay sau khi bộ gán nhãn động căn chỉnh tác vụ (TAL) chọn xong tập mẫu dương và tạo mục tiêu phân loại mềm $\mathbf{t}_{\mathrm{cls}}$ theo (1.2), DW-TAL thực hiện tái hiệu chuẩn theo $\mathrm{IoU}$ thực trên toàn bộ lưới ứng viên:
$$u_{\mathrm{cap}} = \max\!\big(\mathrm{IoU}(B_p, B_g),\; \eta\,g(s_{\mathrm{gt}})\big)$$ (2.19)
$$\mathbf{t}_{\mathrm{cls}}^{\mathrm{cal}} = \min\!\left(\mathbf{t}_{\mathrm{cls}} \cdot \mathrm{clip}\!\left(\frac{u_{\mathrm{cap}}}{(1 - \alpha_{\mathrm{tal}})\,\mathrm{IoU} + \alpha_{\mathrm{tal}}\,S_W + 10^{-4}},\, 0,\, 1{,}15\right),\; u_{\mathrm{cap}}\right)$$ (2.20)
Trong đó: $\mathrm{IoU}(B_p, B_g) \in [0, 1]$ là chỉ số IoU thực tế, $\eta = 0{,}08$ là ngưỡng sàn tối thiểu chỉ kích hoạt trên vật nhỏ khi $\mathrm{IoU} = 0$ ở giai đoạn đầu huấn luyện, và $\mathbf{t}_{\mathrm{cls}}^{\mathrm{cal}}$ là mục tiêu phân loại mềm đã hiệu chuẩn đưa vào hàm mất mát BCE. Hình 2.3(c) đối chiếu trực tiếp hai cơ chế: trên một hộp 4 pixel lệch tâm ($S_W = 0{,}65, \mathrm{IoU} \approx 0$), cơ chế ghép cứng đẩy mục tiêu lên trên $0{,}23\text{–}0{,}55$, còn DW-TAL giới hạn ở mức $\le 0{,}08$ rồi bám tuyến tính theo $\mathrm{IoU}$ thực khi hộp co khít.

Hàm mất mát hồi quy hộp trên tập mẫu dương $\mathcal{P}$ cộng thêm thành phần khoảng cách Wasserstein bị chặn có điều biến kích thước:
$$\mathcal{L}_{\mathrm{box}} = \mathcal{L}_{\mathrm{CIoU}} + \frac{\lambda_{\mathrm{reg}}}{\sum_{u \in \mathcal{P}} w_{\mathrm{pos}}(u)} \sum_{u \in \mathcal{P}} w_{\mathrm{pos}}(u)\,g\big(s_{\mathrm{gt}}(u)\big)\,\frac{D_u}{1 + D_u}$$ (2.21)
Trong đó: $D_u = D(B_p(u), B_g(u))$, $w_{\mathrm{pos}}(u) = \sum_{c=1}^{C} t_{\mathrm{cls},c}^{\mathrm{cal}}(u)$ và $\lambda_{\mathrm{reg}} = 0{,}20$.

#### 2.4.3. Chưng cất phân phối hộp Gauss giảm dần giữa hai nhánh

Nhánh $\mathcal{H}_{1:1}$ chỉ gán 1 ô dương cho mỗi vật thể nên tín hiệu học biên hộp thưa hơn nhánh phụ trợ $\mathcal{H}_{1:M}$. Trong quá trình huấn luyện, trường phân phối hộp $(\boldsymbol{\mu}_{\mathrm{aux}}^{(s)}, \boldsymbol{\sigma}_{\mathrm{aux}}^{(s)})$ của nhánh $\mathcal{H}_{1:M}$ (đã ngắt lan truyền ngược gradient) truyền hướng dẫn hình học sang dự đoán Gauss thô $(\boldsymbol{\mu}_0^{(s)}, \boldsymbol{\sigma}_0^{(s)})$ của nhánh $\mathcal{H}_{1:1}$ trên $s \in \{P_2, P_3\}$ tại các ô có $\hat{p}_{\mathrm{aux}}^{(s)}(u) = \max_c \sigma(\mathbf{s}_{\mathrm{aux}}^{(s)}(u, c)) > 0{,}20$:
$$\mathcal{L}_{\mathrm{distill}} = \lambda_{\mathrm{dist}}\,\kappa(e)\,\frac{\sum_{s \in \{P_2, P_3\}} \sum_{u} w_{\mathrm{dist}}^{(s)}(u)\,\frac{D_{\mathrm{dist}}^{(s)}(u)}{1 + D_{\mathrm{dist}}^{(s)}(u)}}{\sum_{s \in \{P_2, P_3\}} \sum_{u} w_{\mathrm{dist}}^{(s)}(u) + 10^{-6}}$$ (2.22)
Trong đó: $D_{\mathrm{dist}}^{(s)}(u) = D(\mathcal{G}_0^{(s)}(u), \mathcal{G}_{\mathrm{aux}}^{(s)}(u))$, $w_{\mathrm{dist}}^{(s)}(u) = (\hat{p}_{\mathrm{aux}}^{(s)}(u))^2\,\mathbb{I}[\hat{p}_{\mathrm{aux}}^{(s)}(u) > 0{,}20]$, $\lambda_{\mathrm{dist}} = 0{,}25$, và hệ số suy giảm $\kappa(e) = \mathrm{clip}(\gamma_{\mathrm{aux}}(e)/\gamma_{\mathrm{aux}}(0),\, 0{,}05,\, 1{,}0)$ (trong đó $\gamma_{\mathrm{aux}}(e)$ là hệ số trọng số theo lịch trình huấn luyện của nhánh $\mathcal{H}_{1:M}$ tại epoch $e$, khởi tạo $\gamma_{\mathrm{aux}}(0) = 0{,}8$, không phải giá trị hàm mất mát) giúp giảm dần ảnh hưởng của nhánh phụ ở chặng hội tụ cuối. Trình tự suy luận toán học của GauCon tại một mức $s \in \{P_2, P_3\}$ được tóm tắt trong Thuật toán 2.1.

Đầu vào: Bản đồ đặc trưng $\mathbf{X}_s \in \mathbb{R}^{B \times C_s \times H_s \times W_s}$ và tham số của ô cha tại mức $s+1$.

Đầu ra: Khoảng cách biên đã hiệu chỉnh $\mathbf{D}_{\mathrm{ref}}^{(s)}$ và logit phân loại đã hiệu chỉnh $\mathbf{S}_{\mathrm{ref}}^{(s)}$.

1. Tính khoảng cách biên thô $\mathbf{D}_0^{(s)}$ và logit phân loại thô $\mathbf{S}_{\mathrm{raw}}^{(s)}$ từ $\mathbf{X}_s$.
2. Với mỗi ô $u=(y,x)$, chuyển khoảng cách biên thành tâm và độ lệch chuẩn Gauss theo (2.5). Tính điểm tiền cảnh $s_0(u)=\max_c \mathbf{s}_{\mathrm{raw}}^{(s)}(u,c)$.
3. Lấy 8 ô lân cận trong cửa sổ $3\times3$ và đưa tâm hộp về hệ tọa độ cục bộ của ô $u$ theo (2.7).
4. Lấy ô cha $u_p=(\lfloor y/2\rfloor,\lfloor x/2\rfloor)$ từ mức $s+1$ và quy đổi tọa độ theo (2.8).
5. Tính khoảng cách $D_k(u)$ cho 9 ứng viên theo (2.9). Loại các ứng viên vượt biên hoặc khác lớp bằng cách đặt khoảng cách lớn.
6. Tính trọng số $w_k(u)$ bằng softmax của điểm tin cậy và khoảng cách hình học theo (2.10)–(2.11).
7. Cập nhật tâm và độ lệch chuẩn bằng nội suy với hệ số $\beta_{\mathrm{box}}^{(s)}$ theo (2.12)–(2.13).
8. Tính các đặc trưng quan hệ lân cận theo (2.14)–(2.15), sau đó điều chỉnh logit theo (2.16).
9. Chuyển phân phối Gauss đã hiệu chỉnh về bốn khoảng cách biên theo (2.6) và trả về hai đầu ra.

### 2.5. Phân rã luồng tensor, chi phí tham số và căn cứ lựa chọn thiết kế

Bảng 2.2 theo dõi kích thước các tensor trung gian qua GauCon ở độ phân giải đầu vào $1280 \times 1280$ ($C = 1$), và Bảng 2.3 thống kê chi tiết từng tham số học được tăng thêm so với YOLO26n-P2 nền.

Bảng 2.2: Luồng biến đổi tensor qua Head GauCon với đầu vào $1280 \times 1280$ ($C = 1$)

| Giai đoạn xử lý | Ký hiệu Tensor | Kích thước trên $P_2$ ($r_2 = 4$) | Kích thước trên $P_3$ ($r_3 = 8$) | Vai trò trong đồ thị |
|---|---|---|---|---|
| Đầu vào từ Neck PANet | $\mathbf{X}_s$ | $B \times 32 \times 320 \times 320$ | $B \times 64 \times 160 \times 160$ | Bản đồ đặc trưng đầu vào của Head |
| Dự đoán thô nhánh $\mathcal{H}_{1:1}$ | $\mathbf{D}_0^{(s)},\, \mathbf{S}_{\mathrm{raw}}^{(s)}$ | $B \times 4 \times 320^2$, $B \times 1 \times 320^2$ | $B \times 4 \times 160^2$, $B \times 1 \times 160^2$ | Khoảng cách $(l,t,r,b)$ và logit thô |
| Tham số Gauss gốc | $\boldsymbol{\mu}_0^{(s)},\, \boldsymbol{\sigma}_0^{(s)}$ | $B \times 2 \times 320 \times 320$ | $B \times 2 \times 160 \times 160$ | Ánh xạ (2.5) ở độ chính xác FP32 |
| Trượt lấy mẫu theo chỉ số + Ô cha | $\tilde{\boldsymbol{\mu}}_{1..9},\, \tilde{\boldsymbol{\sigma}}_{1..9}$ | $B \times 2 \times 9 \times 320 \times 320$ | $B \times 2 \times 9 \times 160 \times 160$ | 8 ô $3 \times 3$ + 1 ô cha mức $s + 1$ |
| Khoảng cách vận chuyển | $\mathbf{D}_{1..9}$ | $B \times 9 \times 320 \times 320$ | $B \times 9 \times 160 \times 160$ | Khoảng cách (2.9) gần bất biến theo tỉ lệ |
| Đặc trưng quan hệ lân cận | $\mathbf{r}_u$ | $B \times 20 \times 320 \times 320$ | $B \times 20 \times 160 \times 160$ | Ghép $[F_k, M_k, M_{\mathrm{dom}}, M_{\mathrm{sup}}]$ |
| Tầng ẩn cổng phân loại | $\mathbf{h}_u$ | $B \times 16 \times 320 \times 320$ | $B \times 16 \times 160 \times 160$ | $\mathrm{SiLU}(\mathrm{PWConv}_{20\to 16}(\mathbf{r}_u))$ |
| Đầu ra tinh chỉnh | $\mathbf{D}_{\mathrm{ref}}^{(s)},\, \mathbf{S}_{\mathrm{ref}}^{(s)}$ | $B \times 4 \times 320^2$, $B \times 1 \times 320^2$ | $B \times 4 \times 160^2$, $B \times 1 \times 160^2$ | Trọng tâm (2.12) & cổng tiền cảnh (2.16) |

Bảng 2.3: Thống kê tham số học được tăng thêm của GauCon trên $P_2$ và $P_3$ ($C = 1, d_{\mathrm{rel}} = 16$)

| Thành phần trong GauCon | Biểu thức tính số tham số | Số tham số trên 1 mức | Tổng trên $\{P_2, P_3\}$ |
|---|---|---|---|
| Bộ dịch chỉ số lân cận $3 \times 3$ | Chỉ số cố định, không có trọng số khả học | 0 | 0 |
| Tham số trọng tâm nhánh hộp $(\hat{\tau}_{\mathrm{box}}, \hat{\gamma}_{\mathrm{box}}, \hat{\beta}_{\mathrm{box}})$ | $3 \times 1$ tham số vô hướng | 3 | 6 |
| Tham số nhiệt độ gắn kết phân loại $\hat{\tau}_{\mathrm{cls}}$ | $1 \times 1$ tham số vô hướng | 1 | 2 |
| Tầng chiếu ẩn $\mathcal{M}_s^{(1)}$ ($\mathrm{PWConv}_{20 \to 16}$, có bias, không BN) | $20 \times 16 + 16$ | 336 | 672 |
| Tầng chiếu ra $\mathcal{M}_s^{(2)}$ ($\mathrm{PWConv}_{16 \to 1}$, có bias, khởi tạo 0) | $16 \times 1 + 1$ | 17 | 34 |
| Tổng tham số tăng thêm của mô-đun GauCon | $2 \times (3 + 1 + 336 + 17)$ | 357 | 714 |
| Tổng tham số mạng trước khi hợp nhất | Nền: $2.515.976 \to$ GauCon: $2.516.690$ | – | +714 (+0,028%) |
| Tổng tham số suy luận sau khi hợp nhất | Nền: $2.401.660 \to$ GauCon: $2.402.374$ | – | +714 (+0,030%) |

Con số 714 tham số đếm từ Bảng 2.3 khớp chính xác với hiệu số tham số đo được từ mô hình thực tế trước và sau hợp nhất ($2.516.690 - 2.515.976 = 714$ và $2.402.374 - 2.401.660 = 714$). Thay vì gắn thêm các khối tích chập đa nhánh trong Neck (thường tốn hàng chục nghìn tham số trên bản đồ đặc trưng $320 \times 320$), xử lý trực tiếp trên 5 kênh hình học đã giải mã $(\mu_x, \mu_y, \sigma_x, \sigma_y, s_0)$ giữ mức tăng thêm dưới 0,03% tổng số tham số.

### 2.6. Cấu hình huấn luyện đối chứng

Để tách ảnh hưởng của mô-đun và cơ chế gán nhãn khỏi các biến nhiễu khác (như kích thước batch, khởi tạo trọng số hay bộ tối ưu), tất cả mô hình trong thực nghiệm đối chứng chính chạy trên cùng hạ tầng PyTorch có tăng tốc CUDA (FP16) trên GPU NVIDIA L4 24 GB dưới cấu hình thống nhất tại Bảng 2.4.

Bảng 2.4: Cấu hình huấn luyện dùng chung cho các mô hình đối chứng (80 epoch, khởi tạo từ trọng số COCO)

| Nhóm cấu hình | Giá trị thiết lập | Căn cứ kỹ thuật |
|---|---|---|
| Trọng số khởi tạo | Trọng số YOLO26n huấn luyện trên MS COCO | Các tầng tương ứng được nạp từ checkpoint COCO; nhánh $P_2$ và các phần thay đổi được khởi tạo theo cùng quy trình |
| Độ phân giải đầu vào | $1280 \times 1280$ (*letterbox* giữ tỉ lệ) | Bội số 32; đảm bảo lưới $P_2$ đạt $320 \times 320$ |
| Thời lượng & Batch size | 80 epoch, batch size 16, FP16 AMP | Thống nhất batch size 16 trên GPU NVIDIA L4 24 GB |
| Bộ tối ưu & Lịch học | AdamW [29], $\text{lr}_0 = 0{,}0015$, $\text{lrf} = 0{,}05$ (cosine), *warmup* 3 epoch | Tinh chỉnh trọng số tiền huấn luyện ổn định mà không phá vỡ các tầng Backbone đầu |
| Tăng cường dữ liệu | Lật ngang $0{,}5$, dịch $0{,}05$, tỉ lệ $0{,}15$, HSV $(0{,}005; 0{,}10; 0{,}15)$; tắt Mosaic & MixUp | Ghép Mosaic thu nhỏ vật thêm 2–4 lần khiến người 2–6 px rơi xuống dưới 1 px |
| Tham số TAL dùng chung | $\alpha = 0{,}5$, $\beta = 4{,}0$ cho Base, NWD và GauCon | Giữ nguyên hệ số căn chỉnh giữa các cấu hình đối chứng |
| Tham số GauCon & DW-TAL | Kích hoạt $P_2, P_3$, bật FG-Gate; $\tau_{\mathrm{px}} = 20{,}0$, $\beta_{\mathrm{px}} = 3{,}5$, $\beta_{\mathrm{box}}$ khởi tạo 0,06, $\lambda_{\mathrm{tal}} = 0{,}45$, $\lambda_{\mathrm{reg}} = 0{,}20$, $\lambda_{\mathrm{dist}} = 0{,}25$ | Khoanh vùng tác động vào dải $\le 20$ px trên $P_2, P_3$ |
| Đối chứng NWD chuẩn | Trọng số gán nhãn $\alpha_{\mathrm{NWD}} = 0{,}50$, mất mát $\lambda_{\mathrm{NWD}} = 0{,}50$, $C = 12{,}8$ px | Cấu hình gốc của bài báo NWD [13, 14] |
| Chọn mô hình đánh giá | Chọn mô hình tốt nhất theo $mAP_{50}$ trên 84 ảnh kiểm định (*val*), seed = 42 | Tập kiểm tra (*test*) 786 ảnh chỉ chạy đánh giá một lần sau khi kết thúc huấn luyện |

---

## 3. THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

### 3.1. Bộ dữ liệu, quy trình chống rò rỉ chuỗi video và giao thức đánh giá kép

TinyPerson [1] gồm 1.610 khung hình trích cách mỗi 50 khung từ 31 video ven biển, chia sẵn thành 794 ảnh huấn luyện và 816 ảnh kiểm tra. Theo quy ước lọc ảnh quá dày đặc của tác giả [1] (bỏ khung hình trên 220 nhãn hoặc mang cờ đánh dấu ảnh mật độ cực dày đặc), dữ liệu thực nghiệm gồm 746 khung hình huấn luyện và 786 khung hình kiểm tra chính thức. Do các khung hình trong cùng một video có bối cảnh sóng và góc máy gần nhau, chia ngẫu nhiên theo ảnh sẽ gây rò rỉ cảnh quay sang tập kiểm định. Do đó, tập huấn luyện được tách theo tiền tố chuỗi video thành 662 ảnh huấn luyện (*train*) và 84 ảnh kiểm định (*val*).

Bảng 3.1: Phân bố số lượng hộp Ground Truth và phân tầng kích thước trên tập kiểm tra TinyPerson (786 ảnh)

| Thang kích thước (*Scale Bin*) | Định nghĩa theo $s = \sqrt{w \cdot h}$ (pixel ảnh gốc) | Số hộp Ground Truth (GT) | Tỷ trọng trong tập kiểm tra (%) |
|---|---|---|---|
| *tiny1* | $[2,\, 8)$ | 2.333 | 17,05 |
| *tiny2* | $[8,\, 12)$ | 2.778 | 20,30 |
| *tiny3* | $[12,\, 20]$ | 3.767 | 27,52 |
| *tiny* *(gộp 3 dải nhỏ)* | $[2,\, 20]$ | 8.878 | 64,86 |
| *small* | $(20,\, 32]$ | 2.485 | 18,16 |
| *reasonable* | $(32,\, \infty)$ | 2.324 | 16,98 |
| *all* *(toàn bộ đối tượng)* | $[2,\, \infty)$ | 13.687 | 100,00 |

Ngoài 13.687 hộp nhãn thật GT (Bảng 3.1), tập kiểm tra còn chứa 1.989 vùng bỏ qua (*ignore*) và 2.832 vùng không chắc chắn (*uncertain*). Bộ chấm điểm tuân theo quy chuẩn WACV 2020 [1]: hộp GT khớp theo $\mathrm{IoU}$, vùng *ignore*/*uncertain* khớp theo chỉ số giao trên diện tích dự đoán $\mathrm{IoD} = \mathrm{Area}(B_{\mathrm{det}} \cap B_{\mathrm{ign}}) / \mathrm{Area}(B_{\mathrm{det}}) > 0{,}50$ để không phạt các hộp rơi vào đám đông chưa gắn nhãn chi tiết. Bên cạnh $AP_{25}, AP_{50}, AP_{75}$ và $MR_{50}$ (trung bình log tỷ lệ bỏ sót tại 9 mốc FPPI trong $[10^{-2}, 10^0]$; $MR_{50}$ càng thấp càng tốt), mọi cấu hình được chấm thêm chỉ số COCO $AP_{50:95}$ (giới hạn tối đa 500 dự đoán mỗi ảnh), phân rã TIDE [37] tại ngưỡng điểm $0{,}05$ và độ phủ theo khoảng cách láng giềng gần nhất $d_{\mathrm{NN}}$.

Mỗi mô hình tốt nhất trên tập kiểm định được đánh giá ở cùng lúc hai giao thức suy luận:
1. Suy luận theo lát cắt độ phân giải gốc (Tiled 640×512): Chia ảnh gốc thành các lát cắt $640 \times 512$ chồng lấn $100 \times 100$ pixel, suy luận FP16 theo batch trên GPU rồi hợp nhất bằng NMS theo lớp ($\mathrm{IoU} = 0{,}50$, giữ tối đa 2.000 hộp trước NMS và 1.000 hộp sau NMS) kèm lọc hộp cắt ngang biên lát cắt ($\mathrm{IOS} \ge 0{,}85$).
2. Suy luận toàn khung một lượt (Full-Frame 1280×1280): Đưa ảnh qua phép *letterbox* về $1280 \times 1280$ và suy luận một lượt qua mô hình đã hợp nhất không dùng NMS (tối đa 200 dự đoán/ảnh).

Bên cạnh hai giao thức trên, đồ án thực hiện thêm hai kiểm định hậu nghiệm trực tiếp trên tập dự đoán đã lưu: (a) áp dụng Gaussian Soft-NMS [31] ($\sigma = 0{,}5$) trực tiếp lên đầu ra toàn khung của Base làm mốc đối chứng hậu xử lý ngoài đồ thị; và (b) chạy 1.000 lần paired image-level bootstrap (seed = 42) trên 786 ảnh kiểm tra để ước lượng khoảng tin cậy 95% (95% CI) cho sai biệt $\Delta AP_{50}^{all}$ và $\Delta AP_{75}^{all}$. Giá trị $P(\Delta > 0)$ được tính bằng tỷ lệ mẫu bootstrap có sai biệt dương, không phải p-value của một kiểm định giả thuyết.

### 3.2. Kết quả định lượng ở chế độ lát cắt độ phân giải gốc ($640 \times 512$)

Bảng 3.2 và Hình 3.1(a) tổng hợp kết quả đo trên 786 ảnh kiểm tra theo giao thức lát cắt 1:1 cho YOLO26n-P2 Base, YOLO26n-P2 + NWD ($C = 12{,}8$ px) và YOLO26n-P2 + GauCon.

Bảng 3.2: Kết quả đối chứng trên tập kiểm tra TinyPerson ở chế độ lát cắt gốc $640 \times 512$ (80 epoch, pretrained COCO, batch 16, seed 42)

| Thang đo / Chỉ số | Số GT | YOLO26n-P2 Base | YOLO26n-P2 + NWD | YOLO26n-P2 + GauCon | $\Delta$ (GauCon − Base) | $\Delta$ (GauCon − NWD) |
|---|---|---|---|---|---|---|
| *all* $AP_{50}$ (%) $\uparrow$ | 13.687 | 45,58 | 48,45 | 48,53 | +2,95 | +0,08 |
| *tiny* $AP_{50}$ (%) $\uparrow$ | 8.878 | 42,59 | 45,31 | 43,99 | +1,40 | −1,32 |
| *tiny1* (2–8 px) $AP_{50}$ (%) $\uparrow$ | 2.333 | 25,10 | 27,91 | 25,53 | +0,43 | −2,38 |
| *tiny2* (8–12 px) $AP_{50}$ (%) $\uparrow$ | 2.778 | 45,14 | 46,73 | 46,97 | +1,83 | +0,24 |
| *tiny3* (12–20 px) $AP_{50}$ (%) $\uparrow$ | 3.767 | 50,94 | 54,82 | 52,95 | +2,01 | −1,87 |
| *small* (20–32 px) $AP_{50}$ (%) $\uparrow$ | 2.485 | 54,55 | 56,83 | 56,07 | +1,52 | −0,76 |
| *reasonable* (>32 px) $AP_{50}$ (%) $\uparrow$ | 2.324 | 49,78 | 53,51 | 58,11 | +8,33 | +4,60 |
| *all* $AP_{75}$ (%) $\uparrow$ | 13.687 | 6,40 | 6,09 | 7,37 | +0,97 | +1,28 |
| *all* $AP_{25}$ (%) $\uparrow$ | 13.687 | 70,02 | 71,93 | 71,56 | +1,54 | −0,37 |
| *all* $MR_{50}$ (%) $\downarrow$ | 13.687 | 91,95 | 90,74 | 90,31 | −1,64 | −0,43 |
| COCO $AP_{50:95}$ (@500) (%) $\uparrow$ | 13.687 | 15,55 | 16,51 | 16,96 | +1,41 | +0,45 |

*Ghi chú: Giá trị tốt nhất trên từng dòng được in đậm; $\Delta$ tính bằng điểm phần trăm.*

![](docs/report/figures/fig_gaucon_scales.png)

Hình 3.1: Độ chính xác $AP_{50}$ theo bảy thang kích thước trên tập kiểm tra TinyPerson: *(a)* Chế độ lát cắt độ phân giải gốc $640 \times 512$; *(b)* Chế độ suy luận toàn khung $1280 \times 1280$

Số liệu Bảng 3.2 cho thấy sự phân hóa rõ ràng giữa độ phủ ở nhóm siêu nhỏ và độ chính xác định vị chặt:
1. So với YOLO26n-P2 Base: GauCon nâng $AP_{50}^{all}$ từ 45,58% lên 48,53% (+2,95 điểm phần trăm, 95% Bootstrap CI [+1,92; +4,02], $P(\Delta > 0) = 100\%$), hạ $MR_{50}^{all}$ đi 1,64 điểm và cải thiện ở các thang đo, ngoại trừ các trường hợp được nêu riêng (tuy nhiên mức tăng tập trung mạnh nhất ở *reasonable* +8,33 điểm, được phân rã tại Mục 3.4; còn *tiny1* chỉ tăng +0,43 điểm). Tại ngưỡng chặt $AP_{75}^{all}$, GauCon tăng từ 6,40% lên 7,37% (+0,97 điểm, 95% CI [+0,51; +1,62]), đưa COCO $AP_{50:95}$ từ 15,55% lên 16,96%.
2. So với YOLO26n-P2 + NWD (đánh đổi giữa độ phủ nới lỏng và độ khít biên hộp): NWD đạt $AP_{50}^{all} = 48{,}45\%$, tương đương GauCon (+0,08 điểm, 95% CI [−1,06; +1,17], $P = 55{,}7\%$), và vượt GauCon trên hai dải siêu nhỏ *tiny1* (27,91% so với 25,53%, −2,38 điểm) lẫn *tiny3* (54,82% so với 52,95%). Đổi lại, việc NWD nới lỏng biên hộp khiến $AP_{75}^{all}$ rớt xuống dưới Base (6,40% xuống 6,09%), thấp hơn GauCon 1,28 điểm phần trăm (95% CI [+0,54; +1,88], $P = 100\%$), và kéo COCO $AP_{50:95}$ (16,51%) thấp hơn GauCon (16,96%).

### 3.3. Kết quả ở chế độ suy luận toàn khung một lượt ($1280 \times 1280$)

Khi đưa thẳng cả khung hình qua *letterbox* $1280 \times 1280$ và bỏ hoàn toàn lát cắt lẫn bước hậu xử lý NMS, Bảng 3.3 và Hình 3.1(b) phản ánh chất lượng thực chất của nhánh $\mathcal{H}_{1:1}$.

Bảng 3.3: Kết quả đối chứng trên tập kiểm tra TinyPerson ở chế độ suy luận toàn khung $1280 \times 1280$ không dùng NMS (%)

| Thang đo / Chỉ số | YOLO26n-P2 Base | YOLO26n-P2 + NWD | YOLO26n-P2 + GauCon | $\Delta$ (GauCon − Base) | $\Delta$ (GauCon − NWD) |
|---|---|---|---|---|---|
| *all* $AP_{50}$ (%) $\uparrow$ | 43,08 | 43,24 | 44,13 | +1,05 | +0,89 |
| *tiny* $AP_{50}$ (%) $\uparrow$ | 36,28 | 35,74 | 36,06 | −0,22 | +0,32 |
| *tiny1* (2–8 px) $AP_{50}$ (%) $\uparrow$ | 18,71 | 18,46 | 18,34 | −0,37 | −0,12 |
| *tiny2* (8–12 px) $AP_{50}$ (%) $\uparrow$ | 37,48 | 36,53 | 37,90 | +0,42 | +1,37 |
| *tiny3* (12–20 px) $AP_{50}$ (%) $\uparrow$ | 46,23 | 45,94 | 45,79 | −0,44 | −0,06 |
| *small* (20–32 px) $AP_{50}$ (%) $\uparrow$ | 53,07 | 52,90 | 54,32 | +1,25 | +1,42 |
| *reasonable* (>32 px) $AP_{50}$ (%) $\uparrow$ | 58,77 | 62,48 | 63,09 | +4,32 | +0,61 |
| *all* $AP_{75}$ (%) $\uparrow$ | 4,44 | 4,13 | 4,78 | +0,34 | +0,65 |
| *all* $AP_{25}$ (%) $\uparrow$ | 69,29 | 71,44 | 71,19 | +1,90 | −0,25 |
| *all* $MR_{50}$ (%) $\downarrow$ | 91,97 | 92,42 | 91,32 | −0,65 | −1,10 |
| COCO $AP_{50:95}$ (@500) (%) $\uparrow$ | 13,51 | 13,35 | 14,00 | +0,49 | +0,65 |
| Validation $mAP_{50}$ đỉnh (%) $\uparrow$ | 39,96 | 39,67 | 42,24 | +2,28 | +2,57 |

Khi không còn bước NMS gộp lát cắt che bớt nhiễu nền, sự khác biệt giữa hiệu chuẩn tách rời DW-TAL và ghép cứng NWD thể hiện rõ trên Bảng 3.3:
- Ở cấu hình NWD, chỉ số $mAP_{50}$ đỉnh trên tập kiểm định giảm từ 39,96% xuống 39,67%, còn trên tập kiểm tra toàn khung làm COCO $AP_{50:95}$ hạ từ 13,51% xuống 13,35%, $AP_{75}^{all}$ rớt xuống 4,13% và tỷ lệ bỏ sót $MR_{50}$ tăng lên 92,42%.
- Cấu hình GauCon nâng đỉnh $mAP_{50}$ kiểm định lên 42,24% (+2,28 điểm trên Base), dẫn đầu về $AP_{50}^{all}$ (44,13%, 95% CI [+0,01; +2,16] so với Base, $P=97{,}7\%$; [−0,12; +1,90] so với NWD, $P=95{,}6\%$), $AP_{75}^{all}$ (4,78%, 95% CI [+0,13; +1,31] so với NWD, $P=99{,}0\%$), COCO $AP_{50:95}$ (14,00%) và $MR_{50}^{all}$ (91,32%). Khi gắn hậu xử lý Gaussian Soft-NMS ($\sigma = 0{,}5$) trực tiếp lên dự đoán toàn khung của Base, $AP_{50}^{all}$ chỉ tăng nhẹ từ 43,08% lên 43,37% (+0,29 điểm) và $AP_{75}^{all}$ giữ nguyên 4,44% ($MR_{50} = 91{,}93\%$), vẫn thấp hơn GauCon 0,76 điểm $AP_{50}^{all}$ và 0,34 điểm $AP_{75}^{all}$.

Tuy vậy, ở mục tiêu cốt lõi *tiny1* (2–8 px) chế độ toàn khung, GauCon đạt 18,34%, thấp hơn cả Base (18,71%, −0,37 điểm) lẫn NWD (18,46%, −0,12 điểm) (và cũng thấp hơn Base −0,44 điểm ở *tiny3*), mặc dù $AP_{25}^{tiny1}$ tăng từ 45,19% lên 46,41%. Khi co khung ảnh 4K khoảng 3 lần về $1280 \times 1280$, người 2–8 px bị thu về 0,7–2,7 px (chỉ nằm trong 1 ô lưới $P_2$), khiến cơ chế đồng thuận $3 \times 3$ thiếu thông tin từ các ô lân cận.

![](docs/report/figures/fig_gaucon_pareto.png)

Hình 3.2: Biểu đồ tương quan tham số (thang log) và độ chính xác $AP_{50}$ tham chiếu từ tài liệu: đặt các cấu hình 80 epoch 1 lớp của đồ án cạnh các công bố cũ (2 lớp với hai giai đoạn [1, 26] và 300 epoch từ đầu với nano NMS [5]); do khác giao thức đánh giá, bộ lọc và khởi tạo, các mốc từ tài liệu chỉ mang tính tham chiếu quy mô kiến trúc

### 3.4. Phân tích lỗi TIDE, dải con *reasonable* và khoảng tin cậy Bootstrap

Bảng 3.4 và Hình 3.3 phân rã số lượng từng loại lỗi TIDE [37] (lấy ngưỡng điểm $\ge 0{,}05$) và độ phủ theo khoảng cách tâm người gần nhất $d_{\mathrm{NN}}$ từ kết quả đánh giá định lượng.

Bảng 3.4: Phân rã lỗi TIDE (ngưỡng điểm $\ge 0{,}05$) và độ phủ Recall@0.50 theo mật độ đám đông trên tập kiểm tra TinyPerson

| Chế độ & Mô hình | Khớp đúng $\mathrm{TP} \uparrow$ | Lỗi định vị $E_{\mathrm{loc}} \downarrow$ | Hộp trùng $E_{\mathrm{dupe}} \downarrow$ | Báo nhầm nền $E_{\mathrm{bkg}} \downarrow$ | Bỏ sót $E_{\mathrm{miss}} \downarrow$ | Recall cụm dày đặc < 10 px $\uparrow$ | Recall cụm gần $10\text{–}20$ px $\uparrow$ | Recall tách rời $\ge 20$ px $\uparrow$ |
|---|---|---|---|---|---|---|---|---|
| Full-1280: Base | 7.871 | 9.139 | 1.027 | 7.341 | 5.816 | 35,19% | 54,16% | 66,06% |
| Full-1280: + NWD | 8.004 | 9.137 | 858 | 9.021 | 5.683 | 38,71% | 55,02% | 66,24% |
| Full-1280: + GauCon | 8.003 | 8.408 | 799 | 8.683 | 5.684 | 38,06% | 54,80% | 66,53% |
| Tiled 640×512: Base | 8.702 | 9.656 | 937 | 16.422 | 4.985 | 53,12% | 64,37% | 66,66% |
| Tiled 640×512: + NWD | 9.149 | 10.444 | 922 | 22.658 | 4.538 | 57,92% | 66,06% | 70,05% |
| Tiled 640×512: + GauCon | 8.963 | 9.188 | 895 | 21.130 | 4.724 | 55,13% | 65,04% | 69,02% |

![](docs/report/figures/fig_gaucon_diagnostics.png)

Hình 3.3: Phân tích chẩn đoán lỗi chế độ toàn khung $1280 \times 1280$: *(a)* Số lượng từng loại lỗi theo phân rã TIDE; *(b)* Độ phủ Recall@0.50 phân tầng theo khoảng cách láng giềng gần nhất $d_{\mathrm{NN}}$

Bảng 3.4 và Hình 3.3 phản ánh đồng thời các mặt mạnh lẫn đánh đổi của phương pháp:
1. Giảm lỗi định vị ($E_{\mathrm{loc}}$) và lỗi trùng lặp ($E_{\mathrm{dupe}}$): Ở chế độ toàn khung, GauCon và NWD khôi phục cùng số khớp đúng (8.003 so với 8.004 TP, +132 so với Base), nhưng NWD giữ nguyên lỗi lệch biên (9.137 ở Full; tăng lên 10.444 ở Tiled so với 9.656 của Base), còn GauCon hạ $E_{\mathrm{loc}}$ xuống 8.408 (−731 hộp ở Full) và 9.188 (−1.256 hộp so với NWD ở Tiled). GauCon cũng đạt số hộp trùng $E_{\mathrm{dupe}}$ thấp nhất ở cả hai chế độ (799 hộp ở Full, giảm 22,2% từ 1.027 của Base). Tuy nhiên, do chưa có thí nghiệm tách bỏ từng thành phần (*ablation*), mức cải thiện này là hiệu ứng tổng hợp của toàn hệ thống chứ chưa tách riêng đóng góp của trọng tâm Wasserstein hay FG-Gate.
2. Báo nhầm nền ($E_{\mathrm{bkg}}$) và độ phủ cụm dày đặc (< 10 px): Cần ghi nhận thẳng thắn hai mặt hạn chế: (i) Báo nhầm nền ở chế độ lát cắt (Tiled) của GauCon tăng +28,7% so với Base (16.422 lên 21.130), chỉ thấp hơn mức tăng +38,0% của NWD (22.658); lợi thế giảm $E_{\mathrm{bkg}}$ so với NWD thể hiện rõ hơn ở Full-1280 (8.683 so với 9.021), nhưng vẫn cao hơn Base (7.341); (ii) Trên cụm người đứng sát < 10 px, độ phủ Recall@0.50 của GauCon tăng +2,87 điểm so với Base (35,19% lên 38,06% ở Full; 53,12% lên 55,13% ở Tiled) nhưng vẫn thấp hơn NWD ở cả hai chế độ (38,71% ở Full; 57,92% ở Tiled).
3. Phân tích mức tăng ở nhóm *reasonable* (>32 px) và kiểm định Bootstrap (Hình 3.4): GauCon không can thiệp vào $P_4, P_5$ lúc suy luận và cổng $g(s_{\mathrm{gt}})$ giảm mạnh ở các đối tượng lớn hơn 35 px. Vì vậy, mức tăng +8,33 điểm (Tiled) và +4,32 điểm (Full) trên *reasonable* cần được diễn giải thận trọng. Khi phân rã nhóm *reasonable* (2.324 GT) trên Hình 3.4(a), ở Tiled, dải 32–48 px (1.293 GT) tăng 55,05% lên 61,71% (+6,66), 48–64 px (557 GT) tăng 51,36% lên 61,20% (+9,84), còn >64 px (474 GT) tăng 34,30% lên 44,98% (+10,68); tương tự ở Full-1280, dải >64 px tăng 56,16% lên 63,63% (+7,47), trong khi NWD cũng tăng +6,00 điểm ở dải này (62,16%). Kết quả cho thấy mức tăng lớn nhất xuất hiện ở nhóm >48 px, dù GauCon chỉ hiệu chỉnh trực tiếp các đầu phát hiện tại $P_2$ và $P_3$. Một khả năng là thay đổi trong quá trình huấn luyện chung đã ảnh hưởng đến đặc trưng ở Backbone và Neck, cũng như điểm tin cậy của các tầng sâu. Đây mới là giả thuyết phù hợp với phân bố kết quả; chưa thể xác định cơ chế này nếu không có thí nghiệm tách riêng từng thành phần. Hình 3.4(b) báo cáo thêm khoảng tin cậy 95% Bootstrap cho từng cặp đối chứng.

![](docs/report/figures/fig_gaucon_diagnostics_ext.png)

Hình 3.4: Chẩn đoán bổ sung trên tập dự đoán kiểm tra (không huấn luyện thêm): *(a)* Phân rã *reasonable* (>32 px) thành 3 dải con (32–48, 48–64, >64 px) ở Tiled và đối chứng Base + Soft-NMS ($\sigma = 0{,}5$) ở Full-1280; *(b)* Khoảng tin cậy 95% Paired Image-level Bootstrap ($N = 1.000$) kèm xác suất $P(\Delta > 0)$ trên 786 ảnh

### 3.5. Động học hội tụ, đường cong ngưỡng IoU và quỹ đạo tham số nội tại

![](docs/report/figures/fig_gaucon_convergence.png)

Hình 3.5: Quỹ đạo độ chính xác $mAP_{50}$ trên tập kiểm định xuyên suốt 80 epoch huấn luyện của ba mô hình đối chứng

![](docs/report/figures/fig_gaucon_iou_curve.png)

Hình 3.6: Đường cong độ chính xác AP trên tập kiểm định theo ngưỡng đánh giá $\mathrm{IoU}$ từ $0{,}50$ đến $0{,}85$ tại mô hình tốt nhất

Hình 3.5 và Hình 3.6 cho thấy GauCon tách khỏi đường cơ sở Base và NWD từ khoảng epoch 50 trở đi, đạt đỉnh 42,24% tại epoch 70 và duy trì vị trí dẫn đầu đều từ $\mathrm{IoU} = 0{,}50$ tới $0{,}75$ trên tập kiểm định (*val*), trong khi đường của NWD nằm dưới Base bắt đầu từ ngưỡng $\mathrm{IoU} \ge 0{,}55$.

![](docs/report/figures/fig_gaucon_param_dynamics.png)

Hình 3.7: Quỹ đạo thực tế các tham số học được trong đầu phát hiện GauCon qua 80 epoch: *(a)* Bước trọng tâm $\beta_{\mathrm{box}}$ trên $P_2, P_3$ và hàm mất mát chưng cất $\mathcal{L}_{\mathrm{distill}}$; *(b)* Nhiệt độ phân bố $\tau_{\mathrm{box}}$ và $\tau_{\mathrm{cls}}$

Hình 3.7 ghi nhận động học thực tế của các tham số khả học nội tại trong suốt 80 epoch huấn luyện:
- Tham số bước nội suy $\beta_{\mathrm{box}}$ hội tụ giảm từ khởi tạo 0,060 về $\beta_{\mathrm{box}}^{(P_2)} = 0{,}0405$ và $\beta_{\mathrm{box}}^{(P_3)} = 0{,}0456$, nghĩa là thành phần hộp chỉ dịch chuyển khoảng 4% về phía trọng tâm lân cận. Mức bước nhỏ này cho thấy mô hình giữ hiệu chỉnh biên rất bảo thủ; khi chưa có thí nghiệm tắt riêng $\beta_{\mathrm{box}} = 0$, không thể quy toàn bộ mức giảm $E_{\mathrm{loc}}$ hay cải thiện $AP_{75}$ riêng cho phép dịch tâm mà phải đánh giá trong tổng thể cùng DW-TAL và chưng cất $\mathcal{H}_{1:M}$.
- Hệ số nhiệt độ nhánh hộp tăng lên $\tau_{\mathrm{box}}^{(P_2)} = 1{,}524$ và $\tau_{\mathrm{box}}^{(P_3)} = 1{,}598$, trong khi nhánh phân loại phân hóa thành $\tau_{\mathrm{cls}}^{(P_2)} = 1{,}492$ và $\tau_{\mathrm{cls}}^{(P_3)} = 1{,}366$. Song hành là mất mát chưng cất $\mathcal{L}_{\mathrm{distill}}$ giảm đều về mức 0,0248 ở epoch 80.

### 3.6. Phân tích định tính trên đa bối cảnh kiểm tra

![](docs/report/figures/fig_gaucon_qualitative_gallery.png)

Hình 3.8: Minh họa đối chiếu định tính trên ba bối cảnh tập kiểm tra TinyPerson (ngưỡng độ tin cậy cố định $\mathrm{conf} \ge 0{,}25$ cho mọi mô hình): *Hàng 1*: cụm 5 người ven bờ ánh chiều (lát cắt $640 \times 512$); *Hàng 2*: bãi biển đông người có đá nền phức tạp (lát cắt $640 \times 512$); *Hàng 3*: toàn khung $1280 \times 1280$ nhiều mục tiêu phân tán trên sóng biển

Hình 3.8 kết xuất trực tiếp kết quả dự đoán trên tập kiểm tra của ba mô hình tại cùng ngưỡng độ tin cậy $0{,}25$:
- Hàng 1 (Cảnh ven sóng tương phản thấp, lát cắt): Trong khung cảnh ánh sáng yếu ven mép sóng với 5 hộp GT, Base chỉ bắt được 2/5 người (điểm 0,52 và 0,29), NWD không có hộp nào vượt ngưỡng 0,25 (0/5), còn GauCon phát hiện đủ cả 5/5 người với độ tin cậy $0{,}30\text{–}0{,}55$ và không tạo hộp trùng.
- Hàng 2 (Cảnh bãi biển đông có vách đá, lát cắt): Trên vùng chứa 18 người nằm xen cạnh vách đá và các vùng bỏ qua *ignore* (nét đứt vàng cam), Base phát hiện 14 hộp, còn NWD (màu tím) phát sinh tới 58 hộp dự đoán dày đặc lên vùng nền và đá. GauCon giảm xuống còn 30 hộp, giảm bớt mật độ báo nhầm trên nền đá so với NWD, nhưng vẫn còn khoảng 12 dự đoán dương giả tại vùng đá phức tạp. Do Hình 3.8 chỉ chọn minh họa ba bối cảnh điển hình, các kết luận định lượng dựa trên toàn bộ 786 ảnh trong Bảng 3.2–3.4.
- Hàng 3 (Cảnh biển góc rộng nhiều mục tiêu, toàn khung 1280): Ở chế độ toàn khung trên cảnh chứa 19 GT vừa và nhỏ, GauCon dự đoán 18 hộp khớp sát vị trí người (18/19), trong khi Base bỏ sót còn 13/19 và NWD sinh lặp thành 23/19. Ở rìa mép trên nơi đối tượng co xuống chỉ còn 2–3 pixel sau *letterbox*, một vài cá thể rất nhỏ vẫn bị bỏ qua dưới cả ba mô hình, minh họa giới hạn độ phân giải vật lý của chế độ toàn khung một lượt.

### 3.7. Hiệu năng phần cứng và độ phức tạp tính toán trước/sau hợp nhất

Bảng 3.5 ghi nhận phép đo suy luận mô hình trên GPU NVIDIA L4 24 GB ở FP16, batch size 1, sau 15 lượt làm nóng và lấy trung bình 100 lượt chạy đồng bộ CUDA. Đây là phép đo trong một phiên, chưa phải kết quả profiling đa phiên.

Bảng 3.5: Độ phức tạp tính toán, bộ nhớ GPU và tốc độ suy luận một phiên trên NVIDIA L4 (FP16, batch size 1)

| Mô hình | Tham số trước hợp nhất | Tham số sau hợp nhất | GFLOPs trước / sau hợp nhất (@1280) | VRAM đỉnh (MB) | Độ trễ @1280 (ms) | Tốc độ (FPS) | Full $AP_{50}$ (%) | Tiled $AP_{50}$ (%) |
|---|---|---|---|---|---|---|---|---|
| YOLO26n-P2 Base | 2.515.976 | 2.401.660 | 31,948 / 27,683 | 166,75 | 14,39 | 69,51 | 43,08 | 45,58 |
| YOLO26n-P2 + NWD | 2.515.976 | 2.401.660 | 31,948 / 27,683 | 166,75 | 13,45 | 74,36 | 43,24 | 48,45 |
| YOLO26n-P2 + GauCon | 2.516.690 | 2.402.374 | 32,034 / 27,683 | 186,23 | 13,52 | 73,95 | 44,13 | 48,53 |

Kết quả Bảng 3.5 mô tả chi phí tính toán và phép đo suy luận trong một phiên:
- Trước hợp nhất, bộ đo ghi nhận 32,034 GFLOPs cho GauCon và 31,948 GFLOPs cho Base. Sau hợp nhất, bộ đo báo 27,683 GFLOPs cho cả hai; công cụ hiện dùng không tính phần chi phí của MLP $1\times1$ tùy biến trong đầu GauCon. Ước lượng riêng từ số phép nhân-cộng của MLP cho thấy phần này cần khoảng 43 triệu MAC, tương đương 0,086 GFLOPs ở đầu vào $1280\times1280$ và 0,022 GFLOPs ở $640\times640$. Phần MLP vẫn nằm trong đường suy luận.
- Về tốc độ một phiên trên L4, GauCon đạt 73,95 FPS (13,52 ms/khung hình) và tiêu tốn thêm +19,48 MB VRAM cho bộ đệm trung gian 20 kênh tại $P_2$. Việc Base đo được 14,39 ms (69,51 FPS) chậm hơn GauCon 0,86 ms dù ít tầng hơn cho thấy phép đo 1 phiên (100 lượt) chịu nhiễu biến động xung nhịp/trạng thái GPU; kết luận tốc độ cần được kiểm định lại bằng nhiều phiên đo độc lập kèm độ lệch chuẩn (*mean ± std*).

### 3.8. Hạn chế và các yếu tố ảnh hưởng đến độ tin cậy của kết quả

Bảng 3.6 tổng hợp các giới hạn của thực nghiệm và hướng kiểm chứng tiếp theo.

Bảng 3.6: Tổng hợp các hạn chế thực nghiệm và hướng khắc phục trong nghiên cứu tiếp theo

| Hạng mục hạn chế | Tác động đến việc diễn giải kết quả | Hướng khắc phục cụ thể |
|---|---|---|
| Chưa có thí nghiệm tách bỏ từng cơ chế (*ablation*) | Chưa tách được đóng góp riêng của trọng tâm $W_2$ ($\beta_{\mathrm{box}} \approx 0{,}04$), FG-Gate, DW-TAL và chưng cất; cũng chưa cô lập nguồn gốc mức tăng +8,33 điểm ở nhóm *reasonable* (>48 px) | Chạy tối thiểu 3 cấu hình đối chứng tắt từng thành phần ($\beta_{\mathrm{box}} = 0$, tắt DW-TAL, tắt FG-Gate) dưới cùng điều kiện 80 epoch COCO |
| Mỗi cấu hình chỉ có một lần huấn luyện (seed = 42) | Bootstrap theo ảnh ước lượng độ bất định trên tập kiểm tra, không phản ánh biến thiên giữa các lần huấn luyện. Việc lấy mẫu từng ảnh cũng chưa mô hình hóa tương quan giữa các khung hình thuộc cùng video. Chênh lệch Full $AP_{50}$ so với NWD (+0,89 điểm, 95% CI [−0,12; +1,90]) chứa 0. | Lặp lại huấn luyện với nhiều seed; nếu có mã chuỗi video, bổ sung bootstrap phân cụm theo video. |
| Đánh đổi tại *tiny1*, cụm sát < 10 px và $E_{\mathrm{bkg}}$ Tiled | GauCon thua NWD ở *tiny1* (25,53% vs 27,91% Tiled; 18,34% vs 18,46% Full) lẫn Recall cụm < 10 px (38,06% vs 38,71% Full), đồng thời $E_{\mathrm{bkg}}$ Tiled tăng +28,7% so với Base | Điều chỉnh tham số $\eta, \tau_{\mathrm{px}}$ trên tập kiểm định và kết hợp theo dõi chuỗi thời gian cho ảnh *tiny1* |
| Thiếu đối chứng RFLA/BPD-YOLO cùng giao thức & nhiễu đo FPS | Chưa tái huấn luyện RFLA [16], BPD-YOLO [5] hay YOLOv8n-P2 ở cùng chế độ 80 epoch COCO; phép đo L4 1 phiên còn nhiễu làm Base chậm hơn GauCon | Bổ sung đối chứng cùng giao thức và đo lại độ trễ đa phiên (*mean ± std*) trên L4 lẫn Jetson Orin Nano |

---

## KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

Đồ án đã đề xuất và đánh giá GauCon cùng DW-TAL trên YOLO26n-P2 cho bài toán phát hiện người cực nhỏ trong bộ dữ liệu TinyPerson. Đối chiếu theo bốn mục tiêu đề ra ở Mục 1.4:

1. Mục tiêu 1 (Đường cơ sở và quy trình đánh giá chống rò rỉ): Đã xây dựng quy trình huấn luyện và chấm điểm thống nhất trên tập TinyPerson tách biệt hoàn toàn theo tiền tố chuỗi video (662 ảnh *train*, 84 ảnh *val*, 786 ảnh *test* với 13.687 nhãn người GT), đo đồng bộ ở cả giao thức lát cắt $640 \times 512$, toàn khung $1280 \times 1280$, COCO $AP_{50:95}$ và phân rã TIDE.
2. Mục tiêu 2 (Đầu phát hiện GauCon siêu nhẹ bảo toàn tốc độ suy luận): Chỉ áp dụng chọn lọc trên $P_2, P_3$ bằng cách lấy mẫu các ô lân cận theo chỉ số, sử dụng khoảng cách Wasserstein chuẩn hóa theo phương sai cục bộ (gần bất biến theo tỉ lệ), trọng tâm Wasserstein cục bộ và cổng FG-Gate không dùng chuẩn hóa theo batch, GauCon chỉ cộng thêm 714 tham số (+0,030% sau hợp nhất, tổng 2.402.374 tham số). Phép đo một phiên trên NVIDIA L4 ghi nhận 73,95 FPS (13,52 ms/khung hình); cần đo nhiều phiên độc lập trước khi kết luận về tốc độ tương đối.
3. Mục tiêu 3 (Gán nhãn tách rời DW-TAL và chưng cất không phát sinh chi phí suy luận): Cổng điều biến kích thước logistic ($\tau_{\mathrm{px}} = 20$ px) kết hợp giới hạn mục tiêu mềm theo $\mathrm{IoU}$ thực được sử dụng trong cấu hình thực nghiệm; trên chế độ toàn khung, GauCon giữ $E_{\mathrm{bkg}} = 8.683$ thấp hơn NWD (9.021) dù vẫn cao hơn Base (7.341), và trong chế độ lát cắt $E_{\mathrm{bkg}}$ tăng +28,7% so với Base (16.422 lên 21.130).
4. Mục tiêu 4 (Kết quả thực nghiệm đối chứng trong phạm vi đơn seed): Ở chế độ lát cắt $640 \times 512$, GauCon nâng $AP_{50}^{all}$ từ 45,58% lên 48,53% (+2,95 điểm, 95% CI [+1,92; +4,02]) và $AP_{75}^{all}$ lên 7,37% (vượt Base +0,97 điểm và NWD +1,28 điểm). Trên toàn khung $1280 \times 1280$, GauCon đạt 44,13% $AP_{50}^{all}$ (cao hơn cả Base 43,08%, Base + Soft-NMS 43,37% lẫn NWD 43,24%), giảm 22,2% hộp trùng (1.027 xuống 799), đồng thời ghi nhận rõ các hạn chế cần hoàn thiện về nhóm *tiny1* và hiện tượng tăng gián tiếp trên nhóm *reasonable*.
Hướng nghiên cứu tiếp theo gồm ba nội dung. Thứ nhất, thực hiện thí nghiệm ablation để đánh giá riêng bước hiệu chỉnh hộp ($\beta_{\mathrm{box}} = 0$), FG-Gate, DW-TAL và cơ chế chưng cất; đồng thời huấn luyện lặp với 3–5 seed độc lập để ước lượng biến thiên giữa các lần chạy. Thứ hai, tái huấn luyện RFLA và BPD-YOLO theo cùng giao thức 80 epoch với trọng số khởi tạo COCO, đồng thời đánh giá khả năng mở rộng sang bài toán đa lớp trên VisDrone. Thứ ba, đo độ trễ qua nhiều phiên độc lập và báo cáo giá trị trung bình cùng độ lệch chuẩn trước khi tối ưu bằng TensorRT FP16 hoặc INT8 trên NVIDIA Jetson Orin Nano.
Hướng phát triển tiếp theo tập trung vào ba trọng tâm khắc phục trực tiếp các giới hạn đã chỉ ra: (1) bổ sung ngay bộ thí nghiệm tách bỏ từng thành phần (bao gồm cấu hình tắt dịch trọng tâm $\beta_{\mathrm{box}} = 0$) và chạy lặp 3–5 lần khởi tạo ngẫu nhiên (random seed) độc lập để xác định đóng góp chính xác từng cơ chế; (2) bổ sung đối chứng RFLA/BPD-YOLO cùng giao thức 80 epoch COCO và kiểm chứng đa lớp trên VisDrone; và (3) đo tốc độ suy luận lặp nhiều phiên (*mean ± std*) trước khi triển khai TensorRT INT8/FP16 trên bo mạch NVIDIA Jetson Orin Nano.

---

## TÀI LIỆU THAM KHẢO

[1] X. Yu, Y. Gong, N. Jiang, Q. Ye, and Z. Han, "Scale Match for Tiny Person Detection," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2020, pp. 1257–1265.

[2] N. Jiang, X. Yu, X. Peng, Y. Gong, and Z. Han, "SM+: Refined Scale Match for Tiny Person Detection," in *Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2021, pp. 1815–1819.

[3] Y. Gong, X. Yu, Y. Ding, X. Peng, J. Zhao, and Z. Han, "Effective Fusion Factor in FPN for Tiny Object Detection," in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV)*, 2021, pp. 1160–1168.

[4] M. Hong, S. Li, Y. Yang, F. Zhu, Q. Zhao, and L. Lu, "SSPNet: Scale Selection Pyramid Network for Tiny Person Detection From UAV Images," *IEEE Geoscience and Remote Sensing Letters*, vol. 19, pp. 1–5, 2022.

[5] M.-X. Chao, C. Peng, L.-J. Yun, C. Zhang, H.-H. Wang, and Z.-Q. Chen, "A Lightweight Small Object Detection Model for UAV Images Based on Deep Semantic Integration," *Scientific Reports*, vol. 15, art. 31888, 2025, doi: 10.1038/s41598-025-16878-6.

[6] J. Wang, W. Yang, H. Guo, R. Zhang, and G.-S. Xia, "Tiny Object Detection in Aerial Images," in *Proc. 25th Int. Conf. Pattern Recognition (ICPR)*, 2021, pp. 3791–3798.

[7] P. Zhu, L. Wen, D. Du, X. Bian, H. Fan, Q. Hu, and H. Ling, "Detection and Tracking Meet Drones Challenge," *IEEE Trans. Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 44, no. 11, pp. 7380–7399, 2022.

[8] A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, and G. Ding, "YOLOv10: Real-Time End-to-End Object Detection," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 37, 2024, pp. 107984–108011.

[9] Ultralytics, "Ultralytics YOLO26: End-to-End Object Detection Architecture," 2025. [Online]. Available: https://docs.ultralytics.com/models/yolo26/ (Accessed: Nov. 2026).

[10] P. Sun, Y. Jiang, E. Xie, W. Shao, Z. Yuan, C. Wang, and P. Luo, "What Makes for End-to-End Object Detection?," in *Proc. 38th Int. Conf. Machine Learning (ICML)*, 2021, pp. 9934–9944.

[11] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, "DETRs Beat YOLOs on Real-Time Object Detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 16965–16974.

[12] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-End Object Detection with Transformers," in *Proc. European Conf. Computer Vision (ECCV)*, 2020, pp. 213–229.

[13] J. Wang, C. Xu, W. Yang, and L. Yu, "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection," *arXiv preprint arXiv:2110.13389*, 2021.

[14] C. Xu, J. Wang, W. Yang, H. Yu, L. Yu, and G.-S. Xia, "Detecting Tiny Objects in Aerial Images: A Normalized Wasserstein Distance and a New Benchmark," *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 190, pp. 79–93, 2022, doi: 10.1016/j.isprsjprs.2022.06.002.

[15] X. Yang, J. Yan, Q. Ming, W. Wang, X. Zhang, and Q. Tian, "Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss," in *Proc. 38th Int. Conf. Machine Learning (ICML)*, 2021, pp. 11830–11841.

[16] C. Xu, J. Wang, W. Yang, and L. Yu, "RFLA: Gaussian Receptive Field Based Label Assignment for Tiny Object Detection," in *Proc. European Conf. Computer Vision (ECCV)*, 2022, pp. 526–543.

[17] C. Feng, Y. Zhong, Y. Gao, M. R. Scott, and W. Huang, "TOOD: Task-Aligned One-Stage Object Detection," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 3510–3519.

[18] M. Agueh and G. Carlier, "Barycenters in the Wasserstein Space," *SIAM Journal on Mathematical Analysis*, vol. 43, no. 2, pp. 904–924, 2011.

[19] C. Villani, *Optimal Transport: Old and New*, vol. 338 of *Grundlehren der mathematischen Wissenschaften*. Berlin: Springer, 2009.

[20] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO: Common Objects in Context," in *Proc. European Conf. Computer Vision (ECCV)*, 2014, pp. 740–755.

[21] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 28, 2015, pp. 91–99.

[22] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature Pyramid Networks for Object Detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 2117–2125.

[23] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2980–2988.

[24] S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia, "Path Aggregation Network for Instance Segmentation," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 8759–8768.

[25] Z. Tian, C. Shen, H. Chen, and T. He, "FCOS: Fully Convolutional One-Stage Object Detection," in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2019, pp. 9627–9636.

[26] Z. Cai and N. Vasconcelos, "Cascade R-CNN: High Quality Object Detection and Instance Segmentation," *IEEE Trans. Pattern Analysis and Machine Intelligence (TPAMI)*, vol. 43, no. 5, pp. 1483–1498, 2021.

[27] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression," in *Proc. AAAI Conf. Artificial Intelligence*, vol. 34, no. 7, 2020, pp. 12993–13000.

[28] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, J. Li, J. Tang, and J. Yang, "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 21002–21012.

[29] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.

[30] A. Neubeck and L. Van Gool, "Efficient Non-Maximum Suppression," in *Proc. 18th Int. Conf. Pattern Recognition (ICPR)*, vol. 3, 2006, pp. 850–855.

[31] N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, "Soft-NMS — Improving Object Detection with One Line of Code," in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 5561–5569.

[32] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, 2015, pp. 448–456.

[33] G. Jocher and J. Qiu, "Ultralytics YOLO11," 2024. [Online]. Available: https://docs.ultralytics.com/models/yolo11/ (Accessed: Nov. 2026).

[34] R. Solovyev, W. Wang, and T. Gabruseva, "Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models," *Image and Vision Computing*, vol. 107, art. 104117, 2021, doi: 10.1016/j.imavis.2021.104117.

[35] J. Hosang, R. Benenson, and B. Schiele, "Learning Non-Maximum Suppression," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 4507–4515.

[36] H. Zhang, Y. Wang, F. Dayoub, and N. Sünderhauf, "VarifocalNet: An IoU-aware Dense Object Detector," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8514–8523.

[37] D. Bolya, S. Foley, J. Hays, and J. Hoffman, "TIDE: A General Toolbox for Identifying Object Detection Errors," in *Proc. European Conf. Computer Vision (ECCV)*, 2020, pp. 558–573.

[38] F. C. Akyon, S. O. Altinuc, and A. Temizel, "Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection," in *Proc. IEEE Int. Conf. Image Processing (ICIP)*, 2022, pp. 966–970.

---

## PHỤ LỤC

### Phụ lục A. Đặc tả cấu trúc kiến trúc mạng đa tầng YOLO26n-P2 + GauCon

Bảng 3.7 tóm tắt đặc tả từng tầng trong mô hình bốn mức $P_2\text{–}P_5$ ở kích thước đầu vào $1280 \times 1280$ (hệ số mở rộng độ sâu $0{,}50$ và độ rộng $0{,}25$), bao gồm cấu trúc nhánh suy luận $\mathcal{H}_{1:1}$ tích hợp GauCon chọn lọc và nhánh phụ trợ $\mathcal{H}_{1:M}$ chỉ dùng khi huấn luyện.

Bảng 3.7: Đặc tả các tầng kiến trúc mạng YOLO26n-P2 tích hợp GauCon ở độ phân giải đầu vào $1280 \times 1280$

| Khối chức năng | Tầng | Phép toán chính | Kích thước bản đồ đặc trưng | Bước lưới | Vai trò trong kiến trúc |
|---|---|---|---|---|---|
| Backbone | $C_2$ | CSPDarknet giai đoạn nông | $320 \times 320 \times 64$ | 4 | Bảo toàn chi tiết người cực nhỏ 2–8 px |
| Backbone | $C_3$ | CSPDarknet giai đoạn 2 | $160 \times 160 \times 128$ | 8 | Đặc trưng trung gian cho vật 8–20 px |
| Backbone | $C_4$ | CSPDarknet giai đoạn 3 | $80 \times 80 \times 256$ | 16 | Đặc trưng ngữ cảnh đối tượng vừa |
| Backbone | $C_5$ | CSPDarknet + SPPF + C2PSA | $40 \times 40 \times 256$ | 32 | Ngữ cảnh toàn cục và tự chú ý không gian |
| Neck (Top-Down) | $N_4, N_3$ | $\mathrm{Upsample}_{2\times} + \mathrm{Concat} + \mathrm{C3k2}$ | $80^2 \times 128$, $160^2 \times 64$ | 16, 8 | Truyền ngữ nghĩa từ sâu xuống nông |
| Neck (Bottom-Up) | $P_2$ | $\mathrm{Upsample}_{2\times} + \mathrm{Concat}(C_2) + \mathrm{C3k2}$ | $320 \times 320 \times 32$ | 4 | Bản đồ đặc trưng đầu ra độ phân giải cao |
| Neck (Bottom-Up) | $P_3$ | $\mathrm{Conv}_{3\times 3, s=2} + \mathrm{Concat}(N_3) + \mathrm{C3k2}$ | $160 \times 160 \times 64$ | 8 | Bản đồ đặc trưng đầu ra dải nhỏ–vừa |
| Neck (Bottom-Up) | $P_4, P_5$ | $\mathrm{Conv}_{3\times 3, s=2} + \mathrm{Concat} + \mathrm{C3k2}$ | $80^2 \times 128$, $40^2 \times 256$ | 16, 32 | Bản đồ đầu ra vật >28 px và ô cha $P_3$ |
| Head ($\mathcal{H}_{1:1}$) | $P_2, P_3$ | $\Phi_{\mathrm{box}}^{(s)}, \Phi_{\mathrm{cls}}^{(s)}$ + GauCon | $320^2, 160^2$ | 4, 8 | Trọng tâm $W_2$ + FG-Gate (+357 tham số/mức) |
| Head ($\mathcal{H}_{1:1}$) | $P_4, P_5$ | $\Phi_{\mathrm{box}}^{(s)}, \Phi_{\mathrm{cls}}^{(s)}$ + Đồng nhất | $80^2, 40^2$ | 16, 32 | Giữ nguyên dự đoán gốc, tổng 136.000 ô |
| Huấn luyện ($\mathcal{H}_{1:M}$) | $P_2\text{–}P_5$ | Nhánh đa mẫu + DW-TAL + $\mathcal{L}_{\mathrm{distill}}$ | Cả 4 mức | 4–32 | Chưng cất xuống $P_2, P_3$; gỡ bỏ khi suy luận |

### Phụ lục B. Bảng đối chiếu đầy đủ ba ngưỡng IoU ($AP_{25}, AP_{50}, AP_{75}$) trên tập kiểm tra

Bảng 3.8: Chi tiết $AP_{25}, AP_{50}, AP_{75}$ theo từng thang đo ở chế độ lát cắt $640 \times 512$ (%)

| Thang đo | Base $AP_{25}$ | NWD $AP_{25}$ | GauCon $AP_{25}$ | Base $AP_{50}$ | NWD $AP_{50}$ | GauCon $AP_{50}$ | Base $AP_{75}$ | NWD $AP_{75}$ | GauCon $AP_{75}$ |
|---|---|---|---|---|---|---|---|---|---|
| *all* | 70,02 | 71,93 | 71,56 | 45,58 | 48,45 | 48,53 | 6,40 | 6,09 | 7,37 |
| *tiny* | 66,45 | 68,84 | 67,28 | 42,59 | 45,31 | 43,99 | 3,83 | 4,26 | 4,33 |
| *tiny1* | 45,57 | 50,31 | 46,72 | 25,10 | 27,91 | 25,53 | 1,38 | 1,99 | 1,27 |
| *tiny2* | 70,05 | 73,18 | 71,62 | 45,14 | 46,73 | 46,97 | 3,13 | 3,42 | 4,45 |
| *tiny3* | 76,35 | 77,48 | 77,47 | 50,94 | 54,82 | 52,95 | 5,58 | 6,46 | 5,82 |
| *small* | 78,95 | 80,61 | 79,98 | 54,55 | 56,83 | 56,07 | 9,87 | 10,53 | 11,79 |
| *reasonable* | 77,26 | 80,12 | 81,80 | 49,78 | 53,51 | 58,11 | 10,51 | 11,02 | 13,55 |

Bảng 3.9: Chi tiết $AP_{25}, AP_{50}, AP_{75}$ theo từng thang đo ở chế độ suy luận toàn khung $1280 \times 1280$ (%)

| Thang đo | Base $AP_{25}$ | NWD $AP_{25}$ | GauCon $AP_{25}$ | Base $AP_{50}$ | NWD $AP_{50}$ | GauCon $AP_{50}$ | Base $AP_{75}$ | NWD $AP_{75}$ | GauCon $AP_{75}$ |
|---|---|---|---|---|---|---|---|---|---|
| *all* | 69,29 | 71,44 | 71,19 | 43,08 | 43,24 | 44,13 | 4,44 | 4,13 | 4,78 |
| *tiny* | 63,68 | 65,63 | 65,45 | 36,28 | 35,74 | 36,06 | 2,32 | 2,26 | 2,86 |
| *tiny1* | 45,19 | 46,79 | 46,41 | 18,71 | 18,46 | 18,34 | 0,53 | 0,48 | 0,45 |
| *tiny2* | 68,18 | 69,82 | 70,36 | 37,48 | 36,53 | 37,90 | 2,74 | 1,81 | 2,41 |
| *tiny3* | 73,29 | 75,03 | 74,60 | 46,23 | 45,94 | 45,79 | 3,46 | 3,75 | 4,17 |
| *small* | 77,69 | 79,98 | 79,68 | 53,07 | 52,90 | 54,32 | 6,56 | 6,36 | 6,29 |
| *reasonable* | 83,42 | 85,76 | 85,27 | 58,77 | 62,48 | 63,09 | 9,29 | 10,51 | 11,28 |
