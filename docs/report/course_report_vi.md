# BÁO CÁO ĐỒ ÁN MÔN HỌC

## THỊ GIÁC MÁY TÍNH

**Đề tài:** GauSS: Module không gian tỉ lệ Gauss cho phát hiện người cực nhỏ trên ảnh hàng hải độ phân giải cao

**Giảng viên hướng dẫn:** TS. Huỳnh Hữu Hùng

**Nhóm sinh viên thực hiện:**

- Trần Đức Long
- Phan Thị Thảo Tiên

**Đơn vị:** Trường Đại học Bách khoa, Đại học Đà Nẵng

**Năm học:** 2025–2026

---

## MỤC LỤC

## TÓM TẮT

Phát hiện người cực nhỏ trên ảnh hàng hải độ phân giải cao phục vụ bài toán tìm kiếm cứu nạn trên biển: một người ở khoảng cách vài trăm mét chỉ chiếm 2–8 pixel trong khung hình 1280 × 1280, tương đồng về độ sáng và kết cấu với đỉnh sóng. Chúng tôi đề xuất module GauSS với 36.878 tham số (cấu hình tinh gọn 36.361 tham số hiệu dụng, tăng 1,12%) cắm vào tầng ghép stride 4 của detector YOLO26n-P2. Module phân rã đặc trưng thành ba dải tần bằng hai bộ lọc Gauss (độ lệch chuẩn $\sigma$ cố định quanh giá trị thiết kế), tinh chỉnh từng dải ở stride 8, định tuyến theo vị trí dựa trên năng lượng log-scale và tái tạo về stride 4 bằng tàn dư có chặn.

Một đặc điểm phương pháp luận quan trọng là sự lệch thang giữa huấn luyện và suy luận: mô hình được huấn luyện trên ảnh cả khung bị thu nhỏ khoảng ba lần, còn đánh giá chính thực hiện trên các mảnh cắt 640 × 512 ở độ phân giải gốc nguyên bản (không co). Ở chế độ cắt mảnh này, AP50 toàn thang tăng từ 40,63 lên 41,08 (+0,45). Mức tăng tập trung ở thang 8–12 pixel (+0,75), 20–32 pixel (+0,88) và trên 32 pixel (+3,45). Thang 2–8 pixel tăng 3,99 điểm AP25 nhưng giảm 0,03 điểm AP50; thang 12–20 pixel giảm mạnh nhất (−1,09 điểm AP50, miss rate tăng 0,58 điểm). Ngược lại, ở chế độ suy luận cả khung 1280 (khớp thang huấn luyện), AP50 chỉ đạt 35,45 (so với nền 35,32). Trong cấu hình thực nghiệm chứa nhánh C3k2 trung gian chưa lược bỏ, module làm độ trễ mạng tăng thêm 34,7%. Phân tích thành phần với kiểm định bootstrap cho thấy việc loại bỏ nhánh dải giữa và router làm giảm AP50, tuy nhiên phương sai giữa các lần huấn luyện độc lập là đáng kể (đạt hơn 4 điểm AP50 khi kiểm tra nhánh saliency). Do mỗi cấu hình chỉ chạy trên một seed (42), mức chênh lệch tổng thể 0,45 điểm AP50 là một cải thiện khiêm tốn, gắn chặt với pipeline cắt mảnh ở độ phân giải gốc thay vì phản ánh ưu thế vượt trội ở mọi thang đo.

**Abstract.** Detecting tiny persons in high-resolution maritime aerial imagery is critical for maritime search and rescue: a person several hundred meters away covers only 2–8 pixels in a 1280 × 1280 frame, closely resembling wave crests. We propose GauSS, a module with 36.8k parameters (28.2k effective parameters in the streamlined configuration, +1.12%), integrated at the stride-4 fusion stage of YOLO26n-P2. GauSS decomposes features into three spatial-frequency bands using two Gaussian filters whose scale parameters stay fixed at design values, refines bands at stride 8, routes them by position using log-scale energy, and recombines them through bounded residual mapping.

A key methodological aspect is the scale mismatch between training and inference: the network trains on full frames downscaled roughly threefold, yet primary evaluation runs on unscaled 640 × 512 crops at native resolution. Under this native tiled regime, all-scale AP50 increases from 40.63 to 41.08 (+0.45), with targeted gains at 8–12 px (+0.75) and above 32 px (+3.45), while the 12–20 px bin shows the steepest drop (-1.09 AP50, +0.58 MR50). Under full-frame 1280 inference matching the training scale, AP50 reaches only 35.45 (baseline 35.32). In the experimental configuration retaining an unpruned intermediate C3k2 block, neural latency increases by 34.7%. Ablation tests with bootstrap validation show that removing the mid band and hybrid router reduces AP50, though run-to-run training variance remains substantial (over 4 points when probing the inactive saliency gate branch). Evaluated on a single seed (42), the overall 0.45 AP50 gain is modest and strongly tied to the full-resolution tiling pipeline rather than consistent superiority across all scales.

**Từ khóa:** phát hiện người cực nhỏ; scale-space; phân rã dải tần; TinyPerson; YOLO26n



## 1. TỔNG QUAN NGHIÊN CỨU LIÊN QUAN

### 1.1. Bài toán phát hiện người cực nhỏ

Một người rơi xuống biển ở khoảng cách vài trăm mét chỉ chiếm 2–8 pixel trong khung hình 1280 × 1280 của camera trên máy bay không người lái (UAV). Ở cỡ đó, người và đỉnh sóng có cùng độ sáng, cùng kết cấu và cùng kích thước trên ảnh. Detector huấn luyện trên COCO [25] gần như không thấy gì: sau năm lần downsample, tổng hệ số là 32, nên một vật 5 pixel chỉ còn khoảng 0,16 ô ở mức sâu nhất. Một ô không đủ để mạng phân biệt người với nhiễu cảm biến.

![](docs/report/figures/r_gt_distribution.png)

**Hình 1.1: Phân bố *ground truth* theo cạnh vật trong tập test TinyPerson**

Hình 1.1 cho thấy phân bố kích thước của tập test [1]: 65% ground truth có cạnh dưới 20 pixel, trong đó 17% nằm dưới 8 pixel. Phần lớn khối lượng chú thích nằm ở dải mà detector thông thường xử lý kém nhất, nên một cải tiến hiệu quả ngay ở dải này có ý nghĩa thực tế lớn hơn một cải tiến nhỏ rải đều trên toàn thang.

Bài toán có ứng dụng trực tiếp trong tìm kiếm cứu nạn trên biển. Ảnh từ UAV có độ phân giải gốc rất cao, tập TinyPerson [1] gồm ảnh đến 3840 × 2160, nên người trực quan sát bằng mắt gần như không quét hết. Mỗi khung hình phải xử lý trong ngân sách thời gian của thiết bị bay, nên các mô hình vài chục triệu tham số không dùng được dù điểm benchmark cao hơn.

Đề tài được chọn vì bài toán có ứng dụng cứu hộ rõ ràng, không phải một benchmark để tối ưu điểm số. Bộ dữ liệu cũng phù hợp với cách đồ án đánh giá: TinyPerson chia thang đo sẵn theo căn bậc hai diện tích, nên mỗi cải tiến quy được về đúng thang nào được lợi thay vì chỉ nhìn một con số tổng.

### 1.2. Các hướng tiếp cận đã có

#### 1.2.1. Detector hai giai đoạn với backbone huấn luyện trước

Hướng phổ biến nhất là dùng detector hai giai đoạn có *backbone* huấn luyện trước trên ImageNet. Trên TinyPerson, Faster R-CNN [26] đạt 43,55 mAP50 toàn tập, Cascade R-CNN [27] đạt 57,19 và Faster R-CNN kết hợp SSPNet [2] đạt 59,13. Ở đây mAP50 là độ chính xác trung bình (mean Average Precision) tại ngưỡng IoU 0,50 (IoU: tỷ lệ phần trùng giữa khung dự đoán và ground truth). Các mô hình này có từ 32 đến hơn 60 triệu tham số và phụ thuộc trọng số ImageNet.

Ưu điểm là độ chính xác cao và ổn định, nhờ hai lần tinh chỉnh khung và ngân sách tính toán lớn. Nhược điểm nằm ở chi phí: 32–60+ triệu tham số, hai lượt lan truyền, khó đạt thời gian thực trên thiết bị bay. Điểm cao của chúng đến từ ngân sách tính toán nhiều hơn là từ cách xử lý vật cỡ vài pixel.

#### 1.2.2. Thêm mức stride 4 vào detector một giai đoạn

Một cách khác là thêm mức *stride* 4 (P2) vào detector một giai đoạn. Trên TinyPerson, YOLOv8n+P2 đạt 38,0 mAP50 và BPD-YOLO [3] đạt 39,1, theo Bảng 5 của [3]; cả hai dưới 3 triệu tham số và huấn luyện từ đầu.

Ưu điểm là chi phí thấp và giữ được thời gian thực. P2 lấy lại độ phân giải không gian mà phép downsample đã lấy đi. Nhược điểm nằm ở ngữ nghĩa: đặc trưng nông có độ phân giải cao nhưng chưa đủ sâu để phân biệt người với mặt nước lấp lánh, nên nối thẳng C2 vào *head* thường làm tăng báo nhầm trên đỉnh sóng.

#### 1.2.3. Module attention trên neck

Ở mức *neck*, một nhóm công trình gắn module attention để chọn kênh hoặc chọn vùng quan trọng. Ưu điểm là giảm nhiễu nền với chi phí tham số nhỏ. Nhược điểm là không mô hình hóa nội dung tần số của đặc trưng. Biên một người 8 pixel và một gradient nước biến thiên chậm khác nhau chủ yếu ở thang không gian, và attention trên kênh không tách được hai thứ đó.

SEPC [4] đi theo một cách khác: tác giả coi trục tỉ lệ là một chiều không gian riêng và dùng tích chập ba chiều trượt qua cả trục không gian lẫn trục tỉ lệ để nắm tương quan liên mức, với lập luận rằng các mức của pyramid lý tưởng phải thỏa mãn tính chất của pyramid Gauss. Vì pyramid đặc trưng của mạng tích chập không thỏa mãn chính xác tính chất đó, SEPC thêm bước căn bằng tỉ lệ: lấy mẫu lại đặc trưng theo từng mức trước khi chia sẻ nhân, và chuẩn hóa batch trên toàn bộ tensor pyramid thay vì từng mức riêng. Đây là công trình gần với động cơ của GauSS nhất ở mức ý tưởng: cả hai đều xuất phát từ yêu cầu scale-space. Khác biệt nằm ở chỗ SEPC tái cấu trúc tương quan giữa các mức, còn GauSS tách ngay bên trong một mức thành các dải tần số.

#### 1.2.4. Hàm mất mát và tiêu chí ghép cho vật nhỏ

Thay vì sửa kiến trúc, một nhóm công trình sửa cách đo và cách ghép. Ba công trình tiêu biểu:

- **NWD [5]** mô hình hóa mỗi khung như một phân phối Gauss hai chiều rồi đo khoảng cách Wasserstein chuẩn hóa giữa hai phân phối. Với khung 4–8 pixel, IoU gần như biến mất khi lệch một pixel, còn khoảng cách Wasserstein vẫn cho gradient đều.
- **Dot Distance [6]** thay IoU bằng khoảng cách tâm chuẩn hóa theo kích thước vật, nên hai khung nhỏ lệch nửa pixel không bị phạt như hai khung lớn lệch nửa pixel.
- **RFLA [7]** gán nhãn theo trường tiếp nhận hiệu dụng: mỗi ground truth được ghép với các ô có trường tiếp nhận Gauss phủ lên nó, thay vì theo ngưỡng IoU.

Ưu điểm chung là không thêm tham số, bổ trợ được cho mọi kiến trúc và xử lý đúng vấn đề thang đo ở mức tín hiệu huấn luyện. Nhược điểm chung là tín hiệu đã mất sau stride 8 thì không tiêu chí ghép nào lấy lại được: nếu biên vật không còn nằm trên lưới đặc trưng, không có cách gán nhãn nào tạo lại nó. Đồ án vì thế xem hướng này là phần bổ trợ chứ không phải lời giải.

#### 1.2.5. Tăng cường đặc trưng trong miền tần số

Một hướng khác can thiệp vào phổ của đặc trưng. SET [8] phân tích sự suy giảm đặc trưng của vật nhỏ từ góc nhìn tần số và gồm hai module dùng trong lúc huấn luyện: làm mượt nền theo thứ bậc (HBS) để giảm kết cấu nhiễu tần số cao ở vùng nền, và chèn nhiễu đối kháng (API) để tăng độ nổi bật của vật nhỏ trong đặc trưng. Vì cả hai chỉ tác động khi huấn luyện, SET không thêm chi phí suy luận.

Điểm chung với GauSS là cùng xuất phát từ chỗ vật cực nhỏ và nhiễu nền khác nhau ở phân bố tần số chứ không ở ngữ nghĩa. Khác biệt nằm ở vai trò trong hệ thống: SET là một chiến lược huấn luyện, còn GauSS là một khối kiến trúc chạy cả khi suy luận, phân rã đặc trưng stride 4 thành ba dải bằng $\sigma$ tham số hóa cho từng kênh rồi định tuyến theo từng vị trí. Hai cách không loại trừ nhau; đồ án không chạy SET nên không có so sánh trực tiếp, và nhận định trên là suy luận từ mô tả trong [8] chứ chưa phải kết quả đo đạc đối chứng.

#### 1.2.6. Cắt mảnh trước khi suy luận

Cắt ảnh thành mảnh nhỏ trước khi suy luận là cách xử lý ở mức giao thức, tiêu biểu là SAHI [9]. Ưu điểm là tăng recall cho vật nhỏ mà không cần huấn luyện lại. Nhược điểm là chi phí suy luận tỉ lệ với số mảnh: 23,4 mảnh cho một ảnh trung bình của TinyPerson, và kết quả phụ thuộc vào bước ghép kết quả bằng Non-Maximum Suppression (NMS) trên toàn ảnh.

#### 1.2.7. Chọn vùng quan tâm trên đặc trưng độ phân giải cao

QueryDet [10] đi theo hướng khác: dùng đặc trưng mức sâu để dự đoán vị trí có khả năng chứa vật nhỏ, rồi chỉ chạy head ở mức stride 4 trên những vị trí đó. Cách làm này giữ nguyên độ phân giải của mức nông mà giảm mạnh số ô phải xử lý.

Ưu điểm là tăng tốc suy luận rõ rệt và không cần huấn luyện lại từ đầu. Nhược điểm nằm ở chỗ phụ thuộc: nếu mức sâu bỏ sót vật thì mức nông không bao giờ được xét, và bản thân đặc trưng nông vẫn không được xử lý gì thêm: nó chỉ được chọn hoặc không được chọn.

### 1.3. Khoảng trống nghiên cứu

Bảng 1.1 tóm tắt bảy hướng trên theo ba tiêu chí mà bài toán cứu hộ đặt ra: chi phí tham số, khả năng chạy thời gian thực, và mức can thiệp vào tầng đặc trưng nơi vật nhỏ vẫn giữ được tín hiệu.

**Bảng 1.1: So sánh các hướng tiếp cận theo tiêu chí triển khai**

| Hướng tiếp cận | Tham số | Thời gian thực | Can thiệp tầng stride 4 |
|---|---|---|---|
| Hai giai đoạn, huấn luyện trước | 32–60+ M | Không | Không |
| Một giai đoạn + P2 | 1,5–3 M | Có | Chỉ nối thêm mức |
| *Attention* trên *neck* | thấp | Có | Không |
| Hàm mất mát / tiêu chí ghép | 0 | Có | Không |
| Cắt mảnh khi suy luận | 0 | Không (23,4 mảnh) | Không |
| Tăng cường miền tần số (SET) | 0 khi suy luận | Có | Chỉ trong huấn luyện |
| Chọn vùng quan tâm (QueryDet) | thấp | Có | Chỉ chọn vị trí |

**Bảng 1.2: Kết quả công bố trên TinyPerson test (mAP50 toàn tập), chia theo nhóm giao thức**

| Mô hình | Huấn luyện trước | Tham số | AP50 toàn thang |
|---|---|---|---|
| **(a) Bảng gốc WACV 2020 [1]** | | | |
| Faster R-CNN [26] | ImageNet | ~41 M | 43,55 |
| RetinaNet [28] | ImageNet | ~34 M | 30,82 |
| FCOS [29] | ImageNet | ~32 M | 16,90 |
| **(b) Đánh giá lại bởi Hong và cộng sự [2]** | | | |
| FCOS [29] | ImageNet | – | 17,90 |
| RetinaNet [28] | ImageNet | – | 33,53 |
| Faster R-CNN-FPN [26] | ImageNet | ~41 M | 47,35 |
| Cascade R-CNN [27] | ImageNet | >60 M | 57,19 |
| Faster R-CNN + SSPNet [2] | ImageNet | >41 M | 59,13 |
| **(c) Nano-scale, huấn luyện từ đầu [3]** | | | |
| YOLOv8n+P2 | Từ đầu | 2,93 M | 38,00 |
| BPD-YOLO | Từ đầu | 1,50 M | 39,10 |

Bảng 1.2 đặt ba nhóm cạnh nhau. Nhóm (a) và (b) đạt 43–59 điểm nhưng cần 32–60+ triệu tham số và trọng số ImageNet. Nhóm (c) nằm dưới 3 triệu tham số và đạt 38–39 điểm, đúng vùng phù hợp với một module nhẹ. Khoảng cách giữa hai nhóm có thể giải thích phần lớn bằng ngân sách tính toán, trọng số huấn luyện trước và giao thức đánh giá; tách riêng từng yếu tố đòi hỏi thí nghiệm đối chứng mà các công trình trên không thực hiện.

So với SET [8], khác biệt không nằm ở việc có dùng miền tần số hay không (cả hai đều dùng), mà ở vai trò trong hệ thống: SET tác động khi huấn luyện, GauSS là khối kiến trúc chạy cả khi suy luận. Khác biệt còn lại nằm ở mức chi tiết: phân rã theo thang không gian với tham số thiết kế cho từng kênh, quyết định mức tăng cường theo từng vị trí thay vì một hệ số chung.

Khoảng trống mà đồ án nhắm tới, dựa trên việc tra cứu các hướng trên trong các bài báo đã nêu, là chưa thấy module nhẹ nào can thiệp trực tiếp vào tầng ghép stride 4. Đây là nơi thông tin của vật 2–12 pixel còn giữ được, cũng là nơi nhiễu mặt nước mạnh nhất. Một module đặt ở đó cần tách vật khỏi nền theo thang không gian, không phải thêm dung lượng cho mạng.

### 1.4. Mục tiêu và phạm vi

Đồ án đặt bốn mục tiêu.

1. Xây dựng detector YOLO26n-P2 huấn luyện từ đầu trên TinyPerson, tái lập được, chia train/test tách biệt theo chuỗi video.
2. Thiết kế module GauSS với chi phí tham số dưới 2% và kiểm tra tác động của nó trên các thang đo nhỏ.
3. Đánh giá theo giao thức chuẩn của TinyPerson, báo cáo theo từng thang đo và theo ngưỡng IoU.
4. Phân tích thành phần và đo độ trễ, bộ nhớ để biết module còn nằm trong ngân sách thời gian thực hay không.

Phạm vi đánh giá chỉ gồm TinyPerson, mỗi cấu hình chạy một *seed* (42); vì chỉ có một lần chạy, báo cáo không kèm khoảng tin cậy và các chênh lệch nhỏ được đọc theo xu hướng chung. Kết quả đo bằng evaluator tự viết theo mô tả giao thức trong bài báo gốc, không chạy mã TinyBenchmark chính thức, nên các giá trị tuyệt đối nên đọc là "đúng giao thức" chứ không phải "trùng khớp từng số với bảng công bố". Phần cứng huấn luyện và đo là GPU NVIDIA A10G; máy cá nhân chỉ chạy kiểm thử đơn vị trên CPU.

### 1.5. Bố cục báo cáo

Chương 1 trình bày bài toán, bảy hướng tiếp cận đã có kèm ưu nhược điểm và khoảng trống nghiên cứu. Chương 2 trình bày giải pháp: cơ sở lý thuyết cần dùng, kiến trúc module GauSS, công thức từng khối và cấu hình huấn luyện. Chương 3 trình bày đánh giá: giao thức, kết quả theo thang đo, phân tích thành phần, hiệu năng và hạn chế. Sản phẩm kèm theo báo cáo gồm mã nguồn module GauSS và evaluator, trọng số của hai cấu hình đã huấn luyện, cùng các bảng và hình kết quả đánh giá ở chương 3.

## 2. GIẢI PHÁP ĐỀ XUẤT

### 2.1. Cơ sở lý thuyết

#### 2.1.1. Pyramid đặc trưng và giới hạn của stride

Detector một giai đoạn dự đoán trực tiếp khung và nhãn trên lưới đặc trưng ở nhiều stride. FPN [11] (Feature Pyramid Network) ghép các mức đặc trưng từ sâu lên nông để mỗi mức phụ trách một khoảng kích thước. PANet [12] thêm một đường từ nông xuống sâu, nhờ đó thông tin vị trí chính xác của mức nông quay lại được các mức sâu.

Stride là tỉ lệ giữa kích thước ảnh và kích thước lưới đặc trưng. Với vật có cạnh $s$ pixel, số ô mà vật phủ lên một mức stride $r$ là $s/r$. Bảng 2.1 tính cho vật 8 pixel, tức cận trên của thang tiny1.

**Bảng 2.1: Kích thước đặc trưng theo stride với vật 8 pixel ở đầu vào mạng (chế độ không co, xem mục 3.1)**

| Mức | Stride | Kích thước lưới | Số ô (vật 8 px) | Vai trò |
|---|---|---|---|---|
| P2 | 4 | 320 × 320 | 2,0 | Vật 4–16 px, nhận thêm phần dưới vì là mức nông nhất; nơi GauSS can thiệp |
| P3 | 8 | 160 × 160 | 1,0 | Vật 8–32 px |
| P4 | 16 | 80 × 80 | 0,5 | Vật 16–64 px |
| P5 | 32 | 40 × 40 | 0,25 | Vật lớn, ngữ cảnh toàn ảnh |

Cột vai trò là quy ước của đồ án: mỗi mức nhận vật có cạnh từ khoảng $r$ đến $4r$, riêng P2 nhận thêm phần dưới vì là mức nông nhất. Bộ dữ liệu không dùng neo (anchor) nên khoảng này chỉ để đọc số ô, không phải luật gán nhãn.

Vật 8 pixel còn 2 ô ở stride 4 và đúng 1 ô ở stride 8. Từ stride 16 trở đi, vật nhỏ hơn một ô: tâm vật vẫn có thể rơi vào một ô và được gán nhãn, nhưng biên không còn nằm trên lưới, nên tín hiệu học chủ yếu là mức trung bình của cả ô lẫn nền xung quanh. Thêm mức P2 cũng có giá: lưới stride 4 có 16 lần số ô của stride 16, và đặc trưng nông chưa qua đủ tầng tích chập để trở nên chọn lọc, nên mọi kết cấu tần số cao đều đi vào head với biên độ tương đương vật thật. Đây chính là khoảng trống mà GauSS nhắm vào.

Bảng 2.1 tính theo pixel của đầu vào mạng. Ảnh TinyPerson rộng 3.480–3.840 pixel bị *letterbox* về 1280 khi huấn luyện và khi suy luận cả khung, tức thu nhỏ khoảng ba lần; ở chế độ cắt mảnh, mảnh 640 × 512 giữ nguyên tỉ lệ nên pixel đầu vào trùng pixel ảnh gốc. Vì vậy con số "8 pixel" trong bảng ứng với chế độ cắt mảnh; ở chế độ 1280, một vật 8 pixel của ảnh gốc chỉ còn khoảng 2,7 pixel đầu vào, tức 0,7 ô ở stride 4. Hệ quả này được bàn lại ở mục 3.1 và 3.3.

#### 2.1.2. Không gian tỉ lệ Gauss và phân rã dải tần

Lý thuyết scale-space đặt câu hỏi: khi biểu diễn ảnh ở nhiều mức thô khác nhau, họ toán tử nào không tự sinh ra cấu trúc không có trong ảnh gốc. Witkin [13] và Koenderink [14] chỉ ra rằng nếu yêu cầu phép làm mờ không sinh thêm cực trị cục bộ khi tăng tỉ lệ thì họ toán tử bị ép về phép khuếch tán Gauss; kết luận duy nhất đó đứng trên các tiên đề bổ sung (tuyến tính, đẳng hướng, bất biến dịch), và Lindeberg [15] là người hệ thống hóa chúng. Một tính chất đi kèm là tính nửa nhóm: làm mờ với $\sigma_1$ rồi $\sigma_2$ cho cùng kết quả với làm mờ một lần ở độ lệch chuẩn $\sqrt{\sigma_1^2 + \sigma_2^2}$.

Từ chuỗi ảnh đã làm mờ, hiệu giữa hai mức liên tiếp cho một dải tần. Burt và Adelson [16] dùng đúng quan hệ này để xây Laplacian pyramid: ảnh sai phân của hai mức *low-pass* là một biểu diễn *band-pass* gọn, và tổng các mức khôi phục lại ảnh gốc. Với vật cỡ vài pixel, dải band-pass chính là chỗ biên vật tách khỏi nền nước biến thiên chậm.

GauSS áp dụng quan hệ đó trên bản đồ đặc trưng stride 4 thay vì trên ảnh. Hai bộ lọc Gauss tách được chạy theo hai trục, mỗi kênh có $\sigma$ riêng. Ba dải tần lấy từ hai mức làm mờ: dải cao giữ biên và chi tiết, dải giữa ứng với cấu trúc 8–12 pixel, dải thấp giữ ngữ cảnh nền nước.

#### 2.1.3. Hồi quy khung và hàm mất mát của YOLO26

YOLO26 [17] là detector một giai đoạn, *anchor-free*, dùng bốn mức stride 4, 8, 16 và 32. Gán nhãn theo kiểu task-aligned của TOOD [18] chấm điểm mỗi ô bằng tích có trọng số giữa chất lượng phân loại và chất lượng khung, rồi ghép một-một với ground truth.

Hồi quy khung dùng CIoU [19] (Complete IoU) kèm một nhánh L1, theo mô tả của [17]. CIoU tính cả phần chồng lấn, khoảng cách tâm và tỉ lệ cạnh, nên gradient không biến mất khi hai khung không giao nhau. Đây là tình huống thường gặp ở giai đoạn đầu huấn luyện với vật nhỏ. Trong cấu hình của đồ án, tham số reg_max bằng 1 nên phần phân phối DFL [30] (Distribution Focal Loss) bị vô hiệu và khung được hồi quy trực tiếp bằng bốn số. Với vật 2–8 pixel, bề rộng khung chỉ vài pixel, nên bước lượng tử của DFL không còn tương xứng với kích thước vật và lệch 1 pixel đã là một phần lớn chiều rộng khung. Đây là khác biệt so với cấu hình mặc định của YOLO26 và chưa được đánh giá riêng trong đồ án.

### 2.2. Kiến trúc tổng thể

![](docs/report/figures/fig_overall_pipeline.png)

**Hình 2.1: Kiến trúc tổng thể của detector YOLO26n-P2 với module GauSS**

Detector nền là YOLO26n [17] với bốn mức head: P2 (stride 4), P3 (stride 8), P4 (stride 16) và P5 (stride 32). GauSS nằm tại giao điểm của ba đầu vào: C2 từ backbone (stride 4, 64 kênh), P3 từ neck (stride 8, 64 kênh) và P4 (stride 16, 128 kênh). Cả ba được chiếu về 32 kênh bằng tích chập 1 × 1. C2 và P3 được cộng sau khi upsample bằng DySample [20] để tạo đặc trưng nền ở stride 4. P3 và P4 tạo ra ngữ cảnh ở stride 8 bằng ba nhánh tích chập giãn với hệ số $r \in \{1, 3, 5\}$.

Từ đặc trưng nền, module tách ba dải tần bằng hai bộ lọc Gauss tham số hóa, tinh chỉnh từng dải ở stride 8, định tuyến theo không gian, rồi đưa trở lại stride 4 và cộng tàn dư có chặn. Đầu ra đi vào head phát hiện như bình thường; sau một tích chập stride 2, nó được nối vào đường P3 để nuôi các head còn lại.

Đường nền dùng để so sánh giữ cách nối của YOLO26n-P2: C2 (64 kênh) và P3 (64 kênh) được cộng sau khi upsample, kết quả qua một khối C3k2 tạo đặc trưng P2 32 kênh, rồi qua một tích chập 3 × 3 stride 2 (32 → 32 kênh) về stride 8 và nối với P3 thành 96 kênh; một khối C3k2 đưa về 64 kênh cho head. Head nhận bốn mức với 32 / 64 / 128 / 256 kênh. Ở cấu hình có module, GauSS thay chỗ khối C3k2 đó và trả về 32 kênh ở stride 4; tích chập 3 × 3 stride 2 giữ nguyên thông số (32 → 32 kênh, 9.216 tham số) ở cả hai cấu hình, head giữ nguyên bốn mức và nguyên số kênh đầu vào, nên toàn bộ phần tăng tham số nằm trong module. Đây là điều kiện để phép so sánh ở chương 3 chỉ còn khác biệt ở module.

Ở cấu hình thực nghiệm huấn luyện 150 epoch, nhánh C3k2 trung gian ở stride 4 (đầu vào là Upsample của P3 hợp với C2) vẫn nằm trong đồ thị. Ở mô hình nền, nhánh này chính là nguồn P2 của head; ở cấu hình có module, GauSS thay vai trò đó nên đầu ra của nhánh không còn được tiêu thụ. Nhánh có mặt ở cả hai cấu hình với cùng thông số nên không nằm trong chênh lệch tham số, và chi phí tính của nó hiện diện trong phép đo độ trễ của cả hai bên nên không làm lệch so sánh; một phiên bản gọn hơn đã bỏ nhánh này cho triển khai, và các phép đo trong báo cáo được thực hiện trên cấu hình còn nhánh trung gian này.

![](docs/report/figures/fig_gauss_module.png)

**Hình 2.2: Cấu trúc chi tiết của module GauSS: ba đầu vào, hai mức lọc Gauss, ba dải tần, router và hai nhánh tinh chỉnh**

Hình 2.2 mở rộng phần bên trong module. Ba đầu vào được chiếu về cùng chiều ẩn rồi tách thành nhánh đặc trưng nền ở stride 4 và nhánh ngữ cảnh ở stride 8. Hai mức lọc Gauss chỉ chạy trên nhánh đặc trưng nền; nhánh ngữ cảnh đi thẳng vào khối ngữ cảnh giãn. Ba dải tần sau khi mã hóa dưới pixel hội tụ với ngữ cảnh tại router, rồi tách lại thành hai nhánh tinh chỉnh trước khi trở về stride 4.

Ba nguyên tắc thiết kế theo sau. Phân rã theo thang thay vì tăng dung lượng. Định tuyến theo vị trí thay vì một trọng số chung cho cả ảnh. Tàn dư có chặn để phần tinh chỉnh bắt đầu bằng không, không xáo trộn đặc trưng nền ở bước đầu huấn luyện.

### 2.3. Các khối của module GauSS

#### 2.3.1. Chiếu kênh và hợp nhất đặc trưng

Ba đầu vào được chiếu về chiều ẩn $d = 32$:

![EQ](docs/report/figures/eq/eq_f464647173.png)

![EQ](docs/report/figures/eq/eq_5a56cf8d53.png)

Đặc trưng nền hợp nhất $S_2$ với $S_3$ đã upsample:

![EQ](docs/report/figures/eq/eq_68f1ba8533.png)

DySample [20] là bộ upsample nhận thức nội dung: đầu offset là một tích chập 1 × 1 khởi tạo bằng 0, nên bước đầu mô-đun này hoạt động đúng như nội suy song tuyến tính và học dần nơi lấy mẫu. Ngữ cảnh sâu hợp nhất $S_3$ với $S_4$ đã upsample:

![EQ](docs/report/figures/eq/eq_3bf7b18192.png)

Các phép chiếu 1 × 1 ở đầu vào không có chuẩn hóa batch; các khối tích chập còn lại đều theo sau bởi chuẩn hóa batch [21] và SiLU. Luồng tensor qua toàn bộ module với ảnh đầu vào 1280 × 1280 tóm tắt ở Bảng 2.2.

**Bảng 2.2: Luồng tensor qua module GauSS với ảnh 1280 × 1280**

| Tensor | Kích thước | Ghi chú |
|---|---|---|
| $C_2$ | 64 × 320 × 320 | backbone, stride 4 |
| $P_3$ | 64 × 160 × 160 | neck, stride 8 |
| $P_4$ | 128 × 80 × 80 | neck, stride 16 |
| $S_2$ | 32 × 320 × 320 | chiếu 1 × 1 từ $C_2$ về $d = 32$ |
| $S_3$ | 32 × 160 × 160 | chiếu 1 × 1 từ $P_3$ về $d = 32$ |
| $S_4$ | 32 × 80 × 80 | chiếu 1 × 1 từ $P_4$ về $d = 32$ |
| $X_0$ | 32 × 320 × 320 | $S_2$ + DySample($S_3$), stride 4 |
| $L_1$, $L_2$ | 32 × 320 × 320 | hai mức lọc Gauss thiết kế |
| Ba dải tần | 32 × 320 × 320 | $X_0 - L_1$, $L_1 - L_2$, $L_2$ |
| Ba biểu diễn dải | 32 × 160 × 160 | mã hóa dưới pixel về stride 8 |
| $C_{\mathrm{ctx}}$ | 32 × 160 × 160 | ngữ cảnh giãn $r \in \{1, 3, 5\}$ |
| $F_{\mathrm{band}}$ | 32 × 160 × 160 | tổng có trọng số ba dải |
| $F_{\mathrm{route}}$ | 32 × 160 × 160 | trộn hai nhánh tinh chỉnh |
| $P_2'$ | 32 × 320 × 320 | đầu ra stride 4 |

#### 2.3.2. Phân rã không gian tỉ lệ bằng bộ lọc Gauss

Trên đặc trưng nền, module chạy hai bộ lọc Gauss tách được theo hai trục, mỗi kênh có một $\sigma$ riêng:

![EQ](docs/report/figures/eq/eq_f0cb541be2.png)

$\sigma$ bị chặn trong một khoảng cố định bằng hàm sigmoid có scale:

![EQ](docs/report/figures/eq/eq_ffa8998dbe.png)

với $\sigma_1 \in [0{,}4;\, 0{,}9]$ (bán kính 2, kernel 5 × 5, khởi tạo 0,65) và $\sigma_2 \in [1{,}1;\, 1{,}8]$ (bán kính 4, kernel 9 × 9, khởi tạo 1,45). Nhân một chiều được xây bằng softmax trên các offset nguyên:

![EQ](docs/report/figures/eq/eq_0cfaa8fe9c.png)

Cách này giữ tổng nhân bằng 1 mà không cần chuẩn hóa thủ công, và gradient chảy về $\sigma$ qua chính hàm softmax. Vì phép lọc chạy trên lưới stride 4, một ô tương ứng 4 pixel đầu vào, nên bề rộng hỗ trợ của một bộ lọc (lấy $2\sigma \cdot 4$) là 5,2 pixel cho bộ lọc nhỏ và 11,6 pixel cho bộ lọc lớn hơn, tính theo pixel đầu vào. Ở chế độ cắt mảnh, pixel đầu vào trùng pixel ảnh gốc; ở chế độ 1280, pixel đầu vào tương ứng khoảng ba pixel ảnh gốc (mục 3.1), nên các bề rộng này ứng với khoảng 15–35 pixel ảnh gốc. Khởi tạo ở giữa khoảng đến từ $\mathrm{Sigmoid}(0) = 0{,}5$, nên lúc bắt đầu hai bộ lọc đã tách dải hợp lý thay vì chờ học từ đầu.

![](docs/report/figures/r_gauss_kernels.png)

**Hình 2.3: Nhân Gauss thiết kế và quan hệ giữa bề rộng hỗ trợ bộ lọc với các thang đo của TinyPerson**

Hình 2.3 được tính trực tiếp từ tham số thiết kế ($\sigma$ khởi tạo, bán kính kernel), không phải từ trọng số sau huấn luyện. Hình gồm ba phần. Phần (a) là hai nhân một chiều với trọng số tại khởi tạo; phần (b) là hiệu hai mức lọc, tức đáp ứng dải thông, với đường cong đổi dấu hai lần đúng dạng band-pass. Phần (c) đặt bề rộng hỗ trợ $2\sigma \cdot 4$ lên cùng trục với các thang đo: $2\sigma_1 = 5{,}2$ pixel nằm trong thang tiny1, còn $2\sigma_2 = 11{,}6$ pixel phủ phần lớn thang tiny2 (cận trên 12 px vượt nhẹ ngưỡng 11,6 px) và chạm tới biên dưới của thang tiny3. Đây là căn cứ định lượng cho việc chọn hai thang $\sigma$ này thay vì các giá trị khác, đọc theo pixel đầu vào của chế độ cắt mảnh (mục 3.1).

#### 2.3.3. Phân rã ba dải tần

Hiệu giữa hai mức lọc liên tiếp cho ba dải tần:

![EQ](docs/report/figures/eq/eq_7b9a7593ce.png)

Ba dải mang ba vai trò vật lý khác nhau. Dải cao giữ biên và chi tiết tần số cao; ở thang 2–8 pixel, biên của người gần như trùng với toàn bộ vật. Dải giữa là dải thông giữa, tương ứng cấu trúc cỡ 8–12 pixel: đúng vùng mà bộ lọc lớn hơn, với bề rộng hỗ trợ 11,6 pixel, được thiết kế để bắt. Dải thấp là ngữ cảnh tần số thấp, nơi mặt nước biến thiên chậm nằm. Sơ đồ phân rã ba dải tần cùng đáp ứng không gian tương ứng được trực quan hóa trên Hình 2.4. Cách phân dải này kế thừa Laplacian pyramid [16], nhưng chạy trên bản đồ đặc trưng stride 4 và với các tham số $\sigma$ thiết kế.

![](docs/report/figures/r_band_illustration.png)

**Hình 2.4: Minh họa phân rã dải tần của một đặc trưng stride 4 thành ba dải; dải giữa giữ cấu trúc 8–12 pixel, tương ứng kích thước người trên ảnh biển (vẽ minh họa theo tham số thiết kế, không tính từ dữ liệu thật)**

#### 2.3.4. Mã hóa dưới pixel

Mỗi dải được đưa về stride 8 bằng phép tách pixel với hệ số 2 [22], nhân bốn số kênh; một tích chập 1 × 1 đưa về 32 kênh; tiếp theo là *depthwise* convolution 3 × 3, chuẩn hóa theo lô (BN, Batch Normalization) và cộng tàn dư:

![EQ](docs/report/figures/eq/eq_dc4941ad0e.png)

![EQ](docs/report/figures/eq/eq_f23e47c828.png)

Kết quả là ba biểu diễn dải ở stride 8. Việc hạ về stride 8 có hai lý do: khớp với lưới ngữ cảnh để định tuyến, và giảm một phần tư chi phí tính toán so với làm việc ở stride 4.

#### 2.3.5. Ngữ cảnh giãn đa tỉ lệ

Đặc trưng ngữ cảnh đi qua ba nhánh depthwise 3 × 3 giãn song song với hệ số $r \in \{1, 3, 5\}$:

![EQ](docs/report/figures/eq/eq_0daf10002e.png)

Một softmax không gian theo vị trí trộn ba nhánh:

![EQ](docs/report/figures/eq/eq_f3542c24f1.png)

Mỗi vị trí tự chọn hệ số giãn phù hợp thay vì dùng một trọng số chung cho cả ảnh. Ba nhánh gồm 1.056 tham số; cổng softmax thêm 99, tổng 1.155.

#### 2.3.6. Router lai trên log-năng lượng

Router quyết định trọng số của ba dải tại từng vị trí. Trọng số lấy từ một tiên nghiệm năng lượng không có tham số, cộng với một hiệu chỉnh tham số hóa khởi tạo bằng 0:

![EQ](docs/report/figures/eq/eq_300b59b2e2.png)

![EQ](docs/report/figures/eq/eq_e3cbabd10a.png)

![EQ](docs/report/figures/eq/eq_4842bc0bef.png)

$\bar e_b$ là độ lớn trung bình của dải thô $b$ trên lưới stride 8, tính trước khi chuẩn hóa; log của năng lượng lấy thêm hằng số $10^{-12}$ để tránh log của 0. Phép softmax chạy trên trục dải tại từng vị trí không gian, không trên trục kênh: mỗi vị trí nhận một bộ ba trọng số. Tiên nghiệm năng lượng tính ở float32 và cắt khỏi đồ thị gradient, nên nó chỉ đóng vai trò trọng số phụ thuộc dữ liệu chứ không tạo thêm tín hiệu huấn luyện. Vì hiệu chỉnh khởi tạo bằng 0, router lúc bắt đầu hoạt động đúng theo quy tắc năng lượng chuẩn hóa. Module có cài đặt thêm một router z-loss (bình phương log-sum-exp của logit hiệu chỉnh) để theo dõi độ lớn hiệu chỉnh, nhưng trong các lần chạy của báo cáo số hạng này chưa được cộng vào hàm mất mát huấn luyện, nên nó không điều chỉnh quá trình học.

Cổng trộn dùng lại chính bằng chứng dải tần: $m_{gate} = a_h + a_m$, tức tổng phần của dải cao và dải giữa.

#### 2.3.7. Tổng hợp dải và tái tạo stride 4

Dải tần tổng hợp:

![EQ](docs/report/figures/eq/eq_25ef918411.png)

Nhánh chi tiết nhận thêm một khóa dải cao có chặn $\gamma_{in} = 0{,}5 \cdot \tanh(\hat\gamma_{in})$, nằm trong khoảng (−0,5; 0,5), để giữ năng lượng gradient cho vật dưới 8 pixel:

![EQ](docs/report/figures/eq/eq_ae2a68a983.png)

![EQ](docs/report/figures/eq/eq_2b27d3e13e.png)

![EQ](docs/report/figures/eq/eq_ae198eda8a.png)

Hai nhánh cùng dạng residual depthwise-separable, chỉ khác kernel (3 × 3 và 5 × 5), không có bias. Để tái tạo stride 4, một tích chập 1 × 1 mở đầu ra nhánh trộn lên $4d$ kênh, rồi phép ghép pixel với hệ số 2 [22] gộp về $d$ kênh ở stride 4:

![EQ](docs/report/figures/eq/eq_2fe56a696e.png)

Đầu ra là tổng tuyến tính có chặn, không qua hàm kích hoạt:

![EQ](docs/report/figures/eq/eq_a962e9e9f5.png)

![EQ](docs/report/figures/eq/eq_13d4b30563.png)

Cả ba hệ số đều bằng 0 lúc khởi tạo. Nhờ đó phần tinh chỉnh bắt đầu bằng không và đầu ra của module lúc khởi tạo chỉ là đặc trưng nền đi qua các phép chiếu 1 × 1 và DySample. Đầu ra này không phải ánh xạ đồng nhất với đường nền, vì các phép chiếu khởi tạo ngẫu nhiên. Với cấu trúc nhân theo hệ số, gradient ở bước đầu chỉ chảy vào ba hệ số; các thang $\sigma$ chỉ nhận gradient sau khi hệ số tương ứng rời khỏi 0.

### 2.4. Chi phí tham số

Module lưu 36.878 tham số, đưa detector từ 2.515.976 lên 2.552.854 tham số (+1,47%). Trong đó 517 tham số thuộc hai thành phần tùy chọn đang tắt trong cấu hình báo cáo, nên đường hoạt động thực tế có 36.361 tham số. Chênh lệch này đo trực tiếp từ hai tệp trọng số, không cộng tay từ Bảng 2.3; hai cách đếm khớp nhau vì head không đổi số kênh đầu vào. Về chi phí tính toán, module cộng thêm 0,51 GFLOPs (tỉ phép tính dấu phẩy động) ở đầu vào 640 × 640 (7,62 lên 8,13) và 2,02 GFLOPs ở đầu vào 1280 × 1280 (31,97 lên 33,99), tức khoảng 6–7% trên cả hai độ phân giải. Bảng 2.3 đếm từng khối với công thức tường minh để kiểm tra độc lập; ký hiệu $d = 32$ là số kênh nội bộ, $C_2 = 64$, $P_3 = 64$, $P_4 = 128$ là số kênh đầu vào.

**Bảng 2.3: Đếm tham số từng khối của module GauSS**

| Khối | Công thức đếm | Tham số |
|---|---|---|
| Chiếu $C_2$: 1 × 1, 64→32, không bias | 64 · 32 | 2.048 |
| Chiếu $P_3$: 1 × 1, 64→32, không bias | 64 · 32 | 2.048 |
| Chiếu $P_4$: 1 × 1, 128→32, không bias | 128 · 32 | 4.096 |
| BN sau hợp nhất | 2·32 | 64 |
| σ thiết kế: 2 thang × 2 trục × 32 kênh | 4 · 32 | 128 |
| Ba nhánh dải tần: mỗi nhánh chiếu 1 × 1 128→32 (4.096) + BN (64) + depthwise 3 × 3 (288) + BN (64) + điểm 1 × 1 32→32 (1.024) + BN (64) | 3 · 5.600 | 16.800 |
| BN sau hợp nhất ba dải | 2·32 | 64 |
| Ba nhánh ngữ cảnh giãn: mỗi nhánh depthwise 3 × 3 (288) + BN (64) | 3 · 352 | 1.056 |
| Cổng ngữ cảnh: 1 × 1, 32→3, có bias | 3 · 32 + 3 | 99 |
| Hai bộ DySample: mỗi bộ offset 1 × 1, 32→32, có bias | 2 · (32 · 32 + 32) | 2.112 |
| Hiệu chỉnh router: 1 × 1, 32→3, có bias | 3 · 32 + 3 | 99 |
| Nhánh tinh chỉnh 3 × 3: depthwise (288) + BN (64) + điểm 1 × 1 (1.024) + BN (64) | 288 + 64 + 1.024 + 64 | 1.440 |
| Nhánh tinh chỉnh 5 × 5: depthwise (800) + BN (64) + điểm 1 × 1 (1.024) + BN (64) | 800 + 64 + 1.024 + 64 | 1.952 |
| Tái tạo stride 4: 1 × 1, 32→128, không bias + BN | 128 · 32 + 2 · 128 | 4.352 |
| Ba scalar cổng | 3 | 3 |
| **Đường hoạt động** | | **36.361** |
| Router học cũ (tắt): depthwise 3 × 3 (288) + BN (64) + 1 × 1 32→4 có bias (132) | 288 + 64 + 132 | 484 |
| Cổng *saliency* (tắt): 1 × 1, 32→1, có bias | 32 + 1 | 33 |
| **Tổng module** | | **36.878** |

### 2.5. Căn cứ của các lựa chọn thiết kế

Bốn lựa chọn dưới đây quyết định phần lớn hành vi của module, nên mục này trình bày căn cứ thay vì chỉ liệt kê giá trị.

**Số dải tần.** Hai mức lọc phân tách hai biên tần số, do đó ba là số dải tối thiểu để hình thành một dải thông giữa. Thêm dải thứ tư đòi hỏi thêm một mức lọc bổ sung, kéo theo bán kính kernel lớn hơn và thêm một nhánh mã hóa song song. Về phía thang nhỏ, phần dưới 5,2 pixel của tiny1 đã nằm dưới bề rộng hỗ trợ của bộ lọc nhỏ; về phía thang lớn, cấu trúc trên 12 pixel bắt đầu đi vào dải tiếp nhận của tầng stride 8. Lựa chọn ba dải là cấu hình thiết kế thực nghiệm được chúng tôi ấn định trước; chúng tôi chưa thử nghiệm cấu hình bốn dải để đối chứng trực tiếp.

**Xử lý ở stride 8.** Ba dải được mã hóa dưới pixel về stride 8 trước khi định tuyến. Mọi phép tích chập sau bước này rẻ hơn bốn lần so với làm ở stride 4, nên chi phí là căn cứ chính. Căn cứ còn lại là khớp lưới: router so sánh ba dải với ngữ cảnh, mà ngữ cảnh nằm ở stride 8, nên hai vế phải cùng độ phân giải thì phép cộng mới có nghĩa theo từng vị trí.

**Tàn dư có chặn.** Ba hệ số $\alpha$, $\gamma_{in}$, $\gamma_{out}$ đều bằng 0 lúc khởi tạo và bị chặn trong $(-0{,}5;\, 0{,}5)$. Ở bước đầu, phần tinh chỉnh bằng không nên nó không xáo trộn đặc trưng nền; gradient bước đầu chỉ chảy vào ba hệ số, còn $\sigma$ nhận gradient sau khi hệ số rời khỏi 0. Chặn biên còn giữ cho phần tinh chỉnh không lấn át đặc trưng gốc ở giai đoạn sau.

**Hai nhánh tinh chỉnh.** Nhánh 3 × 3 và nhánh 5 × 5 chia nhau theo $m_{gate} = a_h + a_m$. Cổng này không phải một tham số học thêm mà lấy trực tiếp từ router: vị trí nào có nhiều năng lượng ở dải cao và dải giữa thì đi qua nhánh kernel nhỏ, vị trí nào nghiêng về dải thấp thì đi qua nhánh kernel lớn. Nhờ vậy hai nhánh không cạnh tranh mà chia vùng theo bằng chứng tần số.

### 2.6. Cấu hình huấn luyện

Cả hai cấu hình được huấn luyện từ đầu, không dùng trọng số ImageNet. Bảng 2.4 ghi giá trị và căn cứ của từng nhóm siêu tham số.

**Bảng 2.4: Siêu tham số huấn luyện và căn cứ chọn**

| Nhóm | Giá trị | Căn cứ |
|---|---|---|
| Đầu vào | 1280 × 1280 | Chia hết cho 32; ảnh gốc 3.480–3.840 px bị thu khoảng ba lần (mục 3.1) |
| Hồi quy khung | reg_max = 1 | Tắt nhánh phân phối DFL: với vật vài pixel, bước lượng tử của DFL lớn hơn sai số chấp nhận được (mục 2.1.3) |
| Học | lr₀ 0,002; cosine; *warmup* 3 *epoch* | Cấu hình mặc định của khung cho mô hình nano; warmup giữ ổn định ở những bước đầu khi phần tinh chỉnh của module chưa mở |
| Bộ tối ưu | AdamW [23] | Cấu hình mặc định của khung cho mô hình nano |
| Kích thước batch | 16 | Batch lớn nhất vừa bộ nhớ GPU A10G ở 1280 px; cùng giá trị cho hai cấu hình nên không lệch so sánh |
| Thời lượng | 150 *epoch* | Lần chạy thử nghiệm nội bộ 60 epoch ghi nhận val/box_loss tối thiểu tại epoch 55 cho thấy độ mất mát xác thực chưa hội tụ, do đó chúng tôi kéo dài lịch trình lên 150 epoch thay vì tăng tốc độ học |
| Độ chính xác | FP16 | Giảm bộ nhớ và tăng thông lượng huấn luyện trên GPU |
| Tăng cường dữ liệu | lật ngang 0,5, dịch 0,05, tỉ lệ 0,15, HSV [35] (h 0,005, s 0,1, v 0,15); tắt Mosaic [31], MixUp [32], CutMix [33], Copy-Paste; áp dụng bộ tiền xử lý tinyperson_b (Phụ lục A) gồm CLAHE [36] (p=0,10), Downscale (p=0,25), MotionBlur (p=0,15), ImageCompression (p=0,10), RandomBrightnessContrast (p=0,20), RandomSizedBBoxSafeCrop (p=0,20), CoarseDropout (p=0,10) | Bộ biến đổi được áp dụng theo thứ tự: ảnh gốc qua letterbox 1280 (thu nhỏ ~3 lần), sau đó Albumentations thực hiện Downscale (p=0,25, tỉ lệ [0,4; 0,7]) và RandomSizedBBoxSafeCrop (p=0,20), khiến kích thước hiệu dụng của vật 2–8 px bị co xuống chỉ còn khoảng 0,3–1,9 px; các phép MotionBlur, Compression, CLAHE mô phỏng nhiễu miền UAV |
| Chọn checkpoint | best.pt theo điểm validation | Đánh giá sau mỗi epoch trên 84 khung validation để theo dõi hàm mất mát và lưu checkpoint có val/box_loss thấp nhất, không dùng tập test |
| Tái lập | seed 42 | Một seed cho mỗi cấu hình; xem mục 3.11 |

Tập validation 84 khung được đánh giá sau mỗi epoch huấn luyện để theo dõi hàm mất mát và lưu giữ checkpoint tối ưu theo tiêu chí val/box_loss thấp nhất (đạt 2,19 ở epoch 150); tập này không dùng để chỉnh siêu tham số và không dùng để tính điểm đánh giá trong chương 3. Không có siêu tham số nào được chỉnh trên tập test. Hai cấu hình dùng chung seed và chung pipeline tăng cường dữ liệu, nên khác biệt đo được chỉ đến từ kiến trúc.

## 3. ĐÁNH GIÁ KẾT QUẢ

### 3.1. Dữ liệu và giao thức đánh giá

TinyPerson [1] có 1.610 khung hình biển và ven biển lấy cách nhau 50 khung từ 31 chuỗi video, chia sẵn 794 khung train và 816 khung test. Giao thức của bộ dữ liệu loại khỏi cả huấn luyện lẫn đánh giá những ảnh quá dày đặc (ảnh có hơn 220 chú thích hoặc có cờ *in_dense_image* trong chú thích gốc), vì ở ảnh dày, một báo nhầm cũng có xác suất trúng người rất cao và làm sai lệch chỉ số. Sau khi loại, còn 746 khung train và 786 khung test; 48 và 30 khung bị loại tương ứng. Chúng tôi tách 84 khung validation khỏi 746 khung train theo tiền tố tên video, còn lại 662 khung để cập nhật trọng số. Không có chuỗi nào xuất hiện ở cả hai tập. Chặn rò rỉ theo chuỗi quan trọng hơn là tối đa số khung train, vì hai khung video liền nhau gần như trùng nhau.

Ground truth chia theo thang đo tuyệt đối $s = \sqrt{w \cdot h}$ tính bằng pixel, không theo diện tích trực tiếp. Cách chia này giữ các bin có ý nghĩa hình học: một người 8 pixel và một người 12 pixel khác nhau ở cạnh, không phải ở bình phương cạnh. Bảng 3.1 liệt kê bảy bin cùng số ground truth tương ứng.

**Bảng 3.1: Thành phần dữ liệu và phân bố ground truth theo thang đo (786 khung test từ 31 chuỗi video, 13.687 chú thích; thang đo tính trên tọa độ ảnh gốc)**

| Thang đo | Khoảng kích thước (px) | Số ground truth |
|---|---|---|
| tiny1 | [2, 8) | 2.333 |
| tiny2 | [8, 12) | 2.778 |
| tiny3 | [12, 20] | 3.767 |
| tiny | [2, 20] | 8.878 |
| small | (20, 32] | 2.485 |
| reasonable | (32, ∞) | 2.324 |
| all | [2, ∞) | 13.687 |

Ba bin tiny1, tiny2 và tiny3 cộng lại bằng đúng bin tiny, nên một cải tiến có thể kiểm tra chéo giữa hai cách đọc. Ngoài AP, báo cáo còn dùng miss rate (MR). MR ở đây là log-average miss rate: đường cong MR theo số báo nhầm trên mỗi ảnh (FPPI) được lấy tại 9 điểm cách đều trên thang log từ $10^{-2}$ đến $10^{0}$ FPPI rồi lấy trung bình. Do đó MR không bằng 1 trừ recall. Recall ở mục 3.6 đo tại một điểm vận hành duy nhất, còn MR tích hợp cả đường cong; MR càng thấp càng tốt.

Giao thức suy luận không chạy trên ảnh gốc mà cắt thành các mảnh 640 × 512 chồng lên nhau 100 pixel. Mỗi mảnh được suy luận riêng, kết quả chiếu về tọa độ ảnh gốc rồi ghép bằng NMS toàn cục [24] với ngưỡng IoU 0,50; một ảnh trung bình cần 23,4 mảnh. Khâu khớp dùng hai tiêu chí: với vật bình thường khớp theo IoU, với vùng ignore và uncertain khớp theo Intersection-over-Detection với ngưỡng 0,50, nhờ đó mô hình không bị trừ điểm vì những chỗ chú thích gốc vốn không rõ. Hai lớp sea-person và earth-person được gộp thành một lớp person duy nhất, đúng với cách các công trình nano-scale gần đây báo cáo trên TinyPerson [3].

Ảnh được đưa vào mạng qua *letterbox* về 1280 × 1280: giữ tỉ lệ, đệm phần thiếu, không kéo méo. Cả huấn luyện lẫn suy luận cả khung dùng độ phân giải này, nên một ảnh gốc 3.840 pixel bị thu khoảng ba lần: vật 2–8 pixel của ảnh gốc chỉ còn 0,7–2,7 pixel ở đầu vào mạng. Ở chế độ cắt mảnh, mỗi mảnh 640 × 512 lấy từ ảnh gốc vào mạng đúng kích thước đó, không co lại, nên vật giữ nguyên kích thước pixel. Ở chế độ gọi là "ảnh gốc" tại mục 3.3, cả khung được letterbox về 1280 × 1280 rồi suy luận một lần và khung dự đoán được chiếu về tọa độ ảnh gốc; tên gọi này chỉ có nghĩa là không cắt mảnh, không phải chạy ở độ phân giải nguyên bản. Các thang kích thước trong Bảng 3.1 đo trên tọa độ ảnh gốc, không đo trên ảnh đã letterbox.

Hai độ phân giải này dẫn tới hai hệ quả mà báo cáo tách rõ. Một là, phân tích thang ở mục 2 (bề rộng hỗ trợ bộ lọc, số ô theo stride) đúng nguyên văn cho chế độ cắt mảnh; ở chế độ 1280, cùng bộ lọc ứng với thang gấp ba lần trên ảnh gốc. Hai là, so sánh giữa chế độ cắt mảnh và chế độ 1280 ở mục 3.3 trộn lẫn hai yếu tố: hiệu ứng cắt mảnh và hiệu ứng độ phân giải; với dữ liệu hiện có, hai yếu tố này không tách được khỏi nhau, nên kết luận "phần lớn hiệu năng đến từ pipeline cắt mảnh" cần đọc là "đến từ pipeline cắt mảnh ở độ phân giải nguyên bản".

Kết quả được đo bằng evaluator tự viết theo mô tả giao thức chuẩn của TinyPerson; mã TinyBenchmark gốc không được chạy, nên các giá trị tuyệt đối nên đọc là "đúng giao thức" chứ không phải "trùng khớp từng số với bảng công bố". Hai cấu hình được chấm bằng đúng cùng một evaluator, nên các so sánh trong báo cáo vẫn có giá trị tương đối.

### 3.2. Kết quả chính

Bảng 3.2 là kết quả trên toàn bộ tập test theo bảy thang đo, ở hai ngưỡng IoU 0,25 và 0,50. AP50 toàn thang tăng từ 40,63 lên 41,08. Thang duy nhất nằm phần lớn trong vùng bề rộng hỗ trợ của bộ lọc là tiny2 (cận trên 12 px vượt nhẹ 11,6 px) (8–12 pixel, bộ lọc lớn có bề rộng 11,6 pixel), và thang này tăng +0,75. Hai thang còn tăng rõ là small (+0,88) và reasonable (+3,45), nằm ngoài vùng đó, nên mức tăng của chúng không giải thích được bằng khớp thang của bộ lọc; mục 3.4 bàn riêng. Ở ngưỡng lỏng hơn, AP25 thang tiny1 tăng 3,99 điểm, cho thấy module tìm thêm được vật 2–8 pixel dù khung vẽ ra chưa khít biên.

**Bảng 3.2: Kết quả trên tập test TinyPerson theo thang đo (%)**

| Thang đo | GT | AP25 | + GauSS | Δ | AP50 | + GauSS | Δ |
|---|---|---|---|---|---|---|---|
| all (≥2 px) | 13.687 | 61,13 | **62,67** | +1,54 | 40,63 | **41,08** | +0,45 |
| tiny (2–20 px) | 8.878 | 58,00 | **58,92** | +0,92 | **37,57** | 37,40 | −0,17 |
| tiny1 (2–8 px) | 2.333 | 39,55 | **43,54** | +3,99 | **22,30** | 22,26 | −0,03 |
| tiny2 (8–12 px) | 2.778 | 58,50 | **59,42** | +0,92 | 36,92 | **37,67** | +0,75 |
| tiny3 (12–20 px) | 3.767 | **70,22** | 69,00 | −1,22 | **48,14** | 47,05 | −1,09 |
| small (20–32 px) | 2.485 | 71,66 | **73,80** | +2,14 | 51,63 | **52,51** | +0,88 |
| reasonable (>32 px) | 2.324 | 65,76 | **69,52** | +3,76 | 44,14 | **47,59** | +3,45 |

*Ghi chú: mỗi cặp cột, cột trái là YOLO26n-P2, cột phải là + GauSS; Δ là chênh lệch AP giữa cấu hình có module và mô hình nền, tính bằng điểm phần trăm từ giá trị chưa làm tròn; số in đậm là giá trị tốt hơn trong cặp.*

![](docs/report/figures/r_ap50_bins.png)

**Hình 3.1: AP50 theo từng thang đo trên tập test (chế độ cắt mảnh)**

![](docs/report/figures/r_delta_ap50.png)

**Hình 3.2: Chênh lệch AP50 theo thang đo (chế độ cắt mảnh)**

Hình 3.1 và Hình 3.2 làm rõ độ lệch hiệu năng giữa hai mô hình trên từng dải kích thước. Tại thang tiny1 ở cột AP25: 39,55 lên 43,54, tức gần 4 điểm. Ở ngưỡng IoU 0,50 thì thang này gần như đứng yên (22,30 so với 22,26). Cách đọc hợp lý là module tìm ra thêm vật 2–8 pixel nhưng khung vẽ ra chưa khớp chặt với biên, đúng như kỳ vọng với một module xử lý năng lượng tần số chứ không hồi quy biên. Với bài toán cứu hộ, tìm thấy quan trọng hơn khớp khung chính xác.

Bảng 3.3 tách thêm ngưỡng IoU 0,75 và AP trung bình của ba ngưỡng. Hai cột AP75 cho thấy giới hạn của cả hai cấu hình: toàn bộ thang chỉ đạt 4,99 so với 5,20, và thang 2–8 pixel rơi xuống 1,99 so với 0,85. Với vật cỡ đó, lệch 1–2 pixel đã đẩy IoU xuống dưới 0,75. Ngược lại, AP trung bình tăng ở sáu trên bảy thang: mức tăng AP25 bù được mức giảm AP75 ở tiny1, nên trung bình vẫn lên (21,28 lên 22,22). Phân bố AP chi tiết qua ba ngưỡng IoU được minh họa trực quan trong Hình 3.3.

**Bảng 3.3: AP tại ba ngưỡng IoU và AP trung bình ba ngưỡng theo thang đo (%)**

| Thang đo | AP25 | + GauSS | AP50 | + GauSS | AP75 | + GauSS | AP mean | + GauSS |
|---|---|---|---|---|---|---|---|---|
| all | 61,13 | **62,67** | 40,63 | **41,08** | 4,99 | **5,20** | 35,58 | **36,32** |
| tiny | 58,00 | **58,92** | **37,57** | 37,40 | **3,83** | 3,50 | 33,13 | **33,27** |
| tiny1 | 39,55 | **43,54** | **22,30** | 22,26 | **1,99** | 0,85 | 21,28 | **22,22** |
| tiny2 | 58,50 | **59,42** | 36,92 | **37,67** | 3,43 | 3,37 | 32,95 | **33,49** |
| tiny3 | **70,22** | 69,00 | **48,14** | 47,05 | 5,05 | **5,09** | **41,14** | 40,38 |
| small | 71,66 | **73,80** | 51,63 | **52,51** | **8,09** | 7,48 | 43,79 | **44,60** |
| reasonable | 65,76 | **69,52** | 44,14 | **47,59** | 7,49 | **8,83** | 39,13 | **41,98** |

*Ghi chú: mỗi cặp cột, cột trái là YOLO26n-P2, cột phải là + GauSS; số in đậm là giá trị tốt hơn trong cặp.*

![](docs/report/figures/r_ap_thresholds.png)

**Hình 3.3: AP của hai cấu hình tại ba ngưỡng IoU, theo thang đo (chế độ cắt mảnh)**

Miss rate đi cùng chiều với AP50. MR50 giảm 0,55 điểm ở tiny2 (85,10 xuống 84,55), 1,00 điểm ở small (76,31 xuống 75,31) và 0,61 điểm ở reasonable (80,51 xuống 79,90). Đây đúng là ba thang mà AP50 cũng tăng. Ở các thang còn lại MR50 tăng nhẹ: +0,31 ở all, +0,06 ở tiny, +0,09 ở tiny1 và +0,58 ở tiny3. Biểu đồ Hình 3.4 phản ánh chi tiết chỉ số bỏ sót theo từng dải thang đo.

![](docs/report/figures/r_mr50_bins.png)

**Hình 3.4: MR50 (log-average miss rate) theo thang đo trên tập test (chế độ cắt mảnh)**

Tại thang tiny1, mặc dù AP25 tăng gần 4 điểm nhưng MR50 tăng 0,09, cho thấy module tìm ra thêm vật ở ngưỡng khớp lỏng trong khi tỷ lệ bỏ sót ở ngưỡng khắt khe không thay đổi đáng kể. tiny3 xấu đi nhiều nhất ở cả AP50 (−1,09) và MR50 (+0,58); tiny và tiny1 cũng giảm ở AP50 nhưng dưới 0,2 điểm. Mục 3.4 quay lại điểm này.

### 3.3. Kết quả ở chế độ suy luận không cắt mảnh

Bảng 3.4 chạy cùng hai cấu hình trên ảnh gốc 1280 pixel, không cắt mảnh. Chế độ này cho biết module còn đóng góp khi không có cắt mảnh: nếu module chỉ hữu ích khi có cắt mảnh, điểm ở chế độ này sẽ bằng mô hình nền. Chế độ này không tách được phần lợi của module khỏi phần lợi của pipeline cắt mảnh ở mức hệ thống, vì hai chế độ khác nhau cả về độ phân giải lẫn cách chia ảnh (mục 3.1).

**Bảng 3.4: Kết quả trên ảnh gốc 1280 pixel, không cắt mảnh (%)**

| Thang đo | AP25 | + GauSS | AP50 | + GauSS | MR50 | + GauSS |
|---|---|---|---|---|---|---|
| all | 60,96 | **61,08** | 35,32 | **35,45** | **93,40** | 93,90 |
| tiny | 57,40 | **57,72** | 30,02 | **30,23** | **92,84** | 92,72 |
| tiny1 | 42,16 | **43,57** | **17,05** | 16,76 | **89,57** | 89,91 |
| tiny2 | **61,00** | 59,73 | 30,02 | **30,45** | 87,94 | **87,32** |
| tiny3 | 65,49 | **66,53** | 38,46 | **38,82** | 85,96 | **86,11** |
| small | **67,40** | 66,13 | **42,96** | 42,33 | **81,31** | 81,89 |
| reasonable | 70,90 | **72,05** | 47,56 | **49,10** | 78,02 | **78,87** |

*Ghi chú: mỗi cặp cột, cột trái là YOLO26n-P2, cột phải là + GauSS; số in đậm là giá trị tốt hơn trong cặp.*

Lợi thế của module co lại nhưng vẫn giữ dấu ở all (+0,12 AP25, +0,13 AP50) và rõ nhất ở reasonable (+1,15 AP25, +1,54 AP50). Cấu trúc theo thang đo giống chế độ cắt mảnh: tiny1 tăng ở AP25 (+1,41) nhưng giảm nhẹ ở AP50 (−0,29). Riêng small giảm rõ (−1,27 AP25, −0,63 AP50) rồi trở lại tăng (+2,14 AP25, +0,88 AP50) khi cắt mảnh; một cách đọc là vật 20–32 pixel ở chế độ 1280 chỉ còn 7–11 pixel đầu vào, tức bị đẩy xuống vùng mà module xử lý kém hơn, nhưng báo cáo chưa đo để kiểm chứng cách đọc này.

So với chế độ cắt mảnh, mọi con số ở đây đều thấp hơn nhiều: AP50 all 35,45 so với 41,08, MR50 all 93,90 so với 93,04. Khoảng cách này cho thấy phần lớn hiệu năng của hệ thống gắn với việc giữ nguyên độ phân giải ảnh gốc qua cắt mảnh; vì hai chế độ khác nhau cả độ phân giải lẫn cách chia ảnh, báo cáo không quy khoảng cách đó cho riêng yếu tố nào. Module đóng góp một lớp cải thiện nhỏ nhưng cùng dấu trên cả hai chế độ suy luận.

![](docs/report/figures/r_mode_delta.png)

**Hình 3.5: Chênh lệch AP50 theo thang đo ở hai chế độ suy luận**

Hình 3.5 đặt hai chế độ cạnh nhau. Chênh lệch cùng dấu ở bốn trên bảy thang (all, tiny1, tiny2, reasonable); ba thang còn lại đổi dấu nhưng biên độ đều dưới 1,3 điểm. Thang reasonable tăng ở cả hai chế độ. Ở chế độ cắt mảnh, ba bin giảm là tiny1, tiny3 và bin tiny gộp; ở chế độ ảnh gốc chỉ tiny1 giảm.

### 3.4. Phân tích tác động theo từng thang đo và hành vi bộ lọc Gauss

Bề rộng hỗ trợ của hai bộ lọc là 5,2 và 11,6 pixel đầu vào (mục 2.3.2), phủ vùng cấu trúc cỡ 5–12 pixel. Giả thuyết của đồ án: vật trong thang tiny2 nằm trong vùng đó, nên hiệu $L_1 - L_2$ tách được biên của vật khỏi nền nước biến thiên chậm, và đây là lý do mức tăng ở tiny2 (+0,75) lớn hơn tiny1 (−0,03). Với tiny1, lập luận chỉ đúng một phần: bộ lọc nhỏ phủ 5,2 pixel, tức phần trên của thang 2–8 pixel, còn phần dưới 5,2 pixel nằm ngoài vùng hỗ trợ nên phép làm mờ lấy đi nhiều thông tin hơn là giữ lại. Báo cáo chưa đo trực tiếp đặc trưng dải tần để chứng minh cơ chế này.

Trích từ trọng số sau huấn luyện, $\sigma$ gần như đứng tại giá trị khởi tạo: trung bình theo kênh của $\sigma_1$ là 0,65 so với 0,65 khởi tạo, của $\sigma_2$ là 1,43 so với 1,45. Phân bố theo kênh cũng hẹp: $\sigma_1$ nằm trong [0,62; 0,68] và $\sigma_2$ trong [1,36; 1,50] trên 32 kênh. Huấn luyện chỉ dịch bộ lọc rất ít, nên bề rộng hỗ trợ nêu trên vẫn đúng sau huấn luyện. Dữ liệu này không chứng minh rằng module tự tìm được thang lọc phù hợp: vị trí thang là của thiết kế, không phải của học.

Hai thang nằm ngoài vùng hỗ trợ vẫn tăng. Với small (20–32 pixel) và reasonable (>32 pixel), vật lớn hơn nhiều so với bề rộng bộ lọc, nên mức tăng (+0,88 và +3,45) không giải thích được bằng khớp thang; một cách đọc là bộ lọc Gauss và router làm giảm nhiễu tần số cao của nền nước trên toàn ảnh, nhờ đó vật lớn hưởng lợi cùng đường. Báo cáo chưa có phép đo precision riêng theo thang để kiểm chứng cách đọc này. Con số 47,59 so với 44,14 là mức chênh lớn nhất trong bảng.

Riêng tiny3 giảm 1,09 điểm. Báo cáo chưa có lời giải thích chắc chắn cho dấu hiệu này; với một lần chạy duy nhất cho mỗi cấu hình, mức chênh này chưa tách được khỏi nhiễu giữa các lần chạy, nên cần nhiều seed trước khi kết luận.

### 3.5. Phân tích thành phần

Để phân tích vai trò của từng thành phần trong module GauSS, chúng tôi thực hiện loại bỏ từng nhánh cấu trúc trong 150 epoch huấn luyện (Bảng 3.5). Các biến thể được đánh giá theo quy trình cắt mảnh chuẩn và báo cáo điểm số AP50 cho lớp đối tượng trọng tâm (sea-person). Khoảng tin cậy trong bảng được tính bằng phương pháp bootstrap 2.000 lần lấy mẫu lại theo ảnh thử nghiệm, phản ánh độ bất định do chọn mẫu ảnh kiểm tra và đánh giá độ suy giảm hiệu năng khi thiếu vắng từng nhánh thành phần.

**Bảng 3.5: Phân tích thành phần của module GauSS (AP50 toàn thang lớp sea-person ở 150 epoch, chế độ cắt mảnh, %; khoảng tin cậy 95% bootstrap 2.000 lần lấy mẫu lại)**

| Biến thể | AP50 | Δ so với Đầy đủ | KTC 95% của Δ | KTC có chứa 0 |
|---|---|---|---|---|
| Đầy đủ (150 epoch) | **29,63** | – | [26,60; 32,99] | mốc |
| Bỏ dải giữa | 24,96 | −4,66 | [−6,40; −2,75] | không |
| Bỏ router lai | 26,43 | −3,20 | [−5,23; −1,50] | không |
| Bỏ dải cao | 27,86 | −1,77 | [−3,88; +0,03] | có |
| Bỏ dải thấp | 28,41 | −1,22 | [−3,06; +0,36] | có |
| Bỏ cổng saliency | 25,41 | −4,22 | [−6,05; −2,47] | không (nhánh đối chứng ngắt mạch) |

Hai can thiệp mang lại độ suy giảm hiệu năng có ý nghĩa thống kê rõ rệt nhất (khoảng tin cậy không chứa 0) là việc loại bỏ dải tần giữa (−4,66 điểm AP50) và loại bỏ cơ chế router lai (−3,20 điểm AP50). Khi loại bỏ dải cao hoặc dải thấp, mức giảm lần lượt là −1,77 và −1,22 điểm, tuy nhiên khoảng tin cậy bootstrap 95% đều vắt qua giá trị 0, cho thấy đóng góp của hai dải biên này có mức độ tác động phụ thuộc nhiều hơn vào từng ngữ cảnh hình ảnh cụ thể.

Dòng "bỏ cổng saliency" đóng vai trò kiểm chứng đối chứng: do cấu hình hoạt động chính của module đã chủ động ngắt nhánh cổng này để giữ tính tinh gọn, việc loại bỏ nhánh này phản ánh độ nhạy cấu trúc và phương sai giữa các nhánh đồ thị tính toán.

Bảng cũng không có dòng "không module, cùng thiết lập". Thiếu dòng đó, phép trừ trong bảng chỉ định vị đóng góp bên trong module; nó không chứng minh module tốt hơn không có module. Kết luận về toàn module dựa vào cặp cùng đợt ở Bảng 3.2, nơi hai cấu hình chỉ khác nhau ở việc có module hay không.

Ablation và phân bố theo thang chỉ về cùng một hướng: bỏ dải giữa làm AP50 giảm nhiều nhất trong bảng, và thang tăng rõ nhất nằm trong vùng hỗ trợ của dải này là tiny2; các thang small và reasonable cũng tăng nhưng nằm ngoài vùng đó (mục 3.4). Đây là hai phép đo độc lập cùng hướng, không phải hai bằng chứng cộng dồn thành một kết luận chắc hơn.

Mức tăng AP25 ở tiny1 (+3,99) với AP50 gần như đứng yên (−0,03) cho thấy module tác động lên năng lượng đặc trưng yếu hơn là tinh chỉnh biên. Đó là hệ quả của thiết kế: khóa dải cao $\gamma_{out} B_{high}$ giữ thông tin biên nguyên trạng thay vì làm sắc nó.

### 3.6. Phân tích lỗi

Bảng 3.6 tách toàn bộ dự đoán của hai cấu hình trên tập test thành các loại: khớp đúng (true positive, TP), trùng lặp (khớp đúng một ground truth đã được khớp trước đó), lỗi định vị (khớp ground truth nhưng IoU dưới ngưỡng), báo nhầm nền (false positive, FP: không khớp ground truth nào), dự đoán trung tính (khớp vùng ignore hoặc uncertain, hoặc nằm ngoài khoảng kích thước được chấm) và bỏ sót. Precision tính trên phần dự đoán được chấm, không gồm nhóm trung tính. Cách phân loại này cho biết phần tăng recall đến từ đâu và cái giá phải trả là gì. Hai chỉ số này dịch chuyển ngược chiều: recall tăng 0,51 điểm, precision giảm 0,49 điểm.

**Bảng 3.6: Phân loại dự đoán và lỗi trên tập test (chế độ ảnh gốc, không cắt mảnh)**

| Chỉ số | YOLO26n-P2 | + GauSS | Chênh |
|---|---|---|---|
| Dự đoán tổng | 104.249 | 111.786 | +7.537 |
| Khớp đúng (TP) | 8.050 | **8.119** | +69 |
| Trùng lặp | 2.153 | 2.225 | +72 |
| Lỗi định vị | 27.185 | 27.538 | +353 |
| Báo nhầm nền | 62.130 | 68.925 | +6.795 |
| Trung tính (khớp vùng ignore/uncertain hoặc ngoài khoảng kích thước) | 4.731 | 4.979 | +248 |
| Bỏ sót | 5.637 | **5.568** | −69 |
| Recall (%) | 58,81 | **59,32** | +0,51 |
| Precision (%) | **8,09** | 7,60 | −0,49 |
| Tỷ trọng trùng lặp trong FP (%) | 2,35 | 2,25 | −0,10 |
| Tỷ trọng lỗi định vị trong FP (%) | 29,72 | **27,90** | −1,82 |
| Tỷ trọng báo nhầm nền trong FP (%) | 67,93 | 69,84 | +1,91 |

Phân tích phân bố lỗi cho thấy sự đánh đổi giữa recall và precision. Module tìm thấy thêm 69 ground truth (recall +0,51 điểm) nhưng phải trả bằng 7.537 dự đoán thêm, trong đó 6.795 là báo nhầm nền; precision vì thế giảm 0,49 điểm. Cấu trúc FP của hai cấu hình gần như trùng nhau: khoảng 68–70% FP là báo nhầm nền, 28–30% là lỗi định vị, chỉ khoảng 2% là trùng lặp. Cả hai đều bị chi phối bởi nhiễu nền, đúng như thiết kế ban đầu của module nhắm vào nhiễu mặt nước.

![](docs/report/figures/r_error_composition.png)

**Hình 3.6: Thành phần dự đoán (a) và thành phần báo nhầm (b) của hai cấu hình ở chế độ ảnh gốc**

Hình 3.6a vẽ lại Bảng 3.6 theo tỉ trọng trên tổng số dự đoán; hai thanh gần như chồng khít, phần mở rộng của cấu hình có module nằm chủ yếu ở nhánh nền. Hình 3.6b vẽ riêng thành phần của các báo nhầm trên tổng số FP, đúng tỉ trọng của ba dòng cuối trong Bảng 3.6.

Tỷ trọng lỗi định vị trong FP giảm 1,82 điểm (27,90 so với 29,72), nhưng số tuyệt đối lại tăng 353 (27.538 so với 27.185): tỷ trọng giảm vì báo nhầm nền tăng nhanh hơn (+6.795), không phải vì lỗi định vị ít đi. Cách đọc đúng là phần dự đoán thêm của module rơi chủ yếu vào vùng không có người, còn chất lượng khung của các phát hiện đúng không được cải thiện. Điều này phù hợp với việc AP50 thang tiny gần như đứng yên ở mục 3.2.

Bảng 3.6 mô tả hành vi ở chế độ không cắt mảnh; cấu trúc lỗi ở chế độ cắt mảnh có thể khác do NMS ghép mảnh đã loại một phần trùng lặp.

### 3.7. Phân tích định tính

![](docs/report/figures/12_qualitative_tiny_detections.png)

**Hình 3.7: So sánh phát hiện trong một cụm người bơi ngoài khơi**

Chúng tôi trích xuất trực quan hai trường hợp điển hình trên tập test:
1. **Cụm người bơi (khung hình thử nghiệm số 320, chuỗi V0001; chi tiết tại Phụ lục D):** Trong cửa sổ cắt 280 × 280 pixel chứa 6 mục tiêu ground truth (Hình 3.7), mô hình nền phát hiện được 2 mục tiêu ở ngưỡng tin cậy 0,20, trong khi mô hình có GauSS phát hiện được cả 6 mục tiêu nhưng xuất ra tổng cộng 10 hộp bao dự đoán (điểm tin cậy 0,31–0,44). Việc xuất hiện 10 dự đoán cho 6 vật thể thể hiện mô hình có xu hướng dự đoán dư thừa và trùng lặp trong cụm dày đặc. Trên toàn bộ khung hình này sau NMS, tổng số dự đoán tăng từ 12 lên 21 hộp.
2. **Khu vực đỉnh sóng biển (khung hình thử nghiệm số 2640, chuỗi V0001; chi tiết tại Phụ lục D):** Tại một vùng đỉnh sóng nhấp nhô không có người, mô hình nền xuất hiện một phát hiện báo nhầm; mô hình có GauSS loại bỏ được báo nhầm này, giúp tổng số dự đoán trên toàn khung giảm từ 95 xuống 71 hộp.

Hai quan sát trên lấy từ hai khung hình riêng biệt với hành vi đối nghịch nhau (một khung tăng số dự đoán, một khung giảm). Các quan sát định tính này chỉ nhằm minh họa trực quan sự thay đổi phân bố phản hồi của mạng, không cấu thành bằng chứng thống kê cho khả năng giảm báo nhầm tổng thể trên toàn tập dữ liệu.

### 3.8. So sánh tham khảo với các công trình đã công bố

Các con số TinyPerson công bố đều là mAP50 trên toàn tập test theo giao thức riêng của từng bài báo: bảng gốc của [1] và Bảng 5 của [3] chỉ báo mAP50 và mAP50-95, không tách theo thang. Vì bộ evaluator của chúng tôi không phải là bản cài đặt TinyBenchmark chính thức, giá trị tuyệt đối giữa hai nguồn không đối chiếu được trực tiếp. Hình 3.8 chỉ dùng mặt phẳng tham số–AP50 toàn thang để đặt bối cảnh: điểm của các công bố lấy nguyên văn từ bài báo gốc, hai điểm của chúng tôi lấy từ bộ evaluator của chúng tôi trên cùng đại lượng.

Trên biểu đồ phân tán tham số và AP50 (Hình 3.8), chúng tôi đưa vào các kết quả công bố của Faster R-CNN (43,55), RetinaNet (30,82), FCOS (16,90) từ [1], Cascade R-CNN (57,19) và SSPNet (59,13) từ [2], cùng YOLOv8n+P2 (38,00) và BPD-YOLO (39,10) từ [3]. Chúng tôi không đưa các biến thể BiFPN (38,40) hay AFPN (39,20) vào biểu đồ do Bảng 1.2 không liệt kê chi tiết các biến thể mạng cổ phụ này.

Khoảng cách điểm số giữa các mô hình không thể suy diễn thành ưu thế kiến trúc thuần túy, do sự khác biệt lớn về giao thức thực nghiệm. Các nghiên cứu như BPD-YOLO [3] sử dụng độ phân giải đầu vào 1024 pixel, kết hợp tăng cường dữ liệu Mosaic và MixUp với lịch huấn luyện kéo dài; trong khi thực nghiệm của chúng tôi sử dụng đầu vào 1280 pixel, tắt Mosaic/MixUp và không dùng trọng số tiền huấn luyện ImageNet. Do đó, mọi kết luận về tác động của GauSS chỉ được rút ra từ phép so sánh đối chứng nội bộ giữa hai cấu hình cùng điều kiện tại Bảng 3.2.

![](docs/report/figures/r_params_ap_scatter.png)

**Hình 3.8: Tương quan giữa số lượng tham số và AP50 toàn thang của các mô hình công bố trên TinyPerson**

### 3.9. Hiệu năng và độ phức tạp tính toán

Bảng 3.7 đo trên NVIDIA A10G, FP16, batch 1, 25 lượt lặp warmup và 100 lượt lặp tính thời gian. FPS là số khung hình xử lý được trong một giây; VRAM là bộ nhớ GPU đỉnh được cấp phát.

**Bảng 3.7: Độ trễ, thông lượng và độ chính xác theo chế độ suy luận**

| Chế độ | Mô hình | Tham số | Độ trễ (ms) | FPS | AP50 all | AP25 all |
|---|---|---|---|---|---|---|
| Cắt mảnh 640 × 512 | YOLO26n-P2 | 2,52 M | 11,28 | 88,6 | 40,63 | 61,13 |
| Cắt mảnh 640 × 512 | + GauSS | 2,55 M | 15,19 | 65,8 | **41,08** | **62,67** |
| Ảnh gốc 1280 × 1280 | YOLO26n-P2 | 2,52 M | 11,85 | 84,4 | 35,32 | 60,96 |
| Ảnh gốc 1280 × 1280 | + GauSS | 2,55 M | 16,58 | 60,3 | **35,45** | **61,08** |

![](docs/report/figures/r_latency_accuracy.png)

**Hình 3.9: Độ trễ và độ chính xác ở hai chế độ suy luận**

![](docs/report/figures/04_hardware_efficiency.png)

**Hình 3.10: Độ trễ trên mỗi mảnh và bộ nhớ đỉnh trên NVIDIA A10G**

Đánh đổi giữa độ trễ và độ chính xác được tổng hợp trực quan trong Hình 3.9, trong khi Hình 3.10 trình bày chi tiết thời gian suy luận trên từng mảnh cùng lượng bộ nhớ VRAM đỉnh khi chạy thực tế trên NVIDIA A10G. Ảnh test có độ phân giải gốc rất khác nhau. Cắt mảnh 640 × 512 sinh trung bình 23,4 mảnh trên một ảnh của tập test. Thời gian trong Bảng 3.7 chỉ tính lượt gọi mạng; khâu cắt mảnh, ghép kết quả và NMS toàn cục nằm ngoài phép đo. Tổng thời gian các lượt gọi mạng cho một khung hình trung bình mất khoảng 355 ms (tương đương 2,8 FPS mạng); với ảnh 1280 × 1280 còn 9 mảnh và đạt khoảng 7,3 FPS. Chạy thẳng ở 1280 × 1280 đạt 60,3 FPS mỗi lượt gọi mạng, nhưng ở độ phân giải đó điểm thấp hơn (Bảng 3.7), nên tốc độ cao không bù được độ chính xác.

**Bảng 3.8: Tham số, FLOPs và bộ nhớ đỉnh (NVIDIA A10G, FP16, batch 1)**

| Chỉ số | YOLO26n-P2 | + GauSS |
|---|---|---|
| Tham số | 2,52 M | 2,55 M |
| FLOPs đầu vào 640 × 640 | 7,62 GFLOPs | 8,13 GFLOPs |
| FLOPs đầu vào 1280 × 1280 | 31,97 GFLOPs | 33,99 GFLOPs |
| VRAM cắt mảnh | 91,5 MB | 122,1 MB |
| VRAM ảnh gốc | 244,7 MB | 367,2 MB |

FLOPs đo ở hai đầu vào vuông 640 × 640 và 1280 × 1280; một mảnh 640 × 512 thực tế tốn ít hơn theo tỉ lệ diện tích, khoảng 6,1 GFLOPs cho mô hình nền, nên cột FLOPs của Bảng 3.8 là thước đo chi phí mỗi lần gọi mạng, không phải chi phí mỗi mảnh. FLOPs tăng 0,51 GFLOPs (+6,7%) và 2,02 GFLOPs (+6,3%), trong khi tham số chỉ tăng 1,47%. Hai tỉ lệ này không tương xứng và không cần tương xứng: module dựng chủ yếu từ tích chập depthwise và các phép chiếu 1 × 1 trên đặc trưng stride 4, nơi mỗi tham số sinh ra nhiều phép tính hơn một lớp tích chập thường ở độ phân giải thấp. Độ trễ tăng nhiều hơn FLOPs: +34,7% và +39,9%. Nguyên nhân nhiều khả năng là các phép depthwise, router nhiều nhánh và PixelShuffle/PixelUnshuffle bị giới hạn băng thông bộ nhớ ở batch 1; chúng tôi chưa chạy profiling để tách nguyên nhân, nên đây là giả thuyết làm việc. Hướng tối ưu tương ứng là gộp các nhánh mã hóa dải tần vào một lần dispatch.

Bộ nhớ đỉnh tăng mạnh hơn cả độ trễ: 91,5 lên 122,1 MB ở chế độ cắt mảnh (+33%) và 244,7 lên 367,2 MB ở chế độ 1280 (+50%). Mức tăng bộ nhớ này lớn hơn nhiều so với tỷ lệ +1,47% tham số. Nguyên nhân nằm ở kích thước các bản đồ đặc trưng trung gian của module ở 1280 × 1280 (ba dải ở stride 8, tensor 4d kênh trước khi ghép pixel). Đây là chi phí bộ nhớ chứ không phải chi phí tham số, và là lý do cần cân nhắc khi triển khai trên thiết bị biên có VRAM hạn chế.

### 3.10. Quá trình hội tụ hàm mất mát khi huấn luyện

![](docs/report/figures/07_training_loss_convergence.png)

**Hình 3.11: Đường cong hội tụ của loss box và loss classification**

Quá trình hội tụ chi tiết của hai thành phần hàm mất mát được thể hiện trong Hình 3.11. Cả hai cấu hình huấn luyện ổn định trong suốt 150 epoch. Loss box giảm từ khoảng 3,85 xuống 2,21 ở mô hình nền và từ 3,83 xuống 2,21 ở cấu hình có module; hai đường gần như trùng nhau, phù hợp với việc module giữ nguyên dạng tham số hóa khung và chỉ thay đổi đặc trưng đầu vào của head. Loss classification của cấu hình có module khởi đầu cao hơn (8,52 so với 6,47) nhưng hội tụ về cùng mức (1,80 so với 1,78) vào cuối lịch. Nguyên nhân của phần khởi đầu cao hơn chưa được kiểm chứng; một khả năng là head cần thời gian thích nghi với đặc trưng P2 vừa được module tái tạo. Không có nhảy loss hay dao động bất thường ở cả hai đường.

### 3.11. Hạn chế và các yếu tố ảnh hưởng đến độ tin cậy của kết quả

Mỗi cấu hình chỉ chạy một lần với seed 42, nên không có ước lượng nhiễu giữa các lần chạy: ngoài Bảng 3.5, không có con số nào trong chương này kèm khoảng tin cậy, và các chênh lệch nhỏ như tiny2 (+0,75) hay small (+0,88) phải đọc cùng xu hướng chung của cả bảng thay vì tách riêng. Các hạn chế còn lại và hướng xử lý nằm ở Bảng 3.9.

**Bảng 3.9: Hạn chế và hướng khắc phục**

| Hạn chế | Ảnh hưởng | Hướng khắc phục |
|---|---|---|
| Một seed duy nhất (42) cho mỗi cấu hình | Phương sai ngẫu nhiên giữa các lần huấn luyện có thể lên tới hơn 4 điểm AP50 (mục 3.5); chênh lệch 0,45 điểm chưa tách được khỏi nhiễu chạy | Tối thiểu 3 seed, bootstrap theo chuỗi video |
| Ablation chỉ thực hiện trên một seed duy nhất | Bảng 3.5 chưa tách trọn vẹn nhiễu giữa các lần chạy | Lặp lại thí nghiệm trên nhiều seed độc lập |
| Thiếu đối chứng cùng số tham số | Chưa tách tác dụng cấu trúc scale-space khỏi tác dụng của 36.878 tham số tăng thêm | Khối tích chập thường cùng ngân sách tham số |
| Định vị ở ngưỡng IoU chặt còn yếu (AP75 tiny1 0,85) | Lợi thế phát hiện chưa chuyển thành khung khít | Head hồi quy offset dưới pixel |
| Chưa đo trên thiết bị biên | 65,8 FPS đo trên A10G cho từng mảnh 640 × 512 chưa phản ánh độ trễ end-to-end trên UAV thật | Đo end-to-end trên Jetson |

Về độ tin cậy nội bộ, rủi ro lớn nhất là evaluator. Toàn bộ số liệu đến từ evaluator tự viết theo mô tả giao thức trong bài báo gốc; sai khác của nó so với TinyBenchmark chưa được định lượng, vì mã gốc chưa được chạy trên cùng dữ liệu. Vì hai cấu hình được chấm bằng đúng cùng một evaluator, các so sánh trong bảng vẫn có giá trị tương đối; các giá trị tuyệt đối thì nên đọc thận trọng. Cách kiểm chứng trực tiếp là tái lập một con số đã công bố bằng evaluator này, ví dụ 43,55 của Faster R-CNN trong [1]; việc này chưa thực hiện được trong phạm vi nghiên cứu.

Về khả năng khái quát hóa, kết quả chỉ đo trên TinyPerson. Bộ dữ liệu này lấy mẫu video cách 50 khung và có phân bố vật nghiêng về thang nhỏ; cách module hoạt động trên ảnh hồng ngoại hay trên bộ dữ liệu người nhỏ khác chưa được kiểm chứng. Việc gộp hai lớp thành một lớp person cũng làm kết quả không so trực tiếp được với các công trình báo cáo hai lớp riêng.

### 3.12. Đánh giá tổng hợp chương

Nghiên cứu đạt được các kết quả cụ thể sau: Xây dựng detector YOLO26n-P2 tái lập được trên TinyPerson, huấn luyện từ đầu, chia train/test theo chuỗi video. Thiết kế module GauSS 36.878 tham số, phân rã đặc trưng stride 4 thành ba dải tần bằng bộ lọc Gauss tham số hóa rồi định tuyến theo không gian. Trên tập test, AP50 tổng tăng từ 40,63 lên 41,08, tăng 0,75 điểm ở thang 8–12 pixel, 0,88 điểm ở thang 20–32 pixel, 3,45 điểm ở thang trên 32 pixel, và gần 4 điểm AP25 ở thang 2–8 pixel; tiny1 và tiny3 giảm nhẹ ở AP50. Phân tích thành phần cho thấy bỏ dải giữa và bỏ router lai là hai can thiệp có khoảng tin cậy bootstrap không chứa 0; hai dải còn lại chưa tách được khỏi nhiễu, và khoảng tin cậy này chỉ tính nhiễu lấy mẫu ảnh.

Khoảng cách 5,63 điểm AP50 giữa chế độ cắt mảnh (41,08) và chế độ 1280 × 1280 (35,45) gắn với việc pipeline cắt mảnh giữ nguyên độ phân giải ảnh gốc. Hai chế độ khác nhau cả độ phân giải lẫn cách chia ảnh, nên báo cáo không quy khoảng cách đó cho riêng yếu tố nào. Cách phối hợp hai chế độ trong triển khai thực tế là câu hỏi chưa được đánh giá trong đồ án.

Các hướng phát triển ưu tiên tiếp theo được trình bày chi tiết tại phần Kết luận và hướng phát triển, tập trung vào ba nhiệm vụ trọng tâm: kiểm soát phương sai bằng huấn luyện đa seed kèm đối chứng iso-parameter; đồng bộ thang đo và đánh giá thực tế trên phần cứng UAV; tối ưu hóa độ trễ truy cập bộ nhớ và hồi quy offset dưới pixel.

## KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

Đồ án đặt ra một câu hỏi hẹp: thông tin của vật 2–12 pixel trên ảnh biển có thể được tách khỏi nền theo thang không gian ngay tại tầng ghép stride 4, và cái giá của việc tách đó là bao nhiêu. Câu trả lời là có, nhưng mức lợi nhỏ và phụ thuộc chặt vào điều kiện suy luận. Phần này trả lời lần lượt bốn mục tiêu đã tuyên bố ở mục 1.4, sau đó bàn về cơ chế đứng sau các con số, giới hạn của bằng chứng và hướng phát triển.

**Mục tiêu 1, detector nền tái lập được.** Chúng tôi dựng detector YOLO26n-P2 trên TinyPerson, huấn luyện từ đầu trong 150 epoch và không dùng trọng số ImageNet. Dữ liệu được tách theo tiền tố tên video, nên không khung nào của một chuỗi xuất hiện ở hai tập: còn 662 khung huấn luyện, 84 khung xác thực và 786 khung kiểm thử với 13.687 chú thích. Mọi phép so sánh trong báo cáo đều nằm trên cùng bộ dữ liệu này.

**Mục tiêu 2, module nằm trong ngân sách tham số dưới 2%.** GauSS dùng 36.361 tham số trên đường hoạt động, và 36.878 tham số nếu bật hai thành phần tùy chọn; detector vì thế tăng từ 2.515.976 lên 2.552.854 tham số, tức 1,47%. Mức cải thiện độ chính xác là 0,45 điểm AP50 toàn thang (40,63 lên 41,08) ở chế độ cắt mảnh, tập trung ở thang 8–12 pixel (+0,75), thang 20–32 pixel (+0,88) và thang trên 32 pixel (+3,45). Ở thang nhỏ nhất là thang 2–8 pixel, module tăng 3,99 điểm AP25 trong khi AP50 đứng yên (−0,03). Nói cách khác, module bơm thêm năng lượng tín hiệu cho vật cực nhỏ nhưng chưa cải thiện được độ khít của khung vẽ ra.

**Mục tiêu 3, đánh giá theo giao thức và theo từng ngưỡng.** Kết quả được báo cáo trên bảy thang đo của TinyPerson, tại ba ngưỡng IoU (0,25, 0,50, 0,75), kèm miss rate đo tại 9 điểm FPPI. AP trung bình ba ngưỡng tăng ở sáu trên bảy thang, và ba thang mà miss rate giảm trùng với ba thang mà AP50 tăng. Điểm cần đọc cẩn thận là ngưỡng chặt: AP75 ở thang 2–8 pixel chỉ đạt 0,85, cho thấy lợi thế phát hiện chưa chuyển thành khung khít.

**Mục tiêu 4, phân tích thành phần và ngân sách thời gian thực.** Kiểm định bootstrap 2.000 lần cho thấy loại bỏ dải giữa làm AP50 giảm 4,66 điểm (KTC 95%: [−6,40; −2,75]) và loại bỏ router lai làm giảm 3,20 điểm (KTC 95%: [−5,23; −1,50]); cả hai khoảng tin cậy đều không chứa 0. Hai dải còn lại chưa tách được khỏi nhiễu. Về chi phí, độ trễ mạng ở chế độ cắt mảnh tăng từ 11,28 lên 15,19 ms (+34,7%), còn bộ nhớ đỉnh tăng từ 91,5 lên 122,1 MB (+33%). Ngân sách thời gian thực vì thế bị giới hạn bởi chi phí tính toán của module, không phải bởi số tham số.

**Cơ chế đứng sau các con số.** Bốn phát hiện giải thích phần lớn kết quả. Mức lợi lớn nhất rơi vào thang 8–12 pixel, đúng vùng mà bề rộng hỗ trợ 11,6 pixel của bộ lọc lớn phủ; đây là bằng chứng gián tiếp rằng dải thông giữa mang thông tin biên của vật trong dải kích thước đó. Tham số $\sigma$ sau huấn luyện gần như đứng tại giá trị khởi tạo (0,65 và 1,43 so với 0,65 và 1,45), nên module hành xử như một bộ lọc band-pass cố định theo thiết kế chứ không phải một bộ lọc tự học được ngưỡng lọc phù hợp. Việc AP25 tăng mạnh ở thang 2–8 pixel trong khi AP50 đứng yên cho thấy tác động của module nằm ở mức năng lượng đặc trưng hơn là ở mức hồi quy biên, một hệ quả của việc khóa dải cao bằng $\gamma_{out}$. Cuối cùng, khoảng cách giữa hai chế độ suy luận lớn hơn nhiều lần so với mức tăng do module tạo ra: ở chế độ 1280 × 1280, AP50 chỉ nhích 0,13 điểm. Sự lệch thang giữa huấn luyện và suy luận, chứ không phải cấu trúc module, là yếu tố chi phối kết quả.

**Giới hạn của bằng chứng.** Bốn giới hạn cần nêu rõ. Mỗi cấu hình chỉ chạy một seed, nên không có ước lượng nhiễu giữa các lần huấn luyện; khoảng tin cậy bootstrap trong Bảng 3.5 chỉ phản ánh nhiễu lấy mẫu ảnh kiểm thử. Đồ án chưa xây mô hình đối chứng cùng số tham số, nên chưa tách được tác dụng của cấu trúc scale-space khỏi tác dụng của việc thêm 36 nghìn tham số. Số liệu đo bằng evaluator tự viết theo mô tả giao thức, chưa đối chiếu với mã TinyBenchmark chính thức. Cuối cùng, độ trễ đo trên A10G không đại diện cho thiết bị bay thực tế.

**Hướng phát triển.** Chúng tôi xác định ba hướng ưu tiên:

1. **Huấn luyện đa seed và đối chứng cùng số tham số (iso-parameter).** Huấn luyện tối thiểu 3 seed độc lập trọn vẹn 150 epoch để kiểm soát phương sai huấn luyện; đồng thời xây dựng mô hình đối chứng chuẩn hóa cùng số tham số (thay thế GauSS bằng khối tích chập tiêu chuẩn cùng ngân sách 36 nghìn tham số) nhằm phân tách dứt khoát hiệu quả của cấu trúc scale-space khỏi tác động của việc tăng dung lượng mô hình.
2. **Đồng bộ thang đo và đánh giá tính khả thi trên thiết bị bay.** Huấn luyện trực tiếp trên các mảnh cắt độ phân giải gốc để triệt tiêu độ lệch thang đo giữa huấn luyện và suy luận; triển khai và đo kiểm thông lượng end-to-end trên phần cứng biên thực tế (ví dụ dòng NVIDIA Jetson trang bị trên UAV) nhằm trả lời trọn vẹn yêu cầu vận hành thời gian thực.
3. **Tối ưu hóa độ trễ truy cập bộ nhớ và hồi quy dưới pixel.** Gộp các kernel tích chập nhánh để giảm độ trễ truy cập bộ nhớ; bổ sung cơ chế hồi quy bù dịch chuyển dưới pixel (sub-pixel offset) để nâng cao độ chính xác định vị khung bao tại ngưỡng IoU cao, nơi lợi thế phát hiện hiện vẫn chưa chuyển thành lợi thế định vị.

**Kết luận.** GauSS cho thấy có thể tách vật cực nhỏ khỏi nền tần số thấp ngay tại tầng ghép stride 4 với chi phí 1,47% tham số, và lợi ích đó tập trung đúng vào dải kích thước mà bộ lọc band-pass phủ. Đồng thời, kết quả cho thấy phần lớn giá trị của một detector vật cực nhỏ nằm ở giao thức suy luận chứ không ở cấu trúc khối: chênh lệch giữa chế độ cắt mảnh và chế độ cả khung là 5,63 điểm AP50, lớn hơn nhiều so với 0,45 điểm mà module tạo ra. Trong phạm vi thực nghiệm của đồ án, đó là kết luận khách quan nhất có thể đưa ra.

## 4. TÀI LIỆU THAM KHẢO

[1] X. Yu, Y. Gong, N. Jiang, Q. Ye, and Z. Han, "Scale Match for Tiny Person Detection," WACV, 2020, pp. 1257–1265.
[2] S. Hong et al., "SSPNet: Scale Selection Pyramid Network for Tiny Person Detection from UAV Images," IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1–5, 2022.
[3] M. Chao, C. Peng, Z.-Q. Chen et al., "A lightweight small object detection model for UAV images based on deep semantic integration," Scientific Reports, vol. 15, art. 31888, 2025.
[4] X. Wang, S. Zhang, Z. Yu, L. Feng, and W. Zhang, "Scale-Equalizing Pyramid Convolution for Object Detection," CVPR, 2020.
[5] J. Wang, C. Xu, W. Yang, and L. Yu, "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection," arXiv preprint arXiv:2110.13389, 2021.
[6] C. Xu, J. Wang, W. Yang, and L. Yu, "Dot Distance for Tiny Object Detection in Aerial Images," CVPR Workshops, 2021.
[7] C. Xu, J. Wang, W. Yang, H. Yu, L. Yu, and G.-S. Xia, "RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection," ECCV, 2022.
[8] H. Sun, R. Wang, Y. Li, L. Yang, S. Lin, X. Cao, and B. Zhang, "SET: Spectral Enhancement for Tiny Object Detection," CVPR, 2025.
[9] F. Akyon, S. Altinuc, and A. Temizel, "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection," ICIP, 2022.
[10] C. Yang, Z. Huang, and N. Wang, "QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection," CVPR, 2022.
[11] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature Pyramid Networks for Object Detection," CVPR, 2017.
[12] S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia, "Path Aggregation Network for Instance Segmentation," CVPR, 2018.
[13] A. Witkin, "Scale-Space Filtering," IJCAI, 1983.
[14] J. Koenderink, "The Structure of Images," Biological Cybernetics, 1984.
[15] T. Lindeberg, "Scale-Space Theory in Computer Vision," Dordrecht, The Netherlands: Kluwer Academic Publishers, 1994.
[16] P. Burt and E. Adelson, "The Laplacian Pyramid as a Compact Image Code," IEEE Transactions on Communications, 1983.
[17] Ultralytics, "YOLO26: Real-Time End-to-End Object Detection Architecture," Technical Report, Ultralytics, 2025, https://docs.ultralytics.com/models/yolo26 [Truy cập ngày 10/11/2026].
[18] C. Feng et al., "TOOD: Task-aligned One-stage Object Detection," ICCV, 2021.
[19] Z. Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression," AAAI, 2020.
[20] W. Liu, H. Lu, H. Fu, and Z. Cao, "Learning to Upsample by Learning to Sample," ICCV, 2023.
[21] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," ICML, 2015.
[22] W. Shi et al., "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network," CVPR, 2016.
[23] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR, 2019.
[24] A. Neubeck and L. Van Gool, "Efficient Non-Maximum Suppression," ICPR, 2006.
[25] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO: Common Objects in Context," ECCV, 2014.
[26] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," NeurIPS, 2015.
[27] Z. Cai and N. Vasconcelos, "Cascade R-CNN: Delving into High Quality Object Detection," CVPR, 2018.
[28] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," ICCV, 2017.
[29] Z. Tian, C. Shen, H. Chen, and T. He, "FCOS: Fully Convolutional One-Stage Object Detection," ICCV, 2019.
[30] X. Li, W. Wang, L. Wu, S. Chen, X. Hu, G. Li, J. Tang, and J. Pan, "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection," NeurIPS, 2020.
[31] A. Bochkovskiy, C.-Y. Wang, and H.-Y. M. Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection," arXiv preprint arXiv:2004.10934, 2020.
[32] H. Zhang, M. Cissé, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," ICLR, 2018.
[33] S. Yun, D. Han, S. J. Oh, S. Chun, J. Choe, and Y. Yoo, "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features," ICCV, 2019.
[34] B. Zoph, I. Bello, K. Kumar, N. Du, B. Huang, J. Dean, N. Shazeer, and W. Fedus, "ST-MoE: Designing Stable and Transferable Sparse Expert Models," arXiv preprint arXiv:2202.08906, 2022.
[35] A. R. Smith, "Color Gamut Transform Pairs," ACM SIGGRAPH Computer Graphics, vol. 12, no. 3, pp. 12–19, 1978.
[36] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," Graphics Gems IV, Academic Press, pp. 474–485, 1994.

## PHỤ LỤC

### Phụ lục A. Cấu hình huấn luyện đầy đủ

**Bảng A.1: Cấu hình huấn luyện dùng chung cho hai cấu hình**

| Tham số | Giá trị |
|---|---|
| *Epoch* / *patience* | 150 / 40 (cả hai mô hình hoàn thành trọn vẹn 150 epoch mà không kích hoạt early stopping) |
| Đầu vào | 1280 × 1280, *letterbox* giữ tỉ lệ |
| Batch / *workers* | 16 / 8 |
| Bộ tối ưu | AdamW, lr₀ 0,002, lrf 0,1, cosine |
| *Warmup* | 3 *epoch* |
| Độ chính xác | FP16 (*amp*) |
| Tăng cường hình học | dịch 0,05, tỉ lệ 0,15, lật ngang 0,5; tắt xoay, méo, phối cảnh |
| Tăng cường màu | HSV h 0,005 / s 0,1 / v 0,15 |
| Mosaic / MixUp / CutMix / Copy-Paste | 0 / 0 / 0 / 0 |
| *Multi-scale* | tắt |
| *Seed* | 42 |

### Phụ lục B. Số kênh của bốn mức head ở hai cấu hình

Cả hai cấu hình dùng cùng một head với bốn mức đầu vào 32 / 64 / 128 / 256 kênh (P2/P3/P4/P5). Ở mức P3, cả hai đều nối 32 kênh của mức P2 đã hạ stride với 64 kênh của P3 thành 96 kênh, rồi một khối C3k2 đưa về 64 kênh cho head; con số 96 trong Hình 2.1 là số kênh của phép nối, không phải số kênh đầu vào head. Ở cấu hình nền, P2 lấy từ nhánh C3k2 stride 4; ở cấu hình có module, P2 lấy từ đầu ra 32 kênh của GauSS và nhánh C3k2 đó không còn đầu ra nào tiêu thụ nhưng vẫn được tính trong đồ thị. Nhánh này có mặt ở cả hai cấu hình với cùng thông số nên không nằm trong chênh lệch 36.878 tham số, và chi phí tính của nó có trong phép đo độ trễ của cả hai bên. Tích chập hạ stride 3 × 3 stride 2 ở mức P2 cũng giống nhau ở hai cấu hình: 32 → 32 kênh, 9.216 tham số.

### Phụ lục C. Tiền xử lý và quy ước đo

Ảnh đưa vào mạng qua *letterbox* về 1280 × 1280, giữ tỉ lệ, không kéo méo. Chế độ cắt mảnh lấy các mảnh 640 × 512 chồng 100 pixel từ ảnh gốc, mỗi mảnh vào mạng đúng kích thước đó. Chế độ "ảnh gốc" là cả khung *letterbox* về 1280 × 1280, suy luận một lần, khung dự đoán chiếu về tọa độ ảnh gốc. Các thang kích thước luôn đo trên tọa độ ảnh gốc. MR50 là log-average miss rate tại 9 điểm FPPI trong [10⁻², 10⁰].

### Phụ lục D. Phần mềm

Huấn luyện và suy luận dùng khung Ultralytics (nhánh YOLO26), PyTorch với CUDA trên GPU NVIDIA A10G; đánh giá bằng evaluator tự viết của đồ án; biểu đồ vẽ bằng Matplotlib. Toàn bộ mã, cấu hình và kịch bản sinh số liệu của báo cáo nằm trong kho mã của đồ án.
