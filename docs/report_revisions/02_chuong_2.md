# Chương 2. Phương pháp đề xuất

## Chẩn đoán chương hiện tại

Chương 2 hiện có khoảng 2.574 từ, 7 mục và nhiều tiểu mục. Độ dài chưa đủ để
người đọc hiểu và tái lập mô-đun. Ba vấn đề chính:

1. Mô tả BTRM P3 cũ, không khớp mã nguồn hiện tại là BTRM v2 trên P2.
2. Có đoạn lặp nguyên văn trong 2.4 và 2.2.
3. Các công thức được chèn dưới dạng ảnh. Không thể tìm kiếm, không đánh số
   lại được và dễ lệch khi chuyển định dạng.

Chương 2 sau khi sửa cần trả lời bốn câu hỏi:

- BTRM v2 nhận đầu vào nào, đặt ở đâu và thay đổi tensor ra sao?
- Mô-đun tính những tín hiệu gì và kết hợp chúng như thế nào?
- Hai nhánh đối chứng được huấn luyện giống và khác nhau ở điểm nào?
- Phép đánh giá nào đủ để trả lời câu hỏi localization?

## Cấu trúc đề xuất

| Mục | Nội dung | Độ dài mục tiêu |
|---|---|---:|
| 2.1 | Câu hỏi nghiên cứu và giả thuyết | 1,5 trang |
| 2.2 | Thiết kế paired và kiểm soát khởi tạo | 2,5 trang |
| 2.3 | Vị trí tích hợp trên P2 | 2 trang |
| 2.4 | BTRM v2: tín hiệu cục bộ | 3 trang |
| 2.5 | Phép chiếu, cổng và phần dư | 2,5 trang |
| 2.6 | Dữ liệu, huấn luyện và đánh giá | 5 trang |
| 2.7 | Bất biến hiện thực và tái lập | 2,5 trang |
| 2.8 | Đe dọa đến tính hợp lệ | 1,5 trang |

## 2.1. Câu hỏi nghiên cứu và giả thuyết

**Vị trí:** thay toàn bộ mục 2.1 hiện tại, từ heading đến trước heading 2.2.

**Vấn đề:** mục hiện tại lặp lại H01 đã phát biểu ở 1.5, dài 537 từ nhưng không
nêu câu hỏi nghiên cứu thành danh sách. Người đọc không biết chương 3 sẽ trả
lời câu nào trước.

**Nội dung thay thế:**

> Chương này mô tả thiết kế thực nghiệm để kiểm tra H01 ở cấp mô hình và tìm
> hiểu tầng lỗi còn lại khi mô-đun không cải thiện AP@0.75. Ba câu hỏi được
> dùng xuyên suốt báo cáo.
>
> Q1. YOLO26n A2 đạt mức nào trên TinyPerson khi giữ nguyên protocol của repo?
> Câu trả lời gồm AP@0.25, AP@0.50, AP@0.75, APmean, precision và
> recall trên validation và official test.
>
> Q2. BTRM v2 có cải thiện AP@0.75 và recall của dải tiny, tức là căn bậc hai
> diện tích hộp từ 1 đến 20 điểm ảnh, hay không? Câu trả lời được kiểm tra
> trên hai cặp: huấn luyện scratch và warm-start từ cùng checkpoint A2.
>
> Q3. Nếu BTRM v2 không cải thiện AP@0.75, lỗi còn lại nằm ở đâu trong cấu
> trúc TP, miss và FP? Phân tích này dùng chung primary IoU 0,50 và tách FP
> thành localization FP, background FP và duplicate FP.
>
> H01 được phát biểu ở mục 1.5. Báo cáo không đặt lại giả thuyết ở đây.
>
> Để diễn giải phép so sánh, báo cáo xem xét AP@0.75 trên validation và official
> test cùng recall tiny1 trên official test. Đây là các phép đo dùng để thảo
> luận kết quả, không phải ngưỡng quyết định đăng ký trước. Chưa có tài liệu
> chứng minh tiêu chí được chốt trước khi xem kết quả, nên không trình bày
> phần này như một kiểm định tiên nghiệm hay bằng chứng thống kê.

## 2.2. Thiết kế paired và kiểm soát khởi tạo

**Vị trí:** thay toàn bộ mục 2.2 và Bảng 2.1 hiện tại.

**Vấn đề:** Bảng 2.1 liệt kê A0 đến A5 nhưng chỉ A2 và BTRM v2 có kết quả
trong protocol hiện tại. A0, A1, A3 cũ thuộc protocol khác. Trình bày A0-A5
như một ma trận hoàn chỉnh sẽ khiến người đọc hiểu sai rằng tất cả đã chạy.

**Bảng thay thế:**

> Bảng 2.1. Thiết kế paired cho BTRM v2 trên TinyPerson.

| Nhánh | Khởi tạo | Thay đổi kiến trúc | Vai trò | Run |
|---|---|---|---|---|
| A2 scratch | Ngẫu nhiên, seed 42 | Không | Đối chứng scratch | `tiny_A2_p2` |
| BTRM scratch | Ngẫu nhiên, seed 42 | Thêm BTRM v2 trên P2 | Treatment scratch | `tiny_A2_p2_btrm` |
| A2 warm-start | `best.pt` của A2 scratch | Không | Đối chứng warm-start | `tiny_A2_p2_adapt` |
| BTRM warm-start | Cùng `best.pt` của A2 scratch | Thêm BTRM v2, gamma bằng 0 | Treatment warm-start | `tiny_A2_p2_btrm_adapt` |

**Đoạn văn thay thế:**

> Thiết kế có hai cặp so sánh. Cặp scratch kiểm tra toàn bộ quy trình huấn
> luyện khi hai nhánh khởi tạo ngẫu nhiên. Cặp warm-start kiểm tra tác động
> của mô-đun khi hai nhánh bắt đầu từ cùng một checkpoint A2. Cặp thứ hai cần
> thiết vì việc chèn BTRM trước Detect làm thay đổi chuỗi số ngẫu nhiên khởi
> tạo của đầu phát hiện. Chênh lệch ở cặp scratch vì vậy không phải ước lượng
> nhân quả sạch cho riêng mô-đun.
>
> Trong cặp warm-start, cả hai nhánh nạp cùng checkpoint
> `tiny_A2_p2/20261019-194809-8326468-seed42/checkpoints/best.pt` và
> fine-tune 10 epoch với `lr0=0.0002`. BTRM v2 khởi tạo gamma bằng 0, nên tại
> bước cập nhật đầu tiên đầu ra của hai nhánh giống nhau. Đây là điều kiện để
> phép so sánh tập trung vào phần cập nhật do mô-đun tạo ra, không phải khác
> biệt khởi tạo.
>
> A0, A1 và A3 cũ không được đưa vào bảng so sánh. Ba cấu hình đó dùng
> protocol khác, đặc biệt ở batch, mosaic, patience và phiên bản mô-đun. Trộn
> chúng với kết quả hiện tại sẽ tạo ra so sánh không paired. Kết quả cũ chỉ
> được nhắc riêng như một ghi chú lịch sử nếu cần, không đưa vào bảng paired.

**Hình cần chèn:** Hình 2.1, sơ đồ bốn nhánh và hai cặp so sánh. Vị trí sau
Bảng 2.1, trước heading 2.3. Xem `06_hinh_bang_vi_tri.md`.

## 2.3. Vị trí tích hợp trên P2

**Vị trí:** thay toàn bộ mục 2.3, từ heading đến trước heading 2.4.

**Vấn đề:** mục hiện tại mô tả P3, stride 8, vùng lân cận 24 × 24 điểm ảnh và
caption hình bị lỗi font. Mã nguồn hiện tại đặt BTRM trên P2, layer 29, stride
4.

**Nội dung thay thế:**

> YOLO26n A2 trong repo xuất bốn bản đồ đặc trưng P2, P3, P4 và P5 với stride
> lần lượt 4, 8, 16 và 32. Với đầu vào 1.280 × 1.280, kích thước không gian
> tương ứng là 320 × 320, 160 × 160, 80 × 80 và 40 × 40. Một người cao 8 điểm
> ảnh chỉ chiếm khoảng hai ô trên P2 và khoảng một ô trên P3. Vì vậy P2 là
> nhánh còn giữ được nhiều vị trí không gian nhất cho nhóm tiny1.
>
> BTRM v2 được chèn vào nhánh P2, ngay trước Detect. Trong đồ thị model, BTRM
> là layer 29, nhận đầu ra của layer 19 và truyền sang Detect. Detect nhận
> bốn nhánh theo thứ tự `[29, 22, 25, 28]`, tương ứng P2, P3, P4 và P5. Các
> nhánh P3, P4, P5, hàm mất mát và quy tắc gán nhãn giữ nguyên.
>
> Mô-đun giữ nguyên kích thước tensor `[B, 32, H, W]`. Với cấu hình A2, BTRM
> v2 thêm 2.609 tham số, nâng tổng số tham số từ 2.515.976 lên 2.518.585.
> Phần lớn tham số nằm ở phép chiếu 1 × 1 từ 160 kênh xuống 32 kênh và ở cổng
> 1 × 1.
>
> Vị trí này không phải kết quả của một phép tìm kiếm kiến trúc. Nó được chọn
> vì P2 là nhánh duy nhất còn giữ được nhiều hơn một ô cho tiny1. Báo cáo
> không khẳng định P2 là lựa chọn tối ưu. Một thử nghiệm quét vị trí trên P3
> hoặc P4 sẽ là nghiên cứu riêng.

**Hình cần chèn:** Hình 2.2, sơ đồ rút gọn neck với P2 đi qua BTRM rồi vào
Detect, còn P3, P4, P5 đi thẳng vào Detect. Vị trí sau đoạn cuối mục 2.3,
trước heading 2.4. Không lặp lại sơ đồ kiến trúc đầy đủ của Hình 1.3.

## 2.4. BTRM v2: tín hiệu cục bộ

**Vị trí:** thay toàn bộ mục 2.4, gồm đoạn tám điểm vòng, Bảng 2.2 và các công
thức 2.1, 2.2.

**Vấn đề:** toàn bộ mục mô tả toán tử tám điểm và mô hình nền bậc nhất không
còn trong mã nguồn. Hai đoạn gần giống nhau bị lặp.

**Nội dung thay thế:**

> Ký hiệu `X` là bản đồ đặc trưng đầu vào có kích thước `B × C × H × W`, với
> `C = 32` tại P2. BTRM v2 không ước lượng một mặt nền bậc nhất. Mô-đun xây
> dựng năm bản đồ tín hiệu cục bộ từ `X`.
>
> Hai bản đồ tương phản gần và xa:
>
> \[
> M_3 = \operatorname{AvgPool}_{3 \times 3}(X), \qquad
> M_5 = \operatorname{AvgPool}_{5 \times 5}(X)
> \]
>
> Hai bản đồ gradient theo phương ngang và phương dọc, dùng sai phân trung tâm:
>
> \[
> G_x = \frac{1}{2}\left(X_{h,w+1,c} - X_{h,w-1,c}\right), \qquad
> G_y = \frac{1}{2}\left(X_{h+1,w,c} - X_{h-1,w,c}\right)
> \]
>
> và độ lớn gradient:
>
> \[
> |G| = |G_x| + |G_y|
> \]
>
> Năm tín hiệu được ghép theo kênh:
>
> \[
> E = \left[X - M_3,\; X - M_5,\; G_x,\; G_y,\; |G|\right]
> \]
>
> `E` có `5C` kênh. Hai hiệu `X - M_3` và `X - M_5` đo mức khác biệt giữa điểm
> tâm và vùng lân cận ở hai bán kính. `G_x` và `G_y` đo biên theo hai hướng.
> `|G|` đo độ mạnh biên không phụ thuộc dấu.
>
> BTRM v2 không dùng tám điểm rời rạc. Cả hai phép trung bình đều dùng
> replicate padding và giữ nguyên kích thước không gian. Cách làm này tránh
> tạo giá trị 0 giả ở biên ảnh và cho phép áp dụng cho toàn bản đồ P2.

**Bảng cần chèn:** Bảng 2.2, kích thước tensor của từng tín hiệu. Vị trí sau
đoạn cuối mục 2.4, trước heading 2.5.

> Bảng 2.2. Tensor trong BTRM v2 với đầu vào P2.

| Ký hiệu | Công thức | Kích thước | Ý nghĩa |
|---|---|---|---|
| `X` | Đầu vào | `B × 32 × H × W` | Đặc trưng P2 |
| `M_3` | AvgPool 3 × 3 | `B × 32 × H × W` | Nền gần |
| `M_5` | AvgPool 5 × 5 | `B × 32 × H × W` | Nền xa |
| `G_x` | Sai phân ngang | `B × 32 × H × W` | Biên dọc |
| `G_y` | Sai phân dọc | `B × 32 × H × W` | Biên ngang |
| `E` | Ghép kênh | `B × 160 × H × W` | Evidence |
| `P` | Chiếu 1 × 1 | `B × 32 × H × W` | Phần dư đề xuất |
| `S` | Sigmoid cổng | `B × 32 × H × W` | Cổng theo kênh và vị trí |
| `Y` | Đầu ra | `B × 32 × H × W` | Đặc trưng đã điều chế |

**Hình cần chèn:** Hình 2.3, sơ đồ khối BTRM v2. Vị trí sau Bảng 2.2, trước
heading 2.5.

## 2.5. Phép chiếu, cổng và phần dư

**Vị trí:** thay toàn bộ mục 2.5, gồm Tikhonov, phương sai cục bộ, hệ số điều
chế bị chặn và các công thức 2.3 đến 2.8.

**Vấn đề:** các thành phần này không có trong `src/tinydet/modules/btrm.py`.
Giữ lại sẽ khiến chương 2 mô tả một mô-đun không tồn tại.

**Nội dung thay thế:**

> Evidence `E` được chiếu bằng hai tích chập 1 × 1:
>
> \[
> P = W_2\,\operatorname{SiLU}\left(\operatorname{BN}\left(W_1 E\right)\right)
> \]
>
> Trong đó `W_1` ánh xạ `5C` kênh xuống `C/4` kênh, `W_2` ánh xạ `C/4` kênh
> trở lại `C` kênh. Với `C = 32`, lớp ẩn có 8 kênh. BatchNorm và SiLU giúp ổn
> định phân bố của evidence trước khi đưa vào cổng.
>
> Cổng được tính từ chính `P`:
>
> \[
> S = \sigma\left(\operatorname{Conv}_{1 \times 1}(P)\right)
> \]
>
> `S` có cùng kích thước với `P`, nên cổng thay đổi theo từng kênh và từng vị
> trí không gian. Đầu ra cuối cùng:
>
> \[
> Y = X + \gamma\, S \odot P
> \]
>
> `gamma` là một tham số vô hướng, khởi tạo bằng 0. Khi `gamma = 0`, `Y = X`
> chính xác, không phải xấp xỉ. Điều này cho phép huấn luyện bắt đầu từ đúng
> hàm của baseline A2 và tránh sốc gradient cho Detect.
>
> BTRM v2 không chuẩn hóa theo phương sai cục bộ và không chặn `S` trong một
> khoảng cố định. Cơ chế kiểm soát duy nhất là cổng sigmoid và gamma. Vì vậy
> mô-đun không được mô tả như một bộ lọc CFAR. Nếu báo cáo muốn so sánh với
> CFAR, phần so sánh phải nằm ở nghiên cứu liên quan, không nằm trong mô tả
> hiện thực.
>
> Giá trị gamma sau huấn luyện và kết quả ép gamma bằng 0 được báo ở mục 3.8,
> không dùng để lựa chọn lại thiết kế mô-đun từ kết quả test.

## 2.6. Dữ liệu, huấn luyện và đánh giá

**Vị trí:** thay toàn bộ mục 2.6 hiện tại. Chuyển các chi tiết protocol từ
chương 3 vào đây, chỉ giữ kết quả ở chương 3.

**Nội dung thay thế:**

> Dữ liệu huấn luyện là bản derivative YOLO của TinyPerson official_v1. Bản
> derivative có 746 ảnh, gồm 662 ảnh huấn luyện và 84 ảnh validation. Nhãn
> `sea_person` và `earth_person` được gộp thành một lớp `person`. Tập validation
> dùng để chọn checkpoint. Tập official test chỉ dùng để báo cáo kết quả cuối,
> không dùng để chọn mô hình.
>
> Official test có 816 ảnh. Theo protocol TinyPerson, 30 ảnh dense bị loại,
> còn 786 ảnh với 13.687 khung tham chiếu dương tính. Ảnh dense được xác định
> theo trường `in_dense_image` hoặc theo ngưỡng hơn 220 annotation trên một
> ảnh. Việc loại ảnh dense phải được nêu rõ vì nó làm thay đổi mẫu số của mọi
> metric.
>
> Hai nhánh dùng cùng siêu tham số: seed 42, 50 epoch, patience 15, ảnh đầu
> vào 1.280 × 1.280, batch 8, AdamW, `lr0=0.001`, AMP bật và deterministic
> bật. Tăng cường dữ liệu gồm lật ngang 0,5, dịch 0,05, scale 0,15 và thay
> đổi HSV nhẹ. Mosaic, mixup, cutmix và copy-paste đều bằng 0. Hai nhánh chỉ
> khác nhau ở sự có mặt của BTRM v2.
>
> Checkpoint được chọn theo validation. Metric báo cáo lấy từ `best.pt`, không
> lấy từ epoch cuối. Với official test, dự đoán được tạo ở `imgsz=1280`,
> `conf=0.001` và `max_det=200`. Cùng một checkpoint, cùng tham số suy diễn và
> cùng evaluator được dùng cho cả hai nhánh.
>
> Chỉ số được ưu tiên phân tích trong báo cáo là AP@0.75. Các chỉ số khác
> gồm AP@0.25, AP@0.50, APmean, precision, recall và phân rã TP, miss, FP.
> APmean là trung bình của
> AP@0.25, AP@0.50 và AP@0.75. FP được tách
> thành ba loại. Localization FP là dự đoán có IoU với GT trong khoảng
> [0,10; 0,50) với GT dương tính cùng lớp. Background FP là dự đoán có IoU
> dưới 0,10 với mọi GT dương tính cùng lớp sau khi đã loại dự đoán trung tính.
> Duplicate FP có IoU ít nhất 0,50 với một GT dương tính nhưng GT đó đã được
> khớp trước theo thứ tự confidence. Các dự đoán khớp vùng ignore/uncertain,
> ngoài dải kích thước hoặc ảnh dense bị loại không tính như background FP.
>
> AP@0.75 giúp xem xét chất lượng định vị ở ngưỡng IoU cao; AP@0.50 và
> recall giúp đối chiếu khả năng phát hiện. Không có bằng chứng rằng thứ tự
> ưu tiên metric này đã được đăng ký trước thực nghiệm.
>
> Evaluator là implementation nội bộ của repo. Nó chưa được pin với evaluator
> tham chiếu của TinyPerson. Vì vậy các số AP trong báo cáo chỉ dùng để so
> sánh các nhánh trong cùng protocol, không dùng để so với bảng công bố.

## 2.7. Bất biến hiện thực và tái lập

**Vị trí:** thay toàn bộ mục 2.7. Giữ Bảng checklist nhưng cập nhật nội dung
theo BTRM v2.

**Nội dung thay thế:**

> Mô-đun có bảy bất biến cần kiểm tra trước khi huấn luyện. Các bất biến này
> được kiểm tra tự động trong `tests/test_btrm.py`.
>
> 1. Với gamma bằng 0, đầu ra bằng đầu vào tuyệt đối.
> 2. Với đầu vào hằng số, hai hiệu `X - M_3` và `X - M_5` bằng 0, gradient
>    bằng 0 và phần dư bằng 0.
> 3. Đầu ra giữ nguyên kích thước `[B, 32, H, W]`.
> 4. Gradient chảy qua gamma, phép chiếu và cổng.
> 5. Autocast trên CPU giữ dtype đầu ra.
> 6. Đồ thị model có BTRM ở layer 29 và Detect nhận `[29, 22, 25, 28]`.
> 7. Checkpoint lưu và nạp lại được, kể cả khi nạp trọng số Detect từ
>    baseline.
>
> Bảng 2.3. Checklist tái lập tối thiểu.

| Thành phần | Nội dung phải cố định | Nơi lưu |
|---|---|---|
| Kiến trúc | P2, layer 29, thứ tự `[29, 22, 25, 28]` | `configs/model/yolo26n_tiny_A2_p2_btrm.yaml` |
| Toán tử | 3 × 3, 5 × 5, sai phân trung tâm, replicate padding | `src/tinydet/modules/btrm.py` |
| Khởi tạo | gamma bằng 0 | `tests/test_btrm.py` |
| Huấn luyện | seed, epoch, batch, optimizer, augmentation | `configs/experiment/localization_btrm_tinyperson.yaml` |
| Đánh giá | split, checkpoint, conf, max_det, IoU | `docs/BTRM_P2.md` |
| Artifact | run ID, metric, error analysis | `docs/BTRM_REPORT.md` |

## 2.8. Đe dọa đến tính hợp lệ

**Vị trí:** thêm mục mới sau 2.7, trước chương 3.

**Nội dung thay thế:**

> Có bốn đe dọa chính. Thứ nhất, kết quả chỉ có một seed. Chênh lệch AP@0.75
> khoảng 0,003 đến 0,005 có thể thay đổi nếu chạy seed khác. Báo cáo không
> dùng kiểm định thống kê và không tuyên bố mức chênh lệch có ý nghĩa.
>
> Thứ hai, evaluator là implementation nội bộ. Cross-check giữa hai
> implementation nội bộ khớp tuyệt đối, nhưng chưa có so sánh với evaluator
> tham chiếu. Các số AP không nên dùng để so với công bố bên ngoài.
>
> Thứ ba, bản derivative YOLO không giữ đầy đủ metadata ignore, uncertain và
> dense của TinyPerson. Validation trên derivative vì vậy chỉ dùng để chọn
> checkpoint, không dùng để kết luận về benchmark chính thức.
>
> Thứ tư, BTRM v2 chỉ được thử trên TinyPerson. Chưa có kết quả trên VisDrone
> hoặc một bộ dữ liệu người nhỏ khác. Kết luận không được mở rộng ra ngoài
> miền dữ liệu này.
