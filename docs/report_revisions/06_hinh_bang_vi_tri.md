# Hình, bảng và vị trí chèn

File này dành cho công cụ chèn hình. Mỗi mục ghi số hình, nội dung, nguồn dữ
liệu, vị trí chèn theo heading và câu neo trong bản gốc. Không chèn hình nếu
chưa có câu nhắc trong văn bản.

## Quy ước

- Caption hình đặt dưới hình, bắt đầu `Hình X.Y.`.
- Caption bảng đặt trên bảng, bắt đầu `Bảng X.Y.`.
- Không dùng ảnh chụp công thức. Công thức phải là phương trình native.
- Biểu đồ dùng cùng thang đo giữa các nhánh. Ghi rõ đơn vị, số mẫu và split.
- Hình phải có nguồn dữ liệu ở cuối caption.
- Kích thước gợi ý tính theo chiều rộng vùng nội dung khoảng 16 cm.

## Hình cần chèn

| Hình | Nội dung | Nguồn | Vị trí | Kích thước |
|---|---|---|---|---|
| 1.1 | Histogram căn bậc hai diện tích hộp, đường tích lũy và bốn ảnh ví dụ single, sparse, crowded, very_crowded | Annotation TinyPerson và `samples.jsonl` | Sau Bảng 1.1, trước heading 1.2 | 15 × 9 cm |
| 1.2 | IoU theo độ dịch một trục cho hộp 4, 8, 16, 32 điểm ảnh | Công thức IoU | Sau đoạn ví dụ IoU 64 × 64 và 8 × 8, trước đoạn định nghĩa TP/miss/FP | 12 × 6 cm |
| 1.3 | Sơ đồ YOLO26n A2 với P2, P3, P4, P5, stride và vị trí BTRM trước Detect | `configs/model/yolo26n_tiny_A2_p2_btrm.yaml` | Sau Bảng 1.2, trước heading 1.4 | 16 × 8 cm |
| 2.1 | Sơ đồ bốn nhánh A2 scratch, BTRM scratch, A2 warm-start, BTRM warm-start và hai cặp so sánh | Bảng 2.1 | Sau Bảng 2.1, trước heading 2.3 | 16 × 7 cm |
| 2.2 | Sơ đồ rút gọn neck: P2 qua BTRM vào Detect; P3, P4, P5 đi thẳng vào Detect | `configs/model/yolo26n_tiny_A2_p2_btrm.yaml` | Sau đoạn cuối 2.3, trước heading 2.4 | 15 × 7 cm |
| 2.3 | Sơ đồ khối BTRM v2: X, M3, M5, Gx, Gy, evidence 5C, project, gate, gamma, Y | `src/tinydet/modules/btrm.py` | Sau Bảng 2.2, trước heading 2.5 | 16 × 8 cm |
| 3.1 | AP@0.25, AP@0.50, AP@0.75 của bốn nhánh | Bảng 3.4 | Sau Bảng 3.5, trước heading 3.5 | 14 × 7 cm |
| 3.2 | Stacked bar TP, miss, localization FP, background FP, duplicate FP | Bảng 3.6 | Sau Bảng 3.6, trước heading 3.6 | 16 × 7 cm |
| 3.3 | AP@0.75 theo tiny1, tiny2, tiny3, small, reasonable | Bảng 3.7 | Sau Bảng 3.7, trước Bảng 3.8 | 16 × 7 cm |
| 3.4 | Box plot width ratio và center offset cho tiny1, tiny2, tiny3 | Bảng 3.9 và `errors.jsonl` | Sau Bảng 3.9, trước Bảng 3.10 | 16 × 8 cm |
| 3.5 | Precision và recall theo ngưỡng 0,001 đến 0,5 | Bảng 3.10 | Sau Bảng 3.10, trước heading 3.8 | 14 × 7 cm |
| 3.6 | Latency, FPS và bộ nhớ của A2, BTRM scratch, BTRM warm-start | Bảng 3.13 | Sau Bảng 3.13, trước đoạn `Kết quả phân rã lỗi` | 14 × 6 cm |

## Chi tiết từng hình

### Hình 1.1. Phân bố kích thước và mật độ

**Vị trí:** sau Bảng 1.1, trước heading `1.2. Giới hạn biểu diễn và sai số định
vị`.

**Nội dung:** panel trái là histogram căn bậc hai diện tích hộp trên toàn bộ
GT TinyPerson, có đường tích lũy. Đánh dấu các mốc 8, 12, 20 và 32 điểm ảnh.
Panel phải là bốn ảnh ví dụ, mỗi ảnh có khung GT và nhãn mật độ.

**Nguồn:** annotation TinyPerson và `samples.jsonl`. Nếu không có ảnh ví dụ đã
được duyệt, bỏ panel phải và chỉ giữ histogram.

**Caption:**

> Hình 1.1. Phân bố kích thước hộp GT trên TinyPerson và bốn mức mật độ.
> Đường dọc đánh dấu các ngưỡng 8, 12, 20 và 32 điểm ảnh. Nguồn: annotation
> TinyPerson.

### Hình 1.2. IoU theo độ dịch tâm

**Vị trí:** sau đoạn kết thúc bằng `dưới ngưỡng đánh giá dương tiêu chuẩn (IoU
= 0,50)`, trước đoạn bắt đầu `Trong thực nghiệm, nhóm phân tách sai số`.

**Nội dung:** trục x là độ dịch tâm theo một trục, từ 0 đến 4 điểm ảnh. Trục y
là IoU. Bốn đường cho hộp 4, 8, 16 và 32 điểm ảnh. Vẽ đường ngang 0,50 và
0,75.

**Caption:**

> Hình 1.2. IoU theo độ dịch tâm với bốn kích thước hộp. Cùng một độ dịch làm
> IoU giảm nhanh hơn khi hộp nhỏ. Nguồn: tính từ công thức IoU.

### Hình 1.3. Kiến trúc YOLO26n A2

**Vị trí:** sau Bảng 1.2, trước heading `1.4. Động lực nghiên cứu từ miền tần
số`.

**Nội dung:** sơ đồ backbone, neck và Detect. Ghi rõ P2 layer 19, P3 layer 22,
P4 layer 25, P5 layer 28. BTRM nằm giữa layer 19 và Detect. Ghi stride 4, 8,
16, 32.

**Caption:**

> Hình 1.3. Kiến trúc YOLO26n A2 với bốn nhánh đặc trưng. BTRM v2 được đặt
> trên P2 trước Detect. Nguồn: `configs/model/yolo26n_tiny_A2_p2_btrm.yaml`.

### Hình 2.1. Thiết kế paired

**Vị trí:** sau Bảng 2.1, trước heading `2.3. Vị trí tích hợp trên P2`.

**Nội dung:** bốn hộp theo hai hàng. Hàng scratch: A2 và BTRM, cùng khởi tạo
ngẫu nhiên. Hàng warm-start: A2 và BTRM, cùng nạp một checkpoint A2. Mũi tên
so sánh nằm ngang trong từng hàng. Ghi rõ BTRM warm-start có gamma bằng 0.

**Caption:**

> Hình 2.1. Hai cặp so sánh paired. Cặp warm-start kiểm soát khác biệt khởi
> tạo bằng cách cho hai nhánh cùng nạp một checkpoint A2. Nguồn: thiết kế thực
> nghiệm.

### Hình 2.2. Vị trí BTRM trong mạng

**Vị trí:** sau đoạn cuối 2.3 kết thúc bằng `Một thử nghiệm quét vị trí trên P3
hoặc P4 sẽ là nghiên cứu riêng.`, trước heading `2.4`.

**Nội dung:** sơ đồ rút gọn của neck. P2 đi qua BTRM rồi vào Detect. P3, P4,
P5 đi thẳng vào Detect. Ghi tensor `B × 32 × H × W` trước và sau BTRM. Ghi
Detect nhận `[29, 22, 25, 28]`.

**Caption:**

> Hình 2.2. Vị trí BTRM v2 trên nhánh P2. Mô-đun giữ nguyên kích thước tensor
> và chỉ thay đổi P2 trước Detect. Nguồn: `src/tinydet/modules/btrm.py`.

### Hình 2.3. Sơ đồ khối BTRM v2

**Vị trí:** sau Bảng 2.2, trước heading `2.5. Phép chiếu, cổng và phần dư`.

**Nội dung:** X tách thành hai nhánh: AvgPool 3 × 3, AvgPool 5 × 5 và hai nhánh
gradient. Năm tensor ghép thành E 160 kênh. E qua Conv 1 × 1, BN, SiLU, Conv
1 × 1 thành P. P qua Conv 1 × 1 và sigmoid thành S. Đầu ra `Y = X + gamma × S
× P`. Ghi gamma khởi tạo 0.

**Caption:**

> Hình 2.3. Sơ đồ BTRM v2. Năm tín hiệu cục bộ được chiếu, điều chế bằng cổng
> sigmoid và cộng phần dư vào đầu vào. Nguồn: `src/tinydet/modules/btrm.py`.

### Hình 3.1. AP theo ngưỡng IoU

**Vị trí:** sau Bảng 3.5, trước heading `3.5. Phân rã TP, miss và FP`.

**Nội dung:** grouped bar với ba nhóm AP@0.25, AP@0.50, AP@0.75. Bốn màu cho
A2 scratch, BTRM scratch, A2 warm-start, BTRM warm-start. Ghi giá trị trên
đầu cột. Trục y từ 0 đến 0,45.

**Caption:**

> Hình 3.1. AP theo ngưỡng IoU trên official test. BTRM giảm AP ở cả ba ngưỡng
> trong cặp scratch và giảm AP@0.75 trong cặp warm-start. Nguồn: Bảng 3.4.

### Hình 3.2. Phân rã TP, miss và FP

**Vị trí:** sau Bảng 3.6, trước heading `3.6. Kết quả theo kích thước`.

**Nội dung:** stacked bar cho bốn nhánh. Thành phần gồm TP, miss, localization
FP, background FP, duplicate FP. Vì FP rất lớn so với TP, dùng trục y log
hoặc tách hai panel. Panel trái cho TP và miss, panel phải cho ba loại FP.

**Caption:**

> Hình 3.2. Phân rã TP, miss và FP trên official test. Trong cặp scratch, BTRM
> giảm localization FP nhưng tăng background FP và giảm TP. Nguồn: Bảng 3.6.

### Hình 3.3. AP@0.75 theo dải kích thước

**Vị trí:** sau Bảng 3.7, trước Bảng 3.8.

**Nội dung:** grouped bar cho tiny1, tiny2, tiny3, small, reasonable. Bốn màu
theo bốn nhánh. Ghi giá trị. Trục y từ 0 đến 0,05.

**Caption:**

> Hình 3.3. AP@0.75 theo dải kích thước. Trong cặp scratch, BTRM tăng tiny1
> và tiny2 nhưng giảm tiny3 và small. Nguồn: Bảng 3.7.

### Hình 3.4. Hình học hộp lỗi

**Vị trí:** sau Bảng 3.9, trước Bảng 3.10.

**Nội dung:** hai panel box plot. Panel trái là width ratio, panel phải là
center offset. Mỗi panel có ba nhóm tiny1, tiny2, tiny3 và ba màu A2, BTRM
scratch, BTRM warm-start. Vẽ đường tham chiếu 1,0 cho width ratio.

**Caption:**

> Hình 3.4. Phân bố width ratio và center offset của localization FP. Hộp dự
> đoán trung vị hẹp hơn GT; BTRM không giảm center offset. Nguồn: Bảng 3.9 và
> `errors.jsonl`.

### Hình 3.5. Precision và recall theo confidence

**Vị trí:** sau Bảng 3.10, trước heading `3.8. Kiểm tra hiện thực và ablation
gamma`.

**Nội dung:** hai panel. Panel trái là recall theo ngưỡng, panel phải là
precision theo ngưỡng. Bốn đường cho A2 scratch, BTRM scratch, A2 warm-start
và BTRM warm-start. Trục x log scale từ 0,001 đến 0,5.

**Caption:**

> Hình 3.5. Recall và precision theo ngưỡng confidence. BTRM scratch thua A2
> về recall ở mọi ngưỡng; BTRM warm-start gần như trùng đối chứng. Nguồn:
> Bảng 3.10.

### Hình 3.6. Chi phí tính toán

**Vị trí:** sau Bảng 3.13, trước đoạn `Kết quả phân rã lỗi và hình học hộp`.

**Nội dung:** ba panel nhỏ: latency, FPS, bộ nhớ. Mỗi panel có ba cột A2, BTRM
scratch, BTRM warm-start. Ghi giá trị và phần trăm chênh lệch.

**Caption:**

> Hình 3.6. Chi phí suy diễn trên L4, batch 1, ảnh 1.280. BTRM tăng latency
> và bộ nhớ nhưng không cải thiện AP@0.75. Nguồn: Bảng 3.13.

## Hình cũ cần xử lý

| Hình cũ | Nội dung | Cách xử lý |
|---|---|---|
| image1 | Logo bìa | Giữ |
| image2 | Công thức IoU | Thay bằng phương trình native, không dùng ảnh |
| image3 | Sơ đồ BTRM P3 | Thay bằng Hình 2.2 mới |
| image4, image5 | Công thức tám điểm và nền bậc nhất | Xóa |
| image6 đến image11 | Công thức Tikhonov, phương sai, điều chế | Xóa |
| image12 | Biểu đồ AP@0.50 cũ | Thay bằng Hình 3.1 |
| image13 | Biểu đồ AP trung bình COCO cũ, khác APmean hiện tại | Thay bằng Hình 3.1 hoặc bỏ |

## Bảng cần có

| Bảng | Nội dung | Vị trí |
|---|---|---|
| 1.1 | TinyPerson gốc, derivative và official test | Mục 1.1 |
| 1.2 | Bản đồ đặc trưng P2 đến P5 | Mục 1.3 |
| 1.3 | So sánh hướng tiếp cận | Mục 1.6 |
| 2.1 | Thiết kế paired bốn nhánh | Mục 2.2 |
| 2.2 | Tensor trong BTRM v2 | Mục 2.4 |
| 2.3 | Checklist tái lập | Mục 2.7 |
| 3.1 | Run và artifact | Mục 3.1 |
| 3.2 | Dữ liệu và protocol | Mục 3.2 |
| 3.3 | Baseline A2 trên hai split | Mục 3.3 |
| 3.4 | Kết quả bốn nhánh | Mục 3.4 |
| 3.5 | Chênh lệch theo cặp | Mục 3.4 |
| 3.6 | Phân rã TP, miss, FP | Mục 3.5 |
| 3.7 | AP@0.75 theo dải | Mục 3.6 |
| 3.8 | Recall theo dải | Mục 3.6 |
| 3.9 | Hình học localization FP | Mục 3.7 |
| 3.10 | Recall và precision theo confidence | Mục 3.7 |
| 3.11 | Ablation gamma | Mục 3.8 |
| 3.12 | Phân rã lỗi gamma 0 | Mục 3.8 |
| 3.13 | Chi phí tính toán | Mục 3.9 |
| 4.1 | Giả thuyết và bằng chứng | Mục 4.1 |

## Thứ tự chèn

1. Chèn Hình 1.1 đến 1.3 và Bảng 1.1 đến 1.3.
2. Chèn Hình 2.1 đến 2.3 và Bảng 2.1 đến 2.3.
3. Chèn Hình 3.1 đến 3.6 và Bảng 3.1 đến 3.13.
4. Chèn Bảng 4.1.
5. Cập nhật danh mục hình, danh mục bảng và mục lục.

## Kiểm tra sau khi chèn

- Mỗi hình có ít nhất một câu nhắc trước khi xuất hiện.
- Không có hình nào nằm giữa hai đoạn của cùng một ý.
- Caption không bị tách khỏi hình.
- Bảng không vượt quá chiều rộng trang. Bảng 3.9 và 3.10 nên đặt ở chế độ
  ngang hoặc giảm cỡ chữ còn 8 pt.
- Số hình và số bảng trong danh mục khớp với thân bài.
