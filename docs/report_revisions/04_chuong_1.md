# Chương 1. Tổng quan nghiên cứu

## Chẩn đoán chương hiện tại

Chương 1 có khoảng 6.315 từ. Độ dài không thiếu nhưng phân bổ không đều:

- 1.1: 623 từ, ba đoạn, đoạn đầu 275 từ.
- 1.2: 784 từ, bốn đoạn.
- 1.3: 821 từ, bốn đoạn, chỉ mô tả P3, P4, P5.
- 1.4: 837 từ, bốn đoạn, phần miền tần số hơi xa trọng tâm.
- 1.5: 750 từ, sáu đoạn, có hai định nghĩa và H01.
- 1.6: 2.307 từ, bảy đoạn, dài gấp ba lần các mục khác và lặp ý.
- Đóng góp: 193 từ, vẫn nói A0, A2, A3.

Mục tiêu sau khi sửa: 11 đến 13 trang, mỗi mục 700 đến 1.000 từ, 1.6 còn
khoảng 1.300 từ. Không cắt nội dung kỹ thuật cần thiết. Cắt các câu lặp và
thay bằng số liệu, bảng, hình.

## 1.1. Tổng quan về phát hiện người kích thước cực nhỏ

**Vị trí:** mục 1.1, ba đoạn hiện tại.

**Vấn đề:** đoạn đầu dài 275 từ, gộp ba ý: ứng dụng, thống kê TinyPerson và so
sánh với COCO. Đoạn hai lặp lại vai trò của nền. Đoạn ba mô tả phạm vi báo cáo
nhưng không nêu câu hỏi nghiên cứu.

**Cách sửa:** tách đoạn đầu thành ba đoạn. Thêm Bảng 1.1 và Hình 1.1. Giữ các
số liệu TinyPerson gốc, nhưng thêm số liệu bản derivative và official test.

**Cấu trúc thay thế:**

> Đoạn 1. Ứng dụng và lý do bài toán quan trọng. Nêu ba kịch bản: tìm kiếm cứu
> nạn trên biển, camera giám sát bờ biển, thiết bị bay không người lái. Không
> dùng các cụm `đóng vai trò cốt lõi`, `có ý nghĩa to lớn`.
>
> Đoạn 2. Thống kê TinyPerson gốc: 1.610 ảnh, 794 train, 816 test, 72.651
> khung. Kích thước trung bình 18,0 ± 17,4 điểm ảnh. Tỷ lệ kích thước tương
> đối 0,012 ± 0,010, nhỏ hơn khoảng 16 lần so với COCO. Nêu nguồn [1].
>
> Đoạn 3. Bản derivative dùng trong báo cáo: 746 ảnh, 662 train, 84
> validation. Official test: 816 ảnh, loại 30 ảnh dense còn 786 ảnh và 13.687
> GT. Giải thích vì sao phải phân biệt ba cấp này.
>
> Đoạn 4. Ba thách thức: suy giảm độ phân giải, nền phức tạp và mật độ không
> đều. Mỗi thách thức gắn với một hệ quả có thể đo, không chỉ mô tả định tính.
>
> Đoạn 5. Phạm vi báo cáo và ba câu hỏi Q1, Q2, Q3. Câu hỏi chi tiết nằm ở
> 2.1, ở đây chỉ nêu một câu dẫn.

**Bảng cần chèn:** Bảng 1.1, thống kê dữ liệu. Vị trí sau đoạn 3, trước đoạn 4.

> Bảng 1.1. TinyPerson gốc, bản derivative và tập official test.

| Cấp dữ liệu | Tổng ảnh | Train | Validation | Test | Ghi chú |
|---|---:|---:|---:|---:|---|
| TinyPerson gốc | 1.610 | 794 | Không có | 816 | 72.651 GT, hai lớp sea/earth person |
| Derivative YOLO | 746 | 662 | 84 | Không có | Một lớp person, dùng chọn checkpoint |
| Official test sau loại dense | 786 | Không có | Không có | 786 | 13.687 GT, 30 ảnh dense bị loại |

**Hình cần chèn:** Hình 1.1, histogram căn bậc hai diện tích hộp và bốn ảnh ví
dụ ở mức mật độ khác nhau. Vị trí sau Bảng 1.1, trước heading 1.2. Xem
`06_hinh_bang_vi_tri.md`.

**Câu cần xóa hoặc sửa:**

- Xóa `Đặc biệt, nếu xét theo kích thước tương đối...` nếu đã có bảng. Chỉ giữ
  một câu so sánh với COCO.
- Xóa `Tên riêng của TinyPerson và YOLO26n được giữ nguyên; các thuật ngữ toán
  học và xử lý tín hiệu được Việt hóa thống nhất`. Quy ước này thuộc phần mở
  đầu, không thuộc nội dung nghiên cứu.

## 1.2. Giới hạn biểu diễn và sai số định vị

**Vị trí:** mục 1.2, bốn đoạn hiện tại.

**Vấn đề:** ví dụ IoU 64 × 64 và 8 × 8 là điểm mạnh, nên giữ. Tuy nhiên mục
này kết thúc bằng cách phân lỗi thành ba loại chung chung, chưa nối với phân
rã lỗi ở chương 3.

**Cách sửa:** giữ đoạn 1 và ví dụ IoU. Thay đoạn cuối bằng định nghĩa chính
xác của năm loại kết quả sẽ dùng ở chương 3. Thêm Hình 1.2.

**Đoạn thay thế cho đoạn cuối:**

> Trong báo cáo này, mỗi dự đoán được gán một trạng thái chính. TP là dự đoán
> khớp GT dương tính cùng lớp với IoU ít nhất 0,50. Miss là GT dương tính
> không có dự đoán khớp. FP là dự đoán không được gán TP hoặc trung tính.
> Với FP, lấy IoU lớn nhất với GT dương tính cùng lớp trong dải đang xét.
> Duplicate FP là dự đoán có IoU ít nhất 0,50 với một GT đã được khớp trước
> theo thứ tự confidence. Nếu không phải duplicate, localization FP là dự
> đoán có 0,10 ≤ IoU < 0,50; background FP là dự đoán có IoU < 0,10. Các dự
> đoán khớp vùng ignore/uncertain hoặc nằm ngoài dải kích thước là trung tính
> và không bị tính như background FP.

**Hình cần chèn:** Hình 1.2, IoU theo độ dịch tâm cho hộp 4, 8, 16 và 32 điểm
ảnh. Vị trí sau đoạn ví dụ IoU, trước đoạn định nghĩa TP/miss/FP.

**Lưu ý số liệu:** giữ nguyên phép tính 3.844 / (4.096 + 4.096 - 3.844) ≈
0,884 và 36 / (64 + 64 - 36) ≈ 0,391. Hai phép tính này đúng. Không thay bằng
số làm tròn khác.

## 1.3. YOLO26n và bản đồ đặc trưng P2

**Vị trí:** thay toàn bộ mục 1.3.

**Vấn đề:** mục hiện tại chỉ mô tả P3, P4, P5. Không có P2, trong khi BTRM v2
được đặt trên P2. Đoạn về NMS-free, DFL, ProgLoss, STAL và MuSGD có thể giữ
nhưng cần rút gọn vì không phải trọng tâm.

**Nội dung thay thế:**

> YOLO26n là phiên bản Nano của họ YOLO26. Mô hình gồm backbone, neck hợp nhất
> đa tỷ lệ và đầu phát hiện. YOLO26 dùng suy diễn đầu-cuối không cần NMS, bỏ
> Distribution Focal Loss ở nhánh hồi quy và dùng các cơ chế ProgLoss, STAL
> và MuSGD trong huấn luyện [7]. Báo cáo chỉ dùng các thay đổi này ở mức mô
> tả. Không có ablation riêng cho từng cơ chế.
>
> Cấu hình A2 trong repo thêm nhánh P2 vào neck. Mô hình xuất bốn bản đồ đặc
> trưng P2, P3, P4, P5 với stride 4, 8, 16, 32. Với đầu vào 1.280 × 1.280,
> kích thước không gian là 320 × 320, 160 × 160, 80 × 80 và 40 × 40. Bảng
> 1.2 tóm tắt.
>
> Một người cao 8 điểm ảnh chiếm khoảng hai ô trên P2 và một ô trên P3. Vì
> vậy P2 là nhánh còn giữ được nhiều vị trí cho nhóm tiny1. Đây là lý do BTRM
> v2 được đặt trên P2. Báo cáo không khẳng định P2 là lựa chọn tối ưu cho mọi
> bộ dữ liệu.

> Bảng 1.2. Bản đồ đặc trưng của YOLO26n A2 ở đầu vào 1.280.

| Nhánh | Stride | Kích thước | Vai trò |
|---|---:|---|---|
| P2 | 4 | 320 × 320 | Người dưới 12 điểm ảnh |
| P3 | 8 | 160 × 160 | Người 12 đến 32 điểm ảnh |
| P4 | 16 | 80 × 80 | Người 32 đến 96 điểm ảnh |
| P5 | 32 | 40 × 40 | Người lớn hơn 96 điểm ảnh |

**Hình cần chèn:** Hình 1.3, sơ đồ bốn nhánh và vị trí BTRM trên P2. Vị trí
sau Bảng 1.2, trước heading 1.4.

**Câu cần sửa:**

- `bộ hợp nhất đa tỷ lệ PAFPN tạo ra ba nhánh` phải đổi thành `bốn nhánh`.
- `P3 (tầng 16, stride 8), P4 (tầng 19, stride 16), P5 (tầng 22, stride 32)`
  phải đổi thành `P2 (tầng 19, stride 4), P3 (tầng 22, stride 8), P4 (tầng
  25, stride 16), P5 (tầng 28, stride 32)`. Đối chiếu lại số layer với
  `configs/model/yolo26n_tiny_A2_p2.yaml` trước khi chốt.

## 1.4. Động lực từ miền tần số

**Vị trí:** mục 1.4, bốn đoạn hiện tại.

**Vấn đề:** mục này dài 837 từ nhưng chỉ dùng để dẫn tới ý tương phản cục bộ.
Phần Fourier và spectral residual có thể rút còn một phần ba. Nếu giữ dài, nó
làm loãng trọng tâm và khiến người đọc tưởng BTRM dựa trên Fourier.

**Cách sửa:** giữ một đoạn về phân tách tần số thấp và tần số cao. Giữ một
đoạn ngắn về spectral residual như tiền lệ ý tưởng, nhưng nói rõ BTRM không
dùng Fourier. Bỏ các đoạn lặp về ưu điểm của miền tần số.

**Đoạn thay thế:**

> Ảnh tự nhiên có thể được xem như tổng của thành phần biến thiên chậm, tương
> ứng nền và ánh sáng, với thành phần biến thiên nhanh, tương ứng biên và chi
> tiết. Với người cực nhỏ, phần lớn năng lượng nằm ở tần số cao nhưng biên độ
> thấp, nên dễ bị các tầng giảm mẫu làm mất. Cách nhìn này giải thích vì sao
> một toán tử cục bộ nhấn biên có thể bổ sung thông tin cho đặc trưng nội tại.
>
> Hou và Zhang dùng phần dư phổ để làm nổi vùng bất thường trên ảnh [29].
> BTRM v2 chia sẻ ý tưởng loại thành phần nền, nhưng không dùng biến đổi
> Fourier. Mô-đun tính trung bình cục bộ và gradient trực tiếp trên bản đồ đặc
> trưng. Vì vậy spectral residual chỉ là tiền lệ khái niệm, không phải cơ sở
> toán học của mô-đun.

## 1.5. Tương phản cục bộ và sai khác ngữ cảnh

**Vị trí:** mục 1.5, sáu đoạn hiện tại.

**Vấn đề:** mục này định nghĩa hai khái niệm, sau đó phát biểu H01. Cách viết
hiện tại có nhiều câu dài và lặp lại `sai khác ngữ cảnh cục bộ`. H01 cũng
được phát biểu lại ở 2.1.

**Cách sửa:** giữ hai định nghĩa. Rút H01 còn ba câu. Thêm một đoạn nêu cách
diễn giải kết quả ở cấp mô hình. Không dùng từ `bác bỏ` như một kết luận thống
kê khi chỉ có một seed. Không phát biểu lại H01 ở 2.1.

**Đoạn H01 thay thế:**

> H01. Khi căn bậc hai diện tích hộp người từ 1 đến 20 điểm ảnh (dải tiny),
> tín hiệu sai khác giữa vùng ứng viên và nền lân cận suy giảm chậm hơn đặc
> trưng hình thái nội tại. Vì vậy tín hiệu này có thể bổ sung khả năng phân
> biệt người với nền.
>
> H01 được kiểm tra ở hai cấp. Ở cấp đặc trưng, phép đo là AUPRC giữa điểm
> người và điểm nền theo dải kích thước. Ở cấp mô hình, phép đo là AP@0.75 và
> recall tiny1 trên hai cặp paired. Báo cáo này chỉ hoàn tất phép đo cấp mô
> hình. Phép đo cấp đặc trưng chưa có, nên H01 chưa được kết luận đầy đủ.

## 1.6. Nghiên cứu liên quan và khoảng trống

**Vị trí:** mục 1.6, bảy đoạn hiện tại, dài 2.307 từ.

**Vấn đề:** mục này là phần dài nhất của chương 1. Bốn trục nghiên cứu được
trình bày dưới dạng bốn đoạn dài, nhiều câu chồng mệnh đề. Các ý về tiền lệ
của BTRM được nhắc lại ở 1.4 và 1.5.

**Cách sửa:** giữ bốn trục nhưng mỗi trục còn 250 đến 350 từ. Thêm Bảng 1.3 so
sánh. Xóa các câu lặp về việc BTRM không phải ý tưởng hoàn toàn mới. Kết thúc
bằng một đoạn nêu khoảng trống nghiên cứu và ba đóng góp thực tế.

**Bảng cần chèn:** Bảng 1.3, so sánh hướng tiếp cận. Vị trí sau đoạn thứ tư
của 1.6, trước đoạn `Tóm lược khoảng trống nghiên cứu`.

> Bảng 1.3. Các hướng tiếp cận liên quan và khoảng trống của báo cáo.

| Hướng | Đại diện | Can thiệp | Bằng chứng định lượng | Giới hạn |
|---|---|---|---|---|
| Đối sánh thang | TinyPerson, SM+ | Dữ liệu và tiền huấn luyện | AP tổng | Không tách lỗi hộp |
| Trường tiếp nhận | RFLA, adaptive receptive field | Gán nhãn hoặc kiến trúc | AP, recall | Can thiệp sâu vào mạng |
| Tương phản cục bộ | MPCM, ACM, spectral residual | Đặc trưng hoặc ảnh | Chủ yếu trên ảnh hồng ngoại | Chưa kiểm tra trên người RGB |
| Sai khác tái tạo | Difference map, OW-DAR | Bản đồ sai khác | AP trên tiny object | Chưa tách vai trò nền P2 |
| Hồi quy và gán nhãn | NWD, RFLA | Loss hoặc assignment | AP@0.75 | Chưa thử trong báo cáo |
| Báo cáo này | BTRM v2 trên P2 | Đặc trưng trước Detect | AP@0.25/0.50/0.75, TP/FP | Một seed, một dataset |

**Đoạn đóng góp thay thế:**

> Báo cáo có ba đóng góp. Thứ nhất, báo cáo xây dựng một phép so sánh paired
> giữa A2 và BTRM v2 trên cùng dữ liệu, cùng seed và cùng protocol. Thứ hai,
> báo cáo phân rã lỗi thành TP, miss, localization FP, background FP và
> duplicate FP, cho thấy số localization FP giảm nhưng background FP tăng.
> Các số đếm tổng hợp không đủ để quy từng dự đoán cho một kiểu dịch lỗi. Thứ ba,
> báo cáo công bố chi phí tham số, độ trễ và bộ nhớ để người đọc cân nhắc giữa
> lợi ích và chi phí.
>
> Báo cáo không đóng góp một kiến trúc mới. BTRM v2 là công cụ để kiểm tra một
> giả thuyết về tín hiệu cục bộ. Kết quả không ủng hộ giả thuyết ở cấp mô
> hình, nhưng chưa đủ để bác bỏ ở cấp đặc trưng.
