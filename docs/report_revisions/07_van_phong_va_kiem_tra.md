# Văn phong và kiểm tra trước khi bàn giao

File này áp dụng `~/Downloads/SKILL.md` academic-humanizer cho toàn báo cáo.
Mục tiêu là văn phong học thuật tự nhiên, không phải né nhận diện AI.

## Các dấu hiệu cần sửa

### Đoạn dài và chồng mệnh đề

**Vị trí:** 1.1 đoạn đầu 275 từ, 1.2 đoạn đầu 290 từ, 1.6 các đoạn tổng quan
trên 300 từ.

**Cách sửa:** mỗi đoạn 80 đến 150 từ, một ý chính. Tách câu khi có hơn hai
mệnh đề phụ. Không đổi số hoặc citation khi tách.

**Ví dụ:**

> Câu hiện tại: `Khảo sát toàn diện của Nikouei và cộng sự chỉ ra rằng sự suy
> giảm độ phân giải không gian, hiện tượng che khuất cục bộ, mật độ phân bố
> không đồng đều và sự nhiễu loạn từ nền cảnh rộng là những rào cản lớn nhất
> đối với các bộ phát hiện đối tượng nhỏ dựa trên học sâu hiện đại.`
>
> Sửa: `Nikouei và cộng sự tổng hợp bốn khó khăn thường gặp: độ phân giải
> thấp, che khuất, mật độ đối tượng không đều và nhiễu nền [8]. Trên TinyPerson,
> hai khó khăn cuối thể hiện rõ ở ảnh bờ biển đông người và vùng nước có phản
> xạ mạnh [1].`

### Khẳng định quá mức

**Vị trí:** 1.6, đoạn `Các công trình này xác nhận rằng...`; 3.7, đoạn `Sự sụt
giảm nhẹ về AP...`; kết luận.

**Cách sửa:** dùng `cho thấy`, `phù hợp với`, `chưa đủ để kết luận` khi bằng
chứng chỉ từ một seed. Không dùng `chứng minh`, `xác nhận`, `bác bỏ hoàn toàn`
cho kết quả này.

### Từ tăng cường rỗng

**Vị trí:** 1.1 `tính cấp thiết thực tiễn`, 1.6 `tổng quan sâu rộng`, 1.6
`phân biệt rạch ròi`, 2.1 `kiểm định H01 một cách khách quan`.

**Cách sửa:** bỏ các tính từ đánh giá. Viết trực tiếp dữ kiện và phép đo.

### Ngoặc đơn không cần thiết

**Vị trí:** toàn báo cáo.

**Cách sửa:** chỉ dùng ngoặc khi định nghĩa viết tắt hoặc ghi thông số. Ví dụ:

- `tỷ số giao trên hợp (Intersection over Union – IoU)` giữ ở lần đầu.
- `mạng khung xương (Backbone)` giữ ở lần đầu.
- `khung tham chiếu thực tế (ground truth)` giữ ở lần đầu.
- Các lần sau chỉ dùng thuật ngữ đã chọn, không lặp ngoặc.

### Em dash và dấu câu kiểu AI

**Vị trí:** toàn báo cáo.

**Cách sửa:** thay em dash bằng dấu phẩy, dấu hai chấm hoặc tách câu. Không
thay mọi dấu gạch ngang trong tên metric hoặc khoảng số.

### Phủ định đối xứng

**Vị trí:** các câu dạng `không chỉ X mà còn Y`, `không đơn thuần X, mà còn
Y`.

**Cách sửa:** viết hai câu. Câu đầu nói X, câu sau nói Y và chỉ ra quan hệ.

## Checklist nội dung

Trước khi công cụ chỉnh sửa đánh dấu hoàn thành, kiểm tra từng mục:

- [ ] Tóm tắt mô tả BTRM v2 trên P2, không nhắc Tikhonov.
- [ ] Chương 1 có bảng dữ liệu, định nghĩa đúng P2/P3/P4/P5 và H01 chỉ phát
  biểu một lần.
- [ ] Chương 2 không còn tám điểm vòng, mô hình nền bậc nhất, CFAR, chuẩn hóa
  phương sai hoặc hệ số điều chế bị chặn.
- [ ] Chương 2 có công thức `Y = X + gamma × S × P` và gamma khởi tạo 0.
- [ ] Chương 2 có protocol paired và giải thích confound khởi tạo scratch.
- [ ] Chương 3 báo AP@0.25, AP@0.50, AP@0.75 và APmean.
- [ ] Chương 3 phân rã TP, miss, localization FP, background FP và duplicate FP.
- [ ] Chương 3 báo kết quả theo kích thước, confidence, hình học hộp và chi phí.
- [ ] Dải kích thước ghi rõ dùng căn bậc hai diện tích trong tọa độ ảnh gốc,
  không phải sau letterbox 1.280; nói rõ evaluator dùng cận inclusive.
- [ ] Không có câu nào nói tiêu chí H01 hoặc metric chính được đăng ký trước
  khi xem kết quả nếu không có tài liệu chứng minh.
- [ ] Không dùng `bác bỏ H01` như kết luận thống kê khi chỉ có một seed.
- [ ] Không quy chênh lệch tổng hợp thành quan hệ nhân quả giữa từng dự đoán,
  ví dụ `mô-đun dịch lỗi từ localization sang background`.
- [ ] Không khẳng định nút thắt hồi quy/gán nhãn là đã chứng minh; chỉ ghi là
  giả thuyết phù hợp với số liệu phân rã lỗi.
- [ ] Kết luận trả lời Q1, Q2, Q3, nêu giới hạn một seed và evaluator nội bộ.
- [ ] Không có câu nào khẳng định H01 đã được kiểm định ở cấp đặc trưng.
- [ ] Không so sánh AP nội bộ với bảng công bố TinyPerson.
- [ ] Mọi số liệu trong bảng khớp `docs/BTRM_REPORT.md`.
- [ ] Chi phí latency, FPS và bộ nhớ có raw artifact hoặc log nguồn; nếu chưa
  có, ghi rõ nguồn là tài liệu tổng hợp và chưa công bố như số đo độc lập.
- [ ] Không có citation nào không xuất hiện trong danh mục tài liệu.
- [ ] Không có tài liệu nào trong danh mục mà không được trích trong thân bài.

## Checklist hình thức

- [ ] Tổng báo cáo không quá 50 trang không kể mục lục và phụ lục.
- [ ] Chương 1 không quá 15 trang, chương 2 không quá 25 trang, chương 3 không
  quá 20 trang.
- [ ] Mục lục, danh mục hình và danh mục bảng khớp số trang cuối.
- [ ] Hình có caption dưới hình; bảng có caption trên bảng.
- [ ] Công thức là equation native, không phải ảnh chụp.
- [ ] Không có caption lỗi font như `Hình 2.1Vịuy í tícìnhợpủa`.
- [ ] Không có hình P3 cũ trong chương 2.
- [ ] Không có hai biểu đồ AP@0.50/APmean cũ trong chương 3.
- [ ] Thứ tự Hình 3.1 đến 3.6 tăng dần theo vị trí xuất hiện trong thân bài.
- [ ] Hình 3.4 dùng dữ liệu từng hộp trong `errors.jsonl`, không vẽ box plot
  từ bảng trung vị.
- [ ] Mỗi hình được nhắc trong câu trước khi chèn.
- [ ] Bảng 3.9 và 3.10 không tràn lề.
- [ ] Ngày tháng và năm của tài liệu tham khảo được kiểm tra lại.

## Việc chưa có bằng chứng, không được tự viết thêm

- Mean và độ lệch chuẩn qua nhiều seed.
- AUPRC cấp đặc trưng.
- VisDrone hoặc cross-dataset.
- So sánh với evaluator tham chiếu TinyPerson.
- AP-small theo COCO trên ảnh gốc.
- Miss rate từ evaluator tham chiếu.

Những mục này có thể ghi là giới hạn hoặc kế hoạch, nhưng không được viết như
kết quả đã hoàn thành.
