# Hướng dẫn sửa báo cáo TGMT_BaoCao

Bộ file này chỉ chứa hướng dẫn sửa. Không có thao tác nào lên Google Docs.
Mỗi file nêu vị trí theo heading và câu mở đầu trong bản gốc, để công cụ sửa
tìm được đúng đoạn.

## Bản đang dùng làm mốc

- Bản gốc: revision 128, 24/10/2026 19:41:48 giờ Việt Nam.
- Bản sao cục bộ: `/home/longtran/Downloads/TGMT_BaoCao.docx` và
  `/tmp/original_rev128.md`.
- Tổng độ dài: khoảng 10.258 từ, 28 trang. Chương 1 khoảng 6.315 từ, chương 2
  khoảng 2.574 từ, chương 3 khoảng 1.277 từ, kết luận 92 từ.

## Chẩn đoán tổng thể

Báo cáo hiện tại không thiếu mục. Vấn đề là phân bổ nội dung lệch và bằng chứng
không đồng nhất:

- Chương 1 dài nhưng nhiều đoạn lặp ý và mô tả chung.
- Chương 2 dài vừa nhưng mô tả BTRM phiên bản P3 cũ, không khớp mã nguồn hiện
  tại là BTRM v2 trên P2.
- Chương 3 quá ngắn, thiếu AP75, phân rã TP/miss/FP, phân tích theo kích thước,
  chi phí và kiểm định thống kê.
- Kết luận chỉ có một đoạn, không trả lời trực tiếp từng câu hỏi nghiên cứu.
- Một số hình là ảnh chụp công thức, một caption bị lỗi font, một số hình chưa
  được nhắc trong văn bản.

## Nguyên tắc sửa

1. Chọn một phiên bản BTRM duy nhất. Bộ hướng dẫn này dùng BTRM v2 trên P2,
   khớp `src/tinydet/modules/btrm.py`. Không trộn với BTRM P3, tám điểm vòng
   và Tikhonov của bản cũ.
2. Mọi số liệu phải lấy từ artifact trong repo. Không suy ra số mới từ hình
   hoặc từ trí nhớ.
3. Mỗi đoạn nêu một ý. Đoạn quá 150 từ phải tách.
4. Không dùng em dash. Hạn chế ngoặc đơn giải thích tiếng Anh. Định nghĩa thuật
   ngữ một lần trong bảng viết tắt.
5. Không dùng các cụm tăng cường rỗng: đóng vai trò quan trọng, tiềm năng to
   lớn, không chỉ X mà còn Y, mở ra hướng mới, là minh chứng cho.
6. Không tuyên bố BTRM thất bại trong mọi điều kiện. Kết luận chỉ nói rằng
   trong protocol hiện tại, BTRM không cải thiện AP75.
7. Hình và bảng phải được nhắc trong văn bản trước khi xuất hiện.

## Mục tiêu độ dài

Giảng viên giới hạn chương 1 không quá 15 trang, chương 2 không quá 25 trang,
chương 3 không quá 20 trang, toàn báo cáo không quá 50 trang không kể mục lục
và phụ lục. Báo cáo hiện tại còn nhiều chỗ trống, nên có thể tăng chương 2 và
chương 3 mà vẫn dưới giới hạn.

| Phần | Hiện tại | Mục tiêu | Việc chính |
|---|---:|---:|---|
| Tóm tắt | 0,7 trang | 1 trang | Viết lại theo kết quả BTRM v2 |
| Chương 1 | 6,3 nghìn từ | 11 đến 13 trang | Giữ, cắt lặp, thêm số liệu và hình |
| Chương 2 | 2,6 nghìn từ | 16 đến 18 trang | Viết lại BTRM v2, protocol, công thức |
| Chương 3 | 1,3 nghìn từ | 13 đến 15 trang | Viết lại toàn bộ theo artifact |
| Kết luận | 92 từ | 1,5 đến 2 trang | Trả lời từng câu hỏi, nêu giới hạn |
| Tài liệu tham khảo | 41 mục | 35 đến 45 mục | Kiểm tra trích dẫn, bỏ mục không dùng |

Tổng mục tiêu khoảng 43 đến 48 trang, chừa khoảng 2 trang cho phần đầu và tài
liệu tham khảo để không vượt giới hạn 50 trang. Không thêm nội dung chỉ để đủ
độ dài; ưu tiên bảng, hình và diễn giải có artifact.

## Đánh giá mức hoàn thiện từng mục

Thang điểm 1 đến 5. Điểm 1 là sai hoặc thiếu bằng chứng, điểm 3 là đủ ý nhưng
chưa đủ số liệu, điểm 5 là đủ ý và khớp artifact. Điểm 5 vẫn không đồng nghĩa
với sẵn sàng công bố: còn phải xác minh evaluator, số seed, tài liệu tham khảo
và metadata phần cứng.

| Mục | Từ | Điểm | Vấn đề chính | Việc cần làm |
|---|---:|---:|---|---|
| 1.1 | 623 | 3 | Đoạn dài, chưa có hình và bảng dữ liệu | Tách đoạn, thêm Bảng 1.1, Hình 1.1 |
| 1.2 | 784 | 4 | Ví dụ IoU tốt, định nghĩa lỗi chưa gắn chương 3 | Thêm định nghĩa TP/miss/FP, Hình 1.2 |
| 1.3 | 821 | 2 | Chỉ có P3 đến P5, không có P2 | Viết lại theo A2 P2, thêm Bảng 1.2, Hình 1.3 |
| 1.4 | 837 | 3 | Dài, phần Fourier xa trọng tâm | Rút còn một phần ba |
| 1.5 | 750 | 4 | Khái niệm rõ, H01 lặp ở 2.1 | Giữ một phát biểu H01 |
| 1.6 | 2.307 | 3 | Quá dài, lặp, chưa có bảng so sánh | Rút còn 1.300 từ, thêm Bảng 1.3 |
| Đóng góp | 193 | 2 | Vẫn nói A0, A2, A3 | Đổi sang paired A2/BTRM v2 |
| 2.1 | 537 | 3 | H01 lặp, thiếu danh sách Q1-Q3 | Thêm Q1-Q3 và tiêu chí ủng hộ |
| 2.2 | 332 | 2 | Bảng A0-A5 không khớp run thực tế | Đổi thành bảng bốn nhánh paired |
| 2.3 | 332 | 2 | Mô tả P3, caption lỗi font | Viết lại P2, layer 29 |
| 2.4 | 408 | 1 | Tám điểm vòng, nền bậc nhất, đoạn lặp | Xóa, thay bằng BTRM v2 |
| 2.5 | 353 | 1 | Tikhonov, phương sai, điều chế bị chặn | Xóa, thay bằng project/gate/gamma |
| 2.6 | 304 | 3 | Protocol chưa đủ số liệu | Thêm dữ liệu, metric, evaluator |
| 2.7 | 308 | 3 | Có bất biến nhưng thiếu lệnh cụ thể | Thêm checklist và đường dẫn |
| 3.1 | 147 | 2 | Thiếu run ID, commit, cấu hình | Thêm Bảng 3.1 |
| 3.2 | 221 | 1 | Sai số validation, protocol không tái lập | Viết lại theo 746/84/786 |
| 3.3 | 87 | 2 | Báo A0, không báo A2 | Thay bằng baseline A2 |
| 3.4 | 255 | 1 | A0/A2/A3 cũ, không có AP@0.75 | Thay bằng bốn nhánh BTRM v2 |
| 3.5 | 134 | 2 | Chỉ kiểm tra đại số định tính | Thêm TP/miss/FP và ablation gamma |
| 3.6 | 285 | 2 | Bốn giả thuyết không có số liệu | Thay bằng phân tích theo bằng chứng |
| Kết luận | 92 | 1 | Không trả lời Q1-Q3 | Viết lại 700-900 từ |
| Tài liệu | 1.230 | 3 | Nhiều mục, metadata chưa kiểm tra | Kiểm tra trích dẫn và định dạng |

## Thứ tự thực hiện

1. Sửa thuật ngữ và số liệu gốc theo `01_sua_nhanh.md`.
2. Viết lại chương 2 theo `02_chuong_2.md`.
3. Viết lại chương 3 theo `03_chuong_3.md`.
4. Cập nhật chương 1 theo `04_chuong_1.md`.
5. Viết lại tóm tắt, kết luận, tài liệu theo `05_tom_tat_ket_luan_tai_lieu.md`.
6. Chèn hình và bảng theo `06_hinh_bang_vi_tri.md`.
7. Chạy checklist trong `07_van_phong_va_kiem_tra.md`.

## Quy ước số liệu

- Dùng dấu phẩy thập phân và dấu chấm hàng nghìn theo tiếng Việt.
- AP ghi 4 chữ số thập phân trong câu, 6 chữ số trong bảng gốc.
- Chênh lệch tuyệt đối ghi là điểm phần trăm, không ghi phần trăm. Khi AP
  đang ở dạng phân số, nhân chênh lệch với 100 trước khi ghi điểm phần trăm.
- Tên metric thống nhất: AP@0.25, AP@0.50, AP@0.75, APmean.
- APmean là trung bình cộng của AP@0.25, AP@0.50 và AP@0.75, đúng với khóa
  `AP_mean_IoU_0.25_0.50_0.75` trong artifact. Đây không phải AP COCO
  @[0.50:0.95].

## Nguồn số liệu chuẩn

| Nội dung | Nguồn |
|---|---|
| Thiết kế BTRM v2 | `src/tinydet/modules/btrm.py`, `docs/BTRM_P2.md` |
| Bảng AP, phân rã lỗi, chi phí | `docs/BTRM_REPORT.md` |
| Kết quả validation và warm-start | `docs/BTRM_P2.md` |
| Protocol dữ liệu | `docs/agents/DATA_EVALUATION.md`, `docs/TINYPERSON_BENCHMARK.md` |
| Artifact từng run | `docs/BTRM_REPORT.md`, mục Artifact |
| Kết quả cũ A0/A2/A3 | `docs/BTRM_MSR_ANALYSIS.md` |

## Cảnh báo trước khi public

- Chưa có nhiều seed. Không được dùng chữ có ý nghĩa thống kê nếu chưa chạy
  kiểm định.
- Evaluator là implementation nội bộ, chưa đối chiếu với evaluator tham chiếu. Không
  so sánh trực tiếp với bảng công bố của TinyPerson.
- Chưa có AUPRC theo dải kích thước. Không được nói H01 đã được kiểm định ở
  cấp đặc trưng.
- Cấu hình ghi GPU A10 nhưng tài liệu run ghi L4. Phải đối chiếu metadata của
  run trước khi ghi phần cứng.
- Chưa thấy artifact benchmark latency/FPS/bộ nhớ trong `outputs/`. Các số chi
  phí hiện lấy từ `docs/BTRM_REPORT.md`; phải lưu raw log trước khi công bố.
- Dải kích thước của evaluator dùng căn bậc hai diện tích hộp trong tọa độ ảnh
  gốc, không phải tọa độ sau letterbox 1.280. Các cận của evaluator là inclusive,
  còn `DIAGNOSTIC_SIZE_BINS` dùng cận trên loại trừ. Không trộn hai cách gán.
