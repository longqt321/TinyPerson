# Chương 3. Thử nghiệm và kết quả

## Chẩn đoán chương hiện tại

Chương 3 hiện có khoảng 1.277 từ, 7 mục. Đây là chương ngắn nhất nhưng phải
gánh toàn bộ bằng chứng của báo cáo. Các thiếu sót chính:

- Chỉ báo AP@0.50 và APmean, không có AP@0.25 và AP@0.75.
- Không có phân rã TP, miss, FP.
- Không có kết quả theo dải kích thước.
- Không có phân tích confidence, hình học hộp và chi phí.
- Kết quả A0, A2, A3 thuộc protocol cũ, không paired với BTRM v2.
- Hình 3.1 và 3.2 chỉ là biểu đồ AP@0.50 và APmean, không trả lời câu
  hỏi localization.

Chương 3 sau khi sửa cần trả lời Q1, Q2, Q3 ở mục 2.1. Mỗi mục phải có bảng
hoặc hình và một đoạn diễn giải ngắn. Không lặp lại mô tả phương pháp.

## Cấu trúc đề xuất

| Mục | Nội dung | Độ dài mục tiêu |
|---|---|---:|
| 3.1 | Môi trường và artifact | 1 trang |
| 3.2 | Dữ liệu và protocol đánh giá | 1 trang |
| 3.3 | Kết quả baseline A2 | 1 trang |
| 3.4 | Kết quả BTRM v2 scratch và warm-start | 3 trang |
| 3.5 | Phân rã TP, miss và FP | 2 trang |
| 3.6 | Kết quả theo kích thước | 2 trang |
| 3.7 | Hình học hộp và ngưỡng confidence | 2 trang |
| 3.8 | Kiểm tra hiện thực và ablation gamma | 1 trang |
| 3.9 | Chi phí và thảo luận | 1 trang |
| 3.10 | Đe dọa đến tính hợp lệ | 1 trang |

Tổng mục tiêu khoảng 15 trang, còn chỗ cho hình và tránh vượt giới hạn 20 trang
của chương 3. Chỉ thêm diễn giải khi có artifact chứng minh.

## 3.1. Môi trường và artifact

**Vị trí:** thay toàn bộ mục 3.1.

**Vấn đề:** mục hiện tại chỉ có một đoạn 147 từ, thiếu run ID, commit và cấu
hình. Câu `một số phép trích xuất đặc trưng chưa hoàn tất do giới hạn VRAM` là
ghi chú tiến độ, không phải nội dung báo cáo.

**Nội dung thay thế:**

> Các run được thực hiện qua Modal. Bảng 3.1 ghi run ID theo tài liệu, đường
> dẫn checkpoint và metric đang có trong checkout. Chỉ run A2 scratch còn đầy
> đủ thư mục run cục bộ; ba run còn lại chỉ có `best.pt` và artifact metric,
> error analysis. Phải bổ sung thư mục run hoặc ghi rõ nguồn lưu trữ trước khi
> công bố. Phiên bản Python, PyTorch, Ultralytics và GPU cũng phải đối chiếu
> metadata của từng run, không suy ra từ cấu hình hiện tại.
>
> Bảng 3.1 liệt kê năm artifact được dùng trong báo cáo. Bốn run đầu là hai
> cặp paired. Run thứ năm dùng lại checkpoint BTRM scratch và ép gamma bằng 0
> để kiểm tra tác động của phần dư đã học.

> Bảng 3.1. Run và artifact dùng trong báo cáo.

| Nhánh | Run ID theo tài liệu | Checkpoint cục bộ | Metric | Error analysis |
|---|---|---|---|---|
| A2 scratch | `tiny_A2_p2/20261019-194809-8326468-seed42` | `outputs/tiny_A2_p2/20261019-194809-8326468-seed42/checkpoints/best.pt` | `outputs/metrics/tiny_A2_p2-tinyperson-test.json` | `outputs/error_analysis/tiny_A2_p2-test/` |
| BTRM scratch | `tiny_A2_p2_btrm/20261023-233359-8efefb4-seed42` | `outputs/tiny_A2_p2_btrm/best.pt` | `outputs/metrics/tiny_A2_p2_btrm-tinyperson-test.json` | `outputs/error_analysis/tiny_A2_p2_btrm-test/` |
| A2 warm-start | `tiny_A2_p2_adapt/20261024-005159-8efefb4-seed42` | `outputs/tiny_A2_p2_adapt/best.pt` | `outputs/metrics/tiny_A2_p2_adapt-tinyperson-test.json` | `outputs/error_analysis/tiny_A2_p2_adapt-test/` |
| BTRM warm-start | `tiny_A2_p2_btrm_adapt/20261024-005202-8efefb4-seed42` | `outputs/tiny_A2_p2_btrm_adapt/best.pt` | `outputs/metrics/tiny_A2_p2_btrm_adapt-tinyperson-test.json` | `outputs/error_analysis/tiny_A2_p2_btrm_adapt-test/` |
| BTRM gamma 0 | Dùng lại checkpoint BTRM scratch | `outputs/tiny_A2_p2_btrm/best.pt` | `outputs/metrics/tiny_A2_p2_btrm_gamma0-tinyperson-test.json` | `outputs/error_analysis/tiny_A2_p2_btrm_gamma0-test/` |

**Lưu ý kiểm tra:** metadata của run A2 scratch xác nhận GPU NVIDIA L4, Python
3.12.10, PyTorch 2.11.0+cu128 và Ultralytics 8.4.164. Cấu hình experiment vẫn
ghi `gpu: A10`, nên trường này đã lỗi thời. Ba run BTRM không có metadata cục
bộ; nếu không tìm được bản gốc, không ghi phần cứng hoặc phiên bản thư viện
cho ba run đó. Không để hai thông tin phần cứng mâu thuẫn trong cùng báo cáo.

## 3.2. Dữ liệu và protocol đánh giá

**Vị trí:** thay toàn bộ mục 3.2.

**Vấn đề:** số liệu 322 ảnh validation và 12.017 khung không khớp protocol
hiện tại. Câu `tiêu chí chọn tập con chưa được ghi lại đầy đủ` cho thấy báo
cáo đang mô tả một protocol không tái lập được.

**Nội dung thay thế:**

> Bảng 3.2 tóm tắt dữ liệu ở ba cấp. Bản derivative dùng để huấn luyện và chọn
> checkpoint. Tập official test dùng để báo cáo kết quả cuối. Hai cấp này
> không được trộn khi báo cáo metric.

> Bảng 3.2. Dữ liệu và protocol đánh giá.

| Cấp đánh giá | Số ảnh | Số GT | Mục đích | Ghi chú |
|---|---:|---:|---|---|
| Train derivative | 662 | Không báo cáo | Huấn luyện | Nhãn YOLO, gộp hai lớp person |
| Validation derivative | 84 | Không báo cáo | Chọn checkpoint | Không gọi là test |
| Official test | 816 | Không báo cáo | Đánh giá cuối | Có 30 ảnh dense |
| Official test sau loại dense | 786 | 13.687 | Báo cáo AP, recall, FP | Protocol TinyPerson |

> Toàn bộ ảnh được letterbox về 1.280 × 1.280. Dự đoán trên official test
> dùng `conf=0.001` và `max_det=200`. Các ngưỡng này giữ nguyên cho mọi nhánh.
> AP được tính ở IoU 0,25, 0,50 và 0,75. APmean là trung bình cộng của ba
> ngưỡng này, khớp khóa `AP_mean_IoU_0.25_0.50_0.75` trong artifact. APmean
> không phải AP COCO @[0.50:0.95].
>
> Dải kích thước được xác định bằng căn bậc hai diện tích hộp trong tọa độ
> ảnh gốc, sau khi dự đoán được quy đổi về tọa độ đó. Theo evaluator dùng cho
> AP theo dải, tiny1 từ 1 đến 8, tiny2 từ 8 đến 12, tiny3 từ 12 đến 20,
> small từ 20 đến 32 và reasonable từ 32 trở lên; các cận đều được tính vào
> dải nên hộp đúng ở ranh giới có thể thuộc hai dải. Dải `tiny` là từ 1 đến
> 20. Trên artifact hiện tại, tổng số GT của năm dải bằng 13.687, nên không có
> hộp nào nằm đúng ranh giới. Các dải này không phải AP_S, AP_M, AP_L của
> COCO. Với bảng recall và nhãn size của phân tích lỗi, đối chiếu riêng
> `DIAGNOSTIC_SIZE_BINS` vì cách gán ranh giới trong chẩn đoán dùng cận trên
> không bao gồm.

## 3.3. Kết quả baseline A2

**Vị trí:** thay toàn bộ mục 3.4 hiện tại, đổi tên thành `3.3. Kết quả baseline
A2`.

**Vấn đề:** mục hiện tại chỉ có 87 từ và báo A0, không báo A2. A2 mới là đối
chứng trực tiếp của BTRM v2.

**Nội dung thay thế:**

> Bảng 3.3 báo kết quả của A2 trên validation và official test. Trên
> validation, A2 đạt AP@0.50 0,196888 và AP@0.75 0,015852. Trên official test,
> AP@0.50 tăng lên 0,230141 nhưng AP@0.75 chỉ còn 0,023745. Khoảng cách giữa
> hai ngưỡng cho thấy mô hình phát hiện được vùng chứa người nhưng hộp dự đoán
> chưa đủ chính xác ở IoU cao.

> Bảng 3.3. Baseline A2 trên hai split.

| Split | AP@0.25 | AP@0.50 | AP@0.75 | AP@[0.50:0.95] | APmean | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | Không báo cáo | 0,196888 | 0,015852 | 0,059119 | Không báo cáo | 0,358155 | 0,252494 |
| Official test | 0,401268 | 0,230141 | 0,023745 | Không báo cáo | 0,218384 | 0,050629 | 0,521882 |

> Hai cột AP@[0.50:0.95] và APmean không thay thế nhau. Giá trị 0,059119 trên
> validation là mAP chuẩn của Ultralytics trên các ngưỡng 0,50 đến 0,95. Giá
> trị 0,218384 trên official test là APmean của ba ngưỡng 0,25, 0,50 và 0,75.
> Không so sánh trực tiếp hai giá trị này.

> Precision và recall trên official test được tính ở `conf=0.001`, primary
> IoU 0,50. Precision thấp vì ở ngưỡng tin cậy rất thấp, mô hình tạo nhiều dự
> đoán. Con số này không dùng để đánh giá chất lượng vận hành ở ngưỡng triển
> khai. Nó dùng để so sánh cùng điều kiện giữa các nhánh.

## 3.4. Kết quả BTRM v2 scratch và warm-start

**Vị trí:** thay toàn bộ mục 3.5, gồm Bảng 3.1 cũ và các Hình 3.1, 3.2 cũ.

**Vấn đề:** mục hiện tại so A0, A2, A3 của protocol cũ. Không có BTRM v2,
không có warm-start, không có AP@0.75.

**Nội dung thay thế:**

> Bảng 3.4 so sánh bốn nhánh. Ở cặp scratch, BTRM v2 giảm AP@0.25 từ 0,401268
> xuống 0,376787, giảm AP@0.50 từ 0,230141 xuống 0,211951 và giảm AP@0.75 từ
> 0,023745 xuống 0,020200. APmean giảm 1,5405 điểm phần trăm. Recall giảm từ
> 0,521882 xuống 0,511215.
>
> Ở cặp warm-start, BTRM gần như giữ nguyên AP@0.25, giảm AP@0.50 từ 0,234307
> xuống 0,233549 và giảm AP@0.75 từ 0,024719 xuống 0,019529. APmean giảm
> 0,1934 điểm phần trăm. Recall gần như không đổi.

> Bảng 3.4. Kết quả official test của bốn nhánh.

| Nhánh | AP@0.25 | AP@0.50 | AP@0.75 | APmean | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| A2 scratch | 0,401268 | 0,230141 | 0,023745 | 0,218384 | 0,050629 | 0,521882 |
| BTRM scratch | 0,376787 | 0,211951 | 0,020200 | 0,202979 | 0,048606 | 0,511215 |
| A2 warm-start | 0,402600 | 0,234307 | 0,024719 | 0,220542 | 0,053119 | 0,526631 |
| BTRM warm-start | 0,402747 | 0,233549 | 0,019529 | 0,218608 | 0,053191 | 0,526704 |

> Bảng 3.5. Chênh lệch của BTRM so với đối chứng cùng cặp.

| Cặp | AP@0.25 | AP@0.50 | AP@0.75 | APmean | Recall |
|---|---:|---:|---:|---:|---:|
| Scratch | -0,024481 | -0,018190 | -0,003545 | -0,015405 | -0,010667 |
| Warm-start | +0,000147 | -0,000758 | -0,005190 | -0,001934 | +0,000073 |

> Các giá trị trong Bảng 3.5 là chênh lệch AP dạng phân số. Nhân với 100 để
> đổi sang điểm phần trăm.

> Cặp warm-start giảm nhiễu do khởi tạo Detect khác nhau: hai nhánh nạp cùng
> checkpoint A2. AP@0.75 của BTRM thấp hơn đối chứng khoảng 21% tương đối
> trong cặp này. Một seed và hai kiến trúc huấn luyện khác nhau chưa đủ để
> quy toàn bộ chênh lệch cho riêng tín hiệu tương phản. Kết quả không cho
> thấy cải thiện ở cấp mô hình trong protocol đã chạy.
>
> Validation cho xu hướng khác ở recall. BTRM scratch tăng recall từ 0,252494
> lên 0,265798 nhưng giảm precision từ 0,358155 xuống 0,340439. Mức tăng recall
> không giữ được trên official test. Báo cáo không dùng validation để kết luận
> BTRM cải thiện recall.

**Hình cần chèn:** Hình 3.1, biểu đồ AP@0.25, AP@0.50, AP@0.75 của bốn nhánh.
Vị trí sau Bảng 3.5, trước heading 3.5.

## 3.5. Phân rã TP, miss và FP

**Vị trí:** thêm mục mới sau 3.4.

**Nội dung thay thế:**

> Bảng 3.6 phân rã kết quả official test. BTRM scratch giảm localization FP từ
> 39.121 xuống 37.684, tức giảm 1.437 dự đoán có hộp chưa khớp. Tuy nhiên,
> background FP tăng từ 84.359 lên 89.045, tức tăng 4.686. Số TP giảm 146 và
> số miss tăng 146. Tổng FP tăng 3.015.
>
> Số localization FP giảm đồng thời với background FP tăng trong cặp scratch.
> Các số đếm tổng hợp không chứng minh từng dự đoán đã chuyển từ loại lỗi này
> sang loại lỗi khác. Vì vậy không thể nói BTRM cải thiện localization nếu
> chỉ nhìn số localization FP.

> Bảng 3.6. Phân rã lỗi trên official test.

| Chỉ số | A2 scratch | BTRM scratch | A2 warm-start | BTRM warm-start |
|---|---:|---:|---:|---:|
| TP | 7.143 | 6.997 | 7.208 | 7.209 |
| Miss | 6.544 | 6.690 | 6.479 | 6.478 |
| FP | 133.941 | 136.956 | 128.487 | 128.321 |
| Localization FP | 39.121 | 37.684 | 39.910 | 39.950 |
| Background FP | 84.359 | 89.045 | 77.917 | 77.720 |
| Duplicate FP | 10.461 | 10.227 | 10.660 | 10.651 |
| Recall | 0,521882 | 0,511215 | 0,526631 | 0,526704 |
| Precision | 0,050629 | 0,048606 | 0,053119 | 0,053191 |

> Ở cặp warm-start, tổng FP giảm 166 và miss giảm 1. Các thay đổi này rất nhỏ
> so với quy mô 13.687 GT. AP@0.75 vẫn giảm 0,5190 điểm phần trăm. Vì vậy
> không có bằng chứng cho thấy BTRM cải thiện cấu trúc lỗi tổng thể.

**Hình cần chèn:** Hình 3.2, stacked bar TP, miss, localization FP, background
FP và duplicate FP cho bốn nhánh. Vị trí sau Bảng 3.6, trước heading 3.6.

## 3.6. Kết quả theo kích thước

**Vị trí:** thêm mục mới sau 3.5.

**Nội dung thay thế:**

> Bảng 3.7 báo AP@0.75 theo năm dải kích thước. Kết quả không đồng nhất. BTRM
> scratch tăng AP@0.75 ở tiny1 từ 0,003452 lên 0,005055 và ở tiny2 từ 0,011632
> lên 0,015676. Mức tăng này không giữ được ở tiny3 và small. Tiny3 giảm từ
> 0,022933 xuống 0,015623, small giảm từ 0,032881 xuống 0,024034.
>
> Ở cặp warm-start, BTRM giữ gần như nguyên tiny1, tiny2 và tiny3 nhưng
> reasonable giảm từ 0,042397 xuống 0,036896. Kết quả theo dải không đủ để
> quy toàn bộ mức giảm AP@0.75 tổng cho riêng dải reasonable, vì AP tổng được
> tính lại trên toàn tập dự đoán và đối tượng.

> Bảng 3.7. AP@0.75 theo dải kích thước.

| Dải | A2 scratch | BTRM scratch | A2 warm-start | BTRM warm-start |
|---|---:|---:|---:|---:|
| all | 0,023745 | 0,020200 | 0,024719 | 0,019529 |
| tiny | 0,017398 | 0,010198 | 0,018096 | 0,018067 |
| tiny1 | 0,003452 | 0,005055 | 0,005519 | 0,005851 |
| tiny2 | 0,011632 | 0,015676 | 0,017893 | 0,017737 |
| tiny3 | 0,022933 | 0,015623 | 0,022976 | 0,023013 |
| small | 0,032881 | 0,024034 | 0,033673 | 0,033899 |
| reasonable | 0,041583 | 0,039157 | 0,042397 | 0,036896 |

> Bảng 3.8 báo recall theo dải. BTRM scratch giảm recall ở tiny1, tiny2, tiny3
> và small. BTRM warm-start tăng nhẹ tiny1 và tiny3 nhưng giảm tiny2 và small.
> Không có dải nào cho thấy mức tăng ổn định trên cả hai cặp.

> Bảng 3.8. Recall theo dải kích thước.

| Dải | A2 scratch | BTRM scratch | A2 warm-start | BTRM warm-start |
|---|---:|---:|---:|---:|
| tiny1 | 0,2105 | 0,1980 | 0,2105 | 0,2126 |
| tiny2 | 0,4737 | 0,4478 | 0,4590 | 0,4582 |
| tiny3 | 0,5636 | 0,5500 | 0,5641 | 0,5644 |
| small | 0,6414 | 0,6374 | 0,6584 | 0,6559 |
| reasonable | 0,6966 | 0,7035 | 0,7233 | 0,7246 |

**Hình cần chèn:** Hình 3.3, AP@0.75 theo dải kích thước. Vị trí sau Bảng 3.7,
trước Bảng 3.8.

## 3.7. Hình học hộp và ngưỡng confidence

**Vị trí:** thêm mục mới sau 3.6.

**Nội dung thay thế:**

> Bảng 3.9 báo trung vị của width ratio và center offset trên các localization
> FP có điểm ít nhất 0,05 và IoU trong khoảng [0,10; 0,50). Theo định nghĩa
> trong `docs/BTRM_REPORT.md`, width ratio là `pred_width / GT_width`; giá trị
> nhỏ hơn 1 nghĩa là hộp dự đoán hẹp hơn GT. Center offset là khoảng cách tâm
> chia cho cạnh lớn hơn của GT. Khi trích lại từ `errors.jsonl`, phải dùng
> đúng định nghĩa này để không trộn với tỷ số căn bậc hai diện tích.
>
> Hộp lỗi trung vị hẹp hơn GT ở cả ba dải. BTRM scratch kéo width ratio của
> tiny1 từ 0,747 lên 0,842 và tiny2 từ 0,883 lên 0,985, gần 1 hơn. Tuy nhiên,
> center offset gần như không đổi, khoảng 0,24 đến 0,26. Việc sửa chiều rộng
> mà không sửa tâm không đủ để tăng AP@0.75.

> Bảng 3.9. Hình học của localization FP.

| Nhánh | Dải | n | Width ratio trung vị | Center offset trung vị |
|---|---|---:|---:|---:|
| A2 | tiny1 | 466 | 0,747 | 0,247 |
| A2 | tiny2 | 2.273 | 0,883 | 0,242 |
| A2 | tiny3 | 5.223 | 0,908 | 0,249 |
| BTRM scratch | tiny1 | 437 | 0,842 | 0,244 |
| BTRM scratch | tiny2 | 2.151 | 0,985 | 0,243 |
| BTRM scratch | tiny3 | 4.849 | 0,972 | 0,260 |
| A2 warm-start | tiny1 | 488 | 0,701 | 0,249 |
| A2 warm-start | tiny2 | 2.156 | 0,874 | 0,242 |
| A2 warm-start | tiny3 | 5.168 | 0,945 | 0,250 |
| BTRM warm-start | tiny1 | 482 | 0,703 | 0,250 |
| BTRM warm-start | tiny2 | 2.160 | 0,866 | 0,241 |
| BTRM warm-start | tiny3 | 5.164 | 0,956 | 0,249 |

> Bảng 3.10 báo recall và precision tại tám ngưỡng confidence. BTRM scratch
> thua A2 về recall ở mọi ngưỡng. Precision chỉ cao hơn ở ngưỡng 0,5. BTRM
> warm-start gần như trùng đối chứng ở mọi ngưỡng. Không có ngưỡng nào cho
> thấy lợi thế vận hành ổn định.

> Bảng 3.10. Recall và precision theo ngưỡng confidence, định dạng
> recall/precision.

| Ngưỡng | A2 scratch | BTRM scratch | A2 warm-start | BTRM warm-start |
|---:|---:|---:|---:|---:|
| 0,001 | 0,5219/0,0506 | 0,5112/0,0486 | 0,5266/0,0531 | 0,5267/0,0532 |
| 0,005 | 0,5152/0,0717 | 0,5041/0,0636 | 0,5182/0,0746 | 0,5190/0,0748 |
| 0,010 | 0,5076/0,0882 | 0,4954/0,0787 | 0,5103/0,0915 | 0,5116/0,0916 |
| 0,020 | 0,4927/0,1125 | 0,4802/0,1018 | 0,4989/0,1150 | 0,4992/0,1147 |
| 0,050 | 0,4554/0,1626 | 0,4407/0,1532 | 0,4607/0,1603 | 0,4620/0,1602 |
| 0,100 | 0,3943/0,2321 | 0,3777/0,2277 | 0,4031/0,2237 | 0,4041/0,2234 |
| 0,200 | 0,2713/0,4032 | 0,2467/0,3993 | 0,2832/0,3836 | 0,2841/0,3822 |
| 0,500 | 0,0252/0,7913 | 0,0180/0,8098 | 0,0288/0,8191 | 0,0283/0,8134 |

**Hình cần chèn:** Hình 3.4, phân bố width ratio và center offset, sau Bảng
3.9 và trước Bảng 3.10; Hình 3.5, precision và recall theo confidence, sau
Bảng 3.10 và trước heading 3.8. Không tạo box plot từ các giá trị trung vị
trong Bảng 3.9; cần dữ liệu từng hộp trong `errors.jsonl`.

## 3.8. Kiểm tra hiện thực và ablation gamma

**Vị trí:** thay mục 3.6 cũ, đổi tên thành `3.8. Kiểm tra hiện thực và ablation
gamma`.

**Vấn đề:** mục cũ chỉ nói kiểm tra đại số định tính và AUPRC chưa hoàn tất.
Không có kết quả kiểm tra cụ thể.

**Nội dung thay thế:**

> Các bất biến ở mục 2.7 được kiểm tra bằng unit test trên CPU. Kết quả gồm:
> đầu ra bằng đầu vào khi gamma bằng 0; nền hằng số cho phần dư bằng 0;
> gradient chảy qua gamma và phép chiếu; autocast giữ dtype; đồ thị model có
> BTRM ở layer 29; Detect nhận đúng bốn nhánh; checkpoint lưu và nạp lại được.
> Các kiểm tra này xác nhận hiện thực đúng thiết kế, không xác nhận mô-đun có
> ích cho phát hiện.
>
> Bảng 3.11 so sánh cùng một checkpoint BTRM scratch với gamma học được và
> gamma bằng 0. Gamma học được bằng 0,1226. Khi ép gamma bằng 0, AP@0.50 giảm
> 0,6311 điểm phần trăm, AP@0.75 tăng 0,0747 điểm phần trăm và APmean giảm
> 0,0863 điểm phần trăm. Chênh lệch nhỏ và không nhất quán giữa các ngưỡng.
> Không thể dùng ablation này để nói phần dư có ích.

> Bảng 3.11. Ablation gamma trên cùng checkpoint BTRM scratch.

| Cấu hình | AP@0.50 | AP@0.75 | APmean |
|---|---:|---:|---:|
| Gamma học được, 0,1226 | 0,211951 | 0,020200 | 0,202979 |
| Gamma bằng 0 | 0,205640 | 0,020947 | 0,202116 |
| Chênh lệch | -0,006311 | +0,000747 | -0,000863 |

> Bảng 3.12 báo phân rã lỗi của nhánh gamma bằng 0. So với BTRM học gamma,
> nhánh này giảm 6.908 FP nhưng tăng 186 miss và giảm recall từ 0,511215 xuống
> 0,497625. Đây là đánh đổi recall lấy precision, không phải cải thiện chất
> lượng hộp.

> Bảng 3.12. Phân rã lỗi của nhánh gamma bằng 0.

| Chỉ số | BTRM học gamma | BTRM gamma 0 |
|---|---:|---:|
| TP | 6.997 | 6.811 |
| Miss | 6.690 | 6.876 |
| FP | 136.956 | 130.048 |
| Localization FP | 37.684 | 36.739 |
| Background FP | 89.045 | 83.239 |
| Duplicate FP | 10.227 | 10.070 |
| Recall | 0,511215 | 0,497625 |
| Precision | 0,048606 | 0,049767 |

## 3.9. Chi phí và thảo luận

**Vị trí:** thay mục 3.7 cũ, đổi tên thành `3.9. Chi phí và thảo luận`.

**Nội dung thay thế:**

> Bảng 3.13 báo chi phí trên L4 theo `docs/BTRM_REPORT.md`, batch 1, ảnh 1.280.
> BTRM thêm 2.609 tham số nhưng làm latency tăng 7,6%, FPS giảm 7,1%, bộ nhớ
> benchmark tăng 60,3% và thời gian huấn luyện tăng 41,2%. Với cặp warm-start,
> latency tăng khoảng 14%, FPS giảm 12,3% và bộ nhớ tăng 46,7%. Các phần trăm
> lấy từ nguồn, không tính lại từ số đã làm tròn. Chưa thấy artifact benchmark
> riêng trong `outputs/`; phải lưu raw log và chốt GPU theo metadata run trước
> khi công bố các số này.
>
> Chi phí này không đi kèm cải thiện AP@0.75. Vì vậy không nên tăng thêm kênh,
> độ sâu hoặc attention vào cùng họ BTRM trước khi có bằng chứng mới.

> Bảng 3.13. Chi phí tính toán.

| Chỉ số | A2 | BTRM scratch | BTRM warm-start |
|---|---:|---:|---:|
| Tham số | 2.515.976 | 2.518.585 | 2.518.585 |
| Tham số thêm | 0 | 2.609 | 2.609 |
| Latency | 16,37 ms | 17,62 ms | 18,74 ms |
| FPS | 61,08 | 56,76 | 53,35 |
| Bộ nhớ benchmark | 214,8 MB | 344,3 MB | 315,1 MB |
| Thời gian huấn luyện | 1.781 s | 2.515 s | Không có |

> Phân rã lỗi và hình học hộp gợi ý nên thử cải thiện hồi quy hoặc gán nhãn
> như một giả thuyết tiếp theo, chưa xác định được nguyên nhân chính. Trong
> cặp scratch, localization FP giảm còn background FP tăng; hộp lỗi trung vị
> hẹp hơn GT và tâm vẫn lệch khoảng 0,24 đến 0,26 cạnh GT. Những số liệu này
> không loại trừ vai trò của biểu diễn đặc trưng hay điểm confidence.
>
> Hướng tiếp theo hợp lý hơn là cải thiện hồi quy hộp và gán nhãn. Các ứng
> viên gồm RFLA, NWD hoặc một đầu hồi quy phân bố. Báo cáo không thực hiện các
> thí nghiệm đó, nên chỉ nêu như hướng nghiên cứu.

## 3.10. Đe dọa đến tính hợp lệ

**Vị trí:** thêm mục mới sau 3.9.

**Nội dung thay thế:**

> Kết quả chỉ có một seed. Không có mean và độ lệch chuẩn qua nhiều lần khởi
> tạo, nên không thể kết luận mức chênh lệch có ý nghĩa thống kê. Chênh lệch
> AP@0.75 khoảng 0,003 đến 0,005 có thể thay đổi nếu chạy seed khác.
>
> Evaluator là implementation nội bộ của repo. Cross-check giữa hai
> implementation nội bộ khớp tuyệt đối, nhưng chưa có so sánh với evaluator
> tham chiếu của TinyPerson. Các số AP không nên dùng để so với bảng công bố.
>
> Bản derivative YOLO không giữ đầy đủ metadata ignore, uncertain và dense.
> Validation trên derivative chỉ dùng để chọn checkpoint. Kết quả official
> test phụ thuộc vào cách evaluator xử lý ba loại metadata này.
>
> Chưa có AUPRC theo dải kích thước. Vì vậy H01 chưa được kiểm định ở cấp đặc
> trưng. Kết quả cấp mô hình không đủ để nói tín hiệu tương phản cục bộ không
> có thông tin. Nó chỉ cho thấy cách hiện thực hiện tại không chuyển được
> thông tin đó thành AP@0.75.
