# Quy tắc văn phong và trích dẫn

Áp dụng cho mọi văn bản học thuật: báo cáo đồ án, bài báo IEEE, báo cáo nghiên cứu.

## 1. Văn phong

### 1.1 Cấm dùng
Các từ và cụm từ sau không được xuất hiện trong bản nộp:

- Quảng cáo: đột phá, vượt trội, cách mạng, hoàn hảo, tuyệt vời, xuất sắc, tối ưu tuyệt đối, mang tính cách mạng.
- Sáo rỗng: đi sâu, then chốt, bức tranh toàn cảnh, trong thời đại số, kỷ nguyên, là một bước tiến quan trọng, không thể phủ nhận, vô cùng.
- Văn nói: chúng ta, bạn có thể, rất nhiều, hiển nhiên.
- Dấu "!" trong thân bài.

### 1.2 Thận trọng với khẳng định tuyệt đối
Các từ sau chỉ dùng khi có bằng chứng trực tiếp cho đúng phạm vi đó: luôn luôn, không bao giờ, chắc chắn, tuyệt đối, hoàn toàn, duy nhất, tốt nhất, không thể, chứng minh, mọi, tất cả, chưa từng.

Thay bằng mức độ phù hợp với bằng chứng:

| Thay vì | Dùng |
|---|---|
| "Phương pháp chứng minh rằng..." | "Kết quả thực nghiệm cho thấy..." |
| "Phương pháp tốt nhất" | "Trong tập dữ liệu X, phương pháp đạt F1 cao nhất trong các baseline được thử nghiệm" |
| "Luôn luôn đúng" | "Trong các điều kiện khảo sát, ..." |
| "Không có nghiên cứu nào..." | "Trong phạm vi tìm kiếm của đồ án (Mục 2.1), chưa tìm thấy nghiên cứu nào..." |
| "Hoàn toàn chính xác" | "Đạt độ chính xác 92,4% trên tập kiểm tra" |

### 1.3 Nhất quán
- Mức độ khẳng định phải giống nhau giữa tóm tắt, kết quả, kết luận. Nếu kết quả là "khá tốt" thì không viết "xuất sắc" ở kết luận.
- Thuật ngữ chỉ có một tên trong toàn bài. Một đại lượng có một ký hiệu, định nghĩa ở lần xuất hiện đầu.
- Viết tắt định nghĩa ở lần đầu: "Mạng nơ-ron tích chập (CNN)", sau đó chỉ dùng CNN.

### 1.4 Ngôi và thì
- Ngôi: "chúng tôi" hoặc câu bị động, chọn một và giữ đến hết bài.
- Thì hiện tại cho kiến thức đã được thiết lập ("CNN sử dụng các lớp tích chập [3]").
- Thì quá khứ cho việc đã thực hiện ("Chúng tôi huấn luyện mô hình trên...").

### 1.5 Định dạng
- Viết thành đoạn văn. Hạn chế gạch đầu dòng trong thân bài; dùng khi thật sự là danh sách (ví dụ: các bước thuật toán).
- Không in đậm lạm dụng. Không dùng in đậm để nhấn mạnh trong văn bản học thuật.
- Không dùng tiêu đề kiểu slide ("Tại sao điều này quan trọng?").

## 2. Trích dẫn

### 2.1 Quy tắc cốt lõi
Mỗi câu khẳng định hoặc phủ định về thế giới bên ngoài phải thuộc một trong ba loại:

1. **Có trích dẫn [n]** đến nguồn đã được đọc và kiểm tra.
2. **Có tham chiếu nội bộ** đến kết quả trong chính bài: "xem Bảng 3", "như trình bày ở Mục 4.2", "Hình 5 cho thấy...".
3. **Là kết quả của đồ án này** và có mô tả cách đo (tập dữ liệu, cấu hình, số lần lặp). Trường hợp này không cần [n], nhưng phải có chỗ chứng minh trong bài.

Nếu không rơi vào ba loại trên, câu đó phải được gắn `[CẦN NGUỒN]` hoặc sửa mức độ khẳng định, hoặc xóa.

### 2.2 Phủ định
Câu phủ định như "chưa có nghiên cứu nào giải quyết..." là khẳng định mạnh. Chỉ dùng khi có tìm kiếm literature có mô tả (từ khóa, cơ sở dữ liệu, khoảng thời gian). Nếu không có, viết: "Theo tìm hiểu của chúng tôi trong các cơ sở dữ liệu [X], Y là..." hoặc hạ mức khẳng định.

### 2.3 Số liệu
- Mọi con số có đơn vị: "120 ms", "92,4%", "3,2 GB".
- Mỗi số đến từ một trong hai nguồn: kết quả tự đo (có file/log/script sinh ra) hoặc trích dẫn [n]. Không có nguồn thứ ba.
- Số trong tóm tắt, kết quả, kết luận phải khớp nhau. Lấy từ cùng một file kết quả.
- Ghi rõ điều kiện: tập dữ liệu, số lần chạy, seed, độ đo (accuracy hay F1, macro hay micro).

### 2.4 Phân biệt đóng góp của mình và của người khác
- Của mình: "Đề xuất của chúng tôi...", "Trong nghiên cứu này, chúng tôi...".
- Của người khác: "Phương pháp X [5] đạt...", "Theo [7], ...".
- Không viết câu có cả hai trong cùng một mệnh đề nếu có thể gây nhầm lẫn.

### 2.5 Đánh số (IEEE)
- Số [n] được đánh theo thứ tự xuất hiện lần đầu trong văn bản. Trích [1] trước [2]...
- Mỗi tài liệu trong danh mục phải được trích ít nhất một lần.
- Gộp nhiều nguồn: [2], [5] hoặc [2]–[4].
- Không trích tài liệu chỉ dựa vào tiêu đề. Phải đọc ít nhất phần tóm tắt và đoạn liên quan.

### 2.6 Nguồn tài liệu
- Chỉ trích tài liệu đã đọc. Có ghi chú "đã đọc" trong Zotero/Mendeley.
- Ưu tiên nguồn gốc: bài báo gốc, tiêu chuẩn, tài liệu chính thức của nhà cung cấp.
- Với bài báo, ưu tiên bản có DOI. Kiểm tra DOI bằng `verify_refs.py` hoặc Crossref.
- Tài liệu cũ (dưới 5 năm) cho trạng thái hiện tại của lĩnh vực nhanh thay đổi. Nếu dùng tài liệu cũ, nói rõ đó là kiến thức nền.
- Không dùng Wikipedia, blog, bài báo chưa qua bình duyệt làm nguồn cho khẳng định học thuật, trừ khi chỉ dùng cho định nghĩa đơn giản và có ghi chú.

### 2.7 Không được làm
- Không tự sinh danh mục tài liệu bằng AI. Danh mục lấy từ Zotero/Mendeley hoặc file `.bib`.
- Không viết một tài liệu với năm, tác giả, DOI không kiểm chứng được.
- Không paraphrase sát từng chữ của nguồn. Viết lại bằng cách diễn đạt của mình và trích dẫn.
- Không giữ `[CẦN NGUỒN]`, `[CHECK]`, `[TODO]`, `[REMOVE]` trong bản nộp.

## 3. Phần hạn chế và phạm vi
Mỗi báo cáo nghiên cứu phải có:
- Phạm vi: dữ liệu nào, điều kiện nào, kết quả chỉ áp dụng trong đó.
- Hạn chế: thiếu gì (dữ liệu, thời gian, cấu hình), vì sao có thể ảnh hưởng đến kết quả.
- Đe dọa tính hợp lệ (nếu có): rò rỉ dữ liệu, tune siêu tham số trên tập kiểm tra, baseline yếu.

Không viết hạn chế kiểu "có thể cải thiện trong tương lai" như câu duy nhất.

## 4. Kỹ thuật và tái lập
- Mô tả đủ để người khác chạy lại: phiên bản thư viện, siêu tham số, cách chia dữ liệu, seed, phần cứng khi đo thời gian.
- Baseline phải được cấu hình công bằng. Nếu tune baseline ít hơn, nói rõ.
- Đơn vị nhất quán: ms hoặc s, không trộn. Accuracy và F1 không thay thế nhau.
- Không dùng thuật ngữ sai: nhầm train/test, nhầm độ đo, nhầm "độ chính xác" với "độ đúng đắn của mô hình".

## 5. Định dạng

### 5.1 IEEE
- Dùng template chính thức (LaTeX `IEEEtran` hoặc template Word IEEE).
- Caption hình đặt dưới hình; caption bảng đặt trên bảng.
- Danh mục theo kiểu IEEE, sinh từ BibTeX/Zotero.

### 5.2 DOCX cho đồ án
- Tiêu đề dùng style Heading 1/2/3 của Word, để mục lục tự động hoạt động.
- Hình, bảng đánh số bằng Insert Caption; tham chiếu bằng Cross-reference.
- Một font, một cỡ chữ cho thân bài. Kiểm tra tiếng Việt có dấu sau khi copy từ AI.
- Bảng không tràn lề. Công thức dùng Equation editor, không gõ bằng text thường.
