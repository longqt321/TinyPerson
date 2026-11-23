# CẨM NANG TOÀN DIỆN: KỸ THUẬT SOẠN THẢO BÁO CÁO KHOA HỌC & ĐỒ ÁN KỸ THUẬT BẰNG AI
*(The Comprehensive Handbook for AI-Assisted Academic & Engineering Report Generation)*

---

## NGUYÊN TẮC TIÊN QUYẾT: TRÁNH AI SLOP & VIẾT TỰ NHIÊN NHƯ CON NGƯỜI

> [!IMPORTANT]
> **TIÊN ĐỀ VÀNG (GOLDEN AXIOM):**
> 1. **Văn phong người viết (Human Voice) là yêu cầu tối thượng:** Mọi quy tắc kỹ thuật (IMRaD, nhịp điệu Provost, số lượng từ, cấm từ, dấu câu) đều chỉ là công cụ phục vụ mục tiêu viết như một chuyên gia nghiên cứu con người thực thụ.
> 2. **Tuyệt đối không áp dụng máy móc, rập khuôn:** Khi các quy tắc phụ xung đột với tính tự nhiên, logic học thuật hoặc mạch tư duy của văn bản, AI **phải tư duy phản biện (critical thinking) và ưu tiên tính tự nhiên của ngữ cảnh**, tuyệt đối không được "răm rắp nghe theo" một cách máy móc dẫn đến phản tác dụng.
> 3. **Nhận diện và triệt tiêu AI Slop dạng tinh vi:**
>    - *AI Slop không chỉ là những từ hoa mỹ sáo rỗng* (như "đột phá", "bức tranh toàn cảnh", "hơn nữa", "tóm lại").
>    - *AI Slop còn là những câu khẩu hiệu đối lập nhân tạo* được cố tình nhét vào để thỏa mãn quy tắc câu ngắn/nhịp điệu: *"Tham số rẻ, bộ nhớ thì không."*, *"Recall lên, precision xuống."*, *"X, không phải Y, là nút thắt."*, *"Đây là điều kiện cần, chưa phải điều kiện đủ."*
>    - Những câu gượng gạo này làm mất đi tính nghiêm túc của báo cáo khoa học. Người nghiên cứu thực thụ luôn diễn đạt bằng thực tế khách quan và dữ liệu có chủ ngữ, vị ngữ rõ ràng (ví dụ: *"Mức tăng bộ nhớ này lớn hơn nhiều so với tỷ lệ +1,47% tham số do kích thước bản đồ đặc trưng trung gian."* thay vì *"Tham số rẻ, bộ nhớ thì không."*).

---

## MỤC LỤC
1. [Nguyên tắc Tiên quyết: Tránh AI Slop & Viết Tự nhiên như Con người](#nguyên-tắc-tiên-quyết-tránh-ai-slop--viết-tự-nhiên-như-con-người)
2. [Phần 1: Cấu trúc Vĩ mô (Macro-Structure): Mô hình IMRaD & Tỷ lệ Phân bổ Dung lượng](#phần-1-cấu-trúc-vĩ-mô-macro-structure-mô-hình-imrad--tỷ-lệ-phân-bổ-dung-lượng)
3. [Phần 2: Nhận diện Ranh giới Chức năng: Đâu là Method, Đâu là Results? (CARS Model)](#phần-2-nhận-diện-ranh-giới-chức-năng-đâu-là-method-đâu-là-results-cars-model)
4. [Phần 3: Nhịp điệu Văn bản (Burstiness) & Đa dạng hóa Cấu trúc Câu](#phần-3-nhịp-điệu-văn-bản-burstiness--đa-dạng-hóa-cấu-trúc-câu)
5. [Phần 4: Chuẩn mực Dấu câu & Quy tắc dùng Gạch nối / Gạch ngang (Hyphen, En Dash, Em Dash)](#phần-4-chuẩn-mực-dấu-câu--quy-tắc-dùng-gạch-nối--gạch-ngang-hyphen-en-dash-em-dash)
6. [Phần 5: Văn phong Khoa học Thực sự trông như thế nào? (Đối chiếu Before / After)](#phần-5-văn-phong-khoa-học-thực-sự-trông-như-thế-nào-đối-chiếu-before--after)
7. [Phần 6: Danh mục 30 Lỗi Kinh điển Cần tránh khi Viết Báo cáo (Zero-Overlap)](#phần-6-danh-mục-30-lỗi-kinh-điển-cần-tránh-khi-viết-báo-cáo-zero-overlap)
8. [Phần 7: Bộ Quy tắc Prompting Ép AI Viết Đúng Chuẩn Học thuật (Anti-AI Slop)](#phần-7-bộ-quy-tắc-prompting-ép-ai-viết-đúng-chuẩn-học-thuật-anti-ai-slop)
9. [Phần 8: Công cụ & Script Python Tạo File DOCX/PDF Chuẩn Đẹp 100%](#phần-8-công-cụ--script-python-tạo-file-docxpdf-chuẩn-đẹp-100)
10. [Phần 9: Bảng Kiểm tra Chất lượng Trước khi Nộp (Pre-submission Checklist)](#phần-9-bảng-kiểm-tra-chất-lượng-trước-khi-nộp-pre-submission-checklist)

---

## PHẦN 1: CẤU TRÚC VĨ MÔ (MACRO-STRUCTURE): MÔ HÌNH IMRAD & TỶ LỆ PHÂN BỔ DUNG LƯỢNG

Một bài báo khoa học hay báo cáo đồ án kỹ thuật chuẩn mực quốc tế đều vận hành theo **mô hình đồng hồ cát (Hourglass IMRaD)**: Mở rộng ở bối cảnh chung $\rightarrow$ Thu hẹp vào phương pháp và dữ liệu cụ thể $\rightarrow$ Mở rộng ra ý nghĩa và tác động tổng thể.

### 1. Bảng Tỷ lệ Phân bổ Dung lượng Tiêu chuẩn (Dành cho Báo cáo / Khóa luận)

| Phần trong báo cáo | Tỷ lệ % dung lượng | Chức năng học thuật cốt lõi | Nội dung bắt buộc phải có |
|---|:---:|---|---|
| **Tóm tắt (Abstract)** | 2–3% (200–350 từ) | Bản thu nhỏ độc lập của toàn bộ công trình | Vấn đề $\rightarrow$ Phương pháp $\rightarrow$ Kết quả cốt lõi $\rightarrow$ Đóng góp |
| **Chương 1: Mở đầu & Tổng quan (Introduction & Related Work)** | **15–20%** | Bối cảnh rộng $\rightarrow$ Thu hẹp vào khoảng trống (gap) | Động lực bài toán, phân loại các hướng tiếp cận, nhược điểm của công trình trước, tuyên bố mục tiêu (SMART) |
| **Chương 2: Phương pháp đề xuất (Methodology / System Architecture)** | **25–30%** | Chi tiết kỹ thuật phục vụ tái lập thực nghiệm | Mô hình toán học, luồng tensor, cấu trúc khối, siêu tham số thiết kế, căn cứ lựa chọn giải pháp (Trade-off) |
| **Chương 3: Thực nghiệm & Đánh giá (Experiments & Evaluation / Results)** | **25–30%** | Dữ liệu khách quan, bằng chứng định lượng | Giao thức kiểm thử, bảng kết quả định lượng, so sánh baseline, phân tích lỗi (failure modes), kiểm định ablation |
| **Chương 4: Bàn luận & Kết luận (Discussion & Conclusion)** | **15–20%** | Mở rộng ý nghĩa $\rightarrow$ Nhìn nhận hạn chế | Trả lời trực diện mục tiêu ở Chương 1, giải thích cơ chế thành công/thất bại, nêu hạn chế thực tế, hướng phát triển |

> **Quy tắc vàng về phân bổ:** 
> Chương 2 (Phương pháp) và Chương 3 (Kết quả) luôn là "trọng tâm kỹ thuật", chiếm từ **50% đến 60%** tổng dung lượng toàn bài. Tránh tuyệt đối tình trạng Chương Tổng quan dài 40% mà phần kết quả tự làm chỉ chiếm 10-15%.

---

## PHẦN 2: NHẬN DIỆN RANH GIỚI CHỨC NĂNG: ĐÂU LÀ METHOD, ĐÂU LÀ RESULTS? (CARS MODEL)

Một lỗi rất phổ biến ở các bài viết do AI sinh ra là **trộn lẫn phương pháp vào phần kết quả, hoặc đem kết quả bàn luận ngược lên phần phương pháp**. Dựa trên mô hình phân tích thể loại **CARS (Create a Research Space)** của John Swales, ranh giới giữa hai phần này được phân định tuyệt đối như sau:

### 1. Bảng Đối chiếu Nhận diện Ngôn ngữ & Chức năng

| Tiêu chí | Phần Phương pháp (Methodology) | Phần Kết quả (Results) |
|---|---|---|
| **Câu hỏi cốt lõi** | *"Tác giả đã làm việc đó như thế nào?"* | *"Dữ liệu/thực nghiệm đã phát hiện ra điều gì?"* |
| **Hành vi văn bản (Move)** | Mô tả thao tác, tham số, thiết lập, công thức | Công bố con số đo đạc, xu hướng đồ thị, so sánh sai khác |
| **Động từ đặc trưng** | *Thiết kế, tích chập, khởi tạo, chiếu tuyến tính, chuẩn hóa, phân tách, huấn luyện, cắt mảnh, trích xuất* | *Đạt được, tăng/giảm, ghi nhận, bộc lộ, suy giảm, cho thấy, tương quan, vượt trội* |
| **Danh từ đặc trưng** | *Tensor, kênh (channel), độ lệch chuẩn $\sigma$, hàm mất mát, learning rate, kernel, batch size, epoch* | *AP50, mAP, độ trễ (latency), miss rate (MR), khoảng tin cậy (CI), độ lệch chuẩn, $p$-value* |
| **Thì của động từ (Tense)** | **Quá khứ đơn** (thao tác đã thực hiện) hoặc **Hiện tại đơn** (khi mô tả cơ chế toán học cố định của khối) | **Quá khứ đơn** (kết quả đo đạc từ thí nghiệm cụ thể đã chạy) |

### 2. Ví dụ Đối chiếu Nhận diện Tức thì
- 🔴 **Câu Method:** *"Chúng tôi sử dụng hai toán tử lọc Gauss với bán kính hỗ trợ lần lượt là 2 và 4 để phân tách đặc trưng stride 4 thành ba dải tần số riêng biệt."* $\rightarrow$ Đây là **cách thức thiết lập**, phải nằm ở Chương 2.
- 🟢 **Câu Result:** *"Khi loại bỏ nhánh dải tần giữa trong thực nghiệm cắt mảnh, điểm AP50 suy giảm 4,66 điểm phần trăm với khoảng tin cậy bootstrap [−6,40; −2,75]."* $\rightarrow$ Đây là **kết quả đo đạc thu được**, phải nằm ở Chương 3.

---

## PHẦN 3: NHỊP ĐIỆU VĂN BẢN (BURSTINESS) & ĐA DẠNG HÓA CẤU TRÚC CÂU

### 1. Điểm yếu đồng đều cấu trúc của AI (Low Burstiness)
Mô hình ngôn ngữ lớn (LLM) có xu hướng sinh ra các câu có độ dài đều đặn một cách máy móc (thường từ 20 đến 25 từ mỗi câu), cấu trúc lặp đi lặp lại: `[Trạng từ nối], [Chủ ngữ] + [Động từ] + [Bổ ngữ dài]`. Điều này tạo ra một nhịp điệu đều đều, gây mệt mỏi nhận thức (cognitive fatigue) cho người chấm đồ án hoặc người bình duyệt.

### 2. Bài học nhịp điệu của Gary Provost trong Văn phong Khoa học
Một văn bản khoa học xuất sắc cần có sự giao thoa nhịp nhàng giữa **câu đơn ngắn (Short)**, **câu phức trung bình (Medium)** và **câu ghép đa tầng (Long)**:

> **Đoạn văn minh họa nhịp điệu (Burstiness in Academic Prose):**
> 
> *(Câu ngắn - 7 từ)* **Thực nghiệm chứng minh giả thuyết đề xuất.**  
> *(Câu trung bình - 22 từ)* Mô hình tích hợp module đạt mức tăng 0,45 điểm AP50 trên toàn bộ tập dữ liệu kiểm thử ở chế độ cắt mảnh nguyên bản.  
> *(Câu dài nhiều tầng bậc - 38 từ)* Mặc dù các phương pháp tiếp cận truyền thống thường đánh đổi độ trễ tính toán để lấy độ chính xác định vị bằng cách mở rộng các tầng xử lý nặng, kiến trúc đề xuất duy trì chi phí tham số ở mức thấp nhờ cơ chế phân tách dải tần có định hướng.  
> *(Câu ngắn - 5 từ)* **Độ trễ mạng được bảo toàn.**  
> *(Câu trung bình - 20 từ)* Phân tích sâu hơn cho thấy các đối tượng cực nhỏ từ 8 đến 12 pixel nhận được mức cải thiện rõ nét nhất.

### 3. Phân loại & Chức năng của 3 nhóm độ dài câu:

* **Câu ngắn (5–12 từ):** Dùng để khẳng định một kết luận đanh thép, nêu một nhận định then chốt, hoặc làm bản lề chuyển ý giữa hai đoạn phân tích sâu.
  - *Ví dụ chuẩn mực:* *"Cơ chế định tuyến đóng vai trò quyết định."* hoặc *"Sai số này không đáng kể."*
  - ⚠️ **Cảnh báo bẫy AI Slop dạng câu ngắn:** Tuyệt đối không chèn các câu ngắn mang tính khẩu hiệu đối lập nhân tạo hoặc rỗng nghĩa (như *"Tham số rẻ, bộ nhớ thì không."*, *"Recall lên, precision xuống."*, *"X, không phải Y, là nút thắt."*). Câu ngắn của người viết khoa học luôn có thông tin cụ thể hoặc chủ-vị tự nhiên, không phải một slogan quảng cáo.
* **Câu trung bình (15–25 từ):** Đóng vai trò "xương sống" trình bày logic kỹ thuật, mô tả các bước tiến hành hoặc liên kết nguyên nhân - kết quả.
  - *Ví dụ:* *"Bộ lọc Gauss thứ nhất triệt tiêu nhiễu tần số cao của mặt sóng trước khi đặc trưng được đưa vào tầng định tuyến."*
* **Câu dài (30+ từ):** Dùng khi cần thể hiện tư duy đa chiều: tổng hợp lý thuyết, so sánh các biến phụ thuộc kèm theo điều kiện biên hoặc các ràng buộc kỹ thuật phức tạp.

---

## PHẦN 4: CHUẨN MỰC DẤU CÂU & QUY TẮC DÙNG GẠCH NỐI / GẠCH NGANG (HYPHEN, EN DASH, EM DASH)

Trong văn bản học thuật quốc tế (IEEE, Chicago Style) và văn bản kỹ thuật tiếng Việt, việc phân biệt ba loại dấu gạch là thước đo sự chuyên nghiệp của tác giả.

### 1. Bảng Phân biệt 3 Loại Dấu Gạch

| Đặc điểm | Dấu gạch nối (Hyphen) | Dấu gạch ngang trung bình (En Dash) | Dấu gạch ngang dài (Em Dash) |
|---|:---:|:---:|:---:|
| **Ký tự** | `-` | `–` | `—` |
| **Mã Unicode** | `U+002D` | `U+2013` | `U+2014` |
| **Bề rộng tương đương** | Bằng bề rộng chữ nối | Bằng bề rộng chữ `N` | Bằng bề rộng chữ `M` |
| **Phím tắt** | Phím `-` trên bàn phím | `Alt + 0150` (Win) / `Option + -` (Mac) | `Alt + 0151` (Win) / `Shift + Option + -` (Mac) |

### 2. Quy tắc Sử dụng Chi tiết & Hạn chế Em Dash

#### A. Dấu gạch nối (Hyphen: `-`)
* **Chức năng:** Dùng để **nối từ** (từ ghép phiên âm, danh từ ghép kỹ thuật tiếng Anh).
* **Quy tắc khoảng trắng:** **Tuyệt đối không có khoảng trắng** ở hai đầu.
* **Ví dụ chuẩn:**
  - Tiếng Việt phiên âm: *vắc-xin, ra-đa, pít-tông, ca-nô*.
  - Tiếng Anh kỹ thuật: *scale-space, end-to-end, weight-decay, warm-started*.
  - *Sai:* `scale - space`, `vắc - xin`.

#### B. Dấu gạch ngang trung bình (En Dash: `–`) — **Dùng nhiều nhất trong báo cáo kỹ thuật**
* **Chức năng 1 (Chỉ khoảng / Range):** Biểu thị khoảng giá trị số học, thời gian, trang tài liệu, dải kích thước.
  - **Quy tắc:** **Không có khoảng trắng** ở hai đầu khi kẹp giữa hai con số.
  - **Ví dụ chuẩn:** *2–8 pixel, 15–20 ms, năm 2026–2027, trang 45–60, tỉ lệ 0,4–0,7*.
  - *Sai:* `2 - 8 pixel`, `2 – 8 pixel`, `2 -8 pixel`.
* **Chức năng 2 (Nối hai thực thể liên kết ngang hàng):**
  - **Ví dụ chuẩn:** *Quan hệ huấn luyện–suy luận, thuật toán Dijkstra–Moore, trục Bắc–Nam*.

#### C. Dấu gạch ngang dài (Em Dash: `—`) — **Kiêng dùng trong văn phong khoa học**
* **Bản chất:** Em dash mang sắc thái ngắt quãng kịch tính, phong cách báo chí hoặc tiểu thuyết.
* **Quy tắc:** Trong các bài báo khoa học nghiêm ngặt (IEEE/Nature), **hạn chế tối đa dùng Em dash**. Hãy thay thế mệnh đề xen ngang bằng **dấu phẩy (,)**, **dấu ngoặc đơn ()** hoặc viết thành hai câu độc lập.
* **Trường hợp bắt buộc dùng:** Viết liền không khoảng trắng ở hai đầu theo chuẩn quốc tế (`từ trước—mệnh đề xen ngang—từ sau`).

---

## PHẦN 5: VĂN PHONG KHOA HỌC THỰC SỰ TRÔNG NHƯ THẾ NÀO? (ĐỐI CHIẾU BEFORE / AFTER)

| Tình huống viết | ❌ Văn mẫu AI / Tiếp thị (AI-Slop) |  Văn phong khoa học thực sự |
|---|---|---|
| **Công bố kết quả** | *"Là minh chứng hùng hồn cho thấy module đã tạo bước đột phá ngoạn mục, nâng hiệu năng lên một tầm cao mới..."* | *"Trên tập test TinyPerson ở chế độ cắt mảnh 640 × 512, cấu hình tích hợp module đạt AP50 là 41,08%, tăng 0,45 điểm phần trăm so với baseline (40,63%). Mức tăng tập trung ở thang 8–12 px (+0,75 điểm)..."* |
| **Thừa nhận thất bại / hạn chế** | *"Dù mô hình hoạt động vô cùng xuất sắc, vẫn còn vài thách thức nhỏ ở thang 12–20 px nhưng không ảnh hưởng bức tranh toàn cảnh..."* | *"Ở thang 12–20 pixel, cấu hình có module giảm 1,09 điểm AP50 và tỷ lệ bỏ sót tăng 0,58 điểm do dải tần số cao bị chặn cứng bởi hệ số $\gamma_{out}$..."* |
| **Phân tích thành phần (Ablation)** | *"Để giúp người đọc hiểu sâu hơn, phần này sẽ đi sâu vào ablation. Chúng tôi nhận thấy nhánh nào cũng then chốt không thể thiếu..."* | *"Kết quả kiểm định bootstrap 2.000 lần cho thấy việc loại bỏ dải tần giữa và router lai làm AP50 giảm lần lượt 4,66 điểm (KTC 95%: [−6,40; −2,75]) và 3,20 điểm..."* |
| **Tổng quan tài liệu** | *"Có rất nhiều tác giả đã làm. Đầu tiên, tác giả A làm X. Tiếp theo, tác giả B làm Y. Cuối cùng, tác giả C làm Z..."* | *"Các tiếp cận hiện tại chủ yếu mở rộng độ phân giải qua stride 4 hoặc cắt mảnh trước suy luận. Tuy nhiên, các giải pháp này làm tăng 3–5 lần chi phí tính toán hoặc làm phân mảnh ngữ cảnh..."* |
| **Mô tả kiến trúc hệ thống** | *"Sau khi cài đặt PyTorch, chúng ta import module vào mạng. Người dùng chỉ cần truyền tensor vào và module chạy rất tiện lợi..."* | *"Module nhận ba tensor đặc trưng $\{C_2, P_3, P_4\}$. Đầu vào $C_2$ (stride 4, 64 kênh) được chiếu tuyến tính về $d = 32$ kênh qua tích chập $1 \times 1$, rồi phân rã bằng hai bộ lọc Gauss..."* |

---

## PHẦN 6: DANH MỤC 30 LỖI KINH ĐIỂN CẦN TRÁNH KHI VIẾT BÁO CÁO (ZERO-OVERLAP)

### I. Cấu trúc và phương pháp luận (Structural & Methodological)
1. **Biến báo cáo thành sách hướng dẫn cài đặt (User Manual Syndrome):** Dành hàng chục trang chụp màn hình bấm "Next", tạo môi trường ảo, menu IDE thay vì phân tích kiến trúc hệ thống, nguyên lý thuật toán và căn cứ thiết kế kỹ thuật.
2. **Nhầm lẫn giữa Tóm tắt (Abstract) và Mở đầu (Introduction):** Abstract là bản thu nhỏ độc lập của toàn bài; Introduction chỉ dùng để dẫn dắt bối cảnh rộng, động lực và phát biểu bài toán.
3. **Mục tiêu mơ hồ, không đo lường được (Unmeasurable Objectives):** Dùng cụm từ cảm tính như *"Xây dựng hệ thống thông minh/tối ưu"*, thiếu chỉ số định lượng cụ thể (nguyên tắc SMART) để nghiệm thu ở chương kết luận.
4. **Tổng quan tài liệu chỉ liệt kê thụ động (Descriptive vs Critical Review):** Liệt kê tác giả A làm X, tác giả B làm Y như điểm danh; thiếu so sánh đa chiều và không chỉ ra được khoảng trống nghiên cứu (gap).
5. **Thiếu lập luận đánh đổi khi chọn công nghệ (Lack of Trade-off Analysis):** Chọn framework, thuật toán mà không giải thích nguyên nhân kỹ thuật, chi phí đánh đổi và lý do không chọn giải pháp tương đương.
6. **Lỗi rò rỉ dữ liệu thực nghiệm (Data Leakage):** Trộn lẫn dữ liệu giữa tập train/val/test (khung hình cùng một video nằm ở cả hai tập) khiến kết quả đo đạc bị thổi phồng giả tạo.
7. **Đánh giá thiếu đường cơ sở đối chứng (Missing Baseline Benchmarks):** Chỉ đưa ra điểm số của mô hình tự làm mà không chạy thực nghiệm đối chứng trên cùng tập dữ liệu, cùng phần cứng với các giải pháp chuẩn.
8. **Bỏ qua các ca thất bại và điều kiện biên (Ignoring Failure Modes):** Báo cáo chỉ toàn màu hồng, che giấu các trường hợp mô hình dự đoán sai hoặc tắc nghẽn tài nguyên.
9. **Kết luận vượt quá bằng chứng thực nghiệm (Overclaiming):** Khẳng định giải pháp "hoạt động tốt trong mọi điều kiện" trong khi thực nghiệm chỉ thực hiện trên một tập dữ liệu nhỏ hoặc một kịch bản đơn lẻ.
10. **Mục tiêu đầu vào bị bỏ rơi ở đầu ra:** Đưa ra 4 mục tiêu ở Chương 1 nhưng Chương Đánh giá và Kết luận chỉ trả lời 2 mục tiêu, 2 mục tiêu còn lại bị lãng quên.

### II. Xử lý số liệu và kết quả đánh giá (Data, Evaluation & Metrics)
11. **Trưng bày bảng số liệu "chết" thiếu phân tích (Results without Discussion):** Đưa bảng/biểu đồ vào nhưng lời văn chỉ đọc lại số liệu cơ học mà không phân tích nguyên nhân kỹ thuật đằng sau mức tăng/giảm đó.
12. **Mâu thuẫn số liệu nội bộ (Internal Data Inconsistency):** Cùng một chỉ số nhưng con số ở phần Tóm tắt, Bảng kết quả, Biểu đồ và Kết luận lại lệch nhau (do cập nhật dở dang giữa các lần chạy lại).
13. **Đánh đồng sai lệch giữa các đơn vị tính toán:** Nhầm lẫn bản chất giữa các đại lượng tính toán như FLOPs và MACs ($1\text{ MAC} = 2\text{ FLOPs}$), hoặc nhầm lẫn giữa độ trễ của riêng mạng nơ-ron với độ trễ xử lý toàn pipeline.
14. **Báo cáo giá trị trung bình che giấu phương sai:** Chỉ công bố một con số trung bình duy nhất mà không kèm độ lệch chuẩn, khoảng tin cậy (Confidence Interval) hoặc kết quả trên nhiều seed ngẫu nhiên độc lập.
15. **Nhầm lẫn bản chất chỉ số chuyên ngành:** Đánh đồng Miss Rate (MR) là $1 - \text{Recall}$ mà bỏ qua định nghĩa chuẩn (log-average miss rate tích phân trên thang FPPI).

### III. Hình vẽ, bảng biểu và toán học (Visuals & Mathematics)
16. **Hình ảnh/Bảng biểu mồ côi (Floating Figures / Tables):** Đưa hình/bảng vào trang báo cáo nhưng nội dung văn bản không có câu nào dẫn nhập ("như thể hiện ở Hình X.X") và không có dòng bình luận nào về hình đó.
17. **Dùng ảnh chụp màn hình mờ nhòe, thừa thông tin:** Chụp màn hình terminal/IDE mờ nhòe thay vì vẽ sơ đồ vector (SVG/PDF), biểu đồ xuất từ code (Matplotlib vector) hoặc bảng biểu văn bản chuẩn.
18. **Trục đồ thị cắt cụt đánh lừa thị giác (Misleading Truncated Axes):** Cắt xén trục tung không bắt đầu từ 0 nhằm phóng đại mức chênh lệch nhỏ, hoặc đồ thị thiếu tên trục, thiếu đơn vị đo lường và thiếu chú giải (Legend).
19. **Đặt tiêu đề (Caption) sai quy chuẩn học thuật:** Đặt tiêu đề bảng ở dưới đáy bảng, hoặc đặt tiêu đề hình ở trên đỉnh hình (Quy ước quốc tế: Tên Bảng ở trên, Tên Hình ở dưới).
20. **Công thức toán học thiếu định nghĩa biến số:** Đưa phương trình toán học vào bài nhưng không giải thích tường minh từng ký hiệu bên dưới (biến nào là vô hướng, ma trận, không gian chiều nào, đơn vị gì).
21. **Rò rỉ mã nguồn toán thô (Raw LaTeX Code Leaks):** Để lộ các ký tự mã nguồn chưa biên dịch (ví dụ: `\alpha`, `\sigma`, `\times`, `$x$`) trực tiếp trên văn bản in ấn Word/PDF.

### IV. Trích dẫn và liêm chính học thuật (Citations & References)
22. **Trích dẫn kiểu "ném URL trần":** Danh mục tài liệu tham khảo chỉ dán mỗi đường link web trần trụi (dễ chết link), thiếu tác giả, tiêu đề bài viết, tổ chức phát hành và mốc thời gian truy cập (Accessed date).
23. **Không nhất quán định dạng trích dẫn:** Trộn lẫn lộn xộn giữa chuẩn IEEE dạng số `[1]` và chuẩn APA dạng tác giả - năm `(Smith, 2024)` trong cùng một tài liệu.
24. **Trích dẫn dây chuyền không kiểm chứng gốc (Secondary Citation Hazard):** Trích dẫn lại nhận định từ bài báo B nói về công trình A mà không tự mình đọc bài báo gốc A, dẫn đến truyền bá sai lệch số liệu.
25. **Tài liệu tham khảo lệch pha thời gian hoặc thiếu cân bằng:** Đồ án về chủ đề mũi nhọn (AI, Computer Vision) nhưng tài liệu tham khảo toàn giáo trình từ hàng chục năm trước, hoặc ngược lại chỉ toàn link blog cá nhân, thiếu vắng các bài báo khoa học bình duyệt (peer-reviewed).

### V. Văn phong và kỹ thuật trình bày (Tone, Layout & Typesetting)
26. **Lạm dụng văn phong cảm tính và từ ngữ tiếp thị:** Dùng các tính từ hoa mỹ như *"kết quả vô cùng hoàn hảo"*, *"bước đột phá ngoạn mục"*, *"thuật toán siêu việt"* thay vì trình bày trung thực, khách quan bằng dữ liệu định lượng.
27. **Ngôi xưng không nhất quán:** Đoạn thì xưng "tôi", đoạn xưng "chúng tôi", đoạn lại xưng "nhóm em" hoặc lạm dụng thể bị động một cách gượng gạo.
28. **Từ viết tắt không giải nghĩa lần đầu (Undocumented Acronyms):** Sử dụng các thuật ngữ viết tắt chuyên ngành (như IoU, NMS, FPN) ngay từ đầu mà không định nghĩa đầy đủ ở lần xuất hiện đầu tiên và thiếu Danh mục chữ viết tắt.
29. **Dòng mồ côi và ngắt trang lỗi (Widows & Orphans):** Tiêu đề mục (Heading) nằm trơ trọi ở đáy trang cuối trong khi nội dung nhảy sang trang sau; bảng biểu dài bị ngắt đôi mà không có thuộc tính lặp lại hàng tiêu đề (Repeat Header Row).
30. **Mục lục sai lệch số trang thực tế (Desynchronized TOC):** Đánh số trang mục lục thủ công hoặc quên cập nhật lại trường số trang (Update Field) trước khi xuất PDF, khiến số trang trong Mục lục không khớp với nội dung thực tế.

---

## PHẦN 7: BỘ QUY TẮC PROMPTING ÉP AI VIẾT ĐÚNG CHUẨN HỌC THUẬT (ANTI-AI SLOP)

### 1. Khung Negative Constraints (Bắt buộc chèn vào System Prompt)

```markdown
Bạn là một Nhà nghiên cứu khoa học và Biên tập viên học thuật chuyên sâu.
Khi soạn thảo hoặc hiệu chỉnh nội dung báo cáo kỹ thuật, bạn PHẢI tuân thủ các quy tắc sau:

1. NGỮ KHÍ VÀ VĂN PHONG HỌC THUẬT:
   - Dùng văn phong trung tính, thông tin dày đặc (information-dense), cô đọng, khách quan.
   - Tuyệt đối KHÔNG dùng các tính từ hoa mỹ, phóng đại hoặc tiếp thị (ví dụ: "vượt trội", "đột phá", "hoàn hảo", "siêu việt", "mạnh mẽ"). Mọi nhận định chất lượng phải gắn liền với số liệu đo đạc cụ thể.
   - Giả định độc giả là chuyên gia trong ngành; KHÔNG giải thích các khái niệm nền tảng phổ thông (không đóng vai thầy giáo giảng bài, không meta-narrative).

2. CẤU TRÚC VÀ NHỊP ĐIỆU CÂU (BURSTINESS):
   - Đa dạng hóa độ dài câu: xen kẽ linh hoạt giữa câu ngắn (5-12 từ), câu trung bình (15-25 từ) và câu dài phức hợp (30+ từ).
   - Tuyệt đối tránh việc sinh ra hàng loạt câu có cùng một khuôn mẫu ngữ pháp hoặc độ dài xấp xỉ nhau.

3. CÁC TỪ NGỮ VÀ MẪU CÂU BỊ CẤM (BANNED PHRASES):
   - Cấm các từ nối sáo rỗng: "Đầu tiên", "Thứ hai", "Thứ ba", "Hơn nữa", "Tóm lại", "Nhìn chung", "Có thể thấy", "Đáng chú ý", "không chỉ... mà còn".
   - Cấm các từ chuyển ý khoa trương: "delve", "testament", "beacon", "pivotal", "landscape", "tapestry", "bức tranh toàn cảnh", "minh chứng hùng hồn".
   - Cấm các câu khẩu hiệu đối lập nhân tạo: "X, không phải Y, là...", "Tham số rẻ, bộ nhớ thì không", "Recall lên, precision xuống".
   - Bắt đầu trực tiếp vào bản chất kỹ thuật hoặc kết quả; không mở đầu đoạn văn bằng câu dẫn nhập rỗng nghĩa.

4. QUY TẮC SUY NGHĨ PHẢN BIỆN KHI CÓ MÂU THUẪN (CRITICAL THINKING OVER BLIND OBEDIENCE):
   - Mục tiêu tối thượng là tính chân thực, khách quan và văn phong tự nhiên của một nhà nghiên cứu con người.
   - Khi có sự xung đột giữa các quy tắc phụ (ví dụ: quy tắc câu ngắn/nhịp điệu xung đột với tính logic, tự nhiên của câu văn), PHẢI dừng lại suy nghĩ, ưu tiên ngữ cảnh và tính mạch lạc, tuyệt đối KHÔNG được mù quáng làm theo để rồi tạo ra câu văn mang tính "khẩu hiệu" hoặc "AI slop".

4. QUY TẮC DẤU CÂU (PUNCTUATION RULES):
   - Hạn chế tối đa dấu gạch ngang dài (Em dash: —). Thay thế bằng dấu phẩy (,) hoặc cấu trúc câu độc lập.
   - Dùng đúng dấu gạch ngang trung bình (En dash: –) cho khoảng số (ví dụ: 2–8 pixel, 2026–2027) không có khoảng trắng ở hai đầu.
   - Dùng đúng dấu gạch nối (Hyphen: -) cho từ ghép kỹ thuật (scale-space, end-to-end).

5. TRUNG THỰC VỚI SỐ LIỆU VÀ ĐIỀU KIỆN BIÊN:
   - Chỉ được viết dựa trên bằng chứng và dữ liệu được cung cấp. Tuyệt đối không bịa đặt số liệu hoặc suy diễn ngoài phạm vi thực nghiệm.
   - Luôn kèm theo điều kiện biên, độ lệch chuẩn hoặc khoảng tin cậy khi nêu kết quả.
```

---

## PHẦN 8: CÔNG CỤ & SCRIPT PYTHON TẠO FILE DOCX/PDF CHUẨN ĐẸP 100%

### 1. Bộ 4 cờ OpenXML cốt lõi chống vỡ định dạng
- `paragraph_format.keep_with_next = True`: Bắt buộc cho toàn bộ Heading và Caption, ngăn ngừa tuyệt đối hiện tượng tiêu đề nằm trơ trọi cuối trang (Widows/Orphans).
- `w:cantSplit`: Bắt buộc gắn cho từng hàng trong bảng (`w:trPr`), ngăn việc một hàng bị xé đôi giữa 2 trang.
- `w:tblHeader`: Bắt buộc gắn cho hàng tiêu đề của bảng, tự động lặp lại hàng tiêu đề ở trang sau khi bảng dài tràn trang.
- `w:rFonts`: Ép chặt font `Times New Roman` trên cả 3 thuộc tính `ascii`, `hAnsi`, `cs` để văn bản không bị nhảy font mặc định của hệ thống.

---

### 2. Script Python Hoàn chỉnh Tạo file DOCX Chuẩn mực (`build_academic_docx.py`)

```python
"""
Script tự động hóa tạo tài liệu DOCX học thuật chuẩn mực cao:
- Chống ngắt trang lỗi (Widows/Orphans)
- Chống xé hàng trong bảng (cantSplit) và lặp header khi tràn trang (tblHeader)
- Ép font chữ chuẩn toàn diện (w:rFonts)
"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def apply_academic_font(run, font_name="Times New Roman", size_pt=12, bold=False, italic=False, color_rgb=(0, 0, 0)):
    """Ép font chuẩn OpenXML cho tất cả các hệ chữ (Latin, Asian, Complex Scripts)."""
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor(*color_rgb)

    rPr = run._r.get_or_add_rPr()
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), font_name)
    rFonts.set(qn('w:hAnsi'), font_name)
    rFonts.set(qn('w:cs'), font_name)
    rPr.append(rFonts)


def add_academic_heading(doc, text, level=1):
    """Thêm Heading chống mồ côi (keep_with_next = True) và chuẩn giãn dòng."""
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True  # Luôn dính liền với đoạn văn tiếp theo
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)

    run = p.add_run(text)
    size = 14 if level == 1 else (13 if level == 2 else 12)
    apply_academic_font(run, "Times New Roman", size_pt=size, bold=True)
    return p


def add_academic_paragraph(doc, text):
    """Thêm đoạn văn bản thân bài chuẩn học thuật."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.3  # Giãn dòng tiêu chuẩn
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    run = p.add_run(text)
    apply_academic_font(run, "Times New Roman", size_pt=12)
    return p


def format_academic_table(table, col_widths=None):
    """Chuẩn hóa bảng biểu: Lặp header khi qua trang (tblHeader) và cấm xé dòng (cantSplit)."""
    table.autofit = False

    for row_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()

        # 1. Cấm xé ngang một hàng qua 2 trang (cantSplit)
        cantSplit = OxmlElement('w:cantSplit')
        cantSplit.set(qn('w:val'), 'true')
        trPr.append(cantSplit)

        # 2. Hàng tiêu đề đầu tiên: tự động lặp lại khi bảng tràn trang (tblHeader)
        if row_idx == 0:
            tblHeader = OxmlElement('w:tblHeader')
            tblHeader.set(qn('w:val'), 'true')
            trPr.append(tblHeader)

        # 3. Định dạng ô
        for col_idx, cell in enumerate(row.cells):
            if col_widths and col_idx < len(col_widths):
                cell.width = col_widths[col_idx]
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)
                for r in p.runs:
                    apply_academic_font(r, "Times New Roman", size_pt=10.5, bold=(row_idx == 0))


def create_sample_report(output_path="report_sample.docx"):
    doc = Document()

    # Thiết lập căn lề tiêu chuẩn đồ án (Trái 3cm, Phải/Trên/Dưới 2cm)
    for section in doc.sections:
        section.top_margin = Inches(0.79)     # 2.0 cm
        section.bottom_margin = Inches(0.79)  # 2.0 cm
        section.left_margin = Inches(1.18)    # 3.0 cm
        section.right_margin = Inches(0.79)   # 2.0 cm

    add_academic_heading(doc, "Chương 1: TỔNG QUAN HỆ THỐNG", level=1)
    add_academic_paragraph(
        doc,
        "Đoạn văn này được định dạng chuẩn học thuật với font Times New Roman 12pt, căn đều hai bên "
        "và giãn dòng 1.3. Tiêu đề mục phía trên được áp dụng thuộc tính keep_with_next để đảm bảo "
        "không bao giờ bị trơ trọi ở đáy trang (loại bỏ hoàn toàn lỗi orphan heading)."
    )

    # Thêm bảng mẫu
    table = doc.add_table(rows=3, cols=3)
    table.style = 'Table Grid'
    headers = ["Thông số", "Mô hình cơ sở", "Giải pháp đề xuất"]
    data = [
        ["Độ chính xác (AP50, %)", "40,63", "41,08 (+0,45)"],
        ["Số lượng tham số (M)", "2,52", "2,55 (+1,12%)"]
    ]

    for col_idx, h in enumerate(headers):
        table.rows[0].cells[col_idx].text = h
    for row_idx, row_data in enumerate(data, start=1):
        for col_idx, val in enumerate(row_data):
            table.rows[row_idx].cells[col_idx].text = val

    format_academic_table(table, col_widths=[Inches(2.5), Inches(1.8), Inches(1.8)])
    doc.save(output_path)
    print(f"Đã tạo file mẫu thành công: {output_path}")


if __name__ == "__main__":
    create_sample_report()
```

---

### 3. Quy trình 2-Pass Biên dịch PDF & Khớp Mục Lục 100%

```bash
#!/usr/bin/env bash
set -e

# Bước 1: Sinh file DOCX lượt 1 (tạm để trang mục lục là placeholder)
python3 scratch/build_report_docx.py --pass1

# Bước 2: Dùng LibreOffice headless xuất file PDF sơ bộ để xác định vị trí trang
soffice --headless --convert-to pdf --outdir temp/ docs/report/course_report_vi.docx

# Bước 3: Dùng pdftotext bóc tách vị trí trang thực tế của từng Heading
python3 scratch/extract_heading_pages.py temp/course_report_vi.pdf > temp/toc_mapping.json

# Bước 4: Chạy lại lượt 2 để bơm số trang thực tế vào bảng Mục lục
python3 scratch/build_report_docx.py --pass2 --toc-map temp/toc_mapping.json

# Bước 5: Xuất file PDF chính thức cuối cùng
soffice --headless --convert-to pdf --outdir dist/ docs/report/course_report_vi.docx
```

---

## PHẦN 9: BẢNG KIỂM TRA CHẤT LƯỢNG HỌC THUẬT (PRE-SUBMISSION CHECKLIST)

| Hạng mục | Tiêu chí nghiệm thu | Đạt |
|---|---|:---:|
| **Cấu trúc IMRaD** | Chương 2 (Phương pháp) và Chương 3 (Kết quả) chiếm 50–60% dung lượng; tách bạch rõ Method vs Results | [ ] |
| **Nhịp điệu câu** | Độ dài câu đa dạng (xen kẽ câu ngắn 5–12 từ, câu trung bình 15–25 từ, câu dài 30+ từ) | [ ] |
| **Dấu câu** | Phân biệt chuẩn xác Hyphen (`-`), En Dash (`–` cho khoảng số) và kiêng tối đa Em Dash (`—`) | [ ] |
| **Văn phong** | Tuyệt đối không chứa các từ sáo rỗng ("Hơn nữa", "Tóm lại", "đột phá", "siêu việt") | [ ] |
| **Dữ liệu** | 100% số liệu ở Tóm tắt, Bảng kết quả, Biểu đồ và Kết luận khớp nhau từng chữ số | [ ] |
| **Hình vẽ** | Tất cả các hình đều được nhắc đến trong bài ("như thể hiện ở Hình X.X...") | [ ] |
| **Bảng biểu** | Tiêu đề bảng nằm ở trên; các hàng có gắn `cantSplit`, hàng tiêu đề gắn `tblHeader` | [ ] |
| **Toán học** | Mọi phương trình đều giải thích rõ ràng ý nghĩa của từng biến số bên dưới | [ ] |
| **Trình bày** | Không có tiêu đề nằm mồ côi cuối trang (`keep_with_next = True`) | [ ] |
| **Mục lục** | Số trang hiển thị trong Mục lục trùng khớp 100% với trang thực tế trong file PDF | [ ] |
