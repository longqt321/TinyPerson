---
name: academic-writing-discipline
description: Use when drafting, revising, or reviewing a thesis chapter, course project report, IEEE paper, or research report. Enforces scientific tone, mandatory citation of every outside claim, a claim-by-claim evidence map, and a pre-submission audit. Triggers on academic writing, thesis report, IEEE paper, citations, literature review, rebuttal of AI-generated prose, cite this claim, check my draft.
---

# Kỹ năng: Viết học thuật có kiểm soát

## Khi nào dùng
- Viết hoặc sửa báo cáo đồ án, bài báo IEEE, báo cáo nghiên cứu.
- Rà soát bản thảo do AI tạo ra để tìm claim thiếu nguồn và văn phong quảng cáo.
- Chuẩn bị bản nộp: kiểm tra trích dẫn, số liệu, định dạng.

## Nguyên tắc
1. Đọc `skills/academic-writing/STYLE.md` trước khi viết. Đó là nguồn quy tắc duy nhất.
2. Không bịa nguồn, số liệu, kết quả. Nếu thiếu thông tin, đánh dấu `[CẦN NGUỒN]` hoặc hỏi người dùng.
3. Số liệu thực nghiệm chỉ lấy từ file kết quả do người dùng cung cấp. Không tự suy ra số.
4. Không thay đổi nội dung kỹ thuật khi chỉ được yêu cầu sửa văn phong.

## Quy trình

### Bước 1: Viết bản nháp
Viết theo dàn ý và file kết quả. Trong bản nháp, mỗi claim về thế giới bên ngoài nên có [n] nếu đã biết nguồn. Nếu chưa biết, ghi `[CẦN NGUỒN]`.

### Bước 2: Trích xuất claim
Với mỗi câu khẳng định hoặc phủ định, thêm một dòng vào bảng theo `claims/claims_template.csv`; lưu thành `claims/<chương>.csv`:
- `id`, `section`, `sentence` (nguyên văn câu)
- `claim_type`: `SELF` (kết quả của mình), `CIT` (cần nguồn), `INTERNAL` (tham chiếu trong bài), `DEF` (định nghĩa), `OPINION` (nhận định của tác giả)
- `evidence_ref`: khóa trích dẫn hoặc tham chiếu nội bộ (ví dụ `Bảng 3`)
- `source_verified`: Y/N, chỉ Y khi đã đọc nguồn
- `action`: `KEEP` | `REWRITE` (sửa mức độ khẳng định) | `CITE` (thêm nguồn) | `REMOVE`

### Bước 3: Xử lý từng claim
- `SELF`: kiểm tra số trong bảng/hình có khớp với file kết quả không.
- `CIT`: tìm nguồn; đọc phần liên quan; nếu nguồn không chứng minh được claim, giảm mức khẳng định hoặc tìm nguồn khác. Không được bỏ qua.
- `INTERNAL`: kiểm tra mục/hình/bảng được tham chiếu có tồn tại.
- `OPINION`: chuyển thành mệnh đề có điều kiện ("theo quan sát của chúng tôi trên tập X...").

### Bước 4: Kiểm tra tự động
Agent KHÔNG tự chạy build. Người dùng chạy `python tools/build.py docx --skip-checks` để tạo `build/report.md` (bản ghép có bảng và danh mục). Khi file này có sẵn, agent có thể kiểm tra bằng:
```bash
python tools/check_manuscript.py build/report.md
```
Nếu build/report.md chưa có hoặc đã cũ, báo người dùng chạy lại. Sửa mọi ERROR trong chương. Xem từng WARN và ghi lý do nếu giữ nguyên. `verify_refs.py` do người dùng chạy (cần internet).

### Bước 5: Rà soát văn phong
Đối chiếu với mục 1 của `STYLE.md`: từ cấm, khẳng định tuyệt đối, sự nhất quán mức độ khẳng định giữa các chương, dấu "!".

### Bước 6: Báo cáo cho người dùng
Trả về:
- Số claim theo nhãn, số claim đã có nguồn, số claim còn `[CẦN NGUỒN]`.
- Danh sách ERROR đã sửa và còn lại.
- Danh sách nguồn cần người dùng tự kiểm tra (không có DOI hoặc không truy cập được).

## Những điều không được làm
- Không tạo tài liệu tham khảo mới chỉ để lấp chỗ trống trích dẫn.
- Không xác nhận một nguồn "đã đọc" nếu chưa có văn bản của nguồn đó.
- Không gộp nhiều claim khác nhau vào một trích dẫn mà không kiểm tra từng cái.
- Không xóa `[CẦN NGUỒN]` mà không có nguồn thật hoặc không có sửa đổi thật.

## Mẫu phản hồi khi thiếu nguồn
"Câu này cần nguồn để khẳng định mức độ đó. Tôi đã đánh dấu [CẦN NGUỒN]. Bạn có thể cung cấp tài liệu, hoặc tôi có thể hạ mức khẳng định thành: '...'."
