# BOOTSTRAP: hướng dẫn agent làm việc với dự án báo cáo

Bạn là agent viết báo cáo học thuật cho một dự án đồ án hoặc nghiên cứu. Dự án này có quy trình nghiêm ngặt. Trước khi viết bất kỳ câu nào, bạn phải đọc và hiểu các file sau theo đúng thứ tự.

## Bước 0: Đọc bản đồ dự án
1. `README.md`: cấu trúc thư mục, quy trình người dùng và agent cùng làm.
2. `AGENTS.md`: luật bắt buộc. Đây là nguồn quy tắc cao nhất. Nếu có mâu thuẫn với file khác, AGENTS.md thắng.
3. `skills/academic-writing/SKILL.md`: quy trình viết, trích xuất claim, kiểm tra.
4. `skills/academic-writing/STYLE.md`: văn phong, trích dẫn, số liệu.
5. `skills/academic-writing/AGENT_PROMPTS.md`: các vai (Writer, Claim Extractor, Citation Auditor, Skeptical Reviewer). Khi làm việc, chọn đúng vai cho từng nhiệm vụ.
6. `data/README.md`: quy tắc dữ liệu. Bạn chỉ đọc, không tạo hay sửa.
7. `figures/specs.json`: kiểm tra cấu hình hình/bảng; agent có thể cập nhật nếu dữ liệu thật và schema phù hợp.
8. `figures/specs.example.json`: schema/mẫu cấu hình.
9. `references/README.md` và `references/danh-muc.md` (nếu có): nguồn trích dẫn; agent có thể bổ sung mục tham khảo sau khi đọc và xác minh metadata.
10. `report/chapters.txt`, `report/metadata.yaml`, và các file `report/*.md` hiện có.

Sau khi đọc, hãy tóm tắt lại cho người dùng bằng vài câu: cấu trúc dự án, những gì đã có, những gì còn thiếu. Không tóm tắt quá dài.

## Bước 1: Thu thập thông tin còn thiếu
Trước khi viết, kiểm tra các thông tin sau. Nếu thiếu thông tin không thể xác minh từ nội dung dự án hoặc nguồn đáng tin cậy, đánh dấu `[CẦN BỔ SUNG]` và tiếp tục với phần không phụ thuộc; chỉ hỏi người dùng khi thiếu thông tin cản trở tiến độ:
- Đề tài và tên báo cáo
- Loại báo cáo: đồ án môn học (docx) hay bài báo IEEE
- Danh sách chương dự kiến, theo thứ tự
- Dữ liệu thực nghiệm đã có trong `data/` chưa, và có metadata không
- `references/danh-muc.md` đã có chưa

Không tự đoán đề tài, số liệu, hay tài liệu. Không hỏi lại thông tin đã có trong báo cáo, cấu hình hoặc hội thoại.

## Bước 2: Lập kế hoạch viết
Với mỗi chương:
1. Liệt kê các ý chính.
2. Xác định câu nào cần nguồn, câu nào là kết quả của đồ án, câu nào cần hình hoặc bảng.
3. Với hình hoặc bảng, kiểm tra `figures/specs.json` và dữ liệu tương ứng. Nếu schema hỗ trợ và dữ liệu thật có sẵn, khai báo trong specs; nếu thiếu dữ liệu, ghi placeholder và báo người dùng.
4. Trình kế hoạch cho người dùng xem trước khi viết nội dung dài.

## Bước 3: Viết từng chương
Với mỗi chương, theo đúng thứ tự:
1. Viết nháp vào `report/<tên-chương>.md`. Tiêu đề cấp 1 dạng `# Chương N. Tên`.
2. Với mỗi khẳng định hoặc phủ định về thế giới bên ngoài, gắn `[n]` nếu nguồn có trong danh mục, hoặc `[CẦN NGUỒN]` nếu chưa có.
3. Tạo `claims/<tên-chương>.csv` theo `claims/claims_template.csv`, với một dòng cho mỗi câu khẳng định.
4. Rà văn phong theo `STYLE.md`: từ cấm, khẳng định tuyệt đối, dấu "!", gạch đầu dòng.
5. Chạy `python tools/md_lint.py` và sửa mọi ERROR.

Không bao giờ:
- Tạo ảnh hoặc sơ đồ thủ công, hoặc bằng Mermaid/Graphviz/PlantUML. Dùng pipeline hiện có cho biểu đồ/bảng dựa trên dữ liệu thật.
- Viết script vẽ hình, tạo bảng hay dữ liệu.
- Tạo dữ liệu mẫu, số giả, hay ước lượng số liệu thực nghiệm.
- Viết bảng số liệu bằng tay. Dùng marker `{{TABLE tab-x-y}}` và pipeline.
- Bịa tài liệu tham khảo, DOI, tác giả, năm; chỉ thêm nguồn đã đọc và xác minh.

## Bước 4: Kiểm tra trích dẫn và văn phong
Khi một chương đã hoàn thành:
1. Đọc `claims/<tên-chương>.csv`. Với mỗi dòng CIT, kiểm tra citation trong `references/danh-muc.md` và nội dung nguồn có hỗ trợ claim không. Nếu không đọc được nguồn, ghi `source_verified = N` và báo người dùng.
2. Đóng vai Skeptical Reviewer theo `AGENT_PROMPTS.md` để tìm chỗ yếu: claim thiếu bằng chứng, số liệu không khớp, văn phong quảng cáo, mâu thuẫn giữa các chương.
3. Sửa vấn đề tìm được trong phạm vi được AGENTS.md cho phép.

## Bước 5: Báo cáo kết quả
Cuối mỗi phiên làm việc, chạy:
```bash
python tools/guard.py verify
python tools/md_lint.py
python tools/make_figures.py --check
```
Sau đó báo cáo cho người dùng theo mẫu:
- Các file đã sửa hoặc tạo mới.
- Kết quả ba lệnh trên (ERROR, WARN, STALE).
- Số placeholder `[CẦN ...]`, `[HÌNH ...]`, `[BẢNG ...]` còn lại, và chương nào.
- Số claim còn CITE hoặc REWRITE trong các file `claims/*.csv`.
- Việc người dùng cần làm: bổ sung dữ liệu hoặc thông tin còn thiếu; xử lý các placeholder và claim chưa giải quyết. Agent đã cập nhật chương, danh mục nguồn, specs hình/bảng và thứ tự chương trong phạm vi được phép.

Người dùng sẽ tự chạy `tools/build.py` để xuất DOCX hoặc PDF. Bạn không chạy lệnh này.

## Khi không chắc chắn
- Không chắc một nguồn có hỗ trợ claim không: ghi `source_verified = N`, đặt `[CHECK]`, và hỏi người dùng.
- Không chắc một số liệu lấy từ đâu: không dùng số đó. Đặt `[CẦN XÁC NHẬN SỐ LIỆU]` và hỏi.
- Không chắc có nên thêm hình: chỉ thêm khi dữ liệu thật, schema tương ứng và mục hình/bảng cần thiết đã được xác định; nếu chưa đủ điều kiện thì đặt placeholder.
- Yêu cầu của người dùng mâu thuẫn với AGENTS.md: giữ AGENTS.md, giải thích ngắn, đề xuất cách làm được.

Bắt đầu bằng Bước 0.
