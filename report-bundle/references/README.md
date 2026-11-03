# Danh mục tài liệu tham khảo

File `danh-muc.md` được agent bổ sung nguồn sau khi đã tìm, đọc và xác minh metadata. Không bịa hoặc suy đoán thông tin thư mục.

Quy tắc:
- Agent có thể bổ sung mục tham khảo đã tìm, đọc và xác minh metadata từ nguồn gốc đáng tin cậy; người dùng có thể rà soát danh mục.
- Dòng đầu là `# Tài liệu tham khảo`, sau đó mỗi dòng một mục, đánh số `[1]`, `[2]`, ... theo thứ tự xuất hiện trong báo cáo.
- Mỗi mục nên có DOI. Chạy `python tools/verify_refs.py references/danh-muc.md` để đối chiếu.
- `build.py` tự nối file này vào cuối báo cáo.

Ví dụ một dòng (thay bằng dữ liệu thật của bạn):
```
[1] <Tác giả>, "<Tiêu đề>," <Tên tạp chí/hội nghị>, <năm>, doi: <DOI>.
```
