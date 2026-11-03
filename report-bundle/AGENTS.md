# Quy tắc cho agent

Đọc file này trước, rồi đọc `BOOTSTRAP.md` nếu đây là lần đầu làm việc với dự án.

## 1. Phạm vi được phép
Agent được đọc mọi thứ. Agent được tạo và sửa:
- `report/*.md`: các chương của báo cáo, mỗi chương một file.
- `claims/*.csv`: bảng bản đồ bằng chứng, mỗi chương một file.
- `figures/specs.json`: khai báo hình/bảng để pipeline sinh từ dữ liệu thật có sẵn.
- `references/danh-muc.md`: danh mục nguồn đã tìm, đọc và xác minh; không được bịa metadata.
- `report/chapters.txt`: cập nhật thứ tự chương khi được yêu cầu hoàn thiện báo cáo.

Agent được chạy `tools/make_figures.py` để sinh ảnh/bảng từ dữ liệu thật và cấu hình hợp lệ. File trong `data/`, `report/assets/` và `report/tables/` chỉ do pipeline quản lý; không sửa trực tiếp.
Mọi file khác chỉ đọc hoặc không được đụng tới, bao gồm:
- `data/`: dữ liệu thực nghiệm
- `figures/` ngoại trừ `figures/specs.json`
- `report/assets/`, `report/tables/` ngoại trừ sinh tự động bằng pipeline
- `skills/`, `tools/`, `tests/`, `build/`
- `AGENTS.md`, `BOOTSTRAP.md`, `README.md`, `report/metadata.yaml`, `reference.docx`

## 2. Cấm tuyệt đối
Các hành vi sau bị cấm trong mọi trường hợp, kể cả khi người dùng yêu cầu:

1. Không vẽ hoặc tạo ảnh thủ công, không tạo sơ đồ bằng Mermaid, Graphviz, PlantUML hay công cụ tương tự. Hình chỉ được sinh qua pipeline từ dữ liệu thật và specs đã khai báo; sơ đồ kiến trúc chỉ được dùng nếu có dữ liệu/định dạng nguồn được hỗ trợ, nếu không đặt ghi chú cần bổ sung và đề xuất cho người dùng.
2. Không viết script sinh ảnh, bảng hay dữ liệu. Không tạo dữ liệu giả dưới mọi hình thức; không sửa dữ liệu gốc.
3. Không tải ảnh từ internet hoặc từ nguồn ngoài `figures/manual/` do người dùng đặt sẵn.
4. Không sửa trực tiếp các ảnh/bảng sinh trong `report/assets/` và `report/tables/`; có thể chạy pipeline để tạo/cập nhật chúng từ dữ liệu và specs.
5. Không bịa tài liệu tham khảo, DOI, tác giả, năm hoặc tiêu đề. Chỉ ghi tài liệu đã tìm và đọc, thông tin thư mục phải xác minh từ nguồn gốc đáng tin cậy.

Nếu thiếu dữ liệu hoặc hình, không tìm cách thay thế bằng số ước lượng hay hình giả. Đặt ghi chú đúng vị trí và báo người dùng:

    > [HÌNH CẦN BỔ SUNG: fig-2-1, độ đo F1 theo mô hình. Cần dữ liệu trong data/ và mục tương ứng trong figures/specs.json]

## 3. Cách trình bày

### Hình
Hình chỉ được trình bày nếu mục tương ứng đã có trong `figures/specs.json`. Mã hình là `fig-<chương>-<số>`. Ảnh tham chiếu đúng tên file đã được pipeline sinh ra:

    ![Độ đo F1 của các mô hình trên tập kiểm tra](assets/fig-2-1.png)

    **Hình 2.1.** Độ đo F1 của các mô hình trên tập kiểm tra.

- Caption đặt NGAY SAU ảnh, dạng `**Hình x.y.** Mô tả`.
- Số hình đánh liên tục trong chương, khớp với mã `fig-x-y`.
- Alt text không được rỗng.

### Bảng
Không viết bảng số liệu bằng tay (dòng bắt đầu bằng `|`). Chỉ chèn bảng bằng marker, trỏ tới mục `tab-<chương>-<số>` trong `figures/specs.json`. Caption đặt NGAY TRƯỚC marker:

    **Bảng 2.1.** So sánh kết quả trên tập kiểm tra.

    {{TABLE tab-2-1}}

### Trích dẫn và số liệu
- Số liệu trong văn bản phải lấy đúng từ `data/`, không làm tròn khác đi, không suy diễn.
- Mọi khẳng định hoặc phủ định về thế giới bên ngoài phải có `[n]` trỏ tới `references/danh-muc.md`, hoặc tham chiếu nội bộ (Hình, Bảng, Mục), hoặc là kết quả của đồ án có mô tả cách đo.
- Trích dẫn đánh số `[1]`, `[2]`... theo thứ tự xuất hiện lần đầu. Không trích tài liệu chưa đọc.

### Văn phong
Tuân thủ `skills/academic-writing/STYLE.md`. Tóm tắt:
- Không dùng từ quảng cáo hay sáo rỗng (đột phá, vượt trội, cách mạng, đi sâu, then chốt...).
- Không dùng dấu "!" trong thân bài. Không dùng câu hỏi tu từ.
- Khẳng định tuyệt đối (luôn luôn, chắc chắn, chứng minh, tốt nhất...) chỉ khi có bằng chứng đúng phạm vi đó. Nếu không, hạ mức khẳng định.
- Ngôi viết nhất quán ("chúng tôi" hoặc câu bị động).
- Không dùng gạch đầu dòng trong thân bài trừ khi thật sự là danh sách.

## 4. Bảng bản đồ bằng chứng (claims)
Với mỗi chương đang viết, tạo `claims/<tên-chương>.csv` theo mẫu `claims/claims_template.csv`. Mỗi câu khẳng định hoặc phủ định là một dòng, với các cột:
- `claim_type`: SELF, CIT, INTERNAL, DEF, OPINION
- `evidence_ref`: khóa trích dẫn hoặc tham chiếu nội bộ
- `source_verified`: Y chỉ khi đã đọc nguồn
- `action`: KEEP, REWRITE, CITE, REMOVE

Trước khi báo hoàn thành, không để lại dòng nào có action CITE hoặc REWRITE chưa xử lý mà không báo người dùng.

## 5. Khai báo và sinh hình/bảng
Agent có thể thêm cấu hình vào `figures/specs.json` khi cột dữ liệu thật tồn tại trong `data/` và phù hợp với schema của `figures/specs.example.json`. Không sửa dữ liệu gốc hoặc tự tạo bảng số liệu bằng tay. Sau khi cập nhật specs, chạy `python tools/make_figures.py` để pipeline sinh đầu ra; không chỉnh tay đầu ra đó.

Sơ đồ kiến trúc không phải biểu đồ từ dữ liệu và không được vẽ thủ công theo quy định này. Nếu cần sơ đồ, ghi placeholder và nêu nội dung/định dạng cần người dùng cung cấp.

## 6. Thêm chương mới
Agent được tạo file `report/<tên>.md` mới và cập nhật `report/chapters.txt` khi hoàn thiện báo cáo; giữ đúng thứ tự chương. Ghi rõ thay đổi trong phản hồi.

## 7. Quy trình bắt buộc
Người dùng chạy `python tools/guard.py snapshot` trước khi giao việc. Sau khi agent xong, agent chạy và báo kết quả:
```bash
python tools/guard.py verify
python tools/md_lint.py
python tools/make_figures.py --check
```
Nếu `verify` báo thay đổi ngoài phạm vi, phải hoàn tác trước khi kết thúc. Nếu `md_lint` báo ERROR, phải sửa. Nếu `make_figures --check` báo STALE, chạy lại `python tools/make_figures.py` nếu nguyên nhân nằm trong dữ liệu/specs được phép; nếu không, báo người dùng. Không chạy `tools/build.py`, `pandoc`, hay `pip install`.

Phản hồi cuối cùng liệt kê: các file đã sửa, kết quả ba lệnh trên, số placeholder còn lại, số claim còn CITE hoặc REWRITE, và các việc người dùng cần làm.
