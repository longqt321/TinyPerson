# Bộ công cụ viết báo cáo học thuật (bản gộp)

Bản này gộp hai bộ công cụ trước đó:
- Bộ kiểm soát văn phong và trích dẫn: `skills/`, `claims/`, `tools/check_manuscript.py`, `tools/verify_refs.py`.
- Bộ viết báo cáo bằng Markdown, hình và bảng sinh từ dữ liệu: `report/`, `data/`, `figures/`, `tools/guard.py`, `tools/md_lint.py`, `tools/make_figures.py`, `tools/build.py`.

Nguyên tắc:
- Agent viết Markdown trong `report/`, bảng bằng chứng trong `claims/`, khai báo hình/bảng trong `figures/specs.json`, cập nhật `references/danh-muc.md` bằng nguồn đã xác minh, và cập nhật `report/chapters.txt` khi hoàn thiện thứ tự chương.
- Hình và bảng số liệu được sinh từ dữ liệu thật trong `data/` bởi `tools/make_figures.py`; agent có thể chạy pipeline nhưng không sửa dữ liệu hay đầu ra sinh trực tiếp.
- Trích dẫn chỉ dùng nguồn đã tìm, đọc và xác minh; metadata trong danh mục phải lấy từ nguồn gốc đáng tin cậy.
- DOCX và PDF do bạn xuất bằng `tools/build.py`, sau khi mọi kiểm tra đạt.
- Không tạo dữ liệu giả, vẽ ảnh thủ công hoặc viết script sinh hình/bảng. Chi tiết trong `AGENTS.md`.

## 1. Cấu trúc

```
report-bundle/
├── BOOTSTRAP.md              prompt khởi động cho agent (đưa cho agent đọc đầu tiên)
├── AGENTS.md                 luật bắt buộc cho agent
├── README.md                 tài liệu này (cho người dùng)
├── .gitignore
├── reference.docx            (tùy chọn) mẫu kiểu dáng DOCX
├── skills/academic-writing/
│   ├── SKILL.md              quy trình viết, claim, kiểm tra
│   ├── STYLE.md              văn phong, trích dẫn, số liệu
│   └── AGENT_PROMPTS.md      prompt cho các vai agent
├── claims/                   bảng bằng chứng, mỗi chương một CSV (agent ghi)
│   └── claims_template.csv
├── references/               DANH MỤC TÀI LIỆU (agent bổ sung nguồn đã xác minh)
│   ├── README.md
│   └── danh-muc.md
├── data/                     DỮ LIỆU THẬT (bạn quản lý, agent chỉ đọc)
│   └── README.md
├── figures/
│   ├── specs.json            khai báo hình và bảng (agent cập nhật)
│   ├── specs.example.json    mẫu đầy đủ
│   ├── manifest.json         dấu vân tay các file đã sinh (tự sinh)
│   └── manual/               ảnh nguồn do người dùng cung cấp sẵn (agent không tạo)
├── tools/
│   ├── make_figures.py       sinh hình, bảng từ data/ (agent chạy khi cần)
│   ├── md_lint.py            kiểm tra cấu trúc chương và khớp specs.json
│   ├── guard.py              chặn thay đổi ngoài phạm vi
│   ├── check_manuscript.py   kiểm tra văn phong và trích dẫn trên bản ghép
│   ├── verify_refs.py        đối chiếu DOI với Crossref
│   └── build.py              kiểm tra, ghép, xuất DOCX/PDF (bạn chạy)
├── report/
│   ├── metadata.yaml         tiêu đề, tác giả, ngày, ngôn ngữ
│   ├── chapters.txt          thứ tự chương (agent cập nhật khi hoàn thiện)
│   ├── 01-introduction.md, 02-theoretical-background.md   (chương mẫu)
│   ├── assets/               hình PNG do pipeline sinh
│   └── tables/               bảng Markdown do pipeline sinh
├── tests/sample_draft.md     bản thảo mẫu để thử check_manuscript
└── build/                    kết quả xuất (tự tạo)
```

## 2. Phân quyền

| Đối tượng | Được làm | Không được làm |
|---|---|---|
| Agent | Sửa chương trong `report/`, `claims/*.csv`, `figures/specs.json`, `references/danh-muc.md`, `report/chapters.txt`; chạy figure pipeline | Sửa `data/`, `tools/`, `report/assets/` hoặc `report/tables/` trực tiếp; tạo dữ liệu giả/ảnh thủ công; chạy build; pip install |
| Bạn | Mọi việc | — |

## 3. Cài đặt

### 3.1 Python 3.9 trở lên
```bash
python --version        # macOS/Linux có thể cần python3 --version
```
Windows: tick "Add Python to PATH" khi cài từ python.org.

### 3.2 Thư viện
```bash
python -m pip install matplotlib==3.10.8 python-docx
```
- `matplotlib`: bắt buộc, để sinh hình.
- `python-docx`: chỉ cần nếu muốn `check_manuscript.py` đọc file .docx.

`md_lint.py`, `guard.py`, `build.py`, `check_manuscript.py`, `verify_refs.py` không cần thư viện ngoài (trừ khi dùng .docx).

### 3.3 pandoc (để xuất DOCX/PDF)
- Windows: `winget install --id JohnMacFarlane.Pandoc -e`
- macOS: `brew install pandoc`
- Ubuntu/Debian: `sudo apt update && sudo apt install pandoc` (bản apt cũ thì tải từ https://github.com/jgm/pandoc/releases)

Kiểm tra: `pandoc --version`

### 3.4 LaTeX có xelatex (chỉ khi xuất PDF)
- Windows: MiKTeX (https://miktex.org/download)
- macOS: `brew install --cask basictex`
- Ubuntu/Debian: `sudo apt install texlive-xetex texlive-fonts-recommended texlive-lang-other`

Khi xuất PDF, bỏ dấu `#` ở dòng `mainfont` trong `report/metadata.yaml` và điền font có trên máy.

### 3.5 Git và hook (khuyến nghị)
```bash
git init && git add . && git commit -m "Khởi tạo dự án báo cáo"
```
Hook `.git/hooks/pre-commit` (chmod +x trên macOS/Linux):
```bash
#!/bin/sh
python3 tools/md_lint.py || exit 1
python3 tools/guard.py scan || exit 1
python3 tools/make_figures.py --check || exit 1
```
Windows: thay `python3` bằng `python`.

## 4. Quy trình làm việc

### 4.1 Lần đầu thiết lập
1. Điền `report/metadata.yaml`; agent cập nhật `report/chapters.txt` theo thứ tự chương hoàn thiện.
2. Đặt dữ liệu thật vào `data/`, kèm file `.meta.json` đúng quy tắc (xem `data/README.md`).
3. Agent bổ sung `references/danh-muc.md` từ nguồn đã đọc và xác minh (xem `references/README.md`).
4. Không yêu cầu agent vẽ ảnh thủ công; chỉ dùng dữ liệu/ảnh nguồn hợp lệ và pipeline hiện có.
5. Agent khai báo hình và bảng trong `figures/specs.json` (mẫu: `figures/specs.example.json`), rồi chạy pipeline.
6. Chạy:
   ```bash
   python tools/guard.py snapshot
   ```

### 4.2 Giao việc cho agent
Đưa agent đọc `BOOTSTRAP.md` đầu tiên. Agent sẽ đọc các file theo thứ tự và làm theo quy trình trong đó.

### 4.3 Sau khi agent làm xong
```bash
python tools/guard.py verify          # có thay đổi ngoài phạm vi không
python tools/md_lint.py               # cấu trúc chương
python tools/make_figures.py --check  # hình và bảng còn cập nhật không
python tools/verify_refs.py references/danh-muc.md   # DOI có tồn tại không (cần internet)
```
Đọc `claims/*.csv`: xử lý các dòng còn CITE hoặc REWRITE.

Nếu agent đã tạo chương mới: agent cập nhật `report/chapters.txt` đúng thứ tự.

### 4.4 Khi có dữ liệu mới hoặc sửa dữ liệu
```bash
python tools/make_figures.py
```
Sau đó chạy `guard snapshot` lại trước khi giao việc tiếp cho agent.

### 4.5 Xuất file
```bash
python tools/build.py docx
python tools/build.py docx --numbered
python tools/build.py pdf
```
`build.py` lần lượt: chạy md_lint, guard scan, make_figures --check; ghép chương và chèn bảng; nối `references/danh-muc.md`; chạy `check_manuscript.py` trên bản ghép `build/report.md`; rồi mới gọi pandoc. Lỗi ở bước nào thì dừng ở bước đó. `--skip-checks` bỏ qua kiểm tra, không khuyến khích.

Sau khi xuất, mở file và kiểm tra mục lục (trong Word nhấn F9), hình, bảng, ngắt trang.

## 5. Ghi chú quan trọng

- **Số trong văn bản** không được kiểm tra tự động với data. Đối chiếu bằng tay, hoặc nhờ agent ở vai Citation Auditor với file `data/`.
- **Trích dẫn**: `check_manuscript.py` bắt được câu nêu bằng chứng mà không có `[n]`, nhưng không hiểu nội dung. `verify_refs.py` chỉ kiểm tra tài liệu có DOI. Tài liệu không có DOI cần kiểm tra thủ công.
- **Đơn vị trên một biểu đồ**: không đặt các đại lượng khác đơn vị vào cùng một hình.
- **Dữ liệu tổng hợp hợp lệ**: nếu đồ án có chủ đích dùng dữ liệu tổng hợp, luật từ khóa trong `make_figures.py` (`FORBIDDEN_TOKENS`) sẽ chặn. Khi đó sửa có chủ đích và ghi lý do trong báo cáo.
- **Placeholder**: khi còn `[CẦN ...]`, `build.py` vẫn có thể chạy nhưng `check_manuscript.py` sẽ báo ERROR và dừng việc xuất. Đó là chủ ý.

## 6. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `No module named matplotlib` | Chưa cài | `python -m pip install matplotlib==3.10.8` |
| `thiếu file metadata` | Thiếu `.meta.json` | Tạo đúng tên, đủ 4 trường |
| `chứa 'mock'` hoặc từ khóa khác | Dữ liệu hoặc metadata có từ khóa giả lập | Kiểm tra nguồn; không lách bằng đổi tên |
| `STALE` | Dữ liệu, specs hoặc file đầu ra đã đổi | Chạy lại `make_figures.py` |
| `đã bị sửa tay` | File trong assets/ hoặc tables/ bị sửa | Chạy lại `make_figures.py` |
| `Bảng viết tay bị cấm` | Viết bảng bằng `\| ... \|` | Dùng `{{TABLE tab-x-y}}` |
| `check_manuscript báo ERROR` trong build | Còn placeholder, thiếu nguồn, hoặc [n] sai thứ tự | Xem báo cáo, sửa trong chương hoặc danh mục |
| `Chưa có bảng ...` | Chưa chạy make_figures | `python tools/make_figures.py` |
| `pandoc: command not found` | Chưa cài hoặc terminal chưa tải lại PATH | Mở terminal mới |
| `xelatex not found` | Chưa cài LaTeX | Xem 3.4 |
| Tiếng Việt lỗi trong PDF | Chưa đặt `mainfont` | Xem 3.4 |
