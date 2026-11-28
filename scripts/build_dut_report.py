#!/usr/bin/env python3
"""Build the official DUT QĐ 209/QĐ-ĐHBK course report (.docx + .pdf) from course_report_vi.md.

Adapted directly from the project's canonical DUT builder (`build_report_docx.py`), upgraded with:
1. Native Word OMML (`m:oMath`) equation rendering via Pandoc for both `$$...$$` display equations
   (3-column borderless table: 15%--70%--15% with right-aligned `(N.M)` numbering per DUT standard)
   and `$...$` inline math — ZERO screenshot/PNG equations (`![EQ]`) and ZERO raw LaTeX leaks.
2. Exact DUT cover page (Times New Roman, Khoa CNTT logo, student IDs `102210034` & `102210045`,
   class `21TCLC_DT2`, Học kỳ I 2026–2027, GVHD TS. Huỳnh Hữu Hùng).
3. Roman front matter (`i, ii, ...`: MỤC LỤC, DANH MỤC HÌNH VẼ, DANH MỤC BẢNG BIỂU,
   DANH MỤC KÝ HIỆU VÀ CHỮ VIẾT TẮT) and Arabic body (`1, 2, ...`: TÓM TẮT, Chương 1–3,
   KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN, TÀI LIỆU THAM KHẢO, PHỤ LỤC) with running header/footer.
4. Automatic two-pass LibreOffice Writer PDF conversion and exact page-number backfilling for
   TOC, List of Figures, and List of Tables.
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from lxml import etree

MD_PATH = Path("docs/report/course_report_vi.md")
OUT_DOCX = Path("docs/report/course_report_vi.docx")
OUT_PDF = Path("docs/report/course_report_vi.pdf")

FONT = "Times New Roman"
BODY = Pt(13)
LINE = 1.3
H1 = Pt(14)       # "Chương N: ..." bold, centered, ALL CAPS
H2 = Pt(13)       # "N.M." bold, left
H3 = Pt(13)       # "N.M.K." bold italic, left
CAPTION = Pt(13)  # figure/table captions (DUT form: 13 pt)
TABLE_TXT = Pt(11.5)

TITLE_SHORT = (
    "GauCon: Đầu phát hiện đồng thuận Gauss cục bộ cho phát hiện người cực nhỏ trên ảnh hàng hải"
)
COVER_TITLE = (
    "GauCon: Đầu phát hiện đồng thuận Gauss cục bộ\n"
    "và gán nhãn Wasserstein tách rời cho phát hiện\n"
    "người cực nhỏ không dùng NMS trên ảnh hàng hải"
)

CHAPTERS = {
    "1": "TỔNG QUAN NGHIÊN CỨU LIÊN QUAN",
    "2": "GIẢI PHÁP ĐỀ XUẤT",
    "3": "THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ",
}
UNNUMBERED_H1 = {
    "TÓM TẮT",
    "KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN",
    "TÀI LIỆU THAM KHẢO",
    "PHỤ LỤC",
}

ABBREVIATIONS = [
    ("AP", "Average Precision — độ chính xác trung bình tích phân dưới đường cong Precision–Recall"),
    ("AP25 / AP50 / AP75", "Độ chính xác trung bình đánh giá tại ngưỡng IoU 0,25 / 0,50 / 0,75"),
    ("AR", "Average Recall — độ phủ trung bình trên tập ngưỡng IoU chuẩn COCO"),
    ("BCE", "Binary Cross-Entropy — hàm mất mát entropy chéo nhị phân"),
    ("BN", "Batch Normalization (BatchNorm) — chuẩn hóa đặc trưng theo batch"),
    ("CIoU", "Complete Intersection over Union — hàm mất mát hồi quy hộp bao đầy đủ"),
    ("COCO", "Microsoft Common Objects in Context — bộ dữ liệu phát hiện đối tượng chuẩn"),
    ("DFL", "Distribution Focal Loss — hàm mất mát phân phối biên hộp"),
    ("DW-TAL", "Decoupled Wasserstein Task-Aligned Learning — gán nhãn Wasserstein tách rời khâu hiệu chuẩn"),
    ("E_bkg / E_dupe / E_loc / E_miss", "Bốn thành phần lỗi TIDE: báo nhầm nền, hộp trùng lặp, lệch vị trí, bỏ sót đối tượng"),
    ("FG-Gate", "Foreground-Gated Relational MLP — cổng điều biến logit lân cận theo xác suất tiền cảnh"),
    ("FP / FN / TP", "False Positive (báo nhầm) / False Negative (bỏ sót) / True Positive (khớp đúng)"),
    ("FPN / PANet", "Feature Pyramid Network / Path Aggregation Network — khối Neck tháp đặc trưng đa tỉ lệ"),
    ("FPPI", "False Positives Per Image — số dự đoán báo nhầm trung bình trên mỗi ảnh"),
    ("FPS", "Frames Per Second — số khung hình suy luận được trong một giây"),
    ("GauCon", "Local Gaussian Consensus Head — mô-đun đồng thuận Gauss cục bộ (đề xuất)"),
    ("GFLOPs", "Giga Floating-Point Operations — số tỉ phép tính dấu chấm động cho một lượt suy luận"),
    ("GT", "Ground Truth — hộp bao nhãn thật được chú thích thủ công trên ảnh gốc"),
    ("GWD / NWD", "Gaussian Wasserstein Distance / Normalized Wasserstein Distance"),
    ("H_1:1 / H_1:M", "Nhánh suy luận gán nhãn đơn mục tiêu (one-to-one) và nhánh phụ đa mẫu (one-to-many)"),
    ("IoD / IOS", "Intersection over Detection / Intersection over Smaller-Area"),
    ("IoU", "Intersection over Union — tỷ số diện tích phần giao trên phần hợp của hai hộp bao"),
    ("MLP", "Multi-Layer Perceptron — mạng truyền thẳng đa tầng với phép chiếu 1×1"),
    ("MR50", "Log-Average Miss Rate tại ngưỡng IoU 0,50 trên 9 mốc FPPI từ 10⁻² đến 10⁰"),
    ("NMS", "Non-Maximum Suppression — thuật toán hậu xử lý loại bỏ khung bao trùng lặp"),
    ("P2, P3, P4, P5", "Bốn tầng bản đồ đặc trưng có bước lưới (stride) lần lượt bằng 4, 8, 16, 32"),
    ("RFLA", "Receptive Field based Label Assignment — gán nhãn dựa trên trường tiếp nhận Gauss"),
    ("SAHI", "Slicing Aided Hyper Inference — giao thức suy luận cắt ảnh thành các lát cắt chồng lấn"),
    ("SAR", "Search and Rescue — nghiệp vụ tìm kiếm và cứu nạn trên biển"),
    ("SiLU", "Sigmoid-Weighted Linear Unit — hàm kích hoạt phi tuyến x · σ(x)"),
    ("TAL / TOOD", "Task-Aligned Learning / Task-Aligned One-Stage Object Detection"),
    ("TIDE", "Toolbox for Identifying Object Detection Errors — khung phân rã thành phần lỗi phát hiện"),
    ("UAV", "Unmanned Aerial Vehicle — thiết bị bay không người lái"),
    ("VRAM", "Video Random-Access Memory — bộ nhớ truy cập ngẫu nhiên trên GPU"),
    ("WBF", "Weighted Boxes Fusion — thuật toán hợp nhất hộp bao theo trọng số độ tin cậy"),
]


# ---------------------------------------------------------------------------
# Batch Pandoc LaTeX -> Native Word OMML converter (1 subprocess call total)
# ---------------------------------------------------------------------------

def build_omml_cache(md_text: str) -> dict[str, etree._Element]:
    """Convert every unique inline `$...$` and display `$$...$$` formula in md_text into `m:oMath`."""
    display_exprs = [m.group(1).strip() for m in re.finditer(r"^\$\$(.+?)\$\$(?:\s*\([\d.]+\))?\s*$", md_text, re.MULTILINE)]
    cleaned = re.sub(r"^\$\$.+?\$\$(?:\s*\([\d.]+\))?\s*$", "", md_text, flags=re.MULTILINE)
    inline_exprs = [m.group(1).strip() for m in re.finditer(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)", cleaned)]

    unique: list[str] = []
    seen: set[str] = set()
    for expr in display_exprs + inline_exprs:
        # Remove manual \tag{...} if present; equation numbers are placed in right column
        norm = re.sub(r"\\tag\{[^}]*\}", "", expr).strip()
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(norm)

    if not unique:
        return {}

    probe_lines = []
    for idx, expr in enumerate(unique):
        probe_lines.append(f"EQMARK_{idx}_START ${expr}$ EQMARK_{idx}_END")
    probe_md = "\n\n".join(probe_lines) + "\n"

    with tempfile.TemporaryDirectory(prefix="dut-omml-") as tmp:
        out_docx = Path(tmp) / "omml_bank.docx"
        subprocess.run(
            ["pandoc", "-f", "markdown+tex_math_dollars", "-t", "docx", "-o", str(out_docx)],
            input=probe_md.encode("utf-8"),
            check=True,
        )
        with zipfile.ZipFile(out_docx) as zf:
            xml_bytes = zf.read("word/document.xml")

    root = etree.fromstring(xml_bytes)
    ns = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    }
    cache: dict[str, etree._Element] = {}
    for p in root.findall(".//w:body/w:p", ns):
        full_txt = "".join(t.text or "" for t in p.findall(".//w:t", ns))
        m_idx = re.search(r"EQMARK_(\d+)_START", full_txt)
        omath = p.find(".//m:oMath", ns)
        if m_idx is not None and omath is not None:
            idx = int(m_idx.group(1))
            cache[unique[idx]] = copy.deepcopy(omath)
    return cache


# ---------------------------------------------------------------------------
# Low-level OpenXML formatting helpers (strict QĐ 209/QĐ-ĐHBK compliance)
# ---------------------------------------------------------------------------

def set_font(run, size=BODY, bold: bool = False, italic: bool = False, font_name: str = FONT) -> None:
    run.font.name = font_name
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), font_name)


INLINE_TOKEN_RE = re.compile(r"(\*\*.+?\*\*|`.+?`|(?<!\$)\$(?!\$)[^$\n]+?\$(?!\$)|\*[^*\n]+\*)")


def append_omml(paragraph, expr: str, omml_cache: dict[str, etree._Element], size=BODY) -> None:
    norm = re.sub(r"\\tag\{[^}]*\}", "", expr).strip()
    omath = omml_cache.get(norm)
    if omath is not None:
        # Clone lxml element into python-docx's oxml tree cleanly via XML serialization
        omath_xml = etree.tostring(omath, encoding="unicode")
        paragraph._p.append(parse_xml(omath_xml))
    else:
        r = paragraph.add_run(norm)
        set_font(r, size=size, italic=True)


def add_rich_runs(
    p,
    text: str,
    omml_cache: dict[str, etree._Element],
    size=BODY,
    bold: bool = False,
    italic: bool = False,
) -> None:
    for chunk in INLINE_TOKEN_RE.split(text):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**") and len(chunk) > 4:
            inner = chunk[2:-2]
            add_rich_runs(p, inner, omml_cache, size=size, bold=True, italic=italic)
        elif chunk.startswith("`") and chunk.endswith("`") and len(chunk) > 2:
            r = p.add_run(chunk[1:-1])
            set_font(r, size=Pt(max(10.5, size.pt - 1.0)), bold=bold, italic=False, font_name="Courier New")
        elif chunk.startswith("$") and chunk.endswith("$") and len(chunk) > 2:
            append_omml(p, chunk[1:-1], omml_cache, size=size)
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            inner = chunk[1:-1]
            add_rich_runs(p, inner, omml_cache, size=size, bold=bold, italic=True)
        else:
            r = p.add_run(chunk)
            set_font(r, size=size, bold=bold, italic=italic)


def para(
    doc,
    text: str = "",
    omml_cache: dict[str, etree._Element] | None = None,
    *,
    size=BODY,
    bold: bool = False,
    italic: bool = False,
    align=WD_ALIGN_PARAGRAPH.JUSTIFY,
    space_after=None,
    space_before=None,
    indent=None,
    line: float = LINE,
):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(6) if space_after is None else space_after
    pf.space_before = Pt(0) if space_before is None else space_before
    pf.line_spacing = line
    if indent is not None:
        pf.first_line_indent = indent
    if text:
        add_rich_runs(p, text, omml_cache or {}, size=size, bold=bold, italic=italic)
    return p


def add_display_equation(
    doc,
    expr: str,
    eq_num: str,
    omml_cache: dict[str, etree._Element],
) -> None:
    """DUT standard display equation: 3-column borderless table (15% - 70% - 15%), right-aligned (N.M)."""
    t = doc.add_table(rows=1, cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        borders.append(el)
    tblPr.append(borders)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)

    widths_cm = (1.1, 13.4, 1.5)  # sums to 16.0 cm usable width (21.0 - 3.0 - 2.0)
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(Cm(widths_cm[j]).twips)))

    row = t.rows[0]
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))
    for j, w_cm in enumerate(widths_cm):
        cell = row.cells[j]
        cell.width = Cm(w_cm)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Center cell: native OMML equation
    p_mid = row.cells[1].paragraphs[0]
    p_mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_mid.paragraph_format.space_before = Pt(4)
    p_mid.paragraph_format.space_after = Pt(4)
    p_mid.paragraph_format.line_spacing = 1.15
    append_omml(p_mid, expr, omml_cache, size=BODY)

    # Right cell: equation number `(2.1)`
    p_right = row.cells[2].paragraphs[0]
    p_right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_right.paragraph_format.space_before = Pt(4)
    p_right.paragraph_format.space_after = Pt(4)
    if eq_num:
        r_num = p_right.add_run(eq_num)
        set_font(r_num, size=BODY)


def add_page_number_field(p, size=None, italic: bool = False) -> None:
    if size is None:
        size = Pt(10)
    run = p.add_run()
    set_font(run, size=size, italic=italic)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    inner = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), FONT)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size.pt * 2)))
    rPr.append(rFonts)
    rPr.append(sz)
    inner.append(rPr)
    t = OxmlElement("w:t")
    t.text = "1"
    inner.append(t)
    fld.append(inner)
    run._element.addnext(fld)


def set_page_numbering(section, fmt: str = "decimal", start: int = 1) -> None:
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        sectPr.append(pg)
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), str(start))


def clear_footer(section) -> None:
    section.footer.is_linked_to_previous = False
    for p in list(section.footer.paragraphs):
        p.text = ""


def set_table_borders_dut(t) -> None:
    """DUT academic booktabs table: thick top/bottom, thin horizontal rules, no vertical borders."""
    tbl = t._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz, val in (
        ("top", "12", "single"),
        ("bottom", "12", "single"),
        ("insideH", "4", "single"),
        ("left", "0", "none"),
        ("right", "0", "none"),
        ("insideV", "0", "none"),
    ):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), sz)
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tblPr.append(borders)
    hdr = t.rows[0]
    trPr = hdr._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))


def add_caption(doc, caption: str, kind: str, omml_cache: dict[str, etree._Element]):
    """DUT 13pt centered caption: '**Hình N.M.** Rest of caption'."""
    clean = caption.replace("**", "").strip()
    clean = re.sub(rf"^{kind}\s+(\d+\.\d+)\s*[:—–.-]\s*", rf"{kind} \1. ", clean)
    label, sep, rest = clean.partition(". ")
    p = para(doc, "", align=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(8), line=1.15, size=CAPTION)
    r_lbl = p.add_run(label + (". " if sep else ""))
    set_font(r_lbl, size=CAPTION, bold=True)
    if rest:
        add_rich_runs(p, rest, omml_cache, size=CAPTION, bold=False, italic=False)
    return p


def add_table(doc, header: list[str], rows: list[list[str]], caption: str | None, omml_cache: dict[str, etree._Element]):
    if caption:
        p_cap = add_caption(doc, caption, "Bảng", omml_cache)
        p_cap.paragraph_format.space_before = Pt(10)
        p_cap.paragraph_format.space_after = Pt(4)
        p_cap.paragraph_format.keep_with_next = True

    n = len(header)
    font_sz = Pt(10.5) if n >= 8 else (Pt(11.0) if n >= 6 else Pt(11.5))
    t = doc.add_table(rows=1 + len(rows), cols=n)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_table_borders_dut(t)

    tblPr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), "40")
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tblPr.append(mar)

    # Smart proportional widths based on column text lengths so columns never crush
    max_lens = []
    for j in range(n):
        samples = [re.sub(r"[$*`]", "", header[j])] + [
            re.sub(r"[$*`]", "", r[j]) for r in rows if j < len(r)
        ]
        longest = max((len(s) for s in samples), default=6)
        weight = max(5.5, min(float(longest), 34.0 if j == 0 else 22.0))
        if j == 0:
            weight = max(weight, 14.0)
        max_lens.append(weight)
    total_w = sum(max_lens)
    ratios = [w / total_w for w in max_lens]

    usable_cm = 16.0
    for j, ratio in enumerate(ratios):
        for row in t.rows:
            row.cells[j].width = Cm(usable_cm * ratio)
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(Cm(usable_cm * ratios[j]).twips)))

    for j, cell in enumerate(t.rows[0].cells):
        cell.text = ""
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.05
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        add_rich_runs(p, header[j], omml_cache, size=font_sz, bold=True)

    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))

    for i, row_vals in enumerate(rows, start=1):
        for j in range(n):
            val = row_vals[j] if j < len(row_vals) else ""
            cell = t.rows[i].cells[j]
            cell.text = ""
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            # Text columns left-aligned, short numeric/metric columns centered
            plain_val = re.sub(r"[$*`%+−-]", "", val).strip()
            is_numeric_like = j > 0 and (len(plain_val) <= 16 or any(ch.isdigit() for ch in plain_val))
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (is_numeric_like and n >= 4) else (
                WD_ALIGN_PARAGRAPH.LEFT if j == 0 or len(plain_val) > 22 else WD_ALIGN_PARAGRAPH.CENTER
            )
            p.paragraph_format.line_spacing = 1.08
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            add_rich_runs(p, val, omml_cache, size=font_sz)

    para(doc, "", space_after=Pt(4), line=1.0)
    return t


def add_figure(doc, path: str, caption: str, omml_cache: dict[str, etree._Element]) -> None:
    img_path = Path(path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Missing report figure: {img_path}")
    width_cm = 15.2
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            aspect = im.width / max(im.height, 1)
            if aspect < 1.75:
                width_cm = 10.8  # compact single-panel chart
            else:
                width_cm = 15.4  # wide two-panel / architecture chart
    except OSError:
        pass
    if "gallery" in img_path.name:
        width_cm = 15.4
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(img_path), width=Cm(width_cm))
    add_caption(doc, caption, "Hình", omml_cache)


# ---------------------------------------------------------------------------
# Cover, TOC, Lists of Figures/Tables, and Abbreviations (Exact DUT Format)
# ---------------------------------------------------------------------------

def build_cover(doc) -> None:
    para(doc, "", line=1.0, space_after=Pt(0))
    para(doc, "ĐẠI HỌC ĐÀ NẴNG", size=Pt(14), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.2, space_after=Pt(2))
    para(doc, "TRƯỜNG ĐẠI HỌC BÁCH KHOA", size=Pt(14), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.2, space_after=Pt(2))
    para(doc, "KHOA CÔNG NGHỆ THÔNG TIN", size=Pt(14), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.2, space_after=Pt(6))

    logo_path = Path("docs/report/figures/logo_dut_it.jpg")
    if logo_path.is_file():
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(6)
        p_logo.paragraph_format.space_after = Pt(14)
        p_logo.add_run().add_picture(str(logo_path), width=Cm(3.0))

    para(doc, "BÁO CÁO ĐỒ ÁN MÔN HỌC", size=Pt(20), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.2, space_after=Pt(6))
    para(doc, "NGÀNH: CÔNG NGHỆ THÔNG TIN", size=Pt(13), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.2, space_after=Pt(2))
    para(doc, "CHUYÊN NGÀNH: KHOA HỌC MÁY TÍNH", size=Pt(13), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.2, space_after=Pt(18))

    para(doc, "ĐỀ TÀI:", size=Pt(14), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.2, space_after=Pt(4))
    para(doc, COVER_TITLE, size=Pt(16.5), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.3, space_after=Pt(22))

    info = [
        ("Môn học:", "Thị giác máy tính"),
        ("Học kỳ:", "Học kỳ I, 2026–2027"),
        ("Người hướng dẫn:", "TS. Huỳnh Hữu Hùng"),
        ("Sinh viên thực hiện:", "Trần Đức Long"),
        ("Số thẻ sinh viên:", "102210034"),
        ("Lớp:", "21TCLC_DT2"),
        ("Sinh viên thực hiện:", "Phan Thị Thảo Tiên"),
        ("Số thẻ sinh viên:", "102210045"),
        ("Lớp:", "21TCLC_DT2"),
    ]
    for label, value in info:
        p = para(doc, "", align=WD_ALIGN_PARAGRAPH.LEFT, space_after=Pt(2), line=1.25)
        p.paragraph_format.left_indent = Cm(2.8)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(7.6))
        r_lbl = p.add_run(label)
        set_font(r_lbl, bold=True)
        p.add_run("\t")
        r_val = p.add_run(value)
        set_font(r_val, bold=(label in {"Người hướng dẫn:", "Sinh viên thực hiện:"}))

    para(doc, "", line=1.0, space_after=Pt(8))
    para(doc, "Đà Nẵng, tháng 11 năm 2026", size=Pt(13), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, line=1.2, space_after=Pt(0))


def _plain_toc_text(text: str) -> str:
    """Convert markdown/LaTeX math inside heading titles into crisp plain Unicode for TOC & Lists."""
    s = re.sub(r"[$*`]", "", text)
    s = s.replace(r"\times", "×").replace(r"\mathrm{IoU}", "IoU")
    s = re.sub(r"_\{([^}]+)\}", r"\1", s)
    s = re.sub(r"_(\d+)", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def toc_entries_from_md(md_text: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for line in md_text.splitlines():
        m = re.match(r"^(#{2,4})\s+(.*)$", line.strip())
        if not m:
            continue
        level, text = len(m.group(1)), m.group(2).strip()
        if text in ("MỤC LỤC", "THỊ GIÁC MÁY TÍNH"):
            continue
        if level == 2:
            mm = re.match(r"^(\d)\.\s+(.*)$", text)
            if mm and mm.group(1) in CHAPTERS:
                entries.append((2, f"Chương {mm.group(1)}: {mm.group(2).strip()}"))
            elif text in UNNUMBERED_H1:
                entries.append((2, text))
        elif level in (3, 4) and (re.match(r"^\d+\.\d+", text) or re.match(r"^Phụ lục [A-Z]", text)):
            entries.append((level, text))
    return entries


def build_toc(
    doc,
    entries: list[tuple[int, str]],
    page_map: dict[str, int] | None = None,
    front_map: dict[str, str] | None = None,
) -> None:
    para(doc, "MỤC LỤC", size=Pt(14), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=Pt(12), line=1.2)
    front_items = [
        "DANH MỤC HÌNH VẼ",
        "DANH MỤC BẢNG BIỂU",
        "DANH MỤC KÝ HIỆU VÀ CHỮ VIẾT TẮT",
    ]
    for f_item in front_items:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = LINE
        pf.space_after = Pt(2)
        pf.tab_stops.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        r = p.add_run(f_item)
        set_font(r, bold=True)
        pg = (front_map or {}).get(f_item)
        if pg:
            r2 = p.add_run("\t" + pg)
            set_font(r2, bold=True)

    for level, entry in entries:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.line_spacing = LINE
        pf.space_after = Pt(2)
        pf.left_indent = Cm(0.0 if level == 2 else (0.6 if level == 3 else 1.2))
        pf.tab_stops.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        clean_entry = _plain_toc_text(entry)
        r = p.add_run(clean_entry)
        set_font(r, bold=(level == 2))
        pg = (page_map or {}).get(entry)
        if pg is not None:
            r2 = p.add_run("\t" + str(pg))
            set_font(r2, bold=(level == 2))
    doc.add_page_break()


def build_lists(
    doc,
    fig_caps: list[tuple[str, str]],
    tab_caps: list[tuple[str, str]],
    fig_map: dict[str, int] | None,
    tab_map: dict[str, int] | None,
    omml_cache: dict[str, etree._Element],
) -> None:
    for title, caps, pmap, prefix in (
        ("DANH MỤC HÌNH VẼ", fig_caps, fig_map, "Hình"),
        ("DANH MỤC BẢNG BIỂU", tab_caps, tab_map, "Bảng"),
    ):
        para(doc, title, size=Pt(14), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=Pt(0), space_after=Pt(2), line=1.05)
        for num, text in caps:
            clean_cap = text.replace("**", "").strip()
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(16.0), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            p.paragraph_format.left_indent = Cm(1.8)
            p.paragraph_format.first_line_indent = Cm(-1.8)
            label = f"{prefix} {num}. "
            r = p.add_run(label)
            set_font(r, size=Pt(12), bold=True)
            add_rich_runs(p, clean_cap, omml_cache, size=Pt(12))
            pg = (pmap or {}).get(f"{prefix} {num}.")
            if pg is not None:
                r2 = p.add_run("\t" + str(pg))
                set_font(r2, size=Pt(12))
        doc.add_page_break()


def add_algorithm_box(doc, code_text: str, omml_cache: dict[str, etree._Element]) -> None:
    """Render an algorithm block as a 3-rule academic table with native OMML math."""
    lines = [ln.rstrip() for ln in code_text.splitlines() if ln.strip()]
    if not lines:
        return
    title_line = lines[0]
    io_lines = [ln for ln in lines[1:] if ln.startswith(("Đầu vào:", "Đầu ra:", "Input:", "Output:"))]
    step_lines = [ln for ln in lines[1:] if ln not in io_lines]

    t = doc.add_table(rows=3 if io_lines else 2, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz, val in (
        ("top", "12", "single"),
        ("bottom", "12", "single"),
        ("insideH", "4", "single"),
        ("left", "0", "none"),
        ("right", "0", "none"),
        ("insideV", "0", "none"),
    ):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), sz)
        el.set(qn("w:color"), "1F2937")
        borders.append(el)
    tblPr.append(borders)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc in grid.findall(qn("w:gridCol")):
            gc.set(qn("w:w"), str(int(Cm(16.0).twips)))

    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    c0 = t.rows[0].cells[0]
    c0.width = Cm(16.0)
    shd = parse_xml('<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="F1F5F9"/>')
    c0._tc.get_or_add_tcPr().append(shd)
    p0 = c0.paragraphs[0]
    p0.paragraph_format.space_before = Pt(4)
    p0.paragraph_format.space_after = Pt(4)
    p0.paragraph_format.line_spacing = 1.15
    p0.paragraph_format.keep_with_next = True
    hdr, sep, rest = title_line.partition(":")
    r_hdr = p0.add_run(hdr + (":" if sep else ""))
    set_font(r_hdr, size=Pt(12), bold=True)
    if rest:
        add_rich_runs(p0, rest, omml_cache, size=Pt(12))

    row_idx = 1
    if io_lines:
        t.rows[row_idx]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        cio = t.rows[row_idx].cells[0]
        cio.width = Cm(16.0)
        cio.text = ""
        for k, iln in enumerate(io_lines):
            pio = cio.paragraphs[0] if k == 0 else cio.add_paragraph()
            pio.paragraph_format.space_before = Pt(2 if k == 0 else 0)
            pio.paragraph_format.space_after = Pt(2 if k == len(io_lines) - 1 else 1)
            pio.paragraph_format.line_spacing = 1.15
            pio.paragraph_format.keep_with_next = True
            lbl, csep, irest = iln.partition(":")
            r_l = pio.add_run(lbl + csep)
            set_font(r_l, size=Pt(11.5), bold=True)
            add_rich_runs(pio, irest, omml_cache, size=Pt(11.5))
        row_idx += 1

    cb = t.rows[row_idx].cells[0]
    cb.width = Cm(16.0)
    cb.text = ""
    t.rows[row_idx]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    for k, sln in enumerate(step_lines):
        ps = cb.paragraphs[0] if k == 0 else cb.add_paragraph()
        ps.paragraph_format.space_before = Pt(3 if k == 0 else 0)
        ps.paragraph_format.space_after = Pt(4 if k == len(step_lines) - 1 else 1.5)
        ps.paragraph_format.line_spacing = 1.15
        ps.paragraph_format.left_indent = Cm(0.8 if "    " in sln[:10] else 0.2)
        num_part, sp_sep, body_part = sln.strip().partition(" ")
        r_n = ps.add_run(num_part + sp_sep)
        set_font(r_n, size=Pt(11.5), bold=True)
        add_rich_runs(ps, body_part, omml_cache, size=Pt(11.5))
    para(doc, "", space_after=Pt(6), line=1.0)


def build_abbreviations(doc) -> None:
    para(doc, "DANH MỤC KÝ HIỆU VÀ CHỮ VIẾT TẮT", size=Pt(14), bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=Pt(0), space_after=Pt(12), line=1.2)
    t = doc.add_table(rows=1 + len(ABBREVIATIONS), cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    set_table_borders_dut(t)
    widths = (3.8, 12.2)
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(Cm(widths[j]).twips)))

    headers = ("Ký hiệu / Từ viết tắt", "Giải nghĩa đầy đủ")
    for j, htxt in enumerate(headers):
        c = t.rows[0].cells[j]
        c.width = Cm(widths[j])
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j == 0 else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(htxt)
        set_font(r, size=Pt(12), bold=True)

    for i, (sym, meaning) in enumerate(sorted(ABBREVIATIONS, key=lambda kv: kv[0].lower()), start=1):
        row = t.rows[i]
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for j, val in enumerate((sym, meaning)):
            cell = row.cells[j]
            cell.width = Cm(widths[j])
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val)
            set_font(r, size=Pt(12), bold=(j == 0))


# ---------------------------------------------------------------------------
# Markdown block parser (supports native $$...$$ equations, tables, figures, code)
# ---------------------------------------------------------------------------

CAP_FIG_RE = re.compile(r"^\*\*Hình\s+(\d+\.\d+)\s*[:—–.-]\s*(.+)$", re.MULTILINE)
CAP_TAB_RE = re.compile(r"^\*\*Bảng\s+(\d+\.\d+)\s*[:—–.-]\s*(.+)$", re.MULTILINE)
IMG_RE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)$")
DISP_EQ_RE = re.compile(r"^\$\$(.+?)\$\$(?:\s*(\([\d.]+\)))?\s*$")


def tight_ranges(text: str) -> str:
    if "$" in text or text.strip().startswith("!["):
        return text
    return re.sub(r"(?<=\d)\s*–\s*(?=[\d,])", "–", text)


def parse_md(md_text: str) -> list[tuple]:
    raw_lines = md_text.splitlines()
    blocks: list[tuple] = []
    i = 0
    pending_tab_caption: str | None = None

    while i < len(raw_lines):
        raw = raw_lines[i]
        stripped = raw.strip()
        if not stripped or stripped == "---" or stripped.startswith(":::::"):
            i += 1
            continue
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            code = []
            i += 1
            while i < len(raw_lines) and not raw_lines[i].strip().startswith("```"):
                code.append(raw_lines[i])
                i += 1
            i += 1
            blocks.append(("algorithm" if lang == "algorithm" else "code", "\n".join(code)))
            continue
        # Multi-line or single-line $$ ... $$ display equation
        if stripped.startswith("$$"):
            if stripped.endswith("$$") and len(stripped) > 4 or DISP_EQ_RE.match(stripped):
                m_eq = DISP_EQ_RE.match(stripped)
                if m_eq:
                    blocks.append(("eq", m_eq.group(1).strip(), m_eq.group(2) or ""))
                    i += 1
                    continue
            eq_lines = [stripped[2:]]
            i += 1
            eq_num = ""
            while i < len(raw_lines):
                ln = raw_lines[i].strip()
                m_end = re.match(r"^(.*)\$\$\s*(\([\d.]+\))?\s*$", ln)
                if m_end:
                    eq_lines.append(m_end.group(1))
                    eq_num = m_end.group(2) or ""
                    i += 1
                    break
                eq_lines.append(ln)
                i += 1
            expr_joined = " ".join(x for x in eq_lines if x).strip()
            m_tag = re.search(r"\\tag\{([^}]*)\}", expr_joined)
            if m_tag and not eq_num:
                eq_num = f"({m_tag.group(1)})"
                expr_joined = re.sub(r"\\tag\{[^}]*\}", "", expr_joined).strip()
            blocks.append(("eq", expr_joined, eq_num))
            continue

        m_h = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m_h:
            blocks.append(("h", len(m_h.group(1)), m_h.group(2).strip()))
            i += 1
            continue
        if stripped.startswith("**Bảng"):
            pending_tab_caption = stripped
            i += 1
            continue
        if stripped.startswith("|"):
            tbl = []
            while i < len(raw_lines) and raw_lines[i].strip().startswith("|"):
                tbl.append([c.strip() for c in raw_lines[i].strip().strip("|").split("|")])
                i += 1
            if len(tbl) >= 2:
                blocks.append(("table", pending_tab_caption, tbl[0], tbl[2:]))
            pending_tab_caption = None
            continue
        if stripped.startswith("- ") or re.match(r"^\d+\.\s", stripped):
            blocks.append(("item", tight_ranges(stripped)))
            i += 1
            continue
        blocks.append(("p", tight_ranges(stripped)))
        i += 1
    return blocks


# ---------------------------------------------------------------------------
# Pass-2 PDF page discovery (Roman front matter + Arabic body pages)
# ---------------------------------------------------------------------------

def to_roman_upper(n: int) -> str:
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, sym in vals:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def find_heading_pages(
    pdf_path: Path,
    entries: list[tuple[int, str]],
    fig_caps: list[tuple[str, str]],
    tab_caps: list[tuple[str, str]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, str]]:
    info = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True).stdout
    m_pg = re.search(r"Pages:\s+(\d+)", info)
    npages = int(m_pg.group(1)) if m_pg else 1

    page_map: dict[str, int] = {}
    fig_map: dict[str, int] = {}
    tab_map: dict[str, int] = {}
    front_map: dict[str, str] = {}
    body_start: int | None = None

    for pg in range(2, npages + 1):
        txt = subprocess.run(
            ["pdftotext", "-f", str(pg), "-l", str(pg), "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if body_start is None:
            if TITLE_SHORT[:36] in txt:
                body_start = pg
            else:
                roman_idx = pg - 1  # cover is page 1; front matter starts at page 2 = I
                if "MỤC LỤC" not in txt:
                    for f_head in ("DANH MỤC HÌNH VẼ", "DANH MỤC BẢNG BIỂU", "DANH MỤC KÝ HIỆU VÀ CHỮ VIẾT TẮT"):
                        if f_head not in front_map and re.search(rf"^\s*{re.escape(f_head)}\s*$", txt, re.MULTILINE):
                            front_map[f_head] = to_roman_upper(roman_idx)
                continue

        norm_txt = re.sub(r"\s+", " ", txt)
        for _, entry in entries:
            if entry in page_map:
                continue
            probe = _plain_toc_text(entry)[:28].strip()
            if probe and probe in norm_txt:
                page_map[entry] = pg
        for num, _ in fig_caps:
            key = f"Hình {num}."
            if key not in fig_map and key in norm_txt:
                fig_map[key] = pg
        for num, _ in tab_caps:
            key = f"Bảng {num}."
            if key not in tab_map and key in norm_txt:
                tab_map[key] = pg

    offset = (body_start or 2) - 1
    shrink = lambda d: {k: v - offset for k, v in d.items()}
    return shrink(page_map), shrink(fig_map), shrink(tab_map), front_map


def render_docx(
    md_text: str,
    omml_cache: dict[str, etree._Element],
    entries: list[tuple[int, str]],
    fig_caps: list[tuple[str, str]],
    tab_caps: list[tuple[str, str]],
    page_map: dict[str, int] | None = None,
    fig_map: dict[str, int] | None = None,
    tab_map: dict[str, int] | None = None,
    front_map: dict[str, str] | None = None,
) -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = FONT
    style.font.size = BODY
    style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    # Section 0: Cover page (no header/footer, no page number)
    cover = doc.sections[0]
    cover.page_width, cover.page_height = Cm(21.0), Cm(29.7)
    cover.left_margin, cover.right_margin = Cm(3.0), Cm(2.0)
    cover.top_margin, cover.bottom_margin = Cm(2.5), Cm(2.5)
    cover.header_distance, cover.footer_distance = Cm(1.6), Cm(1.6)
    build_cover(doc)

    # Section 1: Front matter (Roman numerals i, ii, iii...)
    front = doc.add_section(WD_SECTION.NEW_PAGE)
    front.page_width, front.page_height = Cm(21.0), Cm(29.7)
    front.left_margin, front.right_margin = Cm(3.0), Cm(2.0)
    front.top_margin, front.bottom_margin = Cm(2.5), Cm(2.5)
    front.header_distance, front.footer_distance = Cm(1.6), Cm(1.6)
    set_page_numbering(front, "lowerRoman", 1)
    front.header.is_linked_to_previous = False
    clear_footer(front)
    fp = front.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number_field(fp, size=Pt(10))

    build_toc(doc, entries, page_map, front_map)
    build_lists(doc, fig_caps, tab_caps, fig_map, tab_map, omml_cache)
    build_abbreviations(doc)

    # Section 2: Main body (Arabic numerals restarting at 1, running header + student/advisor footer)
    body = doc.add_section(WD_SECTION.NEW_PAGE)
    body.page_width, body.page_height = Cm(21.0), Cm(29.7)
    body.left_margin, body.right_margin = Cm(3.0), Cm(2.0)
    body.top_margin, body.bottom_margin = Cm(2.5), Cm(2.5)
    body.header_distance, body.footer_distance = Cm(1.6), Cm(1.6)
    set_page_numbering(body, "decimal", 1)

    body.header.is_linked_to_previous = False
    hp = body.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hr = hp.add_run(TITLE_SHORT)
    set_font(hr, size=Pt(10), italic=False)

    clear_footer(body)
    bp = body.footer.paragraphs[0]
    bp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_ft = bp.add_run("Sinh viên thực hiện: Trần Đức Long, Phan Thị Thảo Tiên     –     GVHD: TS. Huỳnh Hữu Hùng")
    set_font(r_ft, size=Pt(10), italic=True)
    bp.add_run().add_break()
    add_page_number_field(bp, size=Pt(10))

    blocks = parse_md(md_text)
    body_started = False
    pending_fig: str | None = None
    skip_meta_re = re.compile(r"^\*\*(Đề tài|Giảng viên|Nhóm sinh viên|Đơn vị|Năm học):")

    for blk in blocks:
        kind = blk[0]
        if kind == "h":
            level, text = blk[1], blk[2]
            if text in ("BÁO CÁO ĐỒ ÁN MÔN HỌC", "THỊ GIÁC MÁY TÍNH", "MỤC LỤC"):
                continue
            m_ch = re.match(r"^(\d)\.\s+(.*)$", text)
            if level == 2 and (m_ch or text in UNNUMBERED_H1):
                if body_started:
                    doc.add_page_break()
                body_started = True
                if m_ch and m_ch.group(1) in CHAPTERS:
                    title = f"Chương {m_ch.group(1)}: {m_ch.group(2).strip()}"
                elif m_ch:
                    title = m_ch.group(2).strip()
                else:
                    title = text
                p_h1 = para(
                    doc,
                    title,
                    omml_cache,
                    size=H1,
                    bold=True,
                    align=WD_ALIGN_PARAGRAPH.CENTER,
                    space_before=Pt(12),
                    space_after=Pt(12),
                    line=1.2,
                )
                p_h1.paragraph_format.keep_with_next = True
                continue
            if not body_started:
                continue
            if level == 3:
                p_h2 = para(
                    doc,
                    text,
                    omml_cache,
                    size=H2,
                    bold=True,
                    align=WD_ALIGN_PARAGRAPH.LEFT,
                    space_before=Pt(12),
                    space_after=Pt(6),
                    line=1.2,
                )
                p_h2.paragraph_format.keep_with_next = True
            else:
                p_h3 = para(
                    doc,
                    text,
                    omml_cache,
                    size=H3,
                    bold=True,
                    italic=True,
                    align=WD_ALIGN_PARAGRAPH.LEFT,
                    space_before=Pt(10),
                    space_after=Pt(4),
                    line=1.2,
                )
                p_h3.paragraph_format.keep_with_next = True

        elif kind == "eq":
            if not body_started:
                continue
            add_display_equation(doc, blk[1], blk[2], omml_cache)

        elif kind == "p":
            text = blk[1]
            if not body_started or skip_meta_re.match(text):
                continue
            m_img = IMG_RE.match(text)
            if m_img:
                pending_fig = m_img.group(1)
                continue
            m_cf = CAP_FIG_RE.match(text)
            if m_cf and pending_fig:
                add_figure(doc, pending_fig, text, omml_cache)
                pending_fig = None
                continue
            if re.match(r"^\[\d+\]\s", text):
                p_ref = para(doc, text, omml_cache, space_after=Pt(3), align=WD_ALIGN_PARAGRAPH.JUSTIFY)
                p_ref.paragraph_format.left_indent = Cm(1.0)
                p_ref.paragraph_format.first_line_indent = Cm(-1.0)
            elif text.startswith(("*Ghi chú:", "Trong đó:")):
                para(doc, text, omml_cache, space_after=Pt(6), indent=Cm(0.0))
            else:
                para(doc, text, omml_cache, indent=Cm(1.0))

        elif kind == "table":
            if not body_started:
                continue
            add_table(doc, blk[2], blk[3], blk[1], omml_cache)

        elif kind == "algorithm":
            if not body_started:
                continue
            add_algorithm_box(doc, blk[1], omml_cache)

        elif kind == "code":
            if not body_started:
                continue
            for ln in blk[1].splitlines():
                p_cd = para(doc, "", space_after=Pt(0), line=1.05, align=WD_ALIGN_PARAGRAPH.LEFT)
                p_cd.paragraph_format.left_indent = Cm(1.0)
                r = p_cd.add_run(ln if ln else " ")
                set_font(r, size=Pt(10.5), font_name="Courier New")
            para(doc, "", space_after=Pt(6), line=1.0)

        elif kind == "item":
            if not body_started:
                continue
            p_it = para(doc, blk[1], omml_cache, space_after=Pt(3))
            p_it.paragraph_format.left_indent = Cm(1.5)
            p_it.paragraph_format.first_line_indent = Cm(-0.5)

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_DOCX))


def convert_docx_to_pdf(docx_path: Path, pdf_out: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="dut-lo-") as tmp:
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                tmp,
                str(docx_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        generated = Path(tmp) / (docx_path.stem + ".pdf")
        pdf_out.write_bytes(generated.read_bytes())


def main() -> int:
    md_text = MD_PATH.read_text(encoding="utf-8")
    print("[dut-build] Converting LaTeX formulas to native Word OMML via Pandoc...", flush=True)
    omml_cache = build_omml_cache(md_text)
    print(f"[dut-build] Cached {len(omml_cache)} unique native OMML equations.", flush=True)

    entries = toc_entries_from_md(md_text)
    fig_caps = CAP_FIG_RE.findall(md_text)
    tab_caps = CAP_TAB_RE.findall(md_text)

    if "--pass1-only" in sys.argv:
        render_docx(md_text, omml_cache, entries, fig_caps, tab_caps)
        print(f"[dut-build] Saved pass-1 {OUT_DOCX}")
        return 0

    # Pass 1 -> PDF -> extract true page numbers -> Pass 2 -> final PDF
    render_docx(md_text, omml_cache, entries, fig_caps, tab_caps)
    convert_docx_to_pdf(OUT_DOCX, OUT_PDF)
    page_map, fig_map, tab_map, front_map = find_heading_pages(OUT_PDF, entries, fig_caps, tab_caps)

    render_docx(md_text, omml_cache, entries, fig_caps, tab_caps, page_map, fig_map, tab_map, front_map)
    convert_docx_to_pdf(OUT_DOCX, OUT_PDF)
    print(
        f"[dut-build] Completed 2-pass DUT build -> {OUT_DOCX} & {OUT_PDF} | "
        f"TOC={len(page_map)}/{len(entries)}, Figs={len(fig_map)}/{len(fig_caps)}, Tabs={len(tab_map)}/{len(tab_caps)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
