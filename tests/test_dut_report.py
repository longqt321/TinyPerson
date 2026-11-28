from pathlib import Path

import pytest
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

from scripts.build_dut_report import (
    build_toc,
    configure_heading_styles,
    normalize_docx_zip,
    parse_md,
    validate_heading_numbering,
)

ROOT = Path(__file__).resolve().parents[1]


def test_current_report_heading_numbering_is_contiguous() -> None:
    markdown = (ROOT / "docs/report/course_report_vi.md").read_text(encoding="utf-8")
    validate_heading_numbering(markdown)


def test_heading_numbering_rejects_gaps() -> None:
    markdown = "## 1. Chapter\n\n### 1.1. First\n\n### 1.3. Skipped"
    with pytest.raises(ValueError, match="expected 1.2, found 1.3"):
        validate_heading_numbering(markdown)


def test_parser_merges_soft_wrapped_paragraph_lines() -> None:
    blocks = parse_md("First physical line\ncontinues here\n\nSecond paragraph")
    assert blocks == [
        ("p", "First physical line continues here"),
        ("p", "Second paragraph"),
    ]


def test_parser_recognizes_numbered_lists() -> None:
    assert parse_md("1. First item\n2. Second item") == [
        ("item", "1. First item"),
        ("item", "2. Second item"),
    ]


def test_parser_preserves_pipes_inside_escaped_text_and_inline_code() -> None:
    markdown = "| Left | Right |\n| --- | --- |\n| a\\|b | `x|y` |"
    blocks = parse_md(markdown)
    assert blocks == [("table", None, ["Left", "Right"], [["a|b", "`x|y`"]])]


def test_parser_rejects_invalid_table_separator() -> None:
    markdown = "| A | B |\n| --- | nope |\n| x | y |"
    with pytest.raises(ValueError, match="Invalid Markdown table separator"):
        parse_md(markdown)


def test_parser_rejects_table_separator_column_mismatch() -> None:
    markdown = "| A | B |\n| --- | --- | --- |\n| x | y |"
    with pytest.raises(ValueError, match="Invalid Markdown table separator"):
        parse_md(markdown)

def test_parser_rejects_unclosed_code_blocks() -> None:
    with pytest.raises(ValueError, match="Unclosed fenced code block"):
        parse_md("```python\nprint('incomplete')")


def test_heading_styles_keep_existing_report_typography() -> None:
    document = Document()
    configure_heading_styles(document)
    heading = document.styles["Heading 1"]
    assert heading.font.name == "Times New Roman"
    assert heading.font.size.pt == 14
    assert heading.paragraph_format.alignment == WD_ALIGN_PARAGRAPH.CENTER


def test_toc_entries_are_internal_hyperlinks() -> None:
    document = Document()
    entries = [(2, "Chương 1: Overview"), (3, "1.1. Details")]
    build_toc(document, entries, {"Chương 1: Overview": 1, "1.1. Details": 2})
    links = document.element.body.xpath(".//w:hyperlink")
    anchors = [link.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}anchor") for link in links]
    assert "heading_0001" in anchors
    assert "heading_0002" in anchors
    assert "frontmatter_1" in anchors


def test_docx_zip_normalization_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "report.docx"
    Document().save(path)
    normalize_docx_zip(path)
    first = path.read_bytes()
    normalize_docx_zip(path)
    assert path.read_bytes() == first
