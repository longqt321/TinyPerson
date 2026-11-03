#!/usr/bin/env python3
"""check_manuscript.py: kiểm tra tự động bản thảo học thuật.

Dùng:
    python check_manuscript.py draft.md
    python check_manuscript.py draft.md --json report.json
    python check_manuscript.py draft.docx      (cần: pip install python-docx)

Mã thoát: 0 nếu không có ERROR, 1 nếu có ERROR.
Đây là kiểm tra heuristic. WARN cần con người xem lại; không phải lỗi chắc chắn.
"""
import re
import sys
import json
import argparse

BANNED_WORDS = [
    "đột phá", "vượt trội", "cách mạng", "hoàn hảo", "tối ưu tuyệt đối",
    "đi sâu", "then chốt", "bức tranh toàn cảnh", "trong thời đại số",
    "kỷ nguyên", "không thể phủ nhận", "tuyệt vời", "xuất sắc", "vô cùng",
    "chúng ta", "bạn có thể", "rất nhiều", "là một bước tiến quan trọng",
    "mang tính cách mạng", "hiển nhiên",
]

ABSOLUTE_WORDS = [
    "luôn luôn", "luôn", "không bao giờ", "chắc chắn", "tuyệt đối",
    "hoàn toàn", "duy nhất", "tốt nhất", "không thể", "chứng minh",
    "mọi", "tất cả", "chưa từng",
]

EVIDENCE_HINTS = [
    "nghiên cứu cho thấy", "các nghiên cứu", "nhiều nghiên cứu",
    "theo thống kê", "theo báo cáo", "đã được chứng minh", "được chứng minh",
]

ACRONYM_STOPLIST = {"IEEE", "DOI", "URL", "PDF", "DOCX", "TODO", "CHECK", "REMOVE", "CẦN", "SUNG", "HÌNH", "BẢNG", "CIT"}

PLACEHOLDER_RE = re.compile(r"\[(CẦN[^\]]*|CHECK[^\]]*|TODO[^\]]*|REMOVE[^\]]*|HÌNH CẦN[^\]]*|BẢNG CẦN[^\]]*|CIT:[^\]]*)\]")
CITE_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
INTERNAL_REF_RE = re.compile(r"\b(Hình|Bảng|Mục|Phương trình|Chương|Phụ lục|xem|Xem)\b")
NUMBER_UNIT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(%|ms|s|GB|MB|FPS|KB)\b")
REF_HEADING_RE = re.compile(r"^#+\s*(Tài liệu tham khảo|References)\s*$", re.IGNORECASE | re.MULTILINE)
CAPTION_RE = re.compile(r"^\**\s*(Hình|Bảng)\s+(\d+)\s*[.:]")
FIGREF_RE = re.compile(r"\b(Hình|Bảng)\s+(\d+)\b")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_text(path):
    if path.lower().endswith(".docx"):
        try:
            import docx  # python-docx
        except ImportError:
            sys.exit("Cần cài python-docx: pip install python-docx")
        d = docx.Document(path)
        lines = []
        for p in d.paragraphs:
            t = p.text.strip()
            style = (p.style.name or "").lower()
            if not t:
                lines.append("")
            elif style.startswith("heading"):
                nums = re.findall(r"\d+", style)
                lines.append("#" * (int(nums[0]) if nums else 1) + " " + t)
            else:
                lines.append(t)
        return "\n".join(lines)
    with open(path, encoding="utf-8") as f:
        return f.read()


def expand_cite(group):
    nums = []
    for part in re.split(r"\s*,\s*", group.strip()):
        part = part.strip()
        rng = re.split(r"\s*[–-]\s*", part)
        if len(rng) == 2 and rng[0].isdigit() and rng[1].isdigit():
            nums.extend(range(int(rng[0]), int(rng[1]) + 1))
        elif part.isdigit():
            nums.append(int(part))
    return nums


def has_word(text_lower, word):
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text_lower) is not None


def check(text):
    issues = []

    def add(sev, code, line, msg):
        issues.append({"severity": sev, "code": code, "line": line, "message": msg})

    # Tách phần tham chiếu khỏi thân bài
    m = REF_HEADING_RE.search(text)
    if m:
        body_text, refs_text = text[:m.start()], text[m.end():]
        refs_offset = text[:m.end()].count("\n") + 1
    else:
        body_text, refs_text, refs_offset = text, "", 0

    lines = body_text.split("\n")
    in_code = False
    cites_in_order = []   # (num, line)
    body_plain_parts = []
    caption_lines = set()

    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line:
            continue
        if CAPTION_RE.match(line.replace("*", "")):
            caption_lines.add(i)
        if line.startswith("#") or line.startswith("|"):
            body_plain_parts.append(line)
            for mm in PLACEHOLDER_RE.finditer(line):
                add("ERROR", "PLACEHOLDER", i, f"Còn placeholder '{mm.group(0)[:60]}' trong bản nộp")
            continue

        body_plain_parts.append(line)

        for mm in PLACEHOLDER_RE.finditer(line):
            add("ERROR", "PLACEHOLDER", i, f"Còn placeholder '{mm.group(0)[:60]}' trong bản nộp")

        if "!" in line:
            add("WARN", "EXCLAMATION", i, "Có dấu '!' trong văn bản học thuật")

        low = line.lower()
        for w in BANNED_WORDS:
            if has_word(low, w):
                add("WARN", "BANNED_WORD", i, f"Từ/cụm từ cần thay: '{w}'")

        for mm in CITE_RE.finditer(line):
            for n in expand_cite(mm.group(1)):
                cites_in_order.append((n, i))

        for s in SENT_SPLIT.split(line):
            s_low = s.lower()
            has_cite = CITE_RE.search(s) is not None
            has_internal = INTERNAL_REF_RE.search(s) is not None

            hits = [w for w in ABSOLUTE_WORDS if has_word(s_low, w)]
            if hits and not has_cite and not has_internal:
                add("WARN", "ABSOLUTE_NO_CITE", i,
                    f"Khẳng định tuyệt đối ({', '.join(hits)}) không có [n] hoặc tham chiếu nội bộ: '{s[:80]}...'")

            if any(h in s_low for h in EVIDENCE_HINTS) and not has_cite:
                add("ERROR", "EVIDENCE_NO_CITE", i,
                    f"Câu nêu bằng chứng chung nhưng không có [n]: '{s[:80]}...'")

            if NUMBER_UNIT_RE.search(s) and not has_cite and not has_internal:
                add("WARN", "NUMBER_NO_SOURCE", i,
                    f"Số liệu không có [n] hoặc tham chiếu (kiểm tra là kết quả tự đo): '{s[:80]}...'")

    body_plain = "\n".join(body_plain_parts)

    # Thứ tự đánh số [n] theo lần xuất hiện đầu
    seen = []
    expected = 1
    for n, ln in cites_in_order:
        if n in seen:
            continue
        if n != expected:
            add("ERROR", "CITE_ORDER", ln,
                f"Trích [{n}] lần đầu xuất hiện trước [{expected}]. IEEE đánh số theo thứ tự xuất hiện.")
        seen.append(n)
        expected = max(seen) + 1

    # Đối chiếu với danh mục tài liệu
    ref_entries = {}
    if refs_text.strip():
        for j, rl in enumerate(refs_text.split("\n"), start=refs_offset + 1):
            rm = re.match(r"^\s*\[(\d+)\]", rl)
            if rm:
                ref_entries[int(rm.group(1))] = j
        ref_keys_sorted = sorted(ref_entries)
        if ref_keys_sorted != list(range(1, len(ref_keys_sorted) + 1)):
            add("WARN", "REF_NUMBERING", None, "Danh mục tài liệu không đánh số liên tục từ [1]")
    cited_set = set(seen)
    if cited_set and not ref_entries:
        add("ERROR", "NO_REF_SECTION", None, "Có trích dẫn nhưng không tìm thấy mục 'Tài liệu tham khảo'")
    for n in sorted(cited_set - set(ref_entries)):
        add("ERROR", "CITE_MISSING_REF", None, f"[{n}] được trích nhưng không có trong danh mục")
    for n in sorted(set(ref_entries) - cited_set):
        add("WARN", "REF_UNCITED", ref_entries[n], f"Tài liệu [{n}] có trong danh mục nhưng không được trích")

    # Hình, bảng: caption và tham chiếu
    captions = {}
    for i, raw in enumerate(lines, start=1):
        cm = CAPTION_RE.match(raw.strip().replace("*", ""))
        if cm:
            captions[(cm.group(1), int(cm.group(2)))] = i
    referenced = {}
    for i, raw in enumerate(lines, start=1):
        if i in caption_lines:
            continue
        for fm in FIGREF_RE.finditer(raw):
            referenced.setdefault((fm.group(1), int(fm.group(2))), i)
    for key, ln in referenced.items():
        if key not in captions:
            add("ERROR", "FIG_REF_MISSING", ln, f"Tham chiếu {key[0]} {key[1]} nhưng không có caption tương ứng")
    for key, ln in captions.items():
        if key not in referenced:
            add("WARN", "FIG_UNREFERENCED", ln, f"{key[0]} {key[1]} có caption nhưng không được tham chiếu trong bài")

    # Viết tắt: kiểm tra có định nghĩa ở lần đầu không
    acr_seen = []
    for tok in re.findall(r"(?<![\w\-])([A-Z][A-Z0-9]{1,6})(?!\w)", body_plain):
        if tok not in acr_seen and tok not in ACRONYM_STOPLIST:
            acr_seen.append(tok)
    for acr in acr_seen:
        defined = (re.search(r"\(\s*[^()]*?\b" + acr + r"\b[^()]*?\)", body_plain) or
                   re.search(r"\b" + acr + r"\s*\(", body_plain) or
                   re.search(r"\b" + acr + r"\)", body_plain))
        if not defined:
            add("WARN", "ACRONYM_UNDEFINED", None,
                f"Viết tắt '{acr}' có thể chưa được định nghĩa ở lần xuất hiện đầu (kiểm tra thủ công)")

    return issues


def main():
    ap = argparse.ArgumentParser(description="Kiểm tra tự động bản thảo học thuật.")
    ap.add_argument("path")
    ap.add_argument("--json", help="Ghi báo cáo ra file JSON")
    args = ap.parse_args()

    text = load_text(args.path)
    issues = check(text)
    issues.sort(key=lambda x: (x["line"] is None, x["line"] or 0, x["severity"]))

    for it in issues:
        loc = f"dòng {it['line']}" if it["line"] else "toàn bài"
        print(f"[{it['severity']}] {loc}: {it['code']} - {it['message']}")

    n_err = sum(1 for x in issues if x["severity"] == "ERROR")
    n_warn = sum(1 for x in issues if x["severity"] == "WARN")
    print(f"\nTổng: {n_err} ERROR, {n_warn} WARN")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(issues, f, ensure_ascii=False, indent=2)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
