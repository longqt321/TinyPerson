#!/usr/bin/env python3
"""md_lint.py: kiểm tra cấu trúc chương và sự khớp với figures/specs.json.

Kiểm tra:
  - Mỗi chương có đúng một tiêu đề cấp 1; không nhảy cấp tiêu đề.
  - Mọi ảnh phải có mục fig-x-y tương ứng trong specs.json, đúng đường dẫn assets/<output>.
  - Caption hình đặt ngay sau ảnh; caption bảng đặt ngay trước marker {{TABLE tab-x-y}}.
  - Không được viết bảng bằng tay (dòng bắt đầu bằng "|").
  - Số hình, bảng đánh liên tục trong chương và khớp với mã mục.
  - Mọi tham chiếu "Hình x.y" / "Bảng x.y" đều có caption tương ứng.
  - Mục trong specs.json không dùng trong báo cáo: cảnh báo.
  - Gạch đầu dòng, in đậm quá mức; placeholder còn sót.

Dùng: python tools/md_lint.py
Mã thoát 1 nếu có ERROR.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
SPECS = ROOT / "figures" / "specs.json"

IMG_RE = re.compile(r"^!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)\s*$")
CAP_FIG_RE = re.compile(r"^\*\*Hình\s+(\d+)\.(\d+)\.?\*\*\s*(.*)$")
CAP_TAB_RE = re.compile(r"^\*\*Bảng\s+(\d+)\.(\d+)\.?\*\*\s*(.*)$")
MARKER_RE = re.compile(r"^\{\{TABLE\s+(tab-\d+-\d+)\}\}$")
REF_RE = re.compile(r"\b(Hình|Bảng)\s+(\d+)\.(\d+)\b")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
CHAPTER_RE = re.compile(r"Chương\s+(\d+)", re.IGNORECASE)
LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")
PLACEHOLDER_RE = re.compile(r"^>\s*\[(HÌNH|BẢNG|CẦN)[^\]]*\]")

errors, warnings = [], []


def add(level, fname, line, msg):
    loc = f"{fname}:{line}" if line else fname
    (errors if level == "ERROR" else warnings).append(f"[{level}] {loc}: {msg}")


def load_specs():
    if not SPECS.exists():
        return {}
    try:
        items = json.loads(SPECS.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        add("ERROR", "figures/specs.json", None, f"sai cú pháp JSON: {e}")
        return {}
    return {s.get("id"): s for s in items if isinstance(s, dict)}


def load_chapters():
    chap_file = REPORT / "chapters.txt"
    if chap_file.exists():
        names = [l.strip() for l in chap_file.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
    else:
        names = sorted(p.name for p in REPORT.glob("*.md"))
    return [REPORT / n for n in names if (REPORT / n).exists()]


def lint_file(path, specs, used, all_caps, all_refs):
    fname = path.name
    raw_lines = path.read_text(encoding="utf-8").split("\n")

    in_code = False
    clean = []
    for i, raw in enumerate(raw_lines, start=1):
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            clean.append((i, raw.rstrip()))
    n = len(clean)

    def prev_nb(idx):
        j = idx - 1
        while j >= 0 and not clean[j][1].strip():
            j -= 1
        return j if j >= 0 else None

    def next_nb(idx):
        j = idx + 1
        while j < n and not clean[j][1].strip():
            j += 1
        return j if j < n else None

    # Tiêu đề
    h1 = [(i, t) for i, t in clean if (m := HEADING_RE.match(t)) and len(m.group(1)) == 1]
    if not h1:
        add("ERROR", fname, None, "Thiếu tiêu đề cấp 1 (# Chương N. Tên)")
    elif len(h1) > 1:
        add("ERROR", fname, h1[1][0], "Có nhiều hơn một tiêu đề cấp 1 trong một chương")
    chapter_no = None
    if h1:
        cm = CHAPTER_RE.search(h1[0][1])
        if cm:
            chapter_no = int(cm.group(1))
        else:
            add("WARN", fname, h1[0][0], "Tiêu đề chương nên có dạng 'Chương N. Tên'")

    prev_level = 0
    for i, t in clean:
        hm = HEADING_RE.match(t)
        if hm:
            level = len(hm.group(1))
            if prev_level and level > prev_level + 1:
                add("WARN", fname, i, f"Nhảy cấp tiêu đề từ H{prev_level} lên H{level}")
            prev_level = level

    fig_seq, tab_seq = [], []
    body_words = list_lines = content_lines = bold_count = 0
    in_table_block = False

    for idx, (i, t) in enumerate(clean):
        s = t.strip()
        if not s:
            in_table_block = False
            continue

        # Placeholder còn sót
        if PLACEHOLDER_RE.match(s):
            add("WARN", fname, i, f"Còn placeholder cần bổ sung: {s[:70]}")
            continue

        # Ảnh
        im = IMG_RE.match(s)
        if im:
            nxt = next_nb(idx)
            cm = CAP_FIG_RE.match(clean[nxt][1].strip()) if nxt is not None else None
            if not cm:
                add("ERROR", fname, i, "Ảnh không có caption '**Hình x.y.** ...' ngay sau")
                continue
            a, b = int(cm.group(1)), int(cm.group(2))
            fig_seq.append((i, a, b))
            all_caps.add(f"Hình {a}.{b}")
            sid = f"fig-{a}-{b}"
            spec = specs.get(sid)
            if not spec:
                add("ERROR", fname, i, f"Hình {a}.{b} không có mục '{sid}' trong figures/specs.json. Ảnh phải do pipeline tạo")
            else:
                used.add(sid)
                expected = f"assets/{spec.get('output', '')}"
                if im.group(2) != expected:
                    add("ERROR", fname, i, f"đường dẫn ảnh phải là '{expected}', hiện là '{im.group(2)}'")
            if not im.group(1).strip():
                add("ERROR", fname, i, "Ảnh thiếu mô tả (alt text)")
            continue

        # Caption hình: phải đi kèm ảnh ngay phía trên
        cf = CAP_FIG_RE.match(s)
        if cf:
            pv = prev_nb(idx)
            if pv is None or not IMG_RE.match(clean[pv][1].strip()):
                add("ERROR", fname, i, "Caption hình không đi kèm ảnh ngay phía trên")
            all_caps.add(f"Hình {cf.group(1)}.{cf.group(2)}")
            if len(cf.group(3).strip()) < 5:
                add("WARN", fname, i, "Caption hình quá ngắn, cần mô tả rõ hơn")
            continue

        # Caption bảng: phải đi kèm marker ngay phía dưới
        ct = CAP_TAB_RE.match(s)
        if ct:
            all_caps.add(f"Bảng {ct.group(1)}.{ct.group(2)}")
            nxt = next_nb(idx)
            if nxt is None or not MARKER_RE.match(clean[nxt][1].strip()):
                add("ERROR", fname, i, "Caption bảng phải đi ngay trước marker {{TABLE tab-x-y}}")
            continue

        # Marker bảng
        mm = MARKER_RE.match(s)
        if mm:
            sid = mm.group(1)
            a_b = sid.split("-")[1:]
            pv = prev_nb(idx)
            cap = CAP_TAB_RE.match(clean[pv][1].strip()) if pv is not None else None
            if not cap:
                add("ERROR", fname, i, f"Marker {sid} thiếu caption '**Bảng x.y.**' ngay phía trên")
            else:
                if (cap.group(1), cap.group(2)) != tuple(a_b):
                    add("ERROR", fname, i, f"Caption bảng {cap.group(1)}.{cap.group(2)} không khớp với mã {sid}")
                tab_seq.append((i, int(cap.group(1)), int(cap.group(2))))
            spec = specs.get(sid)
            if not spec:
                add("ERROR", fname, i, f"Marker '{sid}' không có trong figures/specs.json")
            elif spec.get("type") != "table":
                add("ERROR", fname, i, f"Mục '{sid}' không phải loại table")
            else:
                used.add(sid)
            continue

        # Bảng viết tay
        if s.startswith("|"):
            if not in_table_block:
                add("ERROR", fname, i, "Bảng viết tay bị cấm. Bảng phải sinh bằng make_figures.py và chèn bằng {{TABLE tab-x-y}}")
            in_table_block = True
            continue
        in_table_block = False

        # Nội dung văn bản
        if not HEADING_RE.match(s):
            content_lines += 1
            if LIST_RE.match(s):
                list_lines += 1
            body_words += len(s.split())
            bold_count += len(BOLD_RE.findall(s))
            if "!" in s:
                add("WARN", fname, i, "Có dấu '!' trong văn bản học thuật")
        for rm in REF_RE.finditer(s):
            all_refs.append((fname, i, f"{rm.group(1)} {rm.group(2)}.{rm.group(3)}"))

    if chapter_no is not None:
        for kind, seq in (("Hình", fig_seq), ("Bảng", tab_seq)):
            expected = 1
            for line, a, b in seq:
                if a != chapter_no:
                    add("ERROR", fname, line, f"{kind} {a}.{b} phải có số chương là {chapter_no}")
                elif b != expected:
                    add("ERROR", fname, line, f"{kind} {a}.{b} đánh số không liên tục (kỳ vọng {chapter_no}.{expected})")
                expected = b + 1

    if content_lines >= 6 and list_lines / content_lines > 0.4:
        add("WARN", fname, None, "Hơn 40% dòng là gạch đầu dòng; nên viết thành đoạn văn")
    if body_words > 0 and bold_count / body_words * 300 > 3:
        add("WARN", fname, None, "In đậm quá nhiều; văn bản học thuật chỉ nên in đậm rất hạn chế")


def main():
    specs = load_specs()
    chapters = load_chapters()
    if not chapters:
        print("Không tìm thấy chương nào trong report/.")
        return 1

    used, all_caps, all_refs = set(), set(), []
    for ch in chapters:
        lint_file(ch, specs, used, all_caps, all_refs)

    for fname, line, ref in all_refs:
        if ref not in all_caps:
            add("ERROR", fname, line, f"Tham chiếu '{ref}' nhưng không có caption tương ứng trong báo cáo")

    for sid in sorted(set(specs) - used):
        add("WARN", "figures/specs.json", None, f"Mục '{sid}' không được dùng trong báo cáo")

    for w in warnings:
        print(w)
    for e in errors:
        print(e)
    print(f"\nmd_lint: {len(errors)} ERROR, {len(warnings)} WARN trên {len(chapters)} chương")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
