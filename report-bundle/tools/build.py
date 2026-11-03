#!/usr/bin/env python3
"""build.py: ghép chương, chèn bảng, ghép danh mục tài liệu, kiểm tra trích dẫn, xuất DOCX/PDF.

CHỈ NGƯỜI DÙNG CHẠY. Agent không được chạy file này (xem AGENTS.md).

Dùng:
  python tools/build.py docx
  python tools/build.py pdf
  python tools/build.py docx --numbered        đánh số mục tự động
  python tools/build.py docx --skip-checks     bỏ qua mọi kiểm tra

Thứ tự:
  1. md_lint, guard scan, make_figures --check
  2. ghép chương (report/chapters.txt), chèn bảng ({{TABLE ...}}), nối references/danh-muc.md
  3. check_manuscript trên bản ghép (từ cấm, claim thiếu nguồn, thứ tự [n], placeholder)
  4. pandoc xuất DOCX hoặc PDF

Yêu cầu: pandoc (và xelatex nếu xuất PDF). Xem README.md.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
BUILD = ROOT / "build"
META = REPORT / "metadata.yaml"
REFERENCE_DOCX = ROOT / "reference.docx"
REFS = ROOT / "references" / "danh-muc.md"
TOOLS = ROOT / "tools"
SPECS = ROOT / "figures" / "specs.json"
MARKER_RE = re.compile(r"^\{\{TABLE\s+(tab-\d+-\d+)\}\}\s*$")

PAGE_BREAK = {
    "docx": "```{=openxml}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```",
    "pdf": "```{=latex}\n\\newpage\n```",
}


def run(cmd, cwd=None):
    print("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, cwd=cwd).returncode


def load_chapters():
    lst = REPORT / "chapters.txt"
    if not lst.exists():
        sys.exit("Thiếu report/chapters.txt")
    names = [l.strip() for l in lst.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    paths = []
    for n in names:
        p = REPORT / n
        if not p.exists():
            sys.exit(f"Không tìm thấy chương: {n}")
        paths.append(p)
    return paths


def load_spec_outputs():
    if not SPECS.exists():
        return {}
    return {s["id"]: s.get("output", "") for s in json.loads(SPECS.read_text(encoding="utf-8"))}


def expand_tables(text, outputs):
    out_lines = []
    for line in text.split("\n"):
        m = MARKER_RE.match(line.strip())
        if m:
            sid = m.group(1)
            name = outputs.get(sid)
            path = REPORT / "tables" / name if name else None
            if not path or not path.exists():
                sys.exit(f"Chưa có bảng {sid}. Chạy: python tools/make_figures.py")
            out_lines.append(path.read_text(encoding="utf-8").rstrip())
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


def assemble(chapters, fmt, outputs):
    parts = []
    for i, ch in enumerate(chapters):
        if i > 0:
            parts.append(PAGE_BREAK[fmt])
        parts.append(expand_tables(ch.read_text(encoding="utf-8").rstrip(), outputs) + "\n")
    if REFS.exists():
        parts.append(PAGE_BREAK[fmt])
        parts.append(REFS.read_text(encoding="utf-8").rstrip() + "\n")
    else:
        print("Cảnh báo: chưa có references/danh-muc.md (danh mục tài liệu tham khảo).")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fmt", choices=["docx", "pdf"])
    ap.add_argument("--numbered", action="store_true", help="đánh số mục tự động")
    ap.add_argument("--skip-checks", action="store_true")
    args = ap.parse_args()

    if shutil.which("pandoc") is None:
        sys.exit("Chưa cài pandoc. Xem README.md, mục Cài đặt.")
    if args.fmt == "pdf" and shutil.which("xelatex") is None:
        sys.exit("Chưa có xelatex (cần cho PDF). Xem README.md, mục Cài đặt.")

    if not args.skip_checks:
        for cmd in ([sys.executable, str(TOOLS / "md_lint.py")],
                    [sys.executable, str(TOOLS / "guard.py"), "scan"],
                    [sys.executable, str(TOOLS / "make_figures.py"), "--check"]):
            if run(cmd, cwd=ROOT) != 0:
                sys.exit(f"{Path(cmd[1]).name} báo lỗi. Sửa lỗi trước khi xuất, hoặc dùng --skip-checks.")

    chapters = load_chapters()
    outputs = load_spec_outputs()
    BUILD.mkdir(exist_ok=True)
    merged = BUILD / "report.md"
    merged.write_text(assemble(chapters, args.fmt, outputs), encoding="utf-8")

    if not args.skip_checks:
        if run([sys.executable, str(TOOLS / "check_manuscript.py"), str(merged)], cwd=ROOT) != 0:
            sys.exit("check_manuscript báo ERROR trên bản ghép. Sửa trong chương hoặc references/danh-muc.md.")

    out = BUILD / f"report.{args.fmt}"
    cmd = [
        "pandoc", str(merged), "-o", str(out),
        "--from", "markdown+raw_attribute",
        "--resource-path", str(REPORT),
        "--toc", "--toc-depth=3",
    ]
    if META.exists():
        cmd += ["--metadata-file", str(META)]
    if args.numbered:
        cmd += ["--number-sections"]
    if args.fmt == "docx" and REFERENCE_DOCX.exists():
        cmd += ["--reference-doc", str(REFERENCE_DOCX)]
    if args.fmt == "pdf":
        cmd += ["--pdf-engine=xelatex"]

    if run(cmd, cwd=REPORT) != 0:
        sys.exit("pandoc gặp lỗi. Xem thông báo phía trên.")
    print(f"\nĐã tạo: {out}")
    print("Nên mở file và kiểm tra mục lục, hình, bảng, ngắt trang trước khi nộp.")


if __name__ == "__main__":
    main()
