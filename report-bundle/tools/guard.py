#!/usr/bin/env python3
"""guard.py: chặn agent sửa file ngoài phạm vi và phát hiện ảnh/bảng không do pipeline tạo.

Lệnh:
  python tools/guard.py scan       kiểm tra tĩnh (không cần snapshot)
  python tools/guard.py snapshot   ghi trạng thái dự án, chạy trước khi giao việc cho agent
  python tools/guard.py verify     so với snapshot, báo mọi thay đổi ngoài phạm vi, rồi chạy scan

Mã thoát 1 nếu có vi phạm.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
ASSETS = REPORT / "assets"
TABLES = REPORT / "tables"
MANIFEST = ROOT / "figures" / "manifest.json"
STATE = ROOT / ".guard"
SNAPSHOT = STATE / "manifest.json"

FORBIDDEN_EXT = {".docx", ".doc", ".dotx", ".pdf", ".pptx", ".ppt", ".xlsx", ".xls",
                 ".odt", ".rtf", ".tex", ".html", ".htm", ".epub", ".zip", ".exe",
                 ".py", ".sh", ".js", ".ipynb", ".r", ".m", ".bat", ".ps1"}
PROTECTED = {"AGENTS.md", "README.md", "reference.docx",
             "report/metadata.yaml", "report/chapters.txt"}
PROTECTED_PREFIX = ("tools/", "build/", "data/", "figures/", "report/assets/",
                    "report/tables/", "skills/", "references/", "tests/", ".guard/", ".git/")
EXCLUDE_DIRS = {".git", ".guard", "build", "__pycache__"}
IGNORE_NAMES = {".gitkeep", ".DS_Store", "Thumbs.db"}
MAX_IMG_BYTES = 5 * 1024 * 1024
IMG_LINK_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG_EXT = {".png", ".jpg", ".jpeg", ".svg"}

errors, warnings = [], []


def rel(p):
    return p.relative_to(ROOT).as_posix()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    if not MANIFEST.exists():
        return {}
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append("figures/manifest.json sai cú pháp JSON")
        return {}


def allowed_change(rp):
    if rp in PROTECTED or rp.startswith(PROTECTED_PREFIX):
        return False, "file/thư mục được bảo vệ (dữ liệu, hình, bảng, công cụ do người dùng quản lý)"
    p = Path(rp)
    if p.parent.as_posix() == "report" and p.suffix.lower() == ".md":
        return True, ""
    if p.parent.as_posix() == "claims" and p.suffix.lower() == ".csv":
        return True, ""
    return False, "ngoài phạm vi: agent chỉ được sửa report/*.md và claims/*.csv"


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if set(p.relative_to(ROOT).parts) & EXCLUDE_DIRS:
            continue
        if p.name in IGNORE_NAMES:
            continue
        yield p


def cmd_snapshot():
    data = {rel(p): sha256(p) for p in iter_files()}
    STATE.mkdir(exist_ok=True)
    SNAPSHOT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi snapshot: {len(data)} file -> {rel(SNAPSHOT)}")
    return 0


def cmd_verify():
    if not SNAPSHOT.exists():
        print("Chưa có snapshot. Chạy: python tools/guard.py snapshot")
        return 1
    old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    new = {rel(p): sha256(p) for p in iter_files()}
    changes = []
    for k in sorted(set(old) | set(new)):
        if k not in old:
            changes.append(("THÊM", k))
        elif k not in new:
            changes.append(("XÓA", k))
        elif old[k] != new[k]:
            changes.append(("SỬA", k))
    if not changes:
        print("Không có thay đổi nào so với snapshot.")
    else:
        print("Các thay đổi so với snapshot:")
    for kind, path in changes:
        ok, reason = allowed_change(path)
        print(f"  [{kind}] {path} -> {'OK' if ok else 'VI PHẠM: ' + reason}")
        if not ok:
            errors.append(f"Thay đổi ngoài phạm vi: {path} ({kind.lower()}): {reason}")
    return scan()


def scan():
    # 1. File định dạng cấm trong report/ và các thư mục sinh nội dung
    for folder in (REPORT, ASSETS, TABLES):
        if folder.exists():
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in FORBIDDEN_EXT:
                    errors.append(f"File định dạng cấm: {rel(p)}")

    # 2. Mọi file trong assets/ và tables/ phải do make_figures.py tạo, và chưa bị sửa tay
    man = load_manifest()
    by_path = {e.get("output_path"): e.get("output_sha256") for e in man.values()}
    for folder in (ASSETS, TABLES):
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if not p.is_file() or p.name in IGNORE_NAMES or p.name == ".gitkeep":
                continue
            r = rel(p)
            if r not in by_path:
                errors.append(f"{r}: không do make_figures.py tạo (không có trong figures/manifest.json)")
            elif sha256(p) != by_path[r]:
                errors.append(f"{r}: đã bị sửa tay so với lần sinh gần nhất")

    # 3. Liên kết ảnh trong các chương
    referenced = set()
    for md in sorted(REPORT.glob("*.md")):
        text = re.sub(r"```.*?```", "", md.read_text(encoding="utf-8"), flags=re.S)
        for m in IMG_LINK_RE.finditer(text):
            alt, target = m.group(1).strip(), m.group(2)
            where = md.name
            if target.startswith(("http://", "https://", "data:")):
                errors.append(f"{where}: ảnh dùng đường dẫn ngoài ({target}). Ảnh phải do pipeline tạo")
                continue
            if not alt:
                errors.append(f"{where}: ảnh '{target}' thiếu mô tả (alt text)")
            target_path = (md.parent / target).resolve()
            try:
                referenced.add(rel(target_path))
            except ValueError:
                errors.append(f"{where}: ảnh '{target}' trỏ ra ngoài dự án")
                continue
            if not target_path.exists():
                errors.append(f"{where}: ảnh không tồn tại: {target}")
                continue
            if target_path.suffix.lower() not in IMG_EXT:
                errors.append(f"{where}: định dạng ảnh không cho phép: {target}")
            if target_path.stat().st_size > MAX_IMG_BYTES:
                warnings.append(f"{where}: ảnh lớn hơn 5 MB: {target}")

    for a in sorted(ASSETS.glob("*")) if ASSETS.exists() else []:
        if a.is_file() and a.name not in IGNORE_NAMES and rel(a) not in referenced:
            warnings.append(f"Ảnh chưa được dùng trong chương nào: {rel(a)}")

    # 4. Danh sách chương
    chapters = REPORT / "chapters.txt"
    if chapters.exists():
        for line in chapters.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name and not name.startswith("#") and not (REPORT / name).exists():
                errors.append(f"chapters.txt liệt kê file không tồn tại: {name}")

    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    print(f"\nGuard: {len(errors)} ERROR, {len(warnings)} WARN")
    return 1 if errors else 0


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"scan", "snapshot", "verify"}:
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd == "snapshot":
        return cmd_snapshot()
    if cmd == "verify":
        return cmd_verify()
    return scan()


if __name__ == "__main__":
    sys.exit(main())
