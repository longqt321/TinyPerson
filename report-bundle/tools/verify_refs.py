#!/usr/bin/env python3
"""verify_refs.py: đối chiếu DOI trong danh mục tài liệu với Crossref.

Dùng:
    python verify_refs.py draft.md

Với mỗi dòng tài liệu có DOI:
  - KHÔNG TỒN TẠI: Crossref không có DOI này (DOI sai hoặc bịa).
  - NGHI NGỜ: tiêu đề chính thức không khớp với dòng tham chiếu trong bài.
  - OK: khớp.
Tài liệu không có DOI không được kiểm tra; phải kiểm tra thủ công.
Cần kết nối internet.
"""
import re
import sys
import json
import time
import urllib.parse
import urllib.request
import urllib.error

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")
WORD_RE = re.compile(r"\w+", re.UNICODE)
USER_AGENT = "verify_refs/1.0 (mailto:your-email@example.com)"  # đổi thành email của bạn


def fetch_title(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.load(r)
        titles = data.get("message", {}).get("title", [])
        return titles[0] if titles else ""
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def words(s):
    return {w.lower() for w in WORD_RE.findall(s) if len(w) > 2}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as f:
        text = f.read()

    m = re.search(r"^#+\s*(Tài liệu tham khảo|References)\s*$", text, re.IGNORECASE | re.MULTILINE)
    refs = text[m.end():] if m else text
    entries = [l for l in refs.split("\n") if DOI_RE.search(l)]

    if not entries:
        print("Không tìm thấy DOI trong danh mục tài liệu.")
        return

    n_bad = 0
    for line in entries:
        doi = DOI_RE.search(line).group(0).rstrip(".,;:)]")
        try:
            title = fetch_title(doi)
        except Exception as e:  # mạng lỗi, rate limit
            print(f"[LỖI MẠNG] {doi}: {e}")
            time.sleep(1)
            continue
        time.sleep(0.2)

        short = line.strip()[:90]
        if title is None:
            print(f"[KHÔNG TỒN TẠI] {doi}  <- {short}")
            n_bad += 1
            continue
        tw, lw = words(title), words(line)
        overlap = len(tw & lw) / len(tw) if tw else 0
        if overlap < 0.6:
            print(f"[NGHI NGỜ] {doi}: tiêu đề Crossref = '{title[:80]}' | dòng bài = '{short}'")
            n_bad += 1
        else:
            print(f"[OK] {doi}")

    print(f"\nĐã kiểm tra {len(entries)} tài liệu có DOI, {n_bad} cần xem lại.")
    print("Tài liệu không có DOI cần kiểm tra thủ công.")


if __name__ == "__main__":
    main()
