#!/usr/bin/env python3
"""make_figures.py: sinh hình và bảng số liệu từ dữ liệu thật trong data/.

CHỈ NGƯỜI DÙNG CHẠY. Agent không được chạy hoặc sửa file này (xem AGENTS.md).

Lệnh:
  python tools/make_figures.py                 sinh tất cả mục trong figures/specs.json
  python tools/make_figures.py --only fig-2-1  chỉ sinh một mục
  python tools/make_figures.py --check         chỉ kiểm tra có cần sinh lại không; không ghi file

Quy tắc:
  - Mỗi file dữ liệu phải có file metadata cùng tên + ".meta.json", gồm nguon, nguoi_tao, ngay_tao, lenh_sinh.
  - Dữ liệu hoặc metadata có dấu hiệu giả lập (mock, demo, fake, synthetic, giả lập, ...) bị từ chối.
  - Hình và bảng chỉ được tạo từ data/ hoặc figures/manual/. Không có nguồn nào khác.
  - Mọi mục đều được kiểm tra trước. Nếu một mục lỗi, không ghi file nào.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "figures" / "specs.json"
MANIFEST = ROOT / "figures" / "manifest.json"
DATA = ROOT / "data"
MANUAL = ROOT / "figures" / "manual"
ASSETS = ROOT / "report" / "assets"
TABLES = ROOT / "report" / "tables"

VERSION = "1.0"
STYLE = {
    "width_in": 6.3,           # khổ ngang vừa trang A4 có lề
    "height_in": 3.8,
    "dpi": 300,
    "font_family": "DejaVu Sans",   # có đủ dấu tiếng Việt
    "font_size": 10,
    "palette": ["#1F4E79", "#C55A11", "#548235", "#7F6000", "#7030A0", "#2E75B6"],
    "decimals": 2,
}
CHART_TYPES = {"bar", "grouped_bar", "line", "scatter"}
MANUAL_EXT = {".png", ".jpg", ".jpeg", ".svg"}
REQUIRED_META = ["nguon", "nguoi_tao", "ngay_tao", "lenh_sinh"]
FORBIDDEN_TOKENS = ["mock", "demo", "fake", "synthetic", "dummy", "placeholder",
                    "giả lập", "dữ liệu giả", "dữ liệu mẫu", "test data", "lorem"]
ID_RE = re.compile(r"^(fig|tab)-(\d+)-(\d+)$")


def rel(p):
    return p.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def fmt_num(v, dec):
    return f"{v:.{dec}f}".replace(".", ",")  # quy ước thập phân kiểu Việt Nam


def load_specs():
    if not SPECS.exists():
        sys.exit("Thiếu figures/specs.json")
    try:
        specs = json.loads(SPECS.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"figures/specs.json sai cú pháp JSON: {e}")
    if not isinstance(specs, list):
        sys.exit("figures/specs.json phải là một danh sách (list)")
    return specs


def load_manifest():
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def validate_meta(data_path):
    errs = []
    meta_path = data_path.with_name(data_path.name + ".meta.json")
    if not meta_path.exists():
        return [f"thiếu file metadata {rel(meta_path)}"], None, b""
    raw = meta_path.read_bytes()
    try:
        meta = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return [f"{rel(meta_path)} không phải JSON hợp lệ: {e}"], None, raw
    for k in REQUIRED_META:
        if not str(meta.get(k, "")).strip():
            errs.append(f"{rel(meta_path)} thiếu trường '{k}'")
    blob = (" ".join(str(v) for v in meta.values()) + " " + data_path.name).lower()
    for tok in FORBIDDEN_TOKENS:
        if re.search(r"(?<!\w)" + re.escape(tok) + r"(?!\w)", blob):
            errs.append(f"{rel(data_path)}: metadata hoặc tên file chứa '{tok}'. Từ chối: không dùng dữ liệu giả lập")
    return errs, meta, raw


def load_csv(data_path):
    with open(data_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = [{k: (v or "").strip() for k, v in r.items() if k is not None} for r in reader]
    return cols, rows


def check_numeric(rows, cols):
    errs = []
    for c in cols:
        for i, r in enumerate(rows, start=2):
            try:
                v = float(r[c])
            except (ValueError, KeyError):
                errs.append(f"cột '{c}' dòng {i}: '{r.get(c, '')}' không phải số (dùng dấu chấm thập phân, ví dụ 0.87)")
                break
            if not math.isfinite(v):
                errs.append(f"cột '{c}' dòng {i}: giá trị không hữu hạn")
                break
    return errs


def prepare(spec):
    """Kiểm tra một mục. Trả về (danh sách lỗi, job)."""
    errs = []
    sid = str(spec.get("id", ""))
    m = ID_RE.match(sid)
    if not m:
        return [f"id '{sid}' sai định dạng (fig-<chương>-<số> hoặc tab-<chương>-<số>)"], None
    kind, ch, num = m.group(1), int(m.group(2)), int(m.group(3))
    stype = spec.get("type")
    output = str(spec.get("output", ""))
    if not output or "/" in output or "\\" in output:
        errs.append("'output' phải là tên file đơn, không có đường dẫn")
    job = {"spec": spec, "id": sid, "kind": kind, "chapter": ch, "number": num,
           "type": stype, "output": output}

    if stype == "manual":
        if kind != "fig":
            errs.append("mục manual phải có id dạng fig-...")
        src_name = str(spec.get("source", ""))
        src = MANUAL / src_name
        if not src_name or not src.is_file():
            errs.append(f"không tìm thấy figures/manual/{src_name}")
        elif src.suffix.lower() not in MANUAL_EXT:
            errs.append("ảnh thủ công phải là .png, .jpg hoặc .svg")
        elif Path(output).suffix.lower() != src.suffix.lower():
            errs.append("'output' phải cùng phần mở rộng với file nguồn")
        else:
            job["source_path"] = src
            job["source_sha"] = sha256_file(src)
            job["out_path"] = ASSETS / output

    elif stype == "chart":
        if kind != "fig":
            errs.append("chart phải có id dạng fig-...")
        if Path(output).suffix.lower() != ".png":
            errs.append("chart phải xuất ra .png")
        chart = spec.get("chart")
        if chart not in CHART_TYPES:
            errs.append(f"'chart' phải là một trong {sorted(CHART_TYPES)}")
        for key in ("data", "x", "x_label", "y_label", "y_unit"):
            if not spec.get(key):
                errs.append(f"thiếu '{key}'")
        ys = spec.get("y")
        if not isinstance(ys, list) or not ys:
            errs.append("'y' phải là danh sách tên cột không rỗng")
            ys = []
        if chart == "bar" and len(ys) != 1:
            errs.append("bar chỉ nhận đúng một cột trong 'y' (dùng grouped_bar cho nhiều cột)")
        if chart == "grouped_bar" and len(ys) < 2:
            errs.append("grouped_bar cần ít nhất hai cột trong 'y'")
        labels = spec.get("series_labels") or ys
        if len(labels) != len(ys):
            errs.append("'series_labels' phải có cùng số phần tử với 'y'")
        numeric = list(ys) + ([spec["x"]] if chart in {"line", "scatter"} and spec.get("x") else [])
        data_path = ROOT / str(spec.get("data", ""))
        if spec.get("data") and not (DATA.resolve() in data_path.resolve().parents):
            errs.append("'data' phải nằm trong thư mục data/")
        elif spec.get("data") and not data_path.is_file():
            errs.append(f"không tìm thấy {spec['data']}")
        else:
            meta_errs, _, meta_raw = validate_meta(data_path)
            errs += meta_errs
            cols, rows = load_csv(data_path)
            if not rows:
                errs.append(f"{spec['data']} không có dòng dữ liệu")
            for c in [spec.get("x")] + ys:
                if c and c not in cols:
                    errs.append(f"{spec['data']} thiếu cột '{c}' (các cột: {', '.join(cols)})")
            if not errs:
                errs += check_numeric(rows, [c for c in numeric if c != spec.get("x") or chart in {"line", "scatter"}])
            job.update(rows=rows, source_sha=sha256_bytes(sha256_file(data_path).encode() + sha256_bytes(meta_raw).encode()))
            job["out_path"] = ASSETS / output

    elif stype == "table":
        if kind != "tab":
            errs.append("table phải có id dạng tab-...")
        if Path(output).suffix.lower() != ".md":
            errs.append("table phải xuất ra .md")
        data_path = ROOT / str(spec.get("data", ""))
        if not spec.get("data") or not (DATA.resolve() in data_path.resolve().parents) or not data_path.is_file():
            errs.append(f"không tìm thấy dữ liệu hợp lệ trong data/: {spec.get('data')}")
        else:
            meta_errs, _, meta_raw = validate_meta(data_path)
            errs += meta_errs
            cols, rows = load_csv(data_path)
            if not rows:
                errs.append("file dữ liệu không có dòng")
            use = spec.get("columns") or cols
            for c in use:
                if c not in cols:
                    errs.append(f"thiếu cột '{c}' trong {spec['data']}")
            numeric = spec.get("numeric")
            if numeric is None:
                numeric = [c for c in use if rows and all(_isfloat(r[c]) for r in rows)]
            if not errs:
                errs += check_numeric(rows, numeric)
            job.update(rows=rows, use_cols=use, numeric=numeric,
                       source_sha=sha256_bytes(sha256_file(data_path).encode() + sha256_bytes(meta_raw).encode()))
            job["out_path"] = TABLES / output

    else:
        errs.append(f"type '{stype}' không hợp lệ (chart | table | manual)")

    return errs, (job if not errs else None)


def _isfloat(v):
    try:
        return math.isfinite(float(v))
    except ValueError:
        return False


def render_chart(job):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    spec, rows = job["spec"], job["rows"]
    dec = spec.get("decimals", STYLE["decimals"])
    ys = spec["y"]
    labels = spec.get("series_labels") or ys
    pal = STYLE["palette"]

    plt.rcParams.update({
        "font.family": STYLE["font_family"],
        "font.size": STYLE["font_size"],
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
    })
    fig, ax = plt.subplots(figsize=(STYLE["width_in"], STYLE["height_in"]))

    if spec["chart"] in {"bar", "grouped_bar"}:
        cats = [r[spec["x"]] for r in rows]
        n, k = len(cats), len(ys)
        width = 0.8 / k
        for j, col in enumerate(ys):
            vals = [float(r[col]) for r in rows]
            pos = [i - 0.4 + width * (j + 0.5) for i in range(n)]
            bars = ax.bar(pos, vals, width=width * 0.92, color=pal[j % len(pal)], label=labels[j])
            for b, v in zip(bars, vals):
                ax.annotate(fmt_num(v, dec), (b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                            fontsize=STYLE["font_size"] - 2)
        ax.set_xticks(range(n))
        ax.set_xticklabels(cats)
    else:
        xv = [float(r[spec["x"]]) for r in rows]
        for j, col in enumerate(ys):
            yv = [float(r[col]) for r in rows]
            if spec["chart"] == "line":
                ax.plot(xv, yv, marker="o", color=pal[j % len(pal)], label=labels[j], linewidth=1.6)
            else:
                ax.scatter(xv, yv, color=pal[j % len(pal)], label=labels[j], s=18)

    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel(spec["y_label"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_num(v, dec)))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6)
    ax.set_axisbelow(True)
    if len(ys) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(job["out_path"], dpi=STYLE["dpi"], metadata={"Software": f"make_figures {VERSION}"})
    plt.close(fig)


def esc_cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def render_table(job):
    spec, rows = job["spec"], job["rows"]
    cols, numeric = job["use_cols"], set(job["numeric"])
    dec_spec = spec.get("decimals", STYLE["decimals"])

    def col_dec(c):
        # decimals có thể là một số chung, hoặc dict theo từng cột: {"f1": 2, "thoi_gian_ms": 0}
        return dec_spec.get(c, STYLE["decimals"]) if isinstance(dec_spec, dict) else dec_spec
    headers = spec.get("headers", {})
    lines = ["| " + " | ".join(esc_cell(headers.get(c, c)) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = [fmt_num(float(r[c]), col_dec(c)) if c in numeric else esc_cell(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    job["out_path"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def spec_sha(spec):
    return sha256_bytes(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def check_stale(jobs, manifest):
    stale = []
    for job in jobs:
        sid, out = job["id"], job["out_path"]
        entry = manifest.get(sid)
        if not out.exists():
            stale.append(f"{sid}: chưa có file {rel(out)}")
            continue
        if not entry:
            stale.append(f"{sid}: chưa có trong manifest")
            continue
        if entry.get("spec_sha256") != spec_sha(job["spec"]):
            stale.append(f"{sid}: specs.json đã đổi")
        if entry.get("source_sha256") != job["source_sha"]:
            stale.append(f"{sid}: dữ liệu nguồn đã đổi")
        if entry.get("output_sha256") != sha256_file(out):
            stale.append(f"{sid}: file đầu ra đã bị sửa tay")
    return stale


def main():
    ap = argparse.ArgumentParser(description="Sinh hình và bảng từ dữ liệu thật.")
    ap.add_argument("--only", help="chỉ xử lý một mục, ví dụ fig-2-1")
    ap.add_argument("--check", action="store_true", help="chỉ kiểm tra, không ghi file")
    args = ap.parse_args()

    specs = load_specs()
    if args.only:
        specs = [s for s in specs if s.get("id") == args.only]
        if not specs:
            sys.exit(f"Không có mục '{args.only}' trong specs.json")
    if not specs:
        print("figures/specs.json rỗng: không có hình hay bảng nào để sinh.")
        return 0

    all_errs, jobs, seen_ids, seen_pos, seen_out = [], [], set(), set(), set()
    for spec in specs:
        sid = str(spec.get("id", "?"))
        if sid in seen_ids:
            all_errs.append(f"{sid}: id bị trùng")
        seen_ids.add(sid)
        errs, job = prepare(spec)
        if job:
            key = (job["kind"], job["chapter"], job["number"])
            if key in seen_pos:
                errs.append("trùng số thứ tự trong chương")
            seen_pos.add(key)
            if job["output"] in seen_out:
                errs.append("trùng tên file đầu ra")
            seen_out.add(job["output"])
        all_errs += [f"{sid}: {e}" for e in errs]
        if job and not errs:
            jobs.append(job)

    if all_errs:
        print("Có lỗi, KHÔNG ghi file nào:")
        for e in all_errs:
            print(f"  [ERROR] {e}")
        return 1

    manifest = load_manifest()
    if args.check:
        stale = check_stale(jobs, manifest)
        if stale:
            print("Cần sinh lại:")
            for s in stale:
                print(f"  [STALE] {s}")
            return 1
        print(f"Tất cả {len(jobs)} mục đều cập nhật.")
        return 0

    ASSETS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        if job["type"] == "manual":
            shutil.copyfile(job["source_path"], job["out_path"])
        elif job["type"] == "chart":
            render_chart(job)
        else:
            render_table(job)
        print(f"  đã sinh {rel(job['out_path'])}")
        manifest[job["id"]] = {
            "type": job["type"],
            "output_path": rel(job["out_path"]),
            "output_sha256": sha256_file(job["out_path"]),
            "source_sha256": job["source_sha"],
            "spec_sha256": spec_sha(job["spec"]),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nXong: {len(jobs)} mục. Manifest: {rel(MANIFEST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
