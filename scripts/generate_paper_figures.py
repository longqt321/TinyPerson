"""Generate IEEE/CVPR-grade Seaborn scientific figures (.pdf + 300 DPI .png) from verified run JSON/CSVs.

Never hardcodes metric values — reads strictly from ``outputs/<exp>/<run_id>/metrics/`` and ``training/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import orjson
import pandas as pd
import polars as pl
import seaborn as sns


def _pl_to_pd(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_dict(as_series=False))

# Okabe-Ito colorblind-safe publication palette
PALETTE = {
    "YOLO11n-P2 (NMS)": "#7F7F7F",
    "YOLO26n-P2 Base": "#0072B2",
    "YOLO26n-P2 + NWD": "#E69F00",
    "YOLO26n-P2 + LGT (Ours)": "#D55E00",
    "w/o DW-TAL": "#56B4E9",
    "w/o FG-Gate & Distill": "#009E73",
}


def configure_scientific_theme() -> None:
    """Configure Seaborn + Matplotlib for IEEE/CVPR single- and double-column typography."""
    sns.set_theme(context="paper", style="ticks", font_scale=1.05)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def _save_dual(fig: plt.Figure, out_dirs: list[Path], stem: str) -> None:
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", format="pdf")
        fig.savefig(d / f"{stem}.png", format="png", dpi=300)
    plt.close(fig)


def plot_scale_breakdown(
    runs: list[tuple[str, Path]],
    json_name: str,
    title_suffix: str,
    out_dirs: list[Path],
    stem: str,
) -> None:
    buckets = [
        ("all", "All"),
        ("tiny", "Tiny\n(2-20)"),
        ("tiny1", "Tiny1\n(2-8)"),
        ("tiny2", "Tiny2\n(8-12)"),
        ("tiny3", "Tiny3\n(12-20)"),
        ("small", "Small\n(20-32)"),
        ("reasonable", "Reas.\n(>32)"),
    ]
    rows = []
    for label, run_dir in runs:
        f = run_dir / "metrics" / json_name
        if not f.is_file():
            continue
        data = orjson.loads(f.read_bytes())
        for key, disp in buckets:
            if key in data:
                rows.append({
                    "Model": label,
                    "Scale": disp,
                    "AP50 (%)": float(data[key]["AP@0.50"]) * 100.0,
                })
    if not rows:
        return
    df = _pl_to_pd(pl.DataFrame(rows))
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    sns.barplot(
        data=df,
        x="Scale",
        y="AP50 (%)",
        hue="Model",
        palette=PALETTE,
        edgecolor="#222222",
        linewidth=0.6,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(f"AP@0.50 (%) [{title_suffix}]")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=7.5, ncol=2, loc="upper left")
    sns.despine(ax=ax)
    _save_dual(fig, out_dirs, stem)


def plot_tide_and_crowded(
    runs: list[tuple[str, Path]],
    suite_name: str,
    out_dirs: list[Path],
    stem: str,
) -> None:
    err_rows = []
    crowd_rows = []
    for label, run_dir in runs:
        f = run_dir / "metrics" / suite_name
        if not f.is_file():
            continue
        tc = orjson.loads(f.read_bytes()).get("tide_and_crowded", {})
        counts = tc.get("counts", {})
        for k, disp in [
            ("FP_Loc", "Loc FP"),
            ("FP_Dupe", "Duplicate FP"),
            ("FP_Bkg", "Background FP"),
            ("FN_Miss", "Missed GT"),
        ]:
            if k in counts:
                err_rows.append({"Model": label, "Category": disp, "Count": int(counts[k])})
        cp = tc.get("crowded_pairs", {})
        for ck, cdisp in [
            ("crowded_lt_10px", "<10 px\n(Crowded)"),
            ("close_10_20px", "10-20 px\n(Close)"),
            ("isolated_ge_20px", ">=20 px\n(Isolated)"),
        ]:
            if ck in cp:
                crowd_rows.append({
                    "Model": label,
                    "Density": cdisp,
                    "Recall@50 (%)": float(cp[ck]["recall_at_50"]) * 100.0,
                })
    if not err_rows or not crowd_rows:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax in (ax1, ax2):
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    sns.barplot(
        data=_pl_to_pd(pl.DataFrame(err_rows)),
        x="Category",
        y="Count",
        hue="Model",
        palette=PALETTE,
        edgecolor="#222222",
        linewidth=0.6,
        ax=ax1,
    )
    ax1.set_xlabel("(a) TIDE Error Counts (Lower is Better)")
    ax1.set_ylabel("Instance Count")
    ax1.legend(frameon=True, fontsize=6.8, loc="upper left")

    sns.barplot(
        data=_pl_to_pd(pl.DataFrame(crowd_rows)),
        x="Density",
        y="Recall@50 (%)",
        hue="Model",
        palette=PALETTE,
        edgecolor="#222222",
        linewidth=0.6,
        ax=ax2,
    )
    ax2.set_xlabel("(b) Nearest-Neighbor Recall (Higher is Better)")
    ax2.set_ylabel("Recall@0.50 (%)")
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

    sns.despine(fig=fig)
    fig.tight_layout()
    _save_dual(fig, out_dirs, stem)


def plot_convergence_curves(
    runs: list[tuple[str, Path]],
    out_dirs: list[Path],
    stem: str,
) -> None:
    frames: list[pl.DataFrame] = []
    for label, run_dir in runs:
        csv_f = run_dir / "training" / "results.csv"
        if not csv_f.is_file():
            continue
        raw = pl.read_csv(csv_f)
        raw = raw.rename({c: c.strip() for c in raw.columns})
        if "metrics/mAP50(B)" not in raw.columns:
            continue
        sub = raw.select(
            pl.col("epoch").cast(pl.Int64).alias("Epoch"),
            (pl.col("metrics/mAP50(B)").cast(pl.Float64) * 100.0).alias("Val mAP@0.50 (%)"),
            (pl.col("metrics/precision(B)").cast(pl.Float64) * 100.0).alias("Val Precision (%)"),
            pl.lit(label).alias("Model"),
        )
        frames.append(sub)
    if not frames:
        return
    all_df = _pl_to_pd(pl.concat(frames))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax in (ax1, ax2):
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)

    sns.lineplot(
        data=all_df, x="Epoch", y="Val mAP@0.50 (%)", hue="Model", palette=PALETTE, linewidth=1.5, ax=ax1
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation mAP@0.50 (%)")
    ax1.legend(frameon=True, fontsize=7.0, loc="lower right")

    sns.lineplot(
        data=all_df, x="Epoch", y="Val Precision (%)", hue="Model", palette=PALETTE, linewidth=1.5, ax=ax2
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Precision (%)")
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

    sns.despine(fig=fig)
    fig.tight_layout()
    _save_dual(fig, out_dirs, stem)


def plot_ablation_ladder(
    abl_runs: list[tuple[str, Path]],
    json_name: str,
    out_dirs: list[Path],
    stem: str,
) -> None:
    rows = []
    for label, run_dir in abl_runs:
        f = run_dir / "metrics" / json_name
        if not f.is_file():
            continue
        d = orjson.loads(f.read_bytes())
        for key, metric_name in [
            ("all", "AP50 (All)"),
            ("tiny1", "AP50 (Tiny1 2-8px)"),
            ("reasonable", "AP50 (Reas. >32px)"),
        ]:
            if key in d:
                rows.append({
                    "Variant": label,
                    "Metric": metric_name,
                    "Score (%)": float(d[key]["AP@0.50"]) * 100.0,
                })
    if not rows:
        return
    df = _pl_to_pd(pl.DataFrame(rows))
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    sns.barplot(
        data=df,
        x="Metric",
        y="Score (%)",
        hue="Variant",
        palette=PALETTE,
        edgecolor="#222222",
        linewidth=0.6,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("AP@0.50 (%)")
    ax.legend(frameon=True, fontsize=7.2, loc="upper left")
    sns.despine(ax=ax)
    _save_dual(fig, out_dirs, stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--prefix", default="tp80", choices=["tp80", "tp"])
    parser.add_argument("--run-id", default="colab-seed42")
    args = parser.parse_args()

    configure_scientific_theme()
    out_dirs = [Path("docs/report/figures"), Path("paper/figures")]

    p = args.prefix
    rid = args.run_id
    root = args.outputs
    headline = [
        ("YOLO11n-P2 (NMS)", root / f"{p}_yolo11n_nms" / rid),
        ("YOLO26n-P2 Base", root / f"{p}_yolo26n_base" / rid),
        ("YOLO26n-P2 + NWD", root / f"{p}_yolo26n_nwd" / rid),
        ("YOLO26n-P2 + LGT (Ours)", root / f"{p}_yolo26n_lgt" / rid),
    ]
    ablation = [
        ("YOLO26n-P2 Base", root / f"{p}_yolo26n_base" / rid),
        ("w/o FG-Gate & Distill", root / f"{p}_abl_nogate" / rid),
        ("w/o DW-TAL", root / f"{p}_abl_nodwtal" / rid),
        ("YOLO26n-P2 + LGT (Ours)", root / f"{p}_yolo26n_lgt" / rid),
    ]

    plot_scale_breakdown(headline, "tinyperson_test.json", "Full-Frame 1280", out_dirs, "fig_lgt_scale_full1280")
    plot_scale_breakdown(headline, "tinyperson_test_tiled.json", "Native Tiled 640x512", out_dirs, "fig_lgt_scale_tiled")
    plot_tide_and_crowded(headline, "coco_test_suite.json", out_dirs, "fig_lgt_tide_crowded")
    plot_convergence_curves(headline, out_dirs, "fig_lgt_convergence")
    plot_ablation_ladder(ablation, "tinyperson_test.json", out_dirs, "fig_lgt_ablation")
    print(f"[figures] Generated Seaborn PDF/PNG figures in {[str(d) for d in out_dirs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
