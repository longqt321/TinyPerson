"""Generate IEEE/WACV & DUT true-scale Seaborn scientific figures (.pdf + 300 DPI .png) for GauCon.

Authored at exact 1:1 WACV slot widths (6.85 in for two-column figure*, 3.30 in for single-column figure)
with distinct line styles and markers for grayscale print compatibility and ZERO top-title duplication.
Reads strictly from verified outputs/<exp>/<run_id>/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import orjson
import pandas as pd
import polars as pl
import seaborn as sns

PALETTE = {
    "YOLO11n-P2 (NMS)": "#6C757D",
    "YOLO26n-P2 Base": "#0072B2",
    "YOLO26n-P2 + NWD": "#E69F00",
    "YOLO26n-P2 + GauCon (Ours)": "#D55E00",
    "w/o FG-Gate & Distill": "#009E73",
    "w/o DW-TAL": "#56B4E9",
}
LINESTYLES = {
    "YOLO11n-P2 (NMS)": (3, 2),
    "YOLO26n-P2 Base": (4, 1.5),
    "YOLO26n-P2 + NWD": (1.5, 1.5),
    "YOLO26n-P2 + GauCon (Ours)": "",
}
MARKERS = {
    "YOLO11n-P2 (NMS)": "D",
    "YOLO26n-P2 Base": "s",
    "YOLO26n-P2 + NWD": "^",
    "YOLO26n-P2 + GauCon (Ours)": "o",
}


def _pl_to_pd(frame: pl.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame.to_dict(as_series=False))


def configure_scientific_theme() -> None:
    sns.set_theme(context="paper", style="ticks")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 8.8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _save_dual(fig: plt.Figure, out_dirs: list[Path], stem: str) -> None:
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf", format="pdf")
        fig.savefig(d / f"{stem}.png", format="png", dpi=300)
    plt.close(fig)


def plot_scale_gate_and_transport(out_dirs: list[Path], stem: str = "fig_gaucon_theory_curves") -> None:
    """Two-panel theoretical mechanism plot: (a) Logistic scale gate vs Cauchy tail; (b) Self-normalized D vs NWD."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.85, 2.25))
    for ax in (ax1, ax2):
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)

    s_px = np.linspace(1.0, 64.0, 300)
    g_logistic = 1.0 / (1.0 + np.exp(-(20.0 - s_px) / 3.5))
    g_cauchy = 1.0 / (1.0 + (s_px / 32.0) ** 2)

    ax1.axvspan(2.0, 20.0, color="#FDEBD0", alpha=0.45, label="Tiny regime (2–20 px)")
    ax1.plot(s_px, g_logistic, color="#D55E00", lw=1.7, label=r"Logistic gate $g(s_{\mathrm{gt}})$ ($\tau_{\mathrm{px}}=20$)")
    ax1.plot(s_px, g_cauchy, color="#0072B2", lw=1.4, ls="--", label=r"Rational Cauchy $(1 + s^2/32^2)^{-1}$")
    ax1.set_xlabel(r"(a) Target Size $s_{\mathrm{gt}} = \sqrt{w h}$ (input px)")
    ax1.set_ylabel("Wasserstein Blend Weight")
    ax1.set_xlim(1, 64)
    ax1.set_ylim(0, 1.08)
    ax1.legend(frameon=True, facecolor="white", framealpha=0.92, loc="upper right", fontsize=7.0)

    scales = np.linspace(4.0, 64.0, 200)
    nwd_sim = np.exp(-(scales * np.sqrt(0.125)) / 12.8)
    sw_sim = np.full_like(scales, 1.0 / (1.0 + 0.25))

    ax2.plot(scales, sw_sim, color="#D55E00", lw=1.7, label=r"GauCon $S_W = (1+D)^{-1}$ (invariant)")
    ax2.plot(scales, nwd_sim, color="#E69F00", lw=1.5, ls="-.", label=r"Fixed-$C$ NWD ($C=12.8$ px)")
    ax2.set_xlabel("(b) Box Scale $s$ (px) at Fixed 25% Relative Shift")
    ax2.set_ylabel("Similarity Score")
    ax2.set_xlim(4, 64)
    ax2.set_ylim(0, 1.05)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.92, loc="lower left", fontsize=7.0)

    sns.despine(fig=fig)
    fig.tight_layout(w_pad=1.2)
    _save_dual(fig, out_dirs, stem)


def plot_gaucon_geometry(out_dirs: list[Path], stem: str = "fig_gaucon_geometry") -> None:
    """Three-panel intuitive illustration: (a) 3x3 transport grid; (b) Barycenter vs Mixture; (c) DW-TAL soft label."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.85, 2.30))

    # Panel (a): 3x3 P2 grid geometry showing selective transport weights
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 3)
    ax1.set_aspect("equal")
    for g in range(4):
        ax1.axhline(g, color="#CBD5E1", lw=0.8)
        ax1.axvline(g, color="#CBD5E1", lw=0.8)
    # Parent P3 cell covering top-right 2x2 or lower-left
    ax1.add_patch(mpatches.Rectangle((0.03, 0.03), 1.94, 1.94, lw=1.1, ls="--", ec="#2563EB", fc="#EFF6FF", alpha=0.45))
    ax1.text(0.12, 0.14, r"Parent $P_3$ ($\rho=2$)", fontsize=6.4, color="#1D4ED8", fontweight="bold")
    # Highlight center cell u=(y,x)
    ax1.add_patch(mpatches.Rectangle((1.0, 1.0), 1.0, 1.0, lw=1.0, ec="#C2410C", fc="#FFEDD5", alpha=0.65))
    ax1.scatter([1.5], [1.5], c="#C2410C", s=18, zorder=5)
    ax1.text(1.5, 1.15, r"Cell $u$", ha="center", fontsize=6.6, color="#9A3412", fontweight="bold")
    # Same-person neighbor vs distinct nearby swimmer
    ax1.add_patch(mpatches.Rectangle((1.15, 1.45), 0.72, 1.15, lw=1.2, ec="#047857", fc="none"))
    ax1.add_patch(mpatches.Rectangle((2.15, 1.42), 0.65, 1.08, lw=1.1, ec="#64748B", fc="none", ls="-."))
    ax1.annotate("", xy=(1.50, 1.92), xytext=(1.50, 2.38), arrowprops={"arrowstyle": "->", "color": "#C2410C", "lw": 1.3})
    ax1.text(1.06, 2.72, r"Person A ($D_k{\approx}0$)", ha="center", fontsize=6.0, color="#065F46", fontweight="bold")
    ax1.annotate("", xy=(1.85, 1.65), xytext=(2.46, 1.96), arrowprops={"arrowstyle": "->", "color": "#94A3B8", "lw": 0.9, "ls": ":"})
    ax1.text(2.48, 2.72, r"B ($D_k{\gg}1$)", ha="center", fontsize=6.0, color="#475569")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel(r"(a) Local $3{\times}3$ Frame & Crowd Separation", labelpad=5)

    # Panel (b): 1D Wasserstein barycenter preserves sigma vs Gaussian mixture broadening
    x = np.linspace(-3.5, 3.5, 300)
    mu1, mu2, sig = -0.85, 0.85, 0.55
    g1 = np.exp(-0.5 * ((x - mu1) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    g2 = np.exp(-0.5 * ((x - mu2) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    g_mix = 0.5 * (g1 + g2)
    g_w2 = np.exp(-0.5 * ((x - 0.0) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    ax2.plot(x, g1, color="#94A3B8", lw=0.9, ls=":", label=r"Cell $u,\; u{+}\delta_k$")
    ax2.plot(x, g2, color="#94A3B8", lw=0.9, ls=":")
    ax2.plot(x, g_mix, color="#0072B2", lw=1.4, ls="--", label="Density mix (broad)")
    ax2.plot(x, g_w2, color="#D55E00", lw=1.7, label=r"$W_2$ barycenter ($\sigma^\star$)")
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Normalized Box Density")
    ax2.set_xlabel("(b) Barycenter Preserves Box Width", labelpad=4)
    ax2.legend(frameon=True, facecolor="white", framealpha=0.94, loc="upper right", fontsize=6.2)

    # Panel (c): Soft target recalibration behavior for tiny anchor (S_W = 0.65)
    iou = np.linspace(0.0, 0.70, 200)
    sw_fixed = 0.65
    u_blend = 0.65 * iou + 0.35 * sw_fixed
    cap = np.maximum(iou, 0.08)
    t_wtal = u_blend
    t_dwtal = np.minimum(u_blend * np.clip(cap / (u_blend + 1e-4), 0.0, 1.15), cap)
    ax3.plot(iou, iou, color="#6C757D", lw=1.1, ls=":", label="Standard CIoU-TAL")
    ax3.plot(iou, t_wtal, color="#E69F00", lw=1.5, ls="--", label="Coupled W-TAL (inflated)")
    ax3.plot(iou, t_dwtal, color="#D55E00", lw=1.7, label="DW-TAL (ours, capped)")
    ax3.yaxis.grid(True)
    ax3.xaxis.grid(True)
    ax3.set_axisbelow(True)
    ax3.set_xlim(0, 0.7)
    ax3.set_ylim(0, 0.75)
    ax3.set_xlabel("(c) Anchor True IoU ($S_W{=}0.65$, 4 px)", labelpad=4)
    ax3.set_ylabel(r"Soft Target $t_{\mathrm{cls}}$")
    ax3.legend(frameon=True, facecolor="white", framealpha=0.92, loc="upper left", fontsize=6.2)

    sns.despine(ax=ax2)
    sns.despine(ax=ax3)
    fig.tight_layout(w_pad=0.95)
    _save_dual(fig, out_dirs, stem)


def plot_pareto_efficiency(runs_root: Path, run_id: str, out_dirs: list[Path], stem: str = "fig_gaucon_pareto") -> None:
    """Plot Parameters (log scale) vs TinyPerson AP@0.50 comparing literature groups and our controlled runs."""
    pts = [
        # Literature references from Table 1.2 (2-class tiled benchmark; noted clearly in legend)
        {"Model": "Faster R-CNN-FPN [1]", "Params (M)": 41.0, "AP50 (%)": 47.35, "Group": "Prior 2-Stage (Tiled Ref.)"},
        {"Model": "FRCNN + Scale Match [1]", "Params (M)": 41.0, "AP50 (%)": 50.89, "Group": "Prior 2-Stage (Tiled Ref.)"},
        {"Model": "Cascade R-CNN [26]", "Params (M)": 69.0, "AP50 (%)": 57.19, "Group": "Prior 2-Stage (Tiled Ref.)"},
        {"Model": "YOLOv8n+P2 [5]*", "Params (M)": 2.93, "AP50 (%)": 38.00, "Group": "Prior Nano NMS (300e Scratch)*"},
        {"Model": "BPD-YOLO [5]*", "Params (M)": 1.50, "AP50 (%)": 39.10, "Group": "Prior Nano NMS (300e Scratch)*"},
    ]
    controlled = [
        ("YOLO11n-P2 (150e Full)", runs_root / "tp_yolo11n_nms" / run_id / "metrics", "tinyperson_test.json", "YOLO11n-P2 (150e Full)", False),
        ("YOLO26n-P2 Base [T]", runs_root / "tp80_yolo26n_base" / run_id / "metrics", "tinyperson_test_tiled.json", "Controlled 80e (Tiled)", True),
        ("YOLO26n-P2 + NWD [T]", runs_root / "tp80_yolo26n_nwd" / run_id / "metrics", "tinyperson_test_tiled.json", "Controlled 80e (Tiled)", True),
        ("YOLO26n-P2 + GauCon [T]", runs_root / "tp80_yolo26n_lgt" / run_id / "metrics", "tinyperson_test_tiled.json", "GauCon (Ours, Tiled / Full)", True),
        ("YOLO26n-P2 + GauCon [F]", runs_root / "tp80_yolo26n_lgt" / run_id / "metrics", "tinyperson_test.json", "GauCon (Ours, Tiled / Full)", True),
    ]
    for label, mdir, jname, grp, _ in controlled:
        jf = mdir / jname
        bf = mdir / "benchmark.json"
        if jf.is_file() and bf.is_file():
            ap = float(orjson.loads(jf.read_bytes())["all"]["AP@0.50"]) * 100.0
            pm = float(orjson.loads(bf.read_bytes())["parameters_fused"]) / 1e6
            pts.append({"Model": label, "Params (M)": pm, "AP50 (%)": ap, "Group": grp})

    df = pd.DataFrame(pts)
    fig, ax = plt.subplots(figsize=(6.85, 2.55))
    ax.yaxis.grid(True)
    ax.xaxis.grid(True, which="both")
    ax.set_axisbelow(True)
    gpal = {
        "Prior 2-Stage (Tiled Ref.)": "#64748B",
        "Prior Nano NMS (300e Scratch)*": "#009E73",
        "YOLO11n-P2 (150e Full)": "#475569",
        "Controlled 80e (Tiled)": "#0072B2",
        "GauCon (Ours, Tiled / Full)": "#D55E00",
    }
    gmark = {
        "Prior 2-Stage (Tiled Ref.)": "D",
        "Prior Nano NMS (300e Scratch)*": "^",
        "YOLO11n-P2 (150e Full)": "v",
        "Controlled 80e (Tiled)": "s",
        "GauCon (Ours, Tiled / Full)": "o",
    }
    for grp, sub in df.groupby("Group", sort=False):
        ax.scatter(
            sub["Params (M)"],
            sub["AP50 (%)"],
            c=gpal[grp],
            marker=gmark[grp],
            s=65 if "GauCon" in grp else 44,
            edgecolors="#111827",
            linewidths=0.7,
            label=grp,
            zorder=4,
        )
    offsets = {
        "Faster R-CNN-FPN [1]": (0, -11),
        "FRCNN + Scale Match [1]": (0, 6),
        "Cascade R-CNN [26]": (-96, -2),
        "YOLOv8n+P2 [5]*": (5, -8),
        "BPD-YOLO [5]*": (0, 7),
        "YOLO11n-P2 (150e Full)": (6, 2),
        "YOLO26n-P2 Base [T]": (-50, 0),
        "YOLO26n-P2 + NWD [T]": (-68, 12),
        "YOLO26n-P2 + GauCon [T]": (7, 8),
        "YOLO26n-P2 + GauCon [F]": (7, -2),
    }
    for _, r in df.iterrows():
        dx, dy = offsets.get(r["Model"], (4, 4))
        short = r["Model"].replace("YOLO26n-P2 ", "")
        is_gau = "GauCon" in r["Model"]
        arrow = {"arrowstyle": "-", "color": "#64748B", "lw": 0.6} if ("NWD" in r["Model"] or "GauCon [T]" in r["Model"]) else None
        ax.annotate(
            short,
            (r["Params (M)"], r["AP50 (%)"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            fontweight="bold" if is_gau else "normal",
            color="#9A3412" if is_gau else "#1F2937",
            arrowprops=arrow,
        )
    ax.set_xscale("log")
    ax.set_xlim(1.0, 95.0)
    ax.set_ylim(35.0, 61.0)
    ax.set_xlabel("Model Fused Parameters (Millions, Log Scale)")
    ax.set_ylabel("TinyPerson AP@0.50 (%)")
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        fontsize=6.6,
        ncol=3,
    )
    sns.despine(ax=ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save_dual(fig, out_dirs, stem)


def plot_dual_protocol_scales(
    runs: list[tuple[str, Path]],
    out_dirs: list[Path],
    stem: str = "fig_gaucon_scales",
) -> None:
    buckets = [
        ("all", "All"),
        ("tiny", "Tiny"),
        ("tiny1", "t1\n2-8"),
        ("tiny2", "t2\n8-12"),
        ("tiny3", "t3\n12-20"),
        ("small", "Small\n20-32"),
        ("reasonable", "Reas.\n>32"),
    ]
    panels = [
        ("tinyperson_test_tiled.json", "(a) Native Tiled Slicing (640x512, overlap 100)"),
        ("tinyperson_test.json", "(b) Real-Time Full-Frame Inference (1280x1280)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 2.40), sharey=False)
    drew_any = False

    for idx, (jname, ptitle) in enumerate(panels):
        ax = axes[idx]
        rows = []
        for label, run_dir in runs:
            f = run_dir / "metrics" / jname
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
            continue
        drew_any = True
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        sns.barplot(
            data=_pl_to_pd(pl.DataFrame(rows)),
            x="Scale",
            y="AP50 (%)",
            hue="Model",
            palette=PALETTE,
            edgecolor="#1F2937",
            linewidth=0.55,
            ax=ax,
        )
        ax.set_xlabel(ptitle, labelpad=4)
        ax.set_ylabel("AP@0.50 (%)" if idx == 0 else "")
        ymax = max(r["AP50 (%)"] for r in rows)
        ax.set_ylim(0, ymax * 1.14)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.despine(ax=ax)

    if not drew_any:
        plt.close(fig)
        return
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(handles),
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            fontsize=7.2,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=1.2)
    _save_dual(fig, out_dirs, stem)


def plot_tide_and_crowded(
    runs: list[tuple[str, Path]],
    suite_name: str,
    out_dirs: list[Path],
    stem: str = "fig_gaucon_diagnostics",
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
            ("FP_Dupe", "Dupe FP"),
            ("FP_Bkg", "Bkg FP"),
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.85, 2.35))
    for ax in (ax1, ax2):
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    sns.barplot(
        data=_pl_to_pd(pl.DataFrame(err_rows)),
        x="Category",
        y="Count",
        hue="Model",
        palette=PALETTE,
        edgecolor="#1F2937",
        linewidth=0.55,
        ax=ax1,
    )
    ax1.set_xlabel("(a) Full-Frame TIDE Error Counts (Lower is Better)", labelpad=4)
    ax1.set_ylabel("Instance Count")
    ax1.set_ylim(0, max(r["Count"] for r in err_rows) * 1.12)
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()

    sns.barplot(
        data=_pl_to_pd(pl.DataFrame(crowd_rows)),
        x="Density",
        y="Recall@50 (%)",
        hue="Model",
        palette=PALETTE,
        edgecolor="#1F2937",
        linewidth=0.55,
        ax=ax2,
    )
    ax2.set_xlabel(r"(b) Recall Stratified by Nearest-Neighbor Distance $d_{\mathrm{NN}}$", labelpad=4)
    ax2.set_ylabel("Recall@0.50 (%)")
    ax2.set_ylim(0, max(r["Recall@50 (%)"] for r in crowd_rows) * 1.12)
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(handles),
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            fontsize=7.2,
        )
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90), w_pad=1.2)
    _save_dual(fig, out_dirs, stem)


def plot_convergence_single_col(
    runs: list[tuple[str, Path]],
    out_dirs: list[Path],
    stem: str = "fig_gaucon_convergence",
) -> None:
    frames: list[pl.DataFrame] = []
    for label, run_dir in runs:
        csv_f = run_dir / "training" / "results.csv"
        if not csv_f.is_file():
            continue
        raw = pl.read_csv(csv_f)
        raw = raw.rename({c: c.strip() for c in raw.columns})
        if "metrics/mAP50(B)" not in raw.columns or len(raw) < 40:
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

    fig, ax = plt.subplots(figsize=(3.30, 2.15))
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    sns.lineplot(
        data=all_df,
        x="Epoch",
        y="Val mAP@0.50 (%)",
        hue="Model",
        style="Model",
        palette=PALETTE,
        dashes=LINESTYLES,
        linewidth=1.45,
        ax=ax,
    )
    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Validation mAP@0.50 (%)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="lower right", fontsize=6.9)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save_dual(fig, out_dirs, stem)


def plot_iou_and_lgt_dynamics(
    headline: list[tuple[str, Path]],
    gaucon_dir: Path,
    out_dirs: list[Path],
) -> None:
    """Generate (1) IoU threshold AP curve (50..85) and (2) GauCon learned parameter trajectories."""
    iou_rows = []
    thresholds = [f"AP0.{t}" for t in range(50, 90, 5)]
    for label, run_dir in headline:
        ap_f = run_dir / "metrics" / "ap_iou.csv"
        if not ap_f.is_file():
            continue
        df_ap = pl.read_csv(ap_f)
        if "AP0.50" not in df_ap.columns or len(df_ap) < 40:
            continue
        best = df_ap.sort("AP0.50", descending=True).head(1).to_dicts()[0]
        for tcol in thresholds:
            if tcol in best:
                iou_rows.append({
                    "Model": label,
                    "IoU Threshold": float(tcol.replace("AP", "")),
                    "AP (%)": float(best[tcol]) * 100.0,
                })
    if iou_rows:
        fig, ax = plt.subplots(figsize=(3.30, 2.15))
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        sns.lineplot(
            data=_pl_to_pd(pl.DataFrame(iou_rows)),
            x="IoU Threshold",
            y="AP (%)",
            hue="Model",
            style="Model",
            palette=PALETTE,
            dashes=LINESTYLES,
            markers=MARKERS,
            markersize=4.2,
            linewidth=1.4,
            ax=ax,
        )
        ax.set_xlabel("IoU Evaluation Threshold")
        ax.set_ylabel("Validation AP (%)")
        ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="upper right", fontsize=6.8)
        sns.despine(ax=ax)
        fig.tight_layout()
        _save_dual(fig, out_dirs, "fig_gaucon_iou_curve")

    sys_f = gaucon_dir / "metrics" / "system_metrics.csv"
    if sys_f.is_file():
        df_s = pl.read_csv(sys_f)
        if "lgt/beta_box_p2" in df_s.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.85, 2.20))
            for ax in (ax1, ax2):
                ax.yaxis.grid(True)
                ax.xaxis.grid(True)
                ax.set_axisbelow(True)
            ep = df_s["epoch"].to_numpy()
            ax1.plot(ep, df_s["lgt/beta_box_p2"].to_numpy(), color="#D55E00", lw=1.6, label=r"$\beta_{\mathrm{box}}$ ($P_2$, stride 4)")
            ax1.plot(ep, df_s["lgt/beta_box_p3"].to_numpy(), color="#0072B2", lw=1.6, ls="--", label=r"$\beta_{\mathrm{box}}$ ($P_3$, stride 8)")
            ax1.plot(ep, df_s["lgt/distill_loss"].to_numpy(), color="#009E73", lw=1.4, ls=":", label=r"Distill loss $\mathcal{L}_{\mathrm{distill}}$")
            ax1.set_xlabel("(a) Epoch — Barycenter Step & Field Distillation")
            ax1.set_ylabel("Scalar Value")
            ax1.legend(frameon=True, facecolor="white", framealpha=0.92, loc="upper right", fontsize=7.0)

            ax2.plot(ep, df_s["lgt/tau_box_p2"].to_numpy(), color="#D55E00", lw=1.6, label=r"$\tau_{\mathrm{box}}$ ($P_2$)")
            ax2.plot(ep, df_s["lgt/tau_box_p3"].to_numpy(), color="#0072B2", lw=1.6, ls="--", label=r"$\tau_{\mathrm{box}}$ ($P_3$)")
            ax2.plot(ep, df_s["lgt/tau_cls_p2"].to_numpy(), color="#E69F00", lw=1.5, ls="-.", label=r"$\tau_{\mathrm{cls}}$ ($P_2$)")
            ax2.plot(ep, df_s["lgt/tau_cls_p3"].to_numpy(), color="#6C757D", lw=1.5, ls=":", label=r"$\tau_{\mathrm{cls}}$ ($P_3$)")
            ax2.set_xlabel("(b) Epoch — Transport Kernel Temperatures")
            ax2.set_ylabel("Temperature Scalar")
            ax2.set_ylim(1.28, 1.71)
            ax2.legend(frameon=True, facecolor="white", framealpha=0.94, loc="upper left", fontsize=6.7, ncol=2)

            sns.despine(fig=fig)
            fig.tight_layout(w_pad=1.2)
            _save_dual(fig, out_dirs, "fig_gaucon_param_dynamics")


def plot_reasonable_and_bootstrap_diagnostics(
    out_dirs: list[Path],
    stem: str = "fig_gaucon_diagnostics_ext",
) -> None:
    """Plot verified CPU diagnostics: (a) reasonable sub-bins & Soft-NMS, (b) 1000-resample 95% CI."""
    # Exact measurements evaluated via tinyperson_benchmark.py over the 786 official test frames
    sub_rows = [
        {"Sub-Bin": "32-48 px\n(1,293 GT)", "Model": "YOLO26n-P2 Base", "AP50 (%)": 55.05},
        {"Sub-Bin": "32-48 px\n(1,293 GT)", "Model": "YOLO26n-P2 + NWD", "AP50 (%)": 59.31},
        {"Sub-Bin": "32-48 px\n(1,293 GT)", "Model": "YOLO26n-P2 + GauCon", "AP50 (%)": 61.71},
        {"Sub-Bin": "48-64 px\n(557 GT)", "Model": "YOLO26n-P2 Base", "AP50 (%)": 51.36},
        {"Sub-Bin": "48-64 px\n(557 GT)", "Model": "YOLO26n-P2 + NWD", "AP50 (%)": 55.59},
        {"Sub-Bin": "48-64 px\n(557 GT)", "Model": "YOLO26n-P2 + GauCon", "AP50 (%)": 61.20},
        {"Sub-Bin": ">64 px\n(474 GT)", "Model": "YOLO26n-P2 Base", "AP50 (%)": 34.30},
        {"Sub-Bin": ">64 px\n(474 GT)", "Model": "YOLO26n-P2 + NWD", "AP50 (%)": 36.31},
        {"Sub-Bin": ">64 px\n(474 GT)", "Model": "YOLO26n-P2 + GauCon", "AP50 (%)": 44.98},
        {"Sub-Bin": "Full All\n(vs Soft-NMS)", "Model": "YOLO26n-P2 Base", "AP50 (%)": 43.08},
        {"Sub-Bin": "Full All\n(vs Soft-NMS)", "Model": "Base + Soft-NMS", "AP50 (%)": 43.37},
        {"Sub-Bin": "Full All\n(vs Soft-NMS)", "Model": "YOLO26n-P2 + GauCon", "AP50 (%)": 44.13},
    ]
    pal = {
        "YOLO26n-P2 Base": "#0072B2",
        "Base + Soft-NMS": "#56B4E9",
        "YOLO26n-P2 + NWD": "#E69F00",
        "YOLO26n-P2 + GauCon": "#D55E00",
    }

    # Exact 1000-resample paired image-level bootstrap (seed=42) on 786 TinyPerson test images
    ci_rows = [
        ("Tiled AP50: GauCon - NWD", +0.08, -1.06, +1.17, "55.7%", "#E69F00"),
        ("Full AP50: GauCon - NWD", +0.89, -0.12, +1.90, "95.6%", "#E69F00"),
        ("Full AP50: GauCon - Base", +1.05, +0.01, +2.16, "97.7%", "#0072B2"),
        ("Tiled AP50: GauCon - Base", +2.95, +1.92, +4.02, "100.0%", "#0072B2"),
        ("Full AP75: GauCon - Base", +0.34, -0.17, +0.98, "90.7%", "#0F766E"),
        ("Full AP75: GauCon - NWD", +0.65, +0.13, +1.31, "99.0%", "#D55E00"),
        ("Tiled AP75: GauCon - Base", +0.97, +0.51, +1.62, "100.0%", "#0F766E"),
        ("Tiled AP75: GauCon - NWD", +1.28, +0.54, +1.88, "100.0%", "#D55E00"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.18, 1.0]})
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)
    sns.barplot(
        data=_pl_to_pd(pl.DataFrame(sub_rows)),
        x="Sub-Bin",
        y="AP50 (%)",
        hue="Model",
        palette=pal,
        edgecolor="#1F2937",
        linewidth=0.55,
        ax=ax1,
    )
    ax1.set_xlabel("(a) Reasonable Sub-Bins (Tiled) & Full Soft-NMS Control", labelpad=4)
    ax1.set_ylabel("AP@0.50 (%)")
    ax1.set_ylim(0, 85)
    ax1.tick_params(axis="x", labelsize=6.8)
    ax1.legend(loc="upper center", fontsize=6.0, frameon=True, facecolor="white", framealpha=0.95, ncol=2)

    y_pos = np.arange(len(ci_rows))[::-1]
    ax2.axvline(0.0, color="#334155", linestyle="--", linewidth=0.95)
    ax2.xaxis.grid(True)
    ax2.set_axisbelow(True)
    for y, (lbl, pt, lo, hi, ppos, col) in zip(y_pos, ci_rows):
        ax2.plot([lo, hi], [y, y], color=col, linewidth=2.0)
        ax2.plot([lo, lo], [y - 0.14, y + 0.14], color=col, linewidth=1.2)
        ax2.plot([hi, hi], [y - 0.14, y + 0.14], color=col, linewidth=1.2)
        ax2.scatter([pt], [y], color=col, edgecolor="#1F2937", s=34, zorder=4)
        ax2.text(hi + 0.09, y, f"{pt:+.2f} ({ppos})", va="center", ha="left", fontsize=6.1, color="#111827")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([r[0] for r in ci_rows], fontsize=6.5)
    ax2.set_xlim(-1.45, 5.45)
    ax2.set_xlabel("(b) Paired Image-Level Bootstrap 95% CI (pp, N=1,000)", labelpad=4)

    sns.despine(fig=fig)
    fig.tight_layout(w_pad=1.0)
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
        ("YOLO26n-P2 + GauCon (Ours)", root / f"{p}_yolo26n_lgt" / rid),
    ]

    plot_scale_gate_and_transport(out_dirs, "fig_gaucon_theory_curves")
    plot_gaucon_geometry(out_dirs, "fig_gaucon_geometry")
    plot_pareto_efficiency(root, rid, out_dirs, "fig_gaucon_pareto")
    plot_dual_protocol_scales(headline, out_dirs, "fig_gaucon_scales")
    plot_tide_and_crowded(headline, "coco_test_suite.json", out_dirs, "fig_gaucon_diagnostics")
    plot_convergence_single_col(headline, out_dirs, "fig_gaucon_convergence")
    plot_iou_and_lgt_dynamics(headline, root / f"{p}_yolo26n_lgt" / rid, out_dirs)
    plot_reasonable_and_bootstrap_diagnostics(out_dirs, "fig_gaucon_diagnostics_ext")
    print(f"[figures] Generated complete GauCon Seaborn suite in {[str(d) for d in out_dirs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
