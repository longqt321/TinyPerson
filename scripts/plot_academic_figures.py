from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# STANDARD CVPR/ICCV/IEEE PUBLICATION STYLE
# ==============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 10.5,
    "figure.dpi": 300,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "axes.grid": True,
    "grid.linestyle": "--",
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)
paper_dir = Path("paper/figures")
paper_dir.mkdir(parents=True, exist_ok=True)

# Standard sober academic palette (colorblind-safe muted tones)
C_BASE = "#2C5D88"    # Navy Slate
C_GAUSS = "#C0392B"   # Deep Crimson/Rust
C_GRAY1 = "#7F8C8D"   # Neutral Slate
C_GRAY2 = "#95A5A6"
C_DARK = "#2C3E50"
C_GREEN = "#27AE60"

def save_dual(fig, stem):
    svg_path = out_dir / f"{stem}.svg"
    pdf_path = paper_dir / f"{stem}.pdf"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Exported {stem} to SVG & PDF")


# 1. AP50 across size bins (Figure 2 in text)
def plot_ap50_by_size_bin():
    categories = [
        "tiny1\n(2–8 px)",
        "tiny2\n(8–12 px)",
        "tiny3\n(12–20 px)",
        "tiny\n(2–20 px)",
        "small\n(20–32 px)",
        "reasonable\n(>32 px)",
        "all\n(≥2 px)"
    ]
    base_vals = [4.72, 6.90, 13.51, 9.07, 17.43, 20.49, 11.95]
    gauss_vals = [5.18, 9.50, 13.37, 9.76, 19.23, 21.20, 12.94]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x - width/2, base_vals, width, label="Baseline (YOLO26n-P2)", color=C_BASE, edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, gauss_vals, width, label="GauSS (Ours)", color=C_GAUSS, edgecolor="black", linewidth=0.6)

    ax.set_ylabel(r"$AP_{50}$ (%)")
    ax.set_xlabel(r"Target Scale Interval $s = \sqrt{w \cdot h}$ (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 25.5)
    ax.legend(frameon=True, edgecolor="none", facecolor="#f8f9fa", loc="upper left")

    # Clean annotations on top with vertical offsets to prevent any collision
    for i in range(len(categories)):
        b_val = base_vals[i]
        g_val = gauss_vals[i]
        
        # When values are extremely close (e.g. tiny3: 13.5% vs 13.4%), stagger heights
        if abs(g_val - b_val) < 0.3:
            b_offset = 0.4
            g_offset = 1.6
        else:
            b_offset = 0.4
            g_offset = 0.4

        ax.text(x[i] - width/2, b_val + b_offset, f"{b_val:.1f}%", ha="center", va="bottom", fontsize=7.2, color=C_BASE)
        ax.text(x[i] + width/2, g_val + g_offset, f"{g_val:.1f}%", ha="center", va="bottom", fontsize=7.2, color=C_GAUSS, weight="bold")

    save_dual(fig, "01_tinyperson_ap50_by_size")


# 2. Miss Rate MR50 (Figure 5 in text)
def plot_mr50_comparison():
    categories = [
        "tiny1\n(2–8 px)",
        "tiny2\n(8–12 px)",
        "tiny3\n(12–20 px)",
        "tiny\n(2–20 px)",
        "small\n(20–32 px)",
        "reasonable\n(>32 px)",
        "all\n(≥2 px)"
    ]
    base_mr = [95.83, 95.10, 92.11, 95.90, 89.98, 89.24, 96.93]
    gauss_mr = [96.01, 93.52, 92.57, 95.87, 88.73, 89.09, 96.73]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x - width/2, base_mr, width, label="Baseline (YOLO26n-P2)", color=C_BASE, edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, gauss_mr, width, label="GauSS (Ours)", color=C_GAUSS, edgecolor="black", linewidth=0.6)

    # Standard miss rate notation: MR_50 (%) \downarrow (removing misleading "Log-Average")
    ax.set_ylabel(r"Miss Rate $MR_{50}$ (%) $\downarrow$")
    ax.set_xlabel(r"Target Scale Interval $s = \sqrt{w \cdot h}$ (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(85, 101.5)
    ax.legend(frameon=True, edgecolor="none", facecolor="#f8f9fa", loc="lower left")

    for i in range(len(categories)):
        b_val = base_mr[i]
        g_val = gauss_mr[i]
        # If very close (e.g. tiny1 95.8 vs 96.0 or all 96.9 vs 96.7), stagger height
        if abs(g_val - b_val) < 0.4:
            b_offset = 0.3
            g_offset = 1.3
        else:
            b_offset = 0.3
            g_offset = 0.3

        ax.text(x[i] - width/2, b_val + b_offset, f"{b_val:.1f}%", ha="center", va="bottom", fontsize=7.2, color=C_BASE)
        ax.text(x[i] + width/2, g_val + g_offset, f"{g_val:.1f}%", ha="center", va="bottom", fontsize=7.2, color=C_GAUSS, weight="bold")

    save_dual(fig, "02_tinyperson_miss_rate")


# 3. Reference Comparison on Ultra-Tiny (tiny1: 2-8 px) (Figure 6 in text)
def plot_international_tiny1_benchmark():
    # (model, tiny1 AP50, params M, protocol group)
    # group "a": original WACV-2020 benchmark numbers (Yu et al.)
    # group "b": GRSL-2022 two-stage zoo re-implementation numbers (Hong et al.)
    # group "c": ours (scratch-trained, tiled protocol)
    data = [
        ("FCOS", 3.39, 32.0, "a"),
        ("RetinaNet", 11.47, 34.0, "a"),
        ("Faster\nR-CNN", 29.21, 41.0, "a"),
        ("FCOS", 2.88, 32.0, "b"),
        ("RetinaNet", 12.24, 34.0, "b"),
        ("Swin-T", 31.92, None, "b"),
        ("R-CNN-FPN\nMSM", 33.79, 41.0, "b"),
        ("RetinaNet\n+SM", 33.90, 34.0, "b"),
        ("Cascade\n+SSPNet", 45.75, 66.0, "b"),
        ("Faster\n+SSPNet", 47.56, 41.0, "b"),
        ("YOLO26n-P2\n(Base)", 4.72, 2.52, "c"),
        ("GauSS\n(Ours)", 5.18, 2.55, "c"),
    ]
    models = [d[0] for d in data]
    ap_tiny1 = [d[1] for d in data]
    params = [d[2] for d in data]
    groups = [d[3] for d in data]

    group_facecolor = {"a": C_GRAY1, "b": "#9DB2C6", "c": C_GAUSS}
    colors = [group_facecolor[g] for g in groups]
    hatch = {"a": None, "b": "//", "c": None}

    x = np.arange(len(models))
    fig, ax1 = plt.subplots(figsize=(10.0, 3.4))

    bars = ax1.bar(
        x, ap_tiny1, width=0.62, color=colors,
        edgecolor=[("#1F3B57" if g == "b" else "black") for g in groups],
        linewidth=0.6,
        hatch=[hatch[g] for g in groups],
        label=r"$AP_{50}^{\mathrm{tiny1}}$ (2–8 px)",
    )
    ax1.set_ylabel(r"$AP_{50}^{\mathrm{tiny1}}$ (%)")
    ax1.set_ylim(0, 55)

    # Protocol group separators and labels
    starts = {"a": None, "b": None, "c": None}
    ends = {}
    for i, g in enumerate(groups):
        if starts[g] is None:
            starts[g] = i - 0.5
        ends[g] = i + 0.5
    labels = {
        "a": "Original WACV-2020 protocol",
        "b": "GRSL-2022 re-evaluated protocol",
        "c": "Ours (scratch, tiled)",
    }
    y_sep = ax1.get_ylim()[1]
    for g in ["a", "b", "c"]:
        ax1.axvline(ends[g], color="#B0B7BF", linewidth=0.8, linestyle=":")
        ax1.text((starts[g] + ends[g]) / 2, y_sep * 0.97, labels[g],
                 ha="center", va="top", fontsize=7.4, style="italic", color="#5A6570")

    ax2 = ax1.twinx()
    px = [i for i, p in enumerate(params) if p is not None]
    pv = [p for p in params if p is not None]
    ax2.plot([i + 0.0 for i in px], pv, color=C_GREEN, marker="s", markersize=5,
             linewidth=1.5, linestyle="--", label="Parameters (M)")
    ax2.set_ylabel("Parameters (Millions)", color=C_GREEN)
    ax2.set_ylim(0, 75)
    ax2.grid(False)

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=7.0)

    # Bar labels
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.8, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=6.8, weight="bold")

    # Point labels for params (ours annotated precisely; "None" params skipped)
    for i, p in enumerate(params):
        if p is None:
            continue
        offset = 1.6 if p > 30 else 1.4
        ax2.text(x[i], p + offset, f"{p:.2f}M" if p < 10 else f"{p:.1f}M",
                 ha="center", va="bottom", fontsize=6.6, color=C_GREEN, weight="bold")

    l1, b1 = ax1.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    # de-duplicate legend entries
    seen, handles, lab = set(), [], []
    for h, t in zip(l1 + l2, b1 + b2):
        if t not in seen:
            seen.add(t)
            handles.append(h)
            lab.append(t)
    ax1.legend(handles, lab, loc="upper left", frameon=True, edgecolor="none",
               facecolor="#f8f9fa", fontsize=7.4)

    save_dual(fig, "03_international_tiny1_tradeoff")


# 4. Hardware Efficiency (FPS & VRAM) (Figure 7 in text)
def plot_hardware_efficiency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    modes = ["Tiled (640×512)", "Full-frame (1280×1280)"]
    x = np.arange(len(modes))
    width = 0.32

    fps_base = [88.6, 84.4]
    fps_gauss = [65.8, 60.3]

    ax1.bar(x - width/2, fps_base, width, label="Baseline", color=C_BASE, edgecolor="black", linewidth=0.6)
    ax1.bar(x + width/2, fps_gauss, width, label="GauSS", color=C_GAUSS, edgecolor="black", linewidth=0.6)
    ax1.axhline(60, color="#7f8c8d", linestyle=":", linewidth=1.2, label="Real-time (60 FPS)")
    ax1.set_ylabel("Throughput (FPS)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right", frameon=True, edgecolor="none", facecolor="#f8f9fa", fontsize=7.5)

    for p in ax1.patches[:4]:
        h = p.get_height()
        ax1.text(p.get_x() + p.get_width()/2., h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, weight="bold")

    vram_base = [91.5, 244.7]
    vram_gauss = [122.1, 367.2]

    ax2.bar(x - width/2, vram_base, width, label="Baseline", color=C_BASE, edgecolor="black", linewidth=0.6)
    ax2.bar(x + width/2, vram_gauss, width, label="GauSS", color=C_GAUSS, edgecolor="black", linewidth=0.6)
    ax2.set_ylabel("Peak VRAM (MB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(modes)
    ax2.set_ylim(0, 450)
    ax2.legend(loc="upper left", frameon=True, edgecolor="none", facecolor="#f8f9fa", fontsize=7.5)

    for p in ax2.patches[:4]:
        h = p.get_height()
        ax2.text(p.get_x() + p.get_width()/2., h + 7.0, f"{h:.1f} MB", ha="center", va="bottom", fontsize=7.2, weight="bold")

    save_dual(fig, "04_hardware_efficiency")


# 5. VisDrone per-class (Figure 13 in text)
def plot_visdrone_per_class():
    classes = ["Pedestrian", "People", "Bicycle", "Car", "Van", "Truck", "Tricycle", "Awning-tricycle", "Bus", "Motor"]
    base_val = [16.76, 10.19, 1.35, 47.59, 17.53, 6.36, 7.03, 3.51, 15.62, 13.72]
    gauss_val = [16.92, 9.64, 1.00, 46.87, 15.73, 7.58, 6.45, 3.72, 15.67, 13.41]

    x = np.arange(len(classes))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x - width/2, base_val, width, label="Baseline (YOLO26n-P2)", color=C_BASE, edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, gauss_val, width, label="GauSS (Ours)", color=C_GAUSS, edgecolor="black", linewidth=0.6)

    ax.set_ylabel(r"$mAP@[0.50:0.95]$ (%)")
    ax.set_xlabel("Object Class (VisDrone-DET Validation)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim(0, 55)
    ax.legend(frameon=True, edgecolor="none", facecolor="#f8f9fa", loc="upper right")

    for i in range(len(classes)):
        top = max(base_val[i], gauss_val[i])
        diff = gauss_val[i] - base_val[i]
        if abs(diff) >= 0.5:
            ax.text(x[i] + width/2, top + 0.8, f"{gauss_val[i]:.1f}%", ha="center", va="bottom", fontsize=7.0, color=C_GAUSS, weight="bold")

    save_dual(fig, "05_visdrone_per_class_val")


# 6. Density-stratified recall (Figure 8 in text)
def plot_density_recall():
    densities = [
        "Sparse\n(<15 targets/img)\n[72.8% of split]",
        "Medium\n(15–50 targets/img)\n[16.9% of split]",
        "Crowded\n(>50 targets/img)\n[10.3% of split]"
    ]
    base_recall = [74.93, 77.71, 66.94]
    gauss_recall = [76.85, 76.87, 65.35]

    x = np.arange(len(densities))
    width = 0.36

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.bar(x - width/2, base_recall, width, label="Baseline", color=C_BASE, edgecolor="black", linewidth=0.6)
    ax.bar(x + width/2, gauss_recall, width, label="GauSS", color=C_GAUSS, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("True Positive Recall (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(densities)
    ax.set_ylim(55, 84)
    ax.legend(frameon=True, edgecolor="none", facecolor="#f8f9fa", loc="upper right")

    for i in range(len(densities)):
        ax.text(x[i] - width/2, base_recall[i] + 0.4, f"{base_recall[i]:.2f}%", ha="center", va="bottom", fontsize=7.5, color=C_BASE)
        ax.text(x[i] + width/2, gauss_recall[i] + 0.4, f"{gauss_recall[i]:.2f}%", ha="center", va="bottom", fontsize=7.5, color=C_GAUSS, weight="bold")

    save_dual(fig, "06_tinyperson_density_recall")


if __name__ == "__main__":
    print("Regenerating Academic Figures (1-6) without overlaps and in publication style...")
    plot_ap50_by_size_bin()
    plot_mr50_comparison()
    plot_international_tiny1_benchmark()
    plot_hardware_efficiency()
    plot_visdrone_per_class()
    plot_density_recall()
