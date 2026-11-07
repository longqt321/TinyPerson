from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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

C_BASE = "#2C5D88"
C_GAUSS = "#C0392B"
C_GREEN = "#27AE60"

def save_dual(fig, stem):
    # tight_layout keeps twin-axis labels and legends from colliding; bbox_inches
    # alone does not reflow the axes grid.
    try:
        fig.tight_layout()
    except UserWarning:
        pass
    svg_path = out_dir / f"{stem}.svg"
    pdf_path = paper_dir / f"{stem}.pdf"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Exported {stem} to SVG & PDF")


def plot_training_loss_curves():
    df_base = pd.read_csv("reports/data/baseline_results.csv")
    df_base.columns = [c.strip() for c in df_base.columns]

    df_dcaf = pd.read_csv("reports/data/dcaf_results.csv")
    df_dcaf.columns = [c.strip() for c in df_dcaf.columns]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    metrics = [
        ("train/box_loss", "Bounding Box Loss", axes[0]),
        ("train/cls_loss", "Classification Loss", axes[1]),
        ("train/l1_loss", "Regression L1 Loss", axes[2]),
    ]

    for col, title, ax in metrics:
        ax.plot(
            df_base["epoch"],
            df_base[col],
            label="Baseline (YOLO26n-P2)",
            color=C_BASE,
            linewidth=1.4,
            alpha=0.85,
        )
        ax.plot(
            df_dcaf["epoch"],
            df_dcaf[col],
            label="GauSS (Ours)",
            color=C_GAUSS,
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
        )

        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Epoch", fontsize=8.5)
        ax.set_ylabel("Loss Value", fontsize=8.5)
        ax.legend(frameon=True, edgecolor="none", facecolor="#f8f9fa", loc="upper right", fontsize=7.2)

    save_dual(fig, "07_training_loss_convergence")


def plot_total_loss_and_lr():
    df_base = pd.read_csv("reports/data/baseline_results.csv")
    df_base.columns = [c.strip() for c in df_base.columns]

    df_dcaf = pd.read_csv("reports/data/dcaf_results.csv")
    df_dcaf.columns = [c.strip() for c in df_dcaf.columns]

    df_base["total_loss"] = df_base["train/box_loss"] + df_base["train/cls_loss"]
    df_dcaf["total_loss"] = df_dcaf["train/box_loss"] + df_dcaf["train/cls_loss"]

    fig, ax1 = plt.subplots(figsize=(5.6, 2.8))

    ax1.plot(
        df_base["epoch"],
        df_base["total_loss"],
        label="Total Loss (Baseline)",
        color=C_BASE,
        linewidth=1.6,
    )
    ax1.plot(
        df_dcaf["epoch"],
        df_dcaf["total_loss"],
        label="Total Loss (GauSS)",
        color=C_GAUSS,
        linewidth=1.6,
        linestyle="-.",
    )

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Total Loss (Box + Cls)")
    ax1.set_ylim(4, 11)

    ax2 = ax1.twinx()
    ax2.plot(
        df_base["epoch"],
        df_base["lr/pg0"],
        label="Learning Rate (Cosine)",
        color=C_GREEN,
        linewidth=1.3,
        linestyle=":",
    )
    ax2.set_ylabel("Learning Rate", color=C_GREEN)
    ax2.set_ylim(0, 0.0025)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, edgecolor="none", facecolor="#f8f9fa", fontsize=7.2)

    save_dual(fig, "08_training_total_loss_lr")


def plot_pareto_frontier():
    # FPS values are per-tile throughput (single 640x512 crop), which is the only
    # throughput comparable across all detectors here. Tiled evaluation of a full
    # TinyPerson frame runs 23.4 crops, so end-to-end throughput is much lower;
    # that is stated explicitly in the axis label and in the GauSS annotation.
    models_data = [
        {"Model": "FCOS (Yu et al.)", "Params": 32.0, "AP_tiny": 16.90, "FPS": 20.8, "Type": "Heavy"},
        {"Model": "RetinaNet (Yu et al.)", "Params": 34.0, "AP_tiny": 30.82, "FPS": 18.2, "Type": "Heavy"},
        {"Model": "Faster R-CNN (Yu et al.)", "Params": 41.0, "AP_tiny": 43.55, "FPS": 15.4, "Type": "Heavy"},
        {"Model": "YOLO26n-P2 (Base)", "Params": 2.52, "AP_tiny": 9.07, "FPS": 88.6, "Type": "Ours"},
        {"Model": "GauSS (Ours)", "Params": 2.55, "AP_tiny": 9.76, "FPS": 65.8, "Type": "Ours"},
    ]
    df = pd.DataFrame(models_data)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    colors = {"Heavy": "#7F8C8D", "Ours": C_GAUSS}
    markers = {"Heavy": "s", "Ours": "*"}

    for _, row in df.iterrows():
        is_ours = row["Type"] == "Ours"
        color = C_BASE if row["Model"].startswith("YOLO26n-P2") else colors[row["Type"]]
        ax.scatter(
            row["FPS"],
            row["AP_tiny"],
            s=220 if is_ours else 80,
            color=color,
            marker=markers[row["Type"]],
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

    # Individual hand-tuned annotation positions to avoid overlapping any points or text
    # FCOS: 20.8 FPS, 16.90%
    ax.annotate("FCOS (Yu et al.)\n(20.8 FPS, 16.9%)", xy=(20.8, 16.9), xytext=(22, 18.5), fontsize=7.2)
    # RetinaNet: 18.2 FPS, 30.82%
    ax.annotate("RetinaNet (Yu et al.)\n(18.2 FPS, 30.8%)", xy=(18.2, 30.82), xytext=(20, 31.5), fontsize=7.2)
    # Faster R-CNN: 15.4 FPS, 43.55%
    ax.annotate("Faster R-CNN (Yu et al.)\n(15.4 FPS, 43.6%)", xy=(15.4, 43.55), xytext=(17, 44.0), fontsize=7.2)
    # YOLO26n-P2 (Base): 88.6 FPS, 9.07%
    ax.annotate("YOLO26n-P2 (Base)\n(88.6 FPS, 9.1%)", xy=(88.6, 9.07), xytext=(65, 4.0),
                fontsize=7.2, color=C_BASE, weight="bold",
                arrowprops={"arrowstyle": "->", "color": C_BASE, "lw": 0.8})
    # GauSS (Ours): 65.8 FPS per tile, 9.76% AP_tiny (tiled inference)
    ax.annotate("GauSS (Ours)\n(65.8 FPS/tile, 9.8%)\n7.3 FPS end-to-end", xy=(65.8, 9.76), xytext=(38, 14.5),
                fontsize=7.2, color=C_GAUSS, weight="bold",
                arrowprops={"arrowstyle": "->", "color": C_GAUSS, "lw": 0.8})

    ax.axvline(60, color="#7F8C8D", linestyle=":", linewidth=1.2, label="Real-time (60 FPS)")
    # Removed \rightarrow arrows from labels per feedback
    ax.set_xlabel("Per-tile Inference Throughput (FPS)")
    ax.set_ylabel(r"$AP_{50}^{\mathrm{tiny}}$ (%)")
    ax.set_xlim(10, 110)
    ax.set_ylim(0, 50)
    ax.legend(loc="upper right", frameon=True, edgecolor="none", facecolor="#f8f9fa", fontsize=7.5)

    save_dual(fig, "09_pareto_frontier_tradeoff")


if __name__ == "__main__":
    print("Regenerating Academic Figures (7-9)...")
    plot_training_loss_curves()
    plot_total_loss_and_lr()
    plot_pareto_frontier()
