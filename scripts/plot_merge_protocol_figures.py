"""Figure 19: merge-protocol scale-bin breakdown (baseline vs GauSS, 150e tiled).

Numbers are read directly from reports/eval/{baseline,gauss_v3}_tinyperson_wacv_150e_tiled.json.
The figure is authored at the final print width (3.5 in, one WACV column) so the
text stays at true size; AP50 is drawn solid and AP25 hatched.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BINS = ["all", "tiny", "tiny1", "tiny2", "tiny3", "small", "reasonable"]
LABELS = {
    "all": "all",
    "tiny": "tiny",
    "tiny1": "t1",
    "tiny2": "t2",
    "tiny3": "t3",
    "small": "small",
    "reasonable": ">32",
}

C_BASE = "#8A97A8"
C_GAUSS = "#E8834A"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load(model: str) -> dict[str, dict[str, float]]:
    path = ROOT / "reports" / "eval" / f"{model}_tinyperson_wacv_150e_tiled.json"
    data = json.loads(path.read_text())["benchmark"]
    return {b: data[b] for b in BINS}


def main() -> None:
    base = load("baseline")
    gauss = load("gauss_v3")

    x = np.arange(len(BINS))
    w = 0.20

    fig, ax = plt.subplots(figsize=(3.15, 1.95))

    series = (
        (base, "AP@0.50", -1.5 * w, C_BASE, None, "Baseline"),
        (gauss, "AP@0.50", -0.5 * w, C_GAUSS, None, "GauSS"),
        (base, "AP@0.25", 0.5 * w, C_BASE, "////", None),
        (gauss, "AP@0.25", 1.5 * w, C_GAUSS, "////", None),
    )
    for source, metric, offset, color, hatch, label in series:
        values = [source[k][metric] * 100 for k in BINS]
        ax.bar(
            x + offset,
            values,
            w,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            hatch=hatch,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in BINS])
    ax.set_ylabel(r"$AP$ (%)")
    ax.set_ylim(0, 62)
    ax.set_yticks([0, 20, 40, 60])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.4)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=C_BASE, edgecolor="black", linewidth=0.35, label="Baseline"),
        Patch(facecolor=C_GAUSS, edgecolor="black", linewidth=0.35, label="GauSS"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.35, hatch="////", label=r"$AP_{25}$"),
        Patch(facecolor="#555555", edgecolor="black", linewidth=0.35, label=r"$AP_{50}$"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=0.9,
        bbox_to_anchor=(0.5, 1.13),
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "19_merge_protocol_scale_bins.pdf", bbox_inches="tight")
    fig.savefig(OUT / "19_merge_protocol_scale_bins.svg", bbox_inches="tight")
    print("wrote", OUT / "19_merge_protocol_scale_bins.pdf")


if __name__ == "__main__":
    main()
