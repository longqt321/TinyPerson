#!/usr/bin/env python3
"""
Generate exact GauSS Architecture Diagram conforming to equations in Section 3:
- Inputs: C2 (stride 4), P3 (stride 8), P4 (stride 16)
- 1x1 Projections to d=32: S2, S3, S4
- Base fusion: X0 = SiLU(BN(S2 + Interp(S3)))
- Separable Gaussian Filters: L1 (sigma_1=0.65), L2 (sigma_2=1.45)
- Band Decomposition: B_high, B_mid, B_low
- Sub-pixel Encoders: PixelUnshuffle(2) + DWConv -> R_high, R_mid, R_low
- Dilated Context: C0 -> DWConv (r=1,2,3) + Softmax gate -> C_ctx
- Spatial Router: U -> DWConv3x3 -> Conv1x1 -> a_bands (3 ch), m_gate (1 ch)
- Band Synthesis: F_band = sum(a_i R_i) + gamma_h R_high
- Saliency Gate: M_sal -> R_low^att = R_low * (1 - 0.5 M_sal)
- Dual-path mixing: m_gate * Branch_3x3 + (1-m_gate) * Branch_5x5 -> F_route
- Reconstruction: Conv1x1 -> PixelShuffle(2) -> X0 + alpha * F_tilde -> P2_aligned
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

out_dir_paper = Path("paper/figures")
out_dir_reports = Path("reports/figures")

def draw_exact_gauss_diagram():
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    # Elegant CVPR/WACV pastel color palette
    c_inp = "#E8EEF5"     # Soft Steel Blue (Inputs)
    c_proj = "#D4E6F1"    # Light Ice Blue (Projections)
    c_gauss = "#FCF3CF"   # Warm Pale Gold (Gaussian Filter)
    c_band = "#EDBB99"    # Warm Terracotta (Bands)
    c_subp = "#D5F5E3"    # Mint Green (Sub-pixel encoder)
    c_ctx = "#E8DAEF"     # Soft Lilac (Dilated Context)
    c_router = "#FADBD8"  # Soft Rose (Spatial Router)
    c_synth = "#FDEDEC"   # Rose White (Synthesis)
    c_out = "#D1F2EB"     # Teal Mist (Reconstruction & Output)
    c_edge = "#2C3E50"    # Deep Charcoal Slate for borders

    def draw_box(x, y, w, h, text, bg, border=c_edge, lw=1.0, fs=7.8, bold=False):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.6",
            facecolor=bg, edgecolor=border, linewidth=lw, zorder=2
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#1A252F", weight=weight, zorder=3)
        return box

    def draw_arrow(x1, y1, x2, y2, color=c_edge, lw=1.1, style="-|>"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops={"arrowstyle": style, "color": color, "lw": lw, "shrinkA": 0, "shrinkB": 0},
            zorder=4,
        )

    # -------------------------------------------------------------
    # 0. Outer GauSS Module Container
    # -------------------------------------------------------------
    outer_box = patches.FancyBboxPatch(
        (0.5, 0.8), 99.0, 48.4,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor="#FAFAFA", edgecolor="#BDC3C7", linewidth=1.2, linestyle="--", zorder=1
    )
    ax.add_patch(outer_box)
    ax.text(2.0, 47.8, r"$\mathbf{GauSS}$: Gaussian Scale-Space Alignment and Refinement Architecture",
            ha="left", va="center", fontsize=10.5, weight="bold", color="#2C3E50")

    # -------------------------------------------------------------
    # 1. Inputs (Left)
    # -------------------------------------------------------------
    draw_box(1.5, 36.5, 10.0, 5.0, r"$\mathbf{C}_2$ (Backbone)" + "\n" + r"$64 \times \frac{H}{4} \times \frac{W}{4}$", c_inp, bold=True)
    draw_box(1.5, 23.5, 10.0, 5.0, r"$\mathbf{P}_3$ (Lateral Neck)" + "\n" + r"$64 \times \frac{H}{8} \times \frac{W}{8}$", c_inp, bold=True)
    draw_box(1.5, 5.5, 10.0, 5.0, r"$\mathbf{P}_4$ (Deep Context)" + "\n" + r"$128 \times \frac{H}{16} \times \frac{W}{16}$", c_inp, bold=True)

    # Projections Conv 1x1 -> d=32
    draw_arrow(11.5, 39.0, 14.5, 39.0)
    draw_box(14.5, 37.0, 7.5, 4.0, r"$\text{Conv}_{1\times 1}$" + "\n" + r"$\to \mathbf{S}_2$ ($d$)", c_proj)

    draw_arrow(11.5, 26.0, 14.5, 26.0)
    draw_box(14.5, 24.0, 7.5, 4.0, r"$\text{Conv}_{1\times 1}$" + "\n" + r"$\to \mathbf{S}_3$ ($d$)", c_proj)

    draw_arrow(11.5, 8.0, 14.5, 8.0)
    draw_box(14.5, 6.0, 7.5, 4.0, r"$\text{Conv}_{1\times 1}$" + "\n" + r"$\to \mathbf{S}_4$ ($d$)", c_proj)

    # -------------------------------------------------------------
    # 2. Base Fused Representation X0
    # -------------------------------------------------------------
    # S3 upsample to stride 4
    draw_arrow(22.0, 26.0, 24.5, 26.0)
    draw_box(24.5, 24.5, 6.5, 3.2, r"$\text{Interp}(\cdot)$" + "\n(Stride 4)", c_proj, fs=7.0)

    # Sum S2 + Interp(S3)
    ax.plot([22.0, 29.5], [39.0, 39.0], color=c_edge, lw=1.1)
    ax.plot([29.5, 29.5], [39.0, 35.5], color=c_edge, lw=1.1)
    draw_arrow(31.0, 26.0, 33.5, 26.0)
    ax.plot([33.5, 33.5], [26.0, 33.5], color=c_edge, lw=1.1)

    # Base feature node
    draw_box(27.0, 32.0, 11.0, 4.0, r"$\mathbf{X}_0 = \text{SiLU}(\text{BN}(\mathbf{S}_2 + \mathbf{S}_3^{\uparrow}))$" + "\n" + r"$\mathbb{R}^{d \times \frac{H}{4} \times \frac{W}{4}}$", c_gauss, bold=True)

    # Shortcut connection from X0 to final reconstruction output
    ax.plot([32.5, 32.5], [36.0, 44.5], color="#7F8C8D", lw=1.2, linestyle=":")
    ax.plot([32.5, 91.5], [44.5, 44.5], color="#7F8C8D", lw=1.2, linestyle=":")
    draw_arrow(91.5, 44.5, 91.5, 32.0, color="#7F8C8D", lw=1.2)
    ax.text(62.0, 45.2, r"Identity Shortcut Connection ($\mathbf{X}_0$)", fontsize=7.5, color="#5D6D7E", ha="center")

    # -------------------------------------------------------------
    # 3. Learnable Separable Gaussian Scale-Space Filtering
    # -------------------------------------------------------------
    draw_arrow(38.0, 34.0, 41.0, 34.0)
    draw_box(41.0, 31.5, 11.5, 5.0, "Learnable Gaussian\nFilters " + r"$(\mathbf{L}_1, \mathbf{L}_2)$" + "\n" + r"$\sigma_1 \approx 0.65, \, \sigma_2 \approx 1.45$", c_gauss)

    # Band Decomposition: B_high, B_mid, B_low
    draw_arrow(52.5, 35.5, 55.0, 38.0)
    draw_box(55.0, 36.5, 8.5, 3.2, r"$\mathbf{B}_{\text{high}} = \mathbf{X}_0 - \mathbf{L}_1$", c_band, fs=7.2)

    draw_arrow(52.5, 34.0, 55.0, 34.0)
    draw_box(55.0, 32.5, 8.5, 3.2, r"$\mathbf{B}_{\text{mid}} = \mathbf{L}_1 - \mathbf{L}_2$", c_band, fs=7.2)

    draw_arrow(52.5, 32.5, 55.0, 30.0)
    draw_box(55.0, 28.5, 8.5, 3.2, r"$\mathbf{B}_{\text{low}} = \mathbf{L}_2$", c_band, fs=7.2)

    # -------------------------------------------------------------
    # 4. Sub-pixel Encoders (PixelUnshuffle + DWConv)
    # -------------------------------------------------------------
    draw_arrow(63.5, 38.1, 66.0, 38.1)
    draw_box(66.0, 36.5, 8.5, 3.2, r"$\text{PixelUnshuf}(2) \to \mathbf{R}_{\text{high}}$", c_subp, fs=7.0)

    draw_arrow(63.5, 34.1, 66.0, 34.1)
    draw_box(66.0, 32.5, 8.5, 3.2, r"$\text{PixelUnshuf}(2) \to \mathbf{R}_{\text{mid}}$", c_subp, fs=7.0)

    draw_arrow(63.5, 30.1, 66.0, 30.1)
    draw_box(66.0, 28.5, 8.5, 3.2, r"$\text{PixelUnshuf}(2) \to \mathbf{R}_{\text{low}}$", c_subp, fs=7.0)

    # -------------------------------------------------------------
    # 5. Multi-Scale Dilated Context Aggregation (Bottom branch)
    # -------------------------------------------------------------
    # S4 upsample
    draw_arrow(22.0, 8.0, 24.5, 8.0)
    draw_box(24.5, 6.5, 6.5, 3.2, r"$\text{Interp}(\cdot)$" + "\n(Stride 8)", c_proj, fs=7.0)

    # C0 = SiLU(BN(S3 + Interp(S4)))
    ax.plot([18.25, 18.25], [24.0, 14.5], color=c_edge, lw=1.1)
    ax.plot([18.25, 33.0], [14.5, 14.5], color=c_edge, lw=1.1)
    ax.plot([31.0, 33.0], [8.0, 13.0], color=c_edge, lw=1.1)

    draw_box(33.0, 11.5, 11.0, 4.0, r"$\mathbf{C}_0 = \text{SiLU}(\text{BN}(\mathbf{S}_3 + \mathbf{S}_4^{\uparrow}))$" + "\n(Context Trunk)", c_ctx, bold=True)

    draw_arrow(44.0, 13.5, 47.0, 13.5)
    draw_box(47.0, 10.5, 13.5, 6.0, "Dilated Depthwise Convs\n" + r"$r_i \in \{1, 2, 3\}$ + Gate $\boldsymbol{\omega}$" + "\n" + r"$\to \mathbf{C}_{\text{ctx}} \in \mathbb{R}^{d \times \frac{H}{8} \times \frac{W}{8}}$", c_ctx)

    # -------------------------------------------------------------
    # 6. Spatial Router & Fusion
    # -------------------------------------------------------------
    # Confluence of R_high, R_mid, R_low, C_ctx
    ax.plot([74.5, 77.0], [38.1, 34.0], color=c_edge, lw=1.1)
    ax.plot([74.5, 77.0], [34.1, 34.0], color=c_edge, lw=1.1)
    ax.plot([74.5, 77.0], [30.1, 34.0], color=c_edge, lw=1.1)

    ax.plot([60.5, 77.0], [13.5, 13.5], color=c_edge, lw=1.1)
    ax.plot([77.0, 77.0], [13.5, 23.0], color=c_edge, lw=1.1)

    draw_box(75.0, 23.0, 11.0, 5.0, "Spatial Router Attention\n" + r"$\mathbf{Z} = \text{Conv}(\text{DWConv}(\mathbf{U}))$" + "\n" + r"$\mathbf{a}_{\text{bands}} \in \mathbb{R}^3, \, m_{\text{gate}} \in \mathbb{R}^1$", c_router, bold=True)

    # -------------------------------------------------------------
    # 7. Dynamic Synthesis & Wave Clutter Suppression
    # -------------------------------------------------------------
    # Synthesis block
    draw_arrow(80.5, 23.0, 80.5, 19.5)
    draw_box(73.5, 14.0, 14.5, 5.5, "Band Synthesis & Clutter Gate\n" + r"$\mathbf{F}_{\text{band}} = \sum a_i \mathbf{R}_i + \gamma_h \mathbf{R}_{\text{high}}$" + "\n" + r"$\mathbf{R}_{\text{low}}^{\text{att}} = \mathbf{R}_{\text{low}} \odot (1 - 0.5 \mathbf{M}_{\text{sal}})$", c_synth, fs=7.2)

    # Dual path mixing
    draw_arrow(80.5, 14.0, 80.5, 10.5)
    draw_box(72.0, 4.5, 17.5, 6.0, "Dual-Branch Structural Routing\n" + r"$\mathbf{F}_{\text{route}} = m_{\text{gate}} \cdot \text{Branch}_{3\times 3}(\mathbf{F}_{\text{band}} + \mathbf{C}_{\text{ctx}})$" + "\n" + r"$+ (1 - m_{\text{gate}}) \cdot \text{Branch}_{5\times 5}(\mathbf{R}_{\text{low}}^{\text{att}} + \mathbf{C}_{\text{ctx}})$", c_synth, fs=7.1, bold=True)

    # -------------------------------------------------------------
    # 8. Reconstruction & Stride-4 Output (Right)
    # -------------------------------------------------------------
    draw_arrow(89.5, 7.5, 91.5, 7.5)
    ax.plot([91.5, 91.5], [7.5, 21.0], color=c_edge, lw=1.1)
    draw_arrow(91.5, 21.0, 91.5, 22.0)

    draw_box(86.5, 22.0, 10.0, 4.2, r"$\text{PixelShuffle}(2)$" + "\n" + r"$\text{Conv}_{1\times 1}^{\to 4d} \to \widetilde{\mathbf{F}}$" + "\n(Stride 4 Expansion)", c_out, fs=7.0)

    # Addition with X0
    draw_arrow(91.5, 26.2, 91.5, 28.0)
    draw_box(87.0, 28.0, 9.0, 4.0, r"$\bigoplus$ Residual" + "\n" + r"$\alpha \cdot \widetilde{\mathbf{F}} + \mathbf{X}_0$", c_out, bold=True)

    # Output node
    draw_arrow(91.5, 32.0, 91.5, 34.5)
    draw_box(85.5, 34.5, 12.0, 4.8, r"$\mathbf{P}_2^{\text{aligned}} = \text{SiLU}(\mathbf{X}_0 + \alpha \widetilde{\mathbf{F}})$" + "\n" + r"$64 \times \frac{H}{4} \times \frac{W}{4}$" + "\n(To P2 Detection Head)", "#A3E4D7", border="#117A65", lw=1.4, bold=True)

    plt.tight_layout()
    fig.savefig(out_dir_paper / "10_gauss_router_architecture.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir_paper / "10_gauss_router_architecture.svg", format="svg", bbox_inches="tight")
    fig.savefig(out_dir_reports / "10_gauss_router_architecture.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir_reports / "10_gauss_router_architecture.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("✓ Successfully generated exact 10_gauss_router_architecture (PDF & SVG)")

if __name__ == "__main__":
    draw_exact_gauss_diagram()
