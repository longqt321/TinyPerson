from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["svg.fonttype"] = "none"

out_dir = Path("paper/figures")
out_dir.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# HÌNH 1: SƠ ĐỒ CHI TIẾT BÊN TRONG MODULE GAUSS (PHONG CÁCH TRANSFORMER/NEURIPS)
# ==============================================================================
def draw_gauss_module_architecture():
    fig, ax = plt.subplots(figsize=(7, 9.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Màu sắc pastel chuẩn học thuật (theo ảnh mẫu bạn gửi)
    c_input = "#FFCCD5"  # Hồng pastel (Input)
    c_attn = "#FFE5D9"  # Cam đào pastel (Router Attention)
    c_norm = "#E2ECE9"  # Xanh ngọc nhạt (Norm / Scale)
    c_conv = "#D8E2DC"  # Xanh xám nhạt (Conv / Linear)
    c_op = "#D0F4DE"  # Xanh lục pastel (Softmax / Math)
    c_res = "#FDE2E4"  # Hồng kem (Residual Add)
    c_box_bg = "#F8F9FA"  # Xám rất nhạt (Container)

    # 1. Khung chứa lớn toàn bộ GauSS Module (Container)
    big_box = patches.FancyBboxPatch(
        (0.8, 1.8),
        8.4,
        10.5,
        boxstyle="round,pad=0.2,rounding_size=0.3",
        facecolor=c_box_bg,
        edgecolor="#212529",
        linewidth=1.8,
        linestyle="--",
    )
    ax.add_patch(big_box)
    ax.text(
        5.0,
        12.0,
        "GauSS: Gaussian Scale-Space Refinement Module",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        color="#212529",
    )

    # 2. Ba luồng Input ở dưới
    inputs = [
        (2.0, 0.8, r"$\mathbf{F}_{P2}^{\text{backbone}}$\n(C2: 64ch, Stride 4)", c_input),
        (5.0, 0.8, r"$\mathbf{F}_{P2}^{\text{neck}}$\n(Upsampled P3: 64ch)", c_input),
        (8.0, 0.8, r"$\mathbf{F}_{P3}^{\text{context}}$\n(Upsampled P4: 128ch)", c_input),
    ]
    for x, y, text, col in inputs:
        box = patches.FancyBboxPatch(
            (x - 1.2, y - 0.4),
            2.4,
            0.8,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=col,
            edgecolor="#212529",
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, weight="bold")

    # Mũi tên từ Input vào GauSS Container
    for x in [2.0, 5.0, 8.0]:
        ax.annotate(
            "",
            xy=(x, 2.2),
            xytext=(x, 1.2),
            arrowprops={"arrowstyle": "-|>", "color": "#212529", "lw": 1.6, "shrinkA": 0, "shrinkB": 0},
        )

    # 3. Khối 1x1 Conv cho Context Stream (Right stream)
    c_conv_box = patches.FancyBboxPatch(
        (6.8, 2.3),
        2.4,
        0.6,
        boxstyle="round,pad=0.1,rounding_size=0.1",
        facecolor=c_conv,
        edgecolor="#212529",
        linewidth=1.3,
    )
    ax.add_patch(c_conv_box)
    ax.text(8.0, 2.6, r"Conv $1\times 1$ (128 $\to$ 64)", ha="center", va="center", fontsize=8.5, weight="bold")

    # 4. Khối Router Attention (Trung tâm)
    router_box = patches.FancyBboxPatch(
        (2.2, 3.7),
        5.6,
        3.8,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor="#FFF1E6",
        edgecolor="#E76F51",
        linewidth=1.8,
    )
    ax.add_patch(router_box)
    ax.text(5.0, 7.15, "Dynamic Router Attention", ha="center", va="center", fontsize=11, weight="bold", color="#9D0208")

    # Chi tiết bên trong Router Attention
    r_sub = [
        (5.0, 6.4, r"Global Spatial Descriptor $\mathbf{S}_{\text{desc}} = \bigoplus \text{AvgPool}(\mathbf{F}_i)$", c_norm, 4.8, 0.5),
        (5.0, 5.5, r"Channel Bottleneck MLP: $\mathbf{W}_2 \cdot \text{SiLU}(\mathbf{W}_1 \cdot \mathbf{S})$", c_attn, 4.4, 0.5),
        (5.0, 4.5, r"Softmax Normalization $\to [\alpha, \beta, \gamma]$", c_op, 3.8, 0.5),
    ]
    for x, y, text, col, w, h in r_sub:
        b = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.1",
            facecolor=col,
            edgecolor="#212529",
            linewidth=1.2,
        )
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, weight="bold")

    # Mũi tên bên trong Router
    ax.annotate(
        "",
        xy=(5.0, 6.15),
        xytext=(5.0, 5.75),
        arrowprops={"arrowstyle": "-|>", "color": "#212529", "lw": 1.4},
    )
    ax.annotate(
        "",
        xy=(5.0, 5.25),
        xytext=(5.0, 4.75),
        arrowprops={"arrowstyle": "-|>", "color": "#212529", "lw": 1.4},
    )

    # Đường kết nối từ 3 luồng vào Router Attention
    ax.plot([2.0, 3.0], [3.2, 3.2], color="#212529", lw=1.4)
    ax.plot([3.0, 3.0], [3.2, 3.7], color="#212529", lw=1.4)
    ax.plot([8.0, 7.0], [3.2, 3.2], color="#212529", lw=1.4)
    ax.plot([7.0, 7.0], [3.2, 3.7], color="#212529", lw=1.4)
    ax.plot([5.0, 5.0], [2.2, 3.7], color="#212529", lw=1.4)

    # 5. Phép tổng hợp gia quyền (Adaptive Scale-Space Fusion)
    # Ba đường ra mang trọng số alpha, beta, gamma
    ax.annotate(
        r"$\alpha$",
        xy=(2.0, 8.2),
        xytext=(3.5, 4.5),
        arrowprops={"arrowstyle": "-|>", "color": "#9D0208", "lw": 1.5, "connectionstyle": "arc3,rad=-0.2"},
        fontsize=10,
        weight="bold",
        color="#9D0208",
    )
    ax.annotate(
        r"$\beta$",
        xy=(5.0, 8.2),
        xytext=(5.0, 4.25),
        arrowprops={"arrowstyle": "-|>", "color": "#9D0208", "lw": 1.5, "connectionstyle": "arc3,rad=0.0"},
        fontsize=10,
        weight="bold",
        color="#9D0208",
    )
    ax.annotate(
        r"$\gamma$",
        xy=(8.0, 8.2),
        xytext=(6.5, 4.5),
        arrowprops={"arrowstyle": "-|>", "color": "#9D0208", "lw": 1.5, "connectionstyle": "arc3,rad=0.2"},
        fontsize=10,
        weight="bold",
        color="#9D0208",
    )

    # Các khối nhân trọng số (Multiply)
    for x in [2.0, 5.0, 8.0]:
        circ = patches.Circle((x, 8.35), 0.22, facecolor="#F8EDEB", edgecolor="#212529", linewidth=1.4)
        ax.add_patch(circ)
        ax.text(x, 8.35, r"$\otimes$", ha="center", va="center", fontsize=11, weight="bold")

    # Đường dẫn thẳng lên khối nhân
    ax.plot([2.0, 2.0], [2.2, 8.13], color="#212529", lw=1.4)
    ax.plot([8.0, 8.0], [2.9, 8.13], color="#212529", lw=1.4)

    # Khối tổng hợp cộng phần tử
    add_circ = patches.Circle((5.0, 9.3), 0.28, facecolor="#D8F3DC", edgecolor="#212529", linewidth=1.5)
    ax.add_patch(add_circ)
    ax.text(5.0, 9.3, r"$\bigoplus$", ha="center", va="center", fontsize=13, weight="bold")

    # Nối 3 phép nhân vào phép cộng
    ax.plot([2.0, 2.0, 4.72], [8.57, 9.3, 9.3], color="#212529", lw=1.4)
    ax.plot([8.0, 8.0, 5.28], [8.57, 9.3, 9.3], color="#212529", lw=1.4)
    ax.annotate("", xy=(5.0, 9.02), xytext=(5.0, 8.57), arrowprops={"arrowstyle": "-|>", "color": "#212529", "lw": 1.4})

    # 6. Khối Depthwise Conv tinh chỉnh (DWConv 3x3)
    dw_box = patches.FancyBboxPatch(
        (2.8, 10.0),
        4.4,
        0.65,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=c_attn,
        edgecolor="#212529",
        linewidth=1.4,
    )
    ax.add_patch(dw_box)
    ax.text(
        5.0,
        10.32,
        r"Gaussian Depthwise Conv $3\times 3$ ($\sigma=1.0$)",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
    )
    ax.annotate("", xy=(5.0, 10.0), xytext=(5.0, 9.58), arrowprops={"arrowstyle": "-|>", "color": "#212529", "lw": 1.4})

    # 7. Nhánh Residual Add (Liên kết tắt từ Backbone P2)
    res_circ = patches.Circle((5.0, 11.2), 0.26, facecolor=c_res, edgecolor="#212529", linewidth=1.5)
    ax.add_patch(res_circ)
    ax.text(5.0, 11.2, r"$\bigoplus$", ha="center", va="center", fontsize=12, weight="bold")
    ax.annotate("", xy=(5.0, 10.94), xytext=(5.0, 10.65), arrowprops={"arrowstyle": "-|>", "color": "#212529", "lw": 1.4})

    # Đường Skip connection uốn vòng từ Backbone P2 (trái)
    ax.plot([1.2, 1.2], [2.2, 11.2], color="#D90429", lw=1.6, linestyle="--")
    ax.plot([2.0, 1.2], [2.2, 2.2], color="#D90429", lw=1.6, linestyle="--")
    ax.annotate(
        "",
        xy=(4.74, 11.2),
        xytext=(1.2, 11.2),
        arrowprops={"arrowstyle": "-|>", "color": "#D90429", "lw": 1.6, "linestyle": "--"},
    )
    ax.text(1.35, 7.5, "Identity Residual Link", rotation=90, fontsize=8, weight="bold", color="#D90429")

    # 8. Output của Module GauSS ở đỉnh
    out_box = patches.FancyBboxPatch(
        (3.0, 12.8),
        4.0,
        0.8,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor="#BEE1E6",
        edgecolor="#212529",
        linewidth=1.6,
    )
    ax.add_patch(out_box)
    ax.text(
        5.0,
        13.2,
        r"$\mathbf{P}_2^{\text{aligned}}$ Feature Map" + "\n(Refined Stride 4, 64 channels)",
        ha="center",
        va="center",
        fontsize=9,
        weight="bold",
    )
    ax.annotate("", xy=(5.0, 12.8), xytext=(5.0, 11.46), arrowprops={"arrowstyle": "-|>", "color": "#212529", "lw": 1.6})

    fig.savefig(out_dir / "10_gauss_module_architecture.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "10_gauss_module_architecture.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("✓ Đã xuất 10_gauss_module_architecture.pdf và .svg")


# ==============================================================================
# HÌNH 2: SƠ ĐỒ TOÀN CẢNH ĐƯỜNG ỐNG PHÁT HIỆN GAUSS-YOLO26N (OVERALL PIPELINE)
# ==============================================================================
def draw_overall_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Màu sắc
    c_img = "#E0E1DD"
    c_backbone = "#C8D6AF"
    c_neck = "#F3C68F"
    c_gauss = "#F28482"
    c_head = "#84A59D"

    # 1. Input Image Box
    b_img = patches.FancyBboxPatch(
        (0.5, 1.8),
        2.2,
        2.4,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=c_img,
        edgecolor="black",
        linewidth=1.3,
    )
    ax.add_patch(b_img)
    ax.text(
        1.6,
        3.0,
        "Input Image\n$1280 \\times 1280$\n(Tiny Person\n$2 \\to 20$ px)",
        ha="center",
        va="center",
        fontsize=5.9,
        weight="bold",
    )

    ax.annotate(
        "",
        xy=(3.3, 3.0),
        xytext=(2.7, 3.0),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.6},
    )

    # 2. CSP Backbone Hierarchy
    b_bb = patches.FancyBboxPatch(
        (3.3, 1.0),
        3.2,
        4.0,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor="#F7FFF7",
        edgecolor="#4F772D",
        linewidth=1.5,
    )
    ax.add_patch(b_bb)
    ax.text(4.9, 4.65, "CSP Backbone", ha="center", va="center", fontsize=4.3, weight="bold", color="#4F772D")

    bb_stages = [
        (4.9, 3.9, "C2 Stage (Stride 4, 64ch)"),
        (4.9, 3.0, "C3 Stage (Stride 8, 128ch)"),
        (4.9, 2.1, "C4 Stage (Stride 16, 128ch)"),
        (4.9, 1.3, "C5 Stage (Stride 32, 256ch)"),
    ]
    for x, y, name in bb_stages:
        box = patches.FancyBboxPatch(
            (x - 1.4, y - 0.25),
            2.8,
            0.5,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=c_backbone,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=5.0, weight="semibold")

    # 3. PANet Multi-Scale Neck
    b_neck = patches.FancyBboxPatch(
        (7.2, 1.0),
        3.4,
        4.0,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor="#FFFDF7",
        edgecolor="#D4A373",
        linewidth=1.5,
    )
    ax.add_patch(b_neck)
    ax.text(8.9, 4.65, "PANet Feature Pyramid", ha="center", va="center", fontsize=4.3, weight="bold", color="#B07D62")

    neck_stages = [
        (8.9, 3.9, "P2 Stage (32ch)"),
        (8.9, 3.0, "P3 Stage (64ch)"),
        (8.9, 2.1, "P4 Stage (128ch)"),
        (8.9, 1.3, "P5 Stage (256ch)"),
    ]
    for x, y, name in neck_stages:
        box = patches.FancyBboxPatch(
            (x - 1.4, y - 0.25),
            2.8,
            0.5,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=c_neck,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=5.0, weight="semibold")

    # Mũi tên từ Backbone sang Neck
    ax.annotate(
        "",
        xy=(7.5, 3.0),
        xytext=(6.3, 3.0),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
    )
    ax.annotate(
        "",
        xy=(7.5, 2.1),
        xytext=(6.3, 2.1),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
    )
    ax.annotate(
        "",
        xy=(7.5, 1.3),
        xytext=(6.3, 1.3),
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
    )

    # 4. KHỐI ĐỘT PHÁ: GAUSS MODULE
    b_g = patches.FancyBboxPatch(
        (11.2, 3.3),
        2.2,
        1.6,
        boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=c_gauss,
        edgecolor="#9D0208",
        linewidth=2.0,
    )
    ax.add_patch(b_g)
    ax.text(
        12.3,
        4.1,
        "GauSS Module\n(Ours)",
        ha="center",
        va="center",
        fontsize=6.8,
        weight="bold",
        color="white",
    )
    ax.text(
        12.3,
        3.55,
        "Scale-Space Align\n+ Router Attention",
        ha="center",
        va="center",
        fontsize=5.0,
        weight="semibold",
        color="white",
    )

    # 3 luồng nối vào GauSS
    ax.annotate(
        "",
        xy=(11.2, 4.4),
        xytext=(6.3, 3.9),
        arrowprops={"arrowstyle": "-|>", "color": "#D90429", "lw": 1.5, "connectionstyle": "arc3,rad=-0.15"},
    )
    ax.annotate(
        "",
        xy=(11.2, 4.0),
        xytext=(10.3, 3.9),
        arrowprops={"arrowstyle": "-|>", "color": "#D90429", "lw": 1.5},
    )
    ax.annotate(
        "",
        xy=(11.2, 3.6),
        xytext=(10.3, 3.0),
        arrowprops={"arrowstyle": "-|>", "color": "#D90429", "lw": 1.5, "connectionstyle": "arc3,rad=-0.1"},
    )

    # 5. Anchor-Free Detect Head
    b_head = patches.FancyBboxPatch(
        (14.0, 1.0),
        1.7,
        4.0,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor="#F0EFEB",
        edgecolor="#283618",
        linewidth=1.5,
    )
    ax.add_patch(b_head)
    ax.text(14.85, 4.65, "Detect Head", ha="center", va="center", fontsize=6.1, weight="bold", color="#283618")

    heads = [
        (14.85, 3.9, "Head P2\n(Tiny Object)"),
        (14.85, 3.0, "Head P3"),
        (14.85, 2.1, "Head P4"),
        (14.85, 1.3, "Head P5"),
    ]
    for x, y, name in heads:
        box = patches.FancyBboxPatch(
            (x - 0.75, y - 0.28),
            1.5,
            0.56,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=c_head,
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(
            x,
            y,
            name,
            ha="center",
            va="center",
            fontsize=4.7,
            weight="bold" if "P2" in name else "normal",
            color="white" if "P2" in name else "black",
        )

    # Mũi tên từ GauSS ra Head P2
    ax.annotate(
        "",
        xy=(14.1, 3.9),
        xytext=(13.4, 4.1),
        arrowprops={"arrowstyle": "-|>", "color": "#D90429", "lw": 2.0},
    )
    # Mũi tên từ Neck ra P3, P4, P5
    for y in [3.0, 2.1, 1.3]:
        ax.annotate(
            "",
            xy=(14.1, y),
            xytext=(10.3, y),
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.2},
        )

    fig.savefig(out_dir / "11_overall_gauss_pipeline.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "11_overall_gauss_pipeline.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    print("✓ Đã xuất 11_overall_gauss_pipeline.pdf và .svg")


if __name__ == "__main__":
    draw_gauss_module_architecture()
    draw_overall_pipeline()
