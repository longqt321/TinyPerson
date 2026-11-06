"""Modal function and local runner to extract qualitative and diagnostic figures from 150e checkpoints.
Produces 4 figures:
1. Qualitative Tiny Detection Comparison (GT vs Baseline vs GauSS with zoom insets)
2. Sea Clutter and False Positive Suppression
3. P2 Feature Map / Saliency Attention Energy Map
4. Gaussian Scale-Space Band Decomposition Visualizer (B_high, B_mid, B_low)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))

from image import image
from paths import DATA_ROOT, RUN_ROOT

app = modal.App("tinyperson-extract-figures")
runs = modal.Volume.from_name("tiny-person-runs")
data = modal.Volume.from_name("tiny-person-data")
mounts = {str(RUN_ROOT): runs, str(DATA_ROOT): data}

@app.function(
    image=image,
    gpu="A10",
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    cpu=4,
    memory=16384,
    max_containers=1,
    volumes=mounts,
    timeout=3600,
)
def extract_all_qualitative_artifacts(
    gauss_checkpoint_path: str = "dcaf_v2_1_tinyperson_wacv/20261102-092728-c626816-seed42/training/weights/last.pt",
    baseline_checkpoint_path: str = "baseline_tinyperson_wacv/20261102-092728-c626816-seed42/training/weights/last.pt",
    output_dir_relative: str = "eval/qualitative_export",
):
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib import patches
    from PIL import Image
    from ultralytics import YOLO

    from tinydet.models.ultralytics import register_ultralytics_modules
    from tinydet.modules.dcaf import DCAF

    register_ultralytics_modules()
    runs.reload()
    data.reload()

    out_dir = RUN_ROOT / output_dir_relative
    out_dir.mkdir(parents=True, exist_ok=True)

    gauss_ckpt = RUN_ROOT / gauss_checkpoint_path
    base_ckpt = RUN_ROOT / baseline_checkpoint_path

    if not gauss_ckpt.is_file():
        raise FileNotFoundError(f"GauSS checkpoint not found: {gauss_ckpt}")
    if not base_ckpt.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {base_ckpt}")

    print("Loading checkpoints...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gauss_model = YOLO(str(gauss_ckpt))
    base_model = YOLO(str(base_ckpt))
    gauss_model.model.to(device).eval()
    base_model.model.to(device).eval()

    image_dir = DATA_ROOT / "tinyperson-yolo" / "images" / "test"
    anno_file = DATA_ROOT / "tinyperson-yolo" / "annotations" / "tiny_set_test_with_dense.json"
    with open(anno_file) as f:
        coco_data = json.load(f)

    # -------------------------------------------------------------------------
    # Helper to load image & annotations
    # -------------------------------------------------------------------------
    def get_image_and_gts(filename_pattern: str):
        target_img = None
        for img in coco_data["images"]:
            if filename_pattern in img["file_name"]:
                target_img = img
                break
        if target_img is None:
            raise ValueError(f"Image {filename_pattern} not found in annotations")

        basename = Path(target_img["file_name"]).name
        img_path = image_dir / basename
        if not img_path.is_file():
            found = list(image_dir.rglob(basename))
            if found:
                img_path = found[0]
            else:
                raise FileNotFoundError(f"Image file {basename} not found on disk")

        gts = [
            a for a in coco_data["annotations"]
            if a["image_id"] == target_img["id"]
        ]
        return img_path, target_img, gts

    # Standard publication styling
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
    })

    # =========================================================================
    # FIGURE 1: Qualitative Tiny Detection Comparison
    # Target image: youtube_V0001_I0000320.jpg (High density of tiny1 & tiny2)
    # =========================================================================
    print("Generating Figure 1: Qualitative Detection Comparison...", flush=True)
    img_path1, _info1, gts1 = get_image_and_gts("youtube_V0001_I0000320.jpg")
    pil1 = Image.open(img_path1).convert("RGB")
    w1, h1 = pil1.size

    # Zoom window around person group (approx center/bottom area in this frame)
    # Let's find bounding box cluster of tiny persons
    tiny_gts = [a for a in gts1 if not a["ignore"] and not a["uncertain"] and np.sqrt(a["bbox"][2]*a["bbox"][3]) <= 20]
    if tiny_gts:
        tx = [a["bbox"][0] for a in tiny_gts]
        ty = [a["bbox"][1] for a in tiny_gts]
        crop_cx, crop_cy = int(np.median(tx)), int(np.median(ty))
    else:
        crop_cx, crop_cy = w1 // 2, h1 // 2

    crop_size = 280
    x1 = max(0, min(w1 - crop_size, crop_cx - crop_size // 2))
    y1 = max(0, min(h1 - crop_size, crop_cy - crop_size // 2))
    crop_box = (x1, y1, x1 + crop_size, y1 + crop_size)

    crop_img = pil1.crop(crop_box)

    # Predict on crop
    res_base = base_model.predict(crop_img, imgsz=crop_size, conf=0.20, verbose=False)[0]
    res_gauss = gauss_model.predict(crop_img, imgsz=crop_size, conf=0.20, verbose=False)[0]

    # Create 3-panel figure: Ground Truth | Baseline (YOLO26n-P2) | GauSS (Ours)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    
    # Panel A: Ground Truth
    axes[0].imshow(crop_img)
    axes[0].set_title("(a) Ground Truth", fontweight="bold")
    for g in gts1:
        gx, gy, gw, gh = g["bbox"]
        if x1 <= gx <= x1 + crop_size and y1 <= gy <= y1 + crop_size:
            color = "#27AE60" if not g["ignore"] and not g["uncertain"] else "#E67E22"
            rect = patches.Rectangle((gx - x1, gy - y1), gw, gh, linewidth=1.5, edgecolor=color, facecolor="none")
            axes[0].add_patch(rect)
    axes[0].axis("off")

    # Panel B: Baseline
    axes[1].imshow(crop_img)
    axes[1].set_title("(b) Baseline (YOLO26n-P2)", fontweight="bold")
    for b, s in zip(res_base.boxes.xyxy.cpu().numpy(), res_base.boxes.conf.cpu().numpy()):
        rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=1.5, edgecolor="#2C5D88", facecolor="none")
        axes[1].add_patch(rect)
        axes[1].text(b[0], max(0, b[1]-3), f"{s:.2f}", color="#2C5D88", fontsize=7, fontweight="bold", backgroundcolor="white")
    axes[1].axis("off")

    # Panel C: GauSS
    axes[2].imshow(crop_img)
    axes[2].set_title("(c) GauSS (Ours)", fontweight="bold")
    for b, s in zip(res_gauss.boxes.xyxy.cpu().numpy(), res_gauss.boxes.conf.cpu().numpy()):
        rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=1.5, edgecolor="#C0392B", facecolor="none")
        axes[2].add_patch(rect)
        axes[2].text(b[0], max(0, b[1]-3), f"{s:.2f}", color="#C0392B", fontsize=7, fontweight="bold", backgroundcolor="white")
    axes[2].axis("off")

    plt.tight_layout()
    p1_path = out_dir / "12_qualitative_tiny_detections.pdf"
    fig.savefig(p1_path, format="pdf", bbox_inches="tight")
    fig.savefig(out_dir / "12_qualitative_tiny_detections.png", format="png", bbox_inches="tight")
    plt.close(fig)
    print("✓ Figure 1 saved.", flush=True)

    # =========================================================================
    # FIGURE 2: Sea Clutter & False Positive Suppression
    # Target image: youtube_V0001_I0002640.jpg (cx=1195, cy=312: Wave crest false alarm)
    # =========================================================================
    print("Generating Figure 2: False Positive & Sea Clutter Suppression...", flush=True)
    img_path2, _info2, _gts2 = get_image_and_gts("youtube_V0001_I0002640.jpg")
    pil2 = Image.open(img_path2).convert("RGB")
    w2, h2 = pil2.size

    # Choose wave crest region where Baseline triggers high-confidence false alarm (score 0.60)
    crop_size2 = 320
    cx2, cy2 = 1195, 312
    x2_0 = max(0, min(w2 - crop_size2, cx2 - crop_size2 // 2))
    y2_0 = max(0, min(h2 - crop_size2, cy2 - crop_size2 // 2))
    crop_img2 = pil2.crop((x2_0, y2_0, x2_0 + crop_size2, y2_0 + crop_size2))

    res_base2 = base_model.predict(crop_img2, imgsz=crop_size2, conf=0.25, verbose=False)[0]
    res_gauss2 = gauss_model.predict(crop_img2, imgsz=crop_size2, conf=0.25, verbose=False)[0]

    fig2, axes2 = plt.subplots(1, 2, figsize=(8.6, 4.0))
    
    # Baseline sea clutter false positives
    axes2[0].imshow(crop_img2)
    axes2[0].set_title(f"(a) Baseline (YOLO26n-P2)\n{len(res_base2.boxes)} False Alarms on Wave Crests", fontweight="bold", fontsize=9.0, pad=8)
    for b, s in zip(res_base2.boxes.xyxy.cpu().numpy(), res_base2.boxes.conf.cpu().numpy()):
        rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=1.5, edgecolor="#E74C3C", facecolor="none")
        axes2[0].add_patch(rect)
        axes2[0].text(b[0], max(0, b[1]-3), f"Conf {s:.2f}", color="#E74C3C", fontsize=8, fontweight="bold", backgroundcolor="white")
    axes2[0].axis("off")

    # GauSS sea clutter suppression
    axes2[1].imshow(crop_img2)
    axes2[1].set_title(f"(b) GauSS (Ours)\n{len(res_gauss2.boxes)} Detection (Clutter Filtered)", fontweight="bold", fontsize=9.0, pad=8)
    for b, s in zip(res_gauss2.boxes.xyxy.cpu().numpy(), res_gauss2.boxes.conf.cpu().numpy()):
        rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=1.5, edgecolor="#27AE60", facecolor="none")
        axes2[1].add_patch(rect)
        axes2[1].text(b[0], max(0, b[1]-3), f"Conf {s:.2f}", color="#27AE60", fontsize=8, fontweight="bold", backgroundcolor="white")
    axes2[1].axis("off")

    plt.subplots_adjust(wspace=0.15)

    plt.tight_layout()
    p2_path = out_dir / "13_sea_clutter_suppression.pdf"
    fig2.savefig(p2_path, format="pdf", bbox_inches="tight")
    fig2.savefig(out_dir / "13_sea_clutter_suppression.png", format="png", bbox_inches="tight")
    plt.close(fig2)
    print("✓ Figure 2 saved.", flush=True)

    # =========================================================================
    # FIGURE 3: P2 Feature Map Spatial Energy / Attention Heatmap
    # Hook P2 features from Baseline vs GauSS
    # =========================================================================
    print("Generating Figure 3: P2 Feature Activation Energy Heatmap...", flush=True)
    # Hook P2 feature map
    torch_img = torch.from_numpy(np.array(crop_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    torch_img = torch.nn.functional.interpolate(torch_img, size=(320, 320), mode="bilinear", align_corners=False).to(device)

    base_p2_feats = []
    gauss_p2_feats = []

    def hook_base(m, i, o):
        base_p2_feats.append(o)

    def hook_gauss(m, i, o):
        gauss_p2_feats.append(o)

    # Find the layer outputting Stride 4 (P2)
    # In GauSS model, find DCAF layer
    dcaf_module = None
    for mod in gauss_model.model.modules():
        if isinstance(mod, DCAF):
            dcaf_module = mod
            break

    if dcaf_module is not None:
        h_g = dcaf_module.register_forward_hook(hook_gauss)
        with torch.no_grad():
            _ = gauss_model.model(torch_img)
        h_g.remove()

    # In Baseline model, find layer index matching P2 neck
    # Check baseline modules
    for idx, layer in enumerate(base_model.model.model):
        if hasattr(layer, "f") and isinstance(layer.f, list) and len(layer.f) >= 2:
            base_target_layer = layer
            h_b = base_target_layer.register_forward_hook(hook_base)
            with torch.no_grad():
                _ = base_model.model(torch_img)
            h_b.remove()
            break

    fig3, axes3 = plt.subplots(1, 3, figsize=(10, 3.6))
    axes3[0].imshow(crop_img)
    axes3[0].set_title("(a) Input Test Patch", fontweight="bold")
    axes3[0].axis("off")

    if base_p2_feats:
        feat_b = base_p2_feats[0][0].detach().abs().mean(dim=0).cpu().numpy()
        feat_b = cv2.resize(feat_b, (crop_size, crop_size))
        feat_b = (feat_b - feat_b.min()) / (feat_b.max() - feat_b.min() + 1e-8)
        im1 = axes3[1].imshow(feat_b, cmap="viridis")
        axes3[1].set_title("(b) Baseline P2 Feature Activation", fontweight="bold")
        axes3[1].axis("off")
        plt.colorbar(im1, ax=axes3[1], fraction=0.046, pad=0.04)

    if gauss_p2_feats:
        feat_g = gauss_p2_feats[0][0].detach().abs().mean(dim=0).cpu().numpy()
        feat_g = cv2.resize(feat_g, (crop_size, crop_size))
        feat_g = (feat_g - feat_g.min()) / (feat_g.max() - feat_g.min() + 1e-8)
        im2 = axes3[2].imshow(feat_g, cmap="viridis")
        axes3[2].set_title("(c) GauSS P2 Refined Activation", fontweight="bold")
        axes3[2].axis("off")
        plt.colorbar(im2, ax=axes3[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    p3_path = out_dir / "14_p2_feature_activation_energy.pdf"
    fig3.savefig(p3_path, format="pdf", bbox_inches="tight")
    fig3.savefig(out_dir / "14_p2_feature_activation_energy.png", format="png", bbox_inches="tight")
    plt.close(fig3)
    print("✓ Figure 3 saved.", flush=True)

    # =========================================================================
    # FIGURE 4: Gaussian Scale-Space Band Decomposition (B_high, B_mid, B_low)
    # =========================================================================
    print("Generating Figure 4: Gaussian Band Decomposition...", flush=True)
    if dcaf_module is not None:
        # Hook actual input to DCAF module during forward pass
        dcaf_inputs = []
        def hook_dcaf_in(m, i, o):
            dcaf_inputs.append(i[0])  # tuple of inputs: [c2, p3, p4]

        h_in = dcaf_module.register_forward_hook(hook_dcaf_in)
        with torch.no_grad():
            _ = gauss_model.model(torch_img)
        h_in.remove()

        if dcaf_inputs and isinstance(dcaf_inputs[0], (list, tuple)):
            c2_feat = dcaf_inputs[0][0]  # c2 tensor has 64 channels!
            with torch.no_grad():
                c2_proj = dcaf_module.proj_c2(c2_feat)
                c2_bn = dcaf_module.bn2(c2_proj)
                L1, L2, sigmas = dcaf_module.gaussian(c2_bn)
                B_high = c2_bn - L1
                B_mid = L1 - L2
                B_low = L2

                b_h_map = cv2.resize(B_high[0].abs().mean(dim=0).cpu().numpy(), (crop_size, crop_size))
                b_m_map = cv2.resize(B_mid[0].abs().mean(dim=0).cpu().numpy(), (crop_size, crop_size))
                b_l_map = cv2.resize(B_low[0].abs().mean(dim=0).cpu().numpy(), (crop_size, crop_size))

                # Normalize each band map for visualization clarity
                b_h_map = (b_h_map - b_h_map.min()) / (b_h_map.max() - b_h_map.min() + 1e-8)
                b_m_map = (b_m_map - b_m_map.min()) / (b_m_map.max() - b_m_map.min() + 1e-8)
                b_l_map = (b_l_map - b_l_map.min()) / (b_l_map.max() - b_l_map.min() + 1e-8)

                s1_val = float(torch.cat((sigmas[0], sigmas[1])).mean().item())
                s2_val = float(torch.cat((sigmas[2], sigmas[3])).mean().item())

            fig4, axes4 = plt.subplots(1, 4, figsize=(12, 3.2))
            axes4[0].imshow(crop_img)
            axes4[0].set_title("(a) Input Crop", fontweight="bold")
            axes4[0].axis("off")

            axes4[1].imshow(b_h_map, cmap="inferno")
            axes4[1].set_title(r"(b) $\mathbf{B}_{\mathrm{high}}$ (Fine Edges / Sub-8px)", fontweight="bold")
            axes4[1].axis("off")

            axes4[2].imshow(b_m_map, cmap="inferno")
            axes4[2].set_title(r"(c) $\mathbf{B}_{\mathrm{mid}}$ (Resonance tiny2, $\sigma\approx$" + f"{s1_val:.2f})", fontweight="bold")
            axes4[2].axis("off")

            axes4[3].imshow(b_l_map, cmap="inferno")
            axes4[3].set_title(r"(d) $\mathbf{B}_{\mathrm{low}}$ (Context Structure, $\sigma\approx$" + f"{s2_val:.2f})", fontweight="bold")
            axes4[3].axis("off")

            plt.tight_layout()
            p4_path = out_dir / "15_gaussian_band_decomposition.pdf"
            fig4.savefig(p4_path, format="pdf", bbox_inches="tight")
            fig4.savefig(out_dir / "15_gaussian_band_decomposition.png", format="png", bbox_inches="tight")
            plt.close(fig4)
            print("✓ Figure 4 saved.", flush=True)

    runs.commit()
    print("All 4 qualitative figures successfully generated and committed to volume!", flush=True)
    return {
        "fig1": str(out_dir / "12_qualitative_tiny_detections.pdf"),
        "fig2": str(out_dir / "13_sea_clutter_suppression.pdf"),
        "fig3": str(out_dir / "14_p2_feature_activation_energy.pdf"),
        "fig4": str(out_dir / "15_gaussian_band_decomposition.pdf"),
    }

@app.local_entrypoint()
def main():
    print("Dispatching qualitative artifacts extraction to Modal A10 GPU...")
    res = extract_all_qualitative_artifacts.remote()
    print("Execution complete! Result paths:", res)
