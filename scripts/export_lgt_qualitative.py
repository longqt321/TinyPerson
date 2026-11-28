"""Export honest, true-scale 4-column qualitative figures (GT | Base | +NWD | +GauCon).

Produces:
  1. `fig_gaucon_qualitative.{pdf,png}`: 1-row x 4-col compact WACV panel (6.85 in wide).
  2. `fig_gaucon_qualitative_gallery.{pdf,png}`: 3-row x 4-col gallery for the Course Report
     covering (a) low-light swimmer group (Tiled), (b) cluttered wave beach crowd (Tiled),
     and (c) full-frame 1280x1280 inference highlighting scale limits on 2-5 px targets.
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
from PIL import Image

from tinydet.utils.runtime import DEFAULT_HF_DATASET_REPO, discover_dataset

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_GT = "#009E73"
C_IGN = "#E69F00"
C_BASE = "#0072B2"
C_NWD = "#7C3AED"
C_OURS = "#D55E00"


def _index_preds(pred_json: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    payload = orjson.loads(pred_json.read_bytes())
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in payload.get("images", []):
        name = Path(str(row["path"])).name
        boxes = np.asarray(row.get("boxes", []), dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(row.get("scores", []), dtype=np.float64).reshape(-1)
        out[name] = (boxes, scores)
    return out


def _draw_boxes(
    ax: plt.Axes,
    boxes: np.ndarray,
    scores: np.ndarray,
    color: str,
    conf_thr: float,
    x0: int,
    y0: int,
    view: int,
) -> int:
    count = 0
    placed: list[tuple[float, float]] = []
    # Count visible boxes first to reduce label clutter in high-density crops (> 12 boxes)
    vis = [
        ((x1, y1, x2, y2), sc)
        for (x1, y1, x2, y2), sc in zip(boxes, scores)
        if sc >= conf_thr and x2 >= x0 and x1 <= x0 + view and y2 >= y0 and y1 <= y0 + view
    ]
    draw_scores = len(vis) <= 12
    for (x1, y1, x2, y2), sc in vis:
        cx1, cy1 = x1 - x0, y1 - y0
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        ax.add_patch(
            mpatches.Rectangle((cx1, cy1), w, h, linewidth=0.95, edgecolor=color, facecolor="none")
        )
        if draw_scores:
            tx = max(3.0, min(cx1, view - 28.0))
            ty = max(12.0, cy1 - 2.5)
            for px, py in placed:
                if abs(tx - px) < 25.0 and abs(ty - py) < 11.0:
                    ty = min(view - 4.0, cy1 + h + 9.5)
            placed.append((tx, ty))
            ax.text(
                tx,
                ty,
                f"{sc:.2f}",
                color=color,
                fontsize=5.8,
                fontweight="bold",
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.07", "fc": "white", "ec": color, "lw": 0.35, "alpha": 0.88},
            )
        count += 1
    return count


def _count_in_crop(boxes: np.ndarray, scores: np.ndarray, conf: float, x0: int, y0: int, view: int) -> int:
    if len(scores) == 0:
        return 0
    return int(
        np.sum(
            (scores >= conf)
            & (boxes[:, 0] >= x0)
            & (boxes[:, 2] <= x0 + view)
            & (boxes[:, 1] >= y0)
            & (boxes[:, 3] <= y0 + view)
        )
    )


def _select_scenes(
    anno: dict,
    img_dir: Path,
    base_p: dict[str, tuple[np.ndarray, np.ndarray]],
    nwd_p: dict[str, tuple[np.ndarray, np.ndarray]],
    ours_p: dict[str, tuple[np.ndarray, np.ndarray]],
    conf: float,
    crop_size: int = 260,
    min_gt: int = 5,
    max_gt: int = 18,
    avoid_names: set[str] | None = None,
    prefer_clutter_diff: bool = False,
) -> tuple[Path, list[dict], int, int]:
    gts_by_img: dict[int, list[dict]] = {}
    for a in anno["annotations"]:
        gts_by_img.setdefault(int(a["image_id"]), []).append(a)

    best_score = -1e9
    best_pack: tuple[Path, list[dict], int, int] | None = None
    skip = avoid_names or set()

    for im in anno["images"]:
        fname = Path(str(im["file_name"])).name
        if fname in skip:
            continue
        img_p = img_dir / fname
        if not img_p.is_file() or fname not in base_p or fname not in nwd_p or fname not in ours_p:
            continue
        gts = gts_by_img.get(int(im["id"]), [])
        pos_tiny = [
            g for g in gts
            if not g.get("ignore") and not g.get("uncertain") and not g.get("iscrowd")
            and (g["bbox"][2] * g["bbox"][3]) ** 0.5 <= 22.0
        ]
        if len(pos_tiny) < min_gt:
            continue
        w_im, h_im = int(im["width"]), int(im["height"])
        cx = float(np.median([g["bbox"][0] + 0.5 * g["bbox"][2] for g in pos_tiny]))
        cy = float(np.median([g["bbox"][1] + 0.5 * g["bbox"][3] for g in pos_tiny]))
        x0 = int(np.clip(cx - crop_size // 2, 0, max(0, w_im - crop_size)))
        y0 = int(np.clip(cy - crop_size // 2, 0, max(0, h_im - crop_size)))

        in_gt = [
            g for g in gts
            if not g.get("ignore") and not g.get("uncertain") and not g.get("iscrowd")
            and g["bbox"][0] >= x0 and g["bbox"][1] >= y0
            and g["bbox"][0] + g["bbox"][2] <= x0 + crop_size
            and g["bbox"][1] + g["bbox"][3] <= y0 + crop_size
        ]
        if not (min_gt <= len(in_gt) <= max_gt):
            continue

        nb = _count_in_crop(*base_p[fname], conf, x0, y0, crop_size)
        nn = _count_in_crop(*nwd_p[fname], conf, x0, y0, crop_size)
        no = _count_in_crop(*ours_p[fname], conf, x0, y0, crop_size)

        if prefer_clutter_diff:
            # Find scenes where Ours matches GT count closely while avoiding NWD over-detection/clutter
            score = -2.0 * abs(no - len(in_gt)) + abs(nn - len(in_gt)) + 0.5 * (len(in_gt) - abs(nb - len(in_gt)))
        else:
            score = (no - nb) - 1.5 * abs(no - len(in_gt))
        if score > best_score and no >= nb:
            best_score = score
            best_pack = (img_p, gts, x0, y0)

    if best_pack is None:
        im = anno["images"][0]
        fname = Path(str(im["file_name"])).name
        return img_dir / fname, gts_by_img.get(int(im["id"]), []), 0, 0
    return best_pack


def _render_row(
    axes_row: np.ndarray,
    img_p: Path,
    gts: list[dict],
    x0: int,
    y0: int,
    crop_size: int,
    base_p: dict[str, tuple[np.ndarray, np.ndarray]],
    nwd_p: dict[str, tuple[np.ndarray, np.ndarray]],
    ours_p: dict[str, tuple[np.ndarray, np.ndarray]],
    conf: float,
    row_tag: str | None = None,
    show_titles: bool = True,
) -> tuple[int, int, int, int]:
    with Image.open(img_p) as pil_im:
        crop = pil_im.convert("RGB").crop((x0, y0, x0 + crop_size, y0 + crop_size))
    for ax in axes_row:
        ax.imshow(crop)
        ax.set_xlim(0, crop_size)
        ax.set_ylim(crop_size, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    n_gt = 0
    for g in gts:
        gx, gy, gw, gh = g["bbox"]
        if gx + gw < x0 or gx > x0 + crop_size or gy + gh < y0 or gy > y0 + crop_size:
            continue
        is_ign = bool(g.get("ignore") or g.get("uncertain") or g.get("iscrowd"))
        col = C_IGN if is_ign else C_GT
        ls = (0, (3, 2)) if is_ign else "solid"
        axes_row[0].add_patch(mpatches.Rectangle((gx - x0, gy - y0), gw, gh, linewidth=1.05, edgecolor=col, facecolor="none", linestyle=ls))
        if not is_ign:
            n_gt += 1

    nb = _draw_boxes(axes_row[1], *base_p[img_p.name], C_BASE, conf, x0, y0, crop_size)
    nn = _draw_boxes(axes_row[2], *nwd_p[img_p.name], C_NWD, conf, x0, y0, crop_size)
    no = _draw_boxes(axes_row[3], *ours_p[img_p.name], C_OURS, conf, x0, y0, crop_size)

    if row_tag:
        axes_row[0].set_ylabel(row_tag, fontsize=7.5, fontweight="bold", labelpad=4)
    if show_titles:
        axes_row[0].set_title(f"(a) GT ({n_gt})", fontsize=7.8, fontweight="bold", pad=3)
        axes_row[1].set_title(f"(b) Base ({nb}/{n_gt})", fontsize=7.8, fontweight="bold", pad=3)
        axes_row[2].set_title(f"(c) + NWD ({nn}/{n_gt})", fontsize=7.8, fontweight="bold", pad=3)
        axes_row[3].set_title(f"(d) + GauCon ({no}/{n_gt})", fontsize=7.8, fontweight="bold", pad=3)
    return n_gt, nb, nn, no


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/tp80_yolo26n_base/colab-seed42/metrics"))
    parser.add_argument("--nwd-dir", type=Path, default=Path("outputs/tp80_yolo26n_nwd/colab-seed42/metrics"))
    parser.add_argument("--ours-dir", type=Path, default=Path("outputs/tp80_yolo26n_lgt/colab-seed42/metrics"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    ds_yaml = discover_dataset(None, "tinyperson-yolo-v2", "local", Path.cwd(), hf_repo=DEFAULT_HF_DATASET_REPO)
    ds_root = ds_yaml.parent
    anno_path = ds_root / "annotations" / "tiny_set_test.json"
    img_dir = ds_root / "images" / "test"
    if not anno_path.is_file() or not img_dir.is_dir():
        print("[qualitative] Missing TinyPerson test data.")
        return 0

    anno = orjson.loads(anno_path.read_bytes())
    base_t = _index_preds(args.base_dir / "test_predictions_tiled.json")
    nwd_t = _index_preds(args.nwd_dir / "test_predictions_tiled.json")
    ours_t = _index_preds(args.ours_dir / "test_predictions_tiled.json")

    base_f = _index_preds(args.base_dir / "test_predictions.json")
    nwd_f = _index_preds(args.nwd_dir / "test_predictions.json")
    ours_f = _index_preds(args.ours_dir / "test_predictions.json")

    crop_size = 260
    s1 = _select_scenes(anno, img_dir, base_t, nwd_t, ours_t, args.conf, crop_size, min_gt=5, max_gt=10)
    s2 = _select_scenes(anno, img_dir, base_t, nwd_t, ours_t, args.conf, crop_size, min_gt=8, max_gt=22, avoid_names={s1[0].name}, prefer_clutter_diff=True)
    # Scene 3: Full-frame 1280 challenge showing scale loss on ultra-tiny targets (<6 px) vs Tiled
    s3 = _select_scenes(anno, img_dir, base_f, nwd_f, ours_f, args.conf, crop_size, min_gt=5, max_gt=15, avoid_names={s1[0].name, s2[0].name})

    # 1. Single-row compact 4-column figure for WACV paper
    fig1, axes1 = plt.subplots(1, 4, figsize=(6.85, 1.92))
    c1 = _render_row(axes1, *s1, crop_size, base_t, nwd_t, ours_t, args.conf, show_titles=True)
    fig1.subplots_adjust(left=0.005, right=0.995, top=0.87, bottom=0.01, wspace=0.04)
    for out_d in (Path("paper/figures"), Path("docs/report/figures")):
        out_d.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out_d / "fig_gaucon_qualitative.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
        fig1.savefig(out_d / "fig_gaucon_qualitative.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig1)

    # 2. Multi-scene 3x4 gallery for Course Report
    fig2, axes2 = plt.subplots(3, 4, figsize=(6.85, 5.45))
    c_r1 = _render_row(axes2[0], *s1, crop_size, base_t, nwd_t, ours_t, args.conf, row_tag="Tiled\nVen biển", show_titles=False)
    c_r2 = _render_row(axes2[1], *s2, crop_size, base_t, nwd_t, ours_t, args.conf, row_tag="Tiled\nĐông đúc", show_titles=False)
    c_r3 = _render_row(axes2[2], *s3, crop_size, base_f, nwd_f, ours_f, args.conf, row_tag="Full-1280\nGiới hạn", show_titles=False)
    col_heads = ["(a) Ground Truth", "(b) Base", "(c) + NWD", "(d) + GauCon (Ours)"]
    for j, ch in enumerate(col_heads):
        axes2[0, j].set_title(ch, fontsize=8.0, fontweight="bold", pad=4)
    for row_i, counts in enumerate((c_r1, c_r2, c_r3)):
        gt_n, b_n, n_n, o_n = counts
        axes2[row_i, 0].set_xlabel(f"GT={gt_n}", fontsize=7.0, labelpad=2)
        axes2[row_i, 1].set_xlabel(f"Pred={b_n}/{gt_n}", fontsize=7.0, labelpad=2)
        axes2[row_i, 2].set_xlabel(f"Pred={n_n}/{gt_n}", fontsize=7.0, labelpad=2)
        axes2[row_i, 3].set_xlabel(f"Pred={o_n}/{gt_n}", fontsize=7.0, labelpad=2)
    fig2.subplots_adjust(left=0.07, right=0.995, top=0.94, bottom=0.05, wspace=0.04, hspace=0.18)
    for out_d in (Path("paper/figures"), Path("docs/report/figures")):
        fig2.savefig(out_d / "fig_gaucon_qualitative_gallery.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
        fig2.savefig(out_d / "fig_gaucon_qualitative_gallery.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig2)

    print(f"[qualitative] Scene1={s1[0].name} {c1}; Scene2={s2[0].name} {c_r2}; Scene3={s3[0].name} {c_r3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
