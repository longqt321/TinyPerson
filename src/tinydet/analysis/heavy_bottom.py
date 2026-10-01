"""Fixed-probe backbone diagnostics for YOLO26 horizontal-box detection."""

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.nms import non_max_suppression

BINS = ("0-8", "8-16", "16-32", ">32")


def size_bin(width: float, height: float) -> str:
    """Half-open bins of max side, measured after letterboxing in input pixels."""
    side = max(width, height)
    return BINS[0] if side < 8 else BINS[1] if side < 16 else BINS[2] if side < 32 else BINS[3]


def letterbox_geometry(width, height, size):
    scale = min(size / width, size / height)
    return scale, (size - round(width * scale)) // 2, (size - round(height * scale)) // 2


def letterbox(image: Image.Image, size: int):
    width, height = image.size
    scale, left, top = letterbox_geometry(width, height, size)
    new_width, new_height = round(width * scale), round(height * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize((new_width, new_height), Image.Resampling.BILINEAR), (left, top))
    return canvas, scale, left, top


def map_box(box, scale, left, top):
    x1, y1, x2, y2 = box
    return [x1 * scale + left, y1 * scale + top, x2 * scale + left, y2 * scale + top]


def resolve_stages(model):
    """Find principal C3k2 stage outputs from installed model graph, before its head."""
    layers = getattr(model, "model", None)
    if layers is None:
        raise ValueError("Expected YOLO26 model with backbone stages")
    first_head = next((i for i, layer in enumerate(layers) if type(layer).__name__ == "C2PSA"), None)
    if first_head is None:
        raise ValueError("Expected YOLO26 backbone C2PSA stage")
    c3 = [i for i, layer in enumerate(layers[:first_head]) if type(layer).__name__ == "C3k2"]
    if len(c3) != 4:
        raise ValueError(f"Expected four YOLO26 backbone C3k2 stages, found {len(c3)}")
    return dict(zip(("C2", "C3", "C4", "C5"), c3))


def dataset_images(dataset_file: Path, split: str = "val"):
    data = yaml.safe_load(dataset_file.read_text())
    root = Path(data["path"]) if data.get("path") else dataset_file.parent
    if data.get("path") and not root.is_absolute():
        root = dataset_file.parent / root
    source = root / data[split]
    if source.is_file():
        return [Path(line) if Path(line).is_absolute() else root / line
                for line in source.read_text().splitlines() if line.strip()]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError(f"Image path lacks images directory: {image_path}")
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def make_probe(dataset_file: Path, output: Path, size: int = 1280, count: int = 200, seed: int = 42):
    """Choose instances deterministically, round-robin across input-pixel size bins."""
    grouped = {name: [] for name in BINS}
    for image_path in dataset_images(dataset_file):
        label = label_path(image_path)
        if not label.is_file():
            continue
        with Image.open(image_path) as image:
            width, height = image.size
            scale, left, top = letterbox_geometry(width, height, size)
        for index, line in enumerate(label.read_text().splitlines()):
            values = [float(v) for v in line.split()]
            if len(values) < 5 or int(values[0]) != 0:
                continue
            cx, cy, bw, bh = values[1:5]
            box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
                   (cx + bw / 2) * width, (cy + bh / 2) * height]
            mapped = map_box(box, scale, left, top)
            bin_name = size_bin(bw * width * scale, bh * height * scale)
            grouped[bin_name].append({"image": str(image_path), "gt_id": index,
                                      "box_original": box, "box_input": mapped, "size_bin": bin_name})
    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < count and any(grouped.values()):
        for name in BINS:
            if grouped[name] and len(selected) < count:
                selected.append(grouped[name].pop())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"dataset": str(dataset_file), "size": size, "seed": seed,
                                  "size_definition": "max side after letterbox, input pixels",
                                  "instances": selected}, indent=2) + "\n")
    return selected


def region_masks(box, height: int, width: int, input_size: int, expansion: float = 2.0):
    if expansion <= 1:
        raise ValueError("Background expansion must exceed one")
    x1, y1, x2, y2 = [float(x) for x in box]
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    def raster(bounds):
        xa, ya, xb, yb = bounds
        left = max(0, min(width-1, math.floor(xa * width / input_size)))
        top = max(0, min(height-1, math.floor(ya * height / input_size)))
        right = max(left+1, min(width, math.ceil(xb * width / input_size)))
        bottom = max(top+1, min(height, math.ceil(yb * height / input_size)))
        mask = torch.zeros((height, width), dtype=torch.bool)
        mask[top:bottom, left:right] = True
        return mask
    obj = raster((x1, y1, x2, y2))
    outer = raster((cx-bw*expansion/2, cy-bh*expansion/2,
                    cx+bw*expansion/2, cy+bh*expansion/2))
    if not (outer & ~obj).any():
        ys, xs = obj.nonzero(as_tuple=True)
        outer[max(int(ys.min())-1, 0):min(int(ys.max())+2, height),
              max(int(xs.min())-1, 0):min(int(xs.max())+2, width)] = True
    return obj, outer & ~obj


def activation_ratio(feature: torch.Tensor, box, input_size: int, expansion: float = 2.0):
    magnitude = feature.abs().mean(0)
    obj, bg = region_masks(box, *magnitude.shape, input_size, expansion)
    obj, bg = obj.to(magnitude.device), bg.to(magnitude.device)
    if not obj.any() or not bg.any():
        return None
    object_mean, background_mean = magnitude[obj].mean(), magnitude[bg].mean()
    return float(object_mean / (background_mean + 1e-9)), float(object_mean), float(background_mean)


def erf_extent(energy: torch.Tensor, center):
    """Smallest centered square containing 50% and 90% input-gradient energy."""
    if not torch.isfinite(energy).all() or energy.sum() <= 0:
        return None
    height, width = energy.shape
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    radius = torch.maximum((xx - center[0]).abs(), (yy - center[1]).abs()).flatten()
    order = torch.argsort(radius)
    cumulative = energy.flatten()[order].cumsum(0) / energy.sum()
    values = []
    for fraction in (.5, .9):
        index = torch.searchsorted(cumulative, fraction).clamp(max=radius.numel() - 1)
        r = int(radius[order[index]])
        side = 2 * r + 1
        area = (min(width, int(center[0]) + r + 1) - max(0, int(center[0]) - r)) * \
               (min(height, int(center[1]) + r + 1) - max(0, int(center[1]) - r))
        values.append((side, area, area / (width * height)))
    return values


def _panel(image, maps, box, output: Path, title: str):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    axes[0].imshow(image)
    for axis, (stage, value) in zip(axes[1:], maps.items()):
        axis.imshow(image)
        value = np.asarray(value, dtype=np.float32)
        value = np.nan_to_num(value)
        value = np.array(Image.fromarray(value).resize(image.size, Image.Resampling.BILINEAR))
        axis.imshow(value / max(float(value.max()), 1e-12), cmap="magma", alpha=.55, vmin=0, vmax=1)
        axis.set_title(stage)
    axes[0].set_title("Input")
    for axis in axes:
        axis.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                                     fill=False, edgecolor="lime", linewidth=1))
        axis.axis("off")
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _summary(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["size_bin"])].append(row[field])
    return [{"stage": stage, "size_bin": bin_name, "count": len(values),
             "mean": float(np.mean(values)), "median": float(np.median(values)),
             "std": float(np.std(values))}
            for (stage, bin_name), values in sorted(grouped.items())]


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def energy_profile(energy, center):
    """Reduce gradient energy to integer-radius shells without a full pixel sort.

    Flooring radius preserves the previous integer-side ERF convention.
    """
    height, width = energy.shape
    yy = torch.arange(height, device=energy.device)[:, None]
    xx = torch.arange(width, device=energy.device)[None, :]
    radius = torch.maximum((xx-center[0]).abs(), (yy-center[1]).abs()).long()
    return torch.zeros(max(height, width) + 1, device=energy.device).scatter_add_(
        0, radius.flatten(), energy.flatten()).cpu()


def profile_extent(profile, center, height, width):
    if not torch.isfinite(profile).all() or profile.sum() <= 0:
        return None
    cumulative = profile.cumsum(0) / profile.sum()
    values = []
    for fraction in (.5, .9):
        r = int(torch.searchsorted(cumulative, fraction).clamp(max=len(profile)-1))
        area = (min(width, int(center[0])+r+1)-max(0, int(center[0])-r)) * (
            min(height, int(center[1])+r+1)-max(0, int(center[1])-r))
        values.append((2*r+1, area, area/(height*width)))
    return values


def collect_diagnostics(checkpoint: Path, probe: Path, output: Path,
                        max_instances: int | None = None, expansion: float = 2.0, panels: int = 3):
    """Model/gradient phase. Save compact CPU tensors for the CPU finalizer."""
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = json.loads(probe.read_text())
    instances, size = config["instances"][:max_instances], config["size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(checkpoint)).model.to(device).eval()
    stages = resolve_stages(model)
    grouped = defaultdict(list)
    for index, row in enumerate(instances):
        grouped[row["image"]].append((index, row))
    output.mkdir(parents=True, exist_ok=True)
    collected, rendered = [], set()
    panel_count = 0
    for image_path, rows in grouped.items():
        with Image.open(image_path) as source:
            image = letterbox(source.convert("RGB"), size)[0]
        array = np.asarray(image, dtype=np.float32).copy() / 255
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).requires_grad_()
        features = {}
        hooks = [model.model[index].register_forward_hook(
            lambda _module, _input, value, name=name, features=features: features.__setitem__(name, value))
            for name, index in stages.items()]
        try:
            processed, raw = model(tensor)
            head = model.model[-1]
            prediction = head._inference(raw["one2one"])
            detections = non_max_suppression(processed.detach().clone(), conf_thres=.05,
                iou_thres=.7, nc=head.nc, max_det=1000, max_time_img=2.0, end2end=head.end2end)[0]
            people = detections[detections[:, 5] == 0]
            activations = {stage: feature[0].detach().abs().mean(0).cpu()
                           for stage, feature in features.items()}
            for row_number, row in rows:
                box = row["box_input"]
                center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                centers = ((prediction[0, :2]+prediction[0, 2:4])/2
                           if head.end2end or head.xyxy else prediction[0, :2])
                index = ((centers[0]-center[0])**2+(centers[1]-center[1])**2).argmin()
                score = prediction[0, 4, index]
                overlap = (bbox_iou(torch.tensor(box, device=device)[None], people[:, :4], xywh=False)
                           .flatten() if len(people) else torch.empty(0, device=device))
                category = ("success" if len(overlap) and ((overlap >= .5) & (people[:, 4] >= .25)).any()
                            else "difficult" if len(overlap) and (overlap >= .1).any() else "false_negative")
                render = panel_count < panels or (category not in rendered and panel_count < panels+3)
                gradients = (torch.autograd.grad(score, tuple(features.values()), retain_graph=True,
                                                allow_unused=True) if render else [None]*len(features))
                item = {"row": row, "index": row_number, "category": category,
                        "activation": activations, "profiles": {}, "gradcam": {}, "erf": {}}
                for (stage, feature), gradient in zip(features.items(), gradients):
                    fmap = feature[0]
                    fx = min(fmap.shape[-1]-1, max(0, int(center[0]*fmap.shape[-1]/size)))
                    fy = min(fmap.shape[-2]-1, max(0, int(center[1]*fmap.shape[-2]/size)))
                    target = fmap[:, fy, fx].abs().mean()
                    energy = torch.autograd.grad(target, tensor, retain_graph=True)[0][0].abs().mean(0).detach()
                    item["profiles"][stage] = energy_profile(energy, center)
                    if render:
                        cam = (F.relu((gradient[0].mean((-2, -1), keepdim=True)*fmap).sum(0))
                               if gradient is not None else torch.zeros_like(activations[stage]))
                        item["gradcam"][stage] = cam.detach().cpu()
                        item["erf"][stage] = energy.cpu()
                collected.append(item)
                if render:
                    rendered.add(category)
                    panel_count += 1
        finally:
            for hook in hooks:
                hook.remove()
    metadata = {"checkpoint": str(checkpoint), "probe": str(probe), "stages": stages,
                "device": device, "instances": len(instances), "forward_passes": len(grouped),
                "panel_categories": sorted(rendered), "background_expansion": expansion,
                "activation": "channel mean absolute value",
                "gradcam_target": "person score of decoded one-to-one candidate nearest GT center",
                "erf_target": "mean absolute stage feature at GT center",
                "erf_extent": "integer-radius centered square enclosing 50/90% input-gradient energy"}
    torch.save({"items": collected, "size": size, "metadata": metadata}, output / "diagnostics.pt")
    return output


def finalize_diagnostics(output: Path):
    """CPU phase: region statistics, ERF quantiles, CSVs and all figures."""
    payload = torch.load(output / "diagnostics.pt", map_location="cpu", weights_only=True)
    size, metadata = payload["size"], payload["metadata"]
    records = []
    for item in payload["items"]:
        row, box = item["row"], item["row"]["box_input"]
        center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        for stage, activation in item["activation"].items():
            base = {"stage": stage, "size_bin": row["size_bin"], "image": row["image"], "gt_id": row["gt_id"]}
            ratio = activation_ratio(activation[None], box, size, metadata["background_expansion"])
            if ratio is not None:
                records.append({**base, "ratio": ratio[0], "object_activation": ratio[1],
                                "background_activation": ratio[2]})
            extent = profile_extent(item["profiles"][stage], center, size, size)
            if extent is not None:
                for fraction, values in zip((50, 90), extent):
                    records.append({**base, "erf": fraction, "side": values[0],
                                    "area": values[1], "area_fraction": values[2]})
        if item["gradcam"]:
            with Image.open(row["image"]) as source:
                image = letterbox(source.convert("RGB"), size)[0]
            for label in ("activation", "gradcam", "erf"):
                _panel(image, item[label], box,
                       output / label / f"probe_{item['index']:03d}_{item['category']}.png", label)
    ratio_rows = [row for row in records if "ratio" in row]
    erf_rows = [row for row in records if "erf" in row]
    _write_csv(output / "metrics" / "activation_ratio.csv", ratio_rows)
    _write_csv(output / "metrics" / "erf.csv", erf_rows)
    _write_csv(output / "metrics" / "activation_ratio_summary.csv", _summary(ratio_rows, "ratio"))
    for fraction in (50, 90):
        _write_csv(output / "metrics" / f"erf{fraction}_summary.csv",
                   _summary([row for row in erf_rows if row["erf"] == fraction], "area_fraction"))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def run_diagnostics(checkpoint: Path, probe: Path, output: Path, max_instances: int | None = None,
                    expansion: float = 2.0, panels: int = 3):
    collect_diagnostics(checkpoint, probe, output, max_instances, expansion, panels)
    return finalize_diagnostics(output)
