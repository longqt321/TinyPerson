"""Prepare a YOLO train/validation derivative from TinyPerson's official erase train JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


def _group_key(filename: str) -> str:
    stem = Path(filename).stem
    match = re.match(r"^(.*)_I\d+$", stem)
    if match:
        return match.group(1)
    match = re.match(r"^(.*?)[_-](?:frame)?\d+$", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(annotations_path: Path, image_root: Path, output: Path,
            val_fraction: float = 0.1, seed: int = 42) -> dict:
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    data = json.loads(annotations_path.read_text())
    if not all(key in data for key in ("images", "annotations", "categories")):
        raise ValueError("Expected original COCO-style TinyPerson annotation JSON")

    images = data["images"]
    if any("corner" in image for image in images):
        raise ValueError("Corner/crop annotations are unsupported; use non-corner erase train JSON")
    if len({Path(image["file_name"]).name for image in images}) != len(images):
        raise ValueError("Image basenames are not unique; refusing ambiguous YOLO filenames")

    by_image = defaultdict(list)
    for ann in data["annotations"]:
        by_image[int(ann["image_id"])].append(ann)
    dense_ids = {
        image_id for image_id, anns in by_image.items()
        if len(anns) > 220 or any(bool(ann.get("in_dense_image", False)) for ann in anns)
    }
    if dense_ids:
        raise ValueError(
            f"Input contains {len(dense_ids)} dense images. Use the official non-dense erase train JSON."
        )
    special = [
        ann for ann in data["annotations"]
        if bool(ann.get("ignore", False)) or bool(ann.get("uncertain", False))
        or bool(ann.get("logo", False))
    ]
    if special:
        raise ValueError(
            f"Found {len(special)} ignore/uncertain/logo annotations. Use matching "
            "erase_with_uncertain_dataset images and annotation JSON."
        )

    image_records = []
    group_to_images = defaultdict(list)
    for image in images:
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        source = image_root / file_name
        if not source.is_file():
            raise FileNotFoundError(f"Image referenced by JSON is missing: {source}")
        with Image.open(source) as raster:
            width, height = raster.size
        declared = (int(image["width"]), int(image["height"]))
        if (width, height) != declared:
            raise ValueError(
                f"Raster/COCO dimension mismatch for {file_name}: raster={(width, height)}, "
                f"JSON={declared}"
            )
        group = _group_key(file_name)
        record = {
            "id": image_id, "file_name": file_name, "basename": Path(file_name).name,
            "width": width, "height": height, "group": group,
            "annotations": by_image.get(image_id, []),
        }
        image_records.append(record)
        group_to_images[group].append(record)

    groups = list(group_to_images)
    if len(groups) < 2:
        raise ValueError("Cannot make a train/val split with fewer than 2 filename groups")
    rng = random.Random(seed)
    rng.shuffle(groups)
    target_val = max(1, round(len(image_records) * val_fraction))
    val_groups, val_count = set(), 0
    for group in groups:
        if val_count >= target_val:
            break
        val_groups.add(group)
        val_count += len(group_to_images[group])
    if len(val_groups) == len(groups):
        val_groups.remove(groups[-1])

    output.mkdir(parents=True, exist_ok=True)
    source_annotation_copy = output / "annotations" / annotations_path.name
    source_annotation_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(annotations_path, source_annotation_copy)
    manifest = []
    split_counts = Counter()
    object_counts = Counter()
    for record in image_records:
        split = "val" if record["group"] in val_groups else "train"
        split_counts[split] += 1
        source = image_root / record["file_name"]
        image_dest = output / "images" / split / record["basename"]
        label_dest = output / "labels" / split / f"{Path(record['basename']).stem}.txt"
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        label_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_dest)

        lines = []
        width, height = record["width"], record["height"]
        for ann in record["annotations"]:
            x, y, bw, bh = map(float, ann["bbox"])
            if bw <= 0 or bh <= 0:
                raise ValueError(f"Non-positive bbox in image {record['file_name']}: {ann['bbox']}")
            if x < -1e-3 or y < -1e-3 or x + bw > width + 1e-3 or y + bh > height + 1e-3:
                raise ValueError(f"Out-of-bounds bbox in image {record['file_name']}: {ann['bbox']}")
            cx, cy = (x + bw / 2) / width, (y + bh / 2) / height
            lines.append(f"0 {cx:.8f} {cy:.8f} {bw / width:.8f} {bh / height:.8f}")
        label_dest.write_text("\n".join(lines) + ("\n" if lines else ""))
        object_counts[split] += len(lines)
        manifest.append({
            "image_id": record["id"], "source_file_name": record["file_name"],
            "output_image": str(image_dest.relative_to(output)),
            "output_label": str(label_dest.relative_to(output)),
            "source_width": width, "source_height": height,
            "group": record["group"], "split": split, "objects": len(lines),
        })

    data_yaml = {
        "path": str(output.resolve()), "train": "images/train",
        "val": "images/val", "names": {0: "person"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    (output / "manifest.json").write_text(json.dumps({
        "source_annotation": str(annotations_path.resolve()),
        "source_annotation_copy": str(source_annotation_copy.relative_to(output)),
        "source_annotation_sha256": _sha256(annotations_path),
        "image_root": str(image_root.resolve()), "seed": seed,
        "validation_fraction_requested": val_fraction,
        "split_strategy": "deterministic group split inferred from filename video/frame suffix",
        "group_count": len(groups),
        "singleton_group_count": sum(len(items) == 1 for items in group_to_images.values()),
        "images": len(image_records), "split_images": dict(split_counts),
        "split_objects": dict(object_counts), "images_manifest": manifest,
    }, indent=2) + "\n")
    return {
        "data_yaml": str((output / "data.yaml").resolve()),
        "images": len(image_records), "split_images": dict(split_counts),
        "split_objects": dict(object_counts), "groups": len(groups),
        "singleton_groups": sum(len(items) == 1 for items in group_to_images.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path,
                        help="erase_with_uncertain_dataset/annotations/tiny_set_train.json")
    parser.add_argument("--image-root", required=True, type=Path,
                        help="Directory containing file_name paths from the JSON")
    parser.add_argument("--output", required=True, type=Path,
                        help="New, empty output directory for YOLO images/labels/data.yaml")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.annotations, args.image_root, args.output,
                             args.val_fraction, args.seed), indent=2))


if __name__ == "__main__":
    main()
