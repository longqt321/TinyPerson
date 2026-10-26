"""Convert Tiny Object Detection datasets from Parquet (Hugging Face) to official YOLO format.

Follows the exact specifications defined in the respective original papers:
1. TinyPerson (Yu et al., WACV 2020):
   - Classes: [sea_person, earth_person] (or single-class 'person')
   - Positive boxes: not ignore and not uncertain
   - Excludes dense images (>220 boxes) per official benchmark
   - Splits: train (746 images) and test/val (786 images)

2. VisDrone2019-DET (Zhu et al., PAMI 2021):
   - 10 evaluated classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
   - Drops category 0 (ignored regions), category 11 (others), and boxes with score == 0
   - Splits: train (6,471 images), val (548 images), test (1,610 images)

3. AI-TOD-v2 (Xu et al., ISPRS 2022 / Wang et al., ICPR 2021):
   - 8 official classes: airplane, bridge, storage-tank, ship, swimming-pool, vehicle, person, wind-mill
   - Splits: train (11,214 images), val (2,804 images), test (14,018 images)
"""

from __future__ import annotations

import argparse
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import yaml


def _clamp(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(val, max_val))


def _to_yolo_bbox(
    x: float, y: float, w: float, h: float, img_w: float, img_h: float
) -> tuple[float, float, float, float] | None:
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    x1 = _clamp(x, 0.0, img_w)
    y1 = _clamp(y, 0.0, img_h)
    x2 = _clamp(x + w, 0.0, img_w)
    y2 = _clamp(y + h, 0.0, img_h)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1e-3 or bh <= 1e-3:
        return None
    cx = (x1 + x2) * 0.5 / img_w
    cy = (y1 + y2) * 0.5 / img_h
    nw = bw / img_w
    nh = bh / img_h
    return _clamp(cx, 0.0, 1.0), _clamp(cy, 0.0, 1.0), _clamp(nw, 0.0, 1.0), _clamp(nh, 0.0, 1.0)


def _write_single_image(
    img_dest: Path, label_dest: Path, img_bytes: bytes, yolo_lines: list[str]
) -> None:
    img_dest.write_bytes(img_bytes)
    label_content = "\n".join(yolo_lines) + ("\n" if yolo_lines else "")
    label_dest.write_text(label_content, encoding="utf-8")


def convert_tinyperson(
    input_dir: Path,
    output_dir: Path,
    single_class: bool = False,
    include_dense: bool = False,
    limit: int | None = None,
    workers: int = 8,
) -> None:
    print(f"\n--- Converting TinyPerson from {input_dir} -> {output_dir} ---")
    splits_map = {
        "train": sorted(glob.glob(str(input_dir / "train-*.parquet"))),
        "val": sorted(glob.glob(str(input_dir / "test-*.parquet"))),
    }

    class_names = ["person"] if single_class else ["sea_person", "earth_person"]

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images, total_boxes = 0, 0

    for split, files in splits_map.items():
        if not files:
            continue
        tasks = []
        for file in files:
            if limit and len(tasks) >= limit:
                break
            df = pl.read_parquet(file)
            for row in df.iter_rows(named=True):
                if limit and len(tasks) >= limit:
                    break
                objs = row["objects"]
                if not include_dense and any(objs.get("in_dense_image", [])):
                    continue

                img_id = row["id"]
                w_img = float(row["width"])
                h_img = float(row["height"])
                img_bytes = row["image"]["bytes"]

                yolo_lines = []
                categories = objs["category"]
                bboxes = objs["bbox"]
                ignores = objs.get("ignore", [False] * len(bboxes))
                uncertains = objs.get("uncertain", [False] * len(bboxes))

                for cat, bbox, ign, unc in zip(categories, bboxes, ignores, uncertains):
                    if ign or unc:
                        continue
                    cls_id = 0 if single_class else int(cat)
                    x, y, w, h = bbox
                    coords = _to_yolo_bbox(x, y, w, h, w_img, h_img)
                    if coords is not None:
                        cx, cy, nw, nh = coords
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                img_dest = output_dir / "images" / split / f"{img_id}.jpg"
                lbl_dest = output_dir / "labels" / split / f"{img_id}.txt"
                if img_dest.is_file() and lbl_dest.is_file():
                    continue
                tasks.append((img_dest, lbl_dest, img_bytes, yolo_lines))
                total_boxes += len(yolo_lines)

        print(f"Writing {len(tasks)} images for split '{split}'...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_write_single_image, *t) for t in tasks]
            for f in futures:
                f.result()
        total_images += len(tasks)

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"TinyPerson done: {total_images} images, {total_boxes} boxes.")


def convert_visdrone(
    input_dir: Path, output_dir: Path, limit: int | None = None, workers: int = 8
) -> None:
    print(f"\n--- Converting VisDrone2019-DET from {input_dir} -> {output_dir} ---")
    splits_map = {
        "train": sorted(glob.glob(str(input_dir / "train-*.parquet"))),
        "val": sorted(glob.glob(str(input_dir / "validation-*.parquet"))),
        "test": sorted(glob.glob(str(input_dir / "test-*.parquet"))),
    }

    class_names = [
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    ]

    for split in splits_map:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images, total_boxes = 0, 0

    for split, files in splits_map.items():
        if not files:
            continue
        tasks = []
        for file in files:
            if limit and len(tasks) >= limit:
                break
            df = pl.read_parquet(file)
            for row in df.iter_rows(named=True):
                if limit and len(tasks) >= limit:
                    break
                img_id = row["id"]
                w_img = float(row["width"])
                h_img = float(row["height"])
                img_bytes = row["image"]["bytes"]

                objs = row["objects"]
                categories = objs["category"]
                bboxes = objs["bbox"]
                scores = objs.get("score", [1] * len(bboxes))

                yolo_lines = []
                for cat, bbox, score in zip(categories, bboxes, scores):
                    if score == 0 or cat < 1 or cat > 10:
                        continue
                    cls_id = int(cat) - 1
                    x, y, w, h = bbox
                    coords = _to_yolo_bbox(float(x), float(y), float(w), float(h), w_img, h_img)
                    if coords is not None:
                        cx, cy, nw, nh = coords
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                img_dest = output_dir / "images" / split / f"{img_id}.jpg"
                lbl_dest = output_dir / "labels" / split / f"{img_id}.txt"
                if img_dest.is_file() and lbl_dest.is_file():
                    continue
                tasks.append((img_dest, lbl_dest, img_bytes, yolo_lines))
                total_boxes += len(yolo_lines)

        print(f"Writing {len(tasks)} images for split '{split}'...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_write_single_image, *t) for t in tasks]
            for f in futures:
                f.result()
        total_images += len(tasks)

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"VisDrone done: {total_images} images, {total_boxes} boxes.")


def convert_aitod(
    input_dir: Path, output_dir: Path, limit: int | None = None, workers: int = 8
) -> None:
    print(f"\n--- Converting AI-TOD-v2 from {input_dir} -> {output_dir} ---")
    splits_map = {
        "train": sorted(glob.glob(str(input_dir / "train-*.parquet"))),
        "val": sorted(glob.glob(str(input_dir / "validation-*.parquet"))),
        "test": sorted(glob.glob(str(input_dir / "test-*.parquet"))),
    }

    class_names = [
        "airplane",
        "bridge",
        "storage-tank",
        "ship",
        "swimming-pool",
        "vehicle",
        "person",
        "wind-mill",
    ]

    for split in splits_map:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images, total_boxes = 0, 0

    for split, files in splits_map.items():
        if not files:
            continue
        split_written = 0
        for idx, file in enumerate(files):
            if limit and split_written >= limit:
                break
            df = pl.read_parquet(file)
            tasks = []
            for row in df.iter_rows(named=True):
                if limit and (split_written + len(tasks)) >= limit:
                    break
                img_id = row["id"]
                w_img = float(row["width"])
                h_img = float(row["height"])
                img_bytes = row["image"]["bytes"]

                objs = row["objects"]
                categories = objs["category"]
                bboxes = objs["bbox"]

                yolo_lines = []
                for cat, bbox in zip(categories, bboxes):
                    cls_id = int(cat)
                    x, y, w, h = bbox
                    coords = _to_yolo_bbox(float(x), float(y), float(w), float(h), w_img, h_img)
                    if coords is not None:
                        cx, cy, nw, nh = coords
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                img_dest = output_dir / "images" / split / f"{img_id}.png"
                lbl_dest = output_dir / "labels" / split / f"{img_id}.txt"
                if img_dest.is_file() and lbl_dest.is_file():
                    continue
                tasks.append((img_dest, lbl_dest, img_bytes, yolo_lines))
                total_boxes += len(yolo_lines)

            if tasks:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(_write_single_image, *t) for t in tasks]
                    for f in futures:
                        f.result()
                split_written += len(tasks)
                print(
                    f"  [{split} file {idx + 1}/{len(files)}] Wrote {len(tasks)} images (split total: {split_written})"
                )
        total_images += split_written

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"AI-TOD-v2 done: {total_images} images, {total_boxes} boxes.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Parquet datasets to YOLO format.")
    parser.add_argument(
        "--dataset",
        choices=["tinyperson", "visdrone", "aitod", "all"],
        default="all",
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("datasets/tiny-object-detection"),
        help="Path to folder containing parquet datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/derived"),
        help="Root output directory for YOLO format datasets.",
    )
    parser.add_argument(
        "--single-class",
        action="store_true",
        help="TinyPerson: merge sea_person and earth_person into single class 'person'.",
    )
    parser.add_argument(
        "--include-dense",
        action="store_true",
        help="TinyPerson: include dense images (>220 boxes) which are excluded in official benchmark.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images per split (useful for testing).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads.")

    args = parser.parse_args()

    if args.dataset in ("tinyperson", "all"):
        tp_in = args.root_dir / "tinyperson" / "detection"
        tp_out = args.output_dir / "tinyperson-yolo"
        if tp_in.is_dir():
            convert_tinyperson(
                tp_in,
                tp_out,
                single_class=args.single_class,
                include_dense=args.include_dense,
                limit=args.limit,
                workers=args.workers,
            )

    if args.dataset in ("visdrone", "all"):
        vd_in = args.root_dir / "visdrone2019-det" / "detection"
        vd_out = args.output_dir / "visdrone-yolo"
        if vd_in.is_dir():
            convert_visdrone(vd_in, vd_out, limit=args.limit, workers=args.workers)

    if args.dataset in ("aitod", "all"):
        at_in = args.root_dir / "aitod-v2" / "detection"
        at_out = args.output_dir / "aitod-yolo"
        if at_in.is_dir():
            convert_aitod(at_in, at_out, limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
