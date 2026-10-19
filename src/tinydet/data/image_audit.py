"""Audit original dataset images and labels before any YOLO resizing.

The audit reads image files directly with Pillow. It records stored pixel
dimensions, formats, modes, EXIF orientation, image decoding errors, label
geometry, and a theoretical square-letterbox scale simulation for the selected
imgsz. It never calls YOLO's resize/letterbox pipeline.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ORIENTATION_TAG = 274


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p05": None, "p25": None,
                "median": None, "p75": None, "p95": None, "max": None}
    sorted_values = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        position = (len(sorted_values) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return sorted_values[low]
        return sorted_values[low] * (high - position) + sorted_values[high] * (position - low)

    return {
        "count": len(sorted_values),
        "min": round(sorted_values[0], 4),
        "p05": round(percentile(0.05), 4),
        "p25": round(percentile(0.25), 4),
        "median": round(statistics.median(sorted_values), 4),
        "p75": round(percentile(0.75), 4),
        "p95": round(percentile(0.95), 4),
        "max": round(sorted_values[-1], 4),
    }


def _resolve_path(value: str | Path, root: Path, yaml_parent: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return yaml_parent / path


def _resolve_image_list_line(
    line: str, source_list: Path, root: Path, yaml_parent: Path
) -> Path | None:
    entry = Path(line.strip()).expanduser()
    candidates = [entry] if entry.is_absolute() else [
        root / entry, source_list.parent / entry, yaml_parent / entry
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate.resolve()
    return None


def _expand_source(
    value: str | Path, root: Path, yaml_parent: Path
) -> tuple[list[Path], list[str]]:
    entries = value if isinstance(value, list) else [value]
    images: list[Path] = []
    missing: list[str] = []
    for entry in entries:
        source = _resolve_path(str(entry), root, yaml_parent)
        entry_text = str(entry)
        if any(char in entry_text for char in "*?["):
            matches = sorted(source.parent.glob(source.name))
            found = [path.resolve() for path in matches
                     if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
            if not found:
                missing.append(f"Pattern matched no images: {entry} (resolved to {source})")
            images.extend(found)
        elif source.suffix.lower() == ".txt" and source.is_file():
            for line_number, line in enumerate(source.read_text().splitlines(), 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                resolved = _resolve_image_list_line(stripped, source, root, yaml_parent)
                if resolved is None:
                    missing.append(
                        f"Image list {source}:{line_number} references missing image {stripped}"
                    )
                else:
                    images.append(resolved)
        elif source.is_dir():
            images.extend(
                path.resolve() for path in sorted(source.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(source.resolve())
        else:
            missing.append(f"Dataset source does not exist or is not an image source: {source}")
    return sorted(set(images)), missing


def _label_candidates(image_path: Path, root: Path) -> list[Path]:
    candidates = []
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            alternate = parts.copy()
            alternate[index] = "labels"
            candidates.append(Path(*alternate).with_suffix(".txt"))
    candidates.extend([
        image_path.with_suffix(".txt"),
        root / "labels" / image_path.name.replace(image_path.suffix, ".txt"),
    ])
    # Common custom layout: root/labels/<split>/<stem>.txt
    for parent in (image_path.parent, image_path.parent.parent):
        candidates.append(parent.parent / "labels" / parent.name / f"{image_path.stem}.txt")
    deduplicated = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved not in seen:
            seen.add(resolved)
            deduplicated.append(resolved)
    return deduplicated


def _parse_yolo_label(path: Path, width: int, height: int) -> tuple[list[dict[str, float]], list[str]]:
    objects: list[dict[str, float]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(token) for token in line.split()]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric annotation")
            continue
        if not values or not all(math.isfinite(value) for value in values):
            errors.append(f"{path}:{line_number}: empty or non-finite annotation")
            continue
        coords = values[1:]
        if len(values) == 5:  # YOLO class, normalized cx, cy, width, height
            bw = abs(coords[2]) * width
            bh = abs(coords[3]) * height
            out_of_range = any(value < -0.05 or value > 1.05 for value in coords)
        elif len(coords) >= 6 and len(coords) % 2 == 0:  # YOLO polygon
            if max(abs(value) for value in coords) <= 1.5:
                xs = [value * width for value in coords[0::2]]
                ys = [value * height for value in coords[1::2]]
                out_of_range = any(value < -0.05 or value > 1.05 for value in coords)
            else:
                # Also report pixel-coordinate polygons rather than silently
                # interpreting them as normalized YOLO segmentation.
                xs, ys = coords[0::2], coords[1::2]
                out_of_range = True
            bw = max(xs) - min(xs)
            bh = max(ys) - min(ys)
        else:
            errors.append(
                f"{path}:{line_number}: expected YOLO box or polygon; got {len(values)} values"
            )
            continue
        if bw <= 0 or bh <= 0:
            errors.append(f"{path}:{line_number}: non-positive box dimensions")
            continue
        objects.append({
            "width_px": float(bw),
            "height_px": float(bh),
            "sqrt_area_px": math.sqrt(float(bw) * float(bh)),
            "max_side_px": max(float(bw), float(bh)),
            "coordinates_out_of_range": float(out_of_range),
        })
    return objects, errors


def audit_dataset(
    data_yaml: str | Path,
    imgsz: int,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit original on-disk raster dimensions and normalized YOLO labels."""
    yaml_path = Path(data_yaml).expanduser().resolve()
    if type(imgsz) is not int or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text())
    if not isinstance(config, dict):
        raise TypeError("Dataset YAML must be a mapping")
    yaml_parent = yaml_path.parent
    configured_root = config.get("path", str(yaml_parent))
    root = _resolve_path(str(configured_root), yaml_parent, yaml_parent).resolve()

    errors: list[str] = []
    warnings: list[str] = []
    split_sources: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        source = config.get(split)
        if source is None:
            continue
        files, missing = _expand_source(source, root, yaml_parent)
        split_sources[split] = files
        errors.extend(missing)
        if not files:
            warnings.append(f"Split '{split}' resolved to zero images")

    if not split_sources or not any(split_sources.values()):
        errors.append("No images found in train/val/test dataset sources")

    signature = hashlib.sha256()
    signature.update(yaml_path.read_bytes())
    for annotation_file in sorted(root.rglob("*.json")):
        if "annotations" in annotation_file.parts or "tiny_set" in annotation_file.name.lower():
            try:
                annotation_stat = annotation_file.stat()
                signature.update(
                    f"{annotation_file.name}\0{annotation_stat.st_size}\0{annotation_stat.st_mtime_ns}\n".encode()
                )
            except OSError:
                signature.update(f"{annotation_file.name}\0unstatable\n".encode())
    signatures = set()
    split_reports: dict[str, Any] = {}
    per_image: list[dict[str, Any]] = []
    global_corrupt = 0
    global_label_files = 0
    global_label_rows = 0
    global_objects = 0

    for split, image_paths in split_sources.items():
        widths: list[float] = []
        heights: list[float] = []
        ratios: list[float] = []
        file_bytes: list[float] = []
        resize_factors: list[float] = []
        raw_sqrt_area: list[float] = []
        resized_sqrt_area: list[float] = []
        raw_max_side: list[float] = []
        resized_max_side: list[float] = []
        formats: Counter[str] = Counter()
        modes: Counter[str] = Counter()
        bit_depths: Counter[str] = Counter()
        exif_orientations: Counter[str] = Counter()
        split_objects = 0
        split_label_files = 0
        split_label_rows = 0
        split_corrupt = 0
        split_out_of_range = 0
        split_image_records = []

        for image_path in image_paths:
            try:
                stat = image_path.stat()
                with Image.open(image_path) as image:
                    native_width, native_height = image.size
                    image_format = image.format or image_path.suffix.lstrip(".").upper()
                    image_mode = image.mode
                    exif = image.getexif()
                    orientation = int(exif.get(ORIENTATION_TAG, 1))
                    image.load()  # Decode the original raster; no resize is performed.
                if native_width <= 0 or native_height <= 0:
                    raise ValueError(f"Invalid native dimensions {native_width}x{native_height}")
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                split_corrupt += 1
                global_corrupt += 1
                errors.append(f"Cannot decode original image {image_path}: {exc}")
                continue

            scale = min(imgsz / native_width, imgsz / native_height)
            resized_width = round(native_width * scale)
            resized_height = round(native_height * scale)
            try:
                relative_path = image_path.relative_to(root).as_posix()
            except ValueError:
                relative_path = image_path.as_posix()

            widths.append(native_width)
            heights.append(native_height)
            ratios.append(native_width / native_height)
            file_bytes.append(stat.st_size)
            resize_factors.append(scale)
            formats[image_format] += 1
            modes[image_mode] += 1
            depth = image_mode
            if image_mode in {"1", "L", "P", "RGB", "YCbCr"}:
                depth = "8-bit/channel-or-palette"
            elif image_mode in {"RGBA", "LA"}:
                depth = "8-bit/channel-with-alpha"
            elif image_mode.startswith("I;16"):
                depth = "16-bit"
            elif image_mode == "I":
                depth = "32-bit-integer"
            elif image_mode == "F":
                depth = "32-bit-float"
            bit_depths[depth] += 1
            exif_orientations[str(orientation)] += 1

            label_path = next(
                (candidate for candidate in _label_candidates(image_path, root)
                 if candidate.is_file()),
                None,
            )
            objects: list[dict[str, float]] = []
            label_errors: list[str] = []
            label_rows = 0
            if label_path is not None:
                split_label_files += 1
                global_label_files += 1
                try:
                    label_text = label_path.read_text()
                    label_rows = sum(1 for line in label_text.splitlines() if line.strip())
                    objects, label_errors = _parse_yolo_label(
                        label_path, native_width, native_height
                    )
                except OSError as exc:
                    label_errors = [f"Cannot read labels {label_path}: {exc}"]
                errors.extend(label_errors)
                split_label_rows += label_rows
                global_label_rows += label_rows
                split_out_of_range += sum(
                    int(row["coordinates_out_of_range"]) for row in objects
                )
                for row in objects:
                    raw_sqrt_area.append(row["sqrt_area_px"])
                    resized_sqrt_area.append(row["sqrt_area_px"] * scale)
                    raw_max_side.append(row["max_side_px"])
                    resized_max_side.append(row["max_side_px"] * scale)
                    split_objects += 1
                    global_objects += 1

            split_image_records.append({
                "path": relative_path,
                "width": native_width,
                "height": native_height,
                "aspect_ratio": round(native_width / native_height, 5),
                "format": image_format,
                "mode": image_mode,
                "exif_orientation": orientation,
                "file_bytes": stat.st_size,
                "letterbox_scale_at_imgsz": round(scale, 6),
                "simulated_letterbox_width": resized_width,
                "simulated_letterbox_height": resized_height,
                "label_path": str(label_path) if label_path else None,
                "label_rows": label_rows,
                "objects_parsed": len(objects),
            })
            if image_path not in signatures:
                signatures.add(image_path)
                label_stamp = ""
                if label_path is not None:
                    try:
                        label_stat = label_path.stat()
                        label_stamp = f"{label_stat.st_size}\0{label_stat.st_mtime_ns}"
                    except OSError:
                        label_stamp = "unstatable"
                signature.update(
                    f"{relative_path}\0{stat.st_size}\0{stat.st_mtime_ns}\0"
                    f"{native_width}\0{native_height}\0{label_rows}\0{label_stamp}\n".encode()
                )

        scale_below_4 = (
            sum(value < 4.0 for value in resized_sqrt_area) / len(resized_sqrt_area)
            if resized_sqrt_area else None
        )
        scale_below_8 = (
            sum(value < 8.0 for value in resized_sqrt_area) / len(resized_sqrt_area)
            if resized_sqrt_area else None
        )
        split_report = {
            "images": len(image_paths),
            "decoded_images": len(split_image_records),
            "corrupt_images": split_corrupt,
            "original_width_px": _quantiles(widths),
            "original_height_px": _quantiles(heights),
            "original_aspect_ratio": _quantiles(ratios),
            "original_file_size_bytes": _quantiles(file_bytes),
            "formats": dict(formats),
            "modes": dict(modes),
            "bit_depth_categories": dict(bit_depths),
            "exif_orientation_counts": dict(exif_orientations),
            "simulated_letterbox_scale": _quantiles(resize_factors),
            "labels": {
                "label_files_found": split_label_files,
                "images_without_label_file": max(0, len(image_paths) - split_label_files),
                "annotation_rows": split_label_rows,
                "parsed_objects": split_objects,
                "coordinate_out_of_range_objects": split_out_of_range,
                "native_sqrt_area_px": _quantiles(raw_sqrt_area),
                "simulated_sqrt_area_px_at_imgsz": _quantiles(resized_sqrt_area),
                "native_max_side_px": _quantiles(raw_max_side),
                "simulated_max_side_px_at_imgsz": _quantiles(resized_max_side),
                "fraction_simulated_sqrt_area_below_4px": (
                    round(scale_below_4, 6) if scale_below_4 is not None else None
                ),
                "fraction_simulated_sqrt_area_below_8px": (
                    round(scale_below_8, 6) if scale_below_8 is not None else None
                ),
            },
        }
        split_reports[split] = split_report
        per_image.extend(split_image_records)

        if resize_factors and statistics.median(resize_factors) < 0.5:
            warnings.append(
                f"Split '{split}' median square-letterbox scale is "
                f"{statistics.median(resize_factors):.3f}; most images lose more than half "
                "their linear resolution at this imgsz."
            )
        if scale_below_4 is not None and scale_below_4 >= 0.2:
            warnings.append(
                f"Split '{split}' has {scale_below_4:.1%} of parsed objects with "
                "simulated sqrt-area below 4 px at this imgsz."
            )
        if split_label_files == 0 and image_paths:
            warnings.append(
                f"Split '{split}' has no matched YOLO label files; object-scale analysis "
                "is unavailable for this split."
            )
        elif split_label_files and split_label_files < len(image_paths):
            warnings.append(
                f"Split '{split}' has {len(image_paths) - split_label_files} images without "
                "label files; verify that these are intentional background images."
            )
        if split_out_of_range:
            warnings.append(
                f"Split '{split}' has {split_out_of_range} parsed objects with coordinates "
                "outside the expected normalized YOLO range."
            )

    # If the original COCO annotations are still present, compare their declared
    # dimensions with the actual raster on disk. This can reveal prior resizing or
    # cropping that normalized YOLO labels alone cannot prove.
    native_by_stem: dict[str, list[dict[str, Any]]] = {}
    for row in per_image:
        native_by_stem.setdefault(Path(row["path"]).stem.lower(), []).append(row)
    coco_checks: list[dict[str, Any]] = []
    for annotation_file in sorted(root.rglob("*.json")):
        if "annotations" not in annotation_file.parts and "tiny_set" not in annotation_file.name.lower():
            continue
        try:
            annotation_doc = json.loads(annotation_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        image_rows = annotation_doc.get("images") if isinstance(annotation_doc, dict) else None
        if not isinstance(image_rows, list) or not image_rows:
            continue
        metadata_rows = [
            row for row in image_rows
            if isinstance(row, dict) and "file_name" in row
            and isinstance(row.get("width"), (int, float))
            and isinstance(row.get("height"), (int, float))
        ]
        if not metadata_rows:
            continue
        matched = 0
        mismatches = []
        for metadata in metadata_rows:
            candidates = native_by_stem.get(Path(str(metadata["file_name"])).stem.lower(), [])
            if not candidates:
                continue
            matched += 1
            expected = (int(metadata["width"]), int(metadata["height"]))
            if not any((row["width"], row["height"]) == expected for row in candidates):
                mismatches.append({
                    "file_name": metadata["file_name"],
                    "annotation_width": expected[0],
                    "annotation_height": expected[1],
                    "on_disk_candidates": [
                        {"path": row["path"], "width": row["width"], "height": row["height"]}
                        for row in candidates[:3]
                    ],
                })
        check = {
            "annotation_file": str(annotation_file),
            "metadata_images": len(metadata_rows),
            "matched_by_stem": matched,
            "dimension_mismatches": len(mismatches),
            "mismatch_examples": mismatches[:10],
        }
        coco_checks.append(check)
        if mismatches:
            errors.append(
                f"Original COCO metadata dimensions disagree with raster dimensions for "
                f"{len(mismatches)} matched images in {annotation_file}; inspect coco_dimension_checks."
            )

    matched_coco_images = sum(item["matched_by_stem"] for item in coco_checks)
    if not coco_checks:
        warnings.append(
            "No source COCO dimension metadata was found. Native dimensions below refer to "
            "the stored raster files; if those files were resized before upload, this audit "
            "cannot reconstruct the pre-upload dimensions."
        )
        coco_dimension_status = "not_found"
    elif matched_coco_images == 0:
        warnings.append(
            "COCO annotation JSON was found, but none of its image names matched a stored "
            "raster by basename. Native dimensions could not be verified against source metadata."
        )
        coco_dimension_status = "unmatched"
    elif any(item["dimension_mismatches"] for item in coco_checks):
        coco_dimension_status = "mismatch"
    elif any(item["matched_by_stem"] < item["metadata_images"] for item in coco_checks):
        coco_dimension_status = "partial"
        warnings.append(
            "Only part of the source COCO image metadata matched stored rasters by basename; "
            "review coco_dimension_checks before claiming original-dimension verification."
        )
    else:
        coco_dimension_status = "verified"

    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "data_yaml": str(yaml_path),
        "dataset_root": str(root),
        "imgsz": imgsz,
        "measurement_basis": "original on-disk raster pixels before YOLO preprocessing",
        "simulated_resize": (
            "square letterbox fit scale min(imgsz/width, imgsz/height); excludes mosaic, "
            "random scale, crop, and any EXIF orientation transform"
        ),
        "dataset_signature_metadata_sha256": signature.hexdigest(),
        "split_count": len(split_reports),
        "images_total": sum(value["images"] for value in split_reports.values()),
        "decoded_images_total": sum(value["decoded_images"] for value in split_reports.values()),
        "corrupt_images_total": global_corrupt,
        "label_files_found_total": global_label_files,
        "annotation_rows_total": global_label_rows,
        "parsed_objects_total": global_objects,
        "splits": split_reports,
        "coco_dimension_checks": coco_checks,
        "coco_dimension_status": coco_dimension_status,
        "per_image": per_image,
        "warnings": warnings,
        "errors": errors,
    }

    if global_label_files == 0:
        report["warnings"].append(
            "No YOLO label files were found at all. Confirm that data.yaml points to the "
            "converted YOLO dataset and labels are paired with the original images."
        )
    parsed_object_fractions = [
        split["labels"]["fraction_simulated_sqrt_area_below_4px"]
        for split in split_reports.values()
        if split["labels"]["fraction_simulated_sqrt_area_below_4px"] is not None
    ]
    report["severe_resolution_loss"] = any(fraction >= 0.5 for fraction in parsed_object_fractions)
    report["passed"] = not errors and report["images_total"] > 0 and global_label_files > 0

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        report["report_path"] = str(destination)
    return report
