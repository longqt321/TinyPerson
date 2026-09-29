import argparse
import os
import shutil
import tempfile
from pathlib import Path

import yaml


SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_CLASSES = {"earth_person", "sea_person"}


def merge_label(source: Path, destination: Path) -> None:
    output = []
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        try:
            class_id = int(fields[0])
        except ValueError as error:
            raise ValueError(f"Invalid class ID in {source}:{line_number}") from error
        if class_id not in {0, 1}:
            raise ValueError(f"Unexpected class ID {class_id} in {source}:{line_number}")
        coordinates = fields[1:]
        if len(coordinates) < 4 or len(coordinates) % 2:
            raise ValueError(f"Expected YOLO box or polygon coordinates in {source}:{line_number}")
        try:
            for coordinate in coordinates:
                float(coordinate)
        except ValueError as error:
            raise ValueError(f"Invalid coordinate in {source}:{line_number}") from error
        fields[0] = "0"
        output.append(" ".join(fields))
    destination.write_text("\n".join(output) + ("\n" if output else ""))


def read_classes(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text())
    names = config.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda key: int(key))]
    if set(names or []) != SOURCE_CLASSES:
        raise ValueError(f"Expected classes {sorted(SOURCE_CLASSES)}, got {names}")
    return names


def merge_dataset(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    source_yaml = source / "data.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(f"Missing dataset config: {source_yaml}")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    if destination == source or source in destination.parents:
        raise ValueError("Output must be outside the source dataset")
    read_classes(source_yaml)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.name}.staging-",
    ) as temporary:
        staging = Path(temporary)
        for split in SPLITS:
            source_images = source / split / "images"
            source_labels = source / split / "labels"
            if not source_images.is_dir() or not source_labels.is_dir():
                continue

            output_images = staging / split / "images"
            output_labels = staging / split / "labels"
            output_images.mkdir(parents=True)
            output_labels.mkdir(parents=True)

            for label in source_labels.glob("*.txt"):
                merge_label(label, output_labels / label.name)

            for image in source_images.iterdir():
                if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                target = output_images / image.name
                try:
                    os.link(image, target)
                except OSError:
                    shutil.copy2(image, target)

        output_config = {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 1,
            "names": ["person"],
        }
        (staging / "data.yaml").write_text(yaml.safe_dump(output_config, sort_keys=False))
        staging.rename(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge TinyTiny earth_person and sea_person classes into person"
    )
    parser.add_argument("--source", type=Path, default=Path("datasets/TinyTiny.v1i.yolo26"))
    parser.add_argument("--output", type=Path, default=Path("datasets/TinyTiny_person"))
    args = parser.parse_args()
    merge_dataset(args.source, args.output)
    print(f"Merged dataset written to {args.output}")
