"""Run a checkpoint on TinyPerson test images in source coordinates (flexible path resolution)."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import orjson
import torch
from PIL import Image
from ultralytics import YOLO

from tinydet.models.ultralytics import register_ultralytics_modules
from tinydet.utils.experiment import write_json

register_ultralytics_modules()


def _resolve_test_image(image_root: Path, raw_file_name: str) -> Path:
    basename = Path(raw_file_name).name
    for candidate in (
        image_root / raw_file_name,
        image_root / "images" / "test" / basename,
        image_root / "test" / raw_file_name,
        image_root / basename,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing official test image {raw_file_name!r} under {image_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions/test.json"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 0:
        parser.error("--batch-size must be positive and --cpu-threads non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    annotation = orjson.loads(args.annotations.read_bytes())
    paths: list[Path] = []
    for row in annotation["images"]:
        path = _resolve_test_image(args.image_root, str(row["file_name"]))
        with Image.open(path) as image:
            actual = image.size
        expected = (int(row["width"]), int(row["height"]))
        if actual != expected:
            raise ValueError(
                f"Source raster mismatch for {path.name}: actual={actual}, COCO={expected}."
            )
        paths.append(path)

    if args.device in (None, "cpu") and args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)

    use_half = torch.cuda.is_available() and str(args.device or "0") != "cpu"
    model = YOLO(str(args.checkpoint))
    predictions = []
    with tempfile.TemporaryDirectory(prefix="tinyperson-test-") as temporary:
        source_list = Path(temporary) / "images.txt"
        source_list.write_text("\n".join(str(path.resolve()) for path in paths) + "\n")
        results = model.predict(
            source=str(source_list),
            stream=True,
            imgsz=args.imgsz,
            batch=args.batch_size,
            conf=args.conf,
            max_det=200,
            device=args.device,
            half=use_half,
            verbose=False,
        )
        for result in results:
            boxes = result.boxes
            predictions.append({
                "path": str(Path(result.path).resolve()),
                "boxes": boxes.xyxy.cpu().tolist() if boxes is not None else [],
                "scores": boxes.conf.cpu().tolist() if boxes is not None else [],
                "classes": [int(v) for v in boxes.cls.cpu().tolist()] if boxes is not None else [],
            })
    write_json(args.output, {"images": predictions})
    print(orjson.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "annotation": str(args.annotations.resolve()),
        "predictions": str(args.output.resolve()),
        "images_predicted": len(predictions),
        "imgsz": args.imgsz,
        "batch_size": args.batch_size,
        "confidence_threshold": args.conf,
        "max_det": 200,
    }, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
