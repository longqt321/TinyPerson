"""Run a checkpoint on the official TinyPerson test images in source coordinates."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from tinydet.models.ultralytics import register_ultralytics_modules

register_ultralytics_modules()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Original tiny_set_test_with_dense.json")
    parser.add_argument("--image-root", required=True, type=Path,
                        help="Directory containing COCO file_name paths")
    parser.add_argument("--output", type=Path, default=Path("predictions/test.json"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch; lower if device memory is limited")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0")
    parser.add_argument("--cpu-threads", type=int, default=0,
                        help="PyTorch CPU threads; 0 keeps PyTorch default")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 0:
        parser.error("--batch-size must be positive and --cpu-threads non-negative")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    annotation = json.loads(args.annotations.read_text())

    def _check_image(row: dict) -> Path:
        path = args.image_root / row["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing official test image: {path}")
        with Image.open(path) as image:
            actual = image.size
        expected = (int(row["width"]), int(row["height"]))
        if actual != expected:
            raise ValueError(
                f"Source raster mismatch for {path.name}: actual={actual}, COCO={expected}. "
                "Do not evaluate resized/reconstructed images as original-resolution TinyPerson."
            )
        return path

    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 2)) as executor:
        paths = list(executor.map(_check_image, annotation["images"]))

    if args.device in (None, "cpu") and args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)

    use_half = (args.device != "cpu" and torch.cuda.is_available()) if args.device is None else (args.device != "cpu")
    model = YOLO(str(args.checkpoint))
    predictions = []
    with tempfile.TemporaryDirectory(prefix="tinyperson-test-") as temporary:
        source_list = Path(temporary) / "images.txt"
        source_list.write_text("\n".join(str(path.resolve()) for path in paths) + "\n")
        results = model.predict(
            source=str(source_list), stream=True, imgsz=args.imgsz,
            batch=args.batch_size, conf=args.conf, max_det=200,
            device=args.device, half=use_half, verbose=False,
        )
        for result in results:
            boxes = result.boxes
            predictions.append({
                "path": str(Path(result.path).resolve()),
                "boxes": boxes.xyxy.cpu().tolist() if boxes is not None else [],
                "scores": boxes.conf.cpu().tolist() if boxes is not None else [],
                "classes": boxes.cls.cpu().tolist() if boxes is not None else [],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"images": predictions}, indent=2) + "\n")
    print(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "annotation": str(args.annotations.resolve()),
        "predictions": str(args.output.resolve()),
        "images_predicted": len(predictions),
        "imgsz": args.imgsz, "batch_size": args.batch_size,
        "confidence_threshold": args.conf, "max_det": 200,
    }, indent=2))


if __name__ == "__main__":
    main()
