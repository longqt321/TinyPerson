"""Run a checkpoint on the official TinyPerson test images in source coordinates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Original tiny_set_test_with_dense.json")
    parser.add_argument("--image-root", required=True, type=Path,
                        help="Directory containing COCO file_name paths")
    parser.add_argument("--output", type=Path, default=Path("predictions/test.json"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    annotation = json.loads(args.annotations.read_text())
    paths = []
    for row in annotation["images"]:
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
        paths.append(path)

    model = YOLO(str(args.checkpoint))
    results = model.predict(
        source=[str(path) for path in paths], stream=True, imgsz=args.imgsz,
        conf=args.conf, max_det=200, device=args.device, verbose=False,
    )
    predictions = []
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
        "imgsz": args.imgsz, "confidence_threshold": args.conf, "max_det": 200,
    }, indent=2))


if __name__ == "__main__":
    main()
