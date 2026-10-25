"""Generate source-coordinate prediction JSON for one YOLO dataset split."""

from __future__ import annotations

import json
from pathlib import Path

from tinydet.utils.dataset import dataset_images


def predict_split(
    checkpoint: str | Path,
    data: str | Path,
    split: str,
    output: str | Path,
    *,
    imgsz: int = 1280,
    conf: float = 0.001,
    batch_size: int = 16,
    limit: int = 0,
    device: str | None = None,
    cpu_threads: int = 0,
) -> dict:
    if imgsz <= 0 or batch_size <= 0 or not 0.0 <= conf <= 1.0 or limit < 0:
        raise ValueError("invalid inference parameters")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    images = sorted(dataset_images(Path(data), split), key=lambda path: path.name)
    if limit:
        images = images[:limit]
    if not images:
        raise ValueError(f"No images found for split {split!r}")
    if cpu_threads:
        import torch

        torch.set_num_threads(cpu_threads)
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    rows = []
    results = model.predict(
        source=[str(path) for path in images],
        stream=True,
        imgsz=imgsz,
        batch=batch_size,
        conf=conf,
        max_det=200,
        device=device,
        verbose=False,
    )
    for result in results:
        boxes = result.boxes
        rows.append({
            "path": str(Path(result.path).resolve()),
            "boxes": boxes.xyxy.cpu().tolist() if boxes is not None else [],
            "scores": boxes.conf.cpu().tolist() if boxes is not None else [],
            "classes": [int(value) for value in boxes.cls.cpu().tolist()] if boxes is not None else [],
        })
    if len(rows) != len(images):
        raise RuntimeError(f"Predicted {len(rows)} images, expected {len(images)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    inference = {
        "checkpoint": str(checkpoint),
        "data": str(Path(data).resolve()),
        "split": split,
        "images": len(rows),
        "imgsz": imgsz,
        "batch_size": batch_size,
        "confidence_threshold": conf,
        "max_det": 200,
        "device": device,
        "limit": limit,
    }
    output.write_text(json.dumps({"images": rows, "inference": inference}, indent=2) + "\n")
    return {**inference, "predictions": str(output.resolve())}
