from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from tinydet.utils.experiment import write_json


def evaluate(run: Path, split: str = "val") -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Full evaluation requires Modal GPU")
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    config = yaml.safe_load((run / "config.yaml").read_text())
    checkpoint = run / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    model = YOLO(str(checkpoint))
    result = model.val(
        data=config["data"]["config"],
        split=split,
        imgsz=config["train"]["imgsz"],
        project=str(run),
        name=f"evaluation-{split}",
        exist_ok=True,
    )
    metrics = {
        "AP": result.box.map,
        "AP50": result.box.map50,
        "AP75": result.box.map75,
        "precision": result.box.mp,
        "recall": result.box.mr,
    }
    write_json(run / "metrics" / f"{split}.json", metrics)
    if split == "test":
        dataset_file = Path(config["data"]["config"])
        dataset = yaml.safe_load(dataset_file.read_text())
        root = Path(dataset.get("path", dataset_file.parent))
        if not root.is_absolute():
            root = dataset_file.parent / root
        test_list = root / dataset["test"]
        images = [
            str(Path(line) if Path(line).is_absolute() else root / line)
            for line in test_list.read_text().splitlines()
            if line.strip()
        ]
        predictions = []
        for batch in model.predict(source=images, stream=True, verbose=False):
            predictions.append(
                {
                    "path": batch.path,
                    "boxes": batch.boxes.xyxy.cpu().tolist(),
                    "scores": batch.boxes.conf.cpu().tolist(),
                    "classes": batch.boxes.cls.cpu().tolist(),
                }
            )
        write_json(run / "predictions" / "test.json", {"images": predictions})
    return metrics
