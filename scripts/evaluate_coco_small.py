"""Compute COCO bbox AP_small for the prepared TinyPerson validation split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Downloaded run directory")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Original COCO tiny_set_train.json")
    parser.add_argument("--manifest", type=Path, required=True, help="Prepared dataset manifest.json")
    parser.add_argument("--predictions", type=Path,
                        help="Defaults to RUN/evaluation-val/predictions.pt")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/metrics/coco_val.json")
    args = parser.parse_args()

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise SystemExit("Install optional evaluator dependency: pip install pycocotools") from exc

    predictions_path = args.predictions or args.run / "evaluation-val" / "predictions.pt"
    payload = torch.load(predictions_path, map_location="cpu", weights_only=True)
    manifest = json.loads(args.manifest.read_text())
    annotation_data = json.loads(args.annotations.read_text())
    split_rows = [row for row in manifest["images_manifest"] if row["split"] == "val"]
    image_by_name = {Path(item["file_name"]).name: item for item in annotation_data["images"]}
    selected = {}
    for row in split_rows:
        image = image_by_name.get(Path(row["source_file_name"]).name)
        if image is None or (int(image["width"]), int(image["height"])) != (
            row["source_width"], row["source_height"]
        ):
            raise ValueError(f"Missing or mismatched original COCO image: {row['source_file_name']}")
        selected[int(image["id"])] = image

    gt_annotations = [
        {**ann, "category_id": 1, "area": float(ann["bbox"][2]) * float(ann["bbox"][3]),
         "iscrowd": int(ann.get("iscrowd", 0))}
        for ann in annotation_data["annotations"] if int(ann["image_id"]) in selected
    ]
    gt = COCO()
    gt.dataset = {
        "images": list(selected.values()), "annotations": gt_annotations,
        "categories": [{"id": 1, "name": "person"}],
    }
    gt.createIndex()

    validator = DetectionValidator(save_dir=predictions_path.parent, args={"plots": False})
    validator.metrics.names = payload["names"]
    predictions, seen = [], set()
    for batch, pred in payload["samples"]:
        image = image_by_name.get(Path(batch["im_file"]).name)
        if image is None or int(image["id"]) not in selected:
            raise ValueError(f"Prediction is outside the manifest validation split: {batch['im_file']}")
        image_id = int(image["id"])
        if image_id in seen:
            raise ValueError(f"Duplicate validation prediction for COCO image {image_id}")
        seen.add(image_id)
        scaled = validator.scale_preds(pred, batch)
        for box, score, _cls in zip(scaled["bboxes"].tolist(), pred["conf"].tolist(),
                                    pred["cls"].tolist()):
            if int(_cls) != 0:
                raise ValueError(f"Expected person class 0, got model class {_cls}")
            x1, y1, x2, y2 = box
            predictions.append({
                "image_id": image_id, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score,
            })
    if seen != set(selected):
        raise ValueError(f"Expected {len(selected)} validation images, collected {len(seen)}")
    if not predictions:
        raise ValueError("No predictions to evaluate")

    dt = gt.loadRes(predictions)
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = sorted(selected)
    evaluator.params.catIds = [1]
    evaluator.params.areaRng = [[0, 1e10], [0, 32**2], [32**2, 96**2], [96**2, 1e10]]
    evaluator.params.areaRngLbl = ["all", "small", "medium", "large"]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    recall_small = evaluator.eval["recall"][:, :, 1, :]
    valid_recall = recall_small[recall_small >= 0]
    result = {
        "protocol": "COCO bbox AP; small area [0, 32^2] in original-image pixels",
        "images": len(selected), "ground_truths": len(gt_annotations),
        "predictions": len(predictions), "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]), "AP_small": float(evaluator.stats[3]),
        "AR_small": float(np.mean(valid_recall)) if valid_recall.size else None,
    }
    output = args.output or args.run / "metrics" / "coco_val.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
