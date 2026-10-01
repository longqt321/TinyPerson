"""GPU prediction collection and CPU-only native Ultralytics metric calculation."""

from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from tinydet.analysis.heavy_bottom import dataset_images
from tinydet.evaluation.iou_metrics import ap_by_iou
from tinydet.utils.experiment import write_json


def preflight(run: Path, split: str):
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    config = yaml.safe_load((run / "config.yaml").read_text())
    checkpoint = run / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    images = dataset_images(Path(config["data"]["config"]), split)
    if not images or any(not image.is_file() for image in images):
        raise ValueError(f"Missing or empty dataset split: {split}")
    return config


class PredictionCollector(DetectionValidator):
    """Keep native preprocessing/inference/NMS; defer matching and AP to CPU."""

    def update_metrics(self, preds, batch):
        for index, pred in enumerate(preds):
            prepared = self._prepare_batch(index, batch)
            self.samples.append((
                {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                 for key, value in prepared.items()},
                {key: value.detach().cpu() for key, value in self._prepare_pred(pred).items()},
            ))

    def init_metrics(self, model):
        super().init_metrics(model)
        self.samples = []

    def get_stats(self):
        torch.save({"names": self.names, "samples": self.samples}, self.save_dir / "predictions.pt")
        return {}

    def finalize_metrics(self):
        pass

    def print_results(self):
        pass


def collect_predictions(run: Path, split: str = "val") -> Path:
    config = preflight(run, split)
    name = f"evaluation-{split}"
    YOLO(str(run / "checkpoints" / "best.pt")).val(
        validator=PredictionCollector, data=config["data"]["config"], split=split,
        imgsz=config["train"]["imgsz"], project=str(run), name=name,
        exist_ok=True, plots=False, save_json=False, save_txt=False,
    )
    return run / name / "predictions.pt"


def finalize_predictions(run: Path, artifact: Path, split: str):

    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    validator = DetectionValidator(save_dir=artifact.parent, args={"plots": False})
    validator.metrics.names = payload["names"]
    exported = []
    for batch, pred in payload["samples"]:
        cls = batch["cls"].numpy()
        validator.metrics.update_stats({
            **validator._process_batch(pred, batch), "target_cls": cls,
            "target_img": np.unique(cls), "conf": pred["conf"].numpy(),
            "pred_cls": pred["cls"].numpy(), "im_name": Path(batch["im_file"]).name,
        })
        scaled = validator.scale_preds(pred, batch)
        exported.append({"path": batch["im_file"], "boxes": scaled["bboxes"].tolist(),
                         "scores": pred["conf"].tolist(), "classes": pred["cls"].tolist()})
    validator.metrics.process(plot=False)
    result = metric_summary(validator.metrics)
    write_json(run / "metrics" / f"{split}.json", result)
    if split == "test":
        write_json(run / "predictions" / "test.json", {"images": exported})
    return result


def metric_summary(result):
    return {"AP": result.box.map, "AP50": result.box.map50, "AP75": result.box.map75,
            "precision": result.box.mp, "recall": result.box.mr, **ap_by_iou(result.box)}


def evaluate(run: Path, split: str = "val") -> dict:
    return finalize_predictions(run, collect_predictions(run, split), split)
