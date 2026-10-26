"""GPU prediction collection and CPU-only native Ultralytics metric calculation.

Evaluation is a separate lifecycle from training: it never imports the training
package and only reads run artifacts. Checkpoints are discovered rather than
assumed, because runs published before `finalize` keep weights under
`training/weights/` while finalized runs also expose `checkpoints/`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator

from tinydet.evaluation.iou_metrics import ap_by_iou
from tinydet.utils.dataset import dataset_images, normalize_dataset_yaml
from tinydet.utils.experiment import write_json

# Evaluation writes into a namespaced folder so repeated runs never overwrite
# artifacts produced by training or by another evaluation configuration.
EVALUATION_ROOT = "evaluation"

CHECKPOINT_LOCATIONS = (
    ("checkpoints", "best.pt"),
    ("training", "weights", "best.pt"),
)


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(run: Path, checkpoint: str | Path | None = None) -> Path:
    """Return the `best.pt` to evaluate, scanning every known location.

    An explicit path wins. Otherwise all known locations are scanned; a run that
    exposes more than one distinct `best.pt` is rejected instead of guessed.
    """
    if checkpoint is not None:
        explicit = Path(checkpoint)
        if not explicit.is_absolute():
            explicit = run / explicit
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    found = [
        run.joinpath(*location)
        for location in CHECKPOINT_LOCATIONS
        if run.joinpath(*location).is_file()
    ]
    if not found:
        raise FileNotFoundError(f"No best.pt under {run}/checkpoints or {run}/training/weights")
    if len({sha256(path) for path in found}) > 1:
        raise ValueError(
            f"Run {run.name} exposes distinct best.pt files "
            f"({', '.join(path.relative_to(run).as_posix() for path in found)}); "
            "pass checkpoint= explicitly to select one"
        )
    return found[0]


def preflight(run: Path, split: str, checkpoint: str | Path | None = None):
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    config = yaml.safe_load((run / "config.yaml").read_text())
    config["data"]["config"] = str(normalize_dataset_yaml(config["data"]["config"]))
    config["checkpoint"] = str(resolve_checkpoint(run, checkpoint))
    images = dataset_images(Path(config["data"]["config"]), split)
    if not images or any(not image.is_file() for image in images):
        raise ValueError(f"Missing or empty dataset split: {split}")
    return config


class PredictionCollector(DetectionValidator):
    """Keep native preprocessing/inference/NMS; defer matching and AP to CPU."""

    def update_metrics(self, preds, batch):
        for index, pred in enumerate(preds):
            prepared = self._prepare_batch(index, batch)
            self.samples.append(
                (
                    {
                        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in prepared.items()
                    },
                    {key: value.detach().cpu() for key, value in self._prepare_pred(pred).items()},
                )
            )

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


def collect_predictions(
    run: Path, split: str = "val", checkpoint: str | Path | None = None
) -> Path:
    config = preflight(run, split, checkpoint)
    name = f"{EVALUATION_ROOT}-{split}"
    YOLO(config["checkpoint"]).val(
        validator=PredictionCollector,
        data=config["data"]["config"],
        split=split,
        imgsz=config["train"]["imgsz"],
        project=str(run),
        name=name,
        exist_ok=True,
        plots=False,
        save_json=False,
        save_txt=False,
    )
    return run / name / "predictions.pt"


def finalize_predictions(run: Path, artifact: Path, split: str):

    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    validator = DetectionValidator(save_dir=artifact.parent, args={"plots": False})
    validator.metrics.names = payload["names"]
    exported = []
    for batch, pred in payload["samples"]:
        cls = batch["cls"].numpy()
        validator.metrics.update_stats(
            {
                **validator._process_batch(pred, batch),
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": pred["conf"].numpy(),
                "pred_cls": pred["cls"].numpy(),
                "im_name": Path(batch["im_file"]).name,
            }
        )
        if split == "test":
            scaled = validator.scale_preds(pred, batch)
            exported.append(
                {
                    "path": batch["im_file"],
                    "boxes": scaled["bboxes"].tolist(),
                    "scores": pred["conf"].tolist(),
                    "classes": pred["cls"].tolist(),
                }
            )
    validator.metrics.process(plot=False)
    result = metric_summary(validator.metrics)
    write_json(run / "metrics" / f"{split}.json", {**result, "source": "evaluation"})
    if split == "test":
        write_json(run / "predictions" / "test.json", {"images": exported})
    return result


def metric_summary(result):
    return {
        "AP": result.box.map,
        "AP50": result.box.map50,
        "AP75": result.box.map75,
        "precision": result.box.mp,
        "recall": result.box.mr,
        **ap_by_iou(result.box),
    }


def evaluate(run: Path, split: str = "val", checkpoint: str | Path | None = None) -> dict:
    return finalize_predictions(run, collect_predictions(run, split, checkpoint), split)
