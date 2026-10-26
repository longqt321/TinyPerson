"""Optional W&B media logging that never interrupts training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

FEATURE_INTERVAL_EPOCHS = 10
FEATURE_IMAGES = 2
FEATURE_SIDE = 160


def feature_heatmaps(output: torch.Tensor, count: int = FEATURE_IMAGES) -> list[np.ndarray]:
    """Return mean absolute activation maps as RGB uint8 arrays."""
    if output.ndim != 4 or output.shape[0] == 0:
        return []
    activation = output[:count].detach().float().abs().mean(1, keepdim=True)
    pooled = F.adaptive_avg_pool2d(activation, (FEATURE_SIDE, FEATURE_SIDE))
    images = []
    for heat in pooled.squeeze(1).cpu().numpy():
        low, high = float(heat.min()), float(heat.max())
        scaled = (heat - low) / (high - low) if high > low else np.zeros_like(heat)
        gray = np.asarray(scaled * 255, dtype=np.uint8)
        images.append(np.repeat(gray[..., None], 3, axis=2))
    return images


def analysis_inputs(dataset_config: Path) -> tuple[Path, Path] | None:
    """Resolve the COCO annotations and split manifest beside a dataset YAML."""
    root = dataset_config.parent
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text())
    candidate = payload.get("source_annotation_copy") or payload.get("source_annotation")
    if not candidate:
        return None
    annotations = Path(candidate)
    if not annotations.is_absolute():
        annotations = root / annotations
    return (annotations, manifest) if annotations.is_file() else None


class WandbMediaCallback:
    """Log final-validation tables/histograms, error analysis, and sampled feature maps."""

    def __init__(self, run: Path, dataset_config: Path) -> None:
        self.run = Path(run)
        self.dataset_config = Path(dataset_config)
        self.enabled = True
        self._hook = None
        self._label = "features/p2"
        self._armed = False
        self._pending: list[np.ndarray] | None = None
        self._validator = None
        self._original_update_metrics = None
        self._rows: list[dict] = []
        self._confidences: list[float] = []
        self._sizes: list[float] = []

    def on_train_start(self, trainer) -> None:
        try:
            validator = getattr(trainer, "validator", None)
            if validator is not None:
                self._validator = validator
                self._original_update_metrics = validator.update_metrics
                validator.update_metrics = MethodType(self._update_metrics, validator)
        except Exception as exc:  # noqa: BLE001 - logging is optional
            self._disable(exc)

    def on_train_epoch_start(self, trainer) -> None:
        self._armed = self._hook is not None and trainer.epoch % FEATURE_INTERVAL_EPOCHS == 0

    def _capture(self, _module, _inputs, output) -> None:
        if not self._armed or not torch.is_tensor(output):
            return
        self._armed = False
        try:
            self._pending = feature_heatmaps(output)
        except Exception as exc:  # noqa: BLE001 - logging is optional
            self._disable(exc)

    def on_train_batch_end(self, trainer) -> None:
        if not self._pending:
            return
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {self._label: [wandb.Image(image) for image in self._pending]},
                    step=trainer.epoch + 1,
                    commit=False,
                )
        except Exception as exc:  # noqa: BLE001 - logging is optional
            self._disable(exc)
        finally:
            self._pending = None

    def _update_metrics(self, validator, preds, batch) -> None:
        original = self._original_update_metrics
        if self.enabled:
            try:
                if validator.seen == 0:
                    self._rows.clear()
                    self._confidences.clear()
                    self._sizes.clear()
                for index, pred in enumerate(preds):
                    prepared = validator._prepare_batch(index, batch)
                    row = {"path": str(prepared["im_file"]), "boxes": [], "scores": [], "classes": []}
                    if pred["conf"].numel():
                        predn = validator._prepare_pred(pred)
                        scaled = validator.scale_preds(predn, prepared)
                        boxes = scaled["bboxes"].detach().cpu().numpy().tolist()
                        scores = predn["conf"].detach().cpu().numpy().tolist()
                        classes = predn["cls"].detach().cpu().numpy().tolist()
                        row.update(boxes=boxes, scores=scores, classes=classes)
                        array = np.asarray(boxes, dtype=float).reshape(-1, 4)
                        area = (array[:, 2] - array[:, 0]) * (array[:, 3] - array[:, 1])
                        self._confidences.extend(scores)
                        self._sizes.extend(np.sqrt(np.clip(area, 0, None)).tolist())
                    self._rows.append(row)
            except Exception as exc:  # noqa: BLE001 - logging is optional
                self._disable(exc)
        return original(preds, batch)

    def on_train_end(self, trainer) -> None:
        try:
            import wandb

            if wandb.run is not None:
                self._log_predictions(wandb, trainer)
                self._log_error_analysis(wandb, trainer)
        except Exception as exc:  # noqa: BLE001 - logging is optional
            self._disable(exc)
        finally:
            self.close()

    def _log_predictions(self, wandb, trainer) -> None:
        metrics = getattr(getattr(trainer.validator, "metrics", None), "box", None)
        image_metrics = getattr(metrics, "image_metrics", {}) if metrics is not None else {}
        if image_metrics:
            table = wandb.Table(
                columns=["image", "precision", "recall", "f1", "tp", "fp", "fn"]
            )
            for name, row in sorted(image_metrics.items()):
                table.add_data(
                    name, row["precision"], row["recall"], row["f1"],
                    row["tp"], row["fp"], row["fn"],
                )
            wandb.log({"val/per_image": table}, step=trainer.epoch + 1, commit=False)
        payload = {}
        if self._confidences:
            payload["val/confidence_histogram"] = wandb.Histogram(self._confidences)
        if self._sizes:
            payload["val/box_size_histogram_px"] = wandb.Histogram(self._sizes)
        if payload:
            wandb.log(payload, step=trainer.epoch + 1, commit=False)

    def _log_error_analysis(self, wandb, trainer) -> None:
        inputs = analysis_inputs(self.dataset_config)
        if inputs is None or not self._rows:
            print("W&B error analysis skipped: val predictions or manifest unavailable", flush=True)
            return
        annotations, manifest = inputs
        from tinydet.error_analysis.core import analyze_files
        from tinydet.error_analysis.report import write_outputs

        output = self.run / "analysis-val"
        output.mkdir(parents=True, exist_ok=True)
        predictions = output / "predictions.json"
        predictions.write_text(json.dumps({"images": self._rows}, indent=2) + "\n")
        result = analyze_files(
            annotations, predictions, manifest_path=manifest, split="val",
            run=self.run, checkpoint=trainer.best,
        )
        paths = write_outputs(result, output, plots=True)
        scalars = {f"analysis/{key}": value for key, value in result.summary["counts"].items()}
        scalars.update(
            {f"analysis/rate_{key}": value
             for key, value in result.summary["rates"].items() if value is not None}
        )
        scalars["analysis/valid"] = int(result.summary["integrity"]["valid"])
        wandb.log(scalars, step=trainer.epoch + 1, commit=False)
        counts = Counter((error["category"], error["subtype"]) for error in result.errors)
        table = wandb.Table(
            columns=["category", "subtype", "count"],
            data=[[category, subtype, count]
                  for (category, subtype), count in counts.most_common()],
        )
        wandb.log({"analysis/errors": table}, step=trainer.epoch + 1, commit=False)
        artifact = wandb.Artifact(name=f"analysis-{wandb.run.id}", type="analysis")
        for path in (paths["summary"], paths["report"], *paths.get("plots", [])):
            artifact.add_file(path)
        wandb.run.log_artifact(artifact)

    def _disable(self, error: Exception) -> None:
        if self.enabled:
            print(f"W&B media logging disabled after error: {error}", flush=True)
        self.enabled = False
        self.close()

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if self._validator is not None and self._original_update_metrics is not None:
            self._validator.update_metrics = self._original_update_metrics
            self._validator = None
            self._original_update_metrics = None
