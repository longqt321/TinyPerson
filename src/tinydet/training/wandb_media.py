"""Optional W&B media logging that never interrupts training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import MethodType

import numpy as np

from tinydet.training.val_predictions import build_prediction_row
from tinydet.utils.atomic import atomic_write_text


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
    """Log final-validation tables/histograms and error analysis once at train end."""

    def __init__(self, run: Path, dataset_config: Path) -> None:
        self.run = Path(run)
        self.dataset_config = Path(dataset_config)
        self.enabled = True
        self._collect_active = False
        self._validator = None
        self._original_update_metrics = None
        self._rows: list[dict] = []
        self._confidences: list[float] = []
        self._sizes: list[float] = []

    def on_train_start(self, trainer) -> None:
        if self._original_update_metrics is not None:
            # Already wrapped: wrapping again would recurse through our own hook.
            return
        try:
            validator = getattr(trainer, "validator", None)
            if validator is not None:
                self._validator = validator
                self._original_update_metrics = validator.update_metrics
                validator.update_metrics = MethodType(self._update_metrics, validator)
        except Exception as exc:  # noqa: BLE001 - logging is optional
            self._disable(exc)

    def on_train_epoch_start(self, trainer) -> None:
        epochs = int(getattr(trainer, "epochs", 0) or 0)
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        self._collect_active = (epochs > 0 and epoch >= epochs) or bool(getattr(trainer, "stop", False))

    def _update_metrics(self, validator, preds, batch) -> None:
        original = self._original_update_metrics
        # Collect only during final_eval() (where validator.training is False) or the last epoch,
        # avoiding 149 wasted full-split GPU->CPU prediction copies across intermediate epochs.
        should_collect = self._collect_active or not getattr(validator, "training", True)
        if self.enabled and should_collect:
            try:
                if validator.seen == 0:
                    self._rows.clear()
                    self._confidences.clear()
                    self._sizes.clear()
                for index, pred in enumerate(preds):
                    row = build_prediction_row(validator, index, pred, batch)
                    if row["scores"]:
                        array = np.asarray(row["boxes"], dtype=float).reshape(-1, 4)
                        area = (array[:, 2] - array[:, 0]) * (array[:, 3] - array[:, 1])
                        self._confidences.extend(row["scores"])
                        self._sizes.extend(np.sqrt(np.clip(area, 0, None)).tolist())
                    self._rows.append(row)
            except Exception as exc:  # noqa: BLE001 - logging is optional
                self._disable(exc)
        return original(preds, batch) if original is not None else None

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
        # Compact JSON: this file is machine-read by the analysis pass, and
        # indent=2 measured 2.4x slower and 1.7x larger for no benefit.
        atomic_write_text(predictions, json.dumps({"images": self._rows}) + "\n")
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
        if self._validator is not None and self._original_update_metrics is not None:
            self._validator.update_metrics = self._original_update_metrics
            self._validator = None
            self._original_update_metrics = None
