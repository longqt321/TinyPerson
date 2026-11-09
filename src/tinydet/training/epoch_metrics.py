"""Per-epoch metric callbacks that Ultralytics does not provide out of the box.

- `GaussDiagnosticsCallback`: dumps `GauSS.get_diagnostics()` (router entropy,
  band weights, gates, sigmas, DySample offsets, optimizer control group) for
  every GauSS module each epoch into `metrics/gauss_diagnostics.csv` and W&B.
- `SystemMetricsCallback`: epoch wall time, VRAM usage, total gradient norm and
  the AMP grad-scaler scale, plus the trainer's `fitness` / `best_fitness` so
  that checkpoint selection can be audited after the fact.
- `SizeStratifiedValCallback`: runs the official TinyPerson benchmark
  (`evaluate_tinyperson`, class_mode="merge") on the per-image validation
  predictions of every epoch and records AP/MR per size bucket
  (tiny1/tiny2/tiny3/small/reasonable). This makes the paper's headline metric
  visible during training instead of only at the end.

All W&B logs use `commit=False` and rely on Ultralytics' built-in wandb
`on_fit_epoch_end` callback (`commit=True`, step = epoch+1) to flush them at
the correct step — the same pattern as `iou_logging.log_ap_by_iou`.
"""

import csv
import json
import math
import time
from pathlib import Path
from types import MethodType

import torch

from tinydet.modules.gauss import GauSS
from tinydet.training.val_predictions import build_prediction_row
from tinydet.utils.atomic import atomic_write_text
from tinydet.utils.csv_log import append_epoch_csv_row

SIZE_BUCKETS = ("tiny1", "tiny2", "tiny3", "small", "reasonable", "all")
IOU_THRESHOLDS = ("0.25", "0.50", "0.75")


class GaussDiagnosticsCallback:
    """Log GauSS internals every epoch (CSV + W&B, detached from autograd)."""

    def __init__(self, wandb_enabled: bool = False) -> None:
        self.wandb_enabled = wandb_enabled
        self.enabled = True
        self.gauss_modules: list[tuple[str, GauSS]] = []
        self._csv_state: dict = {}

    def on_train_start(self, trainer) -> None:
        model = getattr(trainer, "model", None)
        if model is None:
            return
        self.gauss_modules = [(name, m) for name, m in model.named_modules() if isinstance(m, GauSS)]

    def on_fit_epoch_end(self, trainer) -> None:
        if not self.gauss_modules or not self.enabled:
            return
        try:
            self._write(trainer)
        except Exception as exc:  # noqa: BLE001 - logging must never break training
            print(f"GauSS diagnostics logging disabled after error: {exc}", flush=True)
            self.enabled = False

    def _write(self, trainer) -> None:
        rows: dict[str, float] = {}
        for idx, (name, module) in enumerate(self.gauss_modules):
            diag = module.get_diagnostics(optimizer=getattr(trainer, "optimizer", None))
            prefix = "gauss" if len(self.gauss_modules) == 1 else f"gauss/{idx:02d}_{name}"
            for key, value in diag.items():
                rows[f"{prefix}/{key}"] = float(value)

        epoch = trainer.epoch + 1
        path = Path(trainer.save_dir).parent / "metrics" / "gauss_diagnostics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        append_epoch_csv_row(path, epoch, rows, self._csv_state)

        if self.wandb_enabled:
            import wandb

            if wandb.run is not None:
                wandb.log(rows, commit=False)


class SystemMetricsCallback:
    """Log epoch wall time, VRAM, gradient norm and AMP scaler scale per epoch.

    Ultralytics 8.4 fires no `on_before_optimizer_step` callback, so the
    gradient norm is captured by wrapping `trainer.optimizer_step` once at
    train start. Gradients are still AMP-scaled at that point, so the raw norm
    is divided by the scaler scale to recover the true gradient norm.
    """

    def __init__(self, wandb_enabled: bool = False) -> None:
        self.wandb_enabled = wandb_enabled
        self.enabled = True
        self._epoch_start: float | None = None
        self._grad_norm: float | None = None
        self._csv_state: dict = {}

    def on_train_start(self, trainer) -> None:
        if getattr(trainer, "_system_metrics_grad_hook", False):
            return
        trainer._system_metrics_grad_hook = True
        original_step = trainer.optimizer_step
        callback = self

        def optimizer_step_with_grad_norm():
            try:
                scale = trainer.scaler.get_scale() if getattr(trainer, "scaler", None) else 1.0
                grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
                # torch's own total-norm helper (same routine clip_grad_norm_ uses
                # internally, so no hand-rolled sum-of-squares).
                callback._grad_norm = float(torch.nn.utils.get_total_norm(grads)) / max(
                    float(scale), 1e-12
                )
            except Exception:  # noqa: BLE001 - metric capture must never break training
                callback._grad_norm = None
            original_step()

        trainer.optimizer_step = optimizer_step_with_grad_norm

    def on_train_epoch_start(self, trainer) -> None:
        self._epoch_start = time.monotonic()

    def on_fit_epoch_end(self, trainer) -> None:
        try:
            self._collect_and_write(trainer)
        except Exception as exc:  # noqa: BLE001 - logging must never break training
            print(f"System metrics logging disabled after error: {exc}", flush=True)
            self.enabled = False

    def _collect_and_write(self, trainer) -> None:
        rows: dict[str, float] = {}
        if self._epoch_start is not None:
            rows["system/epoch_time_s"] = time.monotonic() - self._epoch_start
            self._epoch_start = None
        if self._grad_norm is not None:
            rows["optim/grad_norm"] = self._grad_norm
            self._grad_norm = None
        scaler = getattr(trainer, "scaler", None)
        if scaler is not None:
            rows["optim/amp_scale"] = float(scaler.get_scale())
        # `trainer.fitness` is a float when validation ran, but the loss-fallback
        # path produces a numpy array; numpy 2 raises on float() of a 1-element
        # array, so coerce via `.item()` instead of crashing the callback.
        for name in ("fitness", "best_fitness"):
            value = getattr(trainer, name, None)
            if value is None:
                continue
            try:
                scalar = float(value.item() if hasattr(value, "item") else value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(scalar):
                rows[f"optim/{name}"] = scalar
        if torch.cuda.is_available():
            rows["system/vram_allocated_gb"] = torch.cuda.memory_allocated() / 2**30
            rows["system/vram_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if not rows:
            return

        epoch = trainer.epoch + 1
        path = Path(trainer.save_dir).parent / "metrics" / "system_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        append_epoch_csv_row(path, epoch, rows, self._csv_state)

        if self.wandb_enabled:
            import wandb

            if wandb.run is not None:
                wandb.log(rows, commit=False)


class SizeStratifiedValCallback:
    """Score the per-epoch validation predictions with the official TinyPerson
    benchmark and record AP/MR per size bucket.

    Predictions are collected by wrapping `validator.update_metrics` — the same
    hook `WandbMediaCallback` uses — so no extra inference pass is needed. The
    collected rows are written to `metrics/val_predictions_latest.json` and
    scored with `evaluate_tinyperson(..., class_mode="merge")`, which guarantees
    the per-epoch numbers use exactly the same protocol (dense-image exclusion,
    200-detection cap, IoD ignore handling, 101-point interpolation) as the
    final benchmark.

    Disabled automatically when the annotation JSON is missing (e.g. VisDrone)
    or when anything raises: a metric failure must never stop training.
    """

    def __init__(
        self,
        dataset_config: str | Path,
        annotation_source: str | Path,
        split: str = "val",
        class_mode: str = "merge",
        workers: int = 0,
        wandb_enabled: bool = False,
    ) -> None:
        self.dataset_config = Path(dataset_config)
        self.annotation_source = Path(annotation_source)
        self.split = split
        self.class_mode = class_mode
        self.workers = workers
        self.wandb_enabled = wandb_enabled
        self.enabled = True
        self.rows: list[dict] = []
        self._fresh = False
        self._original_update_metrics = None
        self._annotations: Path | None = None
        self._last_seconds: float | None = None

    def on_train_start(self, trainer) -> None:
        if self._original_update_metrics is not None:
            # Already wrapped: wrapping again would make the "original" point at
            # our own wrapper and recurse forever.
            return
        if not self.annotation_source.is_file():
            self._disable(f"annotation file not found: {self.annotation_source}")
            return
        try:
            from tinydet.evaluation.tinyperson_benchmark import subset_annotations_to_split

            run = Path(trainer.save_dir).parent
            self._annotations = subset_annotations_to_split(
                self.annotation_source,
                self.dataset_config,
                self.split,
                run / "diagnostics" / f"annotations_{self.split}.json",
            )
            validator = getattr(trainer, "validator", None)
            if validator is None:
                self._disable("trainer has no validator")
                return
            self._original_update_metrics = validator.update_metrics
            validator.update_metrics = MethodType(self._update_metrics, validator)
        except Exception as exc:  # noqa: BLE001 - logging is optional
            self._disable(exc)

    def _update_metrics(self, validator, preds, batch) -> None:
        original = self._original_update_metrics
        if self.enabled:
            try:
                if validator.seen == 0:
                    self.rows.clear()
                    self._fresh = True
                for index, pred in enumerate(preds):
                    self.rows.append(build_prediction_row(validator, index, pred, batch))
            except Exception as exc:  # noqa: BLE001 - logging is optional
                self._disable(exc)
        return original(preds, batch)

    def on_fit_epoch_end(self, trainer) -> None:
        if not self.enabled or self._annotations is None or not self.rows or not self._fresh:
            return
        self._fresh = False  # never score the same predictions twice
        try:
            from tinydet.evaluation.tinyperson_benchmark import evaluate_tinyperson

            run = Path(trainer.save_dir).parent
            predictions = run / "metrics" / "val_predictions_latest.json"
            atomic_write_text(predictions, json.dumps({"images": self.rows}) + "\n")

            started = time.monotonic()
            result = evaluate_tinyperson(
                self._annotations, predictions, workers=self.workers, class_mode=self.class_mode
            )
            self._last_seconds = time.monotonic() - started

            epoch = trainer.epoch + 1
            flat: dict[str, float] = {}
            records = []
            for bucket in SIZE_BUCKETS:
                values = result.get(bucket)
                if not isinstance(values, dict):
                    continue
                record = {"epoch": epoch, "bucket": bucket}
                for threshold in IOU_THRESHOLDS:
                    for metric in ("AP", "MR"):
                        key = f"{metric}@{threshold}"
                        value = values.get(key)
                        record[key] = value
                        if value is not None:
                            flat[f"val_size/{bucket}/{metric}@{threshold}"] = float(value)
                mean_ap = values.get("AP_mean_IoU_0.25_0.50_0.75")
                record["AP_mean"] = mean_ap
                record["n_gt"] = values.get("positive_ground_truths")
                if mean_ap is not None:
                    flat[f"val_size/{bucket}/AP_mean"] = float(mean_ap)
                records.append(record)

            path = run / "metrics" / "size_stratified.csv"
            fieldnames = [
                "epoch", "bucket",
                *[f"{m}@{t}" for t in IOU_THRESHOLDS for m in ("AP", "MR")],
                "AP_mean", "n_gt",
            ]
            with path.open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerows(records)

            if self.wandb_enabled:
                import wandb

                if wandb.run is not None:
                    flat["val_size/eval_seconds"] = self._last_seconds
                    wandb.log(flat, commit=False)
        except Exception as exc:  # noqa: BLE001 - logging is optional
            self._disable(exc)

    def _disable(self, error) -> None:
        if self.enabled:
            self.enabled = False
            print(f"Size-stratified val metrics disabled: {error}", flush=True)


def register_epoch_metrics(model, wandb_enabled: bool = False) -> None:
    """Attach the per-epoch metric callbacks to an Ultralytics model."""
    gauss = GaussDiagnosticsCallback(wandb_enabled)
    model.callbacks["on_train_start"].append(gauss.on_train_start)
    model.callbacks["on_fit_epoch_end"].append(gauss.on_fit_epoch_end)
    system = SystemMetricsCallback(wandb_enabled)
    model.callbacks["on_train_start"].append(system.on_train_start)
    model.callbacks["on_train_epoch_start"].append(system.on_train_epoch_start)
    model.callbacks["on_fit_epoch_end"].append(system.on_fit_epoch_end)


def register_size_stratified_metrics(
    model,
    dataset_config: str | Path,
    annotation_source: str | Path,
    split: str = "val",
    wandb_enabled: bool = False,
) -> SizeStratifiedValCallback:
    """Attach the size-stratified validation callback and return it.

    Must be registered BEFORE `WandbMediaCallback` so that both
    `update_metrics` wrappers chain in a deterministic order.
    """
    callback = SizeStratifiedValCallback(
        dataset_config=dataset_config,
        annotation_source=annotation_source,
        split=split,
        wandb_enabled=wandb_enabled,
    )
    model.callbacks["on_train_start"].append(callback.on_train_start)
    model.callbacks["on_fit_epoch_end"].append(callback.on_fit_epoch_end)
    return callback
