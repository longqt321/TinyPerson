"""Per-epoch diagnostics for P2Refine modules attached to an Ultralytics trainer.

GauSS ran 21 trainings without its diagnostics ever being logged; this callback
ensures P2Refine does not repeat that.

Ultralytics re-creates ``DetectionModel`` inside the trainer, so hooks must attach
to ``trainer.model`` on ``on_train_start``. At the start of each epoch, the first
forward pass records residual/stem ratio and cosine alignment under ``torch.no_grad()``.
At ``on_fit_epoch_end``, statistics and ``w_out_norm`` are written to
``metrics/p2refine_diagnostics.csv`` and logged to W&B with ``commit=False``
(same convention as ``log_ap_by_iou``).
"""

import csv
from pathlib import Path

import torch

from tinydet.modules.p2_refine import P2Refine


class P2RefineDiagnosticsCallback:
    """Ultralytics lifecycle callback tracking P2Refine dynamics."""

    def __init__(self, wandb_enabled: bool = False) -> None:
        self.wandb_enabled = wandb_enabled
        self.refines: list[tuple[str, P2Refine]] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._needs_update = True

    def on_train_start(self, trainer) -> None:
        model = getattr(trainer, "model", None)
        if model is None:
            return
        self.refines = [(name, m) for name, m in model.named_modules() if isinstance(m, P2Refine)]
        self._handles.clear()
        for _, mod in self.refines:
            handle = mod.register_forward_pre_hook(self._capture_hook)
            self._handles.append(handle)
        self._needs_update = True

    def on_train_epoch_start(self, trainer) -> None:
        self._needs_update = True

    def _capture_hook(self, module, args):
        if not self._needs_update:
            return
        inputs = args[0]
        with torch.no_grad():
            module._pending_diag = module.get_diagnostics([t.detach().float() for t in inputs])
        self._needs_update = False

    def on_fit_epoch_end(self, trainer) -> None:
        if not self.refines:
            return
        rows: dict[str, float] = {}
        for idx, (name, module) in enumerate(self.refines):
            diag = getattr(module, "_pending_diag", None)
            if diag is None:
                # Fallback if no forward occurred (e.g. zero batches)
                with torch.no_grad():
                    w_norm = module.out.weight.norm().item()
                diag = {
                    "rho_mean": 0.0,
                    "rho_max": 0.0,
                    "cos_x0_delta": 0.0,
                    "w_out_norm": w_norm,
                }
            prefix = "p2refine" if len(self.refines) == 1 else f"p2refine/{idx:02d}_{name}"
            for key, value in diag.items():
                rows[f"{prefix}/{key}"] = float(value)

        epoch = trainer.epoch + 1
        path = Path(trainer.save_dir).parent / "metrics" / "p2refine_diagnostics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["epoch", *sorted(rows)]
        with path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow({"epoch": epoch, **rows})

        if self.wandb_enabled:
            import wandb

            if wandb.run is not None:
                wandb.log(rows, commit=False)

    def on_train_end(self, trainer) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def register_p2refine_callbacks(model, wandb_enabled: bool = False) -> None:
    """Register P2Refine lifecycle hooks into Ultralytics model.callbacks dictionary."""
    cb = P2RefineDiagnosticsCallback(wandb_enabled=wandb_enabled)
    model.callbacks["on_train_start"].append(cb.on_train_start)
    model.callbacks["on_train_epoch_start"].append(cb.on_train_epoch_start)
    model.callbacks["on_fit_epoch_end"].append(cb.on_fit_epoch_end)
    model.callbacks["on_train_end"].append(cb.on_train_end)
