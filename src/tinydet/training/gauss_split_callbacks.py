"""Trainer callbacks for a model that carries a DetectGaussSplit head.

Two jobs, both required for the experiment to mean what it claims:

1. **Keep the frozen detector frozen.** Ultralytics freezes *parameters* via
   ``freeze=`` but still calls ``model.train()``, which puts every
   ``BatchNorm`` back into training mode. The running statistics then drift on
   the frozen backbone, so the "frozen detector + trained head" premise breaks
   and the head is evaluated against a detector that has silently changed.
   Resetting BN to eval at every epoch start keeps the detector bit-identical
   to the baseline checkpoint.

2. **Log the head's own health.** ``gs_loss`` already reaches ``results.csv``
   and W&B through the criterion's loss dict; the per-batch ``p(second)`` and
   positive rate do not. Kill criterion K3 ("the head found no signal") is read
   from exactly these two numbers, so they must be recorded during the run, not
   reconstructed afterwards.
"""

from __future__ import annotations

from torch import nn

from tinydet.modules.gauss_split_detect import find_gauss_split_head


class GaussSplitCallback:
    """BN-eval guard plus head diagnostics for a DetectGaussSplit model."""

    def __init__(self, wandb_enabled: bool = False) -> None:
        self.wandb_enabled = bool(wandb_enabled)
        self._p_second: list[float] = []
        self._pos_rate: list[float] = []
        self._n_det: list[float] = []

    # ------------------------------------------------------------------ guards
    @staticmethod
    def _freeze_bn(model: nn.Module) -> int:
        n = 0
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                n += 1
        return n

    @staticmethod
    def _freeze_detector(model: nn.Module) -> tuple[int, int]:
        """Train only ``*.head.*``; freeze every other parameter.

        Runs at ``on_train_start`` because Ultralytics' own ``freeze=`` works by
        layer index and cannot express "this layer, except its ``head`` child".
        The optimizer is already built at this point, but AdamW skips parameters
        whose ``grad`` is ``None``, so frozen parameters never update.
        """
        frozen = trainable = 0
        for name, p in model.named_parameters():
            keep = ".head." in name
            p.requires_grad_(keep)
            if keep:
                trainable += p.numel()
            else:
                frozen += p.numel()
        return frozen, trainable

    def on_train_start(self, trainer) -> None:
        model = trainer.model
        frozen, trainable = self._freeze_detector(model)
        head = find_gauss_split_head(getattr(model, "model", model))
        print(
            f"[gauss-split] frozen detector params: {frozen:,} | trainable: {trainable:,} | "
            f"head: {sum(p.numel() for p in head.head.parameters()):,}",
            flush=True,
        )

    def on_train_epoch_start(self, trainer) -> None:
        n_bn = self._freeze_bn(trainer.model)
        if trainer.epoch == 0:
            print(f"[gauss-split] BN modules forced to eval: {n_bn}", flush=True)
        self._p_second.clear()
        self._pos_rate.clear()
        self._n_det.clear()

    def on_train_batch_start(self, trainer) -> None:
        # Ultralytics' ``_model_train`` runs *after* ``on_train_epoch_start`` and
        # re-puts every BatchNorm back into train mode (its own BN guard only
        # covers ``freeze_layer_names``, i.e. layers 0..freeze-1 — not the head
        # layer). Re-asserting eval here is the only hook guaranteed to run
        # after it and before the forward pass, so the frozen detector's
        # running statistics stay bit-identical to the loaded checkpoint.
        self._freeze_bn(trainer.model)

    # -------------------------------------------------------------- diagnostics
    def on_train_batch_end(self, trainer) -> None:
        head = find_gauss_split_head(getattr(trainer.model, "model", trainer.model))
        if head is None or not head.last_stats:
            return
        self._p_second.append(float(head.last_stats.get("p_second_mean", 0.0)))
        self._pos_rate.append(float(head.last_stats.get("pos_rate", 0.0)))
        self._n_det.append(float(head.last_stats.get("n_det", 0.0)))

    def on_fit_epoch_end(self, trainer) -> None:
        if not self._p_second:
            return
        metrics = {
            "gauss_split/p_second_init": self._p_second[0],
            "gauss_split/p_second_mean": sum(self._p_second) / len(self._p_second),
            "gauss_split/pos_rate": sum(self._pos_rate) / len(self._pos_rate),
            "gauss_split/dets_per_image": sum(self._n_det) / len(self._n_det) / max(trainer.batch_size, 1),
        }
        print(
            "  [gauss-split] " + " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items()),
            flush=True,
        )
        if self.wandb_enabled:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(metrics, step=trainer.epoch)
            except Exception as exc:  # noqa: BLE001 - W&B is optional logging
                print(f"[gauss-split] W&B log failed (ignored): {exc}", flush=True)


def register_gauss_split_callbacks(model, wandb_enabled: bool = False) -> bool:
    """Register the callbacks if the model has a DetectGaussSplit head.

    Returns True when registered, so the caller can log whether the experiment
    is actually running the module it thinks it is.
    """
    if find_gauss_split_head(getattr(model, "model", model)) is None:
        return False
    cb = GaussSplitCallback(wandb_enabled=wandb_enabled)
    # on_train_start must run BEFORE _log_active_head so the printed trainable
    # count reflects the freeze.
    model.callbacks["on_train_start"].insert(0, cb.on_train_start)
    model.callbacks["on_train_epoch_start"].append(cb.on_train_epoch_start)
    model.callbacks["on_train_batch_start"].append(cb.on_train_batch_start)
    model.callbacks["on_train_batch_end"].append(cb.on_train_batch_end)
    model.callbacks["on_fit_epoch_end"].append(cb.on_fit_epoch_end)
    return True
