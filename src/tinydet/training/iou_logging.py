from tinydet.evaluation.iou_metrics import ap_by_iou
from tinydet.utils.csv_log import append_epoch_csv_row

_CSV_STATE: dict = {}


def log_ap_by_iou(trainer, wandb_enabled: bool = False) -> None:
    """Save AP by IoU each epoch before Ultralytics commits the W&B step."""
    validator = getattr(trainer, "validator", None)
    metrics = getattr(validator, "metrics", None) if validator is not None else None
    box = getattr(metrics, "box", None) if metrics is not None else None
    if box is None:
        return
    values = ap_by_iou(box)
    epoch = trainer.epoch + 1
    path = trainer.save_dir.parent / "metrics" / "ap_iou.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    append_epoch_csv_row(path, epoch, values, _CSV_STATE)
    if wandb_enabled:
        import wandb

        if wandb.run is not None:
            wandb.log({f"metrics/{name}(B)": value for name, value in values.items()}, commit=False)
