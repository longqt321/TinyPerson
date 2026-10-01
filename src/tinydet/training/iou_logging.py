import csv

from tinydet.evaluation.iou_metrics import ap_by_iou


def log_ap_by_iou(trainer, wandb_enabled: bool = False) -> None:
    """Save AP by IoU each epoch before Ultralytics commits the W&B step."""
    values = ap_by_iou(trainer.validator.metrics.box)
    epoch = trainer.epoch + 1
    path = trainer.save_dir.parent / "metrics" / "ap_iou.csv"
    path.parent.mkdir(exist_ok=True)
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", *values])
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow({"epoch": epoch, **values})
    if wandb_enabled:
        import wandb

        if wandb.run is not None:
            wandb.log({f"metrics/{name}(B)": value for name, value in values.items()}, step=epoch, commit=False)
