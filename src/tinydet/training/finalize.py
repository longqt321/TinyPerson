from pathlib import Path

from tinydet.utils.experiment import metadata, update_status


def finalize(run: Path) -> None:
    """Publish training artifacts and plot curves in a CPU-only worker."""
    if metadata(run)["status"] not in {"TRAINED", "COMPLETED"}:
        raise ValueError(f"Training has not finished: {run}")
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for name in ("best.pt", "last.pt"):
        source = run / "training" / "weights" / name
        if source.is_file():
            source.replace(checkpoints / name)
    if not (checkpoints / "best.pt").is_file():
        raise ValueError(f"Missing best checkpoint: {run}")
    metrics = run / "metrics"
    metrics.mkdir(exist_ok=True)
    source = run / "training" / "results.csv"
    if source.is_file():
        source.replace(metrics / "train.csv")
    if not (metrics / "train.csv").is_file():
        raise ValueError(f"Missing training metrics: {run}")
    from ultralytics.utils.plotting import plot_results

    plot_results(file=str(metrics / "train.csv"))
    update_status(run, "COMPLETED")
