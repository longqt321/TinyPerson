import shutil
from pathlib import Path

from tinydet.utils.experiment import metadata, update_status


def finalize(run: Path) -> None:
    """Publish training artifacts and plot curves in a CPU-only worker."""
    if metadata(run)["status"] not in {"TRAINED", "COMPLETED"}:
        raise ValueError(f"Training has not finished: {run}")

    training = run / "training"
    checkpoints = run / "checkpoints"
    metrics = run / "metrics"

    checkpoints.mkdir(exist_ok=True)
    metrics.mkdir(exist_ok=True)

    for name in ("best.pt", "last.pt"):
        source = training / "weights" / name
        destination = checkpoints / name
        if source.is_file():
            shutil.copy2(source, destination)

    if not (checkpoints / "best.pt").is_file():
        raise ValueError(f"Missing best checkpoint: {run}")

    results = training / "results.csv"
    train_csv = metrics / "train.csv"

    if results.is_file():
        shutil.copy2(results, train_csv)

    if not train_csv.is_file():
        raise ValueError(f"Missing training metrics: {run}")

    from ultralytics.utils.plotting import plot_results

    plot_results(file=str(results), save_dir=str(metrics))

    update_status(run, "COMPLETED")
