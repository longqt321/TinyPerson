from pathlib import Path

import torch

from tinydet.models.registry import build_model
from tinydet.utils.config import load_config
from tinydet.utils.experiment import create_run, update_status
from tinydet.utils.seed import set_seed


def train(
    config_path: str | Path,
    source_git: tuple[str, bool] | None = None,
    dataset_config: str | Path | None = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires Modal GPU")
    config = load_config(config_path)
    if not config.data["config"] or not Path(config.data["config"]).is_file():
        raise ValueError(f"Dataset config does not exist: {config.data['config']}")
    if config.model.get("pretrained") and not Path(config.model["pretrained"]).is_file():
        raise ValueError(f"Pretrained checkpoint does not exist: {config.model['pretrained']}")
    set_seed(config.seed, config.train["deterministic"])
    run = create_run(config, source_git)
    update_status(run, "RUNNING")
    try:
        model = build_model(config.model)
        model.train(data=str(dataset_config or config.data["config"]), project=str(run), name="training", exist_ok=True, save=True, save_period=-1, seed=config.seed, **config.train)
        training = run / "training"
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        for name in ("best.pt", "last.pt"):
            source = training / "weights" / name
            if source.is_file():
                source.replace(checkpoints / name)
        metrics = run / "metrics"
        metrics.mkdir(exist_ok=True)
        csv = training / "results.csv"
        if csv.is_file():
            csv.replace(metrics / "train.csv")
        update_status(run, "COMPLETED")
    except Exception as exc:
        update_status(run, "FAILED", str(exc))
        raise
    return run
