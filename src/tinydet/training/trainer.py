from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from tinydet.models.registry import build_model
from tinydet.training.iou_logging import log_ap_by_iou
from tinydet.utils.config import load_config
from tinydet.utils.experiment import create_run, update_status
from tinydet.utils.seed import set_seed


def train(
    config_path: str | Path,
    source_git: tuple[str, bool] | None = None,
    dataset_config: str | Path | None = None,
    wandb_group: str | None = None,
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
    wandb_run = None
    try:
        model = build_model(config.model)
        from ultralytics.utils import SETTINGS

        project = config.output.get("wandb_project")
        SETTINGS.update({"wandb": bool(project)})
        if project:
            import wandb
            wandb_run = wandb.init(
                project=project,
                name=f"{config.name}-{run.name}",
                group=wandb_group,
                tags=["latest"],
                job_type="train",
                config=asdict(config),
                dir="/tmp",
            )
            print(f"W&B: {wandb_run.url}", flush=True)
            from tinydet.utils.wandb_runs import tag_latest_runs

            try:
                tag_latest_runs(wandb.Api(timeout=15), f"{wandb_run.entity}/{wandb_run.project}", config.name)
            except Exception as exc:
                print(f"Warning: W&B latest tags could not be refreshed: {exc}", flush=True)
        model.callbacks["on_fit_epoch_end"].insert(0, partial(log_ap_by_iou, wandb_enabled=bool(project)))
        training_options = {}
        if "modules" in config.model:
            from tinydet.modules.ultralytics_adapter import ModuleDetectionTrainer

            training_options["trainer"] = partial(ModuleDetectionTrainer, modules=config.model["modules"])
        model.train(**training_options, data=str(dataset_config or config.data["config"]), project=str(run), name="training", exist_ok=True, save=True, save_period=-1, seed=config.seed, **{**config.train, "plots": False})
        update_status(run, "TRAINED")
    except Exception as exc:
        update_status(run, "FAILED", str(exc))
        raise
    finally:
        if wandb_run is not None:
            import wandb

            if wandb.run is not None:
                wandb.finish()
    return run
