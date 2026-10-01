from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import monotonic

import torch

from tinydet.models.registry import build_model
from tinydet.training.iou_logging import log_ap_by_iou
from tinydet.utils.config import load_config
from tinydet.utils.experiment import create_run, metadata, update_status, write_json
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
        model.callbacks["on_fit_epoch_end"].insert(0, partial(log_ap_by_iou, wandb_enabled=bool(project)))
        training_options = {}
        if "modules" in config.model:
            from tinydet.modules.ultralytics_adapter import ModuleDetectionTrainer

            training_options["trainer"] = partial(ModuleDetectionTrainer, modules=config.model["modules"])
        started = monotonic()
        torch.cuda.reset_peak_memory_stats()
        model.train(**training_options, data=str(dataset_config or config.data["config"]), project=str(run), name="training", exist_ok=True, save=True, save_period=-1, seed=config.seed, **{**config.train, "plots": False})
        # Ultralytics already validates the stripped best.pt at the end of training.
        # Keep this result for compare; explicit evaluate still performs fresh validation.
        if Path(model.trainer.best).is_file() and model.trainer.validator.metrics.box.all_ap.size:
            from tinydet.evaluation.evaluator import metric_summary

            best_metrics = metric_summary(model.trainer.validator.metrics)
            best_metrics["source"] = "training_final_best"
            write_json(run / "metrics" / "val.json", best_metrics)
        details = metadata(run)
        if wandb_run is not None:
            details["wandb_project"] = f"{wandb_run.entity}/{wandb_run.project}"
        details["training_seconds"] = monotonic() - started
        details["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
        write_json(run / "metadata.json", details)
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
