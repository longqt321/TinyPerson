from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import monotonic

import torch

from tinydet.models.registry import build_model
from tinydet.training.iou_logging import log_ap_by_iou
from tinydet.training.wandb_media import WandbMediaCallback
from tinydet.utils.config import load_config
from tinydet.utils.dataset import resolve_dataset_yaml
from tinydet.utils.experiment import create_run, metadata, update_status, write_json
from tinydet.utils.seed import set_seed


def _init_wandb(project: str, name: str, group: str | None, config: dict):
    try:
        import wandb

        run = wandb.init(
            project=project,
            name=name,
            group=group,
            tags=["latest"],
            job_type="train",
            config=config,
            dir="/tmp",
        )
        print(f"W&B: {run.url}", flush=True)
        return run
    except Exception as exc:  # noqa: BLE001 - W&B is optional logging
        print(f"W&B disabled after init failure: {exc}", flush=True)
        return None


def train(
    config_path: str | Path,
    source_git: tuple[str, bool] | None = None,
    dataset_config: str | Path | None = None,
    wandb_group: str | None = None,
    dataset_audit: dict | None = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires Modal GPU")
    config = load_config(config_path)
    effective_dataset_config = resolve_dataset_yaml(dataset_config or config.data["config"])
    if config.model.get("pretrained"):
        pretrained_path = Path(config.model["pretrained"])
        if not pretrained_path.is_file():
            if "checkpoints" in pretrained_path.parts:
                alt = Path(str(pretrained_path).replace("/checkpoints/", "/training/weights/"))
                if alt.is_file():
                    config.model["pretrained"] = str(alt)
            elif "training/weights" in str(pretrained_path):
                alt = Path(str(pretrained_path).replace("/training/weights/", "/checkpoints/"))
                if alt.is_file():
                    config.model["pretrained"] = str(alt)
        if not Path(config.model["pretrained"]).is_file():
            raise ValueError(f"Pretrained checkpoint does not exist: {config.model['pretrained']}")
    set_seed(config.seed, config.train["deterministic"])
    run = create_run(config, source_git)
    if dataset_audit is not None:
        write_json(run / "diagnostics" / "dataset_preflight_summary.json", dataset_audit)
        details = metadata(run)
        details["dataset_preflight"] = {
            "report_path": dataset_audit.get("report_path"),
            "source_signature": dataset_audit.get("dataset_signature_metadata_sha256"),
            "imgsz": dataset_audit.get("imgsz"),
            "images_total": dataset_audit.get("images_total"),
            "severe_resolution_loss": dataset_audit.get("severe_resolution_loss"),
        }
        write_json(run / "metadata.json", details)
    update_status(run, "RUNNING")
    wandb_run = None
    diagnostics = None
    media = None
    try:
        model = build_model(config.model)
        from ultralytics.utils import SETTINGS

        project = config.output.get("wandb_project")
        SETTINGS.update({"wandb": bool(project)})
        if project:
            wandb_run = _init_wandb(
                project, f"{config.name}-{run.name}", wandb_group, asdict(config)
            )
            if wandb_run is None:
                project = None
                SETTINGS.update({"wandb": False})
        model.callbacks["on_fit_epoch_end"].insert(
            0, partial(log_ap_by_iou, wandb_enabled=bool(project))
        )
        if project:
            media = WandbMediaCallback(run, effective_dataset_config)
            for event, callback in (
                ("on_train_start", media.on_train_start),
                ("on_train_epoch_start", media.on_train_epoch_start),
                ("on_train_batch_end", media.on_train_batch_end),
                ("on_train_end", media.on_train_end),
            ):
                model.callbacks[event].insert(0, callback)

        started = monotonic()
        torch.cuda.reset_peak_memory_stats()

        model.train(
            data=str(effective_dataset_config),
            project=str(run),
            name="training",
            exist_ok=True,
            save=True,
            save_period=-1,
            seed=config.seed,
            **{**config.train, "plots": bool(project)},
        )

        #   Ultralytics already validates the stripped best.pt at the end of training.
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
        if diagnostics is not None:
            diagnostics.close()
        if media is not None:
            media.close()
        if wandb_run is not None:
            import wandb

            if wandb.run is not None:
                wandb.finish()
    return run
