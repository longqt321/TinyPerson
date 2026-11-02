from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from time import monotonic

import torch

from tinydet.models.registry import build_model
from tinydet.modules.dcaf import configure_dcaf_control_param_group
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


def _log_active_head(trainer) -> None:
    head = trainer.model.model[-1] if hasattr(trainer.model, "model") else trainer.model[-1]
    head_type = type(head).__name__
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print("=" * 70, flush=True)
    print(f"[VERIFIED HEAD] Active Detection Head: {head_type}", flush=True)
    print(f"[VERIFIED HEAD] Total parameters: {total_params:,} (Trainable: {trainable_params:,})", flush=True)
    print("=" * 70, flush=True)


def _setup_dcaf_optimizer(trainer) -> None:
    """Ensure DCAF scale-space and router control parameters have weight_decay=0.0 and sync scheduler."""
    configure_dcaf_control_param_group(trainer.optimizer, trainer.model)
    # If the trainer has already constructed its lr_scheduler (LambdaLR),
    # re-synchronize its lr_lambdas and base_lrs to match the updated param_groups.
    if hasattr(trainer, "scheduler") and hasattr(trainer.scheduler, "lr_lambdas"):
        sched = trainer.scheduler
        num_groups = len(trainer.optimizer.param_groups)
        while len(sched.lr_lambdas) < num_groups:
            sched.lr_lambdas.append(sched.lr_lambdas[0])
        while len(sched.base_lrs) < num_groups:
            idx = len(sched.base_lrs)
            sched.base_lrs.append(trainer.optimizer.param_groups[idx].get("initial_lr", trainer.optimizer.param_groups[idx]["lr"]))



def train(
    config_path: str | Path,
    source_git: tuple[str, bool] | None = None,
    dataset_config: str | Path | None = None,
    wandb_group: str | None = None,
    dataset_audit: dict | None = None,
    run_id: str | None = None,
    on_epoch_end_callback: Callable[[], None] | None = None,
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
    run = create_run(config, source_git, run_id=run_id)
    if dataset_audit is not None and not (run / "diagnostics" / "dataset_preflight_summary.json").is_file():
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
    started = monotonic()
    torch.cuda.reset_peak_memory_stats()

    stage2_cfg = config.stage2 or config.train.get("stage2") or config.train.get("two_stage") or {}
    is_two_stage = bool(stage2_cfg.get("enabled", False))
    train_args_s1 = {k: v for k, v in config.train.items() if k not in ("stage2", "two_stage")}

    training_done_flag = run / "training" / "completed.flag"
    training_best_pt = run / "training" / "weights" / "best.pt"
    if training_done_flag.is_file() and training_best_pt.is_file() and (run / "metadata.json").is_file():
        meta = metadata(run)
        if meta.get("status") == "TRAINED":
            print(f"Training for {config.name} at {run} is already fully completed. Skipping training.", flush=True)
            return run

    try:
        from ultralytics.utils import SETTINGS

        project = config.output.get("wandb_project")
        SETTINGS.update({"wandb": bool(project)})

        if is_two_stage:
            stage1_best = run / "stage1" / "weights" / "best.pt"
            stage1_last = run / "stage1" / "weights" / "last.pt"
            stage1_flag = run / "stage1" / "completed.flag"
            stage1_completed = stage1_best.is_file() and (stage1_flag.is_file() or (run / "metrics" / "val_stage1.json").is_file())

            if stage1_completed:
                print(
                    f"[Stage 1/2] Already completed in previous run. Reusing existing {stage1_best}...",
                    flush=True,
                )
            else:
                resume_s1 = stage1_last.is_file()
                if resume_s1:
                    print(
                        f"[Stage 1/2] Resuming interrupted Stage 1 from {stage1_last}...",
                        flush=True,
                    )
                else:
                    print("=" * 70, flush=True)
                    print(
                        f"[Stage 1/2] Launching exploration training ({train_args_s1.get('epochs', 50)} epochs) "
                        f"for {config.name}...",
                        flush=True,
                    )
                    print("=" * 70, flush=True)

                model = build_model(config.model)
                if project:
                    wandb_run = _init_wandb(
                        project, f"{config.name}-{run.name}-stage1", wandb_group, asdict(config)
                    )
                    if wandb_run is None:
                        project = None
                        SETTINGS.update({"wandb": False})

                model.callbacks["on_fit_epoch_end"].insert(
                    0, partial(log_ap_by_iou, wandb_enabled=bool(project))
                )
                model.callbacks["on_train_start"].insert(0, _log_active_head)
                model.callbacks["on_pretrain_routine_end"].append(_setup_dcaf_optimizer)
                if on_epoch_end_callback is not None:
                    model.callbacks["on_fit_epoch_end"].append(lambda trainer: on_epoch_end_callback())
                if project:
                    media = WandbMediaCallback(run, effective_dataset_config)
                    for event, callback in (
                        ("on_train_start", media.on_train_start),
                        ("on_train_epoch_start", media.on_train_epoch_start),
                        ("on_train_batch_end", media.on_train_batch_end),
                        ("on_train_end", media.on_train_end),
                    ):
                        model.callbacks[event].insert(0, callback)

                model.train(
                    data=str(effective_dataset_config),
                    project=str(run),
                    name="stage1",
                    exist_ok=True,
                    save=True,
                    save_period=-1,
                    seed=config.seed,
                    resume=resume_s1,
                    **{**train_args_s1, "plots": bool(project)},
                )

                if not stage1_best.is_file():
                    stage1_best = Path(model.trainer.best)
                if not stage1_best.is_file():
                    stage1_best = run / "stage1" / "weights" / "best.pt"
                if not stage1_best.is_file():
                    raise FileNotFoundError(f"Stage 1 best checkpoint not found under {run}/stage1/weights")

                if model.trainer.validator.metrics.box.all_ap.size:
                    from tinydet.evaluation.evaluator import metric_summary

                    s1_metrics = metric_summary(model.trainer.validator.metrics)
                    s1_metrics["source"] = "stage1_best"
                    write_json(run / "metrics" / "val_stage1.json", s1_metrics)

                (run / "stage1" / "completed.flag").write_text("done\n")
                if on_epoch_end_callback is not None:
                    on_epoch_end_callback()

                if media is not None:
                    media.close()
                    media = None
                if wandb_run is not None:
                    import wandb

                    if wandb.run is not None:
                        wandb.finish()
                    wandb_run = None

                del model
                torch.cuda.empty_cache()

            # --- STAGE 2 ---
            stage2_train_args = {**train_args_s1}
            for k, v in stage2_cfg.items():
                if k not in ("enabled", "stage2", "two_stage"):
                    stage2_train_args[k] = v

            stage2_last = run / "training" / "weights" / "last.pt"
            resume_s2 = stage2_last.is_file()

            if resume_s2:
                print(
                    f"[Stage 2/2] Resuming interrupted Stage 2 from checkpoint {stage2_last}...",
                    flush=True,
                )
            else:
                print("=" * 70, flush=True)
                print(
                    f"[Stage 2/2] Chaining fine-tuning ({stage2_train_args.get('epochs', 50)} epochs) "
                    f"for {config.name} in SAME container from {stage1_best}...",
                    flush=True,
                )
                print("=" * 70, flush=True)

            stage2_model_config = {
                "config": config.model["config"],
                "pretrained": str(stage2_last if resume_s2 else stage1_best),
            }
            model = build_model(stage2_model_config)

            if project:
                wandb_run = _init_wandb(
                    project, f"{config.name}-{run.name}", wandb_group, {**asdict(config), "stage2": stage2_train_args}
                )
                if wandb_run is None:
                    project = None
                    SETTINGS.update({"wandb": False})

            model.callbacks["on_fit_epoch_end"].insert(
                0, partial(log_ap_by_iou, wandb_enabled=bool(project))
            )
            model.callbacks["on_train_start"].insert(0, _log_active_head)
            model.callbacks["on_pretrain_routine_end"].append(_setup_dcaf_optimizer)
            if on_epoch_end_callback is not None:
                model.callbacks["on_fit_epoch_end"].append(lambda trainer: on_epoch_end_callback())
            if project:
                media = WandbMediaCallback(run, effective_dataset_config)
                for event, callback in (
                    ("on_train_start", media.on_train_start),
                    ("on_train_epoch_start", media.on_train_epoch_start),
                    ("on_train_batch_end", media.on_train_batch_end),
                    ("on_train_end", media.on_train_end),
                ):
                    model.callbacks[event].insert(0, callback)

            model.train(
                data=str(effective_dataset_config),
                project=str(run),
                name="training",
                exist_ok=True,
                save=True,
                save_period=-1,
                seed=config.seed,
                resume=resume_s2,
                **{**stage2_train_args, "plots": bool(project)},
            )
        else:
            last_pt = run / "training" / "weights" / "last.pt"
            resume = last_pt.is_file()
            if resume:
                print(f"Resuming interrupted training from {last_pt}...", flush=True)
                model_cfg = {"config": config.model["config"], "pretrained": str(last_pt)}
            else:
                model_cfg = config.model

            model = build_model(model_cfg)
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
            model.callbacks["on_train_start"].insert(0, _log_active_head)
            model.callbacks["on_pretrain_routine_end"].append(_setup_dcaf_optimizer)
            if on_epoch_end_callback is not None:
                model.callbacks["on_fit_epoch_end"].append(lambda trainer: on_epoch_end_callback())
            if project:
                media = WandbMediaCallback(run, effective_dataset_config)
                for event, callback in (
                    ("on_train_start", media.on_train_start),
                    ("on_train_epoch_start", media.on_train_epoch_start),
                    ("on_train_batch_end", media.on_train_batch_end),
                    ("on_train_end", media.on_train_end),
                ):
                    model.callbacks[event].insert(0, callback)

            model.train(
                data=str(effective_dataset_config),
                project=str(run),
                name="training",
                exist_ok=True,
                save=True,
                save_period=-1,
                seed=config.seed,
                resume=resume,
                **{**train_args_s1, "plots": bool(project)},
            )

        (run / "training" / "completed.flag").write_text("done\n")
        if on_epoch_end_callback is not None:
            on_epoch_end_callback()

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
        if is_two_stage:
            details["two_stage"] = True
            details["stage1_best"] = str(stage1_best)
            details["stage2_epochs"] = stage2_train_args.get("epochs")
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
