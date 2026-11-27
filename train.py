"""Environment-agnostic training CLI for TinyDet experiments (Modal, Colab, Kaggle, local GPU).

Single entry point with full Modal parity:
1. Auto-detects host (``colab``, ``kaggle``, ``modal``, ``local``).
2. Auto-discovers dataset ``data.yaml`` (and stages Google Drive FUSE datasets to
   Colab local SSD ``/content/data_local`` to avoid I/O bottlenecks).
3. Auto-selects a persistent writable ``output_root`` (Drive on Colab,
   ``/kaggle/working/runs`` on Kaggle, ``./runs`` locally).
4. Auto-authenticates W&B from env, Colab ``userdata``, or Kaggle ``UserSecretsClient``,
   using deterministic run IDs (``resume="allow"``) and ``host:<name>`` tags for
   full synchronization with Modal runs.
5. Runs dataset preflight audit (``audit_dataset``) and model sanity checks before
   training, logs all artifacts/metrics, and calls ``finalize(run)`` upon completion.

Examples
--------
One-command run on Google Colab / Kaggle after ``git clone``::

    bash scripts/run_gpu.sh --config configs/experiment/lgt_tinyperson_150e.yaml

CPU dry-run validation (no GPU required)::

    uv run --extra cpu python train.py \
        --config configs/experiment/lgt_tinyperson_150e.yaml \
        --data datasets/derived/tinyperson-yolo-v2/data.yaml --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from tinydet.utils.config import ExperimentConfig, load_config
from tinydet.utils.runtime import (
    DEFAULT_HF_DATASET_REPO,
    default_drive_backup_root,
    default_hf_runs_repo,
    default_output_root,
    detect_host,
    discover_dataset,
    ensure_hf_auth,
    ensure_wandb_auth,
    restore_run_from_drive,
    restore_run_from_hf,
    stable_run_id,
    stage_dataset_to_local_ssd,
    sync_run_to_drive,
    sync_run_to_hf,
)

REPO_ROOT = Path(__file__).resolve().parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TinyDet experiment on any CUDA GPU (Colab, Kaggle, local, Modal).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/experiment/lgt_tinyperson_150e.yaml",
        help="Experiment YAML.",
    )
    parser.add_argument("--data", type=Path, help="Override data.config (file or directory).")
    parser.add_argument(
        "--hf-dataset",
        default=os.environ.get("TINYDET_HF_DATASET", DEFAULT_HF_DATASET_REPO),
        help="Hugging Face dataset repo to auto-download if dataset is not present locally. "
        "Pass empty string '' to disable.",
    )
    parser.add_argument(
        "--hf-runs-repo",
        help="Hugging Face model repo for automatic checkpoint backup & resume "
        "(defaults to <hf_user>/tinydet-runs when HF_TOKEN is set).",
    )
    parser.add_argument(
        "--no-hf-sync",
        action="store_true",
        help="Disable automatic Hugging Face Hub checkpoint backup & resume.",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=5,
        help="Sync checkpoints to Hugging Face Hub / Google Drive every N epochs.",
    )
    parser.add_argument(
        "--data-root", type=Path, help="Override the 'path' key inside the dataset YAML."
    )
    parser.add_argument("--output-root", type=Path, help="Override output.root (run directory).")
    parser.add_argument("--name", help="Override the experiment name (run folder).")
    parser.add_argument(
        "--run-id",
        help="Run folder name under <output-root>/<name>/. Defaults to a stable host+seed ID "
        "on Colab/Kaggle so rerunning the same command resumes automatically.",
    )
    parser.add_argument("--device", help="CUDA device list, e.g. '0' or '0,1'.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs.")
    parser.add_argument("--batch", type=int, help="Override train.batch.")
    parser.add_argument("--imgsz", type=int, help="Override train.imgsz.")
    parser.add_argument("--workers", type=int, help="Override train.workers.")
    parser.add_argument("--seed", type=int, help="Override seed.")
    parser.add_argument("--wandb-project", help="Override W&B project name.")
    parser.add_argument("--wandb-group", help="Optional W&B group name to match Modal campaigns.")
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging for this run."
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="Do not copy Google Drive datasets to Colab local SSD.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip dataset preflight audit before training.",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Skip post-training finalize() step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight checks (config, dataset, model) on CPU and exit without training.",
    )
    return parser.parse_args(argv)


def _resolve_model_config(value: str | Path, experiment_path: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    bases = (
        Path.cwd(),
        REPO_ROOT,
        experiment_path.resolve().parent,
        experiment_path.resolve().parent.parent,
    )
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return (REPO_ROOT / candidate).resolve()


def _resolve_config(
    config: ExperimentConfig,
    args: argparse.Namespace,
    experiment_path: Path,
    host: str,
) -> ExperimentConfig:
    model = dict(config.model)
    model["config"] = str(_resolve_model_config(model["config"], experiment_path))
    if not Path(model["config"]).is_file():
        raise FileNotFoundError(f"Model config does not exist: {model['config']}")
    if model.get("pretrained"):
        model["pretrained"] = str(Path(model["pretrained"]).expanduser().resolve())

    data = dict(config.data)
    if args.data is not None:
        data["config"] = str(args.data.expanduser().resolve())

    output = dict(config.output)
    if args.output_root is not None:
        output["root"] = str(args.output_root.expanduser().resolve())
    elif host != "modal" and str(output.get("root", "")).startswith("/mnt/"):
        output["root"] = str(default_output_root(host, REPO_ROOT))

    if args.wandb_project is not None:
        output["wandb_project"] = args.wandb_project
    if args.no_wandb:
        output["wandb_project"] = None

    train = dict(config.train)
    for key in ("epochs", "batch", "imgsz", "workers"):
        value = getattr(args, key)
        if value is not None:
            train[key] = value
    if args.device is not None:
        train["device"] = args.device
    if host in {"colab", "kaggle"}:
        if args.workers is None:
            train["workers"] = min(int(train.get("workers", 4)), os.cpu_count() or 2)
        train.setdefault("cache", "disk")

    seed = config.seed if args.seed is None else args.seed
    name = config.name if args.name is None else args.name
    return replace(
        config, name=name, seed=seed, model=model, data=data, train=train, output=output
    )


def _print_summary(
    config: ExperimentConfig,
    dataset_yaml: Path,
    host: str,
    run_id: str | None,
    wandb_group: str | None,
) -> None:
    print("=" * 70)
    print(f"Host       : {host}")
    print(f"Experiment : {config.name}")
    print(f"Run ID     : {run_id or '(auto-timestamp)'}")
    print(f"Seed       : {config.seed}")
    print(f"Model YAML : {config.model['config']}")
    print(f"Dataset    : {dataset_yaml}")
    print(f"Run root   : {config.output['root']}")
    print(f"W&B        : {config.output.get('wandb_project') or 'disabled'}"
          f"{f' (group={wandb_group})' if wandb_group else ''}")
    for key in ("epochs", "imgsz", "batch", "workers", "optimizer", "lr0", "device"):
        if key in config.train:
            print(f"{key:<11}: {config.train[key]}")
    print("=" * 70, flush=True)


def _run_model_preflight(config: ExperimentConfig, dataset_yaml: Path) -> None:
    """Validate model build, parameter count, strides, and dataset split visibility."""
    from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params

    from tinydet.models.registry import build_model
    from tinydet.utils.dataset import dataset_images

    model = build_model(config.model)
    inner = model.model
    imgsz = int(config.train["imgsz"])
    layers = len([m for m in inner.modules() if len(m._modules) == 0])
    params = get_num_params(inner)
    gradients = get_num_gradients(inner)
    gflops = get_flops(inner, imgsz)
    strides = [int(s) for s in inner.model[-1].stride.tolist()]
    train_images = len(dataset_images(dataset_yaml, "train"))
    val_images = len(dataset_images(dataset_yaml, "val"))
    print(
        f"[preflight] model: layers={layers}, params={params:,}, grads={gradients:,}, "
        f"GFLOPs={gflops:.3f}@{imgsz}, strides={strides}",
        flush=True,
    )
    print(f"[preflight] dataset splits: train={train_images}, val={val_images}", flush=True)
    if train_images == 0:
        raise ValueError(f"Training split has 0 images in {dataset_yaml}")


def _run_dataset_preflight(config: ExperimentConfig, dataset_yaml: Path) -> dict:
    """Run the same ``audit_dataset`` preflight that Modal executes before training."""
    from tinydet.data.image_audit import audit_dataset

    imgsz = int(config.train["imgsz"])
    workers = min(16, max(2, os.cpu_count() or 4))
    print(f"[preflight] auditing dataset images/labels (imgsz={imgsz}, workers={workers})...", flush=True)
    report = audit_dataset(dataset_yaml, imgsz, workers=workers)
    summary = {
        "imgsz": imgsz,
        "images_total": report.get("images_total"),
        "severe_resolution_loss": report.get("severe_resolution_loss"),
        "dataset_signature_metadata_sha256": report.get("dataset_signature_metadata_sha256"),
    }
    print(
        f"[preflight] dataset audit OK: images={summary['images_total']}, "
        f"signature={str(summary['dataset_signature_metadata_sha256'])[:12]}...",
        flush=True,
    )
    return summary


def _check_output_root(config: ExperimentConfig) -> None:
    root = Path(config.output["root"])
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".tinydet-write-probe"
        probe.write_text("")
        probe.unlink()
    except OSError as error:
        raise SystemExit(
            f"Output root is not writable: {root} ({error}).\n"
            "Pass --output-root to point the run directory at a writable path, e.g.\n"
            "  --output-root /content/drive/MyDrive/tinydet_runs   (Colab, Drive)\n"
            "  --output-root /kaggle/working/runs                  (Kaggle)\n"
            "  --output-root ~/runs                                (any host)"
        ) from error


def _stream_logs_unbuffered() -> None:
    """Flush every line to the notebook cell immediately (Colab/Kaggle pipe stdout)."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True, write_through=True)


def main(argv: list[str] | None = None) -> int:
    _stream_logs_unbuffered()
    args = _parse_args(argv)
    host = detect_host()
    experiment_path = Path(args.config).expanduser().resolve()
    config = _resolve_config(load_config(experiment_path), args, experiment_path, host)

    from tinydet.utils.dataset import normalize_dataset_yaml

    hf_token, hf_user = ensure_hf_auth()
    hf_repo = args.hf_dataset.strip() or None
    discovered = discover_dataset(
        args.data, config.data.get("config"), host, REPO_ROOT, hf_repo=hf_repo
    )
    staged = discovered if args.no_stage else stage_dataset_to_local_ssd(discovered, host)
    dataset_yaml = normalize_dataset_yaml(staged, root=args.data_root)

    effective_run_id = args.run_id
    if effective_run_id is None and host in {"colab", "kaggle"}:
        effective_run_id = stable_run_id(config.seed, host)

    _print_summary(config, dataset_yaml, host, effective_run_id, args.wandb_group)
    _run_model_preflight(config, dataset_yaml)

    dataset_audit: dict | None = None
    if not args.skip_audit:
        dataset_audit = _run_dataset_preflight(config, dataset_yaml)

    if args.dry_run:
        print("[DRY RUN] config, dataset audit, and model all passed. No training performed.", flush=True)
        return 0

    import torch

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Training requires a CUDA GPU; CPU training is "
            "intentionally disabled. Use --dry-run to validate the configuration, or "
            "enable a GPU runtime (Colab T4/L4/A100, Kaggle T4/P100, Modal A10/L4).",
            file=sys.stderr,
        )
        return 1

    print(
        f"[preflight] CUDA GPU: {torch.cuda.get_device_name(0)} "
        f"(torch {torch.__version__}, CUDA {torch.version.cuda})",
        flush=True,
    )

    if config.output.get("wandb_project"):
        key = ensure_wandb_auth()
        status = "authenticated" if key else "no WANDB_API_KEY found (W&B may disable itself)"
        print(f"[preflight] W&B: {status}", flush=True)

    _check_output_root(config)

    from tinydet.training.finalize import finalize
    from tinydet.training.trainer import train

    drive_backup = default_drive_backup_root(host)
    hf_runs_repo = None if args.no_hf_sync else (args.hf_runs_repo or default_hf_runs_repo(hf_user))
    run_local = (
        Path(config.output["root"]) / config.name / effective_run_id
        if effective_run_id
        else None
    )
    print(f"[preflight] HF Auth: {'authenticated (' + hf_user + ')' if hf_user else ('token set' if hf_token else 'not set')}", flush=True)
    print(f"[preflight] HF Backup: {hf_runs_repo or 'disabled'}", flush=True)
    print(f"[preflight] Drive Sync: {drive_backup or 'disabled'}", flush=True)
    if host in {"colab", "kaggle"} and not hf_runs_repo and not drive_backup:
        print(
            "[preflight] WARNING: ephemeral runtime with no HF Backup or Drive Sync — "
            "checkpoints will be lost if the session stops!",
            flush=True,
        )

    if run_local is not None:
        restored = False
        if hf_runs_repo:
            restored = restore_run_from_hf(run_local, hf_runs_repo, hf_token)
        if not restored and drive_backup is not None and Path(config.output["root"]) != drive_backup:
            restore_run_from_drive(run_local, drive_backup)

    sync_every = max(1, int(args.sync_every))
    drive_counter = [0]
    hf_counter = [0]
    use_drive = drive_backup is not None and run_local is not None and Path(config.output["root"]) != drive_backup

    def _on_epoch_end() -> None:
        if use_drive and drive_backup is not None and run_local is not None:
            sync_run_to_drive(run_local, drive_backup, every_n_epochs=sync_every, epoch_counter=drive_counter)
        if hf_runs_repo and run_local is not None:
            sync_run_to_hf(
                run_local,
                hf_runs_repo,
                hf_token,
                every_n_epochs=sync_every,
                epoch_counter=hf_counter,
                run_as_background=True,
            )

    on_epoch_end = _on_epoch_end if (use_drive or (hf_runs_repo and run_local is not None)) else None

    run: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="tinydet-cfg-") as temporary:
            config_file = Path(temporary) / "experiment.yaml"
            config_file.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
            run = train(
                config_file,
                dataset_config=dataset_yaml,
                wandb_group=args.wandb_group,
                dataset_audit=dataset_audit,
                run_id=effective_run_id,
                on_epoch_end_callback=on_epoch_end,
            )
    finally:
        flush_dir = run or run_local
        if flush_dir is not None and flush_dir.is_dir():
            if drive_backup is not None and Path(config.output["root"]) != drive_backup:
                sync_run_to_drive(flush_dir, drive_backup, force=True)
            if hf_runs_repo:
                sync_run_to_hf(
                    flush_dir,
                    hf_runs_repo,
                    hf_token,
                    force=True,
                    run_as_background=False,
                )

    if not args.no_finalize:
        try:
            finalize(run)
            print(f"[finalize] Published checkpoints & curves in: {run}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep trained artifacts even if plot fails
            print(f"[finalize] Warning: {exc}", flush=True)

    print(f"Run directory: {run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
