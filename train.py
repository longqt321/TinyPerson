"""Environment-agnostic training & auto-evaluation CLI for TinyDet experiments."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    infer_dataset_slug,
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
        description="Train and evaluate a TinyDet experiment on any CUDA GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/experiment/paper/tp_yolo26n_lgt.yaml",
        help="Experiment YAML.",
    )
    parser.add_argument("--data", type=Path, help="Override data.config (file or directory).")
    parser.add_argument(
        "--hf-dataset",
        default=os.environ.get("TINYDET_HF_DATASET", DEFAULT_HF_DATASET_REPO),
        help="Hugging Face dataset repo to auto-download if dataset is absent locally.",
    )
    parser.add_argument(
        "--hf-runs-repo",
        help="Hugging Face model repo for automatic checkpoint backup & resume.",
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
    parser.add_argument("--data-root", type=Path, help="Override 'path' inside dataset YAML.")
    parser.add_argument("--output-root", type=Path, help="Override output.root (run directory).")
    parser.add_argument("--name", help="Override the experiment name (run folder).")
    parser.add_argument("--run-id", help="Run folder name under <output-root>/<name>/.")
    parser.add_argument("--device", help="CUDA device list, e.g. '0'.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs.")
    parser.add_argument("--batch", type=int, help="Override train.batch.")
    parser.add_argument("--imgsz", type=int, help="Override train.imgsz.")
    parser.add_argument("--workers", type=int, help="Override train.workers.")
    parser.add_argument("--seed", type=int, help="Override seed.")
    parser.add_argument("--wandb-project", help="Override W&B project name.")
    parser.add_argument("--wandb-group", help="Optional W&B group name.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--no-stage", action="store_true", help="Skip staging Drive datasets to SSD.")
    parser.add_argument("--skip-audit", action="store_true", help="Skip dataset preflight audit.")
    parser.add_argument("--no-finalize", action="store_true", help="Skip post-training finalize().")
    parser.add_argument(
        "--no-auto-eval",
        action="store_true",
        help="Skip post-training GPU benchmark and val/test evaluation suite.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training fit and run finalize + full evaluation suite on an existing run.",
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
        raw_pre = str(model["pretrained"]).strip()
        cand_pre = Path(raw_pre).expanduser()
        model["pretrained"] = str(cand_pre.resolve()) if cand_pre.is_file() else raw_pre

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
            train["workers"] = min(int(train.get("workers", 8)), os.cpu_count() or 2)
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
    if config.model.get("pretrained"):
        print(f"Pretrained : {config.model['pretrained']}")
    print(f"Dataset    : {dataset_yaml}")
    print(f"Run root   : {config.output['root']}")
    print(f"W&B        : {config.output.get('wandb_project') or 'disabled'}"
          f"{f' (group={wandb_group})' if wandb_group else ''}")
    for key in ("epochs", "imgsz", "batch", "workers", "optimizer", "lr0", "device"):
        if key in config.train:
            print(f"{key:<11}: {config.train[key]}")
    print("=" * 70, flush=True)


def _run_model_preflight(config: ExperimentConfig, dataset_yaml: Path) -> None:
    from ultralytics.utils.torch_utils import get_flops, get_num_gradients, get_num_params

    from tinydet.models.registry import build_model
    from tinydet.utils.dataset import dataset_images

    # Override nc in model preflight from dataset_yaml so multi-class models report exact params
    ds_cfg = yaml.safe_load(dataset_yaml.read_text()) or {}
    names = ds_cfg.get("names", {})
    nc = len(names) if isinstance(names, (dict, list)) and names else 1

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
        f"[preflight] model (yaml_nc={getattr(inner, 'nc', nc)}, data_nc={nc}): "
        f"layers={layers}, params={params:,}, grads={gradients:,}, "
        f"GFLOPs={gflops:.3f}@{imgsz}, strides={strides}",
        flush=True,
    )
    print(f"[preflight] dataset splits: train={train_images}, val={val_images}", flush=True)
    if train_images == 0:
        raise ValueError(f"Training split has 0 images in {dataset_yaml}")


def _run_dataset_preflight(config: ExperimentConfig, dataset_yaml: Path) -> dict:
    from tinydet.data.image_audit import audit_dataset

    imgsz = int(config.train["imgsz"])
    print(f"[preflight] auditing dataset images/labels (imgsz={imgsz})...", flush=True)
    report = audit_dataset(dataset_yaml, imgsz, workers=1)
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
        raise SystemExit(f"Output root is not writable: {root} ({error}).") from error


def _predict_split_boxes(
    checkpoint: Path,
    image_paths: list[Path],
    output_json: Path,
    *,
    imgsz: int,
    batch_size: int,
    max_det: int,
    conf: float = 0.001,
) -> Path:
    import torch
    from ultralytics import YOLO

    from tinydet.models.ultralytics import register_ultralytics_modules
    from tinydet.utils.experiment import write_json

    register_ultralytics_modules()
    use_half = torch.cuda.is_available()
    model = YOLO(str(checkpoint))
    predictions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tinydet-pred-") as tmp:
        list_file = Path(tmp) / "sources.txt"
        list_file.write_text("\n".join(str(p.resolve()) for p in image_paths) + "\n")
        results = model.predict(
            source=str(list_file),
            stream=True,
            imgsz=imgsz,
            batch=max(1, min(batch_size, 16)),
            conf=conf,
            max_det=max_det,
            half=use_half,
            verbose=False,
        )
        for res in results:
            boxes = res.boxes
            predictions.append({
                "path": str(Path(res.path).resolve()),
                "boxes": boxes.xyxy.cpu().tolist() if boxes is not None else [],
                "scores": boxes.conf.cpu().tolist() if boxes is not None else [],
                "classes": [int(v) for v in boxes.cls.cpu().tolist()] if boxes is not None else [],
            })
    write_json(output_json, {"images": predictions})
    return output_json


def run_post_training_evaluation(run: Path, dataset_yaml: Path, config: ExperimentConfig) -> None:
    """Execute hardware profiler + full benchmark/diagnostic suite so every run is paper-ready."""
    from tinydet.evaluation.benchmark import benchmark as run_hw_benchmark
    from tinydet.evaluation.coco_benchmark import evaluate_coco_suite
    from tinydet.evaluation.tinyperson_benchmark import (
        evaluate_tinyperson,
        subset_annotations_to_split,
    )
    from tinydet.utils.dataset import dataset_images
    from tinydet.utils.experiment import write_json

    metrics_dir = run / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        checkpoint = run / "training" / "weights" / "best.pt"
    if not checkpoint.is_file():
        print(f"[auto-eval] No best.pt found in {run}; skipping evaluation.", flush=True)
        return

    print("[auto-eval] 1/3 Measuring Pre-Fuse vs Post-Fuse hardware complexity & latency...", flush=True)
    try:
        hw = run_hw_benchmark(run)
        print(
            f"[auto-eval] Hardware: fused_params={hw['parameters_fused']:,}, "
            f"GFLOPs={hw['gflops']}@{hw['imgsz']}, latency={hw['latency_ms']}ms ({hw['fps']} FPS)",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[auto-eval] Warning: hardware benchmark failed: {exc}", flush=True)

    ds_cfg = yaml.safe_load(dataset_yaml.read_text()) or {}
    ds_root = Path(ds_cfg.get("path", dataset_yaml.parent))
    if not ds_root.is_absolute():
        ds_root = (dataset_yaml.parent / ds_root).resolve()
    ann_dir = ds_root / "annotations"
    slug = infer_dataset_slug(str(ds_root) + " " + str(config.data.get("config", "")))
    imgsz = int(config.train["imgsz"])
    batch_sz = int(config.train.get("batch", 16))

    if slug == "tinyperson-yolo-v2":
        print("[auto-eval] 2/3 Scoring TinyPerson val & test splits (WACV ignore-aware + TIDE/Crowded)...", flush=True)
        train_ann = ann_dir / "tiny_set_train.json"
        val_ann = ann_dir / "tiny_set_val.json"
        if not val_ann.is_file() and train_ann.is_file():
            val_ann = subset_annotations_to_split(
                train_ann, dataset_yaml, "val", metrics_dir / "annotations_val.json"
            )
        if val_ann.is_file():
            val_imgs = dataset_images(dataset_yaml, "val")
            val_pred = _predict_split_boxes(
                checkpoint, val_imgs, metrics_dir / "val_predictions.json",
                imgsz=imgsz, batch_size=batch_sz, max_det=200,
            )
            tp_val = evaluate_tinyperson(val_ann, val_pred, class_mode="merge")
            write_json(metrics_dir / "tinyperson_val.json", tp_val)

        test_ann = ann_dir / "tiny_set_test.json"
        test_imgs = dataset_images(dataset_yaml, "test")
        if test_ann.is_file() and test_imgs:
            test_pred = _predict_split_boxes(
                checkpoint, test_imgs, metrics_dir / "test_predictions.json",
                imgsz=imgsz, batch_size=batch_sz, max_det=200,
            )
            tp_test = evaluate_tinyperson(test_ann, test_pred, class_mode="merge")
            write_json(metrics_dir / "tinyperson_test.json", tp_test)
            coco_test = evaluate_coco_suite(test_ann, test_pred, collapse_single_class=True)
            write_json(metrics_dir / "coco_test_suite.json", coco_test)
            print(
                f"[auto-eval] TinyPerson TEST (Full-1280): all AP@0.50={tp_test['all']['AP@0.50']:.4f} | "
                f"tiny AP@0.50={tp_test['tiny']['AP@0.50']:.4f} | "
                f"tiny1 AP@0.50={tp_test['tiny1']['AP@0.50']:.4f} | "
                f"AP@0.75={tp_test['all']['AP@0.75']:.4f} | MR@0.50={tp_test['all']['MR@0.50']:.4f}",
                flush=True,
            )

            # Also score official TOV native-resolution Tiled Slicing (640x512, overlap=100)
            from PIL import Image
            from ultralytics import YOLO

            from tinydet.evaluation.tiled_inference import (
                TILE_H,
                TILE_OVERLAP,
                TILE_W,
                predict_tiled_image,
            )
            from tinydet.models.ultralytics import register_ultralytics_modules

            register_ultralytics_modules()
            tiled_model = YOLO(str(checkpoint))
            tiled_dev = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
            tiled_rows: list[dict[str, Any]] = []
            for img_p in test_imgs:
                with Image.open(img_p) as pil_im:
                    t_boxes, t_scores, t_clses = predict_tiled_image(
                        tiled_model,
                        pil_im,
                        tile_w=TILE_W,
                        tile_h=TILE_H,
                        overlap=TILE_OVERLAP,
                        conf=0.001,
                        device=tiled_dev,
                    )
                tiled_rows.append({
                    "path": str(img_p.resolve()),
                    "boxes": t_boxes,
                    "scores": t_scores,
                    "classes": t_clses,
                })
            tiled_pred_path = metrics_dir / "test_predictions_tiled.json"
            write_json(tiled_pred_path, {"images": tiled_rows})
            tp_tiled = evaluate_tinyperson(test_ann, tiled_pred_path, class_mode="merge")
            write_json(metrics_dir / "tinyperson_test_tiled.json", tp_tiled)
            coco_tiled = evaluate_coco_suite(test_ann, tiled_pred_path, collapse_single_class=True)
            write_json(metrics_dir / "coco_test_tiled_suite.json", coco_tiled)
            print(
                f"[auto-eval] TinyPerson TEST (Tiled-640x512): all AP@0.50={tp_tiled['all']['AP@0.50']:.4f} | "
                f"tiny AP@0.50={tp_tiled['tiny']['AP@0.50']:.4f} | "
                f"tiny1 AP@0.50={tp_tiled['tiny1']['AP@0.50']:.4f} | "
                f"AP@0.75={tp_tiled['all']['AP@0.75']:.4f} | MR@0.50={tp_tiled['all']['MR@0.50']:.4f}",
                flush=True,
            )
    else:
        print("[auto-eval] 2/3 Scoring VisDrone val & test-dev splits (COCO-100 + VisDrone-500 + TIDE)...", flush=True)
        for split_key, ann_name, out_name in (
            ("val", "visdrone_val_coco.json", "coco_val_suite.json"),
            ("test", "visdrone_test_dev_coco.json", "coco_test_suite.json"),
        ):
            ann_file = ann_dir / ann_name
            split_imgs = dataset_images(dataset_yaml, split_key)
            if ann_file.is_file() and split_imgs:
                pred_file = _predict_split_boxes(
                    checkpoint, split_imgs, metrics_dir / f"{split_key}_predictions.json",
                    imgsz=imgsz, batch_size=batch_sz, max_det=500,
                )
                suite = evaluate_coco_suite(ann_file, pred_file, collapse_single_class=False)
                write_json(metrics_dir / out_name, suite)
                vd500 = suite["visdrone_maxdet500"]
                print(
                    f"[auto-eval] VisDrone {split_key.upper()} (maxDet=500): "
                    f"AP50={vd500['AP50']:.4f} | AP={vd500['AP']:.4f} | "
                    f"AP75={vd500['AP75']:.4f} | AP_small={vd500['AP_small']:.4f} | "
                    f"AP_very_tiny={vd500['AP_very_tiny']:.4f}",
                    flush=True,
                )

    (metrics_dir / "eval_complete.flag").write_text("completed\n")
    print(f"[auto-eval] 3/3 Completed all paper evaluations in {metrics_dir}", flush=True)


def _stream_logs_unbuffered() -> None:
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
    if not args.skip_audit and not args.eval_only:
        dataset_audit = _run_dataset_preflight(config, dataset_yaml)

    if args.dry_run:
        print("[DRY RUN] config, dataset audit, and model all passed. No training performed.", flush=True)
        return 0

    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available. Training/benchmark requires a CUDA GPU.", file=sys.stderr)
        return 1

    print(
        f"[preflight] CUDA GPU: {torch.cuda.get_device_name(0)} "
        f"(VRAM {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB, "
        f"torch {torch.__version__}, CUDA {torch.version.cuda})",
        flush=True,
    )

    if config.output.get("wandb_project") and not args.eval_only:
        key = ensure_wandb_auth()
        status = "authenticated" if key else "no WANDB_API_KEY found (W&B disabled)"
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

    if run_local is not None:
        restored = False
        if hf_runs_repo:
            restored = restore_run_from_hf(run_local, hf_runs_repo, hf_token)
        if not restored and drive_backup is not None and Path(config.output["root"]) != drive_backup:
            restore_run_from_drive(run_local, drive_backup)

    already_trained = bool(
        run_local is not None and (run_local / "training" / "completed.flag").is_file()
    )

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

    run: Path | None = run_local if (already_trained or args.eval_only) else None
    try:
        if already_trained or args.eval_only:
            if run is None or not run.is_dir():
                raise FileNotFoundError(f"Cannot run --eval-only on missing run directory: {run}")
            print(f"[resume] Training already completed in {run}; skipping fit.", flush=True)
        else:
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

        if run is not None and not args.no_finalize:
            try:
                from tinydet.utils.experiment import update_status

                if (run / "metadata.json").is_file():
                    meta = yaml.safe_load((run / "metadata.json").read_text()) or {}
                    if meta.get("status") not in {"TRAINED", "COMPLETED"} and (
                        run / "training" / "completed.flag"
                    ).is_file():
                        update_status(run, "TRAINED")
                finalize(run)
                print(f"[finalize] Published checkpoints & curves in: {run}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[finalize] Warning: {exc}", flush=True)

        if run is not None and not args.no_auto_eval:
            if not (run / "metrics" / "eval_complete.flag").is_file() or args.eval_only:
                run_post_training_evaluation(run, dataset_yaml, config)
            else:
                print(f"[auto-eval] Metrics already complete in {run / 'metrics'}; skipping.", flush=True)
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

    print(f"Run directory: {run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
