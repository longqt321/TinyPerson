import hashlib
import json
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import yaml
from image import image
from paths import DATA_ROOT, MODEL_ROOT, RUN_ROOT

import modal
from tinydet.training.trainer import train as train_model

app = modal.App("tiny-person-research")
data = modal.Volume.from_name("tiny-person-data", create_if_missing=True)
models = modal.Volume.from_name("tiny-person-models", create_if_missing=True)
runs = modal.Volume.from_name("tiny-person-runs", create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])
mounts = {str(DATA_ROOT): data, str(MODEL_ROOT): models, str(RUN_ROOT): runs}
# GPU containers (training + evaluation): data loading and augmentation are
# CPU/RAM heavy, so keep a generous request with a capped ceiling.
CPU_LIMIT = (4.0, 8.0)
MEMORY_LIMIT_MB = (12288, 24576)
# CPU-only orchestration/plumbing containers (finalize, preflight, W&B tagging,
# report assembly, the campaign orchestrator). They only copy files and await
# handles, so 1 CPU / 1 GB is enough; the ceiling absorbs a transient spike.
LIGHT_CPU = (1.0, 2.0)
LIGHT_MEMORY_MB = (1024, 2048)
# Dataset audit decodes every image with 24 worker threads, so it needs real
# CPU but far less RAM than the training containers.
AUDIT_CPU = (4.0, 8.0)
AUDIT_MEMORY_MB = (8192, 16384)
# CPU-side post-processing that loads a predictions artifact and computes AP
# (finalize_predictions). Single-threaded numpy, but the artifact can hold
# every val/test prediction, so keep a little headroom.
POSTPROCESS_CPU = (2.0, 4.0)
POSTPROCESS_MEMORY_MB = (4096, 8192)


def _dataset_yaml_path(path: str | Path) -> Path:
    p = Path(os.path.normpath(path))
    if p.suffix not in (".yaml", ".yml"):
        for name in ("data.yaml", "dataset.yaml"):
            candidate = p / name
            if candidate.is_file():
                p = candidate
                break
        else:
            p = p / "data.yaml"
    dataset_yaml = p
    try:
        dataset_yaml.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise ValueError(f"Dataset YAML must be inside the Modal data Volume: {dataset_yaml}") from exc
    if not dataset_yaml.is_absolute() or dataset_yaml.name != "data.yaml":
        raise ValueError(f"Expected an absolute path to data.yaml inside {DATA_ROOT}: {dataset_yaml}")
    return dataset_yaml


def copytree_concurrent(src: Path, dst: Path, max_workers: int = 24) -> None:
    pending = []

    def submit_copy(source: str, destination: str) -> str:
        pending.append(pool.submit(shutil.copy2, source, destination))
        return destination

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        shutil.copytree(
            src,
            dst,
            copy_function=submit_copy,
            dirs_exist_ok=True,
        )
        for future in pending:
            future.result()



@app.function(image=image, cpu=AUDIT_CPU, memory=AUDIT_MEMORY_MB, max_containers=1,
              volumes=mounts, timeout=1800)
def _audit_original_dataset(dataset_yaml: str, imgsz: int, audit_id: str):
    """Audit stored image pixels and paired labels before any YOLO preprocessing."""
    from tinydet.data.image_audit import audit_dataset
    from tinydet.utils.dataset import normalize_dataset_yaml

    data.reload()
    runs.reload()
    dataset_yaml = _dataset_yaml_path(dataset_yaml)
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {dataset_yaml}")

    # 1. Tái sử dụng kết quả audit nếu dataset và imgsz chưa từng thay đổi
    cache_key_data = f"{dataset_yaml}\0{imgsz}\0".encode() + dataset_yaml.read_bytes()
    cache_key = hashlib.sha256(cache_key_data).hexdigest()[:16]
    cache_file = RUN_ROOT / "preflight" / f"cache-{cache_key}.json"
    if cache_file.is_file():
        try:
            cached_result = json.loads(cache_file.read_text())
            print(f"Tái sử dụng kết quả audit đã có: {cache_file.name} (0s, không copy lặp lại)", flush=True)
            return cached_result
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            print(f"Cache preflight không hợp lệ ({exc}), tiến hành audit mới...", flush=True)

    # 2. Nếu chưa có cache, stream decode trực tiếp đa luồng từ Modal Volume (không copy thừa ra /tmp)
    report_path = RUN_ROOT / "preflight" / f"dataset-audit-{audit_id}.json"
    normalized_yaml = normalize_dataset_yaml(dataset_yaml)
    print(f"Bắt đầu audit trực tiếp đa luồng từ {dataset_yaml}...", flush=True)
    report = audit_dataset(normalized_yaml, imgsz, report_path, workers=24)

    result = {
        "passed": report["passed"],
        "data_yaml": str(dataset_yaml),
        "imgsz": report["imgsz"],
        "measurement_basis": report["measurement_basis"],
        "dataset_signature_metadata_sha256": report["dataset_signature_metadata_sha256"],
        "images_total": report["images_total"],
        "decoded_images_total": report["decoded_images_total"],
        "corrupt_images_total": report["corrupt_images_total"],
        "label_files_found_total": report["label_files_found_total"],
        "annotation_rows_total": report["annotation_rows_total"],
        "parsed_objects_total": report["parsed_objects_total"],
        "splits": report["splits"],
        "coco_dimension_checks": report["coco_dimension_checks"],
        "coco_dimension_status": report["coco_dimension_status"],
        "severe_resolution_loss": report["severe_resolution_loss"],
        "warnings": report["warnings"],
        "errors": report["errors"],
        "report_path": str(report_path),
    }

    # Lưu vào cache để mọi lần chạy sau không bao giờ phải audit hay copy lại
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(result, indent=2))
    runs.commit()
    return result


@app.function(image=image, gpu="A10", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=4, volumes=mounts, timeout=21600)
def _train(
    config: str,
    commit: str,
    dirty: bool,
    config_text: str | None = None,
    wandb_group: str | None = None,
    dataset_audit: dict | None = None,
    run_id: str | None = None,
    ablation: str = "",
):
    os.chdir("/root")
    # Must be set before the model is built: GauSS reads the flags in __init__.
    # Passed as an argument (not inherited) because .remote() runs in a new container.
    os.environ["TINYDET_ABLATION"] = ablation
    runs.reload()
    raw_config = yaml.safe_load(config_text) if config_text else yaml.safe_load(Path(config).read_text())
    volume_yaml = _dataset_yaml_path(raw_config["data"]["config"])
    volume_dataset = volume_yaml.parent
    if not volume_yaml.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {volume_yaml}")

    local_cache_root = Path("/tmp/tinydet-datasets")
    local_dataset = local_cache_root / volume_dataset.name
    local_yaml = local_dataset / "data.yaml"

    if not local_yaml.is_file():
        print(f"Copying dataset from {volume_dataset} to local container disk {local_dataset}", flush=True)
        local_dataset.mkdir(parents=True, exist_ok=True)
        copytree_concurrent(volume_dataset, local_dataset)
        dataset_config = yaml.safe_load(local_yaml.read_text())
        dataset_config["path"] = str(local_dataset.resolve())
        local_yaml.write_text(yaml.safe_dump(dataset_config, sort_keys=False))
        print("Dataset copy complete; starting training", flush=True)
    else:
        print(f"Reusing cached local dataset at {local_dataset} (0s copy overhead)", flush=True)

    def on_epoch_commit():
        runs.commit()

    if config_text is not None:
        with tempfile.TemporaryDirectory(prefix="tinydet-cfg-", dir="/tmp") as temporary:
            config_file = Path(temporary) / "experiment.yaml"
            config_file.write_text(config_text)
            run = train_model(
                str(config_file),
                (commit, dirty),
                local_yaml,
                wandb_group,
                dataset_audit,
                run_id=run_id,
                on_epoch_end_callback=on_epoch_commit,
            )
    else:
        run = train_model(
            config,
            (commit, dirty),
            local_yaml,
            wandb_group,
            dataset_audit,
            run_id=run_id,
            on_epoch_end_callback=on_epoch_commit,
        )
    runs.commit()

    # Record the ablation flags actually used, so a finished run can be audited
    # after the fact instead of relying on the launch command.
    if ablation:
        from tinydet.utils.experiment import metadata, write_json

        details = metadata(Path(run))
        details["ablation"] = ablation
        details["ablation_flags"] = sorted(
            part.strip() for part in ablation.split(",") if part.strip()
        )
        write_json(Path(run) / "metadata.json", details)
        runs.commit()
    return str(run)


@app.function(image=image, cpu=LIGHT_CPU, memory=LIGHT_MEMORY_MB, max_containers=4, scaledown_window=2, volumes={str(RUN_ROOT): runs}, timeout=1800)
def _finalize(run: str):
    from tinydet.training.finalize import finalize
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    finalize(resolve_run(RUN_ROOT, run))
    runs.commit()
    from tinydet.utils.experiment import metadata

    return metadata(resolve_run(RUN_ROOT, run)).get("wandb_project")


def _finalize_pipeline(run: str):
    project = _finalize.remote(run)
    if project:
        _organize_wandb.remote(project)


def _collect(handle_map: dict, label: str) -> dict:
    """Drain a {key: handle} map, isolating failures per entry.

    Returns {key: result-or-None}. A failed entry is reported and skipped;
    it never aborts the remaining entries.
    """
    results = {}
    for key, handle in handle_map.items():
        try:
            results[key] = handle.get()
            print(f"[OK] {label} finished for {key}", flush=True)
        except Exception as exc:  # noqa: BLE001 - isolate per-run failures
            results[key] = None
            print(f"[FAILED] {label} for {key}: {exc}", flush=True)
    return results


@app.function(
    image=image,
    cpu=LIGHT_CPU,
    memory=LIGHT_MEMORY_MB,
    max_containers=1,
    volumes=mounts,
    timeout=43200,
    secrets=[wandb_secret],
)
def _orchestrate_campaign(configs: list[str], commit: str, dirty: bool):
    """Detached orchestrator: the full train -> finalize -> eval chain runs
    inside Modal, so a dropped local connection cannot kill the campaign.

    Each phase isolates failures per run (see `_collect`); the summary is
    persisted to the runs Volume at `orchestration/summary-<ts>.json` and also
    returned to the caller.
    """
    from dataclasses import asdict

    from tinydet.utils.config import load_config

    print("=" * 70, flush=True)
    print("ORCHESTRATOR: PARALLEL TRAINING + EVALUATION (NVIDIA A10)", flush=True)
    print(f"Configs: {configs}", flush=True)
    print("=" * 70, flush=True)

    # 1. Preflight audit (shared gate: both configs use the same dataset).
    #    A bad dataset invalidates both runs, so `passed=False` aborts here;
    #    transient container errors are retried once instead of killing the chain.
    exp0 = load_config(configs[0])
    audit = None
    for attempt in (1, 2):
        try:
            audit = _audit_original_dataset.remote(
                str(exp0.data["config"]), int(exp0.train["imgsz"]), uuid.uuid4().hex[:12]
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry transient audit failure
            print(f"[WARN] Preflight audit attempt {attempt} failed: {exc}", flush=True)
    if audit is None:
        raise RuntimeError("Preflight audit could not run (transient failure after retry).")
    if not audit["passed"]:
        raise ValueError("Dataset audit failed!")

    # 2. One independent container per run. A spawn failure is recorded and
    #    must not prevent the other run from being spawned.
    train_handles = {}
    failed_spawn: dict[str, str] = {}
    for cfg_path in configs:
        exp = load_config(cfg_path)
        exp.data["preflight_signature"] = audit["dataset_signature_metadata_sha256"]
        config_text = yaml.safe_dump(asdict(exp), sort_keys=False)
        gpu = exp.hardware.get("gpu", "A10")
        options = {"gpu": gpu}
        if exp.output.get("wandb_project"):
            options["secrets"] = [wandb_secret]
        print(f"--> Spawning {exp.train['epochs']}e training for: {cfg_path}...", flush=True)
        try:
            handle = _train.with_options(**options).spawn(
                cfg_path, commit, dirty, config_text=config_text, dataset_audit=audit
            )
            train_handles[cfg_path] = handle
        except Exception as exc:  # noqa: BLE001 - isolate per-run failures
            failed_spawn[cfg_path] = str(exc)
            print(f"[FAILED] TRAIN spawn for {cfg_path}: {exc}", flush=True)

    train_results = _collect(train_handles, "TRAIN")
    for cfg_path in failed_spawn:
        train_results[cfg_path] = None

    # 3. Finalize each successful run independently.
    run_paths = {}
    failed_finalize = []
    failed_evals = []
    for cfg_path, run_full_path in train_results.items():
        if run_full_path is None:
            continue
        try:
            rel_run = str(Path(run_full_path).relative_to(RUN_ROOT))
            _finalize_pipeline(rel_run)
            run_paths[cfg_path] = rel_run
            print(f"[OK] FINALIZE finished for {rel_run}", flush=True)
        except Exception as exc:  # noqa: BLE001 - isolate per-run failures
            failed_finalize.append(Path(cfg_path).stem)
            print(f"[FAILED] FINALIZE for {run_full_path}: {exc}", flush=True)

    # 4. Tiled evaluation (640x512) per finalized run.
    tiled_handles = {}
    for rel_run in run_paths.values():
        run_name = rel_run.split("/")[0]
        name = f"{run_name}_tiled"
        ckpt = f"{rel_run}/training/weights/best.pt"
        print(f"--> Spawning tiled evaluation for {name} ({ckpt})...", flush=True)
        try:
            tiled_handles[name] = _run_tiled_checkpoint_evaluation.spawn(
                relative_checkpoint=ckpt, name=name
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-run failures
            failed_evals.append(name)
            print(f"[FAILED] TILED EVAL spawn for {name}: {exc}", flush=True)
    tiled_results = _collect(tiled_handles, "TILED EVAL")

    # 5. Full-frame test evaluation (1280px) per finalized run.
    full_handles = {}
    for rel_run in run_paths.values():
        run_name = rel_run.split("/")[0]
        name = f"{run_name}_full"
        ckpt = f"{rel_run}/training/weights/best.pt"
        print(f"--> Spawning full-frame evaluation for {name} ({ckpt})...", flush=True)
        try:
            full_handles[name] = _run_checkpoint_evaluation.spawn(
                relative_checkpoint=ckpt, name=name, split="test"
            )
        except Exception as exc:  # noqa: BLE001 - isolate per-run failures
            failed_evals.append(name)
            print(f"[FAILED] FULL EVAL spawn for {name}: {exc}", flush=True)
    full_results = _collect(full_handles, "FULL EVAL")

    # 6. Persist the summary on the Volume so it survives local disconnects.
    succeeded = [Path(c).stem for c, r in train_results.items() if r is not None]
    failed = [Path(c).stem for c, r in train_results.items() if r is None]
    summary = {
        "configs": configs,
        "commit": commit,
        "dirty": dirty,
        "trained_ok": succeeded,
        "train_failed": failed,
        "finalize_failed": failed_finalize,
        "eval_failed": failed_evals,
        "tiled_eval": {k: (v or {}).get("summary") for k, v in tiled_results.items() if v},
        "full_eval": {k: (v or {}).get("summary") for k, v in full_results.items() if v},
        "finished_at": datetime.now(UTC).isoformat(),
    }
    summary_path = RUN_ROOT / "orchestration" / f"summary-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    runs.commit()
    print("=" * 70, flush=True)
    print(f"SUMMARY saved to volume: {summary_path}", flush=True)
    print(f"  Trained OK : {', '.join(succeeded) if succeeded else '(none)'}", flush=True)
    print(f"  Failed     : {', '.join(failed) if failed else '(none)'}", flush=True)
    if failed_finalize:
        print(f"  Finalize failed: {', '.join(failed_finalize)}", flush=True)
    if failed_evals:
        print(f"  Eval failed    : {', '.join(failed_evals)}", flush=True)
    print("=" * 70, flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} training run(s) failed: {', '.join(failed)}")
    return summary


@app.function(image=image, gpu="A10", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=4, volumes=mounts, timeout=3600)
def _evaluate(run: str, split: str):
    from tinydet.evaluation.evaluator import collect_predictions
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = collect_predictions(resolve_run(RUN_ROOT, run), split)
    runs.commit()
    return str(result)


@app.function(image=image, gpu="A10", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=4, volumes=mounts, timeout=3600)
def _benchmark(run: str):
    from tinydet.evaluation.benchmark import benchmark
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = benchmark(resolve_run(RUN_ROOT, run))
    runs.commit()
    return result


@app.function(image=image, gpu="A10", min_containers=0, buffer_containers=0,
              scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB,
              max_containers=4, volumes=mounts, timeout=3600)
def _analyze_geometry(run: str, split: str = "val", iou: float = 0.5):
    """Measure per-GT center and size errors without using local RAM."""
    import subprocess
    import sys

    from tinydet.utils.dataset import normalize_dataset_yaml
    from tinydet.utils.experiment import resolve_run

    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if not 0.0 < iou <= 1.0:
        raise ValueError("iou must be in (0, 1]")
    runs.reload()
    data.reload()
    run_dir = resolve_run(RUN_ROOT, run)
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    dataset_yaml = normalize_dataset_yaml(config["data"]["config"])
    dataset = yaml.safe_load(dataset_yaml.read_text())
    root = Path(dataset["path"])
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "training" / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No best.pt under {run_dir}")
    output = RUN_ROOT / "error_analysis" / run.replace("/", "__") / f"geometry-{split}"
    if not (output / "geometry_summary.csv").is_file():
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        subprocess.run(
            [
                sys.executable, "/root/scripts/analyze_msr_validation.py",
                "--images", str(root / f"images/{split}"),
                "--labels", str(root / f"labels/{split}"),
                "--checkpoint", f"A2={checkpoint}", "--output", str(output),
                "--imgsz", str(config["train"]["imgsz"]), "--device", "cuda:0",
                "--cpu-threads", "4", "--iou", str(iou),
            ],
            env=env,
            check=True,
        )
        runs.commit()

    return {
        "output": str(output),
        "geometry_summary_csv": (output / "geometry_summary.csv").read_text() if (output / "geometry_summary.csv").is_file() else "",
        "size_summary_csv": (output / "size_summary.csv").read_text() if (output / "size_summary.csv").is_file() else "",
        "error_summary_csv": (output / "error_summary.csv").read_text() if (output / "error_summary.csv").is_file() else "",
        "report_json": (output / "report.json").read_text() if (output / "report.json").is_file() else "{}",
    }


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1, volumes={str(RUN_ROOT): runs}, timeout=300)
def _find_reusable(expected: dict, model_sha256: str):
    from tinydet.utils.reuse import find_reusable_run

    runs.reload()
    return find_reusable_run(RUN_ROOT, expected, model_sha256)


@app.local_entrypoint()
def train(config: str, preflight_only: bool = False, strict_resolution: bool = False, require_source_metadata: bool = False):
    from dataclasses import asdict

    from tinydet.utils.config import load_config
    from tinydet.utils.experiment import git_info

    experiment = load_config(config)
    gpu = experiment.hardware["gpu"]
    if gpu != "A10":
        raise ValueError("Only A10 GPU is allowed by the resource cap")
    audit = _audit_original_dataset.remote(
        str(experiment.data["config"]), int(experiment.train["imgsz"]), uuid.uuid4().hex[:12]
    )
    print("Original-image preflight (before YOLO resizing):", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    if not audit["passed"]:
        raise ValueError("Dataset preflight failed; fix the errors before training.")
    if strict_resolution and audit["severe_resolution_loss"]:
        raise ValueError(
            "Strict resolution preflight failed: at least half of parsed objects are "
            "predicted to have sqrt-area below 4 px. Review the report and choose imgsz."
        )
    if require_source_metadata and audit["coco_dimension_status"] != "verified":
        raise ValueError(
            "Source COCO dimensions are not fully verified against stored rasters "
            f"(status={audit['coco_dimension_status']!r}). Add the original COCO metadata "
            "or resolve image-name/dimension mismatches before training."
        )
    if require_source_metadata and audit["coco_dimension_status"] != "verified":
        raise ValueError(
            "Source COCO dimensions are not fully verified against stored rasters "
            f"(status={audit['coco_dimension_status']!r}). Add the original COCO metadata "
            "or resolve image-name/dimension mismatches before training."
        )
    if preflight_only:
        print("Preflight complete; no training GPU was launched.", flush=True)
        return

    experiment.data["preflight_signature"] = audit["dataset_signature_metadata_sha256"]
    config_text = yaml.safe_dump(asdict(experiment), sort_keys=False)
    commit, dirty = git_info()
    options = {"gpu": gpu}
    if experiment.output.get("wandb_project"):
        options["secrets"] = [wandb_secret]
    run = _train.with_options(**options).remote(
        config, commit, dirty, config_text=config_text, dataset_audit=audit
    )
    _finalize_pipeline(str(Path(run).relative_to(RUN_ROOT)))
    print(run)


def _run_comparison_workflow(
    config: str = "configs/experiment/comparison.yaml",
    start_at: str = "",
    parallel: bool = False,
    reuse: bool = False,
    smoke: bool = False,
    two_stage: bool = False,
    preflight_only: bool = False,
    strict_resolution: bool = False,
    require_source_metadata: bool = False,
    config_text: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
):
    from tinydet.utils.comparison import load_comparison
    from tinydet.utils.experiment import git_info

    experiments = load_comparison(path=config, raw_text=config_text)
    training = [experiment for experiment in experiments if "run" not in experiment]
    existing = [experiment for experiment in experiments if "run" in experiment]
    if not training:
        raise ValueError("At least one model must be trained")

    if two_stage:
        for experiment in training:
            if "stage2" not in experiment and "stage2" not in experiment.get("train", {}):
                experiment["stage2"] = {
                    "enabled": True,
                    "epochs": 50,
                    "patience": 20,
                    "lr0": 0.0006,
                    "lrf": 0.02,
                    "cos_lr": True,
                    "warmup_epochs": 3.0,
                    "close_mosaic": 10,
                    "mosaic": 0.0,
                }
        print("TWO-STAGE MODE: Training Phase 1 + Phase 2 continuously in the same container.", flush=True)

    if smoke:
        for experiment in training:
            experiment["train"] = dict(experiment["train"])
            experiment["train"]["epochs"] = 5
            experiment["train"]["patience"] = max(
                int(experiment["train"].get("patience", 6)), 6
            )
            stage2_dict = experiment.get("stage2") or experiment.get("train", {}).get("stage2")
            if isinstance(stage2_dict, dict) and stage2_dict.get("enabled"):
                stage2_dict["epochs"] = 3
                stage2_dict["patience"] = max(int(stage2_dict.get("patience", 5)), 5)
        print("SMOKE MODE: every model is capped (Stage 1: 5 epochs, Stage 2: 3 epochs).", flush=True)

    configured_paths = {Path(experiment["data"]["config"]) for experiment in training}
    if len(configured_paths) != 1:
        raise ValueError("All compared models must use the same dataset YAML.")
    configured_data = _dataset_yaml_path(next(iter(configured_paths)))
    imgsz_values = {int(experiment["train"]["imgsz"]) for experiment in training}
    if len(imgsz_values) != 1:
        raise ValueError("All compared models must use the same imgsz for a valid comparison.")
    audit = _audit_original_dataset.remote(
        str(configured_data), next(iter(imgsz_values)), uuid.uuid4().hex[:12]
    )
    print("Original-image preflight (before YOLO resizing):", flush=True)
    print(json.dumps(audit, indent=2), flush=True)
    if not audit["passed"]:
        raise ValueError("Dataset preflight failed; no training was started.")
    if strict_resolution and audit["severe_resolution_loss"]:
        raise ValueError(
            "Strict resolution preflight failed: at least half of parsed objects are "
            "predicted to have sqrt-area below 4 px. Review the report and choose imgsz."
        )
    if preflight_only:
        print("Preflight complete; no training GPU was launched.", flush=True)
        return {"preflight": audit}

    # A stable data signature makes reuse conditional on the same source dataset.
    for experiment in training:
        experiment["data"]["preflight_signature"] = audit[
            "dataset_signature_metadata_sha256"
        ]
    names = [experiment["name"] for experiment in experiments]
    if start_at and start_at not in names:
        raise ValueError(f"Unknown model: {start_at}")
    if start_at:
        experiments = experiments[names.index(start_at):]
    training = [experiment for experiment in experiments if "run" not in experiment]
    existing = [experiment for experiment in experiments if "run" in experiment]
    if any(experiment["hardware"]["gpu"] != "A10" for experiment in training):
        raise ValueError("Only A10 GPU is allowed by the resource cap")
    if commit is None or dirty is None:
        commit, dirty = git_info()
    wandb_group = f"compare-{uuid.uuid4().hex[:8]}"

    def finish(reference: str, val_done: bool = False, benchmark_done: bool = False):
        if val_done and benchmark_done:
            print(f"Reusing completed {reference}", flush=True)
            return
        if not val_done:
            print(f"Evaluating {reference}", flush=True)
            _evaluate_pipeline(reference, "val", reuse_training=True)
        if not benchmark_done:
            print(f"Benchmarking {reference}", flush=True)
            _benchmark.remote(reference)
        print(f"Completed {reference}", flush=True)

    for experiment in existing:
        reference = experiment["run"]
        _preflight.remote(reference)
        print(f"Using completed control {reference}", flush=True)
        finish(reference, val_done=True, benchmark_done=True)

    to_train = []
    for experiment in training:
        match = None
        if reuse:
            model_sha256 = hashlib.sha256(Path(experiment["model"]["config"]).read_bytes()).hexdigest()
            match = _find_reusable.remote(experiment, model_sha256)
        if match:
            if match["legacy"]:
                print(f"Warning: {match['run']} has no recorded model YAML hash; verify dataset and model files are unchanged", flush=True)
            finish(match["run"], match["val"], match["benchmark"])
        else:
            to_train.append(experiment)

    session_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    results = []

    if parallel:
        for offset in range(0, len(to_train), 4):
            pending = []
            for experiment in to_train[offset:offset + 4]:
                print(f"Training {experiment['name']}", flush=True)
                options = {"secrets": [wandb_secret]} if experiment["output"].get("wandb_project") else {}
                stable_run_id = f"{session_ts}-{commit}-seed{experiment.get('seed', 42)}"
                pending.append(_train.with_options(**options).spawn(
                    config, commit, dirty, yaml.safe_dump(experiment), wandb_group, audit, run_id=stable_run_id
                ))
            completed = [call.get() for call in pending]
            for run in completed:
                reference = str(Path(run).relative_to(RUN_ROOT))
                _finalize_pipeline(reference)
                finish(reference)
                results.append(reference)
    else:
        for experiment in to_train:
            print(f"Training {experiment['name']}", flush=True)
            options = {"secrets": [wandb_secret]} if experiment["output"].get("wandb_project") else {}
            stable_run_id = f"{session_ts}-{commit}-seed{experiment.get('seed', 42)}"
            run = _train.with_options(**options).remote(
                config, commit, dirty, yaml.safe_dump(experiment), wandb_group, audit, run_id=stable_run_id
            )
            reference = str(Path(run).relative_to(RUN_ROOT))
            _finalize_pipeline(reference)
            finish(reference)
            results.append(reference)

    return {"status": "SUCCESS", "runs": results, "session_ts": session_ts}


@app.function(
    image=image,
    cpu=LIGHT_CPU,
    memory=LIGHT_MEMORY_MB,
    timeout=86400,
    volumes=mounts,
    secrets=[wandb_secret],
)
def run_comparison(
    config: str = "configs/experiment/comparison.yaml",
    start_at: str = "",
    parallel: bool = False,
    reuse: bool = False,
    smoke: bool = False,
    two_stage: bool = False,
    preflight_only: bool = False,
    strict_resolution: bool = False,
    require_source_metadata: bool = False,
    config_text: str | None = None,
    commit: str | None = None,
    dirty: bool | None = None,
):
    """Cloud-hosted comparison orchestrator. Safe to detach with `modal run --detach`."""
    return _run_comparison_workflow(
        config=config,
        start_at=start_at,
        parallel=parallel,
        reuse=reuse,
        smoke=smoke,
        two_stage=two_stage,
        preflight_only=preflight_only,
        strict_resolution=strict_resolution,
        require_source_metadata=require_source_metadata,
        config_text=config_text,
        commit=commit,
        dirty=dirty,
    )


@app.local_entrypoint()
def compare(
    config: str = "configs/experiment/comparison.yaml",
    start_at: str = "",
    parallel: bool = False,
    reuse: bool = False,
    smoke: bool = False,
    two_stage: bool = False,
    kitty: bool = False,
    kitty_mode: str = "auto",
    preflight_only: bool = False,
    strict_resolution: bool = False,
    require_source_metadata: bool = False,
):
    """Audit original data first, then compare models; supports spawning Kitty terminals with --kitty."""
    if kitty:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from launch_parallel_kitty import launch_kitty_terminals

        launch_kitty_terminals(
            config=config,
            mode=kitty_mode,
            smoke=smoke,
            two_stage=two_stage,
        )
        return

    from tinydet.utils.experiment import git_info

    config_path = Path(config)
    config_text = config_path.read_text() if config_path.is_file() else None
    commit, dirty = git_info()

    res = run_comparison.remote(
        config=config,
        start_at=start_at,
        parallel=parallel,
        reuse=reuse,
        smoke=smoke,
        two_stage=two_stage,
        preflight_only=preflight_only,
        strict_resolution=strict_resolution,
        require_source_metadata=require_source_metadata,
        config_text=config_text,
        commit=commit,
        dirty=dirty,
    )
    print(f"Comparison workflow finished: {res}")


@app.local_entrypoint()
def train_comparison_model(
    config: str = "configs/experiment/comparison.yaml",
    model_name: str = "",
    wandb_group: str = "",
    session_ts: str = "",
    smoke: bool = False,
    two_stage: bool = False,
):
    """Train a single model extracted from a comparison YAML with full streaming logs in this terminal."""
    from tinydet.utils.comparison import load_comparison
    from tinydet.utils.experiment import git_info

    if not model_name:
        raise ValueError("--model-name must be specified")

    experiments = load_comparison(path=config)
    matching = [exp for exp in experiments if exp["name"] == model_name]
    if not matching:
        available = [exp["name"] for exp in experiments]
        raise ValueError(f"Model '{model_name}' not found in {config}. Available models: {available}")

    experiment = matching[0]
    if "run" in experiment:
        raise ValueError(f"Model '{model_name}' is already an existing run reference: {experiment['run']}")

    if (
        (two_stage or experiment.get("stage2", {}).get("enabled"))
        and "stage2" not in experiment
        and "stage2" not in experiment.get("train", {})
    ):
        experiment["stage2"] = {
                "enabled": True,
                "epochs": 50,
                "patience": 20,
                "lr0": 0.0006,
                "lrf": 0.02,
                "cos_lr": True,
                "warmup_epochs": 3.0,
                "close_mosaic": 10,
                "mosaic": 0.0,
            }

    if smoke:
        experiment["train"] = dict(experiment["train"])
        experiment["train"]["epochs"] = 5
        experiment["train"]["patience"] = max(int(experiment["train"].get("patience", 6)), 6)
        stage2_dict = experiment.get("stage2") or experiment.get("train", {}).get("stage2")
        if isinstance(stage2_dict, dict) and stage2_dict.get("enabled"):
            stage2_dict["epochs"] = 3
            stage2_dict["patience"] = max(int(stage2_dict.get("patience", 5)), 5)

    configured_data = _dataset_yaml_path(experiment["data"]["config"])
    imgsz = int(experiment["train"]["imgsz"])
    audit = _audit_original_dataset.remote(str(configured_data), imgsz, uuid.uuid4().hex[:12])
    if not audit["passed"]:
        raise ValueError(f"Dataset preflight failed for {model_name}")

    experiment["data"]["preflight_signature"] = audit["dataset_signature_metadata_sha256"]
    commit, dirty = git_info()
    if not session_ts:
        session_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if not wandb_group:
        wandb_group = f"compare-{uuid.uuid4().hex[:8]}"

    stable_run_id = f"{session_ts}-{commit}-seed{experiment.get('seed', 42)}"
    options = {"gpu": experiment["hardware"]["gpu"]}
    if experiment["output"].get("wandb_project"):
        options["secrets"] = [wandb_secret]

    print("=" * 70, flush=True)
    print(f"Launching training for {model_name} (Run ID: {stable_run_id})", flush=True)
    print(f"W&B Group: {wandb_group}", flush=True)
    print("=" * 70, flush=True)

    run = _train.with_options(**options).remote(
        config,
        commit,
        dirty,
        config_text=yaml.safe_dump(experiment),
        wandb_group=wandb_group,
        dataset_audit=audit,
        run_id=stable_run_id,
    )

    reference = str(Path(run).relative_to(RUN_ROOT))
    _finalize_pipeline(reference)

    print(f"Evaluating {reference}...", flush=True)
    _evaluate_pipeline(reference, "val", reuse_training=True)

    print(f"Benchmarking {reference}...", flush=True)
    _benchmark.remote(reference)

    print("=" * 70, flush=True)
    print(f"SUCCESS: Completed training, evaluation, and benchmark for {model_name}", flush=True)
    print(f"Run reference: {reference}", flush=True)
    print("=" * 70, flush=True)


@app.local_entrypoint()
def evaluate(run: str, split: str = "val"):
    print(_evaluate_pipeline(run, split))


@app.local_entrypoint()
def benchmark(run: str):
    _preflight.remote(run)
    print(_benchmark.remote(run))


@app.local_entrypoint()
def analyze_geometry(run: str, split: str = "val", iou: float = 0.5):
    res = _analyze_geometry.remote(run, split, iou)
    out_dir = Path("reports/geometry") / run.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    if res.get("geometry_summary_csv"):
        (out_dir / "geometry_summary.csv").write_text(res["geometry_summary_csv"])
    if res.get("size_summary_csv"):
        (out_dir / "size_summary.csv").write_text(res["size_summary_csv"])
    if res.get("error_summary_csv"):
        (out_dir / "error_summary.csv").write_text(res["error_summary_csv"])
    if res.get("report_json"):
        (out_dir / "report.json").write_text(res["report_json"])
    print(f"Artifacts successfully saved locally to {out_dir}")
    print("\n" + "=" * 35 + " GEOMETRY SUMMARY " + "=" * 35)
    print(res.get("geometry_summary_csv", ""))
    print("\n" + "=" * 35 + " SIZE SUMMARY " + "=" * 35)
    print(res.get("size_summary_csv", ""))


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1, scaledown_window=2, secrets=[wandb_secret], timeout=300)
def _organize_wandb(project: str):
    import wandb

    from tinydet.utils.wandb_runs import tag_latest_runs

    return tag_latest_runs(wandb.Api(timeout=30), project)


@app.local_entrypoint()
def organize_wandb(project: str = "hoangngan/tiny-person-research"):
    """Tag latest run per model; preserve all historical W&B logs."""
    print(_organize_wandb.remote(project))


@app.function(image=image, cpu=POSTPROCESS_CPU, memory=POSTPROCESS_MEMORY_MB, max_containers=1,
              scaledown_window=2, volumes=mounts, timeout=3600)
def _finish_evaluation(run: str, artifact: str, split: str):
    from tinydet.evaluation.evaluator import finalize_predictions
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    target = resolve_run(RUN_ROOT, run)
    result = finalize_predictions(target, Path(artifact), split)
    runs.commit()
    return result


@app.function(image=image, cpu=LIGHT_CPU, memory=LIGHT_MEMORY_MB, max_containers=1,
              scaledown_window=2, volumes=mounts, timeout=600)
def _preflight(run: str, split: str = "", image_path: str = "", reuse_training: bool = False):
    import json

    from tinydet.evaluation.evaluator import preflight
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    target = resolve_run(RUN_ROOT, run)
    checkpoint = target / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        alt = target / "training" / "weights" / "best.pt"
        if alt.is_file():
            (target / "checkpoints").mkdir(parents=True, exist_ok=True)
            shutil.copy2(alt, checkpoint)
            last_alt = target / "training" / "weights" / "last.pt"
            if last_alt.is_file():
                shutil.copy2(last_alt, target / "checkpoints" / "last.pt")
            runs.commit()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if image_path and not Path(image_path).is_file():
        raise FileNotFoundError(image_path)
    if split:
        preflight(target, split)
    # Only compare's just-trained run opts into native final-best validation.
    metric_file = target / "metrics" / "val.json"
    if reuse_training and split == "val" and metric_file.is_file():
        result = json.loads(metric_file.read_text())
        if result.get("source") == "training_final_best":
            return result
    return None


def _evaluate_pipeline(run: str, split: str, reuse_training: bool = False):
    existing = _preflight.remote(run, split, reuse_training=reuse_training)
    if existing is not None:
        return existing
    artifact = _evaluate.remote(run, split)
    return _finish_evaluation.remote(run, artifact, split)


@app.function(
    image=image,
    gpu="A10",
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    cpu=CPU_LIMIT,
    memory=MEMORY_LIMIT_MB,
    max_containers=1,
    volumes=mounts,
    timeout=3600,
)
def _run_comparison_report():
    import subprocess
    import sys

    from ultralytics import YOLO

    from tinydet.evaluation.iou_metrics import ap_by_iou
    from tinydet.models.ultralytics import register_ultralytics_modules
    from tinydet.utils.dataset import normalize_dataset_yaml

    register_ultralytics_modules()
    runs.reload()
    data.reload()

    ckpts = {
        "QAC_v3_ep50": RUN_ROOT / "tiny_A2_p2_qac_v3" / "20261026-110551-6fceb3b-seed42" / "stage1" / "weights" / "best.pt",
        "DFL16_ep50": RUN_ROOT / "tiny_A2_p2_dfl16" / "20261026-110551-6fceb3b-seed42" / "stage1" / "weights" / "best.pt",
        "DFL16_ep100": RUN_ROOT / "tiny_A2_p2_dfl16" / "20261026-110551-6fceb3b-seed42" / "checkpoints" / "best.pt",
    }
    for name, p in ckpts.items():
        if not p.is_file():
            alt = RUN_ROOT / "tiny_A2_p2_dfl16" / "20261026-110551-6fceb3b-seed42" / "training" / "weights" / "best.pt"
            if name == "DFL16_ep100" and alt.is_file():
                ckpts[name] = alt
            else:
                raise FileNotFoundError(f"Missing checkpoint for {name}: {p}")

    raw_data_yaml = DATA_ROOT / "tinyperson-yolo-v2" / "data.yaml"
    if not raw_data_yaml.is_file():
        raise FileNotFoundError(f"Data yaml not found: {raw_data_yaml}")
    data_yaml = normalize_dataset_yaml(raw_data_yaml)

    # 1. Measure official Ultralytics mAP metrics
    map_results = {}
    for name, p in ckpts.items():
        print(f"--> Validating mAP for {name} from {p}...", flush=True)
        model = YOLO(str(p))
        val_res = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=1280,
            batch=8,
            device="cuda:0",
            plots=False,
            save_json=False,
            verbose=False,
        )
        box = val_res.box
        map_results[name] = {
            "mAP50-95": float(box.map),
            "mAP50": float(box.map50),
            "mAP75": float(box.map75),
            "precision": float(box.mp),
            "recall": float(box.mr),
            **ap_by_iou(box),
        }
        del model

    # 2. Measure per-bin accuracy & geometry errors using analyze_msr_validation.py
    out_dir = RUN_ROOT / "comparison_report_qac_v3_vs_dfl16"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = DATA_ROOT / "tinyperson-yolo-v2" / "images" / "val"
    lbl_dir = DATA_ROOT / "tinyperson-yolo-v2" / "labels" / "val"

    cmd = [
        sys.executable,
        "/root/scripts/analyze_msr_validation.py",
        "--images", str(img_dir),
        "--labels", str(lbl_dir),
        "--imgsz", "1280",
        "--device", "cuda:0",
        "--cpu-threads", "4",
        "--output", str(out_dir),
    ]
    for name, p in ckpts.items():
        cmd.extend(["--checkpoint", f"{name}={p}"])

    print("--> Running size-bin & geometry analysis...", flush=True)
    subprocess.run(cmd, check=True)
    runs.commit()

    return {
        "map_results": map_results,
        "size_summary_csv": (out_dir / "size_summary.csv").read_text(),
        "geometry_summary_csv": (out_dir / "geometry_summary.csv").read_text(),
        "error_summary_csv": (out_dir / "error_summary.csv").read_text(),
        "model_comparison_csv": (out_dir / "model_comparison.csv").read_text() if (out_dir / "model_comparison.csv").is_file() else "",
    }


@app.local_entrypoint()
def generate_comparison_report():
    res = _run_comparison_report.remote()
    local_dir = Path("reports/comparison_qac_v3_vs_dfl16")
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "map_results.json").write_text(json.dumps(res["map_results"], indent=2))
    (local_dir / "size_summary.csv").write_text(res["size_summary_csv"])
    (local_dir / "geometry_summary.csv").write_text(res["geometry_summary_csv"])
    (local_dir / "error_summary.csv").write_text(res["error_summary_csv"])
    if res.get("model_comparison_csv"):
        (local_dir / "model_comparison.csv").write_text(res["model_comparison_csv"])
    print(f"Results successfully saved locally to {local_dir}")


def _subset_annotations(
    annotation_source: Path, data_yaml: Path, split: str, destination: Path
) -> Path:
    """Thin wrapper around the shared split-subsetting helper."""
    from tinydet.evaluation.tinyperson_benchmark import subset_annotations_to_split

    return subset_annotations_to_split(annotation_source, data_yaml, split, destination)


@app.function(
    image=image,
    gpu="A10",
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    cpu=CPU_LIMIT,
    memory=MEMORY_LIMIT_MB,
    max_containers=2,
    volumes=mounts,
    timeout=10800,
)
def _run_checkpoint_evaluation(relative_checkpoint: str, name: str, split: str = "val"):
    """Run the full evaluation stack (benchmark, AP75 oracles, error analysis) on one checkpoint.

    Deliberately bypasses `preflight()`: a probe checkpoint lives outside a run
    directory and has no `config.yaml`, but `scripts/evaluate_checkpoint.py` only
    needs the checkpoint, the split YAML and the annotation JSON.
    """
    import subprocess
    import sys

    from tinydet.utils.dataset import normalize_dataset_yaml

    runs.reload()
    data.reload()

    checkpoint = Path(relative_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = RUN_ROOT / relative_checkpoint
    if not checkpoint.is_file() and checkpoint.name == "best.pt":
        # A finalized run may still keep its weights under the training folder.
        for candidate in (
            checkpoint.parent.parent / "training" / "weights" / "best.pt",
            checkpoint.parent.parent / "stage1" / "weights" / "best.pt",
        ):
            if candidate.is_file():
                checkpoint = candidate
                break
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    data_yaml = normalize_dataset_yaml(DATA_ROOT / "tinyperson-yolo-v2" / "data.yaml")
    if split == "test":
        annotation_source = DATA_ROOT / "tinyperson-yolo-v2" / "annotations" / "tiny_set_test.json"
    else:
        # The derivative is built from the erase train JSON, which already has the
        # ignore/uncertain regions painted out, so it carries no special annotations.
        annotation_source = DATA_ROOT / "tinyperson-yolo-v2" / "annotations" / "tiny_set_train.json"
    if not annotation_source.is_file():
        raise FileNotFoundError(annotation_source)

    output_dir = RUN_ROOT / "eval" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations = _subset_annotations(
        annotation_source, data_yaml, split, output_dir / f"annotations_{split}.json"
    )

    command = [
        sys.executable,
        "/root/scripts/evaluate_checkpoint.py",
        "--checkpoint", str(checkpoint),
        "--annotations", str(annotations),
        "--data", str(data_yaml),
        "--output", str(output_dir),
        "--split", split,
        "--class-mode", "merge",
        "--imgsz", "1280",
        "--batch-size", "8",
        "--device", "cuda:0",
        "--workers", "4",
    ]
    print("-->", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd="/root",
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    print(completed.stdout[-4000:], flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"evaluation failed ({completed.returncode})\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
        )
    summary = json.loads((output_dir / "summary.json").read_text())
    runs.commit()
    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "output": str(output_dir),
        "summary": summary,
    }


@app.local_entrypoint()
def evaluate_checkpoint(
    checkpoint: str,
    name: str,
    split: str = "val",
):
    """Evaluate one checkpoint with the repository stack (benchmark + oracles + error analysis).

    `checkpoint` is a path inside the runs Volume, either absolute or relative to its root.
    """
    result = _run_checkpoint_evaluation.remote(checkpoint, name, split)
    output = Path("reports/eval")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = result["summary"]
    print(json.dumps({
        "name": name,
        "checkpoint": result["checkpoint"],
        "benchmark_all": summary.get("benchmark", {}).get("all"),
        "oracle_AP75": {
            arm: value["metrics"]["all"]["AP@0.75"]
            for arm, value in (summary.get("oracle") or {}).items()
            if isinstance(value, dict) and "metrics" in value
        },
    }, indent=2))








@app.function(
    image=image,
    gpu="A10",
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    cpu=CPU_LIMIT,
    memory=MEMORY_LIMIT_MB,
    max_containers=2,
    volumes=mounts,
    timeout=10800,
)
def _run_tiled_checkpoint_evaluation(relative_checkpoint: str, name: str):
    """Run tiled crop + merge evaluation (640x512, overlap 100) on test split."""
    import json

    from PIL import Image
    from ultralytics import YOLO

    from tinydet.evaluation.tiled_inference import predict_tiled_image
    from tinydet.evaluation.tinyperson_benchmark import evaluate_tinyperson
    from tinydet.models.ultralytics import register_ultralytics_modules
    from tinydet.utils.dataset import normalize_dataset_yaml

    register_ultralytics_modules()
    runs.reload()
    data.reload()

    checkpoint = Path(relative_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = RUN_ROOT / relative_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = YOLO(checkpoint)
    # Use the filtered 786-image test annotations
    data_yaml = normalize_dataset_yaml(DATA_ROOT / "tinyperson-yolo-v2" / "data.yaml")
    raw_annotation_source = DATA_ROOT / "tinyperson-yolo-v2" / "annotations" / "tiny_set_test.json"
    if not raw_annotation_source.is_file():
        raise FileNotFoundError(raw_annotation_source)

    output_dir = RUN_ROOT / "eval" / name
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_source = _subset_annotations(
        raw_annotation_source, data_yaml, "test", output_dir / "annotations_test.json"
    )

    data_payload = json.loads(annotation_source.read_text())
    image_dir = DATA_ROOT / "tinyperson-yolo-v2" / "images" / "test"

    predictions = []
    print(f"--> Starting tiled inference for {len(data_payload['images'])} images...", flush=True)

    for idx, img_info in enumerate(data_payload["images"]):
        file_name = img_info["file_name"]
        basename = Path(file_name).name
        img_path = image_dir / basename
        if not img_path.is_file():
            # Check recursive
            found = list(image_dir.rglob(basename))
            if found:
                img_path = found[0]
            else:
                continue

        with Image.open(img_path) as pil_img:
            boxes, scores, classes = predict_tiled_image(
                model=model,
                img=pil_img,
                tile_w=640,
                tile_h=512,
                overlap=100,
                conf=0.001,
                merge_iou=0.50,
                nms_pre=2000,
                max_per_img=1000,
                device="cuda:0",
            )

        predictions.append({
            "path": str(img_path),
            "boxes": boxes,
            "scores": scores,
            "classes": classes,
        })
        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{len(data_payload['images'])} images...", flush=True)

    pred_file = output_dir / "predictions_tiled.json"
    pred_file.write_text(json.dumps({"images": predictions}))

    print("--> Running standardized TinyPerson benchmark on merged predictions...", flush=True)
    bench_result = evaluate_tinyperson(annotation_source, pred_file, workers=4,
                                       class_mode="merge")
    (output_dir / "benchmark.json").write_text(json.dumps(bench_result, indent=2))

    summary = {
        "checkpoint": str(checkpoint),
        "name": name,
        "protocol": "Tiled crop 640x512 + NMS merge 0.50",
        "benchmark": bench_result,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    runs.commit()

    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "summary": summary,
    }


@app.function(
    image=image,
    gpu="A10",
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    cpu=CPU_LIMIT,
    memory=MEMORY_LIMIT_MB,
    max_containers=4,
    volumes=mounts,
    timeout=3600,
)
def _run_visdrone_val(relative_checkpoint: str, name: str, split: str = "val"):
    """Evaluate a checkpoint on VisDrone dataset using YOLO validator."""
    import json

    from ultralytics import YOLO

    from tinydet.models.ultralytics import register_ultralytics_modules

    register_ultralytics_modules()
    runs.reload()
    data.reload()

    checkpoint = Path(relative_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = RUN_ROOT / relative_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    data_yaml = str(DATA_ROOT / "visdrone-yolo" / "data.yaml")
    output_dir = RUN_ROOT / "eval" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--> Validating {checkpoint} on VisDrone {split} split...", flush=True)
    model = YOLO(str(checkpoint))
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=1280,
        batch=8,
        workers=4,
        device="cuda:0",
        project=str(output_dir.parent),
        name=name,
        exist_ok=True,
        plots=False,
    )

    results = {
        "checkpoint": str(checkpoint),
        "name": name,
        "split": split,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "maps_per_class": [float(x) for x in metrics.box.maps],
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    result_file = output_dir / "val_summary.json"
    result_file.write_text(json.dumps(results, indent=2))
    runs.commit()
    print(f"--> Validation completed: mAP50={results['map50']:.4f}, mAP50-95={results['map50_95']:.4f}", flush=True)
    return results
