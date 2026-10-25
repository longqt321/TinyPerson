import hashlib
import json
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
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
CPU_LIMIT = (2.0, 4.0)
MEMORY_LIMIT_MB = (8192, 24576)


def _dataset_yaml_path(path: str | Path) -> Path:
    dataset_yaml = Path(os.path.normpath(path))
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



@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1,
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
    report_path = RUN_ROOT / "preflight" / f"dataset-audit-{audit_id}.json"
    normalized_yaml = normalize_dataset_yaml(dataset_yaml)
    report = audit_dataset(normalized_yaml, imgsz, report_path)
    runs.commit()
    return {
        "passed": report["passed"],
        "data_yaml": report["data_yaml"],
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


@app.function(image=image, gpu="A10", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=4, volumes=mounts, timeout=21600)
def _train(config: str, commit: str, dirty: bool, config_text: str | None = None, wandb_group: str | None = None, dataset_audit: dict | None = None):
    os.chdir("/root")
    raw_config = yaml.safe_load(config_text) if config_text else yaml.safe_load(Path(config).read_text())
    volume_yaml = _dataset_yaml_path(raw_config["data"]["config"])
    volume_dataset = volume_yaml.parent
    if not volume_yaml.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {volume_yaml}")

    with tempfile.TemporaryDirectory(prefix="tinydet-data-", dir="/tmp") as temporary:
        local_dataset = Path(temporary) / "tinyperson"
        print(f"Copying dataset from {volume_dataset} to local container disk", flush=True)
        copytree_concurrent(volume_dataset, local_dataset)
        local_yaml = local_dataset / "data.yaml"
        dataset_config = yaml.safe_load(local_yaml.read_text())
        dataset_config["path"] = str(local_dataset.resolve())
        local_yaml.write_text(yaml.safe_dump(dataset_config, sort_keys=False))
        print("Dataset copy complete; starting training", flush=True)
        if config_text is not None:
            config_file = Path(temporary) / "experiment.yaml"
            config_file.write_text(config_text)
            config = str(config_file)
        run = train_model(config, (commit, dirty), local_yaml, wandb_group, dataset_audit)
        runs.commit()
        return str(run)


@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=4, scaledown_window=2, volumes={str(RUN_ROOT): runs}, timeout=1800)
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


@app.function(image=image, gpu="A10", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _evaluate(run: str, split: str):
    from tinydet.evaluation.evaluator import collect_predictions
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = collect_predictions(resolve_run(RUN_ROOT, run), split)
    runs.commit()
    return str(result)


@app.function(image=image, gpu="A10", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _benchmark(run: str):
    from tinydet.evaluation.benchmark import benchmark
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = benchmark(resolve_run(RUN_ROOT, run))
    runs.commit()
    return result


@app.function(image=image, gpu="A10", min_containers=0, buffer_containers=0,
              scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB,
              max_containers=1, volumes=mounts, timeout=3600)
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
    subprocess.run(
        [
            sys.executable, "/root/scripts/analyze_msr_validation.py",
            "--images", str(root / f"images/{split}"),
            "--labels", str(root / f"labels/{split}"),
            "--checkpoint", f"A2={checkpoint}", "--output", str(output),
            "--imgsz", str(config["train"]["imgsz"]), "--device", "cuda:0",
            "--cpu-threads", "4", "--iou", str(iou),
        ],
        check=True,
    )
    runs.commit()
    return {"output": str(output), "geometry_summary": str(output / "geometry_summary.csv")}


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


@app.local_entrypoint()
def compare(
    config: str = "configs/experiment/comparison.yaml",
    start_at: str = "",
    parallel: bool = False,
    reuse: bool = False,
    smoke: bool = False,
    preflight_only: bool = False,
    strict_resolution: bool = False,
    require_source_metadata: bool = False,
):
    """Audit original data first, then compare models; --smoke caps every run at 5 epochs."""
    from tinydet.utils.comparison import load_comparison
    from tinydet.utils.experiment import git_info

    experiments = load_comparison(config)
    training = [experiment for experiment in experiments if "run" not in experiment]
    existing = [experiment for experiment in experiments if "run" in experiment]
    if not training:
        raise ValueError("At least one model must be trained")
    if smoke:
        for experiment in training:
            experiment["train"] = dict(experiment["train"])
            experiment["train"]["epochs"] = 5
            experiment["train"]["patience"] = max(
                int(experiment["train"].get("patience", 6)), 6
            )
        print("SMOKE MODE: every model is limited to 5 epochs.", flush=True)

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
        return

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

    if parallel:
        for offset in range(0, len(to_train), 4):
            pending = []
            for experiment in to_train[offset:offset + 4]:
                print(f"Training {experiment['name']}", flush=True)
                options = {"secrets": [wandb_secret]} if experiment["output"].get("wandb_project") else {}
                pending.append(_train.with_options(**options).spawn(
                    config, commit, dirty, yaml.safe_dump(experiment), wandb_group, audit
                ))
            completed = [call.get() for call in pending]
            for run in completed:
                reference = str(Path(run).relative_to(RUN_ROOT))
                _finalize_pipeline(reference)
                finish(reference)
    else:
        for experiment in to_train:
            print(f"Training {experiment['name']}", flush=True)
            options = {"secrets": [wandb_secret]} if experiment["output"].get("wandb_project") else {}
            run = _train.with_options(**options).remote(
                config, commit, dirty, yaml.safe_dump(experiment), wandb_group, audit
            )
            reference = str(Path(run).relative_to(RUN_ROOT))
            _finalize_pipeline(reference)
            finish(reference)


@app.local_entrypoint()
def evaluate(run: str, split: str = "val"):
    print(_evaluate_pipeline(run, split))


@app.local_entrypoint()
def benchmark(run: str):
    _preflight.remote(run)
    print(_benchmark.remote(run))


@app.local_entrypoint()
def analyze_geometry(run: str, split: str = "val", iou: float = 0.5):
    print(json.dumps(_analyze_geometry.remote(run, split, iou), indent=2))


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1, scaledown_window=2, secrets=[wandb_secret], timeout=300)
def _organize_wandb(project: str):
    import wandb

    from tinydet.utils.wandb_runs import tag_latest_runs

    return tag_latest_runs(wandb.Api(timeout=30), project)


@app.local_entrypoint()
def organize_wandb(project: str = "hoangngan/tiny-person-research"):
    """Tag latest run per model; preserve all historical W&B logs."""
    print(_organize_wandb.remote(project))


@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1,
              scaledown_window=2, volumes=mounts, timeout=3600)
def _finish_evaluation(run: str, artifact: str, split: str):
    from tinydet.evaluation.evaluator import finalize_predictions
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    target = resolve_run(RUN_ROOT, run)
    result = finalize_predictions(target, Path(artifact), split)
    runs.commit()
    return result


@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1,
              scaledown_window=2, volumes=mounts, timeout=600)
def _preflight(run: str, split: str = "", image_path: str = "", reuse_training: bool = False):
    import json

    from tinydet.evaluation.evaluator import preflight
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    target = resolve_run(RUN_ROOT, run)
    if not (target / "checkpoints" / "best.pt").is_file():
        raise FileNotFoundError(target / "checkpoints" / "best.pt")
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


