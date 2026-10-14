import hashlib
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


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=4, volumes=mounts, timeout=21600)
def _train(config: str, commit: str, dirty: bool, config_text: str | None = None, wandb_group: str | None = None):
    os.chdir("/root")
    volume_dataset = DATA_ROOT / "tinyperson"
    volume_yaml = volume_dataset / "data.yaml"
    if not volume_yaml.is_file():
        raise FileNotFoundError(f"Dataset config does not exist: {volume_yaml}")

    with tempfile.TemporaryDirectory(prefix="tinydet-data-", dir="/tmp") as temporary:
        local_dataset = Path(temporary) / "tinyperson"
        print(f"Copying dataset from {volume_dataset} to local container disk", flush=True)
        copytree_concurrent(volume_dataset, local_dataset)
        print("Dataset copy complete; starting training", flush=True)
        if config_text is not None:
            config_file = Path(temporary) / "experiment.yaml"
            config_file.write_text(config_text)
            config = str(config_file)
        run = train_model(config, (commit, dirty), local_dataset / "data.yaml", wandb_group)
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


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _evaluate(run: str, split: str):
    from tinydet.evaluation.evaluator import collect_predictions
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = collect_predictions(resolve_run(RUN_ROOT, run), split)
    runs.commit()
    return str(result)


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _benchmark(run: str):
    from tinydet.evaluation.benchmark import benchmark
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = benchmark(resolve_run(RUN_ROOT, run))
    runs.commit()
    return result


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1, volumes={str(RUN_ROOT): runs}, timeout=300)
def _find_reusable(expected: dict, model_sha256: str):
    from tinydet.utils.reuse import find_reusable_run

    runs.reload()
    return find_reusable_run(RUN_ROOT, expected, model_sha256)


@app.local_entrypoint()
def train(config: str):
    from tinydet.utils.config import load_config
    from tinydet.utils.experiment import git_info

    experiment = load_config(config)
    gpu = experiment.hardware["gpu"]
    if gpu != "L4":
        raise ValueError("Only L4 GPU is allowed by the resource cap")
    commit, dirty = git_info()
    options = {"gpu": gpu}
    if experiment.output.get("wandb_project"):
        options["secrets"] = [wandb_secret]
    run = _train.with_options(**options).remote(config, commit, dirty)
    _finalize_pipeline(str(Path(run).relative_to(RUN_ROOT)))
    print(run)


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1,
              volumes={str(RUN_ROOT): runs}, timeout=60)
def _has_size_metrics(run: str, split: str):
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    target = resolve_run(RUN_ROOT, run)
    return (target / "metrics" / f"size_{split}.json").is_file()


@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1,
              volumes={str(RUN_ROOT): runs}, timeout=3600)
def _write_btrm_report(references: dict, report: dict, group: str):
    from tinydet.evaluation.btrm_report import write_btrm_report

    runs.reload()
    result = write_btrm_report(RUN_ROOT, references, report, group)
    runs.commit()
    return result


@app.local_entrypoint()
def compare(config: str = "configs/experiment/comparison.yaml", start_at: str = "", parallel: bool = False, reuse: bool = False, evaluate_only: bool = False):
    """Compare models, or safely finish evaluation without any retraining."""
    from tinydet.utils.comparison import load_comparison
    from tinydet.utils.experiment import git_info

    experiments = load_comparison(config)
    raw = yaml.safe_load(Path(config).read_text())
    execution = raw.get("execution", {})
    if not isinstance(execution, dict) or set(execution) - {"parallel"}:
        raise ValueError("execution accepts only a parallel boolean")
    if type(execution.get("parallel", False)) is not bool:
        raise ValueError("execution.parallel must be a boolean")
    parallel = parallel or execution.get("parallel", False)
    report = raw.get("report", {})
    if report and not isinstance(report, dict):
        raise ValueError("report must be a YAML mapping")
    report_enabled = bool(report.get("enabled", False))
    if report_enabled and report.get("split", "val") != "val":
        raise ValueError("Automatic report currently supports validation split only")
    names = [experiment["name"] for experiment in experiments]
    if start_at and start_at not in names:
        raise ValueError(f"Unknown model: {start_at}")
    if start_at:
        experiments = experiments[names.index(start_at):]
    if evaluate_only:
        reuse = True  # Never launch training from a recovery operation.
    if any(experiment["hardware"]["gpu"] != "L4" for experiment in experiments):
        raise ValueError("Only L4 GPU is allowed by the resource cap")
    commit, dirty = git_info()
    wandb_group = f"compare-{uuid.uuid4().hex[:8]}"
    references = {}

    def finish(reference: str, val_done: bool = False, benchmark_done: bool = False):
        if not val_done:
            print(f"Evaluating {reference}", flush=True)
            _evaluate_pipeline(reference, "val", reuse_training=True)
        if not benchmark_done:
            print(f"Benchmarking {reference}", flush=True)
            _benchmark.remote(reference)
        if report_enabled and report.get("size_metrics", True):
            if not _has_size_metrics.remote(reference, "val"):
                print(f"Size-wise AP/recall: {reference}", flush=True)
                artifact = _heavy_bottom_size_evaluate.remote(reference, "val")
                _finish_evaluation.remote(reference, artifact, "val", True)
        print(f"Completed {reference}", flush=True)

    to_train = []
    for experiment in experiments:
        match = None
        if reuse:
            model_sha256 = hashlib.sha256(
                Path(experiment["model"]["config"]).read_bytes()
            ).hexdigest()
            match = _find_reusable.remote(experiment, model_sha256)
        if match:
            if match["legacy"]:
                print(
                    f"Warning: {match['run']} has no model YAML hash; "
                    "check code and dataset before reusing", flush=True,
                )
            finish(match["run"], match["val"], match["benchmark"])
            references[experiment["name"]] = match["run"]
        else:
            if evaluate_only:
                raise FileNotFoundError(
                    f"No reusable completed checkpoint for {experiment['name']}. "
                    "Evaluation-only mode refuses to train; verify Modal Volume run status "
                    "and the exact saved experiment/model YAML."
                )
            to_train.append(experiment)

    errors = []
    if parallel:
        for offset in range(0, len(to_train), 4):
            pending = []
            for experiment in to_train[offset:offset + 4]:
                print(f"Training {experiment['name']}", flush=True)
                options = ({"secrets": [wandb_secret]}
                           if experiment["output"].get("wandb_project") else {})
                call = _train.with_options(**options).spawn(
                    config, commit, dirty, yaml.safe_dump(experiment), wandb_group
                )
                pending.append((experiment["name"], call))
            for name, call in pending:
                try:
                    run = call.get()
                    reference = str(Path(run).relative_to(RUN_ROOT))
                    _finalize_pipeline(reference)
                    finish(reference)
                    references[name] = reference
                except Exception as exc:
                    errors.append((name, str(exc)))
                    print(f"FAILED {name}: {exc}", flush=True)
    else:
        for experiment in to_train:
            name = experiment["name"]
            print(f"Training {name}", flush=True)
            options = ({"secrets": [wandb_secret]}
                       if experiment["output"].get("wandb_project") else {})
            try:
                run = _train.with_options(**options).remote(
                    config, commit, dirty, yaml.safe_dump(experiment), wandb_group
                )
                reference = str(Path(run).relative_to(RUN_ROOT))
                _finalize_pipeline(reference)
                finish(reference)
                references[name] = reference
            except Exception as exc:
                errors.append((name, str(exc)))
                print(f"FAILED {name}: {exc}", flush=True)

    if errors:
        raise RuntimeError(f"Comparison failed for: {errors}; successful runs remain saved")
    if report_enabled:
        required = {report.get(k) for k in ("baseline", "contrast", "proposed")}
        if None in required or len(required) != 3:
            raise ValueError("Report needs distinct baseline/contrast/proposed names")
        if not required <= set(references):
            print(
                f"Skipping incomplete report: missing {required - set(references)}. "
                "Use a full comparison to generate the three-arm report.",
                flush=True,
            )
        else:
            print("BTRM report:", _write_btrm_report.remote(references, report, wandb_group))


@app.local_entrypoint()
def evaluate(run: str, split: str = "val"):
    print(_evaluate_pipeline(run, split))


@app.local_entrypoint()
def benchmark(run: str):
    _preflight.remote(run)
    print(_benchmark.remote(run))


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _erf(run: str, image_path: str, layer: int):
    from pathlib import Path

    from tinydet.analysis.erf import analyze_erf
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = analyze_erf(resolve_run(RUN_ROOT, run), Path(image_path), layer, render=False)
    runs.commit()
    return str(result)


@app.local_entrypoint()
def erf(run: str, image_path: str, layer: int = 16):
    _preflight.remote(run, image_path=image_path)
    output = _erf.remote(run, image_path, layer)
    print(_render_diagnostic.remote(output, layer, "erf"))


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _diagnose(run: str, image_path: str, layer: int):
    from pathlib import Path

    from tinydet.analysis.activations import analyze_activations
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = analyze_activations(resolve_run(RUN_ROOT, run), Path(image_path), layer, render=False)
    runs.commit()
    return str(result)


@app.local_entrypoint()
def diagnose(run: str, image_path: str, layer: int = 16):
    _preflight.remote(run, image_path=image_path)
    output = _diagnose.remote(run, image_path, layer)
    print(_render_diagnostic.remote(output, layer, "activations"))


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
              scaledown_window=2, volumes=mounts, timeout=1800)
def _prepare_heavy_bottom(run: str, checkpoint: str, probe: str, output: str, count: int,
                          size: int, max_instances: int | None):
    import json

    from ultralytics import YOLO

    from tinydet.analysis.heavy_bottom import make_probe, resolve_stages
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    target = resolve_run(RUN_ROOT, run) if run else None
    selected = target / "checkpoints" / "best.pt" if target else Path(checkpoint)
    if not selected.is_file():
        raise FileNotFoundError(selected)
    if count <= 0 or size <= 0 or (max_instances is not None and max_instances <= 0):
        raise ValueError("count, size and max_instances must be positive")
    resolve_stages(YOLO(str(selected)).model)
    destination = (Path(output) if output else
                   target / "diagnostics" / "heavy_bottom" if target else
                   RUN_ROOT / "heavy_bottom" / selected.stem)
    probe_path = Path(probe)
    if not probe_path.is_file():
        make_probe(DATA_ROOT / "tinyperson" / "data.yaml", probe_path, size=size, count=count)
    selected_probe = json.loads(probe_path.read_text())
    if selected_probe["size"] != size:
        raise ValueError("Probe image size differs from --size")
    instances = selected_probe["instances"]
    if not instances or any(not Path(row["image"]).is_file() for row in instances[:max_instances]):
        raise ValueError("Probe contains no instances or missing images")
    print(f"Probe: {len(instances)} instances; processing {len(instances[:max_instances])}", flush=True)
    runs.commit()
    return str(selected), str(probe_path), str(destination), max_instances


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2,
              cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _heavy_bottom_diagnose(checkpoint: str, probe: str, output: str, max_instances: int | None):
    from tinydet.analysis.heavy_bottom import collect_diagnostics

    runs.reload()
    result = collect_diagnostics(Path(checkpoint), Path(probe), Path(output), max_instances)
    runs.commit()
    return str(result)


@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1,
              scaledown_window=2, volumes=mounts, timeout=3600)
def _finish_heavy_bottom(output: str):
    from tinydet.analysis.heavy_bottom import finalize_diagnostics

    runs.reload()
    result = finalize_diagnostics(Path(output))
    runs.commit()
    return str(result)


@app.local_entrypoint()
def heavy_bottom_diagnose(run: str = "", checkpoint: str = "", probe: str = "",
                          output: str = "", count: int = 200, size: int = 1280,
                          max_instances: int = 0):
    """Diagnose an explicit run best.pt or an explicit pretrained checkpoint on Modal."""
    if bool(run) == bool(checkpoint):
        raise ValueError("Specify exactly one of --run or --checkpoint")
    if run.endswith("/RUN_ID"):
        raise ValueError("RUN_ID is a placeholder. List runs with: modal volume ls tiny-person-runs /")
    probe = probe or str(RUN_ROOT / "heavy_bottom" / f"probe_{size}_{count}_seed42.json")
    prepared = _prepare_heavy_bottom.remote(run, checkpoint, probe, output, count, size,
                                           max_instances or None)
    artifact = _heavy_bottom_diagnose.remote(*prepared)
    print(_finish_heavy_bottom.remote(artifact))


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2,
              cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _heavy_bottom_size_evaluate(run: str, split: str):
    from tinydet.evaluation.size_metrics import collect_size_predictions
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    target = resolve_run(RUN_ROOT, run)
    config = yaml.safe_load((target / "config.yaml").read_text())
    artifact = collect_size_predictions(target / "checkpoints" / "best.pt",
        Path(config["data"]["config"]), target / f"evaluation-{split}-size" / "predictions.pt",
        split, config["train"]["imgsz"])
    runs.commit()
    return str(artifact)


@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1,
              scaledown_window=2, volumes=mounts, timeout=3600)
def _finish_evaluation(run: str, artifact: str, split: str, size_metrics: bool = False):
    from tinydet.evaluation.evaluator import finalize_predictions
    from tinydet.evaluation.size_metrics import finalize_size_predictions
    from tinydet.utils.experiment import resolve_run, write_json

    runs.reload()
    target = resolve_run(RUN_ROOT, run)
    if size_metrics:
        result = finalize_size_predictions(Path(artifact))
        write_json(target / "metrics" / f"size_{split}.json", result)
    else:
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


@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1,
              scaledown_window=2, volumes=mounts, timeout=1800)
def _render_diagnostic(output: str, layer: int, kind: str):
    from tinydet.analysis.activations import render_activations
    from tinydet.analysis.erf import render_erf

    runs.reload()
    result = (render_erf if kind == "erf" else render_activations)(Path(output), layer)
    runs.commit()
    return str(result)


@app.local_entrypoint()
def heavy_bottom_size_evaluate(run: str, split: str = "val"):
    _preflight.remote(run, split)
    artifact = _heavy_bottom_size_evaluate.remote(run, split)
    print(_finish_evaluation.remote(run, artifact, split, size_metrics=True))


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0,
              scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB,
              max_containers=1, volumes=mounts, timeout=3600)
def _representation_diagnose(run: str, config_text: str):
    """Analyze a completed frozen checkpoint; never initiate YOLO training."""
    from tinydet.analysis.representation import study
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    data.reload()
    target = resolve_run(RUN_ROOT, run)
    checkpoint = target / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    settings = yaml.safe_load(config_text)
    dataset_file = Path(settings["data"]["config"])
    if not dataset_file.is_file():
        raise FileNotFoundError(dataset_file)
    folder = settings.get("output", {}).get("folder", "diagnostics/representation")
    destination = (target / folder).resolve()
    if target.resolve() not in destination.parents:
        raise ValueError("Output folder must be inside the selected run")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.yaml").write_text(config_text)
    outcome = study(checkpoint, dataset_file, destination, settings,
                    allowed_output_root=target.resolve())
    runs.commit()
    return {"output": str(destination), "summary": outcome}


@app.local_entrypoint()
def representation_diagnose(run: str, config: str = "configs/analysis/representation.yaml"):
    """Run the three frozen-feature studies on one existing checkpoint."""
    from tinydet.utils.experiment import resolve_run

    if run.endswith("/RUN_ID"):
        raise ValueError("Replace RUN_ID with an existing Modal run id")
    source = Path(config)
    if not source.is_file():
        raise FileNotFoundError(source)
    settings = yaml.safe_load(source.read_text())
    if not isinstance(settings, dict):
        raise ValueError("Expected YAML configuration mapping")
    print(_representation_diagnose.remote(run, source.read_text()))


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0,
              scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB,
              max_containers=1, volumes=mounts, timeout=3600)
def _spatial_probe_extract(run: str, config_text: str):
    """GPU-only frozen feature extraction; one L4, no detector training."""
    from tinydet.analysis.spatial_probe import resolve_spatial_checkpoint, save_feature_cache
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    data.reload()
    target = resolve_run(RUN_ROOT, run)
    checkpoint = resolve_spatial_checkpoint(target)
    dataset = DATA_ROOT / "tinyperson" / "data.yaml"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    settings = yaml.safe_load(config_text)
    folder = settings.get("output", {}).get("folder", "diagnostics/spatial_probe_v1")
    destination = (target / folder).resolve()
    if target.resolve() not in destination.parents:
        raise ValueError("Spatial probe output must be inside the selected run")
    result = save_feature_cache(checkpoint, dataset, destination, settings,
                                allowed_output_root=target.resolve())
    runs.commit()
    return result


@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB,
              max_containers=1, scaledown_window=2, volumes={str(RUN_ROOT): runs},
              timeout=3600)
def _spatial_probe_finish(run: str, output: str, config_text: str):
    """CPU-only scikit-learn fitting, AP and group bootstrap; no paid GPU."""
    from tinydet.analysis.spatial_probe import finish_feature_cache
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    target = resolve_run(RUN_ROOT, run)
    result = finish_feature_cache(Path(output), yaml.safe_load(config_text),
                                  allowed_output_root=target.resolve())
    runs.commit()
    return result


@app.local_entrypoint()
def spatial_probe(run: str, config: str = "configs/analysis/spatial_probe.yaml"):
    """Frozen spatial sensitivity probe: L4 extraction, CPU-only statistics."""
    if run.endswith("/RUN_ID"):
        raise ValueError("Replace RUN_ID with a completed Modal run id")
    config_file = Path(config)
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    text = config_file.read_text()
    settings = yaml.safe_load(text)
    if not isinstance(settings, dict):
        raise ValueError("Expected YAML configuration mapping")
    if settings.get("experiment", {}).get("device") != "cuda":
        raise ValueError("Modal spatial probe requires experiment.device: cuda")
    extracted = _spatial_probe_extract.remote(run, text)
    print(f"GPU extraction: {extracted}", flush=True)
    print(f"CPU probe metrics: {_spatial_probe_finish.remote(run, extracted['output'], text)}")


# BTRM diagnostics are strictly post-hoc: they read frozen checkpoints and
# saved validation predictions. Mathematical audit is a hard prerequisite
# for every GPU feature-diagnostic call.
@app.function(image=image, cpu=2.0, memory=4096, max_containers=1,
              scaledown_window=2, volumes=mounts, timeout=600)
def _btrm_math_gate(report_id: str):
    from tinydet.analysis.btrm_math_audit import run_math_audit
    from tinydet.utils.experiment import write_json

    runs.reload()
    report = run_math_audit()  # raises on any numerical violation
    if report["status"] != "PASS":
        raise AssertionError("Cannot proceed to real features without a valid math audit")
    destination = RUN_ROOT / "comparisons" / "btrm_diagnostics" / report_id
    write_json(destination / "math_audit.json", report)
    runs.commit()
    return {"status": report["status"], "checks": report["checks"],
            "max_abs_error": report["max_abs_error"]}


@app.function(image=image, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB,
              max_containers=1, scaledown_window=2, volumes=mounts, timeout=3600)
def _btrm_saved_pr(references: dict, report_id: str, split: str):
    import json

    from tinydet.analysis.btrm_prediction_diagnostics import analyze_prediction_runs
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    root = RUN_ROOT / "comparisons" / "btrm_diagnostics" / report_id
    audit = json.loads((root / "math_audit.json").read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("Mathematical validation failed or was not run")
    paths = {model: resolve_run(RUN_ROOT, ref) for model, ref in references.items()}
    result = analyze_prediction_runs(paths, root, split)
    runs.commit()
    return result


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0,
              scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB,
              max_containers=1, volumes=mounts, timeout=3600)
def _btrm_real_features(run: str, report_id: str, max_images: int, split: str):
    import json

    from tinydet.analysis.btrm_feature_diagnostics import collect_features
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    audit = RUN_ROOT / "comparisons" / "btrm_diagnostics" / report_id / "math_audit.json"
    if json.loads(audit.read_text()).get("status") != "PASS":
        raise RuntimeError("Mathematical validation failed or was not run")
    result = collect_features(resolve_run(RUN_ROOT, run), max_images, split)
    runs.commit()
    return {"run": run, "images": result["images"],
            "gate_gain": result["effective_gain"],
            "saturation": result["all_feature_maps"]["g_btrm_gt_95"]}


@app.function(image=image, cpu=2.0, memory=4096, max_containers=1,
              scaledown_window=2, volumes=mounts, timeout=1200)
def _btrm_finalize_diagnostics(references: dict, report_id: str, split: str):
    from tinydet.analysis.btrm_diagnostic_report import assemble_report
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    root = RUN_ROOT / "comparisons" / "btrm_diagnostics" / report_id
    paths = {model: resolve_run(RUN_ROOT, ref) for model, ref in references.items()}
    result = assemble_report(paths, root, split)
    runs.commit()
    return result


@app.local_entrypoint()
def btrm_diagnose(
    a0: str = "btrm_a0/20261014-063504-0a3e661-seed42",
    a2: str = "btrm_a2/20261014-063507-0a3e661-seed42",
    a3: str = "btrm_a3/20261014-063508-0a3e661-seed42",
    max_images: int = 128,
    split: str = "val",
):
    """Audit equations first; analyze saved predictions and frozen P3 only if PASS."""
    if split not in {"val", "test"} or max_images < 1:
        raise ValueError("Invalid split or max_images")
    references = {"btrm_a0": a0, "btrm_a2": a2, "btrm_a3": a3}
    report_id = "audit-" + uuid.uuid4().hex[:10]
    print("STEP 1/4: independent float64 BTRM audit (CPU, no GPU)", flush=True)
    audit = _btrm_math_gate.remote(report_id)
    print("Math result:", audit, flush=True)
    if audit["status"] != "PASS":
        raise RuntimeError("Math audit did not pass; GPU stage canceled")
    print("STEP 2/4: reanalyze saved detector predictions (CPU)", flush=True)
    print(_btrm_saved_pr.remote(references, report_id, split), flush=True)
    print("STEP 3/4: measure frozen P3 features (GPU L4, batch=1)", flush=True)
    for run in references.values():
        print(_btrm_real_features.remote(run, report_id, max_images, split), flush=True)
    print("STEP 4/4: aggregate and write diagnostics (CPU)", flush=True)
    print(_btrm_finalize_diagnostics.remote(references, report_id, split), flush=True)
