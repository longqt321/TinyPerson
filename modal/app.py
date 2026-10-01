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


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _evaluate(run: str, split: str):
    from tinydet.evaluation.evaluator import evaluate
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = evaluate(resolve_run(RUN_ROOT, run), split)
    runs.commit()
    return result


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
    _finalize.remote(str(Path(run).relative_to(RUN_ROOT)))
    print(run)


@app.local_entrypoint()
def compare(config: str = "configs/experiment/comparison.yaml", start_at: str = "", parallel: bool = False, reuse: bool = False):
    """Compare models; optionally reuse exact completed runs."""
    from tinydet.utils.comparison import load_comparison
    from tinydet.utils.experiment import git_info

    experiments = load_comparison(config)
    names = [experiment["name"] for experiment in experiments]
    if start_at and start_at not in names:
        raise ValueError(f"Unknown model: {start_at}")
    if start_at:
        experiments = experiments[names.index(start_at):]
    if any(experiment["hardware"]["gpu"] != "L4" for experiment in experiments):
        raise ValueError("Only L4 GPU is allowed by the resource cap")
    commit, dirty = git_info()
    wandb_group = f"compare-{uuid.uuid4().hex[:8]}"

    def finish(reference: str, val_done: bool = False, benchmark_done: bool = False):
        if val_done and benchmark_done:
            print(f"Reusing completed {reference}", flush=True)
            return
        if not val_done:
            print(f"Evaluating {reference}", flush=True)
            _evaluate.remote(reference, "val")
        if not benchmark_done:
            print(f"Benchmarking {reference}", flush=True)
            _benchmark.remote(reference)
        print(f"Completed {reference}", flush=True)

    to_train = []
    for experiment in experiments:
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
                pending.append(_train.with_options(**options).spawn(config, commit, dirty, yaml.safe_dump(experiment), wandb_group))
            completed = [call.get() for call in pending]
            for run in completed:
                reference = str(Path(run).relative_to(RUN_ROOT))
                _finalize.remote(reference)
                finish(reference)
    else:
        for experiment in to_train:
            print(f"Training {experiment['name']}", flush=True)
            options = {"secrets": [wandb_secret]} if experiment["output"].get("wandb_project") else {}
            run = _train.with_options(**options).remote(config, commit, dirty, yaml.safe_dump(experiment), wandb_group)
            reference = str(Path(run).relative_to(RUN_ROOT))
            _finalize.remote(reference)
            finish(reference)


@app.local_entrypoint()
def evaluate(run: str, split: str = "val"):
    print(_evaluate.remote(run, split))


@app.local_entrypoint()
def benchmark(run: str):
    print(_benchmark.remote(run))


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _erf(run: str, image_path: str, layer: int):
    from pathlib import Path

    from tinydet.analysis.erf import analyze_erf
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = analyze_erf(resolve_run(RUN_ROOT, run), Path(image_path), layer)
    runs.commit()
    return str(result)


@app.local_entrypoint()
def erf(run: str, image_path: str, layer: int = 16):
    print(_erf.remote(run, image_path, layer))


@app.function(image=image, gpu="L4", min_containers=0, buffer_containers=0, scaledown_window=2, cpu=CPU_LIMIT, memory=MEMORY_LIMIT_MB, max_containers=1, volumes=mounts, timeout=3600)
def _diagnose(run: str, image_path: str, layer: int):
    from pathlib import Path

    from tinydet.analysis.activations import analyze_activations
    from tinydet.utils.experiment import resolve_run

    runs.reload()
    result = analyze_activations(resolve_run(RUN_ROOT, run), Path(image_path), layer)
    runs.commit()
    return str(result)


@app.local_entrypoint()
def diagnose(run: str, image_path: str, layer: int = 16):
    print(_diagnose.remote(run, image_path, layer))


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1, scaledown_window=2, secrets=[wandb_secret], timeout=300)
def _organize_wandb(project: str):
    import wandb
    from tinydet.utils.wandb_runs import tag_latest_runs

    return tag_latest_runs(wandb.Api(timeout=30), project)


@app.local_entrypoint()
def organize_wandb(project: str = "hoangngan/tiny-person-research"):
    """Tag latest run per model; preserve all historical W&B logs."""
    print(_organize_wandb.remote(project))
