import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from image import image
from paths import DATA_ROOT, MODEL_ROOT, RUN_ROOT

import modal
from tinydet.training.trainer import train as train_model

app = modal.App("tiny-person-research")
data = modal.Volume.from_name("tiny-person-data", create_if_missing=True)
models = modal.Volume.from_name("tiny-person-models", create_if_missing=True)
runs = modal.Volume.from_name("tiny-person-runs", create_if_missing=True)
mounts = {str(DATA_ROOT): data, str(MODEL_ROOT): models, str(RUN_ROOT): runs}


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


@app.function(image=image, gpu="L4", volumes=mounts, timeout=86400)
def _train(config: str, commit: str, dirty: bool):
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
        return str(train_model(config, (commit, dirty), local_dataset / "data.yaml"))


@app.function(image=image, gpu="L4", volumes=mounts, timeout=86400)
def _evaluate(run: str, split: str):
    from tinydet.evaluation.evaluator import evaluate
    from tinydet.utils.experiment import resolve_run

    return evaluate(resolve_run(RUN_ROOT, run), split)


@app.function(image=image, gpu="L4", volumes=mounts, timeout=3600)
def _benchmark(run: str):
    from tinydet.evaluation.benchmark import benchmark
    from tinydet.utils.experiment import resolve_run

    return benchmark(resolve_run(RUN_ROOT, run))


@app.local_entrypoint()
def train(config: str):
    from tinydet.utils.config import load_config
    from tinydet.utils.experiment import git_info

    gpu = load_config(config).hardware["gpu"]
    commit, dirty = git_info()
    print(_train.with_options(gpu=gpu).remote(config, commit, dirty))


@app.local_entrypoint()
def evaluate(run: str, split: str = "val"):
    print(_evaluate.remote(run, split))


@app.local_entrypoint()
def benchmark(run: str):
    print(_benchmark.remote(run))


@app.function(image=image, gpu="L4", volumes=mounts, timeout=3600)
def _erf(run: str, image_path: str, layer: int):
    from pathlib import Path

    from tinydet.analysis.erf import analyze_erf
    from tinydet.utils.experiment import resolve_run

    return str(analyze_erf(resolve_run(RUN_ROOT, run), Path(image_path), layer))


@app.local_entrypoint()
def erf(run: str, image_path: str, layer: int = 16):
    print(_erf.remote(run, image_path, layer))


@app.function(image=image, gpu="L4", volumes=mounts, timeout=3600)
def _diagnose(run: str, image_path: str, layer: int):
    from pathlib import Path

    from tinydet.analysis.activations import analyze_activations
    from tinydet.utils.experiment import resolve_run

    return str(analyze_activations(resolve_run(RUN_ROOT, run), Path(image_path), layer))


@app.local_entrypoint()
def diagnose(run: str, image_path: str, layer: int = 16):
    print(_diagnose.remote(run, image_path, layer))
