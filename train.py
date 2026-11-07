"""Environment-agnostic training CLI for TinyDet experiments.

Single entry point for every CUDA host (Modal, Colab, Kaggle, RunPod, ...).
CPU training stays disabled on purpose: ``tinydet.training.trainer.train``
raises when CUDA is unavailable.

Examples
--------
Modal keeps using ``modal/app.py``; this CLI is for GPU hosts without Modal::

    python train.py \
        --config configs/experiment/p2refine_tinyperson_wacv.yaml \
        --data /content/data_local/tinyperson-yolo-v2/data.yaml \
        --output-root /content/drive/MyDrive/tinydet_runs \
        --device 0 --batch 8 --imgsz 1280 --workers 4 \
        --name p2refine_colab_seed42

Validate the whole path without a GPU (no training, no CUDA required)::

    python train.py --config configs/experiment/p2refine_tinyperson_wacv.yaml \
        --data datasets/tinyperson-yolo-v2/data.yaml --dry-run
"""

import argparse
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import yaml

from tinydet.utils.config import ExperimentConfig, load_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a TinyDet experiment on any CUDA GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML.")
    parser.add_argument("--data", type=Path, help="Override data.config (file or directory).")
    parser.add_argument(
        "--data-root", type=Path, help="Override the 'path' key inside the dataset YAML."
    )
    parser.add_argument("--output-root", type=Path, help="Override output.root (run directory).")
    parser.add_argument("--name", help="Override the experiment name (run folder).")
    parser.add_argument("--device", help="CUDA device list, e.g. '0' or '0,1'.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs.")
    parser.add_argument("--batch", type=int, help="Override train.batch.")
    parser.add_argument("--imgsz", type=int, help="Override train.imgsz.")
    parser.add_argument("--workers", type=int, help="Override train.workers.")
    parser.add_argument("--seed", type=int, help="Override seed.")
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable W&B logging for this run."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config, dataset and model, print statistics, then exit without training.",
    )
    return parser.parse_args(argv)


REPO_ROOT = Path(__file__).resolve().parent


def _resolve_model_config(value: str | Path, experiment_path: Path) -> Path:
    """Resolve a model YAML path that may be relative to CWD, repo root, or configs/.

    Experiment YAMLs store ``model.config`` relative to the repository root
    (e.g. ``configs/model/foo.yaml``), so a run started from another working
    directory must still find it.
    """
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


def _resolve_paths(
    config: ExperimentConfig, args: argparse.Namespace, experiment_path: Path
) -> ExperimentConfig:
    """Make every path absolute so the run no longer depends on the working directory."""
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
    if args.no_wandb:
        output["wandb_project"] = None

    train = dict(config.train)
    for key in ("epochs", "batch", "imgsz", "workers"):
        value = getattr(args, key)
        if value is not None:
            train[key] = value
    if args.device is not None:
        train["device"] = args.device

    seed = config.seed if args.seed is None else args.seed
    name = config.name if args.name is None else args.name
    return replace(
        config, name=name, seed=seed, model=model, data=data, train=train, output=output
    )


def _print_summary(config: ExperimentConfig, dataset_yaml: Path) -> None:
    print("=" * 70)
    print(f"Experiment : {config.name}")
    print(f"Seed       : {config.seed}")
    print(f"Model YAML : {config.model['config']}")
    print(f"Dataset    : {dataset_yaml}")
    print(f"Run root   : {config.output['root']}")
    print(f"W&B        : {config.output.get('wandb_project') or 'disabled'}")
    for key in ("epochs", "imgsz", "batch", "workers", "optimizer", "lr0", "device"):
        if key in config.train:
            print(f"{key:<11}: {config.train[key]}")
    print("=" * 70, flush=True)


def _dry_run(config: ExperimentConfig, dataset_yaml: Path) -> None:
    """Build the model and dataset objects on CPU; never trains."""
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
    val_images = len(dataset_images(dataset_yaml, "val"))
    print(f"layers     : {layers}")
    print(f"parameters : {params:,}")
    print(f"gradients  : {gradients:,}")
    print(f"GFLOPs     : {gflops:.3f} @ {imgsz}x{imgsz}")
    print(f"strides    : {strides}")
    print(f"val images : {val_images}")
    print("[DRY RUN] config, dataset and model all resolve. No training performed.", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    experiment_path = Path(args.config).expanduser().resolve()
    config = _resolve_paths(load_config(experiment_path), args, experiment_path)

    from tinydet.utils.dataset import normalize_dataset_yaml, resolve_dataset_yaml

    raw_dataset = resolve_dataset_yaml(config.data["config"])
    dataset_yaml = normalize_dataset_yaml(raw_dataset, root=args.data_root)
    _print_summary(config, dataset_yaml)

    if args.dry_run:
        _dry_run(config, dataset_yaml)
        return 0

    import torch

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Training requires a CUDA GPU; CPU training is "
            "intentionally disabled. Use --dry-run to validate the configuration, or "
            "run on a GPU host (Modal, Colab, Kaggle, RunPod, ...).",
            file=sys.stderr,
        )
        return 1

    from tinydet.training.trainer import train

    with tempfile.TemporaryDirectory(prefix="tinydet-cfg-") as temporary:
        config_file = Path(temporary) / "experiment.yaml"
        config_file.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
        run = train(config_file, dataset_config=dataset_yaml)
    print(f"Run directory: {run}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
