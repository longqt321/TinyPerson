"""Runtime detection, W&B auth sync, and dataset discovery for Modal, Colab, and Kaggle."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import torch


def detect_host() -> str:
    """Identify the active execution host: ``modal``, ``colab``, ``kaggle``, or ``local``."""
    if os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_IS_REMOTE"):
        return "modal"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") or Path("/content").is_dir():
        return "colab"
    return "local"


def ensure_wandb_auth() -> str | None:
    """Resolve ``WANDB_API_KEY`` from env, Colab ``userdata``, or Kaggle ``UserSecretsClient``.

    When found, exports the key into ``os.environ["WANDB_API_KEY"]`` and logs in
    non-interactively so both explicit ``wandb.init`` calls and Ultralytics
    callbacks authenticate identically across Modal, Colab, and Kaggle.
    """
    api_key = os.environ.get("WANDB_API_KEY")

    if not api_key:
        try:
            from google.colab import userdata  # type: ignore[import-not-found]

            api_key = userdata.get("WANDB_API_KEY")
        except Exception:  # noqa: BLE001 - optional Colab runtime
            api_key = None

    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

            api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:  # noqa: BLE001 - optional Kaggle runtime
            api_key = None

    if not api_key:
        return None

    os.environ["WANDB_API_KEY"] = api_key
    try:
        import wandb

        wandb.login(key=api_key, relogin=False)
    except Exception as exc:  # noqa: BLE001 - W&B is optional logging
        print(f"W&B login warning: {exc}", flush=True)
    return api_key


def deterministic_wandb_id(project: str, run_name: str) -> str:
    """Generate a stable 16-char W&B run ID so resumed runs continue in-place."""
    digest = hashlib.sha256(f"{project}::{run_name}".encode()).hexdigest()
    return digest[:16]


def runtime_wandb_metadata() -> dict[str, Any]:
    """Return host and GPU metadata attached to every W&B run for cross-host parity."""
    host = detect_host()
    info: dict[str, Any] = {"runtime_host": host}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
    return info


def default_output_root(host: str, repo_root: Path) -> Path:
    """Choose a persistent, writable run root for the detected host."""
    env_override = os.environ.get("TINYDET_OUTPUT_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    if host == "modal":
        return Path("/mnt/runs")
    if host == "colab":
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.is_dir():
            return drive_root / "tinydet_runs"
        return Path("/content/tinydet_runs")
    if host == "kaggle":
        return Path("/kaggle/working/runs")
    return (repo_root / "runs").resolve()


def stable_run_id(seed: int, host: str) -> str:
    """Deterministic run folder name so rerunning the same command resumes cleanly."""
    return f"{host}-seed{seed}"


def _find_yaml_under(root: Path, preferred_dir: str = "tinyperson-yolo-v2") -> Path | None:
    if not root.is_dir():
        return None
    direct = root / preferred_dir / "data.yaml"
    if direct.is_file():
        return direct.resolve()
    for match in sorted(root.glob(f"**/{preferred_dir}/data.yaml")):
        if match.is_file():
            return match.resolve()
    for match in sorted(root.glob("**/data.yaml")):
        if match.is_file():
            return match.resolve()
    return None


def discover_dataset(
    explicit: Path | None,
    config_data_path: str | None,
    host: str,
    repo_root: Path,
) -> Path:
    """Locate ``data.yaml`` automatically across Modal, Colab, Kaggle, and local hosts."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    if env_data := os.environ.get("TINYDET_DATA"):
        candidates.append(Path(env_data).expanduser())
    if config_data_path:
        candidates.append(Path(config_data_path).expanduser())
        if not Path(config_data_path).is_absolute():
            candidates.append(repo_root / config_data_path)

    candidates.extend(
        [
            repo_root / "datasets" / "derived" / "tinyperson-yolo-v2" / "data.yaml",
            Path("/content/data_local/tinyperson-yolo-v2/data.yaml"),
            Path("/mnt/data/tinyperson-yolo-v2/data.yaml"),
        ]
    )

    for candidate in candidates:
        resolved = candidate / "data.yaml" if candidate.is_dir() else candidate
        if resolved.is_file():
            return resolved.resolve()

    search_roots: list[Path] = []
    if host == "colab":
        search_roots.extend([Path("/content/data_local"), Path("/content/drive/MyDrive")])
    elif host == "kaggle":
        search_roots.extend([Path("/kaggle/input"), Path("/kaggle/working")])
    search_roots.append(repo_root / "datasets")

    for root in search_roots:
        found = _find_yaml_under(root)
        if found is not None:
            return found

    raise FileNotFoundError(
        "Could not locate dataset data.yaml automatically.\n"
        "Pass --data <path/to/data.yaml> or set TINYDET_DATA, or place tinyperson-yolo-v2 under:\n"
        "  - Google Colab : /content/data_local/tinyperson-yolo-v2/data.yaml or Google Drive\n"
        "  - Kaggle       : /kaggle/input/<dataset-slug>/tinyperson-yolo-v2/data.yaml\n"
        "  - Local repo   : datasets/derived/tinyperson-yolo-v2/data.yaml"
    )


def stage_dataset_to_local_ssd(dataset_yaml: Path, host: str) -> Path:
    """Copy a Google Drive FUSE dataset to Colab local SSD so GPU training is not I/O starved."""
    if host != "colab":
        return dataset_yaml
    resolved = dataset_yaml.resolve()
    if not str(resolved).startswith("/content/drive/"):
        return resolved

    source_dir = resolved.parent
    local_dir = Path("/content/data_local") / source_dir.name
    local_yaml = local_dir / resolved.name
    if local_yaml.is_file():
        print(f"[stage] Using cached local SSD dataset: {local_yaml}", flush=True)
        return local_yaml

    print(f"[stage] Copying dataset from Drive FUSE to local SSD: {source_dir} -> {local_dir}", flush=True)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, local_dir, dirs_exist_ok=True)
    return local_yaml
