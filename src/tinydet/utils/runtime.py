"""Runtime detection, secret sync (W&B/HF), and dataset provisioning for Modal, Colab, and Kaggle."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

DEFAULT_HF_DATASET_REPO = "longqt321/tinyperson-yolo"
DEFAULT_HF_ARCHIVE_NAME = "tinyperson-yolo-v2.tar.gz"


def detect_host() -> str:
    """Identify the active execution host: ``modal``, ``colab``, ``kaggle``, or ``local``."""
    if os.environ.get("MODAL_TASK_ID") or os.environ.get("MODAL_IS_REMOTE"):
        return "modal"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") or Path("/content").is_dir():
        return "colab"
    return "local"


def resolve_secret(name: str) -> str | None:
    """Read a secret from environment variables, Colab ``userdata``, or Kaggle Secrets."""
    value = os.environ.get(name)
    if value:
        return value

    with suppress(Exception):
        from google.colab import userdata  # type: ignore[import-not-found]

        value = userdata.get(name)
        if value:
            return value

    with suppress(Exception):
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        value = UserSecretsClient().get_secret(name)
        if value:
            return value

    return None


def ensure_wandb_auth() -> str | None:
    """Resolve ``WANDB_API_KEY`` and authenticate W&B non-interactively across hosts."""
    api_key = resolve_secret("WANDB_API_KEY")
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
        # Write to fast local SSD; sync_run_to_drive() backs up to Drive FUSE.
        return Path("/content/tinydet_runs")
    if host == "kaggle":
        return Path("/kaggle/working/runs")
    return (repo_root / "runs").resolve()


def default_local_data_root(host: str, repo_root: Path) -> Path:
    """Local fast-SSD directory where downloaded/staged datasets are unpacked."""
    if host == "colab":
        return Path("/content/data_local")
    if host == "kaggle":
        return Path("/kaggle/working/data_local")
    if host == "modal":
        return Path("/mnt/data")
    return (repo_root / "datasets" / "derived").resolve()


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


def download_hf_dataset(
    repo_id: str,
    dest_parent: Path,
    archive_name: str = DEFAULT_HF_ARCHIVE_NAME,
) -> Path:
    """Download and extract the verified TinyPerson YOLO archive from Hugging Face."""
    from huggingface_hub import hf_hub_download

    dataset_dir_name = archive_name.removesuffix(".tar.gz")
    target_yaml = dest_parent / dataset_dir_name / "data.yaml"
    if target_yaml.is_file():
        print(f"[dataset] Using cached dataset: {target_yaml}", flush=True)
        return target_yaml.resolve()

    token = resolve_secret("HF_TOKEN")
    print(f"[dataset] Downloading {archive_name} from hf://datasets/{repo_id}...", flush=True)
    archive_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=archive_name,
            repo_type="dataset",
            token=token,
        )
    )

    try:
        sha_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{archive_name}.sha256",
                repo_type="dataset",
                token=token,
            )
        )
        expected = sha_path.read_text().strip().split()[0]
        with archive_path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(
                f"Dataset archive SHA-256 mismatch: expected {expected}, got {actual}"
            )
        print(f"[dataset] SHA-256 verified: {actual[:12]}...", flush=True)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - checksum file is optional on custom repos
        print(f"[dataset] Skipping SHA-256 verification ({exc})", flush=True)

    dest_parent.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] Extracting to {dest_parent}...", flush=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_parent, filter="data")

    if not target_yaml.is_file():
        found = _find_yaml_under(dest_parent, preferred_dir=dataset_dir_name)
        if found is None:
            raise FileNotFoundError(f"Extracted archive did not contain data.yaml under {dest_parent}")
        return found
    return target_yaml.resolve()


def discover_dataset(
    explicit: Path | None,
    config_data_path: str | None,
    host: str,
    repo_root: Path,
    *,
    hf_repo: str | None = DEFAULT_HF_DATASET_REPO,
) -> Path:
    """Locate ``data.yaml`` locally, or auto-download from Hugging Face if absent."""
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
            Path("/kaggle/working/data_local/tinyperson-yolo-v2/data.yaml"),
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

    if hf_repo:
        return download_hf_dataset(hf_repo, default_local_data_root(host, repo_root))

    raise FileNotFoundError(
        "Could not locate dataset data.yaml automatically.\n"
        "Pass --data <path/to/data.yaml>, --hf-dataset <owner/repo>, or set TINYDET_DATA."
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


def default_drive_backup_root(host: str) -> Path | None:
    """Return Google Drive backup root on Colab when Drive is mounted."""
    if host != "colab":
        return None
    drive_root = Path("/content/drive/MyDrive")
    return (drive_root / "tinydet_runs") if drive_root.is_dir() else None


def sync_run_to_drive(
    run_dir: Path,
    drive_root: Path,
    *,
    every_n_epochs: int = 5,
    epoch_counter: list[int] | None = None,
    force: bool = False,
) -> None:
    """Sync essential checkpoints and metrics from local SSD to Google Drive."""
    if not force and epoch_counter is not None:
        epoch_counter[0] += 1
        if epoch_counter[0] % max(1, every_n_epochs) != 0:
            return
    if not run_dir.is_dir():
        return
    dest_dir = drive_root / run_dir.parent.name / run_dir.name
    try:
        for rel in (
            Path("training/weights/last.pt"),
            Path("training/weights/best.pt"),
            Path("training/results.csv"),
        ):
            src = run_dir / rel
            if src.is_file():
                dst = dest_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        metrics_src = run_dir / "metrics"
        if metrics_src.is_dir():
            metrics_dst = dest_dir / "metrics"
            metrics_dst.mkdir(parents=True, exist_ok=True)
            for item in metrics_src.iterdir():
                if item.is_file():
                    shutil.copy2(item, metrics_dst / item.name)
    except Exception as exc:  # noqa: BLE001 - Drive FUSE hiccup must not abort training
        print(f"[sync] Warning: Drive backup failed: {exc}", flush=True)
