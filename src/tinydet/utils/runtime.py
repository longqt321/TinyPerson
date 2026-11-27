"""Runtime detection, secret sync (W&B/HF), and dataset provisioning for Modal, Colab, and Kaggle."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch

DEFAULT_HF_DATASET_REPO = "longqt321/tinydet-benchmarks"
DEFAULT_HF_ARCHIVE_NAME = "tinyperson-yolo-v2.tar.zst"
DATASET_ARCHIVE_MAP: dict[str, str] = {
    "tinyperson-yolo-v2": "tinyperson-yolo-v2.tar.zst",
    "tinyperson": "tinyperson-yolo-v2.tar.zst",
    "visdrone-yolo": "visdrone-yolo.tar.zst",
    "visdrone": "visdrone-yolo.tar.zst",
}
RUN_SYNC_EXACT_FILES = (
    Path("config.yaml"),
    Path("metadata.json"),
    Path("environment.txt"),
    Path("training/args.yaml"),
    Path("training/results.csv"),
    Path("training/completed.flag"),
    Path("training/weights/last.pt"),
    Path("training/weights/best.pt"),
)
RUN_SYNC_ALLOW_PATTERNS = [
    "config.yaml",
    "metadata.json",
    "environment.txt",
    "training/args.yaml",
    "training/results.csv",
    "training/completed.flag",
    "training/weights/last.pt",
    "training/weights/best.pt",
    "checkpoints/*",
    "metrics/*",
]
_HF_ENSURED_REPOS: set[str] = set()
_HF_INFLIGHT_FUTURES: dict[str, Future[Any]] = {}


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
    if value and value.strip():
        return value.strip()

    with suppress(Exception):
        from google.colab import userdata  # type: ignore[import-not-found]

        value = userdata.get(name)
        if value and value.strip():
            return value.strip()

    with suppress(Exception):
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        value = UserSecretsClient().get_secret(name)
        if value and value.strip():
            return value.strip()

    if name == "HF_TOKEN":
        with suppress(Exception):
            from huggingface_hub import get_token

            cached = get_token()
            if cached and cached.strip():
                return cached.strip()

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


def infer_dataset_slug(hint: str | Path | None) -> str:
    """Map any dataset path or YAML hint onto canonical ``tinyperson-yolo-v2`` or ``visdrone-yolo``."""
    text = str(hint or "").lower()
    if "visdrone" in text:
        return "visdrone-yolo"
    return "tinyperson-yolo-v2"


def _find_yaml_under(root: Path, preferred_dir: str = "tinyperson-yolo-v2") -> Path | None:
    if not root.is_dir():
        return None
    direct = root / preferred_dir / "data.yaml"
    if direct.is_file():
        return direct.resolve()
    for match in sorted(root.glob(f"**/{preferred_dir}/data.yaml")):
        if match.is_file():
            return match.resolve()
    return None


def _extract_archive(archive_path: Path, dest_parent: Path) -> None:
    dest_parent.mkdir(parents=True, exist_ok=True)
    name = archive_path.name
    if name.endswith((".tar.zst", ".zst")):
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        with (
            archive_path.open("rb") as compressed,
            dctx.stream_reader(compressed) as reader,
            tarfile.open(fileobj=reader, mode="r|") as tar,
        ):
            tar.extractall(dest_parent, filter="data")
    else:
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(dest_parent, filter="data")


def download_hf_dataset(
    repo_id: str,
    dest_parent: Path,
    archive_name: str = DEFAULT_HF_ARCHIVE_NAME,
) -> Path:
    """Download and stream-extract a ``.tar.zst`` (or ``.tar.gz``) dataset archive from Hugging Face."""
    from huggingface_hub import hf_hub_download

    dataset_dir_name = archive_name.removesuffix(".tar.zst").removesuffix(".tar.gz")
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

    print(f"[dataset] Extracting {archive_name} to {dest_parent}...", flush=True)
    _extract_archive(archive_path, dest_parent)

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
    """Locate ``data.yaml`` locally, or auto-download the matching archive from Hugging Face."""
    slug = infer_dataset_slug(explicit or config_data_path)
    archive_name = DATASET_ARCHIVE_MAP.get(slug, f"{slug}.tar.zst")

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    if env_data := os.environ.get("TINYDET_DATA"):
        candidates.append(Path(env_data).expanduser())
    if config_data_path:
        cfg_p = Path(config_data_path).expanduser()
        if cfg_p.name == "data.yaml":
            candidates.append(cfg_p)
            if not cfg_p.is_absolute():
                candidates.append(repo_root / cfg_p)

    candidates.extend(
        [
            repo_root / "datasets" / "derived" / slug / "data.yaml",
            Path(f"/content/data_local/{slug}/data.yaml"),
            Path(f"/kaggle/working/data_local/{slug}/data.yaml"),
            Path(f"/mnt/data/{slug}/data.yaml"),
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
        found = _find_yaml_under(root, preferred_dir=slug)
        if found is not None:
            return found

    if hf_repo:
        return download_hf_dataset(
            hf_repo,
            default_local_data_root(host, repo_root),
            archive_name=archive_name,
        )

    raise FileNotFoundError(
        f"Could not locate dataset '{slug}/data.yaml' automatically.\n"
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


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _snapshot_run_artifacts(run_dir: Path, dest_dir: Path) -> None:
    """Atomically snapshot essential checkpoints & metrics so concurrent training cannot corrupt uploads."""
    for rel in RUN_SYNC_EXACT_FILES:
        src = run_dir / rel
        if src.is_file():
            _atomic_copy(src, dest_dir / rel)
    for folder_name in ("checkpoints", "metrics"):
        folder_src = run_dir / folder_name
        if folder_src.is_dir():
            folder_dst = dest_dir / folder_name
            for item in folder_src.iterdir():
                if item.is_file():
                    _atomic_copy(item, folder_dst / item.name)


def sync_run_to_drive(
    run_dir: Path,
    drive_root: Path,
    *,
    every_n_epochs: int = 5,
    epoch_counter: list[int] | None = None,
    force: bool = False,
) -> None:
    """Sync essential checkpoints, configs, and metrics from local SSD to Google Drive."""
    if not force and epoch_counter is not None:
        epoch_counter[0] += 1
        if epoch_counter[0] % max(1, every_n_epochs) != 0:
            return
    if not run_dir.is_dir():
        return
    dest_dir = drive_root / run_dir.parent.name / run_dir.name
    try:
        _snapshot_run_artifacts(run_dir, dest_dir)
    except Exception as exc:  # noqa: BLE001 - Drive FUSE hiccup must not abort training
        print(f"[sync] Warning: Drive backup failed: {exc}", flush=True)


def restore_run_from_drive(run_dir: Path, drive_root: Path) -> bool:
    """Restore an interrupted run from Google Drive to local SSD if ``last.pt`` is absent."""
    if (run_dir / "training/weights/last.pt").is_file() or (run_dir / "training/completed.flag").is_file():
        return False
    backup_dir = drive_root / run_dir.parent.name / run_dir.name
    if not (backup_dir / "training/weights/last.pt").is_file() and not (backup_dir / "training/completed.flag").is_file():
        return False
    print(f"[restore] Restoring run from Drive: {backup_dir} -> {run_dir}", flush=True)
    shutil.copytree(backup_dir, run_dir, dirs_exist_ok=True)
    return True


def ensure_hf_auth() -> tuple[str | None, str | None]:
    """Resolve ``HF_TOKEN``, export it to ``os.environ``, and return ``(token, username)``."""
    token = resolve_secret("HF_TOKEN")
    if not token:
        return None, None
    os.environ["HF_TOKEN"] = token
    try:
        from huggingface_hub import HfApi

        username = HfApi(token=token).whoami().get("name")
        return token, str(username) if username else None
    except Exception as exc:  # noqa: BLE001 - network/token check is best-effort
        print(f"[hf] Warning: could not verify HF_TOKEN ({exc})", flush=True)
        return token, None


def default_hf_runs_repo(username: str | None) -> str | None:
    """Return the Hugging Face model repo used for checkpoint backup/resume."""
    if env_repo := os.environ.get("TINYDET_HF_RUNS_REPO"):
        return env_repo.strip() or None
    if username:
        return f"{username}/tinydet-runs"
    return None


def sync_run_to_hf(
    run_dir: Path,
    repo_id: str,
    token: str | None = None,
    *,
    every_n_epochs: int = 5,
    epoch_counter: list[int] | None = None,
    force: bool = False,
    run_as_background: bool = True,
) -> None:
    """Snapshot artifacts to ``_hf_staging`` and upload single-flight to Hugging Face Hub."""
    if not force and epoch_counter is not None:
        epoch_counter[0] += 1
        if epoch_counter[0] % max(1, every_n_epochs) != 0:
            return
    if not run_dir.is_dir():
        return
    try:
        from huggingface_hub import HfApi

        path_in_repo = f"{run_dir.parent.name}/{run_dir.name}"
        prev = _HF_INFLIGHT_FUTURES.pop(path_in_repo, None)
        if prev is not None:
            with suppress(Exception):
                prev.result(timeout=180)

        staging_dir = run_dir / "_hf_staging"
        if staging_dir.is_dir():
            shutil.rmtree(staging_dir, ignore_errors=True)
        _snapshot_run_artifacts(run_dir, staging_dir)

        api = HfApi(token=token)
        if repo_id not in _HF_ENSURED_REPOS:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
            _HF_ENSURED_REPOS.add(repo_id)

        if bool(run_as_background and not force):
            print(
                f"[hf-sync] Queued background backup: {path_in_repo} -> hf://{repo_id}",
                flush=True,
            )
            fut = api.upload_folder(
                folder_path=str(staging_dir),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=RUN_SYNC_ALLOW_PATTERNS,
                commit_message=f"Sync {path_in_repo}",
                run_as_future=True,
            )
            if isinstance(fut, Future):
                _HF_INFLIGHT_FUTURES[path_in_repo] = fut
        else:
            print(
                f"[hf-sync] Uploading final backup: {path_in_repo} -> hf://{repo_id}...",
                flush=True,
            )
            api.upload_folder(
                folder_path=str(staging_dir),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=RUN_SYNC_ALLOW_PATTERNS,
                commit_message=f"Sync {path_in_repo}",
                run_as_future=False,
            )
            print(f"[hf-sync] Backup complete: {path_in_repo}", flush=True)
    except Exception as exc:  # noqa: BLE001 - HF Hub hiccup must not abort training
        print(f"[hf-sync] Warning: HF backup failed: {exc}", flush=True)


def restore_run_from_hf(
    run_dir: Path,
    repo_id: str,
    token: str | None = None,
) -> bool:
    """Restore an existing or interrupted run from Hugging Face Hub if absent locally."""
    if (run_dir / "training/weights/last.pt").is_file() or (run_dir / "training/completed.flag").is_file():
        return False
    try:
        from huggingface_hub import snapshot_download

        path_in_repo = f"{run_dir.parent.name}/{run_dir.name}"
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            allow_patterns=[f"{path_in_repo}/**"],
            ignore_patterns=["**/last.pt"] if False else None,
            local_dir=str(run_dir.parent.parent),
        )
        restored = (run_dir / "training/weights/last.pt").is_file() or (
            run_dir / "training/completed.flag"
        ).is_file()
        if restored:
            print(f"[hf-restore] Restored run from hf://{repo_id}/{path_in_repo}", flush=True)
        return restored
    except Exception as exc:  # noqa: BLE001 - fresh runs have no remote snapshot yet
        # Fallback: if last.pt had a legacy size-mismatch on HF Hub, restore best.pt + flags/metrics
        with suppress(Exception):
            from huggingface_hub import snapshot_download

            path_in_repo = f"{run_dir.parent.name}/{run_dir.name}"
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                token=token,
                allow_patterns=[f"{path_in_repo}/**"],
                ignore_patterns=["**/last.pt"],
                local_dir=str(run_dir.parent.parent),
            )
            if (run_dir / "training/completed.flag").is_file():
                print(f"[hf-restore] Restored completed run (skipping last.pt) from hf://{repo_id}/{path_in_repo}", flush=True)
                return True
        print(f"[hf-restore] No remote checkpoint restored ({exc})", flush=True)
        return False
