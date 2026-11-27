"""Unit tests for cross-host runtime helpers (Modal, Colab, Kaggle, local)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tinydet.utils.dataset import tinyperson_group_key
from tinydet.utils.runtime import (
    default_hf_runs_repo,
    default_output_root,
    detect_host,
    deterministic_wandb_id,
    discover_dataset,
    resolve_secret,
    restore_run_from_drive,
    restore_run_from_hf,
    stable_run_id,
    sync_run_to_drive,
    sync_run_to_hf,
)


def test_detect_host_env_priority() -> None:
    saved = {k: os.environ.get(k) for k in ("MODAL_TASK_ID", "KAGGLE_KERNEL_RUN_TYPE", "COLAB_RELEASE_TAG")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        os.environ["MODAL_TASK_ID"] = "ta-123"
        assert detect_host() == "modal"
        os.environ.pop("MODAL_TASK_ID", None)

        os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
        assert detect_host() == "kaggle"
        os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)

        os.environ["COLAB_RELEASE_TAG"] = "release-2026"
        assert detect_host() == "colab"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_deterministic_wandb_id_is_stable_and_distinct() -> None:
    id1 = deterministic_wandb_id("tiny-person", "gauss-seed42")
    id2 = deterministic_wandb_id("tiny-person", "gauss-seed42")
    id3 = deterministic_wandb_id("tiny-person", "baseline-seed42")
    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 16


def test_default_output_root_per_host() -> None:
    saved = os.environ.pop("TINYDET_OUTPUT_ROOT", None)
    try:
        repo = Path("/tmp/fake_repo")
        assert default_output_root("modal", repo) == Path("/mnt/runs")
        assert default_output_root("kaggle", repo) == Path("/kaggle/working/runs")
        assert default_output_root("local", repo) == repo.resolve() / "runs"
        assert stable_run_id(42, "colab") == "colab-seed42"
    finally:
        if saved is not None:
            os.environ["TINYDET_OUTPUT_ROOT"] = saved


def test_discover_dataset_finds_nested_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ds = root / "datasets" / "derived" / "tinyperson-yolo-v2"
        ds.mkdir(parents=True)
        yaml_file = ds / "data.yaml"
        yaml_file.write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: person\n")
        found = discover_dataset(None, "/nonexistent/data.yaml", "local", root)
        assert found == yaml_file.resolve()


def test_tinyperson_group_key_patterns() -> None:
    assert tinyperson_group_key("labeled_images/youtube_V0001_I0000320.jpg") == "youtube_V0001"
    assert tinyperson_group_key("beach_clip_frame0042.png") == "beach_clip"
    assert tinyperson_group_key("seqA-105.jpg") == "seqA"
    assert tinyperson_group_key("standalone.jpg") == "standalone"


def test_default_hf_runs_repo_resolution() -> None:
    saved = os.environ.pop("TINYDET_HF_RUNS_REPO", None)
    try:
        assert default_hf_runs_repo("longqt321") == "longqt321/tinydet-runs"
        assert default_hf_runs_repo(None) is None
        os.environ["TINYDET_HF_RUNS_REPO"] = "custom/repo"
        assert default_hf_runs_repo("longqt321") == "custom/repo"
    finally:
        if saved is not None:
            os.environ["TINYDET_HF_RUNS_REPO"] = saved


def test_drive_sync_and_restore_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "local_runs" / "lgt_150e" / "colab-seed42"
        drive_root = root / "drive_runs"
        (run_dir / "training" / "weights").mkdir(parents=True)
        (run_dir / "metrics").mkdir(parents=True)
        (run_dir / "config.yaml").write_text("name: lgt_150e\n")
        (run_dir / "training" / "results.csv").write_text("epoch,mAP50\n1,0.1\n")
        (run_dir / "training" / "weights" / "last.pt").write_bytes(b"weights-last")
        (run_dir / "metrics" / "epoch.json").write_text("{}")

        sync_run_to_drive(run_dir, drive_root, force=True)
        assert (drive_root / "lgt_150e" / "colab-seed42" / "config.yaml").is_file()
        assert (drive_root / "lgt_150e" / "colab-seed42" / "training" / "weights" / "last.pt").is_file()

        fresh_run = root / "fresh_runs" / "lgt_150e" / "colab-seed42"
        assert restore_run_from_drive(fresh_run, drive_root) is True
        assert (fresh_run / "training" / "weights" / "last.pt").read_bytes() == b"weights-last"
        assert restore_run_from_drive(fresh_run, drive_root) is False


def test_hf_sync_and_restore_mocked(monkeypatch) -> None:
    uploaded: list[dict] = []

    class FakeHfApi:
        def __init__(self, token=None) -> None:
            self.token = token

        def create_repo(self, **kwargs) -> None:
            pass

        def upload_folder(self, **kwargs) -> None:
            uploaded.append(kwargs)

    def fake_snapshot_download(**kwargs) -> None:
        local_dir = Path(kwargs["local_dir"])
        target = local_dir / "lgt_150e" / "colab-seed42" / "training" / "weights" / "last.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"hf-last")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "runs" / "lgt_150e" / "colab-seed42"
        (run_dir / "training" / "weights").mkdir(parents=True)
        (run_dir / "training" / "weights" / "last.pt").write_bytes(b"local-last")

        sync_run_to_hf(run_dir, "user/tinydet-runs", token="hf_test", force=True)
        assert len(uploaded) == 1
        assert uploaded[0]["path_in_repo"] == "lgt_150e/colab-seed42"

        fresh_run = root / "fresh" / "lgt_150e" / "colab-seed42"
        assert restore_run_from_hf(fresh_run, "user/tinydet-runs", token="hf_test") is True
        assert (fresh_run / "training" / "weights" / "last.pt").read_bytes() == b"hf-last"


def test_resolve_secret_strips_whitespace(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "  hf_abc123\r\n ")
    assert resolve_secret("HF_TOKEN") == "hf_abc123"
    monkeypatch.setenv("HF_TOKEN", "   \r\n ")
    assert resolve_secret("HF_TOKEN") is None
