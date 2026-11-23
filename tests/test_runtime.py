"""Unit tests for cross-host runtime helpers (Modal, Colab, Kaggle, local)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tinydet.utils.runtime import (
    default_output_root,
    detect_host,
    deterministic_wandb_id,
    discover_dataset,
    stable_run_id,
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
