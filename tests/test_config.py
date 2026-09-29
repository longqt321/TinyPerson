from pathlib import Path

import pytest

from tinydet.utils.config import load_config


def test_baseline_config():
    assert load_config("configs/experiment/baseline.yaml").seed == 42


def test_missing_config(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\n")
    with pytest.raises(ValueError, match="Missing config fields"):
        load_config(path)
