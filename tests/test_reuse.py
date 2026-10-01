import json
from pathlib import Path

import yaml

from tinydet.utils.reuse import find_reusable_run


def test_reuse_requires_exact_config_completed_run_checkpoint_and_model_hash(tmp_path: Path):
    expected = {"name": "model", "train": {"epochs": 5}}
    run = tmp_path / "model" / "run-1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").touch()
    (run / "config.yaml").write_text(yaml.safe_dump(expected))
    (run / "metadata.json").write_text(json.dumps({"status": "COMPLETED", "model_config_sha256": "abc"}))
    assert find_reusable_run(tmp_path, expected, "abc")["run"] == "model/run-1"
    assert find_reusable_run(tmp_path, expected, "other") is None
    assert find_reusable_run(tmp_path, {"name": "model", "train": {"epochs": 10}}, "abc") is None
    (run / "checkpoints" / "best.pt").unlink()
    assert find_reusable_run(tmp_path, expected, "abc") is None
