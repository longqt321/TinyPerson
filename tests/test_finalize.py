import json

import pytest

from tinydet.training.finalize import finalize


def test_finalize_requires_finished_training_and_can_retry(tmp_path, monkeypatch):
    run = tmp_path
    meta = run / "metadata.json"
    meta.write_text(json.dumps({"status": "RUNNING"}))
    with pytest.raises(ValueError, match="not finished"):
        finalize(run)
    meta.write_text(json.dumps({"status": "TRAINED"}))
    weights = run / "training" / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"checkpoint")
    (run / "training" / "results.csv").write_text("epoch,time\n1,2\n")
    monkeypatch.setattr("ultralytics.utils.plotting.plot_results", lambda **kwargs: None)
    finalize(run)
    finalize(run)
    assert (run / "checkpoints" / "best.pt").read_bytes() == b"checkpoint"
    assert (run / "metrics" / "train.csv").is_file()
    assert json.loads(meta.read_text())["status"] == "COMPLETED"
