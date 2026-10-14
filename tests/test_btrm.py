"""Unit tests for BTRM mathematics, routing, serialization and experiment fairness."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from tinydet.modules.btrm import (
    GATE_INDEX,
    LocalBackgroundEvidence,
    evidence_score,
    install_feature_module,
    validate_feature_module,
)
from tinydet.utils.comparison import load_comparison


@pytest.mark.parametrize("mode", ["contrast", "btrm"])
def test_identity_gradient_and_shape(mode):
    module = LocalBackgroundEvidence(mode)
    x = torch.randn(2, 5, 19, 17, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.equal(y, x)  # guaranteed no feature alteration at init
    with torch.no_grad():
        module.alpha_raw.fill_(0.3)
    y = module(x)
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(module.alpha_raw.grad)


def test_affine_background_and_isolated_center_anomaly():
    h = w = 25
    xx = torch.arange(w).reshape(1, 1, 1, w).float()
    yy = torch.arange(h).reshape(1, 1, h, 1).float()
    smooth = (.1 + .05 * xx + .07 * yy).expand(1, 4, h, w).clone()
    baseline = evidence_score(smooth, "btrm")
    assert baseline[..., 4:-4, 4:-4].abs().max() < .005
    smooth[..., 12, 12] += 3
    anomaly = evidence_score(smooth, "btrm")
    assert anomaly[0, 0, 12, 12] > 20


def test_projection_contraction_at_every_spatial_location():
    torch.manual_seed(9)
    x = torch.randn(2, 9, 21, 21)
    contrast = evidence_score(x, "contrast")
    projected = evidence_score(x, "btrm")
    assert bool(torch.all(projected <= contrast + 1e-4))


def test_flat_features_finite_and_zero_score():
    x = torch.zeros(2, 8, 13, 13, requires_grad=True)
    gate = LocalBackgroundEvidence("btrm")
    gate.alpha_raw.data.fill_(0.4)
    score = evidence_score(x)
    assert torch.isfinite(score).all()
    assert torch.count_nonzero(score) == 0
    gate(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(gate.alpha_raw.grad)


def test_bad_options_fail_fast():
    for bad in (
        {"mode": "unknown"}, {"mode": "none", "radius": 2},
        {"mode": "btrm", "radius": 0},
        {"mode": "btrm", "saturation": float("nan")},
        {"mode": "btrm", "eps": 0},
    ):
        with pytest.raises(ValueError):
            validate_feature_module(bad)


def test_ablation_shares_training_and_architecture():
    runs = load_comparison("configs/experiment/ablation_btrm.yaml")
    assert len(runs) == 3
    assert [run["name"] for run in runs] == ["btrm_a0", "btrm_a2", "btrm_a3"]
    assert [run["model"]["feature_module"]["mode"] for run in runs] == [
        "none", "contrast", "btrm"
    ]
    assert len({run["model"]["config"] for run in runs}) == 1
    assert runs[0]["train"]["epochs"] == 50
    assert all(run["train"] == runs[0]["train"] for run in runs)
    assert all(run["seed"] == 42 for run in runs)


def test_ultralytics_p3_only_gate_and_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    path = "configs/model/yolo26n-btrm.yaml"
    model = YOLO(path).model
    assert isinstance(model.model[GATE_INDEX], nn.Identity)
    assert model.model[18].f == 16  # P4 still receives raw P3
    assert model.model[24].f == [17, 20, 23]
    unmodified = copy.deepcopy(model).eval()
    install_feature_module(model, {"mode": "btrm"})
    assert isinstance(model.model[GATE_INDEX], LocalBackgroundEvidence)
    assert model.model[GATE_INDEX].i == 17
    assert model.model[GATE_INDEX].f == -1
    a = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        # Recursive nested prediction outputs are not guaranteed by Ultralytics;
        # compare the single feature at the gate instead.
        feature = torch.randn(1, 64, 8, 8)
        assert torch.equal(model.model[GATE_INDEX](feature), feature)
        _ = model(a)
        _ = unmodified(a)
    target = tmp_path / "module.pt"
    torch.save(model, target)
    loaded = torch.load(target, weights_only=False)
    assert isinstance(loaded.model[GATE_INDEX], LocalBackgroundEvidence)
    assert loaded.model[GATE_INDEX].mode == "btrm"
    assert loaded.model[24].f == [17, 20, 23]


def test_size_polygon_parsing():
    pytest.importorskip("ultralytics")
    from tinydet.evaluation.size_metrics import parse_annotation_box

    assert parse_annotation_box("0 0.5 0.5 0.2 0.4") == pytest.approx(
        [.4, .3, .6, .7]
    )
    assert parse_annotation_box("0 0.1 0.2 0.2 0.2 0.2 0.4 0.1 0.4") == [
        .1, .2, .2, .4
    ]
    with pytest.raises(ValueError):
        parse_annotation_box("0 0.1 0.2 0.3 0.4 0.5")


def test_report_delta_is_proposed_minus_contrast():
    pytest.importorskip("ultralytics")
    from tinydet.evaluation.btrm_report import _delta

    assert _delta({"AP50": .3, "latency_ms": 12}, {
        "AP50": .2, "latency_ms": 10
    }) == pytest.approx({"AP50": .1, "latency_ms": 2})


def test_size_prediction_manifest_uses_one_image_per_batch(tmp_path, monkeypatch):
    """Regression: Python list paths produce one giant 322-image batch."""
    from PIL import Image
    from ultralytics.data.build import load_inference_source

    from tinydet.evaluation import size_metrics

    image_paths = []
    for i in range(3):
        path = tmp_path / f"sample_{i}.jpg"
        Image.new("RGB", (64, 64), (50, 90, 130)).save(path)
        image_paths.append(path)
    monkeypatch.setattr(size_metrics, "dataset_images", lambda *_: image_paths)

    calls = []

    class FakeBoxes:
        xyxy = torch.empty((0, 4), dtype=torch.float32)
        conf = torch.empty(0, dtype=torch.float32)
        cls = torch.empty(0, dtype=torch.float32)

    class FakeYOLO:
        def __init__(self, checkpoint):
            assert checkpoint == "fake-checkpoint.pt"

        def predict(self, *, source, batch, imgsz, stream, **kwargs):
            assert isinstance(source, str)  # a .txt path, never an image list
            assert source.endswith(".txt")
            assert batch == 1
            assert imgsz == 1280 and stream is True
            paths = [Path(line) for line in Path(source).read_text().splitlines()]
            assert paths == image_paths
            loader = load_inference_source(source, batch=batch)
            assert type(loader).__name__ == "LoadImagesAndVideos"
            assert loader.bs == 1
            assert loader.nf == 3
            assert len(list(loader)) == 3
            calls.append(source)
            for path in paths:
                yield type("Result", (), {
                    "path": str(path), "orig_shape": (64, 64), "boxes": FakeBoxes()
                })()

    monkeypatch.setattr(size_metrics, "YOLO", FakeYOLO)
    artifact = tmp_path / "size" / "predictions.pt"
    result = size_metrics.collect_size_predictions(
        Path("fake-checkpoint.pt"), tmp_path / "data.yaml", artifact, imgsz=1280
    )
    assert result == artifact
    assert len(calls) == 1
    payload = torch.load(artifact, weights_only=True, map_location="cpu")
    assert len(payload["predictions"]) == 3
    assert payload["imgsz"] == 1280
