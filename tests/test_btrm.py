"""Unit tests for BTRM mathematics, routing, serialization and experiment fairness."""
import copy

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
