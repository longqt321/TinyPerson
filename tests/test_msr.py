"""Tests for the full MSR evidence, adaptive pooling and feature fusion system."""
import pytest
import torch

from tinydet.modules.msr import (
    DETECT_INDEX,
    FUSION_SPECS,
    MSRFusionGate,
    MultiScaleStandardizedResidual,
    install_msr_feature_module,
    validate_feature_module,
)
from tinydet.utils.comparison import load_comparison


@pytest.mark.parametrize("aggregation", ["max", "adaptive"])
def test_standardized_evidence_shape_stability_and_gradients(aggregation):
    module = MultiScaleStandardizedResidual(dilations=(1, 2, 4), aggregation=aggregation)
    x = torch.randn(2, 8, 19, 17, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    if aggregation == "adaptive":
        assert module.scale_bias.grad is not None
        assert torch.isfinite(module.scale_bias.grad).all()
    else:
        assert module.scale_bias is None


def test_standardized_evidence_is_zero_on_constant_features():
    module = MultiScaleStandardizedResidual(aggregation="adaptive")
    x = torch.full((2, 8, 17, 15), 2.5, requires_grad=True)
    y = module(x)
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(module.scale_bias.grad).all()


def test_standardized_evidence_detects_a_local_positive_outlier():
    module = MultiScaleStandardizedResidual(dilations=(1, 2, 4), aggregation="adaptive")
    x = torch.ones(1, 4, 25, 25)
    x[:, :, 12, 12] += 3.0
    response = module(x)
    assert response[0, 0, 12, 12] > 5
    assert response[0, 0, 12, 12] > response[0, 0, 12, 11]


@pytest.mark.parametrize("high_first", [True, False])
@pytest.mark.parametrize("mode", ["local_max", "local_adaptive", "full_max", "full_adaptive"])
def test_fusion_gate_shape_and_backward(high_first, mode):
    torch.manual_seed(7)
    gate = MSRFusionGate(
        high_channels=8, low_channels=4, high_first=high_first, mode=mode,
        dilations=(1, 2, 4),
    )
    high = torch.randn(2, 8, 19, 17, requires_grad=True)
    low = torch.randn(2, 4, 19, 17, requires_grad=True)
    x = torch.cat((high, low), dim=1) if high_first else torch.cat((low, high), dim=1)
    y = gate(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(high.grad).all()
    assert torch.isfinite(low.grad).all()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters())


def test_feature_module_validation():
    assert validate_feature_module(None) == {"mode": "none"}
    assert validate_feature_module({"mode": "none"}) == {"mode": "none"}
    for mode in ("local_max", "local_adaptive", "full_max", "full_adaptive"):
        options = validate_feature_module({"mode": mode, "dilations": [1, 2, 4]})
        assert options["mode"] == mode
        assert options["dilations"] == [1, 2, 4]
    for bad in (
        {"mode": "unknown"},
        {"mode": "none", "dilations": [1]},
        {"mode": "full_adaptive", "dilations": []},
        {"mode": "full_adaptive", "dilations": [1, 1]},
        {"mode": "full_adaptive", "dilations": [0]},
        {"mode": "full_adaptive", "eps": 0},
        {"mode": "full_adaptive", "score_clip": float("nan")},
        {"mode": "full_adaptive", "hidden_ratio": 0},
        {"mode": "full_adaptive", "unknown": True},
    ):
        with pytest.raises(ValueError):
            validate_feature_module(bad)


def test_ablation_settings_are_controlled():
    runs = load_comparison("configs/experiment/ablation_msr.yaml")
    assert [run["name"] for run in runs] == [
        "msr_a0_identity", "msr_a1_local_max", "msr_a2_full_max", "msr_a3_full_adaptive"
    ]
    assert len({run["model"]["config"] for run in runs}) == 1
    for key in ("data", "train", "hardware", "output", "seed"):
        assert all(run[key] == runs[0][key] for run in runs)
    assert all(run["train"]["epochs"] == 50 for run in runs)
    assert [run["model"]["feature_module"]["mode"] for run in runs] == [
        "none", "local_max", "full_max", "full_adaptive"
    ]
    assert all(run["train"]["imgsz"] == 1280 for run in runs)


def test_p2_architecture_and_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("ultralytics")
    from tinydet.models.registry import build_model

    options = {
        "mode": "full_adaptive",
        "dilations": [1, 2, 4],
        "eps": 1e-4,
        "score_clip": 10.0,
        "hidden_ratio": 4,
        "scale_temperature": 1.0,
    }
    model = build_model({
        "config": "configs/model/yolo26n-msr-p2.yaml",
        "pretrained": None,
        "feature_module": options,
    })
    net = model.model
    assert len(net.model) == DETECT_INDEX + 1
    assert net.model[DETECT_INDEX].f == [22, 26, 30, 34]
    for index, spec in FUSION_SPECS.items():
        gate = net.model[index]
        assert isinstance(gate, MSRFusionGate)
        assert (gate.high_channels, gate.low_channels, gate.high_first) == spec
        assert gate.i == index and gate.f == -1

    result = model.predict(torch.zeros(1, 3, 64, 64), device="cpu", verbose=False)
    assert len(result) == 1 and result[0].orig_shape == (64, 64)

    checkpoint = tmp_path / "msr_state.pt"
    torch.save(net.state_dict(), checkpoint)
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    reloaded = build_model({
        "config": "configs/model/yolo26n-msr-p2.yaml",
        "pretrained": None,
        "feature_module": options,
    })
    reloaded.model.load_state_dict(state)
    loaded_net = reloaded.model
    assert isinstance(loaded_net.model[21], MSRFusionGate)
    assert loaded_net.model[21].mode == "full_adaptive"
    assert loaded_net.model[DETECT_INDEX].f == [22, 26, 30, 34]


def test_install_rejects_already_replaced_gates():
    pytest.importorskip("ultralytics")
    from tinydet.models.registry import build_model

    model = build_model({
        "config": "configs/model/yolo26n-msr-p2.yaml",
        "pretrained": None,
        "feature_module": {"mode": "full_adaptive"},
    })
    with pytest.raises(ValueError, match="Identity fusion slot"):
        install_msr_feature_module(model.model, {"mode": "full_max"})
