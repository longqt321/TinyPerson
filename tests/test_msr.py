"""Unit and wiring tests for the multi-scale standardized residual module."""
import pytest
import torch

from tinydet.models.registry import build_model
from tinydet.modules.msr import (
    MSR_GATE_INDEX,
    MultiScaleStandardizedResidual,
    install_msr_feature_module,
    validate_feature_module,
)
from tinydet.utils.comparison import load_comparison


@pytest.mark.parametrize("mode", ["raw", "standardized"])
def test_identity_shape_and_gradients(mode):
    module = MultiScaleStandardizedResidual(mode=mode, dilations=(1, 2, 4))
    x = torch.randn(2, 5, 19, 17, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.equal(y, x)

    with torch.no_grad():
        module.gain_raw.fill_(0.4)
    (module(x).square().mean()).backward()
    assert torch.isfinite(x.grad).all()
    assert module.gain_raw.grad is not None
    assert torch.isfinite(module.gain_raw.grad)


def test_standardized_response_detects_local_positive_outlier():
    module = MultiScaleStandardizedResidual(mode="standardized", dilations=(1, 2, 4))
    x = torch.ones(1, 2, 17, 17)
    x[:, :, 8, 8] += 3.0
    with torch.no_grad():
        module.gain_raw.fill_(0.5)
    y = module(x)
    center_delta = (y - x)[0, 0, 8, 8]
    surrounding_delta = (y - x)[0, 0, 8, 7]
    assert center_delta > 0
    assert center_delta > surrounding_delta


def test_constant_features_are_finite_and_unchanged_at_initialization():
    module = MultiScaleStandardizedResidual(mode="standardized")
    x = torch.full((2, 4, 13, 15), 2.5, requires_grad=True)
    y = module(x)
    assert torch.equal(y, x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(module.gain_raw.grad)


def test_feature_module_validation():
    assert validate_feature_module(None) == {"mode": "none"}
    assert validate_feature_module({"mode": "none"}) == {"mode": "none"}
    assert validate_feature_module({"mode": "standardized", "dilations": [1, 2]}) == {
        "mode": "standardized",
        "dilations": [1, 2],
        "eps": 1e-4,
        "max_gain": 0.1,
        "score_clip": 10.0,
    }
    for bad in (
        {"mode": "unknown"},
        {"mode": "none", "dilations": [1]},
        {"mode": "standardized", "dilations": []},
        {"mode": "standardized", "dilations": [1, 1]},
        {"mode": "standardized", "dilations": [0]},
        {"mode": "standardized", "eps": 0},
        {"mode": "standardized", "max_gain": 0.7},
        {"mode": "standardized", "score_clip": float("nan")},
        {"mode": "standardized", "unknown": True},
    ):
        with pytest.raises(ValueError):
            validate_feature_module(bad)


def test_ablation_settings_are_controlled():
    runs = load_comparison("configs/experiment/ablation_msr.yaml")
    assert [run["name"] for run in runs] == [
        "msr_a0_identity",
        "msr_a1_raw_multiscale",
        "msr_a2_standardized_single",
        "msr_a3_standardized_multiscale",
    ]
    assert len({run["model"]["config"] for run in runs}) == 1
    assert all(run["model"]["feature_module"] for run in runs)
    for key in ("data", "train", "hardware", "output", "seed"):
        assert all(run[key] == runs[0][key] for run in runs)
    assert all(run["train"]["epochs"] == 50 for run in runs)
    assert [run["model"]["feature_module"]["mode"] for run in runs] == [
        "none", "raw", "standardized", "standardized"
    ]
    assert runs[2]["model"]["feature_module"]["dilations"] == [1]
    assert runs[3]["model"]["feature_module"]["dilations"] == [1, 2, 4]


def test_p2_model_gate_routing_prediction_and_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("ultralytics")
    options = {
        "mode": "standardized",
        "dilations": [1, 2, 4],
        "eps": 1e-4,
        "max_gain": 0.1,
        "score_clip": 10.0,
    }
    model = build_model({
        "config": "configs/model/yolo26n-msr-p2.yaml",
        "pretrained": None,
        "feature_module": options,
    })
    net = model.model
    assert isinstance(net.model[MSR_GATE_INDEX], MultiScaleStandardizedResidual)
    assert net.model[MSR_GATE_INDEX].i == MSR_GATE_INDEX
    assert net.model[MSR_GATE_INDEX].f == -1
    assert net.model[21].f == 19
    assert net.model[30].f == [20, 23, 26, 29]
    assert net.stride.tolist() == [4.0, 8.0, 16.0, 32.0]

    result = model.predict(torch.zeros(1, 3, 64, 64), device="cpu", verbose=False)
    assert len(result) == 1 and result[0].orig_shape == (64, 64)

    checkpoint = tmp_path / "msr_model.pt"
    torch.save(net, checkpoint)
    loaded = torch.load(checkpoint, weights_only=False)
    assert isinstance(loaded.model[MSR_GATE_INDEX], MultiScaleStandardizedResidual)
    assert loaded.model[MSR_GATE_INDEX].dilations == (1, 2, 4)
    assert loaded.model[30].f == [20, 23, 26, 29]


def test_install_requires_the_vetted_p2_architecture():
    model = build_model({"config": "configs/model/yolo26n-msr-p2.yaml", "pretrained": None})
    install_msr_feature_module(model.model, {"mode": "standardized"})
    with pytest.raises(ValueError, match="Identity gate"):
        install_msr_feature_module(model.model, {"mode": "standardized"})
