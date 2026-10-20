"""Tests for the YAML-registered P2 MSR refinement block."""
import pytest
import torch

from tinydet.modules.msr import MSR, load_p2_msr_weights
from tinydet.utils.comparison import load_comparison


@pytest.mark.parametrize("shape", [(2, 8, 3, 5), (2, 8, 13, 11)])
def test_msr_refinement_shape_backward_and_optimizer_update(shape):
    module = MSR(8)
    x = torch.randn(*shape, requires_grad=True)
    before = module.gamma.detach().clone()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    loss = (module(x) - torch.randn_like(x)).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters())
    assert module.gamma.grad is not None and module.gamma.grad.abs() > 0
    for branch in (module.projections, module.scale_gates, module.feature_gate, module.refine):
        gradients = [p.grad for p in branch.parameters()]
        assert all(grad is not None for grad in gradients)
        assert sum(grad.abs().sum() for grad in gradients) > 0
    optimizer.step()
    assert not torch.equal(before, module.gamma.detach())


def test_msr_cpu_autocast_and_input_validation():
    module = MSR(8)
    x = torch.randn(2, 8, 13, 11)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = module(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    with pytest.raises(ValueError, match="BCHW"):
        module(torch.randn(2, 8, 13))
    with pytest.raises(ValueError, match="odd"):
        MSR(8, kernels=(3, 4))


def test_p2_msr_graph_loss_and_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("ultralytics")
    from ultralytics.utils import DEFAULT_CFG

    from tinydet.models.registry import build_model

    model = build_model({
        "config": "configs/model/yolo26n_tiny_A2_p2_msr.yaml",
    })
    net = model.model
    assert len(net.model) == 31
    assert isinstance(net.model[29], MSR)
    assert net.model[29].f == 19
    detect = net.model[30]
    assert detect.f == [29, 22, 25, 28]
    assert detect.nc == 1 and net.stride.tolist() == [4.0, 8.0, 16.0, 32.0]

    net.args = DEFAULT_CFG
    net.train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.15, 0.15], [0.4, 0.6, 0.1, 0.12]]),
    }
    loss, _ = net.loss(batch)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert torch.isfinite(net.model[29].gamma.grad)

    state_path = tmp_path / "msr.pt"
    torch.save(net.state_dict(), state_path)
    restored = build_model({
        "config": "configs/model/yolo26n_tiny_A2_p2_msr.yaml",
    })
    restored.model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    restored.model.eval()
    prediction = restored.predict(torch.zeros(1, 3, 64, 64), device="cpu", verbose=False)
    assert len(prediction) == 1 and prediction[0].orig_shape == (64, 64)


def test_p2_msr_layer_index_checkpoint_mapping():
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    from tinydet.models.registry import build_model

    baseline = DetectionModel("configs/model/yolo26n_tiny_A2_p2.yaml", ch=3, verbose=False)
    adapted = build_model({
        "config": "configs/model/yolo26n_tiny_A2_p2_msr.yaml",
    }).model
    report = load_p2_msr_weights(adapted, baseline)
    assert report["remapped"] > 0
    assert torch.equal(adapted.model[30].cv2[0][0].conv.weight,
                       baseline.model[29].cv2[0][0].conv.weight)


def test_p2_msr_comparison_uses_paired_architectures():
    runs = load_comparison("configs/experiment/ablation_msr_p2.yaml")
    assert [run["name"] for run in runs] == [
        "msr_p2_e0_a2_baseline", "msr_p2_e1_a2_msr",
        "msr_p2_e2_a3_no_sppf", "msr_p2_e3_a3_no_sppf_msr",
    ]
    assert runs[0]["model"]["config"] == "configs/model/yolo26n_tiny_A2_p2.yaml"
    assert runs[1]["model"]["config"] == "configs/model/yolo26n_tiny_A2_p2_msr.yaml"
    assert runs[1]["model"].keys() == {"config", "pretrained"}
    assert runs[2]["model"]["config"] == "configs/model/yolo26n_tiny_A3_p2_no_sppf.yaml"
    assert runs[3]["model"]["config"] == "configs/model/yolo26n_tiny_A3_p2_no_sppf_msr.yaml"
    for key in ("data", "train", "hardware", "output", "seed"):
        assert all(run[key] == runs[0][key] for run in runs)
