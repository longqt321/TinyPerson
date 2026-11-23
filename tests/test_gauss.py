"""Unit tests for GauSS v3.2 (extracted from ``src/tinydet/modules/gauss.py``)."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from tinydet.modules.gauss import (
    GauSS,
    LearnableGaussianScaleSpace,
    collect_gauss_control_params,
    collect_gauss_router_zloss,
    compute_gauss_analytical_macs,
    configure_gauss_control_param_group,
    remap_v3_to_v31_state_dict,
    remap_v31_to_v32_state_dict,
)


def test_gauss_v32_contract_and_mechanics() -> None:
    torch.manual_seed(42)

    module = GauSS(c1=[64, 64, 128], c2=32)

    # 1. Static contract
    total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    conv_count = sum(1 for m in module.modules() if isinstance(m, nn.Conv2d))
    ctrl_params = collect_gauss_control_params(module)
    ctrl_count = sum(p.numel() for p in ctrl_params)
    assert total_params == 36_879, f"Expected 36,879 params, got {total_params}"
    assert conv_count == 27, f"Expected 27 Conv2d modules, got {conv_count}"
    assert ctrl_count == 132, f"Expected 132 control scalars, got {ctrl_count}"

    # 2. Analytical MACs
    assert int(compute_gauss_analytical_macs()["total_macs"]) == 1_041_326_080
    assert int(compute_gauss_analytical_macs(learned_router=True)["total_macs"]) == 1_049_518_080
    assert int(compute_gauss_analytical_macs(router_correction=False)["total_macs"]) == 1_038_868_480

    # 3. DySample zero-init == bilinear reference
    xt = torch.randn(2, 32, 13, 17)
    ref_up = F.interpolate(xt, scale_factor=2, mode="bilinear", align_corners=False)
    got_up = module.dy_up3(xt)
    assert got_up.shape == (2, 32, 26, 34)
    assert (got_up - ref_up).abs().max().item() < 1e-5

    # 4. Warm start: near-identity output, live sigma gradient
    c2_t = torch.randn(2, 64, 80, 80, requires_grad=True)
    p3_t = torch.randn(2, 64, 40, 40, requires_grad=True)
    p4_t = torch.randn(2, 128, 20, 20, requires_grad=True)

    with torch.no_grad():
        x0_ref = module.act(
            module.bn2(module.proj_c2(c2_t) + module.dy_up3(module.proj_p3(p3_t)))
        )
    out_p2 = module([c2_t, p3_t, p4_t])
    assert out_p2.shape == (2, 32, 80, 80)
    rel = (out_p2 - x0_ref).norm() / x0_ref.norm().clamp_min(1e-12)
    assert 1e-3 < float(rel.detach()) < 0.35

    module.zero_grad(set_to_none=True)
    out_p2 = module([c2_t, p3_t, p4_t])
    out_p2.square().mean().backward()
    sigma_grads = [
        float(p.grad.abs().max())
        for n, p in module.named_parameters()
        if n.startswith("gaussian.sigma")
    ]
    assert max(sigma_grads) > 1e-8

    with torch.no_grad():
        base = module([c2_t, p3_t, p4_t]).clone()
        module.gaussian.sigma1x_raw.add_(0.4)
        module.gaussian.sigma1y_raw.add_(0.4)
        pert = module([c2_t, p3_t, p4_t])
        sens = float((pert - base).abs().max())
        module.gaussian.sigma1x_raw.sub_(0.4)
        module.gaussian.sigma1y_raw.sub_(0.4)
    assert sens > 1e-3

    # 5. Router correction z-loss & band weights
    assert abs(float(module._z_loss.item()) - (math.log(3.0) ** 2)) < 1e-6
    diag0 = module.last_diag
    assert (
        abs(
            diag0["routing/high_band_mean"]
            + diag0["routing/mid_band_mean"]
            + diag0["routing/low_band_mean"]
            - 1.0
        )
        < 1e-5
    )
    assert torch.allclose(collect_gauss_router_zloss(module), module.router_z_loss())

    # 6. Full gradient coverage except legacy unused params
    module.zero_grad(set_to_none=True)
    out_p2 = module([c2_t, p3_t, p4_t])
    out_p2.square().mean().backward()
    expected_unused = {
        "route_dw.weight",
        "route_bn.weight",
        "route_bn.bias",
        "route_out.weight",
        "route_out.bias",
        "saliency_gate.weight",
        "saliency_gate.bias",
    }
    for name, p in module.named_parameters():
        if name in expected_unused:
            assert p.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert float(p.grad.abs().max().item()) > 0.0
    assert set(module.unused_param_names()) == expected_unused

    # 7. Optimizer control group
    opt = torch.optim.AdamW(
        [{"params": list(module.parameters()), "weight_decay": 0.01, "param_group": "weight"}],
        lr=1e-3,
    )
    configure_gauss_control_param_group(opt, module, base_lr=1e-3)
    diag = module.get_diagnostics(optimizer=opt)
    assert diag["optimizer/control_weight_decay"] == 0.0
    assert diag["optimizer/control_lr"] == 1e-3
    assert diag["routing/mixing_std"] > 0.0
    assert diag["gates/alpha"] > 0.0
    configure_gauss_control_param_group(opt, module, base_lr=1e-3, lr_mult=3.0)
    assert module.get_diagnostics(optimizer=opt)["optimizer/control_lr"] == 3e-3

    # 8. Ablation flags
    os.environ["TINYDET_ABLATION"] = "no_high_band"
    try:
        m_abl = GauSS(c1=[64, 64, 128], c2=32).eval()
        with torch.no_grad():
            m_abl.gamma_out_raw.fill_(2.0)
            y_a = m_abl([c2_t.detach(), p3_t.detach(), p4_t.detach()])
            m_abl.gamma_out_raw.fill_(-100.0)
            y_b = m_abl([c2_t.detach(), p3_t.detach(), p4_t.detach()])
        assert (y_a - y_b).abs().max().item() < 1e-6
    finally:
        os.environ.pop("TINYDET_ABLATION", None)

    os.environ["TINYDET_ABLATION"] = "no_router,no_high_band"
    try:
        m_nr = GauSS(c1=[64, 64, 128], c2=32).eval()
        with torch.no_grad():
            m_nr([c2_t.detach(), p3_t.detach(), p4_t.detach()])
        d_nr = m_nr.last_diag
        w_sum = (
            d_nr["routing/high_band_mean"]
            + d_nr["routing/mid_band_mean"]
            + d_nr["routing/low_band_mean"]
        )
        assert abs(w_sum - 1.0) < 1e-6
        assert d_nr["routing/high_band_mean"] == 0.0
    finally:
        os.environ.pop("TINYDET_ABLATION", None)

    os.environ["TINYDET_ABLATION"] = "learned_router,use_saliency"
    try:
        m_leg = GauSS(c1=[64, 64, 128], c2=32)
        with torch.no_grad():
            m_leg.alpha_raw.fill_(1.0)
        out_leg = m_leg([c2_t.detach(), p3_t.detach(), p4_t.detach()])
        out_leg.square().mean().backward()
        for name, p in m_leg.named_parameters():
            if name in ("router_corr.weight", "router_corr.bias"):
                assert p.grad is None
            else:
                assert p.grad is not None and torch.isfinite(p.grad).all()
                assert float(p.grad.abs().max().item()) > 0.0
        assert m_leg.unused_param_names() == ("router_corr.weight", "router_corr.bias")
    finally:
        os.environ.pop("TINYDET_ABLATION", None)


def test_gauss_checkpoint_remaps() -> None:
    legacy = {
        "alpha_raw": torch.tensor([-6.0]),
        "gamma_h_raw": torch.tensor([-5.0]),
        "model.12.gauss.gamma_h_raw": torch.tensor([-3.0]),
    }
    remapped = remap_v3_to_v31_state_dict(legacy)
    assert torch.allclose(
        0.5 * torch.tanh(remapped["alpha_raw"]), 0.5 * torch.sigmoid(torch.tensor(-6.0)), atol=1e-7
    )
    assert torch.allclose(
        0.5 * torch.tanh(remapped["gamma_in_raw"]),
        0.5 * torch.sigmoid(torch.tensor(-5.0)),
        atol=1e-7,
    )
    assert torch.allclose(
        0.5 * torch.tanh(remapped["gamma_out_raw"]),
        0.5 * torch.sigmoid(torch.tensor(-5.0)),
        atol=1e-7,
    )
    assert "gamma_h_raw" not in remapped

    sd31 = {
        "gaussian.sigma1x_raw": torch.tensor([0.8, -0.4]),
        "gaussian.sigma2y_raw": torch.tensor([1.2]),
        "alpha_raw": torch.tensor([0.3616]),
        "conv.weight": torch.tensor([1.0]),
    }
    sd32 = remap_v31_to_v32_state_dict(sd31)
    assert torch.allclose(sd32["gaussian.sigma1x_raw"], sd31["gaussian.sigma1x_raw"] / 2)
    assert torch.allclose(sd32["gaussian.sigma2y_raw"], sd31["gaussian.sigma2y_raw"] / 2)
    assert torch.equal(sd32["alpha_raw"], sd31["alpha_raw"])

    m32 = LearnableGaussianScaleSpace(2)
    with torch.no_grad():
        m32.sigma1x_raw.copy_(sd32["gaussian.sigma1x_raw"])
        m32.sigma2y_raw.copy_(sd32["gaussian.sigma2y_raw"])
    s32 = m32.compute_sigmas()
    assert torch.allclose(s32[0], 0.4 + 0.5 * torch.sigmoid(sd31["gaussian.sigma1x_raw"]), atol=1e-6)
    assert torch.allclose(s32[3], 1.1 + 0.7 * torch.sigmoid(sd31["gaussian.sigma2y_raw"]), atol=1e-6)
