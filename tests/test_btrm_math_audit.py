"""Mathematical validity gate required before interpreting BTRM performance."""
import numpy as np
import pytest
import torch

from tinydet.analysis.btrm_math_audit import oracle_components, run_math_audit
from tinydet.modules.btrm import evidence_score


def test_independent_float64_oracle_and_gradient_audit():
    report = run_math_audit()
    assert report["status"] == "PASS"
    assert report["checks"] == 72
    assert report["max_abs_error"] < 0.1
    assert len(report["gradient_checks"]) == 4
    assert len(report["gate_checks"]) == 2


def test_oracle_uses_real_two_dimensional_background_variation():
    x = np.zeros((3, 9, 9))
    for y in range(9):
        for z in range(9):
            x[:, y, z] = np.array([1.2, -2.0, 3.1]) + y * np.array([
                0.5, 0.1, -0.2
            ]) + z * np.array([0.2, -0.3, 0.1])
    value = oracle_components(x, 4, 4, "btrm")
    assert value["rank"] == 2
    assert value["score"] == pytest.approx(0, abs=1e-9)
    assert value["projected_norm"] == pytest.approx(0, abs=1e-9)


def test_implementation_uses_fp32_even_for_fp64_input():
    x = torch.randn((1, 4, 9, 9), dtype=torch.float64)
    assert evidence_score(x).dtype == torch.float32


def test_oracle_has_no_center_contamination_of_ring():
    x = np.ones((5, 11, 11), dtype=np.float64)
    baseline = oracle_components(x, 5, 5, "btrm")
    x[:, 5, 5] = 10
    spike = oracle_components(x, 5, 5, "btrm")
    assert baseline["sigma"] == pytest.approx(spike["sigma"])
    assert baseline["mu_norm"] == pytest.approx(spike["mu_norm"])
    assert spike["score"] > 100


def test_projection_cannot_increase_evidence_for_same_input():
    x = torch.randn(3, 32, 14, 15)
    for radius in (1, 2):
        a = evidence_score(x, "contrast", radius=radius)
        b = evidence_score(x, "btrm", radius=radius)
        assert torch.all(b <= a + 1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_feature_input_uses_identical_fp32_internal_score(dtype):
    """CPU emulation of AMP inputs; module computes scores in FP32."""
    from tinydet.modules.btrm import LocalBackgroundEvidence

    x = torch.randn(1, 16, 13, 13).to(dtype)
    actual_score = evidence_score(x, "btrm")
    expected_score = evidence_score(x.float(), "btrm")
    assert actual_score.dtype == torch.float32
    assert torch.equal(actual_score, expected_score)
    gate = LocalBackgroundEvidence("btrm")
    with torch.no_grad():
        gate.alpha_raw.fill_(.35)
    inp = x.detach().clone().requires_grad_(True)
    output = gate(inp)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert torch.isfinite(inp.grad).all()
    assert torch.isfinite(gate.alpha_raw.grad)
