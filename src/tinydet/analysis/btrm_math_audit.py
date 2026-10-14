"""Independent float64 verification of the original BTRM implementation.

The oracle uses explicit NumPy least squares and numpy.linalg.solve. It does
NOT call the implementation's internal 2x2 inverse or its spatial operations.
All comparisons use EXACT float32 input values (as fed to the module), so
differences measure implementation error rather than input quantization.
"""
import math
from datetime import UTC, datetime

import numpy as np
import torch

from tinydet.modules.btrm import OFFSETS, LocalBackgroundEvidence, evidence_score


def oracle_components(x: np.ndarray, row: int, col: int, mode: str,
                      radius: int = 1, ridge_relative: float = 0.05,
                      eps: float = 1e-3) -> dict:
    """Compute one location independently using float64 least squares."""
    if x.ndim != 3 or min(x.shape[1:]) <= 2 * radius:
        raise ValueError("Oracle expects CHW with interior pixels")
    if not (radius <= row < x.shape[1] - radius and
            radius <= col < x.shape[2] - radius):
        raise ValueError("Oracle only evaluates interior positions")
    a = np.asarray(x, dtype=np.float64)
    neighbors = np.array(
        [a[:, row + dy * radius, col + dx * radius] for dy, dx in OFFSETS]
    )
    xy = np.asarray([[dx * radius, dy * radius] for dy, dx in OFFSETS], dtype=np.float64)
    mu = neighbors.mean(axis=0)
    # Separate independent solver (not the optimized sums in btrm.py).
    coefficients = np.linalg.lstsq(xy, neighbors - mu[None, :], rcond=None)[0]
    G = coefficients.T
    center_residual = a[:, row, col] - mu
    gram = G.T @ G
    lam = ridge_relative * np.trace(gram) + eps * eps
    if mode == "btrm":
        q = np.linalg.solve(gram + lam * np.eye(2), G.T @ center_residual)
        residual = center_residual - G @ q
    elif mode == "contrast":
        residual = center_residual.copy()
    else:
        raise ValueError(mode)
    fit = mu[None, :] + xy @ coefficients
    sigma = math.sqrt(float(np.square(neighbors - fit).sum() / neighbors.size + eps * eps))
    magnitude = (math.sqrt(float(residual @ residual + eps * eps)) - eps) / math.sqrt(len(mu))
    return {
        "score": magnitude / sigma,
        "sigma": sigma,
        "mu_norm": float(np.linalg.norm(mu)),
        "center_residual_norm": float(np.linalg.norm(center_residual)),
        "projected_norm": float(np.linalg.norm(residual)),
        "lambda": lam,
        "rank": int(np.linalg.matrix_rank(G)),
        "contraction": float(np.linalg.norm(residual) /
                             max(np.linalg.norm(center_residual), 1e-15)),
    }


def _fixture(name: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(20261014)
    if name == "random":
        return torch.randn((1, 7, 13, 15), generator=g)
    if name == "flat":
        return torch.full((1, 7, 13, 15), 3.25)
    yy, xx = torch.meshgrid(torch.arange(13), torch.arange(15), indexing="ij")
    if name == "affine":
        return torch.stack(
            [0.2 * i + (i + 1) * 0.08 * xx + (i - 2) * 0.06 * yy
             for i in range(7)]
        ).unsqueeze(0).float()
    if name == "near_rank_one":
        return torch.stack(
            [100 + i + (i + 1) * 0.04 * xx + 1e-5 * yy
             for i in range(7)]
        ).unsqueeze(0).float()
    if name == "spike":
        x = torch.randn((1, 7, 13, 15), generator=g) * 0.03 + 15
        x[..., 6, 7] += 5
        return x
    if name == "low_noise":
        return 0.01 * torch.randn((1, 7, 13, 15), generator=g) + 2.0
    raise ValueError(name)


def run_math_audit() -> dict:
    """Fail closed on any math/numeric discrepancy before GPU diagnostics."""
    comparisons = []
    for kind in ("random", "flat", "affine", "near_rank_one", "spike", "low_noise"):
        x = _fixture(kind)
        for radius in (1, 2):
            sites = ((5, 6), (6, 7), (8, 9))
            for mode in ("contrast", "btrm"):
                result = evidence_score(x, mode, radius)
                if not torch.isfinite(result).all():
                    raise AssertionError(f"Nonfinite module score: {kind}/{mode}")
                for row, col in sites:
                    ref = oracle_components(
                        x[0].numpy(), row, col, mode, radius
                    )
                    actual = float(result[0, 0, row, col])
                    expected = ref["score"]
                    absolute_error = abs(actual - expected)
                    allowed = 0.005 + 0.005 * abs(expected)
                    if absolute_error > allowed:
                        raise AssertionError(
                            f"Math mismatch {kind}/{mode}/r{radius}@{row},{col}: "
                            f"optimized={actual:.10g}, float64={expected:.10g}, "
                            f"abs={absolute_error:.4g}, tolerance={allowed:.4g}"
                        )
                    if ref["contraction"] > 1 + 1e-9:
                        raise AssertionError("Ridge residual increased norm")
                    comparisons.append({
                        "case": kind, "mode": mode, "radius": radius,
                        "site": [row, col], "optimized": actual,
                        "float64_oracle": expected, "absolute_error": absolute_error,
                        "relative_error": absolute_error / max(abs(expected), 1e-9),
                        "tolerance": allowed, **ref,
                    })

    # Independently check exact identity, bounded learned gain, and derivative.
    gate_results = []
    for mode in ("contrast", "btrm"):
        m = LocalBackgroundEvidence(mode=mode)
        x = _fixture("random")
        if not torch.equal(m(x), x):
            raise AssertionError("Identity initialization failed")
        with torch.no_grad():
            m.alpha_raw.fill_(0.42)
        val = float(m.effective_gain.detach())
        if not (0 < val < 0.5):
            raise AssertionError("Unbounded effective gain")
        center = (6, 7)
        z = oracle_components(x[0].numpy(), *center, mode)["score"]
        expected_factor = 1 + val * math.tanh(z / m.saturation)
        got = m(x).detach()[0, :, *center].numpy()
        expected = x[0, :, *center].numpy().astype(np.float64) * expected_factor
        error = float(np.max(np.abs(got - expected)))
        if error > 2e-5:
            raise AssertionError(f"Gate mismatch {mode}: {error}")
        gate_results.append({"mode": mode, "effective_gain": val, "gate_max_abs_error": error})

    # Check gradients against central differences without relying on double
    # autograd through the module (the production function intentionally
    # computes FP32, including when given a float64 input).
    gradient_checks = []
    for mode in ("contrast", "btrm"):
        m = LocalBackgroundEvidence(mode=mode)
        with torch.no_grad():
            m.alpha_raw.fill_(0.42)
        x = _fixture("random").requires_grad_()
        value = m(x).square().mean()
        derivative = torch.autograd.grad(value, x)[0]
        for row, col in ((6, 7), (5, 6)):
            ch, step = 2, 0.005
            plus = x.detach().clone()
            minus = x.detach().clone()
            plus[0, ch, row, col] += step
            minus[0, ch, row, col] -= step
            numerical = float((m(plus).square().mean() -
                               m(minus).square().mean()) / (2 * step))
            analytic = float(derivative[0, ch, row, col])
            err = abs(numerical - analytic)
            tol = 0.002 + 0.05 * abs(analytic)
            if not math.isfinite(numerical) or not math.isfinite(analytic) or err > tol:
                raise AssertionError(f"Gradient {mode} mismatch {err} > {tol}")
            gradient_checks.append({
                "mode": mode, "site": [row, col], "autograd": analytic,
                "central_difference": numerical, "absolute_error": err,
            })
    return {
        "status": "PASS", "at": datetime.now(UTC).isoformat(),
        "implementation": "tinydet.modules.btrm.evidence_score (unchanged)",
        "oracle": "independent NumPy float64 least-squares + linear solve",
        "dtype": "float32 actual input; float64 independent arithmetic",
        "checks": len(comparisons), "max_abs_error": max(x["absolute_error"] for x in comparisons),
        "max_rel_error_nonzero": max(
            x["relative_error"] for x in comparisons if x["float64_oracle"] > 1e-5
        ),
        "cases": comparisons,
        "gate_checks": gate_results, "gradient_checks": gradient_checks,
        "limitations": [
            "A correct finite-precision implementation does not establish semantic correctness of the nuisance-background model.",
            "FP16/AMP behavior and real-image activation scales require separate empirical checks.",
            "The ridge contraction theorem requires fixed estimated G, not necessarily fixed G across perturbations of input.",
        ],
    }
