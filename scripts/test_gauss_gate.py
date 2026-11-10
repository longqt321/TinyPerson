"""Unit test for the GauSS gate callback: fake trainer, real diagnostics keys."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinydet.training.gauss_gate import GaussGateCallback


def make_trainer(tmp: Path, epoch: int, diag: dict, map50: float | None):
    return SimpleNamespace(
        epoch=epoch - 1,
        model=SimpleNamespace(),
        metrics={"metrics/mAP50(B)": map50} if map50 is not None else {},
        loss_items=[3.0, 2.0],
        stop=False,
        save_dir=tmp / "training" / "run",
    )


def diag(sigma2_std: float, alpha: float, share: float) -> dict:
    return {
        "gaussian/sigma2_std": sigma2_std,
        "gaussian/sigma2_mean": 1.45,
        "gaussian/sigma1_mean": 0.65,
        "gates/alpha": alpha,
        "contribution/alpha_z_share": share * 0.6,
        "contribution/anchor_share": share * 0.3,
        "contribution/mid_anchor_share": share * 0.1,
    }


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cb = GaussGateCallback()  # defaults = registered thresholds

        # healthy run at e10 and e20 -> no kill
        m = make_trainer(tmp, 10, diag(0.02, 0.13, 0.05), 0.10)
        cb.modules = [("m", SimpleNamespace(last_diag=diag(0.02, 0.13, 0.05)))]
        cb.on_fit_epoch_end(m)
        assert m.stop is not True, "healthy e10 must not kill"
        cb.modules[0][1].last_diag = diag(0.03, 0.16, 0.06)
        m20 = make_trainer(tmp, 20, diag(0.03, 0.16, 0.06), 0.12)
        cb.on_fit_epoch_end(m20)
        assert m20.stop is not True, "healthy e20 must not kill"

        # dead mechanism at e20 -> kill via sigma2_std
        cb2 = GaussGateCallback()
        m2 = make_trainer(tmp, 20, diag(0.008, 0.16, 0.06), 0.12)
        cb2.modules = [("m", SimpleNamespace(last_diag=diag(0.008, 0.16, 0.06)))]
        cb2.on_fit_epoch_end(m2)
        assert m2.stop is True, "sigma2_std below floor must kill"
        assert cb2.verdicts[-1]["checks"]["g_sigma2_std"] is False

        # residual share ~0 (the T3 finding) -> kill even with sigma moving
        cb3 = GaussGateCallback()
        m3 = make_trainer(tmp, 20, diag(0.05, 0.20, 0.001), 0.12)
        cb3.modules = [("m", SimpleNamespace(last_diag=diag(0.05, 0.20, 0.001)))]
        cb3.on_fit_epoch_end(m3)
        assert m3.stop is True, "zero residual share must kill"

        # val collapse -> kill
        cb4 = GaussGateCallback()
        m4 = make_trainer(tmp, 20, diag(0.03, 0.16, 0.06), 0.03)
        cb4.modules = [("m", SimpleNamespace(last_diag=diag(0.03, 0.16, 0.06)))]
        cb4.on_fit_epoch_end(m4)
        assert m4.stop is True, "val collapse must kill"

        # NaN diagnostics -> instant kill at any epoch
        cb5 = GaussGateCallback()
        bad = diag(0.03, 0.16, 0.06)
        bad["gates/alpha"] = float("nan")
        m5 = make_trainer(tmp, 5, bad, None)
        cb5.modules = [("m", SimpleNamespace(last_diag=bad))]
        cb5.on_fit_epoch_end(m5)
        assert m5.stop is True, "non-finite values must kill"

        # same layout as gauss_diagnostics.csv: <save_dir>.parent/metrics
        out = tmp / "training" / "metrics" / "gauss_gate_verdict.json"
        assert out.is_file(), "verdict JSON must be written"
        json.loads(out.read_text())

    print("[PASS] gauss_gate: 5/5 scenarios (healthy pass, 3 kill rules, NaN kill)")


if __name__ == "__main__":
    main()
