"""Pre-registered mechanism gates for GauSS v3.2 training runs (ADR-023).

The criteria document (artifact CRITERIA_V32_PREREG) registers thresholds
that decide, at epoch 10 and 20, whether the v3.2 mechanism is alive
without paying for the full 150-epoch run. This callback implements those
gates mechanically: at each configured checkpoint epoch it reads the GauSS
diagnostics, evaluates the rules, writes a verdict JSON next to the run
metrics, and sets ``trainer.stop = True`` (the same abort path Ultralytics
uses for time-based stops) when a kill rule fires.

Gates (v2 of the criteria — see the audit notes in the artifact):
  g_sigma2_std      : sigma2 per-channel spread must exceed the v3.1 e20
                      value by a margin (mechanism is differentiating).
  g_sigma2_mean     : OBSERVATION ONLY. The paired-delta probes predict the
                      population sigma mean stays at 1.45; a hard gate here
                      would kill healthy runs, so it is recorded, not enforced.
  g_alpha           : warm-started residual gate must keep growing.
  g_residual_share  : the module must actually contribute to the output
                      (alpha_z_share + anchor_share + mid_anchor_share).
  g_loss_finite     : hard NaN/Inf check every epoch (AMP safety with the
                      warm-started residual path).
  g_val_floor       : val mAP50 must not collapse below the floor.

Config (train.gauss_gate, all optional):
    epochs: [10, 20]          # checkpoint epochs (default)
    sigma2_std_min: 0.02      # kill below (v3.1 e20 = 0.011)
    alpha_min: 0.15           # kill below (v3.1 e20 = 0.097, v3.2 init 0.10)
    residual_share_min: 0.02  # kill below (T3: whole residual ~0 of loss)
    val_map50_floor: 0.06     # kill below (v3.1 e20 = 0.102; floor = "not collapsed")
    early_epoch: 10           # early stop epoch
    early_sigma2_std_min: 0.01
    early_alpha_min: 0.12
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tinydet.modules.gauss import GauSS
from tinydet.utils.atomic import atomic_write_text

DEFAULTS: dict = {
    "epochs": [10, 20],
    "sigma2_std_min": 0.02,
    "alpha_min": 0.15,
    "residual_share_min": 0.02,
    "val_map50_floor": 0.06,
    "early_epoch": 10,
    "early_sigma2_std_min": 0.01,
    "early_alpha_min": 0.12,
}


class GaussGateCallback:
    """Evaluate the pre-registered v3.2 mechanism gates at fixed epochs."""

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.modules: list[tuple[str, GauSS]] = []
        self.verdicts: list[dict] = []

    def on_train_start(self, trainer) -> None:
        model = getattr(trainer, "model", None)
        if model is None:
            return
        self.modules = [(n, m) for n, m in model.named_modules() if isinstance(m, GauSS)]
        if not self.modules:
            print("[gauss-gate] no GauSS module found; gate disabled", flush=True)

    def on_fit_epoch_end(self, trainer) -> None:
        if not self.modules:
            return
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        self._check_finite(trainer, epoch)
        if epoch not in self.cfg["epochs"]:
            return

        diag = self.modules[0][1].last_diag
        val_map50 = self._val_map50(trainer)
        early = epoch == int(self.cfg["early_epoch"])
        checks = {
            "g_sigma2_std": (
                float(diag.get("gaussian/sigma2_std", 0.0))
                >= (self.cfg["early_sigma2_std_min"] if early else self.cfg["sigma2_std_min"])
            ),
            "g_alpha": (
                float(diag.get("gates/alpha", 0.0))
                >= (self.cfg["early_alpha_min"] if early else self.cfg["alpha_min"])
            ),
            "g_residual_share": (
                float(diag.get("contribution/alpha_z_share", 0.0))
                + float(diag.get("contribution/anchor_share", 0.0))
                + float(diag.get("contribution/mid_anchor_share", 0.0))
                >= self.cfg["residual_share_min"]
            ),
            "g_val_floor": val_map50 is None or val_map50 >= self.cfg["val_map50_floor"],
        }
        verdict = {
            "epoch": epoch,
            "phase": "early" if early else "main",
            "observed": {
                "sigma2_std": diag.get("gaussian/sigma2_std"),
                "sigma2_mean": diag.get("gaussian/sigma2_mean"),
                "sigma1_mean": diag.get("gaussian/sigma1_mean"),
                "gates/alpha": diag.get("gates/alpha"),
                "residual_share_sum": (
                    float(diag.get("contribution/alpha_z_share", 0.0))
                    + float(diag.get("contribution/anchor_share", 0.0))
                    + float(diag.get("contribution/mid_anchor_share", 0.0))
                ),
                "val_map50": val_map50,
            },
            "checks": checks,
            "kill": not all(checks.values()),
            "thresholds": {
                k: v for k, v in self.cfg.items() if k.endswith(("_min", "_floor"))
            },
        }
        self.verdicts.append(verdict)
        self._write(trainer)
        print(f"[gauss-gate] e{epoch} verdict: {json.dumps(verdict)}", flush=True)
        if verdict["kill"]:
            print(
                f"[gauss-gate] KILL at epoch {epoch}: failed "
                f"{[k for k, ok in checks.items() if not ok]}; stopping training",
                flush=True,
            )
            trainer.stop = True

    def _check_finite(self, trainer, epoch: int) -> None:
        """g_loss_finite: NaN/Inf in the loss or diagnostics is an instant kill."""
        losses = getattr(trainer, "loss_items", None)
        vals = []
        if losses is not None:
            # Ultralytics >=8.4 passes a dict {'box_loss': tensor, ...}; older
            # versions pass a tensor. Iterate values only, and never let a
            # non-numeric entry kill the run (verified in a real CPU run).
            items = losses.values() if isinstance(losses, dict) else losses
            vals.extend(
                float(v) for v in items
                if isinstance(v, (int, float)) or hasattr(v, "item")
            )
        diag = self.modules[0][1].last_diag if self.modules else {}
        vals.extend(
            float(v)
            for k, v in diag.items()
            if isinstance(v, float) and k.startswith(("gaussian/", "gates/", "contribution/"))
        )
        if vals and not all(math.isfinite(v) for v in vals):
            print(f"[gauss-gate] KILL at epoch {epoch}: non-finite loss/diag", flush=True)
            self.verdicts.append({"epoch": epoch, "checks": {"g_loss_finite": False}, "kill": True})
            self._write(trainer)
            trainer.stop = True

    @staticmethod
    def _val_map50(trainer) -> float | None:
        metrics = getattr(trainer, "metrics", None) or {}
        for key in ("metrics/mAP50(B)", "metrics/mAP50"):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _write(self, trainer) -> None:
        run = Path(getattr(trainer, "save_dir", ".")).parent
        out = run / "metrics" / "gauss_gate_verdict.json"
        try:
            atomic_write_text(out, json.dumps(self.verdicts, indent=2))
        except OSError as exc:
            print(f"[gauss-gate] could not write verdict: {exc}", flush=True)


def register_gauss_gate(model, cfg: dict | None = None) -> GaussGateCallback:
    """Attach the gate callback; call where other trainer callbacks are wired."""
    cb = GaussGateCallback(cfg)
    model.callbacks["on_train_start"].append(cb.on_train_start)
    model.callbacks["on_fit_epoch_end"].append(cb.on_fit_epoch_end)
    return cb
