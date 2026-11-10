"""GauSS mechanism probes — consolidated, corrected measurement tool.

Subcommands (all run on CPU with the REAL YOLO detection loss):
  grad       T2: per-batch sigma gradient norms + sign consistency + SNR
  ident      T3: paired-delta identifiability (sigma shifts vs controls)
  minitrain  T4: short real-loss training trace of sigma/gates

Measurement hygiene (fixes for the audit findings A1-A4):
  - All loss measurements run in model.eval() so BatchNorm running stats are
    frozen during measurement (train-mode no-grad forwards mutate them).
  - When loading a v3.1 checkpoint, gamma_mid_raw is zeroed so the measured
    model is exactly v3.1 (the warm-start value is a v3.2-only path).
  - The loss hyper-parameters are read from the experiment config and
    asserted equal to Ultralytics defaults; a mismatch aborts instead of
    silently measuring a different objective.

Usage:
  uv run --extra cpu python scripts/gauss_probe.py ident \
      --config configs/experiment/gauss_v32_tinyperson_wacv_150e.yaml \
      [--checkpoint PATH] [--batches 8] [--batch-size 4] [--imgsz 320] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tinydet.models.registry import build_model
from tinydet.modules.gauss import remap_v31_to_v32_state_dict

DEFAULT_DATASET = REPO / "datasets/derived/tinyperson-yolo-v2"


def load_batch(paths: list[Path], imgsz: int, root: Path) -> dict:
    imgs, cls, boxes, bidx = [], [], [], []
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
        imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1))
        lab = root / "labels/train" / (p.stem + ".txt")
        if lab.is_file():
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls.append([float(parts[0])])
                boxes.append([float(v) for v in parts[1:5]])
                bidx.append([i])
    return {
        "img": torch.stack(imgs).float() / 255.0,
        "cls": torch.tensor(cls, dtype=torch.float32).reshape(-1, 1),
        "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "batch_idx": torch.tensor(bidx, dtype=torch.float32).reshape(-1),
    }


def build(config_path: Path, checkpoint: Path | None) -> torch.nn.Module:
    cfg = yaml.safe_load(config_path.read_text())
    model = build_model(cfg["model"]).model
    from ultralytics.cfg import get_cfg

    hyp = get_cfg()
    # A3 fix: the probe objective must match the training objective. Assert
    # the loss gains against the experiment config when it specifies them.
    train_cfg = cfg.get("train", {})
    for k in ("box", "cls", "dfl"):
        if k in train_cfg and float(train_cfg[k]) != float(getattr(hyp, k)):
            raise SystemExit(
                f"loss gain mismatch: config {k}={train_cfg[k]} vs ultralytics "
                f"default {getattr(hyp, k)} — probes must use the training hyp"
            )
    model.args = hyp
    model.criterion = None

    if checkpoint is not None:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = remap_v31_to_v32_state_dict(ck["model"].state_dict())
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # A2 fix: a v3.1 checkpoint has NO gamma_mid key at all, so it must be
        # zeroed on the MODEL (not in the state dict) — otherwise the v3.2
        # warm-start value 0.05 keeps an untrained path active and the probe
        # would not measure v3.1.
        with torch.no_grad():
            for name, prm in model.named_parameters():
                if name.endswith("gamma_mid_raw"):
                    prm.zero_()
        print(f"loaded checkpoint {checkpoint.name} (remapped); "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    # NOTE: measurements must run in train() mode — the detection loss is
    # defined on raw train-mode predictions (eval mode decodes boxes and is
    # ~2000x less sensitive). BN running stats are instead frozen by
    # snapshot/restore inside measure_losses(), which keeps the model state
    # identical across conditions without changing the forward path.
    model.train()
    return model


def sigma_param_names(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    named = dict(model.named_parameters())
    s1 = [n for n in named if n.endswith(("sigma1x_raw", "sigma1y_raw"))]
    s2 = [n for n in named if n.endswith(("sigma2x_raw", "sigma2y_raw"))]
    return s1, s2


def measure_losses(model: torch.nn.Module, batches: list[dict]) -> np.ndarray:
    """Loss on every batch in train mode, with BN running stats frozen.

    Train mode is required (the loss is defined on raw train-mode
    predictions). Running-stat drift is prevented by snapshotting the
    buffers before the measurement and restoring them afterwards, so every
    condition sees exactly the same model state.
    """
    bufs = {n: t.detach().clone() for n, t in model.named_buffers() if "running" in n}
    out = []
    with torch.no_grad():
        for b in batches:
            l, _ = model.loss(b)
            out.append(float(l.sum()))
    with torch.no_grad():
        for n, t in bufs.items():
            dict(model.named_buffers())[n].copy_(t)
    return np.array(out)


def cmd_grad(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    model = build(args.config, args.checkpoint)
    root = Path(args.dataset)
    files = sorted((root / "images/train").glob("*.jpg"))[: args.batch_size * args.batches]
    batches = [load_batch(files[i * args.batch_size:(i + 1) * args.batch_size],
                          args.imgsz, root) for i in range(args.batches)]
    named = dict(model.named_parameters())
    s1, s2 = sigma_param_names(model)

    # Gradients need train-mode graph, but BN stats must not drift across
    # batches: snapshot buffers, run each batch, restore.
    bufs = {n: t.detach().clone() for n, t in model.named_buffers() if "running" in n}
    model.train()
    per_batch: list[torch.Tensor] = []
    conv_norms: list[float] = []
    conv_names = [n for n, p in named.items() if p.ndim > 1 and "gaussian" not in n][:40]
    for b in batches:
        model.zero_grad(set_to_none=True)
        l, _ = model.loss(b)
        l.sum().backward()
        g = torch.cat([named[n].grad.flatten() for n in s1 + s2])
        per_batch.append(g.clone())
        cn = [float(named[n].grad.norm()) for n in conv_names if named[n].grad is not None]
        conv_norms.append(sum(cn) / len(cn))
        with torch.no_grad():
            for n, t in bufs.items():
                dict(model.named_buffers())[n].copy_(t)
    model.eval()

    G = torch.stack(per_batch)
    mean_g, std_g = G.mean(0), G.std(0, unbiased=True)
    snr = mean_g.abs() / (std_g / np.sqrt(len(G))).clamp_min(1e-30)
    # channel count is the module hidden dim (32), not 64 — compute from
    # the tensors themselves instead of assuming (audit fix: the old
    # probe's hardcoded 64 merged sigma1 and sigma2 into one slice)
    n1 = sum(dict(model.named_parameters())[n].numel() for n in s1)
    print(f"batches={len(G)} x {args.batch_size} imgs @ {args.imgsz}px")
    print(f"|g_sigma| per batch: {[round(float(x.norm()), 5) for x in per_batch]}")
    print(f"mean|g_conv| per batch: {[round(v, 2) for v in conv_norms]}")
    print(f"sign consistency (weighted): {float(torch.sign(G).mean(0).abs().mean()):.3f}")
    print(f"SNR sigma1: median={float(snr[:n1].median()):.3f} max={float(snr[:n1].max()):.3f}")
    print(f"SNR sigma2: median={float(snr[n1:].median()):.3f} max={float(snr[n1:].max()):.3f}")
    print(f"frac sigma SNR>2: {float((snr > 2).float().mean()):.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "sigma_grad_norms": [float(x.norm()) for x in per_batch],
            "conv_grad_norms": conv_norms,
            "sign_consistency": float(torch.sign(G).mean(0).abs().mean()),
            "snr_sigma1_median": float(snr[:n1].median()),
            "snr_sigma2_median": float(snr[n1:].median()),
            "frac_snr_gt2": float((snr > 2).float().mean()),
        }, indent=2))


def cmd_ident(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    model = build(args.config, args.checkpoint)
    root = Path(args.dataset)
    files = sorted((root / "images/train").glob("*.jpg"))[: args.batch_size * args.batches]
    batches = [load_batch(files[i * args.batch_size:(i + 1) * args.batch_size],
                          args.imgsz, root) for i in range(args.batches)]
    named = dict(model.named_parameters())
    s1, s2 = sigma_param_names(model)

    base = measure_losses(model, batches)
    print(f"baseline loss: mean={base.mean():.3f} std={base.std(ddof=1):.3f} "
          f"(train mode, BN buffers frozen)")

    results: list[dict] = []

    def paired(tag: str, apply, restore) -> None:
        apply()
        cur = measure_losses(model, batches)
        restore()
        d = cur - base
        sem = d.std(ddof=1) / np.sqrt(len(d))
        t = float(d.mean() / sem) if sem > 0 else 0.0
        print(f"  {tag:44s} dMean={d.mean():+9.4f} sem={sem:.4f} t={t:+7.2f}")
        results.append({"tag": tag, "dmean": float(d.mean()), "sem": float(sem), "t": t})

    def shifter(names: list[str], delta: float):
        saved = {n: named[n].detach().clone() for n in names}

        def apply() -> None:
            with torch.no_grad():
                for n in names:
                    named[n].add_(delta)

        def restore() -> None:
            with torch.no_grad():
                for n, v in saved.items():
                    named[n].copy_(v)

        return apply, restore

    for names, label in ((s1, "sigma1"), (s2, "sigma2")):
        for d in (-2.0, 2.0):
            apply, restore = shifter(names, d)
            paired(f"{label} raw {d:+.1f} (full range)", apply, restore)

    gate_names = [n for n in named if n.endswith(
        ("alpha_raw", "gamma_in_raw", "gamma_out_raw", "gamma_mid_raw"))]
    saved_g = {n: named[n].detach().clone() for n in gate_names}

    def zero_gates() -> None:
        with torch.no_grad():
            for n in gate_names:
                named[n].zero_()

    def restore_gates() -> None:
        with torch.no_grad():
            for n, v in saved_g.items():
                named[n].copy_(v)

    paired("gates -> 0 (remove residual)", zero_gates, restore_gates)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


def cmd_minitrain(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    from tinydet.modules.gauss import (  # noqa: F401
        collect_gauss_control_params,
        configure_gauss_control_param_group,
    )

    root = Path(args.dataset)
    files = sorted((root / "images/train").glob("*.jpg"))[: args.batch_size * 6]
    batches = [load_batch(files[i * args.batch_size:(i + 1) * args.batch_size],
                          args.imgsz, root) for i in range(6)]
    trace_out = {}
    for mult in (1.0, args.lr_mult):
        model = build(args.config, None)
        model.train()  # training runs in train mode by design
        named = dict(model.named_parameters())
        s1, s2 = sigma_param_names(model)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0005)
        configure_gauss_control_param_group(opt, model, lr_mult=mult)
        trace = {"step": [], "s1_mean": [], "s2_mean": [], "s1_std": [], "s2_std": [],
                 "alpha": [], "loss": []}
        for step in range(args.steps):
            b = batches[step % len(batches)]
            opt.zero_grad(set_to_none=True)
            l, _ = model.loss(b)
            l.sum().backward()
            opt.step()
            if step % 10 == 0 or step == args.steps - 1:
                s1v = torch.cat([named[n].detach() for n in s1])
                s2v = torch.cat([named[n].detach() for n in s2])
                trace["step"].append(step)
                trace["s1_mean"].append(float(s1v.mean()))
                trace["s2_mean"].append(float(s2v.mean()))
                trace["s1_std"].append(float(s1v.std()))
                trace["s2_std"].append(float(s2v.std()))
                trace["alpha"].append(
                    float(0.5 * torch.tanh(named["model.20.alpha_raw"].detach())))
                trace["loss"].append(float(l.sum()))
        print(f"\n== lr_mult={mult} ==")
        for i, st in enumerate(trace["step"]):
            print(f"  step {st:4d}: loss={trace['loss'][i]:8.3f} | "
                  f"s1={trace['s1_mean'][i]:+.4f} (std {trace['s1_std'][i]:.4f}) | "
                  f"s2={trace['s2_mean'][i]:+.4f} (std {trace['s2_std'][i]:.4f}) | "
                  f"a={trace['alpha'][i]:+.3f}")
        trace_out[f"m{mult}"] = trace
    if args.json:
        Path(args.json).write_text(json.dumps(trace_out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("grad", "ident", "minitrain"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path,
                       default=REPO / "configs/experiment/gauss_v32_tinyperson_wacv_150e.yaml")
        p.add_argument("--checkpoint", type=Path, default=None)
        p.add_argument("--dataset", default=str(DEFAULT_DATASET))
        p.add_argument("--batches", type=int, default=8)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--imgsz", type=int, default=320)
        p.add_argument("--threads", type=int, default=16)
        p.add_argument("--json", default=None)
        if name == "minitrain":
            p.add_argument("--steps", type=int, default=150)
            p.add_argument("--lr-mult", type=float, default=3.0)
        p.set_defaults(func={"grad": cmd_grad, "ident": cmd_ident,
                             "minitrain": cmd_minitrain}[name])
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
