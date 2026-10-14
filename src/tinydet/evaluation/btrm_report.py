"""Automatic three-arm BTRM ablation report from persisted Modal run artifacts.

This computes paired *descriptive* deltas, NOT significance tests. With one
training seed there is no evidence of cross-seed statistical significance.
"""
import csv
import json
import math
from pathlib import Path

import torch
import yaml

from tinydet.modules.btrm import GATE_INDEX, LocalBackgroundEvidence
from tinydet.utils.experiment import metadata, resolve_run, write_json


def _safe_group(group: str) -> str:
    if not group or not group.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid group identifier")
    return group


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required comparison artifact missing: {path}")
    return json.loads(path.read_text())


def _gain(checkpoint: Path):
    """Read the actual learned gate strength from the saved best checkpoint."""
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    gate = model.model.model[GATE_INDEX]
    if not isinstance(gate, LocalBackgroundEvidence):
        return None
    return float(gate.effective_gain.detach().float().cpu())


def _flatten(run: Path, split: str, need_size: bool) -> dict:
    config = yaml.safe_load((run / "config.yaml").read_text())
    details = metadata(run)
    if details.get("status") != "COMPLETED":
        raise ValueError(f"Run not COMPLETED: {run}")
    if not (run / "checkpoints" / "best.pt").is_file():
        raise FileNotFoundError(run / "checkpoints" / "best.pt")
    overall = _load_json(run / "metrics" / f"{split}.json")
    benchmark = _load_json(run / "metrics" / "benchmark.json")
    size = (_load_json(run / "metrics" / f"size_{split}.json")
            if need_size else {})
    record = {
        "model": config["name"],
        "run": str(run),
        "seed": config["seed"],
        "git_commit": details.get("git_commit"),
        "git_dirty": details.get("git_dirty"),
        "model_sha256": details.get("model_config_sha256"),
        "mode": config["model"].get("feature_module", {}).get("mode", "none"),
        "AP50": overall["AP50"],
        "AP50_95": overall["AP"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "parameters": benchmark["parameters"],
        "latency_ms": benchmark["latency_ms"],
        "gpu_memory_mb": benchmark["gpu_memory_mb"],
        "effective_gain": _gain(run / "checkpoints" / "best.pt"),
    }
    for size_bin in ("0-8", "8-16", "16-32", ">32"):
        prefix = "size_" + size_bin.replace("-", "_").replace(">", "gt")
        if need_size:
            if size_bin not in size:
                raise ValueError(f"Missing size bin {size_bin}: {run}")
            record[f"{prefix}_GT"] = size[size_bin]["GT"]
            record[f"{prefix}_AP50"] = size[size_bin]["AP50"]
            record[f"{prefix}_AP50_95"] = size[size_bin]["AP50-95"]
            record[f"{prefix}_recall50_conf0.001"] = size[size_bin]["Recall50_conf0.001"]
    return record, config


def _delta(after: dict, before: dict):
    # Absolute differences; + AP is favorable and + latency is unfavorable.
    result = {}
    for key in after:
        if key in {"model", "run", "seed", "git_commit", "git_dirty", "model_sha256", "mode"}:
            continue
        a, b = after[key], before.get(key)
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
            if math.isfinite(float(a)) and math.isfinite(float(b)):
                result[key] = float(a) - float(b)
    return result


def write_btrm_report(root: Path, references: dict, config: dict, group: str) -> dict:
    """Persist reproducible comparison artifacts after all 3 runs finish."""
    _safe_group(group)
    split = config.get("split", "val")
    if split not in {"val", "test"}:
        raise ValueError("report.split must be val or test")
    roles = ("baseline", "contrast", "proposed")
    names = [config.get(key) for key in roles]
    if not all(isinstance(n, str) and n for n in names) or len(set(names)) != 3:
        raise ValueError("report baseline/contrast/proposed must be distinct models")
    if set(names) - set(references):
        raise ValueError(f"Missing ablation runs: {set(names) - set(references)}")
    need_size = bool(config.get("size_metrics", True))
    records, saved_configs = {}, {}
    for name in names:
        run = resolve_run(root, references[name])
        records[name], saved_configs[name] = _flatten(run, split, need_size)
        if records[name]["model"] != name:
            raise ValueError(f"Run identity mismatch: {name}")
    # Different models and module settings only. Everything else must match.
    base = saved_configs[names[0]]
    for name in names[1:]:
        current = saved_configs[name]
        for field in ("seed", "data", "train", "hardware"):
            if base[field] != current[field]:
                raise ValueError(f"Unmatched {field} between comparison arms")
        if base["model"]["config"] != current["model"]["config"]:
            raise ValueError("Comparison requires a shared architectural YAML")
    if [records[n]["mode"] for n in names] != ["none", "contrast", "btrm"]:
        raise ValueError("Arms must be none, contrast, btrm in that order")
    if any(records[n]["git_dirty"] for n in names):
        print("WARNING: at least one experiment ran from dirty git state", flush=True)

    a0, a2, a3 = (records[name] for name in names)
    deltas = {
        "A2_minus_A0": _delta(a2, a0),
        "A3_minus_A2": _delta(a3, a2),
        "A3_minus_A0": _delta(a3, a0),
    }
    dest = root / "comparisons" / "btrm" / group
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest / "summary.json", {
        "research_question": "Contribution of tangent-space removal beyond normalized contrast",
        "split": split, "runs": references, "arms": records,
        "absolute_deltas": deltas,
        "interpretation": (
            "Single-seed deltas are descriptive, not statistical significance. "
            "Native overall AP and custom size-bin AP may use different matching protocols."
        ),
    })
    columns = list(a0)
    with (dest / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records[n] for n in names)
    with (dest / "deltas.csv").open("w", newline="") as file:
        keys = sorted(set().union(*(set(v) for v in deltas.values())))
        writer = csv.DictWriter(file, fieldnames=["comparison", *keys])
        writer.writeheader()
        writer.writerows({"comparison": k, **v} for k, v in deltas.items())

    relevant = [
        "AP50", "AP50_95", "size_0_8_AP50", "size_8_16_AP50",
        "size_0_8_recall50_conf0.001", "size_8_16_recall50_conf0.001",
        "latency_ms", "gpu_memory_mb", "effective_gain",
    ]
    lines = [
        "# BTRM ablation: " + group, "",
        "One seed, same 50-epoch schedule and same YOLO architecture for A0/A2/A3.",
        "**Deltas are descriptive, not statistically significant claims.**", "",
        "| Metric | A0 | A2 | A3 | A3-A2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in relevant:
        vals = [records[n].get(key) for n in names]
        delta = deltas["A3_minus_A2"].get(key)
        def fmt(v):
            return f"{v:.6f}" if isinstance(v, (float, int)) else "n/a"
        lines.append("| " + key + " | " + " | ".join(map(fmt, [*vals, delta])) + " |")
    lines += [
        "", "## Interpretation gate",
        "A3 > A2 in tiny-bin AP is required to support a specific contribution",
        "from nuisance-tangent suppression. A3 > A0 alone is insufficient.",
        "Validate false positives, latency, cross-seed stability, and an independent",
        "test split before making a publication-level performance claim.",
    ]
    (dest / "report.md").write_text("\n".join(lines) + "\n")
    return {"output": str(dest), "runs": references, "deltas": deltas}
