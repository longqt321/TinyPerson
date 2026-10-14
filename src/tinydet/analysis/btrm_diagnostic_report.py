"""Assemble post-hoc BTRM mechanism diagnostics after a mandatory math pass."""
import csv
import json
from pathlib import Path

import yaml

from tinydet.utils.experiment import metadata, write_json


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence not produced: {path}")
    return json.loads(path.read_text())


def _read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence not produced: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _fmt(value, digits=5):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def _group_stat(report, bin_name, label, key):
    return (report.get("by_size", {}).get(bin_name, {})
            .get(label, {}).get(key, {}).get("median"))


def _pair_report(records: list[dict], arm: str, bin_name: str, threshold: float):
    return next(
        row for row in records
        if row["model"] == arm and row["bin"] == bin_name
        and float(row["confidence_threshold"]) == threshold
    )


def _plots(destination: Path, feature_rows: dict, pr_rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only plot real A2/A3 gate variables, not a hypothetical baseline gate.
    for model in ("btrm_a2", "btrm_a3"):
        subset = [r for r in feature_rows[model] if r["size_bin"] == "0-8"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for group in ("GT_center", "BG"):
            samples = [
                float(r["g_btrm" if model == "btrm_a3" else "g_contrast"])
                for r in subset if r["group"] == group
            ]
            if samples:
                ax.hist(samples, bins=30, alpha=0.5, density=True, label=f"{group} n={len(samples)}")
        ax.set_xlabel("Actual branch's g = tanh(z / saturation)")
        ax.set_ylabel("Density")
        ax.set_title(f"{model}: GT versus near-background gate, 0-8px")
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / f"{model}_gate_hist_0_8.png", dpi=150)
        plt.close(fig)

    for bin_name in ("0-8", "8-16"):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for model in ("btrm_a0", "btrm_a2", "btrm_a3"):
            rows = sorted(
                (r for r in pr_rows if r["model"] == model and r["bin"] == bin_name),
                key=lambda r: float(r["confidence_threshold"])
            )
            # precision vs recall at fixed thresholds; NOT a full AP curve.
            recall = [float(r["recall"]) for r in rows if r["recall"] and r["precision"]]
            precision = [float(r["precision"]) for r in rows if r["recall"] and r["precision"]]
            if recall:
                ax.plot(recall, precision, marker="o", label=model)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall, IoU=0.5")
        ax.set_ylabel("Precision")
        ax.set_title(f"Confidence-threshold PR samples: {bin_name}px")
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / f"pr_thresholds_{bin_name.replace('-', '_')}.png", dpi=150)
        plt.close(fig)


def assemble_report(references: dict[str, Path], destination: Path, split: str = "val") -> dict:
    """Use only independently verified formula and saved observational outputs."""
    math = _read_json(destination / "math_audit.json")
    if math["status"] != "PASS":
        raise RuntimeError("Math gate failed; analysis is prohibited")
    prediction = _read_json(destination / "prediction_summary.json")
    pr_rows = _read_csv(destination / "precision_recall_thresholds.csv")
    feature = {}
    feature_rows = {}
    saved_configs = {}
    for model, run in references.items():
        feature[model] = _read_json(
            run / "diagnostics" / "btrm_mechanism" / split / "summary.json"
        )
        feature_rows[model] = _read_csv(
            run / "diagnostics" / "btrm_mechanism" / split / "samples.csv"
        )
        saved_configs[model] = yaml.safe_load((run / "config.yaml").read_text())
        if feature[model]["run"] != str(run):
            raise ValueError("Feature report checkpoint identity mismatch")

    a0 = saved_configs["btrm_a0"]
    for arm in ("btrm_a2", "btrm_a3"):
        for key in ("seed", "data", "train", "hardware"):
            if a0[key] != saved_configs[arm][key]:
                raise ValueError(f"Unmatched {key}: {arm}")
    commits = {metadata(run)["git_commit"] for run in references.values()}
    if len(commits) != 1:
        raise ValueError("Cannot compare different training commits")
    counts = {key: val["selected_images"] for key, val in feature.items()}
    if len(set(counts.values())) != 1:
        raise ValueError("Feature subsets are not matched")
    a0_images = {
        r["image"] for r in feature_rows["btrm_a0"] if r["group"] == "GT_center"
    }
    for arm in ("btrm_a2", "btrm_a3"):
        if {r["image"] for r in feature_rows[arm] if r["group"] == "GT_center"} != a0_images:
            raise ValueError("Different P3 sample images; comparison invalid")

    lines = [
        "# BTRM mechanism diagnostics", "",
        "## Gate 1: mathematics and numerical implementation",
        (f"Status: **{math['status']}**; comparisons: {math['checks']}; "
        f"max absolute error: {math['max_abs_error']:.9g}."),
        (f"Max reported relative error (nonzero oracle): "
        f"{math['max_rel_error_nonzero']:.6g}. "
        "Large *relative* errors near zero can coexist with tiny absolute errors."),
        "The oracle uses independent NumPy float64 least-squares and solve.",
        ("This verifies the specified algebra within tested numerical tolerances, "
        "NOT the semantic assumption that local gradients correspond to background."),
        "",
        "## Gate 2: observed checkpoint modulation",
        "| Arm | Learned alpha (effective gain) | P3 gate parity max abs error | g>=0.95 map fraction (mean) |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("btrm_a0", "btrm_a2", "btrm_a3"):
        f = feature[arm]
        stats = f["all_feature_maps"]
        metric = ("g_btrm_all_sat95" if arm == "btrm_a3" else "g_contrast_all_sat95")
        lines.append(
            f"| {arm} | {_fmt(f['effective_gain'])} | "
            f"{_fmt(stats['modulation_abs_max_error'].get('p99'), 7)} | "
            f"{_fmt(stats[metric].get('mean'))} |"
        )
    lines.extend([
        "",
        "Note: A0 has no trainable gate. Its computed z/g are counterfactual probes only.",
        "",
        "## GT-center versus background: median evidence",
        ("GT centers are oracle sampled. Background negatives are matched within image "
        "but cannot be assumed to be truly empty if annotation is incomplete."),
        "| Model | Size (px) | GT z | BG z | GT gate g | BG gate g | ring GT cells (median) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for arm in ("btrm_a0", "btrm_a2", "btrm_a3"):
        key = "btrm" if arm == "btrm_a3" else "contrast"
        for size in ("0-8", "8-16", "16-32", ">32"):
            zcol, gcol = "z_" + key, "g_" + key
            stats = [
                _group_stat(feature[arm], size, "GT_center", zcol),
                _group_stat(feature[arm], size, "BG", zcol),
                _group_stat(feature[arm], size, "GT_center", gcol),
                _group_stat(feature[arm], size, "BG", gcol),
                _group_stat(
                    feature[arm], size, "GT_center", "ring_inside_gt_cell_centers"
                ),
            ]
            lines.append(f"| {arm} | {size} | " + " | ".join(_fmt(x) for x in stats) + " |")
    lines.extend([
        "", "## Saved detector predictions: confidence threshold diagnostics",
        ("These are the **actual saved detector predictions**; no new inference "
        "or training is performed for this section."),
        "| Bin | Conf | Model | TP | FP | Precision | Recall | FP/image |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for size in ("0-8", "8-16"):
        for threshold in (.001, .05, .25):
            for arm in ("btrm_a0", "btrm_a2", "btrm_a3"):
                row = _pair_report(pr_rows, arm, size, threshold)
                lines.append(
                    f"| {size} | {threshold:g} | {arm} | {row['TP']} | "
                    f"{row['FP']} | {_fmt(float(row['precision']) if row['precision'] else None)} | "
                    f"{_fmt(float(row['recall']) if row['recall'] else None)} | "
                    f"{_fmt(float(row['false_positives_per_image']))} |"
                )
    change = prediction["tiny_recall_A3_minus_A0_at_0.001"]
    lines.extend([
        "", "## Tiny 0-8px TP difference (A3 versus A0, conf>=0.001)",
        (f"Net TP: {change['delta_TP']:+d}; net FP: {change['delta_FP']:+d}; "
        f"newly matched GT: {change['newly_matched_GT']}; "
        f"previously matched GT lost: {change['previously_matched_GT_lost']}."),
        "",
        "## Decision protocol",
        ("1. If math or checkpoint parity fails: **numerical/integration bug**; "
        "do not interpret model effects."),
        ("2. If the P3 gate is nearly constant/saturated: modulation may have "
        "lost spatial selectivity; confirm with full per-image CSV and histograms."),
        ("3. If z/g cannot separate GT-center from nearby BG: evidence may not "
        "be object-specific (or GT/background sampling is noisy)."),
        ("4. If recall improves only at conf=0.001 while FP rises strongly or "
        "the effect disappears at higher confidence: revisit score ranking "
        "and calibration, not just recall."),
        "",
        ("No claim of significant mAP improvement is justified from this "
        "one-seed diagnostic. All checkpoints remain unchanged."),
    ])
    (destination / "mechanism_report.md").write_text("\n".join(lines) + "\n")
    _plots(destination, feature_rows, pr_rows)
    combined = {
        "math": {
            "status": math["status"], "checks": math["checks"],
            "max_abs_error": math["max_abs_error"],
            "max_rel_error_nonzero": math["max_rel_error_nonzero"],
        },
        "runs": {model: str(path) for model, path in references.items()},
        "trained_commit": next(iter(commits)),
        "checkpoint_feature": feature,
        "saved_prediction": prediction,
        "limits": [
            "No GPU retraining; all inferences for P3 features use frozen checkpoints.",
            "The P3 GT-center probe has oracle location bias.",
            "No uncertainty estimate over training seeds.",
        ],
    }
    write_json(destination / "mechanism_summary.json", combined)
    return {"output": str(destination), "status": "COMPLETE",
            "report": str(destination / "mechanism_report.md")}
