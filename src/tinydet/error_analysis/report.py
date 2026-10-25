"""Human-readable and visual outputs built only from validated analysis records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tinydet.error_analysis.core import AnalysisResult


def _json(value: Any) -> str:
    return "n/a" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_json(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(result: AnalysisResult) -> str:
    summary = result.summary
    counts = summary["counts"]
    rates = summary["rates"]
    integrity = summary["integrity"]
    provenance = summary["provenance"]
    lines = [
        f"# Error analysis — {summary['dataset']['split']}",
        "",
        "## Provenance",
        "",
        f"- Run: `{provenance['run'] or 'not supplied'}`",
        f"- Checkpoint: `{provenance['checkpoint'] or 'not supplied'}`",
        f"- Checkpoint SHA-256: `{provenance['checkpoint_sha256'] or 'not supplied'}`",
        f"- Dataset split: `{summary['dataset']['split']}`",
        f"- Annotations SHA-256: `{provenance['annotations_sha256']}`",
        f"- Manifest SHA-256: `{provenance['manifest_sha256'] or 'not supplied'}`",
        f"- Predictions SHA-256: `{provenance['predictions_sha256']}`",
        f"- Analysis config SHA-256: `{provenance['analysis_config_sha256']}`",
        f"- Implementation: `{provenance['implementation_version']}`",
        "",
        "## Validity",
        "",
        f"- Status: **{'VALID' if integrity['valid'] else 'INVALID'}**",
        f"- Expected images: {summary['dataset']['expected_images']}",
        f"- Analyzed images: {summary['dataset']['analyzed_images']}",
        f"- Dense images excluded by protocol: {summary['dataset']['dense_images_excluded']}",
        f"- Predictions source: `{summary['dataset']['predictions_source']}`",
    ]
    if integrity["issues"]:
        lines.extend(["", "### Integrity findings", ""])
        lines.extend(f"- `{issue['code']}`: {issue['message']}"
                     for issue in integrity["issues"][:100])
    if not all(integrity["invariants"].values()):
        lines.extend(["", f"- Invariant failure: `{integrity['invariants']}`"])

    lines.extend([
        "",
        "## Observed counts",
        "",
        _table(
            ["quantity", "count", "denominator", "rate"],
            [
                ["positive ground truths", summary["positive_ground_truths"], "", ""],
                ["matched ground truths", counts["true_positives"],
                 summary["positive_ground_truths"], _json(rates["recall"])],
                ["missed ground truths", counts["misses"],
                 summary["positive_ground_truths"], _json(rates["miss_rate"])],
                ["predictions", counts["predictions"], "", ""],
                ["true positives", counts["true_positives"], "", ""],
                ["false positives", counts["false_positives"],
                 counts["true_positives"] + counts["false_positives"],
                 _json(rates["precision"])],
                ["duplicates", counts["duplicates"], counts["false_positives"],
                 _json(rates["duplicate_share_of_false_positives"])],
                ["localization errors", counts["localization_errors"],
                 counts["false_positives"],
                 _json(rates["localization_share_of_false_positives"])],
                ["background false positives", counts["background_false_positives"],
                 counts["false_positives"],
                 _json(rates["background_share_of_false_positives"])],
                ["neutral predictions", counts["neutral_predictions"], "", ""],
            ],
        ),
        "",
        ("These categories are exclusive for valid person predictions: each prediction is "
         "TP, FP, or neutral, and each positive ground truth is matched or missed. "
         "Precision here uses the predictions retained after the configured confidence "
         "threshold."),
        "",
        "## Recall by object size",
        "",
        _table(
            ["diagnostic size bin", "ground truths", "matched", "missed", "recall", "miss rate"],
            [
                [name, row["positive_ground_truths"], row["matched"], row["misses"],
                 _json(row["recall"]), _json(row["miss_rate"])]
                for name, row in summary["aggregation"]["by_size_bin"].items()
            ],
        ),
        "",
        ("Diagnostic size bins are half-open `[low, high)` in sqrt(pixel area). The AP "
         "cross-check uses the repository evaluator's own inclusive size ranges."),
        "",
        "## Recall and false positives by image density",
        "",
        _table(
            ["density bin", "images", "ground truths", "matched", "missed", "recall",
             "false positives", "FP/image"],
            [
                [name, row["images"], row["positive_ground_truths"], row["matched"],
                 row["misses"], _json(row["recall"]), row["false_positives"],
                 _json(row["false_positives_per_image"])]
                for name, row in summary["aggregation"]["by_density"].items()
            ],
        ),
        "",
        "## Border versus interior objects",
        "",
        _table(
            ["position", "ground truths", "matched", "missed", "recall"],
            [
                [name, row["positive_ground_truths"], row["matched"], row["misses"],
                 _json(row["recall"])]
                for name, row in summary["aggregation"]["by_boundary"].items()
            ],
        ),
        "",
        "## Confidence operating points",
        "",
        _table(
            ["score threshold", "predictions", "TP", "FP", "recall", "precision"],
            [
                [row["score_threshold"], row["predictions"], row["true_positives"],
                 row["false_positives"], _json(row["recall"]), _json(row["precision"])]
                for row in summary["aggregation"]["confidence_operating_points"]
            ],
        ),
        "",
        ("The cumulative points reuse the primary-threshold matching outcomes. They are not "
         "a replacement for a full precision-recall curve at each threshold."),
        "",
        "## Diagnostic AP cross-check",
    ])
    crosscheck = summary["metric_crosscheck"]
    if crosscheck.get("available"):
        lines.extend([
            "",
            f"Status: **{crosscheck['status']}** against the repository TinyPerson evaluator.",
            "",
            _table(
                ["bin:metric", "diagnostic", "reference", "absolute difference"],
                [
                    [key, row["diagnostic"], row["reference"], row["abs_diff"]]
                    for key, row in crosscheck["comparisons"].items()
                ],
            ),
        ])
    else:
        lines.extend(["", f"Unavailable: {crosscheck.get('reason', 'unknown reason')}"])

    comparison = summary.get("comparison")
    if comparison:
        lines.extend(["", "## Paired comparison", ""])
        if comparison.get("available"):
            lines.extend([
                f"- Shared images: {comparison['shared_images']}",
                f"- Clean → error (regressed): {comparison['regressed_images']}",
                f"- Error → clean (corrected): {comparison['corrected_images']}",
                f"- Count deltas: `{json.dumps(comparison['count_deltas'], sort_keys=True)}`",
            ])
        else:
            lines.append(f"Unavailable: {comparison.get('reason')}")

    lines.extend([
        "",
        "## Investigation priorities",
        "",
        ("Ranking is by observed error count, then rate. It is a triage order, not causal "
         "attribution."),
        "",
        _table(
            ["category", "group", "count", "denominator", "rate", "metric affected"],
            [
                [row["category"], row["group"], row["count"], row["denominator"],
                 _json(row["rate"]), row["metric_impact"]]
                for row in summary["aggregation"]["priorities"]
            ],
        ),
        "",
        "## Hypotheses and follow-up",
        "",
        ("The tables above are measured facts. They do not identify a root cause. Candidate "
         "hypotheses must be tested with a controlled change and the same analysis pipeline."),
    ])
    if summary["aggregation"]["priorities"]:
        top = summary["aggregation"]["priorities"][0]
        lines.append(
            f"- HYPOTHESIS: the `{top['group']}` group may deserve investigation because it "
            f"contains {top['count']} observed errors; test whether a targeted intervention "
            "changes this group without degrading the paired control group."
        )
    lines.extend([
        ("- NEXT: rerun this analysis on a paired checkpoint with identical dataset, split, "
         "score threshold, and matching configuration; compare sample-level regressions and "
         "corrections."),
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    lines.append("")
    return "\n".join(lines)


def _plots(result: AnalysisResult, output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    counts = result.summary["counts"]
    written = []
    labels = ["miss", "duplicate", "localization", "background FP"]
    values = [counts["misses"], counts["duplicates"], counts["localization_errors"],
              counts["background_false_positives"]]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#b74d4d", "#c28b2c", "#b8792f", "#596a8a"])
    axis.set_ylabel("Count")
    axis.set_title("Primary-matching error categories")
    figure.tight_layout()
    path = output / "error_categories.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path))

    size_rows = result.summary["aggregation"]["by_size_bin"]
    names = list(size_rows)
    recalls = [size_rows[name]["recall"] or 0.0 for name in names]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(names, recalls, color="#3f6d5c")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recall")
    axis.set_title("Recall by diagnostic object-size bin")
    figure.tight_layout()
    path = output / "recall_by_size_bin.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path))

    tp = [prediction["score"] for record in result.samples
          for prediction in record["predictions"]
          if prediction["outcome"]["status"] == "true_positive"]
    fp = [prediction["score"] for record in result.samples
          for prediction in record["predictions"]
          if prediction["outcome"]["status"] == "false_positive"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist([tp, fp], bins=20, range=(0, 1), label=["TP", "FP"], color=["#3f6d5c", "#b74d4d"])
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Predictions")
    axis.set_title("Confidence distribution for matched and false predictions")
    axis.legend()
    figure.tight_layout()
    path = output / "confidence_distribution.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(str(path))
    return written


def write_outputs(result: AnalysisResult, output: str | Path, *, plots: bool = False) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result.summary, indent=2) + "\n")
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in result.samples))
    (output / "errors.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in result.errors))
    (output / "report.md").write_text(render_report(result))
    paths = {
        "summary": str(output / "summary.json"),
        "samples": str(output / "samples.jsonl"),
        "errors": str(output / "errors.jsonl"),
        "report": str(output / "report.md"),
    }
    if plots:
        paths["plots"] = _plots(result, output / "plots")
    return paths
