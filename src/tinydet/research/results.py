"""Collect run artifacts and report paired experiment results."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import yaml

REPORT_METRICS = (
    "AP",
    "AP50",
    "AP75",
    "precision",
    "recall",
    "APsmall",
    "APtiny",
    "latency_ms",
    "gpu_memory_mb",
    "peak_vram_gb",
    "parameters",
)
CSV_COLUMNS = (
    "experiment",
    "run_id",
    "seed",
    "status",
    "metrics_source",
    "git_commit",
    "git_dirty",
    "gpu",
    "epochs",
    *REPORT_METRICS,
)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _epochs(run: Path) -> int | None:
    results = run / "training" / "results.csv"
    try:
        with results.open(newline="") as stream:
            return max(sum(1 for _ in csv.reader(stream)) - 1, 0)
    except OSError:
        return None


def collect_runs(root: str | Path) -> list[dict]:
    """Read every downloaded run under ``root/<experiment>/<run-id>``."""
    rows = []
    for metadata_file in sorted(Path(root).glob("*/*/metadata.json")):
        run = metadata_file.parent
        metadata = _read_json(metadata_file)
        val = _read_json(run / "metrics" / "val.json")
        benchmark = _read_json(run / "metrics" / "benchmark.json")
        rows.append({
            "experiment": metadata.get("experiment", run.parent.name),
            "run_id": metadata.get("run_id", run.name),
            "seed": metadata.get("seed"),
            "status": metadata.get("status"),
            "metrics_source": val.get("source"),
            "git_commit": metadata.get("git_commit"),
            "git_dirty": metadata.get("git_dirty"),
            "gpu": metadata.get("gpu"),
            "epochs": _epochs(run),
            **{name: val.get(name) for name in REPORT_METRICS[:7]},
            **{name: benchmark.get(name) for name in REPORT_METRICS[7:10]},
            "peak_vram_gb": metadata.get("peak_vram_gb"),
            "parameters": benchmark.get("parameters"),
        })
    return rows


def _rank(row: dict) -> tuple:
    """Prefer finished, independently evaluated, then newest artifacts."""
    return (
        row.get("AP") is not None,
        row.get("status") == "COMPLETED",
        row.get("metrics_source") == "evaluation",
        str(row.get("run_id", "")),
    )


def primary_runs(rows: list[dict]) -> list[dict]:
    """Choose one run per experiment and seed without cherry-picking seeds."""
    selected = {}
    for row in rows:
        key = (row.get("experiment"), row.get("seed"))
        if key not in selected or _rank(row) > _rank(selected[key]):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (str(row.get("experiment")), row.get("seed") or -1),
    )


def _mean_std(values: list[float]) -> tuple[float, float | None]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate primary runs by experiment across seeds."""
    grouped: dict[str, list[dict]] = {}
    for row in primary_runs(rows):
        grouped.setdefault(str(row["experiment"]), []).append(row)
    summary = []
    for experiment, runs in sorted(grouped.items()):
        result = {"experiment": experiment, "runs": len(runs), "seeds": sorted(
            row["seed"] for row in runs if row.get("seed") is not None
        )}
        for metric in REPORT_METRICS:
            values = [float(row[metric]) for row in runs if row.get(metric) is not None]
            result[metric] = _mean_std(values) if values else (None, None)
        summary.append(result)
    return summary


def paired_deltas(rows: list[dict], comparisons: list[dict]) -> list[dict]:
    """Compute treatment minus control per seed for explicit arm pairs."""
    indexed = {(row["experiment"], row.get("seed")): row for row in primary_runs(rows)}
    output = []
    for comparison in comparisons:
        control, treatment = comparison.get("control"), comparison.get("treatment")
        if not isinstance(control, str) or not isinstance(treatment, str):
            raise TypeError("Each comparison needs string control and treatment names")
        seeds = sorted({
            seed for experiment, seed in indexed
            if experiment in {control, treatment} and seed is not None
        })
        deltas = {metric: [] for metric in REPORT_METRICS}
        used = 0
        for seed in seeds:
            before = indexed.get((control, seed))
            after = indexed.get((treatment, seed))
            if before is None or after is None:
                continue
            used += 1
            for metric in REPORT_METRICS:
                if before.get(metric) is not None and after.get(metric) is not None:
                    deltas[metric].append(float(after[metric]) - float(before[metric]))
        output.append({
            "name": comparison.get("name", f"{treatment}-{control}"),
            "control": control,
            "treatment": treatment,
            "seeds": used,
            **{
                metric: _mean_std(values) if values else (None, None)
                for metric, values in deltas.items()
            },
        })
    return output


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format(value: tuple[float, float | None] | None) -> str:
    if value is None or value[0] is None:
        return "-"
    mean, std = value
    return f"{mean:.5f}" if std is None else f"{mean:.5f} ± {std:.5f}"


def render_markdown(rows: list[dict], comparisons: list[dict], root: str | Path) -> str:
    """Render a compact table suitable for a research log."""
    lines = [
        "# Experiment results",
        "",
        f"Artifact root: `{root}`",
        "",
        "## Primary runs across seeds",
        "",
        "| Experiment | Seeds | AP | AP50 | Precision | Recall | Latency ms | Peak VRAM GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summarize(rows):
        seeds = ",".join(str(seed) for seed in row["seeds"]) or "-"
        lines.append(
            f"| {row['experiment']} | {seeds} | {_format(row['AP'])} | "
            f"{_format(row['AP50'])} | {_format(row['precision'])} | "
            f"{_format(row['recall'])} | {_format(row['latency_ms'])} | "
            f"{_format(row['peak_vram_gb'])} |"
        )
    lines.extend(["", "## Paired deltas (treatment - control)", ""])
    if not comparisons:
        lines.append("No `comparisons` entries in the experiment config.")
    else:
        lines.extend([
            "| Comparison | Seeds | ΔAP | ΔAP50 | ΔPrecision | ΔRecall | ΔLatency ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in paired_deltas(rows, comparisons):
            lines.append(
                f"| {row['name']} | {row['seeds']} | {_format(row['AP'])} | "
                f"{_format(row['AP50'])} | {_format(row['precision'])} | "
                f"{_format(row['recall'])} | {_format(row['latency_ms'])} |"
            )
    return "\n".join(lines) + "\n"


def load_comparisons(path: str | Path) -> list[dict]:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise TypeError("Experiment config must be a YAML mapping")
    comparisons = raw.get("comparisons", [])
    if not isinstance(comparisons, list) or any(not isinstance(item, dict) for item in comparisons):
        raise ValueError("comparisons must be a list of mappings")
    return comparisons
