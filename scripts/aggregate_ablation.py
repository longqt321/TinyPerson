"""Aggregate component-removal ablation results into results.csv and summary.md.

Scope (user decision, 2026-11-05): the 5 component-removal ablations at seed 42,
compared against the full-GauSS seed-42 reference. No multi-seed statistics are
produced because only one seed was trained.

Inputs
    reports/eval/ablation/{run}.json        eval payload (frozen TOV tiled protocol)
    reports/eval/ablation/{run}.train.json  run metadata written by the runner
    reports/eval/gauss_v2_1_tinyperson_150e_tiled.json
                                            full-GauSS seed-42 reference evaluation

Outputs
    reports/ablation/results.csv  one row per run, reference row included
    reports/ablation/summary.md   deltas vs full GauSS, incomplete list

Usage
    uv run --extra cpu --group dev python scripts/aggregate_ablation.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

EVAL_DIR = Path("reports/eval/ablation")
OUT_DIR = Path("reports/ablation")

SEED = 42
ABLATIONS = (
    "no_high_band",
    "no_mid_band",
    "no_low_band",
    "no_router",
    "no_saliency",
)

# Full-GauSS seed-42 reference, trained and evaluated by the paper pipeline.
REFERENCE_NAME = "gauss_v2_1_tinyperson_wacv"
REFERENCE_EVAL = Path("reports/eval/gauss_v2_1_tinyperson_150e_tiled.json")

METRIC_ROWS = (
    ("AP50_all", "all", "AP@0.50"),
    ("AP50_tiny1", "tiny1", "AP@0.50"),
    ("AP50_tiny2", "tiny2", "AP@0.50"),
    ("AP50_tiny3", "tiny3", "AP@0.50"),
    ("AP50_small", "small", "AP@0.50"),
    ("MR50_tiny2", "tiny2", "MR@0.50"),
    ("AP75_all", "all", "AP@0.75"),
)


def _load_bench(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("summary", {}).get("benchmark")


def _load_train_meta(run: str) -> dict:
    path = EVAL_DIR / f"{run}.train.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _metric(bench: dict | None, bin_name: str, key: str) -> float | None:
    if not bench:
        return None
    value = bench.get(bin_name, {}).get(key)
    return None if value is None else round(100.0 * float(value), 4)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    ref_bench = _load_bench(REFERENCE_EVAL)
    ref_row: dict = {
        "run": REFERENCE_NAME,
        "ablation": "none (full GauSS, paper run)",
        "seed": SEED,
        "status": "REFERENCE" if ref_bench else "REFERENCE_MISSING",
        "training_seconds": "",
        "peak_vram_gb": "",
        "error": "" if ref_bench else f"{REFERENCE_EVAL} not found",
    }
    for out_name, bin_name, key in METRIC_ROWS:
        ref_row[out_name] = _metric(ref_bench, bin_name, key)
    rows.append(ref_row)

    for ablation in ABLATIONS:
        run = f"abl_gauss_{ablation}_seed{SEED}"
        bench = _load_bench(EVAL_DIR / f"{run}.json")
        meta = _load_train_meta(run)
        row: dict = {
            "run": run,
            "ablation": ablation,
            "seed": SEED,
            "status": "TRAINED_EVALUATED" if bench else "INCOMPLETE",
            "training_seconds": meta.get("training_seconds", ""),
            "peak_vram_gb": meta.get("peak_vram_gb", ""),
            "error": meta.get("error", ""),
        }
        for out_name, bin_name, key in METRIC_ROWS:
            row[out_name] = _metric(bench, bin_name, key)
        rows.append(row)

    csv_path = OUT_DIR / "results.csv"
    fieldnames = ["run", "ablation", "seed", "status", "training_seconds",
                  "peak_vram_gb", "error", *[name for name, _, _ in METRIC_ROWS]]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ----- summary.md -----
    metric_names = [name for name, _, _ in METRIC_ROWS]
    lines = [
        "# Component-removal ablation summary",
        "",
        (
            "Protocol: frozen TOV tiled evaluation (tile 640x512, overlap 100, "
            "global NMS IoU 0.50, top-2000 pre-NMS, max 1000 detections/image, "
            "IoD > 0.50 ignore/uncertain)."
        ),
        "",
        (
            f"Seed: {SEED} for every run, including the reference. "
            "Single seed only: no variance estimate is available."
        ),
        "",
        (
            f"Reference: `{REFERENCE_NAME}` (full GauSS, seed {SEED}), trained by the "
            "paper pipeline with a config identical to the ablation configs apart from "
            "`name`/`seed`."
        ),
        "",
        "## Absolute metrics (AP/MR in %)",
        "",
        "| Run | Ablation | " + " | ".join(metric_names) + " |",
        "|" + "---|" * (len(metric_names) + 2),
    ]
    for row in rows:
        cells = [_fmt(row[name]) for name in metric_names]
        lines.append(f"| {row['run']} | {row['ablation']} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Delta vs full GauSS (percentage points)")
    lines.append("")
    lines.append("Negative = the removed component helped. "
                 "`MR50_tiny2` is a miss rate, so a positive delta there means worse.")
    lines.append("")
    lines.append("| Ablation | " + " | ".join(metric_names) + " |")
    lines.append("|" + "---|" * (len(metric_names) + 1))
    for row in rows[1:]:
        if row["status"] != "TRAINED_EVALUATED" or ref_row["status"] != "REFERENCE":
            lines.append(f"| {row['ablation']} | incomplete |"
                         + "---|" * len(metric_names))
            continue
        cells = []
        for name in metric_names:
            if row[name] is None or ref_row[name] is None:
                cells.append("-")
            else:
                cells.append(f"{row[name] - ref_row[name]:+.2f}")
        lines.append(f"| {row['ablation']} | " + " | ".join(cells) + " |")
    lines.append("")

    incomplete = [r["run"] for r in rows if r["status"] not in
                  ("TRAINED_EVALUATED", "REFERENCE")]
    lines.append("## Incomplete / not-run")
    lines.append("")
    if incomplete:
        for run in incomplete:
            row = next(r for r in rows if r["run"] == run)
            reason = row.get("error") or row["status"]
            lines.append(f"- `{run}`: {reason}")
    else:
        lines.append("None - all planned runs trained and evaluated.")
    lines.append("")

    lines.append("## Scope and limitations")
    lines.append("")
    lines.append("- Only the 5 component-removal ablations were trained, at a single "
                 "seed. No baseline model and no multi-seed runs were trained.")
    lines.append("- Because n = 1 per configuration, no confidence interval can be "
                 "reported. Differences below run-to-run seed noise cannot be "
                 "distinguished from real effects.")
    lines.append("- The reference run was produced by the paper pipeline; it was not "
                 "retrained for this study. Its config matches the ablation configs "
                 "apart from `name` and `seed`.")
    lines.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path} and {OUT_DIR / 'summary.md'}")
    print(f"complete: {len(rows) - len(incomplete)}/{len(rows)}")


if __name__ == "__main__":
    main()
