import argparse
import csv
import json
from pathlib import Path


def list_runs(root: Path) -> list[dict]:
    rows = []
    for metadata_file in sorted(root.glob("*/*/metadata.json")):
        meta = json.loads(metadata_file.read_text())
        metrics_file = metadata_file.parent / "metrics" / "val.json"
        metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}
        benchmark_file = metadata_file.parent / "metrics" / "benchmark.json"
        bench = json.loads(benchmark_file.read_text()) if benchmark_file.is_file() else {}
        rows.append({**{key: meta.get(key) for key in ("experiment", "run_id", "seed", "git_commit", "status", "gpu")}, **{key: metrics.get(key) for key in ("AP", "AP50", "AP75", "APtiny")}, **{key: bench.get(key) for key in ("parameters", "gflops", "latency_ms")}})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List downloaded experiment runs")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = list_runs(args.root)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["experiment", "run_id", "seed", "git_commit", "status", "gpu", "AP", "AP50", "AP75", "APtiny", "parameters", "gflops", "latency_ms"])
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(" ".join(f"{key}={value}" for key, value in row.items()))
