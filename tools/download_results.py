import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download selected run artifacts")
    parser.add_argument("run", help="experiment/run-id")
    parser.add_argument("--include", nargs="+", choices=["metadata.json", "config.yaml", "environment.txt", "metrics", "diagnostics", "checkpoints"], default=["metadata.json", "config.yaml", "environment.txt", "metrics"])
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    parts = Path(args.run).parts
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        parser.error("run must be experiment/run-id")
    destination = args.output.joinpath(*parts)
    destination.mkdir(parents=True, exist_ok=True)
    for item in args.include:
        subprocess.run(["modal", "volume", "get", "tiny-person-runs", f"/{args.run}/{item}", str(destination / item)], check=True)
