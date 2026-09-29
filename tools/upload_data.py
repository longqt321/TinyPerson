import argparse
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset or pretrained model to Modal Volume")
    parser.add_argument("kind", choices=["data", "model"])
    parser.add_argument("source", type=Path)
    parser.add_argument("--remote", required=True, help="Path inside selected volume")
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f"Source does not exist: {args.source}")
    volume = "tiny-person-data" if args.kind == "data" else "tiny-person-models"
    subprocess.run(["modal", "volume", "put", volume, str(args.source), args.remote], check=True)
