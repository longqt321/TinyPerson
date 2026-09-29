import argparse
from pathlib import Path

from tinydet.analysis.erf import analyze_erf
from tinydet.utils.experiment import resolve_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ERF gradient for one image on Modal GPU")
    parser.add_argument("--run", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=16)
    args = parser.parse_args()
    print(analyze_erf(resolve_run("/mnt/runs", args.run), args.image, args.layer))
