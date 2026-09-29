import argparse

from tinydet.evaluation.evaluator import evaluate
from tinydet.utils.experiment import resolve_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate existing run on Modal GPU")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    print(evaluate(resolve_run("/mnt/runs", args.run), args.split))
