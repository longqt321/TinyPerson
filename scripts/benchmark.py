import argparse

from tinydet.evaluation.benchmark import benchmark
from tinydet.utils.experiment import resolve_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark existing run on Modal GPU")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    print(benchmark(resolve_run("/mnt/runs", args.run)))
