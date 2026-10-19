"""Evaluate saved YOLO test predictions with TinyPerson's ignore-aware protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinydet.evaluation.tinyperson_benchmark import evaluate_tinyperson


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Original TinyPerson COCO JSON, e.g. tiny_set_test_with_dense.json")
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Repository export predictions/test.json from a completed test evaluation")
    parser.add_argument("--output", type=Path, default=Path("metrics/tinyperson_test.json"))
    args = parser.parse_args()
    result = evaluate_tinyperson(args.annotations, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
