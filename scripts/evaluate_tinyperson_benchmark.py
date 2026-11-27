"""Evaluate saved YOLO test predictions with TinyPerson's ignore-aware protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

import orjson

from tinydet.evaluation.tinyperson_benchmark import evaluate_tinyperson
from tinydet.utils.experiment import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Original TinyPerson COCO JSON, e.g. tiny_set_test_with_dense.json")
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Repository export predictions/test.json from a completed test evaluation")
    parser.add_argument("--output", type=Path, default=Path("metrics/tinyperson_test.json"))
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU processes across size ranges; 0 uses up to 7 cores")
    parser.add_argument("--class-mode", default="per_class",
                        choices=["per_class", "merge", "primary"],
                        help="How detector classes map onto TinyPerson categories. "
                             "A single-class model must use 'merge'.")
    args = parser.parse_args()
    result = evaluate_tinyperson(args.annotations, args.predictions, args.workers,
                                 class_mode=args.class_mode)
    write_json(args.output, result)
    print(orjson.dumps(result, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode())


if __name__ == "__main__":
    main()
