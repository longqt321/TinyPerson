"""CLI for independent BTRM math/numerical audit, no GPU or dataset required."""
import argparse
import json
from pathlib import Path

from tinydet.analysis.btrm_math_audit import run_math_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_math_audit()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "max_abs_error": report["max_abs_error"],
        "max_rel_error_nonzero": report["max_rel_error_nonzero"],
        "gate_checks": report["gate_checks"],
        "gradient_checks": report["gradient_checks"],
    }, indent=2))


if __name__ == "__main__":
    main()
