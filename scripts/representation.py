"""Run representation diagnostics with a local checkpoint and local dataset."""
import argparse
import json
from pathlib import Path

import yaml

from tinydet.analysis.representation import study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/representation.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    print(json.dumps(study(args.checkpoint, args.data, args.output, config), indent=2))


if __name__ == "__main__":
    main()
