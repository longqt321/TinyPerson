"""Run frozen-feature diagnostics, writing only below the repository outputs root."""
import argparse
import json
from pathlib import Path

import yaml

from tinydet.analysis.representation import study, validate_output_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis/representation.yaml"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory inside the repository's outputs/ directory")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    safe_root = (repo_root / "outputs").resolve()
    output = validate_output_directory(safe_root, args.output)
    config = yaml.safe_load(args.config.read_text())
    print(json.dumps(study(args.checkpoint, args.data, output, config,
                           allowed_output_root=safe_root), indent=2))


if __name__ == "__main__":
    main()
