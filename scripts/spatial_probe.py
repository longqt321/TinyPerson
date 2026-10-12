"""Run spatial probe on frozen YOLO26 features (no detector training)."""
import argparse
import json
from pathlib import Path

import yaml

from tinydet.analysis.spatial_probe import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/analysis/spatial_probe.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text())
    print(json.dumps(run(args.checkpoint, args.data, args.output, config,
                         allowed_output_root=root / "outputs"), indent=2))


if __name__ == "__main__":
    main()
