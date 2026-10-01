"""Local Heavy-Bottom probes, architecture statistics, and diagnostics."""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from tinydet.analysis.heavy_bottom import make_probe, resolve_stages, run_diagnostics
from tinydet.analysis.heavy_bottom_report import generate_report
from tinydet.evaluation.size_metrics import evaluate_size
from tinydet.utils.comparison import load_comparison


def architecture_stats(config: str):
    results = []
    for experiment in load_comparison(config):
        model = YOLO(experiment["model"]["config"]).model
        stages = resolve_stages(model)
        results.append({"name": experiment["name"], "model": experiment["model"]["config"],
                        "params": sum(p.numel() for p in model.parameters()),
                        "gflops_640": get_flops(model, imgsz=640),
                        "stages": {name: {"index": index, "blocks": len(model.model[index].m),
                                          "channels": model.model[index].cv2.conv.out_channels}
                                   for name, index in stages.items()}})
    return results


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--data", required=True)
    probe.add_argument("--output", required=True)
    probe.add_argument("--size", type=int, default=1280)
    probe.add_argument("--count", type=int, default=200)
    probe.add_argument("--seed", type=int, default=42)
    stats = commands.add_parser("stats")
    stats.add_argument("--config", default="configs/experiment/heavy_bottom.yaml")
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--checkpoint", required=True)
    diagnose.add_argument("--probe", required=True)
    diagnose.add_argument("--output", required=True)
    diagnose.add_argument("--max-instances", type=int)
    evaluate = commands.add_parser("evaluate-size")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--split", default="val")
    evaluate.add_argument("--size", type=int, default=1280)
    report = commands.add_parser("report")
    report.add_argument("--run", action="append", required=True, help="name=/path/to/run")
    report.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "probe":
        print(len(make_probe(Path(args.data), Path(args.output), args.size, args.count, args.seed)))
    elif args.command == "stats":
        print(json.dumps(architecture_stats(args.config), indent=2))
    elif args.command == "diagnose":
        print(run_diagnostics(Path(args.checkpoint), Path(args.probe), Path(args.output), args.max_instances))
    elif args.command == "evaluate-size":
        print(json.dumps(evaluate_size(Path(args.checkpoint), Path(args.data), args.split, args.size), indent=2))
    else:
        runs = dict(item.split("=", 1) for item in args.run)
        print(generate_report({name: Path(path) for name, path in runs.items()}, Path(args.output)))


if __name__ == "__main__":
    main()
