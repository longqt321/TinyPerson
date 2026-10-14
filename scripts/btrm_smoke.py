"""CPU preflight before renting 3 Modal L4 GPUs.

Checks three exact model variants, full YOLO forward and checkpoint roundtrip.
"""
import argparse
import tempfile
from pathlib import Path

import torch
import yaml

from tinydet.modules.btrm import GATE_INDEX, LocalBackgroundEvidence, install_feature_module
from tinydet.modules.ultralytics_adapter import ModuleDetectionModel
from tinydet.utils.comparison import load_comparison


def smoke(config: str, imgsz: int = 128) -> None:
    if imgsz < 64 or imgsz % 32:
        raise ValueError("Smoke imgsz must be >=64 and divisible by 32")
    experiments = load_comparison(config)
    for experiment in experiments:
        name = experiment["name"]
        options = experiment["model"].get("feature_module")
        if options is None:
            raise ValueError(f"Missing feature_module for {name}")
        cfg = yaml.safe_load(Path(experiment["model"]["config"]).read_text())
        model = ModuleDetectionModel(cfg, nc=1, ch=3, verbose=False)
        install_feature_module(model, options)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, imgsz, imgsz))
        gate = model.model[GATE_INDEX]
        is_module = isinstance(gate, LocalBackgroundEvidence)
        assert is_module == (options["mode"] != "none")
        assert model.model[18].f == 16
        assert model.model[24].f == [17, 20, 23]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "roundtrip.pt"
            torch.save(model, checkpoint)
            reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert isinstance(reloaded.model[GATE_INDEX], LocalBackgroundEvidence) == is_module
            reloaded.eval()
            with torch.no_grad():
                _ = reloaded(torch.randn(1, 3, imgsz, imgsz))
        print(f"PASS {name} mode={options['mode']} forward/checkpoint, gate={gate.__class__.__name__}", flush=True)
    print("PASS all BTRM arms", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/ablation_btrm.yaml")
    parser.add_argument("--imgsz", type=int, default=128)
    arguments = parser.parse_args()
    smoke(arguments.config, arguments.imgsz)
