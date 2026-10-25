"""Parse and smoke-test YOLO26 tiny-ablation model YAMLs on CPU."""

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from ultralytics.nn.tasks import DetectionModel

DEFAULT_MODELS = [
    Path("configs/model/yolo26n_tiny_A0_baseline.yaml"),
    Path("configs/model/yolo26n_tiny_A1_no_sppf.yaml"),
    Path("configs/model/yolo26n_tiny_A2_p2.yaml"),
    Path("configs/model/yolo26n_tiny_A3_p2_no_sppf.yaml"),
]


def validate(path: Path, imgsz: int) -> None:
    config = yaml.safe_load(path.read_text())
    if config.get("nc") != 1 or config.get("scale") != "n":
        raise ValueError(f"Expected nc=1 and scale=n: {path}")

    if "_msr" in path.stem or "_btrm" in path.stem:
        from tinydet.models.ultralytics import register_ultralytics_modules

        register_ultralytics_modules()
    model = DetectionModel(cfg=str(path), ch=3, verbose=False)
    model.eval()
    head = model.model[-1]
    expected = [4, 8, 16, 32] if "_A2_" in path.name or "_A3_" in path.name else [8, 16, 32]
    strides = [int(value) for value in head.stride.tolist()]
    if head.nc != 1 or strides != expected:
        raise ValueError(f"Unexpected detection head for {path}: nc={head.nc}, strides={strides}")

    feature_shapes = []
    hook = head.register_forward_pre_hook(
        lambda _module, args: feature_shapes.extend(tuple(feature.shape) for feature in args[0])
    )
    with torch.inference_mode():
        model(torch.zeros(1, 3, imgsz, imgsz))
    hook.remove()
    if len(feature_shapes) != len(expected):
        raise ValueError(f"Expected {len(expected)} detection inputs, got {feature_shapes}")

    no_sppf = "no_sppf" in path.name
    if no_sppf != isinstance(model.model[9], nn.Identity):
        raise ValueError(f"Unexpected layer 9 module in {path}: {type(model.model[9]).__name__}")

    layers, params, gradients, gflops = model.info(imgsz=imgsz, verbose=True)
    print(f"{path.name}: nc={head.nc}, strides={strides}, features={feature_shapes}, "
          f"layers={layers}, parameters={params}, gradients={gradients}, GFLOPs={gflops:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    for path in args.models:
        validate(path, args.imgsz)


if __name__ == "__main__":
    main()
