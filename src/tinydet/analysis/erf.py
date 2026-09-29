from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO


def analyze_erf(run: Path, image_path: Path, layer: int = 16) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("ERF analysis requires Modal GPU")
    if not image_path.is_file():
        raise ValueError(f"Image does not exist: {image_path}")
    config = yaml.safe_load((run / "config.yaml").read_text())
    checkpoint = run / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    model = YOLO(str(checkpoint)).model.cuda().eval()
    if not 0 <= layer < len(model.model):
        raise ValueError(f"Invalid layer index: {layer}")
    size = config["train"]["imgsz"]
    pixels = np.asarray(Image.open(image_path).convert("RGB").resize((size, size)), dtype=np.float32) / 255
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).cuda().requires_grad_()
    activations = []
    handle = model.model[layer].register_forward_hook(lambda _module, _input, output: activations.append(output))
    try:
        model(tensor)
        feature = activations[0]
        feature[0, :, feature.shape[-2] // 2, feature.shape[-1] // 2].mean().backward()
    finally:
        handle.remove()
    gradient = tensor.grad.detach().abs().mean(dim=1).squeeze().cpu().numpy()
    output = run / "diagnostics" / "erf"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f"layer{layer}.npy", gradient)
    plt.imsave(output / f"layer{layer}.png", gradient, cmap="magma")
    return output
