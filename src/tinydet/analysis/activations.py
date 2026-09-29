from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from tinydet.analysis.frequency import save_frequency


def analyze_activations(run: Path, image_path: Path, layer: int = 16) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Activation analysis requires Modal GPU")
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
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).cuda()
    outputs = []
    handle = model.model[layer].register_forward_hook(lambda _module, _input, output: outputs.append(output.detach().cpu()))
    try:
        with torch.no_grad():
            model(tensor)
    finally:
        handle.remove()
    feature = outputs[0].squeeze(0).mean(dim=0).numpy()
    output = run / "diagnostics" / "activations" / f"layer{layer}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "feature.npy", feature)
    plt.imsave(output / "feature.png", feature, cmap="viridis")
    save_frequency(feature, run / "diagnostics" / "frequency" / f"layer{layer}")
    return output
