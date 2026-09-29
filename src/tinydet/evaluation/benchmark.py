from pathlib import Path
from time import perf_counter

import torch
import yaml
from ultralytics import YOLO

from tinydet.utils.experiment import write_json


def benchmark(run: Path, iterations: int = 100, warmup: int = 10) -> dict:
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires Modal GPU")
    config = yaml.safe_load((run / "config.yaml").read_text())
    checkpoint = run / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    model = YOLO(str(checkpoint))
    size = config["train"]["imgsz"]
    batch = 1
    sample = torch.zeros(batch, 3, size, size, device="cuda")
    for _ in range(warmup):
        model.predict(sample, verbose=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for _ in range(iterations):
        model.predict(sample, verbose=False)
    torch.cuda.synchronize()
    latency = (perf_counter() - start) * 1000 / iterations
    result = {"parameters": sum(p.numel() for p in model.model.parameters()), "gflops": None, "gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2, "latency_ms": latency, "fps": 1000 / latency, "batch": batch, "imgsz": size, "gpu": torch.cuda.get_device_name(), "iterations": iterations, "warmup": warmup}
    write_json(run / "metrics" / "benchmark.json", result)
    return result
