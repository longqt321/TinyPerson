"""GPU complexity & latency profiler measuring both Pre-Fuse (train) and Post-Fuse (deploy) graphs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from tinydet.models.ultralytics import register_ultralytics_modules
from tinydet.utils.experiment import write_json


def _time_forward_ms(net: torch.nn.Module, sample: torch.Tensor, warmup: int, iterations: int) -> tuple[float, float]:
    net.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = net(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(iterations):
            _ = net(sample)
        end_ev.record()
        torch.cuda.synchronize()
        elapsed_ms = float(start_ev.elapsed_time(end_ev)) / iterations
        peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0**2)
    return elapsed_ms, peak_mb


def benchmark(run: Path, iterations: int = 100, warmup: int = 15) -> dict[str, Any]:
    register_ultralytics_modules()
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires a CUDA GPU")
    config = yaml.safe_load((run / "config.yaml").read_text())
    checkpoint = run / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        checkpoint = run / "training" / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")

    model = YOLO(str(checkpoint))
    size = int(config["train"]["imgsz"])
    use_half = bool(config["train"].get("amp", True))

    raw_net = cast(Any, copy.deepcopy(model.model)).to("cuda").eval()
    params_train = sum(p.numel() for p in raw_net.parameters())
    gflops_train = round(float(get_flops(raw_net, size)), 3)
    gflops_640_train = round(float(get_flops(raw_net, 640)), 3)

    fused_net = cast(Any, copy.deepcopy(model.model)).to("cuda").eval()
    fused_net.fuse()
    params_fused = sum(p.numel() for p in fused_net.parameters())
    gflops_fused = round(float(get_flops(fused_net, size)), 3)
    gflops_640_fused = round(float(get_flops(fused_net, 640)), 3)

    dtype = torch.float16 if use_half else torch.float32
    if use_half:
        fused_net = fused_net.half()
    sample = torch.zeros(1, 3, size, size, device="cuda", dtype=dtype)
    latency_ms, gpu_memory_mb = _time_forward_ms(fused_net, sample, warmup=warmup, iterations=iterations)

    result = {
        "parameters": params_fused,
        "parameters_pre_fuse": params_train,
        "parameters_fused": params_fused,
        "gflops": gflops_fused,
        "gflops_pre_fuse": gflops_train,
        "gflops_640_fused": gflops_640_fused,
        "gflops_640_pre_fuse": gflops_640_train,
        "gpu_memory_mb": round(gpu_memory_mb, 2),
        "latency_ms": round(latency_ms, 3),
        "fps": round(1000.0 / max(latency_ms, 1e-6), 2),
        "batch": 1,
        "imgsz": size,
        "half": use_half,
        "gpu": torch.cuda.get_device_name(0),
        "iterations": iterations,
        "warmup": warmup,
    }
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    write_json(run / "metrics" / "benchmark.json", result)
    return result
