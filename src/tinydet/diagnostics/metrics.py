"""Pure tensor summaries and aggregation rules for MSR observability."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sampled_summary(
    tensor: Tensor, sampled_elements: int, *, thresholds: tuple[float, float] | None = None
) -> dict[str, Tensor]:
    """Summarize deterministic strided samples; never return activation-sized tensors."""
    if sampled_elements <= 0:
        raise ValueError("sampled_elements must be positive")
    values = tensor.detach().reshape(-1)
    stride = max(math.ceil(values.numel() / sampled_elements), 1)
    values = values[::stride][:sampled_elements].float()
    finite = torch.isfinite(values)
    mask = finite.to(values.dtype)
    count = mask.sum().clamp_min(1)
    clean = torch.where(finite, values, 0.0)
    mean = clean.sum() / count
    centered = torch.where(finite, values - mean, 0.0)
    std = (centered.square().sum() / count).sqrt()
    magnitude = clean.abs()
    minimum = torch.where(finite, values, torch.inf).amin()
    maximum = torch.where(finite, values, -torch.inf).amax()
    summary = {
        "mean": mean,
        "std": std,
        "rms": (clean.square().sum() / count).sqrt(),
        "abs_mean": magnitude.sum() / count,
        "min": minimum,
        "max": maximum,
        "finite_frac": mask.mean(),
        "zero_frac": ((values == 0) & finite).to(values.dtype).mean(),
    }
    if thresholds is not None:
        low, high = thresholds
        summary["near_zero_frac"] = (((values <= low) & finite).to(values.dtype).sum() / count)
        summary["near_one_frac"] = (((values >= high) & finite).to(values.dtype).sum() / count)
    return {key: value.detach() for key, value in summary.items()}


def aggregation_rule(metric_name: str) -> str:
    """Describe epoch aggregation for a sampled metric."""
    leaf = metric_name.rsplit("/", 1)[-1]
    if leaf == "min":
        return "minimum_of_sampled_batch_minima"
    if leaf == "max":
        return "maximum_of_sampled_batch_maxima"
    if "/grad/" in metric_name and "_parameter_" in leaf:
        return "epoch_end_parameter_snapshot"
    return "batch_size_weighted_mean_of_periodic_batch_summaries"

