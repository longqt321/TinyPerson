"""Validated, low-overhead diagnostics settings."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DiagnosticsConfig:
    enabled: bool = False
    collect_interval_steps: int = 100
    terminal_interval_steps: int = 100
    file_interval_steps: int = 100
    wandb_interval_epochs: int = 1
    gradient_interval_steps: int = 100
    epoch_summary: bool = True
    file_logging: bool = True
    gate_metrics: bool = True
    feature_metrics: bool = True
    residual_metrics: bool = True
    gradient_metrics: bool = False
    numerical_health: bool = True
    performance_metrics: bool = False
    cuda_timing: bool = False
    sampled_elements: int = 4096
    flush_records: int = 10

    @classmethod
    def parse(cls, value: dict[str, Any] | None) -> "DiagnosticsConfig":
        value = {} if value is None else value
        if not isinstance(value, dict):
            raise TypeError("diagnostics must be a mapping")
        unknown = set(value) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown diagnostics settings: {', '.join(sorted(unknown))}")
        parsed = cls(**value)
        for name in (
            "collect_interval_steps", "terminal_interval_steps", "file_interval_steps",
            "wandb_interval_epochs", "gradient_interval_steps", "sampled_elements", "flush_records",
        ):
            if type(getattr(parsed, name)) is not int or getattr(parsed, name) <= 0:
                raise ValueError(f"diagnostics.{name} must be a positive integer")
        for name in (
            "enabled", "epoch_summary", "file_logging", "gate_metrics", "feature_metrics",
            "residual_metrics", "gradient_metrics", "numerical_health", "performance_metrics",
            "cuda_timing",
        ):
            if type(getattr(parsed, name)) is not bool:
                raise ValueError(f"diagnostics.{name} must be a boolean")
        if parsed.terminal_interval_steps % parsed.collect_interval_steps:
            raise ValueError("terminal interval must be a multiple of collect_interval_steps")
        if parsed.file_interval_steps % parsed.collect_interval_steps:
            raise ValueError("file interval must be a multiple of collect_interval_steps")
        if parsed.gradient_interval_steps % parsed.collect_interval_steps:
            raise ValueError("gradient interval must be a multiple of collect_interval_steps")
        return parsed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
