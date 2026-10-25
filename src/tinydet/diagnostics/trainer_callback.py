"""Ultralytics callback adapter for optional MSR diagnostics."""

# Sink and collector failures must be isolated from training by design.
# ruff: noqa: BLE001, S110

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from tinydet.diagnostics.collector import MSRCollector
from tinydet.diagnostics.config import DiagnosticsConfig
from tinydet.diagnostics.metrics import aggregation_rule
from tinydet.diagnostics.sinks.file_sink import JsonlSink
from tinydet.diagnostics.sinks.terminal_sink import TerminalSink
from tinydet.diagnostics.sinks.wandb_sink import WandbSink

LOGGER = logging.getLogger(__name__)


class _EpochAccumulator:
    def __init__(self):
        self.values = defaultdict(dict)

    def add(self, module_id: str, metrics: dict[str, float | None], batch_weight: int) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            state = self.values[module_id].setdefault(key, {"sum": 0.0, "weight": 0,
                                                             "min": None, "max": None})
            if key.rsplit("/", 1)[-1] == "min":
                state["min"] = value if state["min"] is None else min(state["min"], value)
            elif key.rsplit("/", 1)[-1] == "max":
                state["max"] = value if state["max"] is None else max(state["max"], value)
            else:
                state["sum"] += value * batch_weight
                state["weight"] += batch_weight

    def finish(self) -> dict[str, dict[str, float | None]]:
        output = {}
        for module_id, metrics in self.values.items():
            output[module_id] = {}
            for key, state in metrics.items():
                if key.rsplit("/", 1)[-1] == "min":
                    output[module_id][key] = state["min"]
                elif key.rsplit("/", 1)[-1] == "max":
                    output[module_id][key] = state["max"]
                else:
                    output[module_id][key] = state["sum"] / state["weight"] if state["weight"] else None
        return output


class MSRDiagnosticsCallback:
    """Collect periodic scalar samples and isolate each output sink failure."""

    def __init__(self, settings: dict, wandb_enabled: bool = False) -> None:
        self.config = DiagnosticsConfig.parse(settings)
        self.collector = MSRCollector(
            sampled_elements=self.config.sampled_elements,
            feature_metrics=self.config.feature_metrics,
            numerical_health=self.config.numerical_health,
            performance_metrics=self.config.performance_metrics,
            cuda_timing=self.config.cuda_timing,
        )
        self.wandb_enabled = wandb_enabled
        self.file_sink = None
        self.terminal_sink = TerminalSink()
        self.wandb_sink = WandbSink()
        self.failed_sinks: set[str] = set()
        self.epoch_metrics = _EpochAccumulator()
        self.pending_sink_metrics: dict[str, dict[str, float]] = defaultdict(dict)
        self.batch_index = 0
        self.current_step = 0
        self.collecting = False
        self.enabled = self.config.enabled

    def on_train_start(self, trainer) -> None:
        try:
            if not self.enabled or not self.collector.discover(trainer.model):
                self.enabled = False
                return
        except Exception as exc:
            self._fail("collector", exc)
            return
        if self.config.file_logging:
            try:
                run_root = Path(trainer.save_dir).parent
                self.file_sink = JsonlSink(
                    run_root / "diagnostics" / "msr_metrics.jsonl", self.config.flush_records
                )
            except Exception as exc:
                self._fail("file", exc)

    def on_train_epoch_start(self, trainer) -> None:
        self.batch_index = 0
        self.epoch_metrics = _EpochAccumulator()

    def on_train_batch_start(self, trainer) -> None:
        if not self.enabled:
            return
        try:
            self.batch_index += 1
            steps_per_epoch = max(len(trainer.train_loader), 1)
            self.current_step = trainer.epoch * steps_per_epoch + self.batch_index
            self.collecting = self.current_step % self.config.collect_interval_steps == 0
            gradient_step = (
                self.config.gradient_metrics
                and self.current_step % self.config.gradient_interval_steps == 0
            )
            if not (self.collecting or gradient_step):
                return
            self.collector.begin_step(
                self.current_step,
                collect=self.collecting,
                gradients=gradient_step,
                trainer=trainer,
                gate_metrics=self.config.gate_metrics,
                residual_metrics=self.config.residual_metrics,
            )
        except Exception as exc:
            self._fail("collector", exc)

    def on_train_batch_end(self, trainer) -> None:
        if not self.enabled or not (self.collecting or self.collector.gradient_handles):
            return
        start = time.perf_counter()
        try:
            values = self.collector.end_step()
            elapsed_ms = (time.perf_counter() - start) * 1000
            batch_weight = max(int(getattr(trainer, "batch_size", 1)), 1)
            if self.config.performance_metrics:
                for metrics in values.values():
                    metrics["performance/collection_host_ms"] = elapsed_ms
            records = self._records(trainer.epoch + 1, self.current_step, values, "periodic_sample")
            for module_id, record in records.items():
                self.epoch_metrics.add(module_id, record["metrics"], batch_weight)
            self._emit_samples(records)
        except Exception as exc:
            self._fail("collector", exc)
        finally:
            self.collecting = False

    def on_train_epoch_end(self, trainer) -> None:
        if not self.enabled:
            return
        try:
            epoch = trainer.epoch + 1
            values = self.epoch_metrics.finish()
            if self.config.gradient_metrics:
                for module_id, metrics in self.collector.epoch_parameter_snapshot().items():
                    values.setdefault(module_id, {}).update(metrics)
            if self.config.performance_metrics:
                for module_id, milliseconds in self.collector.finish_epoch_timing().items():
                    values.setdefault(module_id, {})["performance/forward_ms"] = milliseconds
            if not values:
                return
            records = self._records(epoch, self.current_step, values, "epoch_summary")
            if epoch % self.config.wandb_interval_epochs == 0 and self.wandb_enabled:
                flat_metrics = {
                    f"{module_id}/{key}": value
                    for module_id, record in records.items()
                    for key, value in record["metrics"].items()
                }
                self._sink("wandb", lambda: self.wandb_sink.write(flat_metrics, epoch))
            if self.config.epoch_summary:
                for record in records.values():
                    self._write_file(record)
                    self._write_terminal(record)
        except Exception as exc:
            self._fail("epoch", exc)
        finally:
            try:
                self.collector.end_step()
            except Exception as exc:
                self._fail("collector", exc)

    def on_train_end(self, _trainer=None) -> None:
        self.close()

    def _records(self, epoch: int, step: int, values: dict, aggregation: str) -> dict:
        result = {}
        for module_id, metrics in sorted(values.items()):
            if not metrics:
                continue
            block = self.collector.modules[module_id]
            result[module_id] = {
                "epoch": epoch,
                "global_step": step,
                "module": module_id,
                "fusion_mode": block.fusion.fusion_mode,
                "aggregation": aggregation,
                "timestamp": datetime.now(UTC).isoformat(),
                "sampled_elements": self.config.sampled_elements,
                "metrics": dict(sorted(metrics.items())),
            }
        if self.config.performance_metrics:
            global_metrics = self.pending_sink_metrics.pop("global", {})
            for module_id, record in result.items():
                record["metrics"].update(self.pending_sink_metrics.pop(module_id, {}))
                record["metrics"].update(global_metrics)
                sink_times = [value for key, value in record["metrics"].items()
                              if key.startswith("performance/") and key.endswith("_host_ms")]
                record["metrics"]["performance/total_host_overhead_ms"] = sum(sink_times)
                record["metrics"] = dict(sorted(record["metrics"].items()))
        if aggregation == "epoch_summary":
            for record in result.values():
                record["aggregation_rules"] = {
                    key: aggregation_rule(key) for key in sorted(record["metrics"])
                }
        return result

    def _emit_samples(self, records: dict) -> None:
        for record in records.values():
            if record["aggregation"] == "periodic_sample":
                step = record["global_step"]
                if step % self.config.file_interval_steps == 0:
                    self._write_file(record)
                if step % self.config.terminal_interval_steps == 0:
                    self._write_terminal(record)

    def _write_file(self, record: dict) -> None:
        if self.file_sink is not None:
            self._sink("file", lambda: self.file_sink.write(record), record["module"])

    def _write_terminal(self, record: dict) -> None:
        self._sink("terminal", lambda: self.terminal_sink.write(record), record["module"])

    def _sink(self, name: str, operation: Callable, module_id: str | None = None):
        if name in self.failed_sinks:
            return None
        started = time.perf_counter()
        try:
            return operation()
        except Exception as exc:
            self._fail(name, exc)
            return None
        finally:
            if self.config.performance_metrics:
                elapsed = (time.perf_counter() - started) * 1000
                target = module_id or "global"
                self.pending_sink_metrics[target][f"performance/{name}_host_ms"] = elapsed

    def _fail(self, name: str, error: Exception) -> None:
        if name not in self.failed_sinks:
            self.failed_sinks.add(name)
            try:
                LOGGER.warning("MSR %s diagnostics disabled after logging error: %s", name, error)
            except Exception:
                pass
        if name == "file" and self.file_sink is not None:
            try:
                self.file_sink.close()
            except Exception:
                pass
            self.file_sink = None
        if name == "collector":
            self.enabled = False
            try:
                self.collector.close()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self.collector.close()
        except Exception as exc:
            self._fail("collector", exc)
        if self.file_sink is not None:
            try:
                self.file_sink.close()
            except Exception as exc:
                self._fail("file", exc)
            self.file_sink = None
