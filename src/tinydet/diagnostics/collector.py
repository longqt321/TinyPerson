"""Low-frequency MSR scalar collection; sinks and trainer stay separate."""

from __future__ import annotations

import math
import time
from collections import defaultdict

import torch
from torch import Tensor, nn

from tinydet.diagnostics.metrics import sampled_summary
from tinydet.modules.msr import MSR


class MSRCollector:
    """Attach temporary hooks only on configured diagnostic batches."""

    def __init__(self, sampled_elements: int, feature_metrics: bool,
                 numerical_health: bool, performance_metrics: bool,
                 cuda_timing: bool = False) -> None:
        self.sampled_elements = sampled_elements
        self.feature_metrics = feature_metrics
        self.numerical_health = numerical_health
        self.performance_metrics = performance_metrics
        self.cuda_timing = cuda_timing
        self.modules: dict[str, MSR] = {}
        self.handles = []
        self.pending: dict[str, dict[str, Tensor | float | None]] = defaultdict(dict)
        self.gradient_handles = []
        self.gradient_pending: dict[str, dict[str, Tensor | float | None]] = defaultdict(dict)
        self.grad_groups: dict[str, dict[str, list[nn.Parameter]]] = {}
        self.performance_events: dict[str, list[tuple[object, object]]] = defaultdict(list)
        self.step = 0
        self.active = False

    def discover(self, model: nn.Module) -> bool:
        self.modules = {
            f"layer_{int(module.i)}": module
            for module in model.modules()
            if isinstance(module, MSR) and hasattr(module, "i")
        }
        if not self.modules:
            self.modules = {
                f"layer_{index}": module
                for index, module in enumerate(m for m in model.modules() if isinstance(m, MSR))
            }
        self.grad_groups = {name: _parameter_groups(module) for name, module in self.modules.items()}
        return bool(self.modules)

    def begin_step(self, step: int, collect: bool, gradients: bool, trainer=None,
                   gate_metrics: bool = True, residual_metrics: bool = True) -> None:
        self._remove_handles()
        self.pending.clear()
        self.gradient_pending.clear()
        for block in self.modules.values():
            block.fusion.diagnostics_enabled = False
            block.refinement.diagnostics_enabled = False
        self.step = step
        self.active = collect
        if not (collect or gradients):
            return
        for module_id, block in self.modules.items():
            block.fusion.diagnostics_enabled = collect and gate_metrics
            block.refinement.diagnostics_enabled = collect and residual_metrics
            block.refinement.sampled_elements = self.sampled_elements
            if collect:
                self._attach_forward_hooks(module_id, block)
                if self.performance_metrics and (
                    next(block.parameters()).device.type != "cuda" or self.cuda_timing
                ):
                    self._attach_timer(module_id, block)
            if gradients:
                self._attach_gradient_hooks(module_id, trainer)

    def _attach_forward_hooks(self, module_id: str, block: MSR) -> None:
        if self.feature_metrics:
            for scale in block.single_scales:
                scale_id = scale.kernel
                self.handles.append(scale.register_forward_hook(
                    lambda _module, _inputs, output, mid=module_id, sid=scale_id:
                    self._capture(mid, f"evidence/scale_{sid}", output)
                ))
            self.handles.append(block.modulation.register_forward_hook(
                lambda _module, _inputs, output, mid=module_id:
                self._capture(mid, "modulation/modulated", output)
            ))
        if self.feature_metrics or self.numerical_health:
            self.handles.append(block.modulation.gate.register_forward_hook(
                lambda _module, _inputs, output, mid=module_id:
                self._capture(mid, "modulation/gate", output, thresholds=(0.05, 0.95))
            ))
        if self.numerical_health:
            self.handles.append(block.register_forward_hook(
                lambda _module, inputs, output, mid=module_id:
                self._capture(mid, "output", output)
            ))

    def _attach_timer(self, module_id: str, block: MSR) -> None:
        if next(block.parameters()).device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            pair = [start, end]
            self.handles.append(block.register_forward_pre_hook(
                lambda _module, _inputs, event=start: event.record()
            ))
            self.handles.append(block.register_forward_hook(
                lambda _module, _inputs, _output, event=end, events=pair, mid=module_id:
                (event.record(), self.performance_events[mid].append(tuple(events)))
            ))
        else:
            start = [0.0]
            self.handles.append(block.register_forward_pre_hook(
                lambda _module, _inputs, token=start: token.__setitem__(0, time.perf_counter())
            ))
            self.handles.append(block.register_forward_hook(
                lambda _module, _inputs, _output, mid=module_id, token=start:
                self.performance_events[mid].append((token[0], time.perf_counter()))
            ))

    def _attach_gradient_hooks(self, module_id: str, trainer) -> None:
        scaler = getattr(trainer, "scaler", None)
        try:
            scale = float(scaler.get_scale())
        except (AttributeError, RuntimeError, TypeError):
            scale = 1.0
        for group, parameters in self.grad_groups[module_id].items():
            state = {
                "squares": [], "finite": [], "seen": 0, "expected": len(parameters),
                "scale": scale, "parameter_norm": _group_parameter_norm(parameters),
            }
            self.gradient_pending[module_id][group] = state
            for parameter in parameters:
                def collect(gradient: Tensor, target=state):
                    grad = gradient.detach().float() / target["scale"]
                    target["squares"].append(grad.square().sum())
                    target["finite"].append(torch.isfinite(grad).all().to(torch.float32))
                    target["seen"] += 1
                    return gradient

                self.gradient_handles.append(parameter.register_hook(collect))

    def _capture(self, module_id: str, prefix: str, tensor: Tensor,
                 thresholds: tuple[float, float] | None = None) -> None:
        for name, value in sampled_summary(tensor, self.sampled_elements, thresholds=thresholds).items():
            if prefix == "output" and name == "finite_frac":
                key = "output/finite_frac"
            elif prefix == "modulation/gate" and name == "near_zero_frac":
                key = "modulation/near_zero_frac"
            elif prefix == "modulation/gate" and name == "near_one_frac":
                key = "modulation/near_one_frac"
            elif prefix == "modulation/gate" and name == "finite_frac":
                key = "modulation/gate_finite_frac"
            elif prefix == "modulation/gate" and name in {"mean", "std", "min", "max"}:
                key = f"modulation/gate_{name}"
            elif prefix == "modulation/modulated" and name in {"rms", "abs_mean"}:
                key = f"modulation/modulated_{name}"
            elif prefix.startswith("evidence/"):
                key = f"{prefix}/{name}"
            else:
                continue
            self.pending[module_id][key] = value.detach()

    def end_step(self) -> dict[str, dict[str, float | None]]:
        self._remove_handles()
        if not self.modules:
            self.active = False
            return {}
        records: dict[str, dict[str, float | None]] = {}
        tensor_values: list[Tensor] = []
        tensor_targets: list[tuple[str, str]] = []
        for module_id, block in self.modules.items():
            if self.active:
                self.pending[module_id].update(block.fusion.consume_diagnostics())
                self.pending[module_id].update(block.refinement.consume_diagnostics())
            for key, value in self.pending.get(module_id, {}).items():
                if isinstance(value, Tensor):
                    tensor_values.append(value.detach().reshape(()))
                    tensor_targets.append((module_id, key))
            for group, state in self.gradient_pending.get(module_id, {}).items():
                seen = state["seen"]
                squares = state["squares"]
                finite = state["finite"]
                prefix = f"grad/{group}"
                if seen:
                    tensor_values.extend((torch.stack(squares).sum().sqrt(), torch.stack(finite).mean()))
                    tensor_targets.extend(((module_id, f"{prefix}_grad_norm"),
                                           (module_id, f"{prefix}_grad_finite_frac")))
                else:
                    self.pending[module_id][f"{prefix}_grad_norm"] = None
                    self.pending[module_id][f"{prefix}_grad_finite_frac"] = None
                self.pending[module_id][f"{prefix}_grad_missing_frac"] = 1.0 - seen / state["expected"]
                if seen:
                    tensor_values.append(state["parameter_norm"])
                    tensor_targets.append((module_id, f"{prefix}_parameter_norm"))
                    self.pending[module_id][f"{prefix}_grad_rms_numel"] = sum(
                        parameter.numel() for parameter in self.grad_groups[module_id][group]
                    )
                    state["grad_numel"] = self.pending[module_id][f"{prefix}_grad_rms_numel"]
                    self.pending[module_id][f"{prefix}_grad_rms"] = None
                    self.pending[module_id][f"{prefix}_grad_to_parameter_ratio"] = None
        if tensor_values:
            devices = {value.device for value in tensor_values}
            if len(devices) == 1:
                copied = torch.stack([value.reshape(()) for value in tensor_values]).cpu().tolist()
                for (module_id, key), value in zip(tensor_targets, copied):
                    self.pending[module_id][key] = float(value)
        for module_id, values in self.pending.items():
            records[module_id] = {}
            for key, value in values.items():
                if isinstance(value, Tensor):
                    # One batched device transfer above handles tensors still present here.
                    continue
                if key.endswith("_grad_rms_numel"):
                    continue
                records[module_id][key] = _finite_or_none(value)
            for group, state in self.gradient_pending.get(module_id, {}).items():
                prefix = f"grad/{group}"
                norm = records[module_id].get(f"{prefix}_grad_norm")
                if norm is not None:
                    numel = state.get("grad_numel", 1)
                    parameter_norm = records[module_id].get(f"{prefix}_parameter_norm")
                    records[module_id][f"{prefix}_grad_rms"] = norm / math.sqrt(numel)
                    records[module_id][f"{prefix}_grad_to_parameter_ratio"] = (
                        norm / parameter_norm if parameter_norm else None
                    )
        self.active = False
        self.pending.clear()
        self.gradient_pending.clear()
        for block in self.modules.values():
            block.fusion.diagnostics_enabled = False
            block.refinement.diagnostics_enabled = False
        return records

    def epoch_parameter_snapshot(self) -> dict[str, dict[str, float | None]]:
        result = {}
        device_tensors: list[Tensor] = []
        targets: list[tuple[str, str]] = []
        finite_values: list[Tensor] = []
        finite_targets: list[tuple[str, str]] = []
        for module_id, groups in self.grad_groups.items():
            for group, parameters in groups.items():
                tensors = [parameter.detach().float() for parameter in parameters]
                device_tensors.append(torch.stack([tensor.square().sum() for tensor in tensors]).sum())
                targets.append((module_id, f"grad/{group}_parameter_norm"))
                count = sum(parameter.numel() for parameter in parameters)
                result.setdefault(module_id, {})[f"grad/{group}_parameter_rms"] = None
                result[module_id][f"grad/{group}_parameter_numel"] = count
                finite_values.append(torch.stack([
                    torch.isfinite(tensor).all().to(torch.float32) for tensor in tensors
                ]).mean())
                finite_targets.append((module_id, f"grad/{group}_parameter_finite_frac"))
        all_values = device_tensors + finite_values
        if all_values and len({value.device for value in all_values}) == 1:
            copied = torch.stack(all_values).cpu().tolist()
            for (module_id, key), value in zip(targets + finite_targets, copied):
                if key.endswith("_parameter_norm"):
                    value = math.sqrt(max(float(value), 0.0))
                    rms_key = key.replace("_parameter_norm", "_parameter_rms")
                    count = result[module_id].pop(rms_key.replace("_parameter_rms", "_parameter_numel"))
                    result[module_id][rms_key] = value / math.sqrt(count)
                result.setdefault(module_id, {})[key] = float(value)
        for module_id, values in result.items():
            for key, value in list(values.items()):
                if key.endswith("_parameter_numel"):
                    values.pop(key)
        return result

    def finish_epoch_timing(self) -> dict[str, float]:
        result = {}
        for module_id, pairs in self.performance_events.items():
            if not pairs:
                continue
            last_end = pairs[-1][1]
            if hasattr(last_end, "synchronize"):
                last_end.synchronize()
                elapsed = [start.elapsed_time(end) for start, end in pairs]
            else:
                elapsed = [(end - start) * 1000 for start, end in pairs]
            result[module_id] = sum(elapsed) / len(elapsed)
            pairs.clear()
        return result

    def close(self) -> None:
        self.end_step()
        for block in self.modules.values():
            block.fusion.diagnostics_enabled = False
            block.refinement.diagnostics_enabled = False

    def _remove_handles(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for handle in self.gradient_handles:
            handle.remove()
        self.gradient_handles.clear()


def _parameter_groups(block: MSR) -> dict[str, list[nn.Parameter]]:
    groups = {
        f"scale_{scale.kernel}/projection": [scale.projection[0].weight]
        for scale in block.single_scales
    }
    if block.fusion.fusion_mode == "adaptive":
        groups.update({
            f"fusion_scale_{scale_id}/gate": [parameter for name, parameter in gate.named_parameters()
                                                  if name.endswith("weight")]
            for scale_id, gate in zip(block.fusion.scale_ids, block.fusion.scale_gates)
        })
    elif block.fusion.fusion_mode == "sk_softmax":
        groups["fusion_sk/projection"] = [block.fusion.sk_reduce.weight, block.fusion.sk_expand.weight]
    groups["egm"] = [block.modulation.gate[0].weight]
    groups["refinement"] = [block.refinement.refine.weight]
    groups["residual/gamma"] = [block.gamma]
    return groups


def _group_parameter_norm(parameters: list[nn.Parameter]) -> Tensor:
    return torch.stack([parameter.detach().float().square().sum() for parameter in parameters]).sum().sqrt()


def _finite_or_none(value):
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None
