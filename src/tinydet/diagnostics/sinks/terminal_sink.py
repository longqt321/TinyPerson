"""Compact MSR summaries for training terminals."""


class TerminalSink:
    def write(self, record: dict) -> None:
        metrics = record["metrics"]
        module = record["module"]
        step = record["global_step"]
        epoch = record["epoch"]
        weights = []
        for key, value in sorted(metrics.items()):
            if key.startswith("fusion/scale_") and key.endswith("/weight_mean"):
                scale = key.split("/")[1].removeprefix("scale_")
                weights.append(f"k{scale}={_format(value)}")
        prefix = f"[MSR][epoch={epoch}][step={step}] {module} {record['fusion_mode']}"
        parts = []
        if weights:
            parts.append("weights=" + ",".join(weights))
            if "fusion/normalized_entropy" in metrics:
                parts.append(f"entropy={_format(metrics['fusion/normalized_entropy'])}")
        elif record["fusion_mode"] == "max":
            parts.append("weights=N/A")
        gate_mean = metrics.get("modulation/gate_mean")
        update_ratio = metrics.get("residual/update_input_ratio")
        if gate_mean is not None:
            parts.append(
                "mod_gate=" + _format(gate_mean)
                + " near0=" + _format(metrics.get("modulation/near_zero_frac"))
                + " near1=" + _format(metrics.get("modulation/near_one_frac"))
            )
        if "residual/gamma" in metrics:
            parts.append(
                "gamma=" + _format(metrics.get("residual/gamma"))
                + " update/input=" + _format(update_ratio)
            )
        if "output/finite_frac" in metrics:
            parts.append("finite=" + _format(metrics["output/finite_frac"]))
        print(prefix + (" " + " ".join(parts) if parts else ""), flush=True)


def _format(value):
    return "N/A" if value is None else f"{value:.4g}"
