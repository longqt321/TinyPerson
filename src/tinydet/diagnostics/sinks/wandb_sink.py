"""W&B sink that reuses the active training run."""


class WandbSink:
    def write(self, metrics: dict[str, float | None], step: int) -> bool:
        import wandb

        if wandb.run is None:
            return False
        payload = {
            f"msr/{key}": value for key, value in metrics.items()
            if value is not None
        }
        if payload:
            wandb.log(payload, step=step, commit=False)
        return True
