"""Checkpoint transfer helpers for related detection architectures."""

from torch import nn


def load_p2_relocated_weights(target: nn.Module, source: nn.Module) -> dict[str, int]:
    """Transfer matching weights when a P2 module moves the Detect head by one layer."""
    from ultralytics.utils import LOGGER

    source_state = source.float().state_dict()
    target_state = target.state_dict()
    source_head = f"model.{len(source.model) - 1}."
    target_head = f"model.{len(target.model) - 1}."
    transferred, remapped, skipped = {}, 0, 0
    for key, value in source_state.items():
        candidates = [key]
        if key.startswith(source_head):
            candidates.append(target_head + key[len(source_head):])
        for index, target_key in enumerate(candidates):
            if target_key in target_state and target_state[target_key].shape == value.shape:
                transferred[target_key] = value
                remapped += index > 0
                break
        else:
            skipped += 1
    if not any(key.startswith(target_head) for key in transferred):
        raise RuntimeError("No relocated detection-head weights matched the P2 model")
    if len(transferred) < 0.8 * len(source_state):
        raise RuntimeError(
            f"Only {len(transferred)}/{len(source_state)} checkpoint tensors matched P2 model"
        )
    target.load_state_dict(transferred, strict=False)
    missing = len(target_state) - len(transferred)
    LOGGER.info(
        "Transferred %d/%d checkpoint tensors (%d layer-index remapped, %d source skipped, "
        "%d target tensors left initialized)",
        len(transferred), len(source_state), remapped, skipped, missing,
    )
    return {
        "transferred": len(transferred), "remapped": remapped,
        "skipped": skipped, "missing": missing,
    }
