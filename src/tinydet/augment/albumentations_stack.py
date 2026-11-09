"""Albumentations augmentation presets for TinyPerson training.

Ultralytics accepts custom Albumentations transforms through the
`augmentations` hyperparameter (see `ultralytics.data.augment.build_transforms`,
which passes `getattr(hyp, "augmentations", None)` to `Albumentations(...)`).
The transforms are passed as serialized dicts (`A.from_dict` format) so they
stay YAML/JSON-serializable in run configs and checkpoints.

Preset `tinyperson_b` (ADR-018) models the TinyPerson acquisition domain:
downscaled resolution, camera shake, JPEG compression, lighting variation,
a bbox-safe zoom crop that magnifies tiny instances, and coarse occlusion.
All transforms keep the 1280x1280 letterboxed shape, so batching is unaffected.
"""

from __future__ import annotations

from typing import Any

PRESET_TINYPERSON_B: list[dict[str, Any]] = [
    {
        "__version__": "2.0.8",
        "transform": {"__class_fullname__": "Downscale", "p": 0.25, "scale_range": [0.4, 0.7]},
    },
    {
        "__version__": "2.0.8",
        "transform": {"__class_fullname__": "MotionBlur", "p": 0.15, "blur_limit": 7},
    },
    {
        "__version__": "2.0.8",
        "transform": {"__class_fullname__": "ImageCompression", "p": 0.10, "quality_range": [30, 70]},
    },
    {
        "__version__": "2.0.8",
        "transform": {"__class_fullname__": "RandomBrightnessContrast", "p": 0.20},
    },
    {
        "__version__": "2.0.8",
        "transform": {"__class_fullname__": "CLAHE", "p": 0.10},
    },
    {
        "__version__": "2.0.8",
        "transform": {
            "__class_fullname__": "RandomSizedBBoxSafeCrop",
            "p": 0.20,
            "height": 1280,
            "width": 1280,
            "erosion_rate": 0.0,
        },
    },
    {
        "__version__": "2.0.8",
        "transform": {"__class_fullname__": "CoarseDropout", "p": 0.10},
    },
]

PRESETS: dict[str, list[dict[str, Any]]] = {"tinyperson_b": PRESET_TINYPERSON_B}


def resolve_augmentations(spec: Any) -> list[dict[str, Any]] | None:
    """Resolve the `augmentations_preset` experiment key into serialized dicts.

    Args:
        spec: Either a preset name from `PRESETS`, or an explicit list of
            `A.from_dict`-compatible transform dicts (passed through unchanged).

    Returns:
        A list of serialized transform dicts, or `None` when no augmentation
        preset is configured (Ultralytics then uses its own defaults).
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"Unknown augmentation preset {spec!r}; available: {sorted(PRESETS)}")
        return PRESETS[spec]
    if isinstance(spec, list) and all(isinstance(item, dict) for item in spec):
        return spec
    raise TypeError(f"augmentations_preset must be a preset name or a list of transform dicts, got {type(spec)}")
