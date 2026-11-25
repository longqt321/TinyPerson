"""Albumentations augmentation presets for TinyPerson training.

Ultralytics accepts custom Albumentations transforms through the
`augmentations` hyperparameter (see `ultralytics.data.augment.build_transforms`,
which passes `getattr(hyp, "augmentations", None)` to `Albumentations(...)`).
The transforms are passed as serialized dicts (`A.from_dict` format) so they
stay YAML/JSON-serializable in run configs and checkpoints.

Preset `tinyperson_b` (ADR-018) models the TinyPerson acquisition domain:
downscaled resolution, camera shake, JPEG compression, lighting variation,
a bbox-safe zoom crop that magnifies tiny instances, and coarse occlusion.
All transforms keep the letterboxed shape at the run's imgsz, so batching is
unaffected. Size-dependent transforms are generated from the run's imgsz by
`resolve_augmentations(spec, imgsz)`; hardcoding a size here broke every run
whose imgsz differed (L-025: a 1280 crop inside an imgsz-320 smoke run made
~20% of the batch come out 1280x1280 and the DataLoader collate crashed).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def preset_tinyperson_b(imgsz: int) -> list[dict[str, Any]]:
    """Build the ADR-018 preset at the run's imgsz.

    `RandomSizedBBoxSafeCrop` resizes the bbox-safe crop to (height, width), so
    that size must equal the letterboxed training shape or the DataLoader
    collate crashes on mixed image sizes (L-025).
    """
    return [
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
                "height": imgsz,
                "width": imgsz,
                "erosion_rate": 0.0,
            },
        },
        {
            "__version__": "2.0.8",
            "transform": {"__class_fullname__": "CoarseDropout", "p": 0.10},
        },
    ]


def preset_tinyperson_safe(imgsz: int) -> list[dict[str, Any]]:
    """Tiny-object-safe augmentation preset preserving sub-8px spatial fidelity.

    Drops destructive degradations from ``tinyperson_b`` (``Downscale``,
    ``MotionBlur(7)``, heavy JPEG ``ImageCompression``, and bbox-blind
    ``CoarseDropout``) that erase ``tiny1`` (2-8 px) instances while leaving
    their ground-truth boxes labeled as positives. Retains bbox-safe zoom
    magnification and mild radiometric perturbations for sea/beach glare.
    """
    return [
        {
            "__version__": "2.0.8",
            "transform": {
                "__class_fullname__": "RandomBrightnessContrast",
                "p": 0.20,
                "brightness_limit": [-0.15, 0.15],
                "contrast_limit": [-0.15, 0.15],
            },
        },
        {
            "__version__": "2.0.8",
            "transform": {
                "__class_fullname__": "CLAHE",
                "p": 0.10,
                "clip_limit": 2.0,
            },
        },
        {
            "__version__": "2.0.8",
            "transform": {
                "__class_fullname__": "RandomSizedBBoxSafeCrop",
                "p": 0.25,
                "height": imgsz,
                "width": imgsz,
                "erosion_rate": 0.0,
            },
        },
    ]


PRESETS: dict[str, Callable[[int], list[dict[str, Any]]]] = {
    "tinyperson_b": preset_tinyperson_b,
    "tinyperson_safe": preset_tinyperson_safe,
}


def _validate_shape_preserving(spec: list[dict[str, Any]], imgsz: int) -> None:
    """Reject explicit transform lists whose crop size differs from the run's imgsz."""
    for item in spec:
        transform = item.get("transform", {})
        height, width = transform.get("height"), transform.get("width")
        if (height is not None and height != imgsz) or (width is not None and width != imgsz):
            raise ValueError(
                f"augmentation {transform.get('__class_fullname__')} declares "
                f"height={height}, width={width} but the run trains at imgsz={imgsz}; "
                "a mismatched crop size changes the image shape mid-batch and crashes "
                "the DataLoader collate. Set height/width to imgsz."
            )


def resolve_augmentations(spec: Any, imgsz: int) -> list[dict[str, Any]] | None:
    """Resolve the `augmentations_preset` experiment key into serialized dicts.

    Args:
        spec: Either a preset name from `PRESETS`, or an explicit list of
            `A.from_dict`-compatible transform dicts (validated against imgsz).
        imgsz: The run's training imgsz; presets generate size-dependent
            transforms (e.g. RandomSizedBBoxSafeCrop) at this size.

    Returns:
        A list of serialized transform dicts, or `None` when no augmentation
        preset is configured (Ultralytics then uses its own defaults).
    """
    if spec is None:
        return None
    if type(imgsz) is not int or imgsz <= 0:
        raise ValueError(f"imgsz must be a positive int, got {imgsz!r}")
    if isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"Unknown augmentation preset {spec!r}; available: {sorted(PRESETS)}")
        return PRESETS[spec](imgsz)
    if isinstance(spec, list) and all(isinstance(item, dict) for item in spec):
        _validate_shape_preserving(spec, imgsz)
        return spec
    raise TypeError(f"augmentations_preset must be a preset name or a list of transform dicts, got {type(spec)}")
