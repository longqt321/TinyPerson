"""Regression tests for the augmentation preset shape contract (L-025).

The ADR-018 preset used to hardcode `RandomSizedBBoxSafeCrop` at 1280x1280.
At any imgsz != 1280 the crop resized ~20% of the batch to 1280x1280 and the
DataLoader collate crashed with "stack expects each tensor to be equal size".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tinydet.augment import PRESETS, resolve_augmentations


def _crop_of(spec):
    crops = [t["transform"] for t in spec if t["transform"]["__class_fullname__"] == "RandomSizedBBoxSafeCrop"]
    assert len(crops) == 1
    return crops[0]


def test_preset_scales_crop_to_imgsz_320():
    spec = resolve_augmentations("tinyperson_b", 320)
    crop = _crop_of(spec)
    assert crop["height"] == 320 and crop["width"] == 320


def test_preset_scales_crop_to_imgsz_1280():
    spec = resolve_augmentations("tinyperson_b", 1280)
    crop = _crop_of(spec)
    assert crop["height"] == 1280 and crop["width"] == 1280


def test_unknown_preset_still_raises():
    with pytest.raises(ValueError, match="Unknown augmentation preset"):
        resolve_augmentations("nope", 320)


def test_explicit_list_with_wrong_crop_size_raises():
    bad = [
        {
            "__version__": "2.0.8",
            "transform": {"__class_fullname__": "RandomSizedBBoxSafeCrop", "p": 0.2, "height": 1280, "width": 1280},
        }
    ]
    with pytest.raises(ValueError, match="imgsz=320"):
        resolve_augmentations(bad, 320)


def test_explicit_list_with_matching_crop_size_passes():
    ok = [
        {
            "__version__": "2.0.8",
            "transform": {"__class_fullname__": "RandomSizedBBoxSafeCrop", "p": 0.2, "height": 640, "width": 640},
        }
    ]
    assert resolve_augmentations(ok, 640) == ok


def test_invalid_imgsz_raises():
    with pytest.raises(ValueError, match="imgsz"):
        resolve_augmentations("tinyperson_b", 0)


def test_real_pipeline_keeps_every_image_at_imgsz(tmp_path=None):
    """End-to-end: the actual YOLODataset transform stack must be shape-uniform.

    `tmp_path` accepts a pytest-provided directory; the scratch runner calls
    this with no argument, so a temp dir is created on demand.
    """
    import shutil
    import tempfile

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp(prefix="gs-aug-"))
        cleanup = True
    else:
        cleanup = False
    try:
        _run_pipeline(tmp_path)
    finally:
        if cleanup:
            shutil.rmtree(tmp_path, ignore_errors=True)


def _run_pipeline(root: Path) -> None:
    import cv2
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset

    root = Path(root) / "ds"
    (root / "images/train").mkdir(parents=True)
    (root / "labels/train").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(6):
        cv2.imwrite(str(root / f"images/train/{i}.jpg"), rng.integers(0, 255, (765, 1360, 3), dtype=np.uint8))
        (root / f"labels/train/{i}.txt").write_text("0 0.5 0.5 0.02 0.03\n")
    (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/train\nnames:\n  0: person\n")

    cfg = get_cfg(overrides={
        "data": str(root / "data.yaml"), "imgsz": 320, "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0,
        "degrees": 0.0, "translate": 0.05, "scale": 0.15, "shear": 0.0, "perspective": 0.0,
        "fliplr": 0.5, "flipud": 0.0, "hsv_h": 0.005, "hsv_s": 0.1, "hsv_v": 0.15,
        "augmentations": resolve_augmentations("tinyperson_b", 320),
    })
    ds = build_yolo_dataset(cfg, str(root / "images/train"), batch=4, data={"names": {0: "person"}, "nc": 1, "channels": 3}, mode="train", rect=False, stride=32)
    shapes = {ds[i % len(ds)]["img"].shape for i in range(60)}
    assert shapes == {(3, 320, 320)}, shapes


def test_presets_registry_is_factory_based():
    assert {"tinyperson_b", "tinyperson_safe"} <= set(PRESETS)
    for name, factory in PRESETS.items():
        assert callable(factory), name


def test_tinyperson_safe_scales_crop_and_drops_destructive_ops():
    for imgsz in (320, 640, 1280):
        spec = resolve_augmentations("tinyperson_safe", imgsz)
        crop = _crop_of(spec)
        assert crop["height"] == imgsz and crop["width"] == imgsz
        names = {item["transform"]["__class_fullname__"] for item in spec}
        assert names.isdisjoint({"Downscale", "MotionBlur", "ImageCompression", "CoarseDropout"})
