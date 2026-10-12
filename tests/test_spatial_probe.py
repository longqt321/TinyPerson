"""Regression tests for sampling geometry, group splits and matching controls."""
import numpy as np
import pytest
import torch

from tinydet.analysis.spatial_probe import (
    matched_negative, paired_group_bootstrap, sample_feature, source_group, split_groups,
)


def test_sampler_constant_map_is_invariant():
    feature = torch.full((1, 4, 8, 8), 7.0)
    box = [20., 20., 40., 40.]
    for method in ("legacy_center", "center", "roi_align"):
        np.testing.assert_allclose(sample_feature(feature, box, 64, method), 7.)
    # Local average pool has zero padding: test away from borders.
    np.testing.assert_allclose(sample_feature(feature, box, 64, "local_mean_3x3"), 7.)


def test_sampler_roi_and_center_find_hotspot():
    feature = torch.zeros((1, 1, 4, 4))
    feature[0, 0, 1, 1] = 8.
    box = [16., 16., 32., 32.]
    assert sample_feature(feature, box, 64, "legacy_center")[0] == 8.
    assert sample_feature(feature, box, 64, "center")[0] > 0.
    assert sample_feature(feature, box, 64, "roi_align")[0] > 0.


def test_matched_negative_has_same_size_and_no_object_overlap():
    rng = np.random.default_rng(12)
    target = {"box": [24., 24., 32., 36.]}
    candidate = matched_negative(target, [target], 64, rng)
    assert candidate is not None
    np.testing.assert_allclose(
        [candidate[2] - candidate[0], candidate[3] - candidate[1]], [8., 12.]
    )
    from tinydet.analysis.spatial_probe import overlaps
    assert overlaps(candidate, target["box"]) == 0


def test_video_group_parsing_and_explicit_failure():
    from pathlib import Path
    assert source_group(Path("bb_V0031_I0002680_jpg.rf.abcd.jpg"), "video") == "video:0031"
    with pytest.raises(ValueError, match="source video"):
        source_group(Path("unknown.jpg"), "video")
    assert source_group(Path("unknown.jpg"), "image").endswith("unknown.jpg")


def test_group_split_no_leakage_and_is_repeatable():
    rows = [{"group": "video:" + str(i), "label": y}
            for i in range(10) for y in (0, 1)]
    train, test = split_groups(rows, seed=42, train_fraction=0.6)
    assert not ({rows[i]["group"] for i in train} & {rows[i]["group"] for i in test})
    train_again, test_again = split_groups(rows, seed=42, train_fraction=0.6)
    np.testing.assert_array_equal(train, train_again)
    np.testing.assert_array_equal(test, test_again)


def test_bootstrap_identical_scores_has_zero_delta():
    rows = [{"group": "a", "label": 1}, {"group": "a", "label": 0},
            {"group": "b", "label": 1}, {"group": "b", "label": 0}]
    scores = np.array([.9, .1, .8, .2])
    low, high = paired_group_bootstrap(rows, scores, scores, 50, 0)
    assert low == 0. and high == 0.


def test_batched_sampler_matches_single_sample():
    from tinydet.analysis.spatial_probe import sample_features

    feature = torch.arange(64., dtype=torch.float32).reshape(1, 1, 8, 8)
    boxes = [[8., 8., 16., 16.], [24., 24., 40., 40.]]
    for method in ("legacy_center", "center", "local_mean_3x3", "roi_align"):
        batch = sample_features(feature, boxes, 64, method)
        singles = np.stack([sample_feature(feature, b, 64, method) for b in boxes])
        np.testing.assert_allclose(batch, singles, rtol=1e-5, atol=1e-5)


def test_checkpoint_resolution_prefers_finalized(tmp_path):
    from tinydet.analysis.spatial_probe import resolve_spatial_checkpoint

    finalized = tmp_path / "checkpoints" / "best.pt"
    raw = tmp_path / "training" / "weights" / "best.pt"
    finalized.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    finalized.write_bytes(b"finalized")
    raw.write_bytes(b"raw")
    assert resolve_spatial_checkpoint(tmp_path) == finalized


def test_checkpoint_resolution_accepts_raw_ultralytics_run(tmp_path):
    from tinydet.analysis.spatial_probe import resolve_spatial_checkpoint

    raw = tmp_path / "training" / "weights" / "best.pt"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw")
    assert resolve_spatial_checkpoint(tmp_path) == raw


def test_checkpoint_resolution_missing_lists_both_paths(tmp_path):
    from tinydet.analysis.spatial_probe import resolve_spatial_checkpoint

    with pytest.raises(FileNotFoundError) as exc:
        resolve_spatial_checkpoint(tmp_path)
    assert "checkpoints/best.pt" in str(exc.value)
    assert "training/weights/best.pt" in str(exc.value)
