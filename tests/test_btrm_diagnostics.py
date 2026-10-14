"""BTRM detector diagnostics: prevent GT leakage, bad matching or mislabelled saturation."""
import numpy as np
import pytest
import torch

from tinydet.analysis.btrm_feature_diagnostics import (
    _feature_summary,
    _random_background,
    _scalar,
    ring_gt_occupancy_proxy,
)
from tinydet.analysis.btrm_prediction_diagnostics import (
    BINS,
    THRESHOLDS,
    _prepare,
    confidence_counts,
)


def _sample(image, gt, predictions):
    return {
        "image": image,
        "gt": [{"id": idx, "box": box, "bin": size}
               for idx, (box, size) in enumerate(gt)],
        "pred": [(np.array(box, dtype=float), score) for box, score in predictions],
    }


def test_confidence_threshold_matching_and_excluded_gt_ignored():
    small_box = [4.0, 4.0, 10.0, 10.0]
    large_box = [50.0, 50.0, 90.0, 90.0]
    samples = [_sample("a.jpg", [
        (small_box, "0-8"), (large_box, ">32"),
    ], [
        (small_box, .8),
        (large_box, .7),
        ([20.0, 20.0, 25.0, 25.0], .5),
    ])]
    prepared = _prepare(samples)
    low, hits = confidence_counts(samples, prepared, "0-8", .001)
    assert low["GT"] == 1
    assert low["TP"] == 1
    assert low["ignored_predictions"] == 1
    assert low["FP"] == 1
    assert low["precision"] == pytest.approx(.5)
    assert hits == {("a.jpg", 0)}
    high, hits_high = confidence_counts(samples, prepared, "0-8", .9)
    assert high["TP"] == 0
    assert high["FP"] == 0
    assert hits_high == set()


def test_repeated_prediction_cannot_double_match_gt():
    box = [0., 0., 6., 6.]
    samples = [_sample("b.jpg", [(box, "0-8")], [(box, .9), (box, .8)])]
    row, hits = confidence_counts(samples, _prepare(samples), "0-8", .001)
    assert row["TP"] == 1
    assert row["FP"] == 1
    assert len(hits) == 1


def test_background_sampler_excludes_all_annotations():
    import random

    anchor = [50., 50., 56., 56.]
    other = [20., 20., 40., 40.]
    rng = random.Random(42)
    for _ in range(25):
        result = _random_background(
            anchor, [anchor, other], 256, 256, rng, (0, 0, 256, 256)
        )
        if result is None:
            continue
        x, y, _ = result
        assert not (34 <= x <= 72 and 34 <= y <= 72)
        assert not (4 <= x <= 56 and 4 <= y <= 56)


def test_score_sampling_uses_bchw_without_rescaling_errors():
    tensor = torch.arange(36).reshape(1, 1, 6, 6)
    assert _scalar(tensor, 5.0, 5.0, 60) == pytest.approx(0)
    assert _scalar(tensor, 35.0, 45.0, 60) == pytest.approx(27)


def test_summary_exposes_the_actual_sat_fraction_and_gain():
    records = [
        {"group": "GT_center", "size_bin": "0-8",
         "z_contrast": 2., "z_btrm": 1., "g_contrast": .7, "g_btrm": .4},
        {"group": "BG", "size_bin": "0-8",
         "z_contrast": 1., "z_btrm": .1, "g_contrast": .4, "g_btrm": .03},
    ]
    per_image = [{
        "g_contrast_all_sat95": .5, "g_btrm_all_sat95": .1,
        "g_contrast_all_p50": .8, "g_btrm_all_p50": .7,
        "g_contrast_all_p95": 1., "g_btrm_all_p95": .9,
        "z_ratio_median": .6, "modulation_abs_max_error": 1e-5,
    }]
    report = _feature_summary(records, per_image, gain=-.2)
    assert report["effective_gain"] == -.2
    assert report["by_size"]["0-8"]["GT_center"]["z_btrm"]["median"] == 1
    assert report["all_feature_maps"]["g_btrm_all_sat95"]["mean"] == .1


def test_threshold_grid_and_size_bins_are_prespecified():
    assert BINS == ("0-8", "8-16", "16-32", ">32")
    assert THRESHOLDS[0] == .001
    assert .05 in THRESHOLDS


def test_ring_inside_gt_box_proxy_responds_to_object_size():
    # P3 cells are 8px wide when input is 128px and feature map is 16x16.
    # A 3px GT cannot occupy ring neighbor cell centers, a 30px GT can.
    assert ring_gt_occupancy_proxy(
        [63., 63., 66., 66.], 64.5, 64.5, 16, 16, 128
    ) == 0
    assert ring_gt_occupancy_proxy(
        [50., 50., 80., 80.], 65., 65., 16, 16, 128
    ) >= 4
