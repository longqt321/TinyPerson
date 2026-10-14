"""BTRM detector diagnostics: prevent GT leakage, bad matching or mislabelled saturation."""
import csv
import json

import numpy as np
import pytest
import torch
import yaml

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
         "z_contrast": 2., "z_btrm": 1., "g_contrast": .7, "g_btrm": .4,
         "ring_inside_gt_cell_centers": 0},
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


def test_report_assembly_from_three_mock_frozen_runs(tmp_path):
    """A complete CPU-only report test; nothing runs on GPUs or loads real checkpoints."""
    from tinydet.analysis.btrm_diagnostic_report import assemble_report

    root = tmp_path / "reports"
    root.mkdir()
    (root / "math_audit.json").write_text(json.dumps({
        "status": "PASS", "checks": 72, "max_abs_error": 0.00026,
        "max_rel_error_nonzero": 2.5,
    }))
    (root / "prediction_summary.json").write_text(json.dumps({
        "tiny_recall_A3_minus_A0_at_0.001": {
            "delta_TP": 2, "delta_FP": 5, "newly_matched_GT": 3,
            "previously_matched_GT_lost": 1,
        }
    }))
    rows = []
    for model in ("btrm_a0", "btrm_a2", "btrm_a3"):
        for bin_name in ("0-8", "8-16"):
            for threshold in (0.001, 0.05, 0.25):
                rows.append({
                    "model": model, "bin": bin_name,
                    "confidence_threshold": threshold,
                    "TP": 3, "FP": 10, "precision": .23,
                    "recall": .06, "false_positives_per_image": 2.5,
                })
    with (root / "precision_recall_thresholds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    refs = {}
    for name, mode, gain in (
        ("btrm_a0", "none", None),
        ("btrm_a2", "contrast", .11),
        ("btrm_a3", "btrm", -.07),
    ):
        run = tmp_path / name / "checkpoint-1"
        refs[name] = run
        run.mkdir(parents=True)
        (run / "metadata.json").write_text(json.dumps({"git_commit": "abc"}))
        (run / "config.yaml").write_text(yaml.safe_dump({
            "seed": 42, "data": {"config": "data.yaml"},
            "train": {"imgsz": 1280}, "hardware": {"gpu": "L4"},
            "model": {"config": "some-model.yaml", "feature_module": {"mode": mode}},
        }))
        dest = run / "diagnostics" / "btrm_mechanism" / "val"
        dest.mkdir(parents=True)
        field = {
            "mean": .1, "median": .1, "p99": .1, "count": 1, "p05": .1, "p95": .1
        }
        maps = {
            key: field for key in (
                "g_contrast_all_sat95", "g_btrm_all_sat95",
                "g_contrast_all_p50", "g_btrm_all_p50",
                "g_contrast_all_p95", "g_btrm_all_p95",
                "z_ratio_median", "modulation_abs_max_error",
            )
        }
        (dest / "summary.json").write_text(json.dumps({
            "run": str(run), "selected_images": 1,
            "effective_gain": gain,
            "all_feature_maps": maps,
            "by_size": {
                "0-8": {
                    label: {
                        **{k: field for k in ("z_contrast", "z_btrm",
                                             "g_contrast", "g_btrm")},
                        "ring_inside_gt_cell_centers": field,
                    }
                    for label in ("GT_center", "BG")
                },
            },
        }))
        samples = [{
            "image": "sample.jpg", "group": group, "size_bin": "0-8",
            "g_contrast": .2 if group == "BG" else .6,
            "g_btrm": .1 if group == "BG" else .5,
        } for group in ("GT_center", "BG")]
        with (dest / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
    result = assemble_report(refs, root, "val")
    assert result["status"] == "COMPLETE"
    assert "PASS" in (root / "mechanism_report.md").read_text()
    assert "Net TP: +2; net FP: +5" in (root / "mechanism_report.md").read_text()
    assert (root / "mechanism_summary.json").is_file()
    assert (root / "btrm_a3_gate_hist_0_8.png").is_file()
    assert (root / "pr_thresholds_0_8.png").is_file()


def test_frozen_feature_diagnostics_full_mock_forward(tmp_path, monkeypatch):
    """Exercise actual P3 hooks, preprocessing, GT sampling and gate parity on CPU."""
    import cv2
    from torch import nn

    from tinydet.analysis import btrm_feature_diagnostics as diagnostic
    from tinydet.modules.btrm import LocalBackgroundEvidence

    images = tmp_path / "dataset" / "images" / "val"
    labels = tmp_path / "dataset" / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    path = images / "synthetic.jpg"
    pixels = np.random.default_rng(11).integers(
        0, 255, size=(128, 128, 3), dtype=np.uint8
    )
    assert cv2.imwrite(str(path), pixels)
    (labels / "synthetic.txt").write_text(
        "0 0.48 0.48 0.52 0.48 0.52 0.52 0.48 0.52\n"
    )

    run = tmp_path / "btrm_a3" / "fake-run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "best.pt").write_bytes(b"test placeholder")
    (run / "config.yaml").write_text(yaml.safe_dump({
        "data": {"config": str(tmp_path / "unused.yaml")},
        "train": {"imgsz": 128},
        "model": {"feature_module": {"mode": "btrm"}},
    }))
    monkeypatch.setattr(diagnostic, "dataset_images", lambda *_: [path])

    class FakeNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList(nn.Identity() for _ in range(18))
            self.model[16] = nn.Conv2d(3, 7, 3, stride=8, padding=1)
            self.model[17] = LocalBackgroundEvidence("btrm")
            with torch.no_grad():
                self.model[17].alpha_raw.fill_(0.42)

        def forward(self, x):
            return self.model[17](self.model[16](x))

    class FakeYOLO:
        def __init__(self, _checkpoint):
            self.model = FakeNetwork()

    monkeypatch.setattr(diagnostic, "YOLO", FakeYOLO)
    result = diagnostic.collect_features(
        run, max_images=1, split="val", device_override="cpu"
    )
    assert result["images"] == 1
    assert result["mode"] == "btrm"
    assert result["records"] >= 1
    assert result["all_feature_maps"]["modulation_abs_max_error"]["p99"] < 1e-5
    output = run / "diagnostics" / "btrm_mechanism" / "val"
    assert (output / "samples.csv").is_file()
    assert (output / "per_image.csv").is_file()
    assert (output / "summary.json").is_file()
