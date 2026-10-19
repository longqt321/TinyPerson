import json
from pathlib import Path

import pytest

from tinydet.evaluation.tinyperson_benchmark import evaluate_tinyperson


def _write_fixture(root: Path):
    annotations = {
        "type": "instance",
        "images": [
            {"id": 1, "file_name": "labeled_images/a.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "labeled_images/b.jpg", "width": 100, "height": 100},
            {"id": 3, "file_name": "labeled_images/c.jpg", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 1, "name": "sea_person"},
            {"id": 2, "name": "earth_person"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10],
             "ignore": False, "uncertain": False, "in_dense_image": False},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10],
             "ignore": True, "uncertain": False, "in_dense_image": False},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [20, 20, 4, 4],
             "ignore": False, "uncertain": False, "in_dense_image": False},
            {"id": 4, "image_id": 3, "category_id": 1, "bbox": [1, 1, 5, 5],
             "ignore": False, "uncertain": False, "in_dense_image": True},
        ],
    }
    predictions = {"images": [
        {"path": "/data/test/a.jpg", "boxes": [
            [10, 10, 20, 20], [50, 50, 60, 60]],
         "scores": [0.99, 0.80], "classes": [0, 0]},
        {"path": "/data/test/b.jpg", "boxes": [[20, 20, 24, 24]],
         "scores": [0.90], "classes": [0]},
        {"path": "/data/test/c.jpg", "boxes": [[1, 1, 6, 6]],
         "scores": [0.95], "classes": [0]},
    ]}
    annotation_path, prediction_path = root / "test.json", root / "predictions.json"
    annotation_path.write_text(json.dumps(annotations))
    prediction_path.write_text(json.dumps(predictions))
    return annotation_path, prediction_path


def test_tinyperson_protocol_gates_dense_and_ignore_regions(tmp_path):
    annotations, predictions = _write_fixture(tmp_path)
    result = evaluate_tinyperson(annotations, predictions)

    assert result["protocol"]["source_images"] == 3
    assert result["protocol"]["dense_images_excluded"] == 1
    assert result["protocol"]["images_evaluated"] == 2
    assert result["all"]["AP@0.50"] == pytest.approx(1.0)
    assert result["tiny1"]["AP@0.50"] == pytest.approx(1.0)
    assert result["tiny2"]["AP@0.50"] == pytest.approx(1.0)
    assert result["small"]["AP@0.50"] is None
    assert result["all"]["positive_ground_truths"] == 2


def test_tinyperson_evaluator_rejects_partial_prediction_split(tmp_path):
    annotations, predictions = _write_fixture(tmp_path)
    payload = json.loads(predictions.read_text())
    payload["images"].pop()
    predictions.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Missing predictions"):
        evaluate_tinyperson(annotations, predictions)


def test_tinyperson_evaluator_rejects_unknown_prediction_image(tmp_path):
    annotations, predictions = _write_fixture(tmp_path)
    payload = json.loads(predictions.read_text())
    payload["images"][0]["path"] = "/data/test/not_in_annotations.jpg"
    predictions.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="absent from official annotation JSON"):
        evaluate_tinyperson(annotations, predictions)
