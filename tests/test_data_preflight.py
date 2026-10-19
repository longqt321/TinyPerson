"""Tests for raw-raster preflight; no YOLO resize path is used."""
import json
from pathlib import Path

from PIL import Image

from tinydet.data.image_audit import audit_dataset


def _make_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (28, 90, 160)).save(path)


def test_audit_uses_original_dimensions_and_simulates_letterbox(tmp_path):
    _make_image(tmp_path / "images/train/large.jpg", (1920, 1080))
    _make_image(tmp_path / "images/val/native.jpg", (1280, 720))
    (tmp_path / "labels/train").mkdir(parents=True)
    (tmp_path / "labels/val").mkdir(parents=True)
    # One normalized YOLO box and one normalized polygon.
    (tmp_path / "labels/train/large.txt").write_text(
        "0 0.5 0.5 0.01 0.02\n"
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
    )
    (tmp_path / "labels/val/native.txt").write_text("")

    dataset_yaml = tmp_path / "data.yaml"
    dataset_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: [person]\n"
    )
    output = tmp_path / "audit.json"
    report = audit_dataset(dataset_yaml, imgsz=1280, output_path=output)

    assert report["passed"]
    assert report["coco_dimension_status"] == "not_found"
    assert report["images_total"] == 2
    train = report["splits"]["train"]
    assert train["original_width_px"]["median"] == 1920
    assert train["original_height_px"]["median"] == 1080
    assert train["labels"]["parsed_objects"] == 2
    # Letterbox fit scale is computed from the stored 1920x1080 raster, not an
    # already-resized tensor; 1280 / 1920 = 2/3.
    large = next(row for row in report["per_image"] if row["path"].endswith("large.jpg"))
    assert large["width"] == 1920
    assert large["height"] == 1080
    assert abs(large["letterbox_scale_at_imgsz"] - 2 / 3) < 1e-5
    assert large["simulated_letterbox_width"] == 1280
    assert large["simulated_letterbox_height"] == 720
    assert report["splits"]["val"]["simulated_letterbox_scale"]["median"] == 1
    saved = json.loads(output.read_text())
    assert len(saved["per_image"]) == 2


def test_audit_reports_corrupt_original_image_as_error(tmp_path):
    image_path = tmp_path / "images/train/corrupt.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not an image")
    labels = tmp_path / "labels/train"
    labels.mkdir(parents=True)
    (labels / "corrupt.txt").write_text("")
    dataset_yaml = tmp_path / "data.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nnames: [person]\n")

    report = audit_dataset(dataset_yaml, imgsz=640)
    assert not report["passed"]
    assert report["corrupt_images_total"] == 1
    assert any("Cannot decode original image" in error for error in report["errors"])


def test_audit_flags_extreme_loss_after_resize_without_claiming_to_resize(tmp_path):
    _make_image(tmp_path / "images/train/uhd.jpg", (3840, 2160))
    label_dir = tmp_path / "labels/train"
    label_dir.mkdir(parents=True)
    # 6x6 pixels in the original raster; at imgsz=640 this becomes ~1x1.
    (label_dir / "uhd.txt").write_text("0 0.5 0.5 0.0015625 0.0027777778\n")
    dataset_yaml = tmp_path / "data.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nnames: [person]\n")

    report = audit_dataset(dataset_yaml, imgsz=640)
    assert report["passed"]
    assert report["severe_resolution_loss"]
    record = report["per_image"][0]
    assert record["width"] == 3840
    assert record["height"] == 2160
    assert record["simulated_letterbox_width"] == 640
    assert record["simulated_letterbox_height"] == 360


def test_audit_compares_on_disk_pixels_with_original_coco_dimensions(tmp_path):
    import json

    _make_image(tmp_path / "images/train/person.jpg", (100, 50))
    label_dir = tmp_path / "labels/train"
    label_dir.mkdir(parents=True)
    (label_dir / "person.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    (annotation_dir / "tiny_set_train.json").write_text(json.dumps({
        "images": [{
            "file_name": "labeled_images/person.jpg",
            "width": 200,
            "height": 100,
            "id": 1,
        }],
        "annotations": [],
        "categories": [],
    }))
    dataset_yaml = tmp_path / "data.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nnames: [person]\n")

    report = audit_dataset(dataset_yaml, imgsz=640)
    assert not report["passed"]
    assert report["coco_dimension_status"] == "mismatch"
    check = report["coco_dimension_checks"][0]
    assert check["matched_by_stem"] == 1
    assert check["dimension_mismatches"] == 1
    assert any("Original COCO metadata dimensions disagree" in error for error in report["errors"])
