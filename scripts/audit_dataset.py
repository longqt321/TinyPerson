"""Audit original dataset images before detector preprocessing."""
import argparse
import json
import sys
from pathlib import Path

from tinydet.data.image_audit import audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect original raster dimensions, YOLO labels and resize risk."
    )
    parser.add_argument("--data", default="/mnt/data/tinyperson/data.yaml",
                        help="YOLO dataset YAML")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Square letterbox side used for the resize-risk simulation")
    parser.add_argument("--output", default="outputs/preflight/image-audit.json",
                        help="Full JSON report path")
    parser.add_argument("--strict-resolution", action="store_true",
                        help="Exit nonzero when at least half of labelled objects are below 4 px sqrt-area after resizing")
    args = parser.parse_args()

    report = audit_dataset(args.data, args.imgsz, args.output)
    summary = {
        key: report[key] for key in (
            "passed", "data_yaml", "dataset_root", "imgsz",
            "measurement_basis", "dataset_signature_metadata_sha256",
            "images_total", "decoded_images_total", "corrupt_images_total",
            "label_files_found_total", "annotation_rows_total",
            "parsed_objects_total", "splits", "severe_resolution_loss",
            "warnings", "errors", "report_path",
        )
    }
    # The full report stores per-image metadata in JSON; avoid dumping thousands
    # of records into the terminal.
    print(json.dumps(summary, indent=2))
    if not report["passed"]:
        return 2
    if args.strict_resolution and report["severe_resolution_loss"]:
        print(
            "ERROR: resize-risk threshold exceeded. Review the report and choose imgsz "
            "explicitly before training.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
