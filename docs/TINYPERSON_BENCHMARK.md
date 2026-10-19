# TinyPerson benchmark protocol in this repository

## Why this is separate from native YOLO validation

The native Ultralytics validator consumes YOLO labels and reports COCO-style AP. A YOLO label file cannot retain TinyPerson's ignore, uncertain, in_dense_image, or source image metadata. Native validation remains useful for training/selection, but is not by itself the TinyPerson benchmark.

The reference TinyBenchmark protocol (see [annotation rules](https://github.com/ucas-vg/PointTinyBenchmark/blob/master/dataset/TinyPerson_annotation.md) and [reference evaluator](https://github.com/sxy1122/TinyBenchmark/blob/master/tiny_benchmark/MyPackage/tools/evaluate/evaluate_tiny.py)) specifies:

- sea_person and earth_person are one person class.
- Only normal boxes are positive ground truth.
- Images with more than 220 annotations / marked dense are excluded.
- ignore and uncertain boxes are neutral, not positives or negatives; the reference evaluator uses IoU matching and an IoD fallback for ignored regions.
- AP uses IoU thresholds 0.25, 0.50, 0.75, a maximum of 200 detections per image, and area ranges all, tiny, tiny1, tiny2, tiny3, small, reasonable, measured by square-root bbox area in original pixels.
- The reference code also reports miss rate (MR). The standalone evaluator added here reports AP only; MR/reference cross-validation remains a follow-up before publication-grade claims.

## 1. Prepare YOLO train/validation from official training data

Use the non-corner erase annotation and matching erased image archive. This preserves the reference training treatment for ignore/uncertain/logo regions and excludes dense images. Do not feed a corner JSON to this script.

\`\`\`bash
uv run --extra cpu python scripts/prepare_tinyperson_yolo.py \\
  --annotations /data/TinyPerson/erase_with_uncertain_dataset/annotations/tiny_set_train.json \\
  --image-root /data/TinyPerson/erase_with_uncertain_dataset/train \\
  --output /data/derived/tinyperson-yolo
\`\`\`

The script validates every image's raster dimensions against COCO metadata, refuses dense/corner data and remaining special annotations, maps both person categories to class 0, and writes a deterministic train/validation split grouped by the filename's inferred video/frame prefix. Inspect manifest.json, especially singleton_group_count: filenames that do not encode a video group fall back to singleton groups, so confirm grouping is sensible for the downloaded release. The official test split is never used for validation.

Upload the derived dataset to the Modal data volume and point the experiment YAML's data.config at its data.yaml location. Keep the original archives and JSON unchanged.

## 2. Predict on the official test split at source resolution

Download the chosen checkpoint from the Modal run volume and use the original official test images plus the JSON with dense images (tiny_set_test_with_dense.json). The predictor verifies every source raster against its declared COCO width/height and saves boxes in original-image coordinates.

\`\`\`bash
uv run --extra cpu python tools/download_results.py MODEL/RUN_ID
uv run --extra cpu python scripts/predict_tinyperson_test.py \\
  --checkpoint /path/to/best.pt \\
  --annotations /data/TinyPerson/annotations/tiny_set_test_with_dense.json \\
  --image-root /data/TinyPerson/test \\
  --output predictions/tinyperson-test-predictions.json \\
  --imgsz 1280
\`\`\`

Set image-root to the directory under which the JSON's file_name values resolve. If the test archive is under a different root, adjust that path; do not rename/recreate images to force a match.

## 3. Calculate TinyPerson AP

\`\`\`bash
uv run --extra cpu python scripts/evaluate_tinyperson_benchmark.py \\
  --annotations /data/TinyPerson/annotations/tiny_set_test_with_dense.json \\
  --predictions predictions/tinyperson-test-predictions.json \\
  --output metrics/tinyperson_test.json
\`\`\`

The evaluator rejects missing/extra/duplicate image predictions to prevent accidental evaluation on the repository's custom 70/20/10 split. It excludes dense images, combines person categories, treats ignore/uncertain regions as neutral, caps detections at 200 per image, and writes per-size AP at IoU 0.25/0.50/0.75. The output records source paths and protocol details.

## Important limitations

- This repository implementation is a standalone AP evaluator derived from the published protocol, not a vendored copy of the legacy TinyBenchmark stack. Before publication, cross-check its scores against the reference evaluator on a fixed prediction file and use the reference evaluator for official MR.
- The size ranges follow the reference evaluator's inclusive area bounds: all [1, 100000], tiny [1, 20], tiny1 [1, 8], tiny2 [8, 12], tiny3 [12, 20], small [20, 32], and reasonable [32, 100000], where each value is sqrt(box area) in original pixels.
- The predictor is a separate inference pass. Its preprocessing, confidence threshold, NMS, and checkpoint must be recorded and kept identical when comparing MSR variants. The official benchmark evaluator itself does not define the detector's inference settings.
- Do not select the best checkpoint or tune MSR using official test metrics. Use only the validation split derived from official training images for model selection.
