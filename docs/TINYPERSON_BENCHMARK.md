# Dữ liệu và benchmark TinyPerson

## Hai loại đánh giá

YOLO validation trên derivative dùng label YOLO và phù hợp để chọn checkpoint giữa các run. Nó không giữ đầy đủ metadata `ignore`, `uncertain`, `in_dense_image` của TinyPerson, nên không đồng nhất với benchmark TinyPerson chính thức.

Script `evaluate_tinyperson_benchmark.py` tính AP theo các size bins TinyPerson từ dự đoán trên ảnh gốc và annotation gốc. Script này là implementation trong repository, không phải bản evaluator chuẩn được vendored. Trước khi công bố kết quả chính thức, đối chiếu fixed prediction file với reference evaluator; metric miss rate (MR) cần reference evaluator.

## Chuẩn bị train/validation derivative

Dùng erase annotation train không phải corner/crop cùng ảnh tương ứng. Script từ chối ảnh dense và annotation đặc biệt còn sót, kiểm tra kích thước raster theo COCO, gộp `sea_person` và `earth_person` về class `person`, rồi tạo split train/val xác định theo nhóm tên file.

```bash
python scripts/prepare_tinyperson_yolo.py \
  --annotations /data/TinyPerson/erase_with_uncertain_dataset/annotations/tiny_set_train.json \
  --image-root /data/TinyPerson/erase_with_uncertain_dataset/train \
  --output /data/derived/tinyperson-yolo
```

Xem `manifest.json`, đặc biệt cách suy ra group từ filename và số `singleton_group_count`. Giữ nguyên dữ liệu gốc. Upload derivative vào versioned path trong Modal Volume:

```bash
python tools/upload_data.py data /data/derived/tinyperson-yolo \
  --remote /tinyperson_official_v1
modal run modal/app.py::compare \
  --config configs/experiment/ablation_msr_p2.yaml \
  --preflight-only --require-source-metadata
```

Config hiện tại dùng `/mnt/data/tinyperson_official_v1/data.yaml`. Preflight kiểm tra ảnh/label gốc, signature dữ liệu và COCO dimensions trước khi mở GPU. Preflight pass không thay thế kiểm tra chất lượng split.

## Đánh giá validation theo COCO AP-small

`evaluate_coco_small.py` dùng annotation COCO train gốc, `manifest.json` để lấy thành viên validation, tọa độ bbox ảnh gốc và evaluator `pycocotools`. Small theo COCO là diện tích nhỏ hơn $32^2$ pixel² trên ảnh gốc. Cài optional dependency và chạy sau khi tải artifacts:

```fish
modal volume get tiny-person-runs /MODEL/RUN_ID/evaluation-val outputs/MODEL/RUN_ID/evaluation-val
modal volume get tiny-person-data /tinyperson_official_v1/annotations/tiny_set_train.json data/tiny_set_train.json
modal volume get tiny-person-data /tinyperson_official_v1/manifest.json data/manifest.json
uv run --extra cpu --with pycocotools python scripts/evaluate_coco_small.py \
  --run outputs/MODEL/RUN_ID \
  --annotations data/tiny_set_train.json --manifest data/manifest.json
```

Metric này chưa được xác minh trên môi trường thiếu dataset và `pycocotools`; không ghi AP-small trước khi script chạy thành công.

## Official test

Chỉ đánh giá test sau khi đã cố định kiến trúc, checkpoint selection và inference parameters dựa trên validation. Dùng ảnh test gốc và `tiny_set_test_with_dense.json`:

```bash
python scripts/predict_tinyperson_test.py \
  --checkpoint /path/to/best.pt \
  --annotations /data/TinyPerson/annotations/tiny_set_test_with_dense.json \
  --image-root /data/TinyPerson/test \
  --output predictions/tinyperson-test-predictions.json \
  --imgsz 1280 --batch-size 16
python scripts/evaluate_tinyperson_benchmark.py \
  --annotations /data/TinyPerson/annotations/tiny_set_test_with_dense.json \
  --predictions predictions/tinyperson-test-predictions.json \
  --output metrics/tinyperson_test.json --workers 7
```

Giữ cùng checkpoint policy, imgsz, confidence, postprocessing và max detections khi so model. Evaluator loại dense images, gộp hai category person, xử lý ignore/uncertain như vùng neutral, giới hạn 200 detections/image và báo AP ở IoU 0.25/0.50/0.75 theo size bins. Không dùng official test để chọn checkpoint hoặc điều chỉnh MSR.
