# Tiny Person Detection Research

Nghiên cứu phát hiện người rất nhỏ bằng YOLO26n trên TinyPerson. Nhánh hiện tại kiểm tra hai câu hỏi theo thứ tự: feature P2 có giúp giữ chi tiết của vật thể nhỏ không, và module Multi-Scale Residual (MSR) có cải thiện feature P2 không.

## Hướng nghiên cứu hiện tại

P2 có stride 4 và được tạo ở neck sau khi hợp nhất feature backbone độ phân giải cao với ngữ nghĩa top-down từ stride 8, 16 và 32. MSR được đặt riêng trên P2 ngay trước Detect. Module giữ nguyên shape và chỉ thay đổi nhánh P2; P3/P4/P5, backbone, loss và label assignment giữ nguyên.

Thí nghiệm đối chứng nằm ở `configs/experiment/ablation_msr_p2.yaml`:

| Run | Model | So sánh dùng để trả lời |
|---|---|---|
| E0 | A2: P2 + SPPF | Baseline chính |
| E1 | A2 + MSR trên P2 | E1 với E0: MSR có ích khi giữ SPPF? |
| E2 | A3: P2, bỏ SPPF | Baseline phụ |
| E3 | A3 + MSR trên P2 | E3 với E2: MSR có ích khi bỏ SPPF? |

Không dùng E1 với E2 để kết luận riêng về MSR vì hai model khác SPPF. Biểu đồ W&B được chia sẻ chạy tới epoch 50, nhưng ảnh biểu đồ không cung cấp số metric chính xác. Artifact của bốn run này chưa có trong workspace; không dùng ước lượng đọc từ ảnh làm kết quả định lượng.

| Run | Run ID trên biểu đồ | AP@[.50:.95] | AP50 | Precision | Recall | AP-small |
|---|---|---:|---:|---:|---:|---:|
| E0 A2 baseline | `20261020-034105-8326468-seed42` | Chưa lấy CSV | Chưa lấy CSV | Chưa lấy CSV | Chưa lấy CSV | Chưa tính |
| E1 A2 + MSR | `20261020-034106-8326468-seed42` | Chưa lấy CSV | Chưa lấy CSV | Chưa lấy CSV | Chưa lấy CSV | Chưa tính |
| E2 A3 bỏ SPPF | `20261020-034109-8326468-seed42` | Chưa lấy CSV | Chưa lấy CSV | Chưa lấy CSV | Chưa lấy CSV | Chưa tính |
| E3 A3 + MSR | `20261020-034109-8326468-seed42` | Chưa lấy CSV | Chưa lấy CSV | Chưa lấy CSV | Chưa lấy CSV | Chưa tính |

Tải artifacts để cập nhật bảng bằng số liệu gốc:

```fish
python tools/download_results.py msr_p2_e0_a2_baseline/20261020-034105-8326468-seed42 --include metrics checkpoints
python tools/download_results.py msr_p2_e1_a2_msr/20261020-034106-8326468-seed42 --include metrics checkpoints
python tools/download_results.py msr_p2_e2_a3_no_sppf/20261020-034109-8326468-seed42 --include metrics checkpoints
python tools/download_results.py msr_p2_e3_a3_no_sppf_msr/20261020-034109-8326468-seed42 --include metrics checkpoints
```

Sau khi tải, lấy metric tốt nhất từ `metrics/val.json` và metric theo epoch từ `metrics/train.csv`; tính AP-small bằng evaluator ở [TINYPERSON_BENCHMARK.md](docs/TINYPERSON_BENCHMARK.md). Không dùng metric của epoch cuối thay cho checkpoint `best.pt`.

Các cấu hình và tài liệu của nhánh thử nghiệm cũ đã được dọn khỏi luồng nghiên cứu hiện tại. Kết quả so sánh YOLO26n trước đây được giữ trong [archive](docs/archive/YOLO26n_ablation_2026-09.md); không gộp chúng với kết quả MSR.

## Cấu trúc liên quan

- `src/tinydet/modules/msr.py`: module MSR P2 và nạp trọng số baseline với remap Detect.
- `configs/model/yolo26n_tiny_A*.yaml`: baseline, ablation P2/SPPF và hai model có MSR.
- `configs/experiment/ablation_msr_p2.yaml`: bốn run đối chứng với chung dataset, seed và hyperparameter.
- `src/tinydet/training/`, `src/tinydet/evaluation/`: train, lưu artifact, validation và benchmark.
- `modal/app.py`: Modal entrypoints cho train, compare, evaluate, benchmark và phân tích.
- `scripts/`: smoke test, xác minh graph, chuẩn bị dữ liệu và metric TinyPerson/COCO small.
- `tools/`: upload dữ liệu, tải run và tổng hợp kết quả.
- `docs/`: mô tả kiến trúc MSR và giao thức dữ liệu/đánh giá.

## Môi trường và kiểm tra local

Yêu cầu Python 3.12. Môi trường local dùng PyTorch CPU; train chính chạy trên Modal L4.

```bash
uv sync --extra cpu --group dev
uv run --extra cpu pytest
uv run --extra cpu ruff check .
```

Xác minh model MSR parse được, feature shape và Detect strides:

```bash
uv run --extra cpu python scripts/validate_tiny_ablation.py \
  configs/model/yolo26n_tiny_A2_p2_msr.yaml \
  configs/model/yolo26n_tiny_A3_p2_no_sppf_msr.yaml --imgsz 640
```

Chạy smoke test CPU qua criterion YOLO thật, backward, optimizer step, checkpoint save/load và predict:

```bash
uv run --extra cpu python scripts/smoke_msr.py
```

Kiểm tra này không thay thế AMP CUDA hoặc train trên dữ liệu thật. Chạy toàn bộ test và lint sau thay đổi mã nguồn; ghi rõ GPU test đã thực hiện hay chưa.

## Dataset

Experiment MSR dùng `/mnt/data/tinyperson_official_v1/data.yaml` trong Modal volume `tiny-person-data`. Dataset không được lưu trong repo. Trước mỗi compare, Modal chạy preflight trên ảnh và annotation gốc trước bước resize YOLO.

Artifact preflight gần nhất ghi nhận 746 ảnh: 662 train và 84 validation; 14,825 và 3,608 đối tượng; không phát hiện ảnh lỗi, thiếu label hay bbox vượt biên. Kích thước COCO khớp ảnh raster. Ở `imgsz=1280`, median căn bậc hai diện tích bbox là 8.94 px trên train và 10.04 px trên validation. Các con số mô tả derivative hiện dùng; kiểm tra `diagnostics/dataset_preflight_summary.json` của run để xác nhận signature của lần chạy cụ thể.

## Chạy experiment trên Modal

Đăng nhập Modal và cấu hình Secret `wandb-secret` nếu cần W&B. Đảm bảo đã upload dataset đúng volume/path. Từ repo root và venv đã kích hoạt:

```bash
modal run --detach modal/app.py::compare \
  --config configs/experiment/ablation_msr_p2.yaml \
  --parallel
```

`--detach` giữ app chạy nếu terminal ngắt. `--parallel` chạy tối đa bốn model cùng lúc theo giới hạn trong `modal/app.py`; compare vẫn chờ train, validation và benchmark hoàn tất trước khi kết thúc. Smoke 5 epoch:

```bash
modal run --detach modal/app.py::compare \
  --config configs/experiment/ablation_msr_p2.yaml \
  --smoke --parallel
```

Trạng thái run và artifacts lưu trong Modal volume `tiny-person-runs`, dưới `/mnt/runs/<model>/<run-id>/`. Mỗi run có config, metadata, môi trường, checkpoint, metrics và evaluation. W&B là log bổ sung, không thay thế artifacts trong volume.

- Tiếp tục danh sách từ một model trở đi: thêm `--start-at msr_p2_e1_a2_msr` với tên model phù hợp. Lệnh này **train lại** từ model đó, không resume epoch từ `last.pt`.
- Tái sử dụng kết quả hoàn tất đúng cấu hình: thêm `--reuse`. Chỉ run `COMPLETED` có `best.pt` mới được reuse; `--reuse` không tiếp tục train run bị dở.
- Nếu terminal báo mất kết nối, trước tiên kiểm tra run/checkpoint trong volume. Lỗi client không tự chứng minh GPU worker đã dừng.

Tải kết quả sau khi xác định model và run ID:

```fish
python tools/download_results.py MODEL/RUN_ID --include metadata.json config.yaml environment.txt metrics checkpoints evaluation
```

## Đánh giá

Dùng validation để chọn checkpoint. So sánh AP@[.50:.95], AP50, AP75, precision và recall; báo cáo thêm AP nhỏ theo evaluator đã xác minh. Dataset chỉ có validation derivative thì gọi kết quả là validation, không gọi là test.

So sánh MSR theo từng cặp E1/E0 và E3/E2. Nếu chênh lệch nhỏ, lặp seed 42, 43, 44 cho cả baseline và MSR, báo cáo mean/std. Đo latency và memory trên cùng L4, resolution, batch size; tách riêng model latency khỏi preprocessing/postprocessing.

Giao thức benchmark chính thức của TinyPerson, gồm ignore/dense annotations và ảnh gốc, nằm trong [TINYPERSON_BENCHMARK.md](docs/TINYPERSON_BENCHMARK.md). Không dùng official test để chọn model hoặc hyperparameter.

## Trạng thái kết luận

MSR đã được nối trực tiếp trong model YAML, parse qua namespace Ultralytics, và chạy smoke test local trên CPU. Đồ thị huấn luyện được chia sẻ cho thấy metric các model còn sát nhau. Hiện chưa đủ bằng chứng để tuyên bố MSR cải thiện tiny-object detection; cần xác nhận số liệu checkpoint tốt nhất, AP nhỏ và độ ổn định qua seed.
