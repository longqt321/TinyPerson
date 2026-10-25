# Tiny Person Detection Research

Nghiên cứu phát hiện người rất nhỏ bằng YOLO26n trên TinyPerson. Nhánh hiện tại kiểm tra hai câu hỏi theo thứ tự: feature P2 có giúp giữ chi tiết của vật thể nhỏ không, và module Multi-Scale Residual (MSR) có cải thiện feature P2 không.

Repo này là **research project**, không phải application project. Ưu tiên là câu
hỏi khoa học, kiểm soát biến, reproducibility, tốc độ vòng lặp thí nghiệm và
bảng kết quả. Bảo mật ứng dụng, nghiệp vụ sản phẩm và test pyramid đầy đủ không
phải mục tiêu. Đọc [Research practices](docs/RESEARCH_PRACTICES.md) trước khi
thay đổi protocol hoặc kết quả.

## Hướng nghiên cứu hiện tại

P2 có stride 4 và được tạo ở neck sau khi hợp nhất feature backbone độ phân giải cao với ngữ nghĩa top-down từ stride 8, 16 và 32. MSR được đặt riêng trên P2 ngay trước Detect. Module giữ nguyên shape và chỉ thay đổi nhánh P2; P3/P4/P5, backbone, loss và label assignment giữ nguyên.

Thí nghiệm đối chứng nằm ở `configs/experiment/ablation_msr_p2.yaml`:

| Run | Model | So sánh dùng để trả lời |
|---|---|---|
| E0 | A2: P2 + SPPF | Baseline chính |
| E1 | A2 + MSR trên P2 | E1 với E0: MSR có ích khi giữ SPPF? |
| E2 | A3: P2, bỏ SPPF | Baseline phụ |
| E3 | A3 + MSR trên P2 | E3 với E2: MSR có ích khi bỏ SPPF? |

Không dùng E1 với E2 để kết luận riêng về MSR vì hai model khác SPPF. Kết quả Modal, gồm metric best checkpoint và các lượt chạy từ tối 19/10 đến trưa 20/10, nằm trong [báo cáo kết quả](docs/modal_results_2026-10-20.md).

| Run | Run ID trên biểu đồ | AP@[.50:.95] | AP50 | Precision | Recall | AP-small |
|---|---|---:|---:|---:|---:|---:|
| E0 A2 baseline | `20261020-034105-8326468-seed42` | 0.05912 | 0.19689 | 0.35815 | 0.25249 | Chưa tính |
| E1 A2 + MSR | `20261020-034106-8326468-seed42` | 0.05975 | 0.19155 | 0.33292 | 0.27217 | Chưa tính |
| E2 A3 bỏ SPPF | `20261020-034109-8326468-seed42` | 0.05591 | 0.17992 | 0.33326 | 0.26053 | Chưa tính |
| E3 A3 + MSR | `20261020-034109-8326468-seed42` | 0.05294 | 0.17576 | 0.29808 | 0.25277 | Chưa tính |

Metric là validation trên best checkpoint. Tải artifact gốc để kiểm tra checkpoint/epoch:

```fish
python tools/download_results.py msr_p2_e0_a2_baseline/20261020-034105-8326468-seed42 --include metrics checkpoints
python tools/download_results.py msr_p2_e1_a2_msr/20261020-034106-8326468-seed42 --include metrics checkpoints
python tools/download_results.py msr_p2_e2_a3_no_sppf/20261020-034109-8326468-seed42 --include metrics checkpoints
python tools/download_results.py msr_p2_e3_a3_no_sppf_msr/20261020-034109-8326468-seed42 --include metrics checkpoints
```

Sau khi tải, lấy metric tốt nhất từ `metrics/val.json` và metric theo epoch từ `training/results.csv`; tính AP-small bằng evaluator ở [TINYPERSON_BENCHMARK.md](docs/TINYPERSON_BENCHMARK.md). Không dùng metric của epoch cuối thay cho checkpoint `best.pt`.

Các cấu hình và tài liệu của nhánh thử nghiệm cũ đã được dọn khỏi luồng nghiên cứu hiện tại. Kết quả so sánh YOLO26n trước đây được giữ trong [archive](docs/archive/YOLO26n_ablation_2026-09.md); không gộp chúng với kết quả MSR.

## Cấu trúc liên quan

- `src/tinydet/modules/msr.py`: module MSR P2 và nạp trọng số baseline với remap Detect.
- `src/tinydet/modules/btrm.py`, `configs/experiment/localization_btrm_*.yaml`: module localization P2 và protocol paired TinyPerson/VisDrone; xem [BTRM_P2.md](docs/BTRM_P2.md).
- `configs/model/yolo26n_tiny_A*.yaml`: baseline, ablation P2/SPPF và hai model có MSR.
- `configs/experiment/ablation_msr_p2.yaml`: bốn run đối chứng với chung dataset, seed và hyperparameter.
- `src/tinydet/training/`, `src/tinydet/evaluation/`: train, lưu artifact, validation và benchmark.
- `src/tinydet/research/results.py`: gom run, chọn primary run, tính mean/std và paired delta.
- `modal/app.py`: Modal entrypoints cho train, compare, evaluate, benchmark và phân tích.
- `scripts/`: smoke test, xác minh graph, chuẩn bị dữ liệu, metric TinyPerson/COCO small và error analysis.
- `tools/`: upload dữ liệu, tải run và tổng hợp kết quả.
- `docs/`: mô tả kiến trúc MSR và giao thức dữ liệu/đánh giá.
- [MSR diagnostics](docs/msr_diagnostics.md): metrics nội bộ, aggregation, W&B namespace và JSONL.
- [Error analysis](docs/error_analysis/README.md): record lỗi theo sample, decomposition
  miss/FP, recall theo size/density/boundary, paired comparison và report local.

## Môi trường và kiểm tra local

Yêu cầu Python 3.12. Môi trường local dùng PyTorch CPU; train chính chạy trên Modal L4.

```bash
uv sync --extra cpu --group dev
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

Kiểm tra này không thay thế AMP CUDA hoặc train trên dữ liệu thật. Chạy lint sau thay đổi mã nguồn; ghi rõ GPU test đã thực hiện hay chưa.

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

Để chạy nhiều seed paired, thay `seed: 42` bằng `seeds: [42, 43, 44]` trong
experiment config. Mỗi seed tạo một run riêng trong cùng experiment directory.
`--parallel` vẫn giới hạn bốn job đồng thời.

Trạng thái run và artifacts lưu trong Modal volume `tiny-person-runs`, dưới `/mnt/runs/<model>/<run-id>/`. Mỗi run có config, metadata, môi trường, checkpoint, metrics và evaluation. W&B là log bổ sung, không thay thế artifacts trong volume.

- Tiếp tục danh sách từ một model trở đi: thêm `--start-at msr_p2_e1_a2_msr` với tên model phù hợp. Lệnh này **train lại** từ model đó, không resume epoch từ `last.pt`.
- Tái sử dụng kết quả hoàn tất đúng cấu hình: thêm `--reuse`. Chỉ run `COMPLETED` có `best.pt` mới được reuse; `--reuse` không tiếp tục train run bị dở.
- Nếu terminal báo mất kết nối, trước tiên kiểm tra run/checkpoint trong volume. Lỗi client không tự chứng minh GPU worker đã dừng.

Tải kết quả sau khi xác định model và run ID:

```fish
python tools/download_results.py MODEL/RUN_ID --include metadata.json config.yaml environment.txt metrics checkpoints evaluation
```

Tổng hợp kết quả đã tải:

```bash
python tools/list_runs.py --root outputs \
  --config configs/experiment/ablation_msr_p2.yaml \
  --csv outputs/results.csv \
  --markdown outputs/results.md
```

Tool chọn một primary run cho mỗi experiment/seed, ưu tiên `COMPLETED`, metric
đánh giá độc lập, rồi run mới nhất. Bảng paired delta dùng `comparisons` trong
experiment config. Đây là bảng làm việc, không thay thế kiểm tra artifact gốc.

## Đánh giá

Dùng validation để chọn checkpoint. So sánh AP@[.50:.95], AP50, AP75, precision và recall; báo cáo thêm AP nhỏ theo evaluator đã xác minh. Dataset chỉ có validation derivative thì gọi kết quả là validation, không gọi là test.

So sánh MSR theo từng cặp E1/E0 và E3/E2. Nếu chênh lệch nhỏ, lặp seed 42, 43, 44 cho cả baseline và MSR, báo cáo mean/std. Đo latency và memory trên cùng L4, resolution, batch size; tách riêng model latency khỏi preprocessing/postprocessing.

Giao thức benchmark chính thức của TinyPerson, gồm ignore/dense annotations và ảnh gốc, nằm trong [TINYPERSON_BENCHMARK.md](docs/TINYPERSON_BENCHMARK.md). Không dùng official test để chọn model hoặc hyperparameter.

## Trạng thái kết luận

MSR đã được nối trực tiếp trong model YAML, parse qua namespace Ultralytics, và chạy smoke test local trên CPU. E1 tăng recall nhưng không cải thiện AP75; các fusion mode đều dưới baseline AP75. Trong warm-start kiểm soát initialization, MSR tăng precision/AP50 nhưng official-test AP75 giảm từ 0.02472 xuống 0.01986. BTRM v2 cũng không cải thiện localization: warm-start AP75 giảm từ 0.013398 xuống 0.013371 trên validation và từ 0.02472 xuống 0.01953 trên official test. Phân tích lỗi rerun cho thấy box lỗi trung vị hẹp hơn GT và lệch tâm, tức nút thắt là regression/assignment chứ không phải thiếu contrast; tăng độ phức tạp BTRM chưa phải hướng hợp lý. Xem [BTRM_REPORT.md](docs/BTRM_REPORT.md).

**SADR-P2 (8-bin DFL thay `reg_max=1`) là kết quả dương đầu tiên:** cùng 50 epoch, seed 42 và protocol A2, validation AP75 tăng `0.015852 → 0.019406`; official test AP mean `0.218384 → 0.236031`, AP50 `0.230141 → 0.258871`, recall `0.521882 → 0.558048`, TP `7.143 → 7.638`, background FP `84.359 → 78.208`. AP75 tổng đi ngang (`0.023745 → 0.023302`) do thay đổi ranking, dù số GT đạt IoU≥0.75 tăng `1.877 → 2.177` và AP75 của tiny1/tiny2/tiny3 đều tăng. Đây là kiến trúc class-agnostic, nhưng mới có một seed và một dataset; chưa đủ claim tổng quát. Chi tiết, công thức và tiêu chí bác bỏ: [SADR_P2.md](docs/SADR_P2.md).
