# Tiny Person Detection Research: GauCon

Nghiên cứu phát hiện người cực nhỏ trong ảnh hàng hải bằng YOLO26n-P2. Phương pháp hiện tại gồm đầu phát hiện đồng thuận Gauss cục bộ **GauCon** và cơ chế gán nhãn Wasserstein tách rời **DW-TAL**. Tài liệu nghiên cứu chính:

- Báo cáo đồ án tiếng Việt: [docs/report/course_report_vi.md](docs/report/course_report_vi.md).
- Bài báo tiếng Anh: [paper/main.tex](paper/main.tex).
- Trạng thái thí nghiệm và provenance: [docs/report/00_state.md](docs/report/00_state.md).

## 1. Phương pháp

GauCon hoạt động trên hai tầng đặc trưng $P_2$ và $P_3$ của YOLO26n-P2. Mỗi hộp dự đoán được biểu diễn bằng một phân phối Gauss 2D. Các hộp từ ô lân cận và ô cha được quy đổi về cùng hệ tọa độ, so sánh bằng khoảng cách Wasserstein chuẩn hóa theo phương sai cục bộ, rồi kết hợp bằng nội suy có hệ số học được. Nhánh phân loại dùng thông tin quan hệ lân cận để điều chỉnh logit.

DW-TAL dùng độ tương đồng Wasserstein có điều biến theo kích thước đối tượng trong bước chọn ứng viên dương. Mục tiêu phân loại sau đó được hiệu chuẩn theo IoU thực tế. Chưng cất phân phối hộp từ nhánh giám sát đa mẫu chỉ được dùng trong huấn luyện.

## 2. Kết quả thực nghiệm

Chiến dịch đối chứng hiện tại dùng 80 epoch, trọng số khởi tạo COCO, batch size 16, seed 42 và GPU NVIDIA L4. TinyPerson được đánh giá trên 786 ảnh kiểm tra, với một lớp person. Giao thức tiled sử dụng NMS để ghép kết quả từ các lát cắt; giao thức full-frame chạy một lượt không dùng NMS.

| Chỉ số | Base | + NWD | + GauCon |
|---|---:|---:|---:|
| Tiled $AP_{50}^{all}$ | 45,58% | 48,45% | **48,53%** |
| Tiled $AP_{75}^{all}$ | 6,40% | 6,09% | **7,37%** |
| Full-frame $AP_{50}^{all}$ | 43,08% | 43,24% | **44,13%** |
| Full-frame $AP_{75}^{all}$ | 4,44% | 4,13% | **4,78%** |

GauCon thêm 714 tham số, đưa tổng số tham số sau hợp nhất lên 2.402.374. Phép đo một phiên trên L4 ghi nhận 73,95 FPS ở đầu vào 1280, nhưng chưa có profiling đa phiên để kết luận về chênh lệch tốc độ.

### Giới hạn hiện tại

- Mỗi cấu hình mới có một lần huấn luyện với seed 42; chưa đo biến thiên giữa các seed.
- Chưa có thí nghiệm ablation để tách đóng góp của từng thành phần.
- GauCon thấp hơn NWD ở nhóm tiny1 và làm tăng false positives nền trong giao thức tiled.
- Bootstrap hiện lấy mẫu theo ảnh, chưa mô hình hóa tương quan giữa các khung hình cùng video.
- Lợi thế AP50 của GauCon so với NWD chưa được xác nhận rõ bằng khoảng tin cậy ở cả hai giao thức.

## 3. Cấu trúc Repo

- `configs/model/yolo26n_p2_lgt_v2.yaml`: cấu hình YOLO26n-P2 với GauCon và DW-TAL.
- `configs/experiment/paper_fast/`: cấu hình chiến dịch đối chứng 80 epoch.
- `src/tinydet/modules/lgt_detect.py`: đầu phát hiện GauCon.
- `src/tinydet/models/ultralytics.py`: tích hợp DW-TAL và các hàm mất mát.
- `scripts/build_dut_report.py`: tạo DOCX và PDF theo định dạng báo cáo DUT.
- `docs/report/`: nguồn báo cáo, trạng thái thí nghiệm và hình minh họa.
- `paper/`: nguồn bài báo LaTeX.

---

## 4. Các Lệnh Thực thi Chính

### Chạy 1 lệnh trên Google Colab / Kaggle (tự cài `uv` + PyTorch CUDA, tự tải dataset từ Hugging Face, preflight, W&B sync & resume)
1. Thêm secret `WANDB_API_KEY` trong **Colab Secrets** (hoặc **Kaggle Add-ons → Secrets**; thêm `HF_TOKEN` nếu chuyển dataset repo sang private).
2. Chạy đúng 1 lệnh sau khi `git clone`:
```bash
git clone <repo-url> && cd thigiacmaytinh
bash scripts/run_gpu.sh --config configs/experiment/paper_fast/tp80_yolo26n_lgt.yaml
```
- **Tự động hoàn toàn**: cài `uv`, `uv sync --extra gpu --frozen` (Python 3.14 + `torch+cu128` theo `uv.lock`), phát hiện môi trường (`colab`/`kaggle`), tự tìm dataset local hoặc **tự tải `tinyperson-yolo-v2.tar.gz` từ Hugging Face (`longqt321/tinyperson-yolo`) + xác minh SHA-256 + giải nén vào SSD local**, chạy `audit_dataset` + model preflight, đăng nhập W&B từ Secrets với tag `host:colab`/`host:kaggle` và deterministic run ID (`resume="allow"`) đồng bộ hoàn toàn với Modal, lưu checkpoint bền vững (Google Drive / `/kaggle/working/runs`), và gọi `finalize()`.
- Chạy lại đúng lệnh trên sau khi bị ngắt session sẽ **tự động resume** từ `last.pt`.
- Truyền thêm cờ tùy chọn nếu cần: `--hf-dataset <owner/repo>`, `--data <path>`, `--output-root <dir>`, `--wandb-group <group>`, `--batch 8`, `--epochs 150`.

### Kiểm tra nhanh trên CPU (lint & dry-run)
```bash
uv sync --extra cpu --group dev
uv run --extra cpu --group dev ruff check .
bash scripts/run_gpu.sh --config configs/experiment/paper_fast/tp80_yolo26n_lgt.yaml --dry-run
```

Train trên CPU bị chặn có chủ đích: cả `train.py` và `tinydet.training.trainer.train` đều từ chối khi không có CUDA.

### Chạy chiến dịch trên Modal (detached)
```bash
modal run -d launch_campaign.py::main
modal run launch_campaign.py::fetch
```
