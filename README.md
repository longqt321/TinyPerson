# Tiny Person Detection Research: GauSS

Nghiên cứu phát hiện người rất nhỏ (Tiny Person Detection) bằng YOLO26n trên TinyPerson với tầng đặc trưng P2 kết hợp **GauSS** (*Gaussian Scale-Space Alignment & Refinement*, triển khai trong [`src/tinydet/modules/gauss.py`](src/tinydet/modules/gauss.py)).

Repo này là **research project**. Ưu tiên là câu hỏi khoa học, kiểm soát biến đối chứng, reproducibility, tốc độ vòng lặp thí nghiệm và tính chuẩn xác của số liệu. Đọc [Research practices](docs/RESEARCH_PRACTICES.md), [Báo cáo đồ án](docs/report/course_report_vi.md) và [Paper](paper/main.tex).

---

## 1. Phương pháp Đề xuất: GauSS (v3.1 / v3.2)

Module tinh chỉnh đặc trưng P2 của neck từ **C2 (stride 4), P3 (stride 8) và P4 (stride 16)**, khai thác phân rã đa băng tần Gaussian:

1. **Gaussian Scale-Space Decomposition**: hai bộ lọc tách biệt khởi tạo ở $\sigma_1 = 0.65$, $\sigma_2 = 1.45$ tạo 3 dải $B_{\text{high}}, B_{\text{mid}}, B_{\text{low}}$.
2. **Sub-pixel Band Encoders**: `PixelUnshuffle(2)` + depthwise conv cho từng dải; upsampling bằng DySample.
3. **Dilated Context Branch**: depthwise conv đa tốc độ giãn $r \in \{1, 3, 5\}$.
4. **Hybrid Energy Router**: sinh trọng số trộn 3 dải theo từng vị trí không gian.
5. **Bounded Residual Output**: $X_0 + \alpha z + \gamma_{\text{out}} B_{\text{high}}$ với cổng $0.5 \tanh(\cdot)$.

Mã nguồn triển khai: [`src/tinydet/modules/gauss.py`](src/tinydet/modules/gauss.py).
Cấu hình chuẩn khớp checkpoint báo cáo (v3.1): [`configs/model/yolo26n_tiny_A2_p2_gauss_v31_tinyperson.yaml`](configs/model/yolo26n_tiny_A2_p2_gauss_v31_tinyperson.yaml).

---

## 2. Kết quả Thực nghiệm Chính thức trên TinyPerson Test (Merge Protocol)

Giao thức: mô hình 1 lớp (`person`) huấn luyện 150 epoch (letterbox 1280, seed 42), đánh giá bằng `class_mode="merge"` trên 786 ảnh test (13,687 positive GT). Nguồn JSON: `reports/eval/{baseline,gauss_v3}_tinyperson_wacv_150e_{tiled,full}.json`.

### Tiled Slicing ($640 \times 512$, overlap 100, global NMS IoU 0.50)

| Mô hình | Tham số | AP50 (all) | AP50 (tiny) | AP50 (tiny1) | AP50 (tiny2) | AP50 (tiny3) | AP50 (small) | AP50 (reasonable) | MR50 (all) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline YOLO26n-P2** | 2.516M | 40.63% | **37.57%** | **22.30%** | 36.92% | **48.14%** | 51.63% | 44.14% | **92.73%** |
| **+ GauSS (v3.1)** | 2.553M | **41.08%** | 37.40% | 22.26% | **37.67%** | 47.05% | **52.51%** | **47.59%** | 93.04% |

### Full-Frame (letterbox 1280)

| Mô hình | AP50 (all) | AP50 (tiny) | AP50 (tiny1) | AP50 (tiny2) | AP50 (tiny3) | AP50 (small) | AP50 (reasonable) | MR50 (all) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline YOLO26n-P2** | 35.32% | 30.02% | **17.05%** | 30.02% | 38.46% | **42.96%** | 47.56% | **93.40%** |
| **+ GauSS (v3.1)** | **35.45%** | **30.23%** | 16.76% | **30.45%** | **38.82%** | 42.33% | **49.10%** | 93.90% |

* **Overhead (v3.1 checkpoint)**: +36,878 tham số (+1.47%; bao gồm nhánh C3k2 trung gian 8,704 params không dùng; cấu hình làm sạch v3.2 tăng +28,174 params, +1.12%).
* Kết quả **một seed duy nhất (seed 42)** trên evaluator standalone; chưa có kiểm định ý nghĩa thống kê đa seed.

Chi tiết đầy đủ nằm trong:
- Báo cáo đồ án (tiếng Việt): [`docs/report/course_report_vi.md`](docs/report/course_report_vi.md) và bản xuất [`BaoCao_DoAn_GauSS_TinyPerson.pdf`](BaoCao_DoAn_GauSS_TinyPerson.pdf) / [`.docx`](BaoCao_DoAn_GauSS_TinyPerson.docx).
- Paper LaTeX (tiếng Anh): [`paper/main.tex`](paper/main.tex).
- Nhật ký trạng thái & provenance: [`docs/report/00_state.md`](docs/report/00_state.md).

---

## 3. Cấu trúc Repo

- `configs/model/`: cấu hình Ultralytics YAML cho Baseline YOLO26n-P2, GauSS v3.1/v3.2, P2Refine, GaussSplit.
- `configs/experiment/`: cấu hình thực nghiệm TinyPerson, VisDrone và `ablation/`.
- `src/tinydet/`: mã nguồn gói chính (`modules/`, `training/`, `evaluation/`, `error_analysis/`, `augment/`, `data/`, `models/`, `utils/`).
- `train.py`: cổng vào huấn luyện độc lập môi trường (Modal, Colab, Kaggle, RunPod).
- `modal/app.py` & `launch_campaign.py`: entrypoints chạy chiến dịch huấn luyện/đánh giá detached trên Modal Cloud.
- `reports/eval/` & `reports/data/`: dữ liệu đánh giá JSON, ablation và log training phục vụ báo cáo.
- `paper/` & `docs/report/`: nguồn bài báo LaTeX và báo cáo đồ án tiếng Việt.

---

## 4. Các Lệnh Thực thi Chính

### Kiểm tra nhanh trên CPU (lint & dry-run)
```bash
uv sync --extra cpu --group dev
uv run --extra cpu --group dev ruff check .
uv run --extra cpu --group dev python train.py \
  --config configs/experiment/gauss_v3_tinyperson_wacv_150e.yaml \
  --data datasets/derived/tinyperson-yolo-v2/data.yaml --dry-run
```

Train trên CPU bị chặn có chủ đích: cả `train.py` và `tinydet.training.trainer.train` đều từ chối khi không có CUDA.

### Huấn luyện trên GPU host bất kỳ (Colab, Kaggle, RunPod, ...)
```bash
pip install -q uv
uv sync --extra gpu --group dev

uv run --extra gpu python train.py \
  --config configs/experiment/gauss_v3_tinyperson_wacv_150e.yaml \
  --data /content/data_local/tinyperson-yolo-v2/data.yaml \
  --output-root /content/drive/MyDrive/tinydet_runs \
  --device 0 --batch 8 --imgsz 1280 --workers 4 \
  --name gauss_v3_seed42 --run-id run1
```

Lưu ý:
- Luôn truyền `--extra gpu` cho `uv run` để giữ đúng bản `torch` CUDA đã khóa trong `uv.lock`.
- Muốn resume khi run bị ngắt giữa chừng, phải truyền lại đúng `--run-id`.

### Chạy chiến dịch trên Modal (detached)
```bash
modal run -d launch_campaign.py::main
modal run launch_campaign.py::fetch
```
