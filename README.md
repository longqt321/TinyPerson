# Tiny Person Detection Research: GauSS v2.1

Nghiên cứu phát hiện người rất nhỏ (Tiny Person Detection) bằng YOLO26n trên TinyPerson với tầng đặc trưng P2 kết hợp **GauSS** (*Gaussian Scale-Space Alignment & Refinement*, triển khai trong [`src/tinydet/modules/gauss.py`](src/tinydet/modules/gauss.py)).

Repo này là **research project**. Ưu tiên là câu hỏi khoa học, kiểm soát biến đối chứng, reproducibility, tốc độ vòng lặp thí nghiệm và tính chuẩn xác của số liệu. Đọc [Research practices](docs/RESEARCH_PRACTICES.md) và [Báo cáo tổng kết GauSS](reports/gauss_final_report.md).

---

## 1. Phương pháp Đề xuất: GauSS v2.1

Module thay thế đặc trưng P2 của neck bằng một nhánh tinh chỉnh từ **C2 (stride 4), P3 (stride 8) và P4 (stride 16)**, khai thác phân rã đa băng tần Gaussian:

1. **Learnable Gaussian Scale-Space**: hai bộ lọc tách biệt $\sigma_1 = 0.65$, $\sigma_2 = 1.45$ tạo phân rã 3 băng $B_{\text{high}}, B_{\text{mid}}, B_{\text{low}}$.
2. **Sub-pixel Encoders**: `PixelUnshuffle(2)` + depthwise conv cho từng băng.
3. **Dilated Context Branch**: depthwise conv với $r \in \{1, 2, 3\}$ + softmax gate.
4. **Spatial Router**: sinh hệ số trộn băng $a_{\text{bands}}$ và cổng $m_{\text{gate}}$.
5. **Band Synthesis**: $F_{\text{band}} = \sum_i a_i R_i + \gamma_h R_{\text{high}}$.

Mã nguồn triển khai: [`src/tinydet/modules/gauss.py`](src/tinydet/modules/gauss.py) — `GauSS`, `LearnableGaussianScaleSpace`, `BandBranch`, `DilatedContextBranch`, `RefinementBranch`.

---

## 2. Kết quả Thực nghiệm Chính thức trên TinyPerson Test

Giao thức: tiled slicing $640 \times 512$, overlap 100, global NMS IoU 0.50, 786 ảnh test, 13,687 positive GT, seed 42.

| Mô hình | Tham số | GFLOPs (tiled) | AP50 (all) | AP50 (tiny 2–20px) | AP50 (tiny1 2–8px) | AP50 (tiny2 8–12px) | MR50 (all) | FPS (A10G) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline YOLO26n-P2** | 2.52M | 7.62 | 11.95% | 9.07% | 4.72% | 6.90% | 96.93% | 88.6 |
| **GauSS (GauSS v2.1)** | 2.55M | 8.13 | **12.94%** | **9.76%** | **5.18%** | **9.50%** | **96.73%** | 65.8 |

* **AP50 (all)**: +0.98 điểm phần trăm (+8.2% tương đối) so với baseline 150 epoch.
* **AP50 (tiny2, 8–12px)**: +2.59 điểm phần trăm — mức tăng lớn nhất.
* **Overhead**: chỉ +34,666 tham số (+1.38%) và +0.51 GFLOPs (+6.7%).
* Kết quả **một seed duy nhất**, evaluator standalone; chưa có kiểm định ý nghĩa thống kê.

Chi tiết đầy đủ (đa ngưỡng IoU, full-frame, VisDrone, so sánh quốc tế) nằm trong [Báo cáo tổng kết GauSS](reports/gauss_final_report.md) và [Tổng hợp GauSS vs Baseline](reports/gauss_vs_baseline_summary.md).

---

## 3. Cấu trúc Repo

- `configs/model/`:
  - `yolo26n_tiny_A2_p2.yaml`: Cấu hình ablation A2 gốc (nc=1).
  - `yolo26n_tiny_A2_p2_tinyperson.yaml`: Baseline YOLO26n-P2 cho TinyPerson.
  - `yolo26n_tiny_A2_p2_gauss_tinyperson.yaml`: Mô hình GauSS (GauSS v2.1).
  - `yolo26n_tiny_A2_p2_visdrone.yaml` / `..._gauss_visdrone.yaml`: Cặp tương ứng cho VisDrone.
- `configs/experiment/`: cặp paired baseline/GauSS cho TinyPerson, VisDrone TPAMI, VisDrone fine-tune; `ablation/` chứa 5 ablation bỏ thành phần.
- `src/tinydet/modules/`: `gauss.py` — module GauSS duy nhất còn lại; `p2_refine.py` — P2Refine.
- `train.py`: cổng vào huấn luyện độc lập môi trường (Modal, Colab, Kaggle, RunPod).
- `modal/app.py`: Modal cloud entrypoints để huấn luyện (`train`, `compare`), đánh giá (`evaluate`, `evaluate_checkpoint`), đo tốc độ (`benchmark`), phân tích hình học (`analyze_geometry`).
- `modal/ablation_runner.py`: chạy 5 ablation song song trên 5 GPU A10.
- `reports/gauss_final_report.md`: báo cáo khoa học hoàn chỉnh nhất với đầy đủ số liệu.

---

## 4. Các Lệnh Thực thi Chính trên Modal

### Huấn luyện trên GPU host bất kỳ (Colab, Kaggle, RunPod, ...):
```bash
pip install -q uv
uv sync --extra gpu --group dev

uv run --extra gpu python train.py \
  --config configs/experiment/p2refine_tinyperson_wacv.yaml \
  --data /content/data_local/tinyperson-yolo-v2/data.yaml \
  --output-root /content/drive/MyDrive/tinydet_runs \
  --device 0 --batch 8 --imgsz 1280 --workers 4 \
  --name p2refine_colab_seed42 --run-id run1
```

Lưu ý:
- Luôn truyền `--extra gpu` cho `uv run`. Gõ `uv run python ...` không có extra
  sẽ kéo một bản torch khác từ PyPI đè lên bản `2.11.0+cu128` đã khoá.
- Muốn resume khi run bị ngắt giữa chừng, phải truyền lại đúng `--run-id`.
  Không có `--run-id` thì mỗi lần chạy tạo thư mục run mới (theo timestamp)
  và train lại từ epoch 0.

Kiểm tra cấu hình mà không cần GPU (chạy được trên CPU):
```bash
uv run --extra cpu --group dev python train.py \
  --config configs/experiment/p2refine_tinyperson_wacv.yaml \
  --data datasets/derived/tinyperson-yolo-v2/data.yaml --dry-run
```

Train trên CPU bị chặn có chủ đích: cả `train.py` và `tinydet.training.trainer.train`
đều từ chối khi không có CUDA.

### Huấn luyện so sánh song song (Baseline vs GauSS):
```bash
modal run modal/app.py::compare --config configs/experiment/gauss_v2_1_tinyperson_wacv.yaml --parallel
```

### Đánh giá checkpoint tốt nhất:
```bash
modal run modal/app.py::evaluate \
  --run gauss_v2_1_tinyperson_wacv/<run-id> \
  --split val
```

### Phân tích sai số hình học theo dải kích thước:
```bash
modal run modal/app.py::analyze_geometry \
  --run gauss_v2_1_tinyperson_wacv/<run-id> \
  --split val \
  --iou 0.5
```

### Chạy 5 ablation bỏ thành phần (seed 42, song song):
```bash
modal run modal/ablation_runner.py::run_all
```
