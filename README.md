# Tiny Person Detection Research: D-QAC (QAC v2)

Nghiên cứu phát hiện người rất nhỏ (Tiny Person Detection) bằng YOLO26n trên TinyPerson với tầng đặc trưng P2 kết hợp **Distributional Quality-Aware Calibration (D-QAC / QAC v2)**.

Repo này là **research project**. Ưu tiên là câu hỏi khoa học, kiểm soát biến đối chứng, reproducibility, tốc độ vòng lặp thí nghiệm và tính chuẩn xác của số liệu. Đọc [Báo cáo toàn diện QAC v2](docs/QAC_V2_REPORT.md) và [Research practices](docs/RESEARCH_PRACTICES.md).

---

## 1. Phương pháp Đề xuất: D-QAC (QAC v2)

Phương pháp giải quyết mâu thuẫn cốt lõi **Confidence-Localization Misalignment** trên tầng P2 thông qua 2 cơ chế chính:
1. **DUGC (Distributional Uncertainty-Gated Calibration)**: Trích xuất top-5 xác suất phân phối biên từ DFL logits (chỉ 88 tham số) kết hợp cơ chế phủ quyết **Semantic-Veto** để triệt tiêu False Positive ở nền sóng biển và cát.
2. **SATAR (Scale-Adaptive Task Alignment Recalibration)**: Kết hợp khoảng cách Gaussian Wasserstein chuẩn hóa (NWD) với số mũ thích ứng diện tích $\beta(A) \in [1.5, 6.0]$, bảo vệ gradient của các anchor siêu nhỏ không bị triệt tiêu.

Mã nguồn triển khai:
* [`src/tinydet/modules/qac_v2.py`](src/tinydet/modules/qac_v2.py): `QACDetectV2`, `QualityAwareDetectionLossV2`, `QualityAwareE2ELossV2`.
* [`src/tinydet/modules/qac.py`](src/tinydet/modules/qac.py): Phiên bản v1 (đối chứng ablation).

---

## 2. Kết quả Thực nghiệm Chính thức trên TinyPerson Validation

| Mô hình | Cơ chế | Tham số | GFLOPs | mAP50-95(B) | mAP50(B) | AP75(B) | Precision | Recall | Tốc độ A10 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A2 P2** | Standard Detect (reg_max=1) | 2.52M | 7.6 | 0.0557 (5.57%) | 0.1840 (18.40%) | 0.0114 (1.14%) | 0.3427 (34.27%) | 0.2456 (24.56%) | 9.8 ms |
| **QAC v1** | Feature Gate + Floor Clamp | 3.24M | 15.6 | 0.0657 (6.57%) | 0.2078 (20.78%) | 0.0203 (2.03%) | 0.3556 (35.56%) | 0.2741 (27.41%) | 10.6 ms |
| **QAC v2 (D-QAC)** | **DUGC + SATAR (Proposed)** | **3.24M** | **15.6** | **0.0716 (7.16%)** | **0.2307 (23.07%)** | **0.0210 (2.10%)** | **0.3959 (39.59%)** | **0.2824 (28.24%)** | **10.6 ms (~94 FPS)** |

* **mAP50-95**: Tăng **+28.5%** tương đối so với Baseline.
* **mAP50**: Tăng **+25.4%** tương đối so với Baseline.
* **AP75**: Tăng **+84.2%** tương đối so với Baseline.
* **Precision**: Đạt tiệm cận **40%** (39.59%), khẳng định tính ưu việt của cơ chế Semantic-Veto.

Chi tiết phân rã sai số theo từng dải kích thước (0-8px, 8-16px, 16-32px) nằm trong [Báo cáo QAC v2](docs/QAC_V2_REPORT.md).

---

## 3. Cấu trúc Repo Đã Tinh gọn

- `configs/model/`:
  - `yolo26n_tiny_A2_p2.yaml`: Cấu hình baseline YOLO26n với neck P2.
  - `yolo26n_tiny_A2_p2_qac.yaml`: Mô hình QAC v1.
  - `yolo26n_tiny_A2_p2_qac_v2.yaml`: Mô hình QAC v2 tối ưu toán học (reg_max=16, topk=5, NWD).
- `configs/experiment/`:
  - `qac_v2_tinyperson.yaml`: Thực nghiệm so sánh song song 3 mô hình từ đầu.
  - `qac_v2_resume50_tinyperson.yaml`: Cấu hình fine-tune thêm 50 epochs với Cosine LR scheduler.
  - `qac_tinyperson.yaml`: Thực nghiệm 2 chiều Baseline vs QAC v1.
- `src/tinydet/modules/`:
  - `qac_v2.py`: Module D-QAC (DUGC + SATAR).
  - `qac.py`: Module QAC v1.
- `modal/app.py`: Modal cloud entrypoints để huấn luyện (`train`, `compare`), đánh giá (`evaluate`), đo tốc độ (`benchmark`), và phân tích hình học (`analyze_geometry`).
- `docs/QAC_V2_REPORT.md`: Báo cáo khoa học hoàn chỉnh nhất với đầy đủ số liệu và chứng minh toán học.

---

## 4. Các Lệnh Thực thi Chính trên Modal

### Huấn luyện so sánh song song 3 mô hình (Baseline vs v1 vs v2):
```bash
modal run modal/app.py::compare --config configs/experiment/qac_v2_tinyperson.yaml --parallel
```

### Đánh giá độc lập checkpoint tốt nhất của QAC v2:
```bash
modal run modal/app.py::evaluate \
  --run tiny_A2_p2_qac_v2/20261025-201303-b086541-seed42 \
  --split val
```

### Phân tích sai số hình học theo dải kích thước:
```bash
modal run modal/app.py::analyze_geometry \
  --run tiny_A2_p2_qac_v2/20261025-201303-b086541-seed42 \
  --split val \
  --iou 0.5
```
