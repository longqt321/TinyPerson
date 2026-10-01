# RFLA và NWD trên YOLO26n chỉ P3

## Phạm vi

Đã triển khai hai module PyTorch, adapter Ultralytics và cấu hình ablation 2×2. Kiểm tra nhỏ chạy trên CPU; chưa train 50 epoch trên TinyPerson, nên chưa có kết luận về accuracy.

Các module chỉ thay đổi training. Kiến trúc inference, số tham số và cách đánh giá IoU giữ nguyên. Cả bốn nhánh dùng cùng YAML `yolo26n-p3-only.yaml`, 144.106 tham số trước fuse.

## Cơ sở thuật toán

- RFLA: [bài ECCV 2022](https://arxiv.org/abs/2208.08738), [code chính thức](https://github.com/Chasel-Tsui/mmdet-rfla). Điểm đặc trưng được mô hình hóa bằng Gaussian receptive field; GT bằng Gaussian có sigma `(w/2, h/2)`. Điểm RFD là `1/(1 + KL(RF || GT))`. HLA chọn top-k, thu nhỏ sigma bằng hệ số 0,9 rồi bổ sung top-1 vào những điểm chưa được gán. Khi nhiều GT tranh một điểm, ưu tiên GT nhỏ hơn.
- NWD: [bài gốc](https://arxiv.org/abs/2110.13389), [code chính thức](https://github.com/jwwangchn/NWD). Với box dạng `(cx, cy, w, h)`, `W² = Δcx² + Δcy² + Δw²/4 + Δh²/4`; `NWD = exp(-sqrt(W²)/C)`; loss là `1-NWD`. Cấu hình hiện tại dùng `C=12.8` pixel làm điểm khởi đầu từ nghiên cứu gốc, chưa tối ưu cho TinyPerson.

## Cách chuyển sang YOLO

NWD thay riêng loss CIoU với `weight=1.0`. Nếu đặt weight trong (0,1), loss là `(1-weight)*CIoU_loss + weight*NWD_loss`. Phần DFL hoặc L1, classification, gán nhãn và hậu xử lý không đổi trong nhánh chỉ NWD. Box được đổi từ đơn vị feature-grid về pixel trước khi tính NWD. Đây là ablation NWD regression, không phải tái hiện cả NWD assignment/NMS trong bài gốc.

RFLA thay assigner, dùng nhãn one-hot cho mẫu dương thay cho quality score của TAL. Với YOLO26, nhánh one-to-many dùng HLA [3,1]; nhánh one-to-one dùng top-1 và tắt bước bổ sung để giữ tối đa một positive/GT. Các trọng số và lịch cập nhật E2E của Ultralytics được giữ nguyên. Đây là adaptation cho YOLO, không phải kết quả tái lập detector MMDetection trong bài.

RF trong ablation được khai báo tường minh: đường convolution dài nhất đến output box của P3 có TRF lần lượt 3, 7, 23, 31, 63, 127 và 159 pixel. Hai convolution 3×3 cuối thuộc Detect. Với `erf_fraction=0.5`, sigma là `159*0.5/2=39.75` pixel. Đây là xấp xỉ ERF từ TRF, không phải ERF đo bằng gradient.

Do chỉ có một mức P3 và mọi điểm có cùng sigma, thu nhỏ sigma không thay thứ tự RFD của các điểm đối với cùng GT. Khi không có tie đặc biệt, bước HLA thứ hai không thêm điểm mới trên bản này. Không được quy mọi thay đổi AP cho bước HLA bổ sung.

## Tương thích

- Kiểm tra trên Ultralytics 8.4.164; Modal image đã khóa đúng phiên bản này, khớp `uv.lock` local.
- YOLO26 reg_max=1: hỗ trợ chọn điểm ngoài GT vì khoảng cách box có dấu.
- YOLOv8/YOLO11 DFL: NWD dùng trực tiếp. RFLA cần `inside_only: true` để tránh giao mục tiêu khoảng cách âm cho DFL. Đây là biến thể có ràng buộc, không thể khắc phục GT không chứa điểm nào như RFLA đầy đủ. Mặc định báo lỗi nếu ghép DFL với outside-GT RFLA.
- Hỗ trợ horizontal-box detection qua API loss Ultralytics đã kiểm thử. Các repo YOLOv5/v7 độc lập, OBB, segmentation và pose cần adapter khác; không tuyên bố tương thích mọi YOLO.
- Nếu đổi kiến trúc, nên cung cấp `receptive_fields` theo từng level của Detect. Nếu bỏ danh sách, adapter dùng proxy `sigma_stride_ratio * stride` (mặc định 4); proxy này không phải TRF đo hay suy ra tự động từ mạng.
- Không sửa package Ultralytics hay monkeypatch toàn cục. Cấu hình module nằm trong model YAML/checkpoint, tồn tại qua EMA và reload.

## Ablation đã cấu hình

File: `configs/experiment/ablation_rfla_nwd.yaml`.

| Tên run | RFLA | NWD regression |
|---|---|---|
| yolo26n_p3_ablation_baseline | Tắt | Tắt |
| yolo26n_p3_ablation_rfla | Bật | Tắt |
| yolo26n_p3_ablation_nwd | Tắt | Bật |
| yolo26n_p3_ablation_rfla_nwd | Bật | Bật |

Dùng cùng TinyPerson, seed 42, AdamW, batch 16, imgsz 1280, epochs 50, patience 15 và toàn bộ augmentation giống comparison hiện tại. Patience có thể dừng sớm: 50 là số epoch tối đa. Nếu muốn bắt buộc đủ 50 epoch, đặt patience: 0 cho cả bốn nhánh trước khi chạy.

```bash
uv run --extra cpu modal run modal/app.py::compare \
  --config configs/experiment/ablation_rfla_nwd.yaml --parallel
```

Lệnh cấp tối đa bốn container L4 cho bước train. Trần RAM mỗi container 24 GiB; GPU rảnh có scaledown_window=2. Đây vẫn là chạy GPU có phí. Không cần đổi model YAML để bật/tắt module.

Metrics giữ AP@0.50…0.95, mAP50–95 và W&B. So sánh best epoch theo cùng mAP50–95; đánh giá lại best checkpoint bằng cùng validation/test. Các run này có tên riêng nên không thay tag latest của bốn kiến trúc cũ.

## Dùng bên ngoài comparison

```python
from functools import partial
from ultralytics import YOLO
from tinydet.modules.ultralytics_adapter import ModuleDetectionTrainer

modules = {
    "rfla": {"enabled": True, "receptive_fields": [159.0]},
    "nwd": {"enabled": True, "constant": 12.8, "weight": 1.0},
}
model = YOLO("configs/model/yolo26n-p3-only.yaml")
model.train(
    trainer=partial(ModuleDetectionTrainer, modules=modules),
    data="path/to/data.yaml", epochs=50, imgsz=1280, batch=16,
)
```

Với YOLO26 nhiều level, phải đổi danh sách receptive_fields hoặc chọn proxy stride tường minh. Với DFL, thêm `inside_only: true` vào rfla.

## Kiểm tra đã chạy

- Công thức NWD, gradient ở box trùng/không trùng và đơn vị pixel.
- KLD có RFD=1 khi hai Gaussian trùng nhau.
- Mẫu ngoài GT, ưu tiên GT nhỏ khi va chạm, empty GT, bổ sung HLA và giới hạn one-to-one.
- Forward/backward cả bốn nhánh trên P3, gradient hữu hạn, giữ số params, serialize/reload cấu hình.
- Forward/backward YOLOv8 và YOLO11 với NWD + RFLA có ràng buộc DFL.
- Train một epoch, validation, save best checkpoint, reload và predict cho cả bốn nhánh trên dữ liệu tổng hợp CPU; không dùng AP của dữ liệu tổng hợp để đánh giá nghiên cứu.

Chạy kiểm tra: `uv run --extra cpu pytest -q tests/test_tiny_modules.py`.
