# Hướng cải tiến MSR

Tài liệu này tổng hợp bằng chứng hiện có về MSR và chốt thí nghiệm kế tiếp.
Mục tiêu không phải thêm một module mới, mà kiểm tra một giả thuyết hẹp có thể
bác bỏ được.

## Bằng chứng hiện có

Baseline là A2 P2. Các run fusion dùng cùng graph, dataset, seed 42 và
50 epoch; metric là validation của `best.pt`.

| Run | AP | AP50 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| A2 baseline | 0.059119 | 0.196888 | 0.015852 | 0.358155 | 0.252494 |
| MSR adaptive | 0.053852 | 0.174575 | 0.015724 | 0.305060 | 0.238914 |
| MSR mean | 0.055488 | 0.184428 | 0.013999 | 0.328125 | 0.265708 |
| MSR max | 0.050229 | 0.172751 | 0.010232 | 0.317191 | 0.239468 |

MSR warm-start từ checkpoint A2 cũng cho cùng dấu hiệu: tăng recall/AP50 ở một
số run nhưng giảm AP75 trên validation và official test. MSR hiện là module
evidence/recall, không phải module localization.

Diagnostics của các run fusion cho thấy:

- Gate fusion adaptive gần như đều: normalized entropy `0.937`, trọng số ba
  scale `0.349 / 0.329 / 0.322`. Gate không chọn được scale.
- `gamma` học tăng từ `0.001` lên khoảng `0.10-0.12`; tỉ lệ
  `RMS(update)/RMS(input)` khoảng `3.1-3.3%`. MSR không phải identity, nhưng
  hướng cập nhật không cải thiện AP75.
- Gate modulation trung bình khoảng `0.47-0.57`, không bão hòa. Vấn đề không
  phải gate bị kẹt, mà là evidence đưa vào chưa đủ chính xác.
- Mọi fusion mode đều dưới baseline ở AP75. Tinh chỉnh fusion thêm không còn
  biện minh từ dữ liệu.

## Giả thuyết ưu tiên

### H1: giảm khuếch đại evidence ở vùng nền phẳng

MSR chuẩn hóa residual theo độ lệch chuẩn cục bộ. Ở vùng phẳng, mẫu số nhỏ nên
`z` dễ chạm `z_max=10`, biến nhiễu nhỏ thành evidence mạnh. Hạ `z_max` xuống
`3.0` là thay đổi một biến, không đổi tham số, graph hay fusion.

Dự đoán: precision và AP75 tăng; recall tiny1 không giảm quá 1 điểm.

### H2: tăng sàn phương sai

Thay `sigma_min` từ `1e-3` lên `1e-2` cũng làm giảm khuếch đại ở vùng phẳng,
nhưng theo cơ chế khác H1. Chạy như treatment riêng, không gộp với H1.

### H3: precision gate hoặc nhánh phụ

Nếu H1/H2 thất bại, không tinh chỉnh fusion nữa. Hướng tiếp theo phải đổi vai
trò của MSR:

- gate evidence theo độ tin cậy/contrast thay vì chỉ theo `z`;
- dùng MSR như nhánh phụ/deep supervision, giữ nguyên P2 chính;
- hoặc chuyển sang box regression/assignment, nơi AP75 đang là nút thắt.

## Protocol thí nghiệm

Config: `configs/experiment/msr_evidence_robustness.yaml`.

- Control: MSR hiện tại, `z_max=10`, `sigma_min=1e-3`, fusion adaptive.
- Treatment 1: `configs/model/yolo26n_tiny_A2_p2_msr_z3.yaml`.
- Treatment 2: `configs/model/yolo26n_tiny_A2_p2_msr_sigma01.yaml`.
- Dataset, seed, epoch, imgsz, batch, optimizer, augmentation và diagnostics
  giữ nguyên.
- Chạy seed 42 trước. Chỉ mở `seeds: [42, 43, 44]` nếu một treatment vượt
  control ở AP75 và không đánh đổi recall tiny1.

Metric quyết định:

1. AP75 validation là metric chính.
2. AP, AP50, precision, recall và tiny1 recall là điều kiện đi kèm.
3. Phân rã FP background/localization/duplicate và latency/VRAM.
4. AP75 official test chỉ dùng sau khi khóa checkpoint, không dùng để tune.

Tiêu chí dừng: nếu cả H1 và H2 không tăng AP75 ít nhất `+0.002` tuyệt đối so
với control, dừng tinh chỉnh evidence normalization và chuyển sang H3/H4.

## Lệnh chạy

```bash
modal run --detach modal/app.py::compare \
  --config configs/experiment/msr_evidence_robustness.yaml --parallel
```

Chưa chạy thí nghiệm này. Không có kết luận cải tiến nào trước khi có artifact.
