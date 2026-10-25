# BTRM trên P2: module localization cho tiny person

Tổng hợp đầy đủ, gồm phân tích lỗi/ngưỡng/localization rerun:
[BTRM_REPORT.md](BTRM_REPORT.md).

## Câu hỏi

Benchmark TinyPerson cho thấy AP giảm từ 40,13% ở IoU 0.25 xuống 2,37% ở IoU
0.75. 51,1% miss có detection chồng lấn IoU 0.1-0.5, tức model đã phát hiện
đúng vùng nhưng box chưa chính xác. Đây là nút thắt localization, không phải
recall thuần. BTRM kiểm tra giả thuyết: loại bỏ trend nền cục bộ trên feature P2
trước Detect sẽ làm biên vật thể nhỏ rõ hơn và cải thiện AP75/tiny1.

## Thiết kế v2

`src/tinydet/modules/btrm.py` là module shape-preserving, đặt trên P2. Thiết kế
v2 thay phép trừ nền đơn scale bằng contrast đa scale và gradient có hướng:

\[
E=[X-M_3,\ X-M_5,\ G_x,\ G_y,\ \lVert G\rVert],\quad
P=\operatorname{MLP}(E),\quad S=\sigma(\operatorname{Conv}_{1\times1}(P))
\]

\[
Y=X+\gamma\,S\odot P
\]

- `M_3`, `M_5` là trung bình cục bộ 3x3 và 5x5 với replicate padding.
- `G_x`, `G_y` là central difference ngang/dọc; `|G| = |G_x| + |G_y|`.
- Projection 1x1 + BN + SiLU + 1x1 học cách chọn evidence; gate sigmoid theo
  kênh/vị trí; `gamma` khởi tạo 0 nên `Y == X` chính xác tại initialization.
- Evidence được tính bằng dtype hiện tại của feature map để tương thích AMP;
  output giữ dtype gốc.
- Với C=32, module thêm 2.609 tham số.
- Module không đổi shape `[B,32,H,W]`, chỉ thay đổi nhánh P2 trước Detect;
  P3/P4/P5, loss và label assignment giữ nguyên.

## Bất biến kiến trúc

- Model: `configs/model/yolo26n_tiny_A2_p2_btrm.yaml`.
- BTRM là layer 29, nhận P2 từ layer 19.
- Detect nhận `[29, 22, 25, 28]` theo đúng thứ tự P2, P3, P4, P5.
- Strides phải là `[4, 8, 16, 32]`; `tests/test_btrm.py` kiểm tra graph,
  checkpoint roundtrip và CPU predict.
- Baseline A2 chỉ khác ở chỗ không có layer BTRM; mọi layer khác giữ nguyên.

## Protocol cố định

Hai file experiment là paired comparison, baseline A2 được giữ cố định:

| Dataset | Config | Baseline | Treatment |
|---|---|---|---|
| TinyPerson | `configs/experiment/localization_btrm_tinyperson.yaml` | `tiny_A2_p2` | `tiny_A2_p2_btrm` |
| VisDrone | `configs/experiment/localization_btrm_visdrone.yaml` | `visdrone_A2_p2` | `visdrone_A2_p2_btrm` |

Siêu tham số giống hệt `ablation_msr_p2.yaml`: seed 42, 50 epoch, patience 15,
imgsz 1280, batch 8, AdamW, lr0 0.001, AMP và deterministic bật, augmentation
giống nhau. Chỉ `data.config` và tên arm khác giữa hai dataset. Không đổi NMS,
không đổi evaluator, không đổi split. Khi cần variance, thay `seed` bằng
`seeds: [42, 43, 44]` và giữ nguyên mọi trường khác.

Diagnostics tắt cho cả baseline và treatment vì BTRM chưa có collector riêng;
hai arm vì vậy vẫn dùng cùng một config ngoài model.

## Dữ liệu

- TinyPerson: `/mnt/data/tinyperson_official_v1/data.yaml` trong Modal volume
  `tiny-person-data`.
- VisDrone: `/mnt/data/visdrone_official_v1/data.yaml`, phải là derivative
  một class (`pedestrian` hoặc `people`) nằm trong cùng volume. Chưa có trong
  repo; cần chuẩn bị và upload trước khi chạy.
- Không dùng AI-TOD/AI-TOD-v2 cho cross-validation person-only vì hai bộ này
  không có class person.

## Chạy

```bash
uv run --extra cpu python scripts/validate_tiny_ablation.py \
  configs/model/yolo26n_tiny_A2_p2_btrm.yaml --imgsz 640
uv run --extra cpu pytest tests/test_btrm.py
modal run --detach modal/app.py::compare \
  --config configs/experiment/localization_btrm_tinyperson.yaml --parallel
```

Chọn checkpoint theo validation, báo metric từ `best.pt`. Metric chính là AP75;
báo kèm mAP, AP50, recall tiny1 và decomposition miss/FP. Không chọn model bằng
official test.

## Kết quả paired trên TinyPerson

Protocol: official_v1, seed 42, 50 epoch, batch 8, mosaic 0, patience 15.
Control là run đầy đủ `tiny_A2_p2/20261019-194809-8326468-seed42`; treatment là
`tiny_A2_p2_btrm/20261023-233359-8efefb4-seed42`. Cả hai dùng `best.pt` theo
validation; official test chỉ dùng để báo cáo cuối.

Lưu ý: đây là so sánh scratch. Vì BTRM nằm trước Detect, việc chèn module làm
thay đổi chuỗi RNG khởi tạo của Detect head, nên chênh lệch scratch không phải
ước lượng nhân quả sạch cho riêng module. Phần adaptation bên dưới kiểm soát
điểm này bằng cách cho cả hai arm khởi tạo từ cùng checkpoint A2.

Validation:

| Run | AP | AP50 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| A2 baseline | 0.059119 | 0.196888 | 0.015852 | 0.358155 | 0.252494 |
| A2 + BTRM v2 | 0.058988 | 0.196493 | 0.012272 | 0.340439 | 0.265798 |
| Delta | -0.000131 | -0.000395 | -0.003580 | -0.017716 | +0.013304 |

Official TinyPerson test, 786 ảnh sau khi loại dense:

| Bin | AP75 baseline | AP75 BTRM | Delta | AP mean baseline | AP mean BTRM |
|---|---:|---:|---:|---:|---:|
| all | 0.023745 | 0.020200 | -0.003545 | 0.218384 | 0.202979 |
| tiny | 0.017398 | 0.010198 | -0.007201 | 0.198792 | 0.187280 |
| tiny1 | 0.003452 | 0.005055 | +0.001603 | 0.140160 | 0.123638 |
| tiny2 | 0.011632 | 0.015676 | +0.004044 | 0.220722 | 0.210255 |
| tiny3 | 0.022933 | 0.015623 | -0.007311 | 0.232133 | 0.222256 |
| small | 0.032881 | 0.024034 | -0.008847 | 0.246987 | 0.225260 |
| reasonable | 0.041583 | 0.039157 | -0.002426 | 0.275387 | 0.250684 |

AP75 tăng ở tiny1/tiny2 nhưng giảm mạnh hơn ở tiny3/small; AP trung bình và
AP50 giảm ở mọi bin. Vì vậy không có bằng chứng BTRM cải thiện localization
tổng thể.

Decomposition trên official test:

| Count | Baseline | BTRM v2 |
|---|---:|---:|
| true positives | 7.143 | 6.997 |
| misses | 6.544 | 6.690 |
| false positives | 133.941 | 136.956 |
| localization FP | 39.121 | 37.684 |
| background FP | 84.359 | 89.045 |
| duplicate FP | 10.461 | 10.227 |

BTRM giảm 1.437 localization FP nhưng tăng 4.686 background FP và mất 146 TP;
recall official test giảm từ 52,19% xuống 51,12%. Trên validation, recall tăng
1,33 điểm nhưng precision giảm 1,77 điểm, cho thấy lợi ích recall không ổn định
qua split.

Chi phí trên L4, batch 1, ảnh 1280:

| Metric | Baseline | BTRM v2 | Delta |
|---|---:|---:|---:|
| parameters | 2.515.976 | 2.518.585 | +2.609 |
| latency | 16,37 ms | 17,62 ms | +7,6% |
| FPS | 61,08 | 56,76 | -7,1% |
| benchmark GPU memory | 214,8 MB | 344,3 MB | +60,3% |
| training time | 1.781 s | 2.515 s | +41,2% |

Kết luận: giả thuyết "local contrast/edge trên P2 sẽ cải thiện AP75" bị bác bỏ
trong protocol hiện tại. Module dịch lỗi từ localization sang background FP
nhiều hơn lượng lỗi localization loại bỏ được. Không nên thêm nhánh hoặc tăng
dung lượng BTRM trước khi có thí nghiệm mới chứng minh tín hiệu biên không
khuếch đại nền.

## Adaptation kiểm soát initialization

Để loại bỏ confound khởi tạo, cả control và treatment cùng load
`tiny_A2_p2/20261019-194809-8326468-seed42/checkpoints/best.pt`, rồi fine-tune
10 epoch với `lr0=0.0002`. BTRM vẫn khởi tạo `gamma=0`, nên tại bước đầu hai
model có cùng output. Config: `configs/experiment/localization_btrm_adapt_tinyperson.yaml`.

Validation:

| Run | AP | AP50 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| A2 warm-start, control | 0.059767 | 0.197801 | 0.013398 | 0.335431 | 0.258315 |
| A2 + BTRM v2 warm-start | 0.059162 | 0.196943 | 0.013371 | 0.330918 | 0.264135 |
| Delta | -0.000606 | -0.000858 | -0.000027 | -0.004514 | +0.005820 |

Official TinyPerson test:

| Run | AP@0.25 | AP@0.50 | AP@0.75 | AP mean |
|---|---:|---:|---:|---:|
| A2 warm-start, control | 0.40260 | 0.23431 | 0.02472 | 0.22054 |
| A2 + BTRM v2 warm-start | 0.40275 | 0.23355 | 0.01953 | 0.21861 |
| Delta | +0.00015 | -0.00076 | -0.00519 | -0.00193 |

AP75 vẫn giảm 21% tương đối dù hai arm bắt đầu từ cùng trọng số. BTRM tăng nhẹ
tiny1 AP75 (`0.00552 → 0.00585`) nhưng làm giảm mạnh `reasonable`
(`0.04240 → 0.03690`), cho thấy module tác động lên cả vật thể lớn hơn qua
feature P2 thay vì chỉ tinh chỉnh biên vật thể nhỏ.

Chi phí warm-start trên L4: latency `16,44 → 18,74 ms` (+14,0%), FPS
`60,81 → 53,35` (-12,3%), benchmark memory `214,8 → 315,1 MB` (+46,7%), cho
2.609 tham số thêm.

Kiểm tra cùng một checkpoint với `gamma` học được và `gamma=0`:

| Scratch BTRM checkpoint | AP@0.50 | AP@0.75 | AP mean |
|---|---:|---:|---:|
| learned gamma = 0.1226 | 0.21195 | 0.02020 | 0.20298 |
| gamma = 0 | 0.20564 | 0.02095 | 0.20212 |

Residual học được tăng AP50 nhưng giảm AP75 trên cùng trọng số. Đây là tác động
module nhỏ và không có lợi cho localization.

## Vì sao tăng độ phức tạp chưa hợp lý

Rerun localization error trên official test (score >= 0.05, IoU 0.1-0.5) với
định nghĩa `pred_width / matched_GT_width` cho thấy box lỗi trung vị **hẹp hơn**
GT: tiny1 `0.747` (A2) và `0.842` (BTRM scratch); tiny2 `0.883` và `0.985`.
Center offset trung vị khoảng `0.24-0.26` cạnh GT ở cả hai. BTRM scratch kéo
width ratio gần 1 nhưng không giảm center offset; warm-start gần như không đổi.
Bảng đầy đủ nằm trong [BTRM_REPORT.md](BTRM_REPORT.md).

Lỗi còn lại là regression/assignment và scale của box, không phải thiếu độ
tương phản hay edge detail. Vì vậy thêm channel, depthwise conv hoặc attention
vào cùng họ BTRM sẽ làm tăng chi phí nhưng không giải quyết đúng tầng lỗi.
Nghiên cứu tiếp theo nên nhắm vào box regression/assignment (ví dụ
scale-aware regression, RFLA/NWD hoặc một refinement head có huấn luyện), không
phải tiếp tục làm BTRM phức tạp hơn.

## Giới hạn

- Kết quả trên là một seed; chưa có mean/std qua nhiều seed.
- Bán kính cơ sở cố định bằng 1 pixel feature (4 px ảnh ở P2), tương ứng cửa sổ
  3x3 và 5x5. Nếu vật thể nhỏ hơn hoặc nền có gradient mạnh hơn, cần thử radius
  khác trong một ablation riêng.
- BTRM chỉ chuẩn hóa tương phản cục bộ; nó không sửa regression head, không thay
  NMS và không giải quyết trực tiếp giới hạn `max_det=200`.

Xem thêm [phân tích BTRM cũ và MSR](BTRM_MSR_ANALYSIS.md).
