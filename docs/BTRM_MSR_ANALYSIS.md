# Phân tích BTRM cũ và tiềm năng MSR

Tài liệu này tách hai câu hỏi: vì sao BTRM cũ thua baseline, và MSR hiện tại có
phải là hướng localization hay không. Số liệu lấy từ artifact trên Modal Volume
`tiny-person-runs`; không dùng W&B curve làm nguồn chính.

## BTRM cũ: kết quả và nguyên nhân

Ba run `btrm_a0` (không module), `btrm_a2` (contrast), `btrm_a3` (BTRM) dùng
cùng graph `configs/model/yolo26n-btrm.yaml`, dataset `/mnt/data/tinyperson/data.yaml`,
batch 16, mosaic 1.0, HSV 0.015/0.7/0.4, patience 50. Đây là protocol khác
baseline A2 hiện tại (batch 8, mosaic 0, HSV 0.005/0.1/0.15, patience 15), nên
không được so trực tiếp với bảng A2/MSR.

| Run | AP | AP50 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| btrm_a0, no module | 0.07090 | 0.24525 | 0.01814 | 0.35423 | 0.32837 |
| btrm_a2, contrast | 0.06990 | 0.24430 | 0.01643 | 0.34783 | 0.32683 |
| btrm_a3, BTRM | 0.06825 | 0.23440 | 0.01662 | 0.34931 | 0.32887 |

Trong cùng protocol cũ, BTRM làm AP giảm 0.00265 và AP75 giảm 0.00152. Theo
size bin, AP50 của `8-16 px` giảm từ 0.07547 xuống 0.05765, trong khi `>32 px`
tăng từ 0.22491 lên 0.23059. Đây là dấu hiệu module loại bỏ nền nhưng làm mất
tín hiệu vật thể nhỏ.

Nguyên nhân thiết kế:

- Module nằm trên P3 (stride 8), trong khi tiny1 chỉ rộng 1-2 ô ở P2.
- Evidence là một map đơn kênh; gain là scalar bị chặn. Module gần như không có
  khả năng học biểu diễn theo kênh hay theo vị trí.
- Phép trừ nền cứng loại bỏ cả low-frequency của vật thể, đúng với lỗi tăng
  mạnh ở bin 8-16 px.
- Protocol khác baseline hiện tại; không thể dùng kết quả cũ để kết luận cho
  BTRM P2 mới.

## MSR: bằng chứng hiện có

MSR-P2 là ablation E0-E3 trên cùng dataset official_v1, seed 42, batch 8,
mosaic 0. Metric từ `best.pt` trên validation:

| Run | AP | AP50 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| E0, A2 baseline | 0.059119 | 0.196888 | 0.015852 | 0.358155 | 0.252494 |
| E1, A2 + MSR | 0.059755 | 0.191554 | 0.015836 | 0.332923 | 0.272173 |
| E2, A3 no SPPF | 0.055914 | 0.179919 | 0.014362 | 0.333259 | 0.260532 |
| E3, A3 + MSR | 0.052938 | 0.175762 | 0.012215 | 0.298083 | 0.252772 |

Diễn giải:

- E1 tăng recall 1.97 điểm nhưng giảm precision 2.52 điểm, AP50 giảm 0.53 điểm;
  AP75 gần như không đổi (-0.000016). MSR đang đổi precision lấy recall, không
  cải thiện chất lượng box.
- E3 giảm cả AP, AP50, AP75 và recall, nên MSR không ổn định khi bỏ SPPF.
- Các fusion mode cũng không giúp AP75: adaptive 0.01572, mean 0.01400,
  sk_softmax 0.01309, max 0.01023 trên run family `msr_fusion_*`; tất cả đều
  dưới baseline 0.01585.

Kết luận: MSR có tiềm năng như module tăng recall/nhạy cảm với vật thể nhỏ, không
phải module localization. Hướng cải tiến hợp lý là thêm cổng precision (chỉ
khuếch đại evidence gần biên/vật thể) hoặc dùng MSR như nhánh phụ, không dùng
local-variance normalization trần vì nó khuếch đại nền nhiễu. Chưa có bằng chứng
MSR cải thiện AP75.

## BTRM v2 trên P2: kết quả

Thiết kế mới thay thế phép trừ nền đơn scale bằng biểu diễn biên đa scale:

\[
M_1=\operatorname{Avg}_{3}(X),\quad M_2=\operatorname{Avg}_{5}(X),\quad
G_x=\partial_x X,\quad G_y=\partial_y X
\]

\[
E=[X-M_1,\ X-M_2,\ G_x,\ G_y,\ \lVert G\rVert],\quad
P=\operatorname{MLP}(E),\quad S=\sigma(\operatorname{Conv}_{1\times1}(P))
\]

\[
Y=X+\gamma\,S\odot P
\]

- P2, shape-preserving, side branch trước Detect; P3/P4/P5 không đổi.
- Có cả contrast gần (3x3) và xa (5x5), cùng gradient ngang/dọc để mã hóa biên.
- Gate theo kênh và vị trí; `gamma=0` nên identity chính xác lúc init.
- Với C=32, module thêm 2.609 tham số (baseline 2.515.976; treatment 2.518.585).
- Đã chạy paired 50 epoch trên TinyPerson official_v1, seed 42. Kết quả đầy đủ
  nằm trong [BTRM_P2.md](BTRM_P2.md).

| Metric | A2 baseline | A2 + BTRM v2 | Delta |
|---|---:|---:|---:|
| Validation AP | 0.059119 | 0.058988 | -0.000131 |
| Validation AP50 | 0.196888 | 0.196493 | -0.000395 |
| Validation AP75 | 0.015852 | 0.012272 | -0.003580 |
| Validation precision | 0.358155 | 0.340439 | -0.017716 |
| Validation recall | 0.252494 | 0.265798 | +0.013304 |
| Official-test AP@0.75 | 0.023745 | 0.020200 | -0.003545 |
| Official-test AP mean | 0.218384 | 0.202979 | -0.015405 |
| Official-test recall | 0.521882 | 0.511215 | -0.010667 |

Trên official test, BTRM giảm 1.437 localization FP nhưng tăng 4.686 background
FP và mất 146 TP. AP75 chỉ tăng ở tiny1/tiny2, trong khi tiny3 và small giảm đủ
mạnh để kéo AP75 tổng thể xuống. Đây không phải là bằng chứng localization được
cải thiện; module chỉ dịch một phần lỗi hộp sang false positive nền.

Chi phí: +2.609 tham số nhưng latency tăng 7,6%, FPS giảm 7,1%, bộ nhớ benchmark
tăng 60,3% và thời gian train tăng 41,2% trên L4. Với một seed, kết luận hợp lý
nhất là bác bỏ hướng BTRM hiện tại, không phải tinh chỉnh thêm cùng lúc nhiều
biến.

## Protocol kiểm chứng

- Control: `tiny_A2_p2/20261019-194809-8326468-seed42` (đã kiểm tra config khớp
  hoàn toàn với experiment mới, nên `--reuse` dùng lại được).
- Treatment: `tiny_A2_p2_btrm/20261023-233359-8efefb4-seed42`, cùng data, seed,
  epoch, batch, augmentation.
- Metric chính: AP75, sau đó AP50/AP, recall tiny1, precision và decomposition
  localization/background FP.
- Run treatment ban đầu `20261023-232551-8efefb4-seed42` thất bại ở validation
  cuối vì lỗi AMP dtype; run này không được dùng làm bằng chứng. Run
  `20261023-233359-8efefb4-seed42` là run đã sửa và hoàn tất.
- Official test dùng cùng `imgsz=1280`, `conf=0.001`, `max_det=200`, 816 ảnh;
  evaluator loại 30 ảnh dense và chấm 786 ảnh.

## Kết luận về MSR

Hai hướng hiện có cùng một dấu hiệu: module bổ sung evidence làm tăng recall
hoặc độ nhạy nhưng không tăng AP75. Với MSR, E1 tăng recall 1,97 điểm trong khi
AP75 gần như không đổi và precision giảm 2,52 điểm; các fusion mode đều dưới
baseline AP75. Với BTRM v2, recall validation tăng nhưng AP75 giảm 0,00358 và
official test giảm 0,00355.

Tiềm năng của MSR vì vậy nằm ở vai trò nhánh recall/evidence, không phải module
localization. Muốn biến nó thành cải thiện AP75 cần một cơ chế chọn evidence có
precision cao hơn (ví dụ gate theo độ tin cậy của box hoặc dùng MSR như nhánh
phụ trong lúc train), và phải đánh giá bằng paired AP75 trước khi thêm phức tạp.
Hiện tại chưa có bằng chứng cho thấy MSR cải thiện AP75. Phần adaptation kiểm
soát initialization bên dưới xác nhận lại kết luận này.

## MSR adaptation kiểm soát initialization

Để tách tác động MSR khỏi khác biệt khởi tạo, control và treatment cùng load
checkpoint A2 baseline và fine-tune 10 epoch với `lr0=0.0002`
(`configs/experiment/localization_msr_adapt_tinyperson.yaml`). Control dùng lại
đúng run `tiny_A2_p2_adapt/20261024-005159-8efefb4-seed42`; treatment là
`tiny_A2_p2_msr_adapt/20261024-011446-8efefb4-seed42`.

Validation:

| Run | AP | AP50 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| A2 warm-start, control | 0.059767 | 0.197801 | 0.013398 | 0.335431 | 0.258315 |
| A2 + MSR warm-start | 0.058721 | 0.199970 | 0.012733 | 0.353529 | 0.259424 |
| Delta | -0.001046 | +0.002169 | -0.000665 | +0.018098 | +0.001109 |

Official TinyPerson test:

| Run | AP@0.25 | AP@0.50 | AP@0.75 | AP mean |
|---|---:|---:|---:|---:|
| A2 warm-start, control | 0.40260 | 0.23431 | 0.02472 | 0.22054 |
| A2 + MSR warm-start | 0.39719 | 0.23229 | 0.01986 | 0.21645 |
| Delta | -0.00541 | -0.00202 | -0.00486 | -0.00409 |

MSR tăng AP75 ở tiny1 (`0.00552 → 0.00615`) và tiny3 (`0.02298 → 0.02424`)
nhưng giảm mạnh ở small (`0.03367 → 0.02689`) và reasonable
(`0.04240 → 0.04028`). Trên validation, MSR tăng precision 1,81 điểm và AP50
0,22 điểm nhưng vẫn giảm AP75. Đây là hành vi của một module evidence/recall có
thể cải thiện chất lượng xếp hạng ở IoU thấp, không phải module localization.

Chi phí warm-start trên L4: latency `16,44 → 19,85 ms` (+20,7%), FPS
`60,81 → 50,37` (-17,2%), benchmark memory `214,8 → 293,8 MB`, với 9.337 tham
số thêm.

Đánh giá: MSR có tiềm năng như một nhánh phụ tăng recall/precision ở IoU thấp,
đặc biệt cho tiny1/tiny3. Chưa có bằng chứng nó cải thiện AP75 tổng thể; để dùng
trong nghiên cứu localization cần một cơ chế chọn evidence hoặc refinement riêng,
và phải giữ AP75 của small/reasonable.
