# SADR-P2: giả thuyết distributional box regression cho tiny object

**Trạng thái:** SADR8 (8 bin) đã có kết quả paired seed 42: validation AP75
tăng, official-test AP mean/AP50/recall tăng, AP75 tổng gần như đi ngang và
giảm nhẹ. Chưa đủ để gọi là thành công theo gate AP75. Ablation 4/16 bin đang
chờ GPU. SADR ở đây là tên ablation *Scalar-to-Distributional Anchor Distances*,
không khẳng định mới về mặt thuật toán: DFL là thành phần có sẵn trong
Ultralytics. Câu hỏi mới trong dự án là liệu nó có giúp regression trên
YOLO26n-P2 với tiny object không.

## Vấn đề đo được

Baseline A2 trên official TinyPerson test (một seed) đạt AP@0.25 = 0.40127,
AP@0.50 = 0.23014, AP@0.75 = 0.02374; riêng tiny1 AP@0.75 = 0.00345.
Ở IoU 0.50 có 39.121 localization FP và 6.544 miss trên 13.687 positive GT.
BTRM và MSR warm-start đều giảm official AP75 so với control 0.02472:
lần lượt 0.01953 và 0.01986. NWD cũ trên một protocol P2 khác cũng giảm
validation AP75 0.02179 → 0.01863. Các số này chỉ định hướng giả thuyết,
không được trộn làm paired comparison.

YOLO26n hiện dùng `reg_max=1`: mỗi biên dự đoán một khoảng cách thực đến
anchor, tối ưu CIoU + L1 chuẩn hoá theo kích thước ảnh. Với ảnh 1280 px,
sai số 1 px trên mỗi tọa độ chỉ đóng góp `1/1280` vào L1 ngang, nhưng trên
box vuông cạnh 8 px, dịch ngang 1 px đã làm IoU giảm từ 1 xuống
`7/9 ≈ 0.778`; dịch 2 px làm IoU còn `6/10 = 0.6`. Muốn IoU ≥ 0.75 khi
chỉ dịch một trục, phải có `|δ|/w ≤ 1/7`, tức ≤1.14 px ở 8 px hoặc ≤0.57 px
ở 4 px. Đây là phân tích độ nhạy hình học, không phải dự báo AP.

## Can thiệp nhỏ nhất có thể kiểm chứng

Thay `reg_max=1` bằng `reg_max=8` trong cùng graph A2 (P2/P3/P4/P5,
strides 4/8/16/32). Cho mỗi cạnh và anchor, head dự đoán logits `z_k`
với `k=0..7`, khoảng cách liên tục bằng kỳ vọng
`d = Σ_k k · softmax(z)_k`. Với target khoảng cách `t` tính theo ô feature,
DFL tối ưu hai bin kề `⌊t⌋, ⌈t⌉`, với trọng số tuyến tính; CIoU được giữ
nguyên. Ultralytics dùng `bbox2dist` clamp target vào `[0, 6.99]` trên
*từng level*; do đó box cần khoảng cách >7 ô có thể bị bóp cụt nhánh DFL.
Đó là rủi ro đáng kể ở P2 đối với object lớn và ở P3/P4/P5 đối với box rộng.
Toàn head đổi từ 1 sang 8 bin, **không** phải module riêng P2; không được
claim là scale-adaptive cho đến khi có head per-level thật.

Mô hình thêm 201.960 tham số (2.515.976 → 2.717.936, +8.0%), không thêm
feature block. Đối chứng và treatment train scratch 50 epoch, seed 42, cùng
dataset `tinyperson_official_v1`, imgsz 1280, augmentation và hyperparameter;
control `tiny_A2_p2/20261019-194809-8326468-seed42` chỉ tái dùng khi
config và signature khớp. Chọn `best.pt` trên validation, rồi đánh giá test
với cùng `conf=0.001`, `max_det=200` và evaluator nội bộ. Cấu hình:
`configs/experiment/localization_sadr_tinyperson.yaml`.

## Kết quả SADR8, seed 42

Validation (`best.pt`, 84 ảnh):

| Metric | A2 control | SADR8 | Delta |
|---|---:|---:|---:|
| AP | 0.059119 | 0.068522 | +0.009403 |
| AP50 | 0.196888 | 0.213177 | +0.016289 |
| AP75 | 0.015852 | 0.019406 | +0.003554 |
| Precision | 0.358155 | 0.354153 | -0.004002 |
| Recall | 0.252494 | 0.279933 | +0.027439 |

Official test (786 ảnh, cùng evaluator/protocol):

| Metric | A2 control | SADR8 | Delta |
|---|---:|---:|---:|
| AP@0.25 | 0.401268 | 0.425920 | +0.024652 |
| AP@0.50 | 0.230141 | 0.258871 | +0.028730 |
| AP@0.75 | 0.023745 | 0.023302 | -0.000443 |
| AP mean | 0.218384 | 0.236031 | +0.017647 |
| tiny1 AP75 | 0.003452 | 0.005788 | +0.002336 |
| tiny2 AP75 | 0.011632 | 0.018344 | +0.006712 |
| tiny3 AP75 | 0.022933 | 0.023428 | +0.000495 |
| small AP75 | 0.032881 | 0.035170 | +0.002289 |
| reasonable AP75 | 0.041583 | 0.045630 | +0.004047 |

Error analysis tại IoU 0.50: TP `7.143 → 7.638`, miss `6.544 → 6.049`,
background FP `84.359 → 78.208`, localization FP `39.121 → 38.549`.
Đếm GT đạt IoU ≥ 0.75 trực tiếp từ samples: `1.877 → 2.177` (+300).
Vì vậy AP75 tổng giảm nhẹ **không** do box kém hơn; đó là thay đổi ranking
confidence giữa các mức IoU. AP75 “tiny” gộp cũng giảm nhẹ
`0.017398 → 0.016066` dù tiny1/tiny2/tiny3 riêng lẻ đều tăng; AP theo size-bin
không phân rã được vì detection ngoài range được xử lý neutral.
Benchmark L4 batch 1: 16.06 ms (A2 control 16.44 ms), 260.9 MB.

## Tiêu chí kết luận và tiếp theo

- Primary: validation AP75, sau đó official-test AP75 và AP mean tăng *đồng thời*;
  không chấp nhận trade AP75 lấy AP50. Kiểm tra tiny1/tiny2/tiny3 không sụt
  mạnh, latency/VRAM và độ phân tán từ paired seeds 42/43/44.
- Nếu AP75 tăng nhưng mean giảm: không coi là thành công; phân tích DFL clamp
  và từng size-bin. Nếu không tăng: dừng DFL, thử proposal assignment hoặc
  refine chỉ sau khi định lượng phân bố độ lệch box trên **validation**.
- Kiểm định tính tổng quát: cùng hypothesis trên benchmark tiny-object nhiều
  lớp khác (không dùng chỉ TinyPerson); báo cả AP-small và AP75 theo lớp/bin.
  Một seed TinyPerson không đủ kết luận “cải thiện tiny object chung”.
- Reference evaluator TinyPerson chưa được pin trong repo; official-test số
  từ evaluator nội bộ chỉ có giá trị trong protocol nhất quán của dự án.
