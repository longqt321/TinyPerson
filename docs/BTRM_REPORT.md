# BTRM: tổng hợp kết quả và phân tích lỗi

## Phạm vi

Báo cáo gom toàn bộ bằng chứng hiện có cho BTRM v2 trên P2:

| Nhóm | Run/artifact |
|---|---|
| BTRM cũ | `btrm_a0`, `btrm_a2`, `btrm_a3`; protocol `tinyperson` cũ |
| BTRM v2 scratch | `tiny_A2_p2_btrm/20261023-233359-8efefb4-seed42` |
| BTRM v2 warm-start | `tiny_A2_p2_btrm_adapt/20261024-005202-8efefb4-seed42` |
| Control scratch | `tiny_A2_p2/20261019-194809-8326468-seed42` |
| Control warm-start | `tiny_A2_p2_adapt/20261024-005159-8efefb4-seed42` |
| Gamma=0 ablation | Cùng BTRM scratch checkpoint, ép `gamma=0` |

Official test dùng 786 ảnh sau khi loại 30 ảnh dense, `conf=0.001`,
`max_det=200`, evaluator nội bộ của repo. Đây không phải reference evaluator
được pin.

## Kết quả AP

### Official TinyPerson test

| Metric | A2 | A2+BTRM scratch | A2 adapt | A2+BTRM adapt | BTRM gamma=0 |
|---|---:|---:|---:|---:|---:|
| AP@0.25 | 0.401268 | 0.376787 | 0.402600 | 0.402747 | 0.379760 |
| AP@0.50 | 0.230141 | 0.211951 | 0.234307 | 0.233549 | 0.205640 |
| AP@0.75 | 0.023745 | 0.020200 | 0.024719 | 0.019529 | 0.020947 |
| AP mean | 0.218384 | 0.202979 | 0.220542 | 0.218608 | 0.202116 |

### AP75 theo size bin

| Bin | A2 | BTRM scratch | A2 adapt | BTRM adapt | BTRM gamma=0 |
|---|---:|---:|---:|---:|---:|
| all | 0.023745 | 0.020200 | 0.024719 | 0.019529 | 0.020947 |
| tiny | 0.017398 | 0.010198 | 0.018096 | 0.018067 | 0.009314 |
| tiny1 | 0.003452 | 0.005055 | 0.005519 | 0.005851 | 0.002238 |
| tiny2 | 0.011632 | 0.015676 | 0.017893 | 0.017737 | 0.013795 |
| tiny3 | 0.022933 | 0.015623 | 0.022976 | 0.023013 | 0.014075 |
| small | 0.032881 | 0.024034 | 0.033673 | 0.033899 | 0.026315 |
| reasonable | 0.041583 | 0.039157 | 0.042397 | 0.036896 | 0.043566 |

Scratch tăng AP75 ở tiny1/tiny2 nhưng giảm mạnh tiny3/small. Warm-start gần như
giữ tiny1/tiny2/tiny3/small nhưng AP75 tổng giảm vì `reasonable` giảm
`0.042397 → 0.036896`.

### Validation

| Run | AP | AP50 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| A2 scratch | 0.059119 | 0.196888 | 0.015852 | 0.358155 | 0.252494 |
| BTRM scratch | 0.058988 | 0.196493 | 0.012272 | 0.340439 | 0.265798 |
| A2 adapt | 0.059767 | 0.197801 | 0.013398 | 0.335431 | 0.258315 |
| BTRM adapt | 0.059162 | 0.196943 | 0.013371 | 0.330918 | 0.264135 |

## Phân tích lỗi rerun

Nguồn: `outputs/error_analysis/*-test/summary.json`, `samples.jsonl`,
`errors.jsonl`. Primary IoU `0.50`, localization FP dùng IoU `0.10-0.50`,
score threshold `0.001`.

| Run | TP | Miss | FP | Loc FP | Background FP | Duplicate FP | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 | 7.143 | 6.544 | 133.941 | 39.121 | 84.359 | 10.461 | 0.521882 | 0.050629 |
| BTRM scratch | 6.997 | 6.690 | 136.956 | 37.684 | 89.045 | 10.227 | 0.511215 | 0.048606 |
| A2 adapt | 7.208 | 6.479 | 128.487 | 39.910 | 77.917 | 10.660 | 0.526631 | 0.053119 |
| BTRM adapt | 7.209 | 6.478 | 128.321 | 39.950 | 77.720 | 10.651 | 0.526704 | 0.053191 |
| BTRM gamma=0 | 6.811 | 6.876 | 130.048 | 36.739 | 83.239 | 10.070 | 0.497625 | 0.049767 |

Scratch: BTRM giảm `1.437` localization FP nhưng tăng `4.686` background FP,
mất `146` TP. Warm-start: gần như không đổi ở mức tổng (`-166` FP, `-1` miss),
trong khi AP75 vẫn giảm. Gamma=0: giảm `6.908` FP so với BTRM learned gamma
nhưng mất thêm `186` miss và giảm mạnh recall.

| So sánh paired | Miss delta | FP delta |
|---|---:|---:|
| BTRM scratch vs A2 | +146 | +3.015 |
| BTRM adapt vs A2 adapt | -1 | -166 |
| BTRM gamma=0 vs BTRM learned gamma | +186 | -6.908 |

### Tỷ lệ lỗi

| Run | Loc share FP | Background share FP | Duplicate share FP |
|---|---:|---:|---:|
| A2 | 0.292076 | 0.629822 | 0.078102 |
| BTRM scratch | 0.275154 | 0.650172 | 0.074674 |
| A2 adapt | 0.310615 | 0.606419 | 0.082966 |
| BTRM adapt | 0.311329 | 0.605669 | 0.083003 |
| BTRM gamma=0 | 0.282503 | 0.640064 | 0.077433 |

## Báo cáo ngưỡng confidence

Ô là `recall/precision`. Đây là cumulative operating points tại primary
matching IoU 0.50, không phải full PR curve.

| Threshold | A2 | BTRM scratch | A2 adapt | BTRM adapt | BTRM gamma=0 |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.5219/0.0506 | 0.5112/0.0486 | 0.5266/0.0531 | 0.5267/0.0532 | 0.4976/0.0498 |
| 0.005 | 0.5152/0.0717 | 0.5041/0.0636 | 0.5182/0.0746 | 0.5190/0.0748 | 0.4865/0.0716 |
| 0.010 | 0.5076/0.0882 | 0.4954/0.0787 | 0.5103/0.0915 | 0.5116/0.0916 | 0.4757/0.0913 |
| 0.020 | 0.4927/0.1125 | 0.4802/0.1018 | 0.4989/0.1150 | 0.4992/0.1147 | 0.4563/0.1191 |
| 0.050 | 0.4554/0.1626 | 0.4407/0.1532 | 0.4607/0.1603 | 0.4620/0.1602 | 0.4024/0.1841 |
| 0.100 | 0.3943/0.2321 | 0.3777/0.2277 | 0.4031/0.2237 | 0.4041/0.2234 | 0.3205/0.2889 |
| 0.200 | 0.2713/0.4032 | 0.2467/0.3993 | 0.2832/0.3836 | 0.2841/0.3822 | 0.1748/0.5063 |
| 0.500 | 0.0252/0.7913 | 0.0180/0.8098 | 0.0288/0.8191 | 0.0283/0.8134 | 0.0073/0.8333 |

BTRM scratch thua baseline ở mọi ngưỡng về recall; precision chỉ nhỉnh hơn ở
`0.5`. BTRM adapt gần như trùng control ở mọi ngưỡng, không tạo lợi thế vận hành.
Gamma=0 tăng precision ở ngưỡng cao nhưng mất recall rất mạnh.

## Cấu trúc lỗi localization

Median trên các localization FP có `score >= 0.05` và `0.10 <= IoU < 0.50`.
`width ratio = pred_width / GT_width`; `< 1` nghĩa là box dự đoán hẹp hơn GT.
`center offset` là khoảng cách tâm chia cho cạnh GT lớn hơn.

| Run | Bin | n | Width ratio median | Center offset median |
|---|---|---:|---:|---:|
| A2 | tiny1 | 466 | 0.747 | 0.247 |
| A2 | tiny2 | 2.273 | 0.883 | 0.242 |
| A2 | tiny3 | 5.223 | 0.908 | 0.249 |
| BTRM scratch | tiny1 | 437 | 0.842 | 0.244 |
| BTRM scratch | tiny2 | 2.151 | 0.985 | 0.243 |
| BTRM scratch | tiny3 | 4.849 | 0.972 | 0.260 |
| A2 adapt | tiny1 | 488 | 0.701 | 0.249 |
| A2 adapt | tiny2 | 2.156 | 0.874 | 0.242 |
| A2 adapt | tiny3 | 5.168 | 0.945 | 0.250 |
| BTRM adapt | tiny1 | 482 | 0.703 | 0.250 |
| BTRM adapt | tiny2 | 2.160 | 0.866 | 0.241 |
| BTRM adapt | tiny3 | 5.164 | 0.956 | 0.249 |

Theo định nghĩa rerun này, box lỗi trung vị **hẹp hơn** GT, không phải rộng hơn
như mô tả trước đây. BTRM scratch đẩy width ratio tiny1/tiny2 về gần 1 nhưng
không cải thiện center offset; warm-start gần như không đổi. Lỗi còn lại vẫn là
regression/assignment và scale, không phải thiếu contrast cục bộ.

## Recall theo size bin

| Bin | A2 | BTRM scratch | A2 adapt | BTRM adapt | BTRM gamma=0 |
|---|---:|---:|---:|---:|---:|
| tiny1 | 0.2105 | 0.1980 | 0.2105 | 0.2126 | 0.1689 |
| tiny2 | 0.4737 | 0.4478 | 0.4590 | 0.4582 | 0.4150 |
| tiny3 | 0.5636 | 0.5500 | 0.5641 | 0.5644 | 0.5317 |
| small | 0.6414 | 0.6374 | 0.6584 | 0.6559 | 0.6314 |
| reasonable | 0.6966 | 0.7035 | 0.7233 | 0.7246 | 0.7281 |

## BTRM cũ

Protocol `tinyperson` cũ, không dùng official_v1, nên chỉ mang tính lịch sử:

| Run | AP | AP50 | AP75 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| `btrm_a0`, no module | 0.07090 | 0.24525 | 0.01814 | 0.35423 | 0.32837 |
| `btrm_a2`, contrast | 0.06990 | 0.24430 | 0.01643 | 0.34783 | 0.32683 |
| `btrm_a3`, BTRM | 0.06825 | 0.23440 | 0.01662 | 0.34931 | 0.32887 |

## Chi phí

Trên L4, batch 1, ảnh 1280:

| Metric | A2 | BTRM scratch | BTRM adapt |
|---|---:|---:|---:|
| Parameters | 2.515.976 | 2.518.585 (+2.609) | 2.518.585 (+2.609) |
| Latency | 16,37 ms | 17,62 ms (+7,6%) | 18,74 ms (+14,0%) |
| FPS | 61,08 | 56,76 (-7,1%) | 53,35 (-12,3%) |
| Benchmark memory | 214,8 MB | 344,3 MB (+60,3%) | 315,1 MB (+46,7%) |
| Training time | 1.781 s | 2.515 s (+41,2%) | n/a |

## Kết luận

1. BTRM không cải thiện AP75 tổng thể: scratch `0.023745 → 0.020200`, warm-start
   `0.024719 → 0.019529`.
2. BTRM dịch lỗi từ localization sang background FP nhiều hơn lượng localization
   FP loại bỏ được, và làm mất TP.
3. Gamma=0 không cứu được hướng này; learned residual chủ yếu đổi trade-off
   recall/precision, không sửa regression.
4. Cấu trúc box lỗi còn lại là regression/assignment, không phải thiếu edge
   detail. Tăng độ phức tạp trong cùng họ BTRM không có cơ sở từ dữ liệu.
5. Kết quả chỉ có một seed; chưa có mean/std. Không nên dùng BTRM làm module mới
   cho bài báo.
6. Chưa có run BTRM trên VisDrone; config đã có nhưng chưa có artifact.

## Artifact

- `outputs/metrics/tiny_A2_p2_btrm-tinyperson-test.json`
- `outputs/metrics/tiny_A2_p2_btrm_adapt-tinyperson-test.json`
- `outputs/metrics/tiny_A2_p2_btrm_gamma0-tinyperson-test.json`
- `outputs/error_analysis/tiny_A2_p2_btrm-test/`
- `outputs/error_analysis/tiny_A2_p2_btrm_adapt-test/`
- `outputs/error_analysis/tiny_A2_p2_btrm_gamma0-test/`
- `outputs/error_analysis/tiny_A2_p2_adapt-test/`

Lưu ý: ba run adapt/gamma0 có cross-check lệch cực đại `2.7e-7`, chỉ là sai số
số học; scratch BTRM khớp tuyệt đối.
