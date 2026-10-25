# Tổng hợp kết quả Modal — 19–20/10/2026

Phạm vi: các run trong Volume `tiny-person-runs` khởi chạy từ 20:43 ngày 19/10 đến 12:24 ngày 20/10, giờ Việt Nam. Có 31 run thuộc ba nhóm: MSR fusion cũ, ablation kiến trúc A0–A3 và MSR trên P2.

Tất cả run dùng dataset `/mnt/data/tinyperson_official_v1/data.yaml`, 746 ảnh (662 train, 84 validation), seed 42, GPU NVIDIA L4, Python 3.12.10, PyTorch 2.11.0+cu128, Ultralytics 8.4.164. Nhóm A0–A3 và MSR-P2 dùng commit `8326468`; nhóm MSR cũ dùng `6a15122`. Cả hai commit có `git_dirty=true`; commit ID một mình không tái lập chính xác code nếu thiếu config/YAML của run.

## MSR-P2: kết quả chính

Metric lấy từ `metrics/val.json` của `best.pt`; AP là mAP@[.50:.95]. CSV của các run full có đủ 50 epoch. Đây là validation, không phải test. AP-small chưa được tính.

| Experiment | Cấu hình | Đường dẫn run trong Volume | AP | AP50 | Precision | Recall | Peak VRAM train |
|---|---|---|---:|---:|---:|---:|---:|
| E0 | A2 P2 + SPPF | `msr_p2_e0_a2_baseline/20261020-034105-8326468-seed42` | 0.05912 | 0.19689 | 0.35815 | 0.25249 | 10.15 GB |
| E1 | A2 + MSR trên P2 | `msr_p2_e1_a2_msr/20261020-034106-8326468-seed42` | 0.05975 | 0.19155 | 0.33292 | 0.27217 | 13.91 GB |
| E2 | A3 P2, bỏ SPPF | `msr_p2_e2_a3_no_sppf/20261020-034109-8326468-seed42` | 0.05591 | 0.17992 | 0.33326 | 0.26053 | 10.07 GB |
| E3 | A3 + MSR trên P2 | `msr_p2_e3_a3_no_sppf_msr/20261020-034109-8326468-seed42` | 0.05294 | 0.17576 | 0.29808 | 0.25277 | 13.83 GB |

So sánh theo cặp:

- **E1 − E0 (MSR, giữ SPPF):** AP `+0.00064` (+0.064 điểm phần trăm), recall `+0.01968` (+1.97 điểm), AP50 `−0.00533` (−0.53 điểm), precision `−0.02523` (−2.52 điểm). Peak VRAM tăng khoảng 3.76 GB.
- **E3 − E2 (MSR, bỏ SPPF):** AP `−0.00298` (−0.30 điểm), recall `−0.00776` (−0.78 điểm), AP50 `−0.00416` (−0.42 điểm), precision `−0.03518` (−3.52 điểm). Peak VRAM tăng khoảng 3.76 GB.

Rerun E1/E2/E3 lúc 12:24 giờ Việt Nam cho AP lần lượt `0.05975`, `0.05589`, `0.05293`, gần như trùng lượt chính. Không có rerun E0. **Kết quả chưa cho thấy MSR cải thiện nhất quán:** E1 nhỉnh AP rất ít và tăng recall, còn E3 giảm AP/recall; cả hai đều tốn thêm bộ nhớ. Một seed chưa đủ kết luận thống kê.

Metadata của run full ghi `TRAINED`, không phải `COMPLETED`; CSV có đủ 50 epoch và validation JSON có mặt, nhưng finalize/benchmark chưa hoàn tất. `modal app list --json` lúc 13:12 ngày 20/10 cho thấy hai app gần nhất đều `stopped`, 0 task; không thấy task đang chạy. Benchmark inference của các run full không có.

## Ablation kiến trúc Tiny A0–A3

| Model | Cấu hình | Đường dẫn run trong Volume | Epoch | AP | AP50 | Precision | Recall | Latency* |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A0 | Baseline | `tiny_A0_baseline/20261019-194809-8326468-seed42` | 50 | 0.04810 | 0.17132 | 0.32493 | 0.23531 | 13.85 ms |
| A1 | Bỏ SPPF | `tiny_A1_no_sppf/20261019-194809-8326468-seed42` | 50 | 0.05029 | 0.17459 | 0.32234 | 0.24446 | 14.09 ms |
| A2 | P2 + SPPF | `tiny_A2_p2/20261019-194809-8326468-seed42` | 50 | 0.05912 | 0.19689 | 0.35815 | 0.25249 | 16.37 ms |
| A3 | P2, bỏ SPPF | `tiny_A3_p2_no_sppf/20261019-194811-8326468-seed42` | 50 | 0.05591 | 0.17992 | 0.33326 | 0.26053 | 16.52 ms |

A2 có AP cao nhất. So với A0, A2 tăng AP `0.01102` (+1.10 điểm phần trăm), recall `0.01718` (+1.72 điểm), latency tăng khoảng 2.52 ms/ảnh. A3 tăng recall `0.02522` so với A0 nhưng precision thấp hơn. Đây là một seed, trên validation; không có AP-small.

`*` Đo trên L4, batch 1, imgsz 1280, warm-up 10, 100 lượt; chỉ tính model latency. `gflops` trong artifact là `null`, nên không báo cáo FLOPs.

## Baseline có thể dùng lại

| Ưu tiên | Checkpoint đã hoàn tất | Dùng cho | Tình trạng |
|---|---|---|---|
| Chính | `tiny_A2_p2/20261019-194809-8326468-seed42/checkpoints/best.pt` | Baseline P2 + SPPF cho E1/MSR-P2 ablation | `COMPLETED`, 50 epoch, có `best.pt`, `last.pt`, validation và benchmark. YAML SHA khớp YAML của E0 hiện tại (`792eed…`). |
| Phụ | `tiny_A3_p2_no_sppf/20261019-194811-8326468-seed42/checkpoints/best.pt` | Baseline P2 bỏ SPPF cho E3/MSR-P2 ablation | `COMPLETED`, 50 epoch, có đủ artifacts. YAML SHA khớp YAML của E2 hiện tại (`aaaaa6…`). |
| Tùy nhu cầu | A0 `tiny_A0_baseline/20261019-194809-8326468-seed42`; A1 `tiny_A1_no_sppf/20261019-194809-8326468-seed42` | So sánh lại tác động P2/SPPF | Cả hai `COMPLETED`, 50 epoch, có checkpoint/metric/benchmark; không cần train lại nếu câu hỏi còn dùng các baseline này. |

A2/A3 có cùng seed 42, dataset signature, epoch, augmentation, optimizer và hyperparameter với E0/E2. Vì vậy có thể giữ chúng làm hai đối chứng seed 42, không cần train baseline lại khi so với biến thể mới trên cùng giao thức. Muốn báo cáo nhiều seed, cần train các cặp seed mới; checkpoint seed 42 chỉ đóng góp seed 42.

Lưu ý về tự động reuse: `--reuse` yêu cầu status `COMPLETED`, file `checkpoints/best.pt`, và config YAML đã lưu phải khớp hoàn toàn, gồm cả tên experiment. Checkpoint A2/A3 đạt điều kiện này với tên `tiny_A2_p2`/`tiny_A3_p2_no_sppf`; cấu hình `ablation_msr_p2.yaml` hiện dùng tên `msr_p2_e0_a2_baseline`/`msr_p2_e2_a3_no_sppf`, nên `--reuse` không tự nhận ra checkpoint khác tên. Dùng lại tự động thì đặt tên baseline như run đã hoàn tất và giữ nguyên các trường config, hoặc đánh giá trực tiếp checkpoint đã lưu.

Run E0/E2 trong nhóm MSR-P2 mới cũng có đủ 50 epoch, validation và weights tại `training/weights/best.pt`, nhưng metadata là `TRAINED` và chưa có `checkpoints/best.pt`; vì vậy chúng chưa qua điều kiện `--reuse`. Không cần huấn luyện lại để dùng thủ công; có thể finalize artifact trước nếu muốn reuse theo đúng tên experiment mới.

## Các run còn lại

| Nhóm | Run IDs | Artifact/trạng thái | Kết luận sử dụng |
|---|---|---|---|
| MSR fusion cũ, smoke | A0 `20261019-135927-6a15122-seed42`; A1 `20261019-135937-6a15122-seed42`; A2 `20261019-135940-6a15122-seed42`; A3 `20261019-135940-6a15122-seed42` | Cả 4 hoàn tất 5 epoch. AP: 0.00324, 0.00426, 0.00519, 0.00891. | Smoke test, không so sánh chất lượng cuối; nhánh cũ, không phải MSR-P2. |
| MSR fusion cũ, lượt dài | A0 `20261019-142706-6a15122-seed42`; A1 `20261019-142801-6a15122-seed42`; A2 `20261019-142828-6a15122-seed42`; A3 `20261019-142858-6a15122-seed42` | A0 có 50 epoch, AP 0.06411. A1/A2 có 39 epoch (best AP 0.06070/0.05528); A3 có 44 epoch (best AP 0.05612). Metadata ba run sau vẫn `RUNNING`, không có validation JSON cuối. | Artifact dừng trước 50 epoch; không xem là kết quả cuối. Khác commit/nhánh với MSR-P2. |
| MSR fusion cũ, retry | A0 `20261019-180246-6a15122-seed42`; A1/A2 `20261019-180248-6a15122-seed42`; A3 `20261019-180251-6a15122-seed42` | Metadata `RUNNING`; A0 mới 2 epoch, ba run còn lại chưa có CSV huấn luyện. | Không có artifact chứng minh còn task chạy; chưa có metric cuối. |
| A0–A3, lần khởi chạy đầu | Cả bốn có run ID `20261019-194348-8326468-seed42` | Metadata `RUNNING`, chỉ có args/metadata, không CSV/checkpoint huấn luyện. | Không có artifact huấn luyện; dùng lần hoàn tất `194809/194811`. |
| MSR-P2 smoke | E0 `20261020-031442-8326468-seed42`; E1 `20261020-031449-8326468-seed42`; E2 `20261020-031706-8326468-seed42`; E3 `20261020-031707-8326468-seed42` | Cả 4 hoàn tất 5 epoch. AP: 0.00555, 0.00618, 0.00646, 0.00386. | Chỉ xác minh pipeline; không xếp hạng model. |
| MSR-P2, lượt chính | E0 `20261020-034105-8326468-seed42`; E1 `20261020-034106-8326468-seed42`; E2/E3 `20261020-034109-8326468-seed42` | Đủ 50 epoch, có validation JSON; metadata `TRAINED`. E2/E3 dùng cùng ID nhưng nằm trong thư mục experiment riêng. | Metric ở bảng đầu. |
| MSR-P2, retry | E1 `20261020-052417-8326468-seed42`; E2/E3 `20261020-052429-8326468-seed42` | Đủ 50 epoch, có validation JSON; metadata `TRAINED`. | Gần trùng lượt chính; không có E0 retry. |

## Artifact và giới hạn

Artifact gốc nằm trên Modal Volume `tiny-person-runs`, dưới `<experiment>/<run-id>/`. Tải run bằng:

```fish
python tools/download_results.py EXPERIMENT/RUN_ID --include metadata.json config.yaml metrics checkpoints
```

Metric hiện có là mAP validation chuẩn của pipeline, không phải COCO AP-small. Chưa có AP-small, test-set result, nhiều-seed mean/std hoặc benchmark inference cho MSR-P2 full runs. Không kết luận MSR cải thiện tiny-object detection từ metric tổng thể hiện tại.
