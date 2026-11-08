# Phân tích kết quả GauSS

## Cơ sở phân tích

Bằng chứng chia thành hai giao thức, không gộp chung:

1. **Validation YOLO derivative:** GauSS cycle 2 có `AP=0.08010`, `AP50=0.26172`, `AP75=0.02203`, precision `0.40447`, recall `0.31513` tại `reports/gauss_autonomous/cycle_2/val.json:1-17`; đây là validation, không phải official test. Baseline tương ứng trong báo cáo tổng kết là 100 epoch, nhưng cần đối chiếu artifact thô trước khi so sánh thống kê.
2. **TinyPerson standalone tiled test:** raw benchmark JSON tại `reports/eval/*_tinyperson_150e_tiled.json`, 786 ảnh, 13,687 positive GT, seed42. Đây là một lần chạy cho mỗi phương pháp, checkpoint khai báo `last.pt`, evaluator standalone; không tuyên bố đây là official reference evaluator.

## Kết quả primary: tiled test

| Claim | Bằng chứng | Độ tin cậy | Ghi chú |
|---|---|---|---|
| GauSS có AP@0.50 cao hơn baseline tổng thể 0.98 điểm phần trăm (12.94% vs 11.95%). | Hai JSON 150e tiled, `summary.benchmark.all.AP@0.50` | Thấp | Một seed; chưa có CI/test. |
| AP@0.50 tăng rõ nhất ở size-bin tiny2: +2.59 điểm (9.50% vs 6.90%). | Hai JSON, `tiny2.AP@0.50` | Thấp | Kết quả theo một split/run. |
| GauSS cải thiện AP@0.50 ở tiny, tiny1, tiny2, small và reasonable; giảm nhẹ ở tiny3. | Cùng hai JSON, các size-bin | Thấp | Không đồng nhất giữa bins. |
| GauSS giảm AP@0.25 tổng thể 0.61 điểm (20.68% vs 21.28%). | `all.AP@0.25` | Thấp | Đánh đổi cần báo cáo. |
| Tiled AP@0.75 tổng thể tăng khoảng 0.11 điểm; nhưng `tiny2.AP@0.75` giảm khoảng 0.22 điểm. | `all.AP@0.75`, `tiny2.AP@0.75` | Thấp | Metric nghiêm ngặt có kết quả hỗn hợp. |
| GauSS giảm MR@0.50 tổng thể khoảng 0.21 điểm; tiny1 MR@0.50 xấu hơn khoảng 0.18 điểm. | `all.MR@0.50`, `tiny1.MR@0.50` | Thấp | Không kết luận cải thiện đồng đều. |

## Full-frame test

Raw JSON tại `reports/eval/baseline_tinyperson_150e_full.json` và `reports/eval/gauss_v2_1_tinyperson_150e_full.json` cho thấy all AP@0.50 lần lượt 12.43% và 12.53% (+0.10 điểm), nhưng all AP@0.75 giảm từ 1.96% xuống 1.31%. Tiny AP@0.50 giảm nhẹ từ 9.97% xuống 9.76%. Do đó kết quả full-frame chỉ hỗ trợ nhận định lợi ích nhỏ ở AP@0.50 tổng thể, không hỗ trợ cải thiện toàn diện.

## Validation và test không thay thế nhau

Validation cycle 2 (`reports/eval/gauss_v2_1_val_suite.json`) báo AP@0.50=26.69% trên 84 ảnh derivative validation, còn test standalone báo AP@0.50=12.94% trên 786 ảnh. Split, evaluator và protocol khác nhau; không so trực tiếp các số này và không gọi validation là test.

## Giới hạn và khoảng trống

- Chỉ seed42 được xác định ở raw test artifacts; không có ước lượng biến thiên qua seed hay kiểm định thống kê.
- Eval artifacts trỏ tới `training/weights/last.pt`; không giả định đó là checkpoint tốt nhất.
- JSON ghi 786 ảnh và 0 dense images excluded. Chưa chứng minh tương đương official TinyPerson evaluator/protocol.
- Báo cáo tổng kết GauSS có số tổng hợp validation và thông tin kiến trúc; số test phải lấy từ JSON thô nếu có lệch.
- Cần xác nhận cấu hình và metadata cho paired training conditions, hardware/profiling provenance và checkpoint SHA256 trước khi đưa claim đó vào bản cuối.
- Cần nhiều seed, ablation riêng từng thay đổi của GauSS v2.1, và đánh giá bằng reference evaluator để có kết luận mạnh hơn.
