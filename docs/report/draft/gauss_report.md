# GauSS v2.1: Contextual Feature Refinement for Tiny-Person Detection

## Tóm tắt

Phát hiện người có kích thước rất nhỏ vẫn khó do tín hiệu đối tượng yếu và nền phức tạp. Báo cáo này khảo sát GauSS v2.1, module tinh chỉnh đặc trưng ngữ cảnh bất đối xứng, gắn vào YOLO26n-P2. GauSS v2.1 được huấn luyện 150 epoch với seed 42 trên cấu hình TinyPerson trong repo. Trên đánh giá tiled standalone gồm 786 ảnh, AP@0.50 tổng thể đạt 12.94%, cao hơn baseline YOLO26n-P2 11.95% khoảng 0.98 điểm phần trăm. Mức tăng lớn nhất trong các size-bin được ghi nhận ở nhóm tiny2 (8–12 px), từ 6.90% lên 9.50%. Tuy vậy AP@0.25 tổng thể và một số kết quả theo kích thước giảm; đánh giá full-frame cho thấy chênh lệch AP@0.50 nhỏ hơn nhiều. Kết quả hiện tại chỉ gồm một seed và evaluator standalone, nên cần được xem là bằng chứng sơ bộ, chưa đủ để kết luận cải thiện có ý nghĩa thống kê hoặc tuân thủ hoàn toàn benchmark chính thức.

## 1. Giới thiệu

Phát hiện người cực nhỏ có ứng dụng trong giám sát trên không và phân tích cảnh đông người, nhưng đối tượng chiếm ít pixel nên đặc trưng không gian dễ lẫn với nền. Mạng phát hiện một giai đoạn gọn nhẹ có lợi thế về chi phí suy luận, song việc giữ lại chi tiết độ phân giải cao thường làm tăng đáp ứng nền và false positive.

Báo cáo này tập trung vào GauSS v2.1 trong repo TinyDet. GauSS là viết tắt của *Gaussian Scale-Space Feature Alignment and Refinement*. Thiết kế đặt trên pipeline YOLO26n với nhánh P2, khai thác đặc trưng đa băng tần, ngữ cảnh giãn nở và hiệu chỉnh bất đối xứng. Phiên bản v2.1 bổ sung neo giữ tín hiệu cao tần và cơ chế giảm đáp ứng nền theo mô tả thiết kế trong `reports/gauss_autonomous/cycle_2/architecture.md`.

Bằng chứng hiện có cho thấy GauSS v2.1 đạt AP@0.50 cao hơn baseline trong đánh giá tiled ở một số kích thước, nổi bật nhất là tiny2. Cải thiện không đồng đều: AP@0.25 tổng thể giảm, nhóm tiny3 giảm nhẹ tại AP@0.50, và full-frame AP@0.50 chỉ tăng khoảng 0.10 điểm phần trăm. Báo cáo trình bày các kết quả tích cực lẫn các đánh đổi, đồng thời giới hạn kết luận theo provenance hiện có.

Đóng góp của báo cáo:

1. Mô tả GauSS v2.1 và cấu hình huấn luyện được lưu trong repository.
2. Tổng hợp kết quả tiled và full-frame riêng biệt, không trộn validation derivative với test.
3. Nêu rõ giới hạn về một seed, checkpoint `last.pt`, và evaluator standalone.

## 2. Liên quan

Nghiên cứu phát hiện vật thể nhỏ thường sử dụng đặc trưng độ phân giải cao, thông tin đa tỷ lệ và ngữ cảnh rộng để bù cho tín hiệu cục bộ yếu. GauSS nằm trong hướng kết hợp đặc trưng chi tiết với ngữ cảnh đa tỷ lệ. Các nguồn học thuật cụ thể chưa được xác minh trong phiên này; không đưa trích dẫn chưa kiểm chứng vào bản thảo.

Các thiết kế đa băng tần phân tách cấu trúc thô và chi tiết cao tần; convolution giãn nở mở rộng receptive field mà không tăng mạnh độ phân giải trung gian. GauSS kết hợp các ý tưởng này với nhánh hiệu chỉnh bất đối xứng. Cần hoàn thành tìm kiếm và xác minh tài liệu theo sáu lớp trước khi gửi báo cáo hoặc bài báo như một bản học thuật hoàn chỉnh.

## 3. Phương pháp

### 3.1. Kiến trúc

GauSS v2.1 được tích hợp vào YOLO26n-P2. Theo tài liệu thiết kế cycle 2, module gồm ba ý tưởng chính: phân rã đáp ứng thành các dải tần, định tuyến ngữ cảnh qua các nhánh convolution giãn nở, và hiệu chỉnh đặc trưng chi tiết so với cấu trúc. Cycle 2 bổ sung đường neo tín hiệu cao tần và gating theo độ nổi bật nền. Các chi tiết toán học và sơ đồ cần được đối chiếu với YAML/model implementation hiện hành trước khi dùng như đặc tả cuối.

### 3.2. Thiết lập huấn luyện

Cấu hình `configs/experiment/gauss_v2_1_tinyperson_wacv.yaml` đặt seed 42, 150 epoch, ảnh 1280, batch 8, AdamW, learning rate ban đầu 0.002, cosine schedule, warmup 5 epoch, AMP bật và GPU A10. Baseline dùng cấu hình tương ứng `configs/experiment/baseline_tinyperson_wacv.yaml`; các trường huấn luyện được khai báo giống nhau, ngoại trừ model config. `deterministic` đặt false, vì vậy seed cố định không đồng nghĩa với tái lập bitwise.

### 3.3. Đánh giá

Kết quả tiled dùng crop 640×512 và NMS merge 0.50. JSON khai báo 786 ảnh, 13,687 positive ground truths, tối đa 200 detections mỗi ảnh và IoU thresholds 0.25, 0.50, 0.75. Bộ đánh giá được ghi là standalone implementation. Full-frame dùng ảnh 1280. Hai chế độ được báo cáo riêng vì chúng tạo điều kiện suy luận khác nhau. Các artifact hiện trỏ checkpoint `training/weights/last.pt`; báo cáo không đổi nhãn thành best checkpoint.

## 4. Thực nghiệm

### 4.1. Kết quả tiled

| Nhóm | Baseline AP@0.50 | GauSS v2.1 AP@0.50 | Δ (điểm %) | Baseline AP@0.25 | GauSS v2.1 AP@0.25 |
|---|---:|---:|---:|---:|---:|
| All | 11.95% | **12.94%** | +0.98 | **21.28%** | 20.68% |
| Tiny (2–20 px) | 9.07% | **9.76%** | +0.69 | **15.96%** | 15.42% |
| Tiny1 (2–8 px) | 4.72% | **5.18%** | +0.46 | **9.68%** | 8.76% |
| Tiny2 (8–12 px) | 6.90% | **9.50%** | +2.59 | 13.20% | **14.40%** |
| Tiny3 (12–20 px) | **13.51%** | 13.37% | −0.14 | **22.05%** | 20.73% |
| Small (20–32 px) | 17.43% | **19.23%** | +1.80 | 28.60% | **28.92%** |
| Reasonable (>32 px) | 20.49% | **21.20%** | +0.72 | **36.04%** | 34.06% |

GauSS v2.1 improves AP@0.50 for five of six displayed size groups, with largest difference in tiny2. It does not improve AP@0.25 overall or in most bins. This indicates metric- and size-dependent behavior rather than uniform improvement. Each method has one seed; differences are descriptive only.

At AP@0.75 overall, GauSS reaches 1.40% versus 1.28% baseline. Tiny2 AP@0.75 decreases from 1.17% to 0.95%, while tiny1 increases from 0.41% to 1.11%. These small absolute values warrant cautious interpretation.

### 4.2. Full-frame results

| Metric | Baseline | GauSS v2.1 | Δ (điểm %) |
|---|---:|---:|---:|
| All AP@0.50 | 12.43% | **12.53%** | +0.10 |
| Tiny AP@0.50 | **9.97%** | 9.76% | −0.21 |
| Tiny1 AP@0.50 | 5.66% | **5.79%** | +0.13 |
| Small AP@0.50 | 15.80% | **16.35%** | +0.55 |
| All AP@0.75 | **1.96%** | 1.31% | −0.66 |

Full-frame results show a smaller overall AP@0.50 difference than tiled results and a decrease in overall AP@0.75. GauSS’s observed effect therefore depends on inference protocol.

### 4.3. Training and validation observations

The repository’s GauSS cycle 2 validation artifact reports AP 8.01%, AP50 26.17%, AP75 2.20%, precision 40.45%, and recall 31.51% (`reports/gauss_autonomous/cycle_2/val.json`). These are validation metrics from the YOLO derivative, not the standalone TinyPerson test metrics above. We report them separately and do not use them as test performance.

## 5. Thảo luận và hạn chế

GauSS v2.1’s clearest observed advantage is tiled AP@0.50 for tiny2 and the aggregate AP@0.50. The lower AP@0.25 and mixed size-bin behavior prevent a blanket claim that the method improves detection quality under every criterion. Full-frame gains are modest and AP@0.75 declines, so claims about deployment benefit require additional evaluation.

All training comparisons use seed 42 and `deterministic: false`. No across-seed variance, confidence interval, or statistical test is available. The benchmark JSON points to `last.pt`; the evaluation should be rerun or checkpoint provenance verified before final submission, especially because project reporting guidance prefers `best.pt`. The standalone evaluator reports 786 images with no dense-image exclusions; equivalence to the official reference protocol has not been established. Hardware profiling claims from existing reports are omitted because raw provenance has not yet been verified.

The current evidence also does not isolate every GauSS v2.1 component with controlled ablations. Thus, observed differences cannot be attributed causally to a specific mechanism. Additional seeds, component ablations, checkpoint verification, and reference-evaluator testing are needed for stronger conclusions.

## 6. Kết luận

This report summarizes GauSS v2.1 results available in the repository. In one seed, tiled AP@0.50 improves by 0.98 percentage points overall and 2.59 points for tiny2 compared with the paired baseline artifact. AP@0.25 decreases overall, size-bin results are mixed, and full-frame AP@0.75 is lower. These findings support further investigation, not a statistically established or protocol-certified performance claim. Multi-seed experiments and benchmark-protocol verification remain necessary.

## Tài liệu tham khảo

Chưa thêm tài liệu ngoài. Cần xác minh nguồn gốc và metadata trước khi đưa tài liệu học thuật vào danh mục.
