# IDEA-05 — Xếp hạng độ tin cậy theo chất lượng box

> Primitive tham chiếu; không train riêng. Chỉ dùng như thành phần trong composite.

## Thông tin
- mã: IDEA-05
- trạng thái: đề xuất
- ưu tiên: P0
- nút thắt: xếp hạng
- chi phí ước tính: 1 lần chạy × 50 epoch trên A10; 2 lần nếu cần ablation
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Cho classification target phụ thuộc chất lượng box (IoU-aware) sẽ đưa các box
chất lượng cao lên trên các FP, giúp tăng AP75 mà không cần đổi regression.

## Bằng chứng
- 436 GT có best IoU `>=0.75`; trong đó 179 box có confidence `<0.25`.
- AP75 chỉ `0.016198` dù nhiều box đã đạt hình học tốt.
- TAL4 tăng precision/recall nhưng không tăng AP75, gợi ý xếp hạng là một nút
  thắt riêng.
- 39.121 localization FP ở ngưỡng IoU 0.50 cho thấy nhiều box gần đúng nhưng
  không được xếp hạng đúng.

## Cơ chế
Thay target classification nhị phân bằng target mềm `cls * IoU^gamma` hoặc thêm
pairwise ranking loss giữa box có IoU cao và box có IoU thấp. Box đúng sẽ có
score cao hơn, AP75 được hưởng lợi vì AP phụ thuộc thứ tự confidence.

## Tính mới và công trình liên quan
IoU-aware classification và ranking loss đã có trong literature. Đây là ablation
có kiểm soát trên đối tượng nhỏ, không phải module mới. Phải so sánh với
confidence calibration và NMS.

## Thay đổi tối thiểu
- Sửa classification loss trong `v8DetectionLoss`; không đổi assigner/head.
- Cấu hình: `quality_gamma`, `rank_weight`.
- Không đổi box loss, augmentation hay optimizer.

## Giao thức thực nghiệm
- Control: A2 baseline.
- Treatment: quality-aware target hoặc ranking loss.
- 50 epoch, imgsz 1280, batch 8, cùng data và GPU A10.

## Chỉ số và ngưỡng
- Chính: AP75.
- Ngưỡng: AP75 tăng `+0.0015`; AP50 không giảm quá `0.002`; precision không giảm.
- Theo dõi confidence của box `IoU>=0.75` trước/sau.

## Loại bỏ thành phần và bác bỏ
- Gamma `0.5`, `1.0`.
- Soft target so với pairwise ranking.
- Bác bỏ: nếu score của box tốt không tăng, cơ chế sai.
- Tiêu chí dừng: dừng nếu AP50 giảm mạnh hoặc AP75 không đổi.

## Rủi ro
- Có thể đẩy score của box chất lượng thấp lên.
- Ranking loss có thể xung đột với BCE.
- AP50 có thể giảm nếu target quá mềm.

## Tuyên bố trong bài báo
- Nếu dương: “xếp hạng theo chất lượng chuyển các box đã localization đúng thành
  AP75”.
- Nếu âm: AP75 không bị giới hạn bởi xếp hạng.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, W&B, CSV calibration confidence-theo-IoU, `val.json`.

## Quyết định
- GO nếu AP75 tăng và calibration không xấu.
- NO-GO nếu chỉ AP50 tăng.

## Điểm
tác động 4, bằng chứng 3, tính mới 4, chi phí 4, rủi ro 3, khả thi 4; tổng 22/30.
