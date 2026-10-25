# IDEA-12 — Trần Oracle AP75 và phân rã lỗi

> Phân tích offline; không train. Được phép chạy như bước chẩn đoán trước composite.

## Thông tin
- mã: IDEA-12
- trạng thái: đề xuất
- ưu tiên: P0
- nút thắt: phân tích
- chi phí ước tính: 0 train; 1 script trên prediction có sẵn
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Phân tích oracle sẽ đo headroom AP75 khi cho phép localization hoàn hảo hoặc
xếp hạng hoàn hảo, từ đó xác định ý tưởng nào có thể tạo ra cải thiện thật.

## Bằng chứng
- 436 GT `IoU>=0.75`; 1.157 GT `0.50–0.75`; 866 GT `0.10–0.50`;
  1.149 GT `<0.10`.
- 179/436 box tốt có confidence `<0.25`.
- AP75 hiện tại `0.016198`; cần biết bao nhiêu do box, bao nhiêu do xếp hạng.

## Cơ chế
Từ prediction của checkpoint, tạo các biến thể:
1. thay box bằng GT cho detection đã match;
2. thay confidence bằng oracle IoU ranking;
3. làm cả hai.
Chạy lại evaluator để tính AP75 và AP75 theo size bin.

## Tính mới và công trình liên quan
Đây là phân tích, không phải model. Không được dùng để tuyên bố cải thiện; nó là
upper bound chẩn đoán.

## Thay đổi tối thiểu
- Thêm script `scripts/oracle_ap75.py` đọc prediction JSON/CSV.
- Không train, không sửa checkpoint.
- Không dùng official test để tune; chỉ validation.

## Giao thức thực nghiệm
- Input: prediction JSON của A2/E2/E3.
- Output: oracle AP75, oracle AP75 tiny1/tiny2/tiny3, số GT bắt được.
- So sánh ba biến thể.

## Chỉ số và ngưỡng
- Ngưỡng: nếu oracle AP75 thấp (ví dụ <0.03), dùng các ý tưởng chỉ về regression;
  nếu oracle AP75 cao, ưu tiên ranking/assignment.
- Báo cáo headroom tuyệt đối và tương đối.

## Loại bỏ thành phần và bác bỏ
- Box-only, score-only, box+score.
- Per-size bin.
- Bác bỏ: nếu oracle không cao hơn baseline đáng kể, không có headroom.
- Tiêu chí dừng: nếu oracle AP75 không cao hơn baseline đáng kể, không dùng kết
  quả để biện minh cho module localization mới.

## Rủi ro
- Oracle lạc quan, không achievable.
- Dễ bị dùng sai để cherry-pick.
- Phải giữ nguyên evaluator protocol.

## Tuyên bố trong bài báo
- Nếu dương: phân rã giới hạn trên của AP75 và định hướng phương pháp.
- Nếu âm: AP75 bị giới hạn bởi detection, không phải localization.

## Sản phẩm lưu trữ
- Script, prediction input, `oracle_summary.json`, `oracle_by_size.csv`.

## Quyết định
- GO: dùng kết quả để chọn top 3 ý tưởng.
- NO-GO: không có.

## Điểm
tác động 5, bằng chứng 5, tính mới 2, chi phí 5, rủi ro 1, khả thi 5; tổng 23/30.
