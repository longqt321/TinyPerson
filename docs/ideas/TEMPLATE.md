# <ID> — <Tên ý tưởng>

## Thông tin
- mã: <IDEA-XX>
- trạng thái: đề xuất
- ưu tiên: P0 | P1 | P2
- nút thắt: bỏ sót | sai lệch vị trí gần đúng | xếp hạng | tiny1
- chi phí ước tính: <số lần chạy, GPU-giờ, credit>
- phụ trách: chưa gán
- ngày: 2026-10-24

## Giả thuyết
Một câu, có chỉ số và đối tượng. Ví dụ: “Gán nhãn có nhận biết tâm làm tăng
AP75 của tiny1 mà không giảm AP50.”

## Bằng chứng
- Số liệu hiện tại liên quan trực tiếp tới ý tưởng.
- Chỉ ra nhóm GT, sai số tâm, confidence hoặc kết quả âm đã biết.
- Ghi rõ nguồn: mã lần chạy, file CSV/JSON, tài liệu.

## Cơ chế
Chuỗi nhân quả:
`lỗi đo được → thay đổi cụ thể → gradient/gán nhãn thay đổi → chỉ số`.
Không dùng câu chung chung như “học đặc trưng tốt hơn”.

## Tính mới và công trình liên quan
- Đây là module, loss, gán nhãn, augmentation hay hậu xử lý?
- Khác gì MSR, BTRM, SADR, SCG, TAL, DFL, NWD hoặc các phương pháp đã biết?
- Nếu là thành phần có sẵn, nói rõ đây là ablation, không phải module mới.

## Thay đổi tối thiểu
- File/hàm/class sẽ sửa.
- Cấu hình mới.
- Biến duy nhất thay đổi so với control.
- Không đổi đồng thời loss, gán nhãn, augmentation và optimizer.

## Giao thức thực nghiệm
- Control và treatment.
- Seed 42; chỉ mở rộng 42/43/44 sau khi qua ngưỡng.
- 50 epoch, imgsz 1280, batch 8, AdamW, cùng augmentation.
- Cùng GPU SKU; chọn `best.pt` trên validation.
- Chỉ chạy official test sau khi freeze.

## Chỉ số và ngưỡng
- Chính: AP75; AP75 tiny1/tiny2/tiny3.
- Phụ: AP, AP50, precision, recall, số GT `IoU>=0.75`, số GT `0.1–0.5`,
  số GT `IoU<0.1`, localization FP.
- Ngưỡng đề xuất: AP75 tăng ít nhất `<mức>` tuyệt đối, AP50/AP trung bình không
  giảm quá `<mức>`, tiny1 không sụt; tái lập 3 seed trước khi tuyên bố.

## Loại bỏ thành phần và bác bỏ
- Bỏ thành phần chính thì có quay về control?
- Đổi siêu tham số chính thì kết quả còn không?
- Điều kiện nào bác bỏ giả thuyết?
- Tiêu chí dừng: dừng sau bao nhiêu lần chạy/seed nếu không đạt.

## Rủi ro
- Nhiễu seed, khác GPU, sai lệch annotation, metric artifact, rò rỉ dữ liệu.
- Chi phí vượt credit.
- Kết quả dương giả do tăng recall nhưng giảm precision.

## Tuyên bố trong bài báo
- Nếu dương: được phép tuyên bố điều gì?
- Nếu âm: kết quả âm có giá trị gì?
- Không tuyên bố tổng quát từ một dataset hoặc một seed.

## Sản phẩm lưu trữ
- Mã lần chạy, cấu hình, model YAML, dự án W&B.
- `val.json`, `results.csv`, CSV phân tích lỗi.

## Quyết định
- GO nếu qua ngưỡng.
- NO-GO nếu không.
- Việc tiếp theo cụ thể.

## Điểm
tác động `<1–5>`, bằng chứng `<1–5>`, tính mới `<1–5>`, chi phí `<1–5>`,
rủi ro `<1–5>`, khả thi `<1–5>`; tổng `<tổng>/30`.
