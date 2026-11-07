# Trạng thái báo cáo DCAF

## Phạm vi
- Handoff dossier DCAF v2.1 và baseline YOLO26n-P2 cho model bên ngoài viết báo cáo.
- Tiếng Việt; không viết thêm báo cáo nghiên cứu.
- Không mở hoặc trích xuất nội dung QAC hay `report-bundle/`.

## Tiến độ
- Giai đoạn 0: hoàn tất; phạm vi DCAF-only, tiếng Việt, technical report Markdown.
- Giai đoạn 1: inventory tại `01_inventory.md`; kết quả ưu tiên lấy từ JSON DCAF và baseline paired.
- Giai đoạn 2: phân tích DCAF-only tại `03_analysis.md`; một seed, kết luận mô tả.
- Giai đoạn 3: chưa tìm nguồn ngoài để tránh citation chưa kiểm chứng; related work trong draft là khung chủ đề, chưa phải tổng quan hoàn chỉnh.
- Giai đoạn 4: outline tại `04_outline.md`.
- Giai đoạn 5: bảng kết quả trong draft đối chiếu thủ công với JSON; chưa có script sinh tự động hoặc hình.
- Giai đoạn 6: draft tiếng Việt tại `draft/dcaf_report.md`.
- Giai đoạn 7: review ghi tại `99_review.md`; chưa biên dịch định dạng submission.

## Cảnh báo provenance
- Artifact eval 150e trỏ `training/weights/last.pt`; ghi đúng checkpoint này, không gọi là `best.pt`.
- Giao thức artifact khai báo 786 ảnh, dense excluded = 0; mô tả là standalone implementation, không khẳng định tuân thủ hoàn toàn official benchmark.
- Kết quả đơn seed; không kiểm định ý nghĩa thống kê.
- Draft là báo cáo nội bộ có giới hạn bằng chứng, chưa phải bản nộp học thuật do related work chưa xác minh và venue chưa biết.

## Ghi chú dọn dẹp 2026-11-06
- Tài liệu QAC/QAR riêng lẻ đã xóa; hồ sơ lịch sử A0–A3/MSR/BTRM/SADR/QAC còn lại duy nhất ở `reports/experiment_results_report.md`.
- Báo cáo trùng lặp đã xóa: `DCAF_FINAL_REPORT.md` và `TONG_HOP_KET_QUA_DCAF_VA_BASELINE.md` (byte-identical với bản GauSS).
- Tên file tài liệu đã chuyển sang tiếng Anh snake_case; link trong `docs/` và `reports/` đã cập nhật theo.
- Đã xóa thêm: `.superpowers/sdd/jaunty-painting-iverson/progress.md` (ledger của phiên trước), `.history/` (backup của VS Code extension, untracked), và `reports/geometry/tiny_A2_p2_qac_v2*/` (artifact hình học của QAC v2).


## Giai đoạn 8 — khắc phục phản biện (2026-11-06)

### Sửa lỗi evaluator (ưu tiên cao nhất)

- `evaluate_tinyperson` bỏ mọi dự đoán `class_id != 0` nhưng vẫn giữ ground truth
  của cả hai category. Đã thêm `class_mode` với `per_class` (mặc định, đúng
  COCO/TOV), `merge`, và `primary` (hành vi cũ, chỉ để tái lập).
- Cổng hồi quy đã chạy: `class_mode="primary"` cho GauSS 12.9360/5.1776/9.4962 và
  Baseline 11.9538/4.7164/6.9036 (all/tiny1/tiny2), khớp số đã công bố.
- Kết quả đúng (tiled, 150e, per-class):
  `sea_person` GauSS 29.63 vs Base 27.52 (+2.11 pp), tiny2 23.05 vs 17.68 (+5.37 pp);
  `earth_person` GauSS 31.65 vs Base 34.03 (−2.38 pp), tiny2 28.69 vs 31.73 (−3.04 pp);
  class-agnostic 29.97 vs 31.18 (−1.21 pp).
- Lưu tại `reports/eval/{dcaf,baseline}_{tiled,full}_per_class.json`.

### Ablation

- 5 run seed-42 dừng ở epoch 78/150 (`MANIFEST.json`: hết credit Modal), không
  phải 150. `metadata.json` vẫn ghi `status: RUNNING`.
- Cột `mAP50` bằng 0 ở mọi epoch vì `train.val: false` — không phải lỗi dữ liệu.
- Cờ ablation có hiệu lực: 5 checkpoint có sha256 trọng số EMA khác nhau và
  `gamma_h_raw` khác nhau; forward pass đổi 3.6% khi bật cờ trên cùng trọng số.
- Đã re-score toàn bộ 5 run sang per-class. `no_high_band` 27.86/24.35/30.20/30.11,
  `no_mid_band` 24.96/22.74/31.67/31.11, `no_low_band` 28.41/23.65/32.43/29.96,
  `no_router` 26.43/24.85/27.51/29.72 (sea all / sea tiny2 / earth all / earth tiny2).
  `no_saliency` đang chạy inference CPU.

### Bản thảo

- Table 4 mới (per-class) là bảng chính; Table 3 cũ giữ lại nhưng gắn nhãn legacy.
- Abstract, Introduction, §5.1, §6, Conclusion viết lại theo phạm vi hàng hải.
- Sửa công thức: σ tường minh (Eq. 9), ω là spatial `R^{3×H/8×W/8}` (Eq. 12),
  định nghĩa `Branch_k` (Eq. 19), Table 2 thêm `bn3` (64) để tổng đúng 34.666.
- Figure 3: 64 → 32 kênh. Figure 2: 5 nhãn kênh sai đã sửa, bỏ `suptitle`.
- Figure 1: trục x ghi "Per-tile", chú thích thêm 7.3 FPS end-to-end.
- Loss curves chuyển sang Appendix A; footnote tác giả đã hiển thị đúng chân trang 1
  (kiểm bằng `pdftotext -bbox` trên `main.pdf`: y=705.6, trang cao 792).
- Biên dịch sạch: 0 lỗi, 0 overfull hbox, 16 trang.

### Chưa xác minh

- Chưa có đối chứng conv chuẩn cùng số tham số (~35k) — cần train mới, ~3h A10.
- `no_saliency` chưa có kết quả tại thời điểm ghi mục này; dòng trong `tab:ablation`
  vẫn để `running`.
- Chưa có địa chỉ email tác giả liên hệ (đã để `TODO(authors)` trong `main.tex`).
- **Vượt giới hạn trang:** WACV 2026 giới hạn 8 trang nội dung (không tính tham khảo).
  Đo được: bản gốc 10 trang nội dung (references từ trang 11), bản hiện tại 12 trang
  (references từ trang 13). Cần cắt 4 trang — chưa thực hiện, là quyết định biên tập.
- Figure 9/10 in ra ở ~2.6 pt vì nguồn rộng 715 pt nhưng in trong 48% cột (238 pt).
  Đã sửa `figsize` trong `modal/extract_figures.py` nhưng **chưa sinh lại hình** —
  script cần GPU Modal, và bản sinh lại chưa được kiểm chứng.
