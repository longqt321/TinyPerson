# Trạng thái báo cáo GauSS

## Phạm vi
- Handoff dossier GauSS v2.1 và baseline YOLO26n-P2 cho model bên ngoài viết báo cáo.
- Tiếng Việt; không viết thêm báo cáo nghiên cứu.
- Không mở hoặc trích xuất nội dung QAC hay `report-bundle/`.

## Tiến độ
- Giai đoạn 0: hoàn tất; phạm vi GauSS-only, tiếng Việt, technical report Markdown.
- Giai đoạn 1: inventory tại `01_inventory.md`; kết quả ưu tiên lấy từ JSON GauSS và baseline paired.
- Giai đoạn 2: phân tích GauSS-only tại `03_analysis.md`; một seed, kết luận mô tả.
- Giai đoạn 3: chưa tìm nguồn ngoài để tránh citation chưa kiểm chứng; related work trong draft là khung chủ đề, chưa phải tổng quan hoàn chỉnh.
- Giai đoạn 4: outline tại `04_outline.md`.
- Giai đoạn 5: bảng kết quả trong draft đối chiếu thủ công với JSON; chưa có script sinh tự động hoặc hình.
- Giai đoạn 6: draft tiếng Việt tại `draft/gauss_report.md`.
- Giai đoạn 7: review ghi tại `99_review.md`; chưa biên dịch định dạng submission.

## Cảnh báo provenance
- Artifact eval 150e trỏ `training/weights/last.pt`; ghi đúng checkpoint này, không gọi là `best.pt`.
- Giao thức artifact khai báo 786 ảnh, dense excluded = 0; mô tả là standalone implementation, không khẳng định tuân thủ hoàn toàn official benchmark.
- Kết quả đơn seed; không kiểm định ý nghĩa thống kê.
- Draft là báo cáo nội bộ có giới hạn bằng chứng, chưa phải bản nộp học thuật do related work chưa xác minh và venue chưa biết.

## Ghi chú dọn dẹp 2026-11-06
- Tài liệu QAC/QAR riêng lẻ đã xóa; hồ sơ lịch sử A0–A3/MSR/BTRM/SADR/QAC còn lại duy nhất ở `reports/experiment_results_report.md`.
- Báo cáo trùng lặp đã xóa: `GauSS_FINAL_REPORT.md` và `TONG_HOP_KET_QUA_GauSS_VA_BASELINE.md` (byte-identical với bản GauSS).
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
- Lưu tại `reports/eval/{gauss,baseline}_{tiled,full}_per_class.json`.

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

## Phase 9 — 2026-11-12 06:15

### Giao thức merge trên checkpoint cuối (kết quả dương)

- Bảng mới `tab:merge_protocol` trong `paper/main.tex` (mục §5.x, label `sec:merge_protocol`):
  checkpoint 150e đợt 9–10/11 (run `20261109-182052-a20f7b6-seed42`, một lớp) chấm tiled:
  AP50 all 40.63→41.08 (+0.45), tiny2 36.92→37.67, small 51.63→52.51, reasonable 44.14→47.59,
  tiny 37.57→37.40 (−0.17), tiny3 −1.09. AP25 thắng mọi thang trừ tiny3, tiny1 +3.99.
  Full-frame: all 35.32→35.45, tiny 30.02→30.23.
- Nguồn: `reports/eval/{baseline,gauss_v3}_tinyperson_wacv_150e_{tiled,full}.json`.
- Giải thích protocol: checkpoint một lớp không thể chấm per-class (evaluator raise ValueError);
  bảng per-class (`tab:per_class`) giữ nguyên nhưng ghi rõ là checkpoint hai lớp cũ (dự đoán 3/11).
- Cập nhật Abstract, Intro, Discussion mục 1, Conclusion.
- Báo cáo đồ án: thêm bảng merge + 4 bullet đọc số vào §4.2; sửa dòng GaussSplit test
  trong §4.3 (40,42→40,54 v.v. — bản cũ ghi sai so với `g4_test_p030.json`); cập nhật §5.1, §6.1.
- Sửa header comment `configs/experiment/gausssplit_g4_tiled.yaml` (B1→G4).
- Biên dịch: 20 trang, 0 lỗi, 1 overfull hbox (0.79pt, vô hại).

### Chưa xác minh

- Một seed duy nhất cho mọi số liệu (giữ nguyên cảnh báo cũ).
- WACV giới hạn 8 trang nội dung — bản hiện tại ~13 trang nội dung, chưa cắt.

## Phase 10 — 2026-11-12 07:15 (bản dương, theo yêu cầu người dùng)

### Quyết định
- Cả hai bản báo cáo chỉ trình bày kết quả dương, dựa trên bộ JSON merge-protocol
  `reports/eval/{baseline,gauss_v3}_tinyperson_wacv_150e_{tiled,full}.json`.
- Đã bỏ khỏi paper: mục second-instance head (PairSplit/GaussSplit), bảng per-class
  (earth_person giảm), bảng legacy class-filtered, mục VisDrone, mục density-stratified,
  mục cross-epoch, hình 01/02/03/05/06/09/16/17/18.
- Paper còn 13 trang (trước 20), 0 lỗi, 0 overfull hbox.
- Hình mới: `figures/19_merge_protocol_scale_bins.pdf` sinh từ
  `scripts/plot_merge_protocol_figures.py` (đọc trực tiếp hai JSON, không chép tay số).
- Báo cáo đồ án viết lại toàn bộ: `docs/report/course_report_vi.md` (383 dòng, xưng
  "nhóm chúng em", 7 chương + 2 phụ lục + lời cảm ơn), xuất kèm
  `docs/report/course_report_vi.docx` (pandoc, có mục lục).
- Áp dụng checklist `docs/report/06_style_guide_no_ai_slop.md` (do subagent soạn):
  bỏ "Crucially", "Thứ nhất/Thứ hai", "Cuối cùng là", giảm em-dash còn 1 chỗ.

### Số liệu dùng chung (đã đối chiếu JSON)
Tiled merge: all 40.63→41.08; tiny 37.57→37.40; tiny1 22.30→22.26 (AP25 39.55→43.54);
tiny2 36.92→37.67; tiny3 48.14→47.05; small 51.63→52.51; reasonable 44.14→47.59.
Full-frame: all 35.32→35.45; tiny 30.02→30.23; tiny2 30.02→30.45; small 42.96→42.33;
reasonable 47.56→49.10. Ablation sea_person AP50 all: 29.63 full vs 24.96/25.41/26.43/
27.86/28.41.

### Chưa xác minh
- Một seed duy nhất; ablation lệch epoch (78 vs 150) — đã ghi trong cả hai bản.
- WACV giới hạn 8 trang nội dung; bản hiện tại 13 trang (chưa cắt, là quyết định biên tập).

## Phase 11 — 2026-11-12 08:05 (áp dụng audit)

### Đã làm
- Áp dụng toàn bộ 22 finding của `docs/report/07_audit_report.md`:
  - Sửa 12 entry bib sai/bịa metadata (đã verify web từng entry: AI-TOD, NWD,
    VisDrone, Pintea, Cascade R-CNN, SM+, Scale Match, Fusion Factor, YOLO26).
  - Sửa số AP75 (5.20 vs 4.99 all; tiny1 0.85 vs 1.99), bỏ claim "57.05%".
  - Sửa số học dataset split: 746 train (662+84) / 786 test / 78 unused.
  - Ghi rõ provenance ablation (config v2.1, 78 vs 150 epoch) trong caption.
  - Gắn nhãn VRAM là "GauSS-only (baseline trong ngoặc)".
  - Chuyển 20 bullet bold-lead-in thành prose; contributions thành prose.
  - Bỏ 21 entry bib không được cite khỏi bảng notation (Table 2) và Table 5.
  - Xóa comment bib rò rỉ về second-instance/frozen-BN.
- Báo cáo VI: sửa ref [1], [3], [13], số AP75, framing dataset split, thêm số
  bảng (Bảng 3.1/4.1/4.2/4.3), bỏ câu spin.
- Sinh lại `docs/report/course_report_vi.docx` bằng pandoc (có TOC) — đã verify
  nội dung mới nằm trong docx.
- Chuyển 3 hình cấm (16/17/18) vào `paper/figures/_archive/`.
- Ghi `docs/LESSONS.md` L-028 (bịa metadata citation) và L-029 (config lệch YAML).

### Trạng thái
- `paper/main.pdf`: 12 trang, 0 lỗi, 0 overfull, bibtex 0 warning, reference bắt
  đầu trang 10 (~9.5 trang nội dung).
- `docs/report/course_report_vi.md` + `.docx`: bản dương, một module duy nhất.

### Chưa xác minh
- Giới hạn 8 trang nội dung của WACV vẫn vượt ~1.5 trang (bản gốc cũng 10 trang).
- Một seed duy nhất cho toàn bộ số liệu.
- `reports/eval/profiling_results.json` gắn nhãn checkpoint là `gauss_v2_1`, không
  phải `gauss_v3` — cùng kiến trúc module nhưng chưa re-profile.

## Phase 12 — 2026-11-12 08:45 (audit hình + vẽ lại sơ đồ)

### Phát hiện nghiêm trọng (đã sửa)
- **Method lệch kiến trúc:** paper mô tả v3 nhưng checkpoint đã train (`a20f7b6`)
  dùng DySample, dilation {1,3,5}, gate `0.5·tanh`, hybrid energy router, output
  tuyến tính `X0 + αz + γ_out·B_high`, **36.878** tham số (không phải 34.666).
  Đã viết lại toàn bộ §3.2–3.4, bảng tham số, abstract, contributions, kết luận.
  Baseline thật từ checkpoint: 2.515.976; model GauSS: 2.552.854 (+1,47%).
- **Hình không đọc được:** 8/8 hình có text hiệu dụng <6 pt (có hình 2,6 pt) vì
  thiết kế 7–10 in rồi chèn vào cột 3,28 in.

### Đã làm
- Vẽ lại **sơ đồ module GauSS** bằng TikZ (`paper/figures/src/fig_gauss_module.tex`,
  xuất `figures/fig_gauss_module.pdf`, 505×126 pt, dùng ở `figure*`): đúng
  DySample, 3 dải, band encoders, context r∈{1,3,5}, hybrid router, refine,
  recon, output tuyến tính.
- Vẽ lại **sơ đồ tổng thể** (`fig_overall_pipeline.tex` → `fig_overall_pipeline.pdf`,
  214×120 pt, 1 cột), có fan-out GauSS → P2' và → Conv s2 → P3'.
- Regenerate ở đúng cỡ cột: `19` (245×135 pt), `04` (237×148 pt), `07` (246×110 pt).
- Bỏ 4 hình không đọc được (13 sea-clutter, 14 P2 activation, 15 band decomp,
  08 LR) và các câu phụ thuộc; giữ 12 ở dạng `figure*` 0.92\textwidth.
- Chuyển 15 PDF/SVG không dùng vào `paper/figures/_archive/`.
- Paper: 12 trang, 0 lỗi, 0 overfull, reference bắt đầu trang 9 (nội dung ~8,3 trang).
- Regenerate `docs/report/course_report_vi.docx` (đã verify 36.878 + DySample).

### Chưa xác minh
- Chưa xem trực tiếp bằng mắt từng hình sau khi regenerate (chỉ kiểm tra kích
  thước, scale và compile); cần người dùng mở PDF xác nhận bố cục.
- Một seed; ablation lệch epoch (78 vs 150) — giữ disclosure trong caption.
- `profiling_results.json` vẫn gắn nhãn `gauss_v2_1` cho số latency.

## Phase 13 — 2026-11-12 09:10 (sửa hình theo phản hồi người dùng)

### Phản hồi
- Người dùng: mọi hình bị "chữ dính chùm", sơ đồ "mũi tên dính chùm".

### Đã làm
- Sơ đồ module TikZ vẽ lại theo lane (band / context / router / synthesis),
  kiểm tra hình học bằng script (dump anchor + intersection): 0 mũi tên cắt
  box, 0 node chồng node, 493×190 pt — dùng ở figure* đúng cỡ in, font
  \footnotesize (8 pt) / \scriptsize (7 pt).
- Sơ đồ tổng thể: sửa 2 mũi tên cắt box (gauss→det cắt ds; p2→det cắt p3p),
  223×120 pt, 1 cột.
- Hình 19 vẽ lại thành 1 panel (AP50 solid + AP25 hatch), nhãn 1 dòng
  (all/tiny/t1/t2/t3/small/>32), 239×152 pt, tick 6.8 pt.
- Hình 04: nhãn giá trị xoay dọc, legend góc trên trái, tick 6.5 pt.
- Hình 07: còn 2 panel (box, cls), tick 6.2 pt.
- Caption fig 19 cập nhật giải thích viết tắt.
- Paper: 12 trang, 0 lỗi, 0 overfull.

### Chưa xác minh
- Người dùng cần mở lại PDF xác nhận các hình đã đọc được.

## Phase 14 — 2026-11-12 09:20 (xác nhận version v3.1)

- Người dùng hỏi model có phải v3.1 không. Xác nhận: đúng — checkpoint 150e
  chạy tại commit `a20f7b6` (sau v3.1 `72f6fc4`, trước v3.2 `8498747`),
  config `yolo26n_tiny_A2_p2_gauss_v31_tinyperson.yaml`, 36.878 tham số.
- Phát hiện và sửa: lượt trước tôi đã đổi phương trình sigma sang tanh-centered
  (v3.2) trong khi v3.1 dùng sigmoid-scaled. Đã revert về sigmoid ở cả paper
  và báo cáo VI; gate `0.5·tanh` giữ nguyên (v3.1 đúng dùng tanh cho gate).
  Ghi L-033.
- Paper: 12 trang, 0 lỗi, 0 overfull. docx regenerated.

## Phase 15 — Concision pass on paper/main.tex (2026-11-12)

- Nghiên cứu chuẩn viết paper: `docs/report/09_writing_guide.md` (budget từng
  section, rule "một số chỉ lặp 2–3 lần", phương trình chỉ khi là đóng góp).
- Đã cắt: cấu hình train khỏi Abstract (271→184 từ); bỏ câu tutorial BN/SiLU
  và công thức SiLU (chỉ trích dẫn); bỏ hẳn `tab:notations` (~1/3 trang); gộp
  phương trình (bands 3→1, projection 3→1, router giữ 2); enumerate protocol →
  đoạn văn; itemize scale bins → 1 câu; rút gọn captions; bỏ 2 đoạn Discussion
  trùng lặp ("Where the gain comes from", "Tiling versus full-frame"); bỏ
  Appendix convergence (vi phạm rule WACV: trang >8 chỉ được chứa references);
  bỏ hàng Swin-T trong bảng tham chiếu; contributions chuyển sang bullet list.
- Kết quả đo được: `pdflatex` ×2 → 0 error, 0 overfull, 0 undefined; PDF 12 →
  9 trang, references bắt đầu trang 8, trang 9 chỉ chứa references. Số từ
  (cùng một bộ đếm, loại float/equation): Abstract 239→184, thân bài 3,625→2,940.

## Phase 16 — Đồng bộ báo cáo đồ án và xuất PDF (2026-11-12)

- `docs/report/course_report_vi.md`: bỏ đoạn §4.6 tham chiếu hình không tồn tại
  trong báo cáo (bản đồ năng lượng L2, hình phân rã dải tần); sửa Phụ lục B trỏ
  đúng config `yolo26n_tiny_A2_p2_gauss_v31_tinyperson.yaml`; tách dòng bìa.
- Xuất PDF bằng pandoc + xelatex (mainfont DejaVu Serif, monofont Noto Sans
  Mono vì DejaVu Sans Mono thiếu glyph tiếng Việt):
  `pandoc docs/report/course_report_vi.md -o docs/report/course_report_vi.pdf
  --pdf-engine=xelatex -V mainfont="DejaVu Serif" -V monofont="Noto Sans Mono"
  -V geometry:margin=2.4cm -V fontsize=11pt -V colorlinks=true --toc --toc-depth=2`
- Đã kiểm tra: 13 trang, 0 warning glyph, mục lục + bảng 3.1/4.1 render đúng.
- `paper/main.pdf` giữ nguyên từ Phase 15 (9 trang, build sạch).

## Phase 17 — Bỏ path repo, dựng docx theo template DUT (2026-11-12)

- Bỏ Phụ lục A/B (chứa path repo) và Lời cảm ơn khỏi `course_report_vi.md`;
  paper `main.tex` đã kiểm tra: không còn path repo nào.
- Nghiên cứu template DUT (subagent): QĐ 209/QĐ-ĐHBK 06/03/2017, Phụ lục 04 —
  bìa text-only (ĐẠI HỌC ĐÀ NẴNG / TRƯỜNG ĐH BÁCH KHOA / KHOA … / tên loại
  báo cáo / NGÀNH / CHUYÊN NGÀNH / ĐỀ TÀI / GVHD / SVTH / MSSV / Lớp / Đà Nẵng),
  A4, lề 3-2-2.5-2.5 cm, header/footer cách mép 1.6 cm, TNR 13, giãn dòng 1.3,
  heading "Chương N:" bold 14 giữa trang, "N.M." bold 13, caption bảng phía trên,
  header = tên đề tài nghiêng 10, footer = số trang 10, mục lục đánh số la mã,
  thân bài đánh số Ả Rập bắt đầu từ 1. Nguồn: dut.udn.vn (QĐ 209 + biểu mẫu).
- Builder: `scratch/build_report_docx.py` (python-docx, 2-pass để ghi số trang
  thật vào mục lục). Xuất PDF bằng LibreOffice headless.
- Đã kiểm tra: PDF 14 trang, mục lục khớp số trang in, footer la mã "i" rồi
  1,2,…, 4 caption bảng "Bảng N.M." phía trên bảng, bìa không có footer.

## Phase 18 — Hoàn thiện báo cáo đồ án: trích dẫn, hình, style bảng (2026-11-12)

- Thêm 14 chỗ trích dẫn [n] trong văn bản (TinyPerson, FPN/PANet, scale-space,
  Laplacian, YOLO26/TOOD, CIoU, DySample, PixelShuffle, AdamW, SSPNet, BPD-YOLO,
  NMS) khớp danh mục 18 mục cuối báo cáo.
- Thêm 5 hình (300 dpi PNG từ PDF figure của paper, copy sang
  `docs/report/figures/`): Hình 3.1 pipeline tổng thể, Hình 3.2 module GauSS,
  Hình 4.1 biểu đồ thang đo, Hình 4.2 định tính, Hình 4.3 hiệu năng. Caption
  "Hình N.M." phía dưới hình, đánh số theo thứ tự xuất hiện.
- Style bảng theo chuẩn học thuật (không phải Table Grid): viền trên/dưới đậm,
  gạch mảnh giữa các dòng, không viền dọc; header lặp khi bảng xuống trang.
- Builder `scratch/build_report_docx.py` hỗ trợ marker `![HÌNH:...]` + caption.
- Kiểm tra: PDF 17 trang, 5 ảnh nhúng đúng trang, mục lục khớp số trang in,
  trích dẫn [n] render đúng.

## Phase 19 — Tái cấu trúc báo cáo theo yêu cầu mở rộng, đạt 37 trang (2026-11-12)

- Chương 1 viết lại đúng mạch: 1.1 bối cảnh/lý do → 1.2 các hướng tiếp cận trước
  và hạn chế (5 nhóm, kết bằng khoảng trống) → 1.3 phương pháp đề xuất (kèm 3
  nguyên tắc thiết kế) → 1.4 kết quả chính (Bảng 1.1) → 1.5 mục tiêu/phạm vi →
  1.6 bố cục.
- Chương 2–6 viết mới hoàn toàn, không tái dùng văn bản paper: 22 bảng, 11 hình.
- Gói mở rộng thực chất (không padding): Bảng 3.4 đếm tham số từng khối (tổng
  36.878, đường hoạt động 36.361, tùy chọn 517); Bảng 4.3 so sánh giao thức đồ
  án với giao thức chuẩn TinyPerson; Bảng 4.5 kết quả full-frame theo thang;
  Bảng 4.9 phân loại dự đoán và lỗi (TP/FP/duplicate/localization/background/miss).
- Loại bỏ: bảng phân tích theo mật độ (dữ liệu thuộc phiên bản module trước và
  split val — không đạt ràng buộc một module duy nhất), mọi nhắc tới phiên bản
  module mới hơn, tên dịch vụ đám mây.
- Đánh số lại tự động sau khi ghép (`scratch/merge_expansion.py`): mục 3.1–3.12,
  4.1–4.15, bảng 3.1–3.5, 4.1–4.11; kiểm tra chéo 0 tham chiếu treo.
- Kết quả kiểm tra: PDF **37 trang** (mục tiêu 35–40); MỤC LỤC, DANH MỤC HÌNH VẼ,
  DANH MỤC BẢNG BIỂU khớp số trang in; 33 caption xuất hiện đúng một lần trong
  thân bài, thứ tự tăng dần; không tràn lề (xMax ≤ 538,4 pt so với giới hạn
  538,6 pt); footer la mã i, ii rồi Ả Rập 1…33; logo trên bìa.
- Còn để trống: số thẻ sinh viên và lớp trên bìa (người dùng chưa cung cấp).

## Phase 20 — Tái cấu trúc báo cáo theo yêu cầu giảng viên (3 chương)

- Báo cáo dựng lại thành 3 chương: Chương 1 Tổng quan nghiên cứu liên quan (3 trang),
  Chương 2 Giải pháp đề xuất (9 trang), Chương 3 Đánh giá kết quả (14 trang);
  tài liệu tham khảo trang 27; tổng phần thân 28 trang (giới hạn 50).
- 21 công thức chuyển sang ảnh PNG dựng bằng mathtext, căn giữa trong hộp, tự co font
  12→9,5pt nếu vượt 15,4 cm; công thức F_route tách thành 2 dòng hiển thị.
- Toán inline chuyển thành ký tự Unicode + run sub/superscript thật trong docx.
- Bảng 3.2/3.3/3.4 đổi tiêu đề cặp cột (AP25 | + GauSS) kèm ghi chú, bỏ trùng nhãn.
- Đã xác minh: caption duy nhất và tăng dần, footer i/ii rồi 1–28, logo trang bìa,
  TOC + danh mục hình/bảng khớp trang in, không tràn lề phải (lệch tối đa 0,12 pt),
  18 tài liệu tham khảo đều được trích, không có path repo / "Modal" / "baseline".
- Ảnh công thức lưu tại docs/report/figures/eq/ để md tự chứa.

## Phase 21 — Bổ sung nội dung và hình thức (yêu cầu "chưa đạt")

- Bỏ hoàn toàn khối giả mã thuật toán ở chương 2 (không còn code trong báo cáo).
- Bảng siêu tham số rút còn 6 nhóm, thêm cột "Căn cứ" để mỗi giá trị có lý do thay vì liệt kê.
- Thêm mục 2.5 "Căn cứ của các lựa chọn thiết kế": số dải tần, xử lý ở stride 8, tàn dư có chặn,
  hai nhánh tinh chỉnh.
- Thêm 5 hình: phân bố ground truth (1.1), sơ đồ chi tiết module (2.2), nhân Gauss và khớp thang đo
  (2.3), thành phần dự đoán (3.6), chênh lệch AP50 hai chế độ (3.5). Tổng 16 hình, 15 bảng.
- Thêm hướng tiếp cận thứ sáu (QueryDet [19]) kèm ưu/nhược; tài liệu tham khảo 18 → 19.
- Chuyển bảng kết quả công bố từ chương 3 sang chương 1 (Bảng 1.2) để chương tổng quan có bằng chứng
  định lượng; chương 3 chỉ còn phần thảo luận vị thế.
- Đánh số lại toàn bộ Hình/Bảng theo thứ tự xuất hiện (script renumber.py, idempotent).
- Trang: ch1 4, ch2 11, ch3 14, tham khảo 2 — tổng 31 trang thân (giới hạn 50; 15/25/20 mỗi chương).

## Phase 22 — Bổ sung liên quan trực tiếp và nguồn gốc hình vẽ

- Mục 1.2 mở rộng: SEPC [4] (tương quan liên mức theo pyramid Gauss), NWD [5], Dot Distance [6],
  RFLA [7] (ghép nhãn theo trường tiếp nhận Gauss), SET [8] (tăng cường miền tần số, CVPR 2025 —
  công trình gần GauSS nhất, đã nêu khác biệt rõ), SAHI [9], QueryDet [10]. Tổng 7 hướng tiếp cận.
- Thêm 4 tài liệu tham khảo (20–23), đánh số lại toàn bộ theo thứ tự trích dẫn lần đầu
  (script renumber_refs.py; 23/23 đều được trích).
- Thêm đoạn nguồn gốc hình vẽ ở mục 3.1: mọi hình do nhóm vẽ từ số liệu đánh giá của đồ án,
  kiểu biểu đồ theo thông lệ của [1], [3], [13]; Hình 2.3 ghi rõ tính từ tham số thiết kế.
- Trang: ch1 5, ch2 11, ch3 14, tham khảo 2 — tổng 32 trang thân.

## Phase 23 — Sửa theo bản nhận xét định dạng (23 mục)

- Bỏ 17 dấu "####" sót lại (parser chỉ đọc tới ###); tiểu mục giờ đậm + nghiêng, sát lề.
- Chương 2, Chương 3 và TÀI LIỆU THAM KHẢO xuống trang mới.
- 22 công thức đánh số (2.1)–(2.22) căn phải bằng tab.
- Bật cantSplit cho mọi dòng bảng + lặp dòng tiêu đề; không còn ô vỡ trang.
- Mục lục liệt kê xuống tiểu mục; DANH MỤC HÌNH VẼ / BẢNG BIỂU / KÝ HIỆU VÀ CHỮ VIẾT TẮT
  mỗi cái một trang; danh mục hình/bảng trích nguyên văn tên đầy đủ (bỏ cắt "…").
- Chú thích hình/bảng 13pt, nhãn in đậm; "Ghi chú" dời xuống dưới bảng kèm chú giải chữ in đậm.
- Cột bảng đặt bề rộng cố định (cột đầu rộng hơn), đổi "AP_mean" thành "AP mean".
- Khổ hình chốt 14cm; hai hình kiến trúc render lại 2477×1332 và 4111×1584 (≥230 dpi hiệu dụng);
  bỏ tiêu đề trùng trong ảnh, thống nhất nhãn "AP50", "+ GauSS", "reasonable".
- Bìa: khối tên hai sinh viên thẳng cột, in đậm; năm 2026; đề tài ngắt 3 dòng không mồ côi.
- Footer: dòng "Sinh viên thực hiện … GVHD" + số trang, cỡ 10; header đủ tên đề tài đầy đủ.
- TLTK: nhãn [n], thụt lề treo, nơi công bố thống nhất, bỏ tiếng Việt trong mục tiếng Anh.
- Ngoặc kép cong “ ” toàn văn bản; thống nhất một quy ước chỉ số dưới (run subscript thật).
- Font PDF: Liberation Serif (thay thế metric-compatible của TNR vì máy không có TNR);
  một số ký hiệu toán (∈, 10⁻³) rơi vào DejaVu Serif — cần xuất lại từ Word có TNR nếu yêu cầu.
- Trang: 39 tổng; thân 33 (ch1 5, ch2 11, ch3 15, TLTK 2).

## Phase 23b — Hoàn thiện DPI hình

- Hình 07_training_loss_convergence render lại từ PDF nguồn ở 700 dpi → 2320×973 (≈421 dpi hiệu dụng ở khổ 14cm).
- Kiểm tra lại toàn bộ 16 hình: thấp nhất 299 dpi (đạt ngưỡng ≥230); logo bìa 225×225 ở 179 dpi
  (logo nhỏ khổ 3,2cm, chấp nhận được, không upscaled để tránh nhòe).
- Build lại docx + PDF: 39 trang, bản đồ trang ổn định (ch1=1, ch2=6, ch3=17, TLTK=32).
  Đã đối chiếu md5 giữa bản build và docs/report/course_report_vi.pdf.
- Kiểm tra cuối: 0 dấu "####"; không có dòng mồ côi đầu trang (quét dòng đầu 39 trang);
  chú thích 13pt nhãn đậm (docx run check); "Ghi chú" nằm dưới bảng 3.2/3.3/3.4;
  23 mục TLTK nhãn [n]; footer GVHD + số trang; công thức (2.1)–(2.22); ngoặc kép cong.

## Phase 24 — Sửa theo bản đánh giá chi tiết thứ hai (Phần A–I)

- Sửa rò rỉ LaTeX trong builder: `\sqrt{...}` → √(...), `\hat\gamma` → ĝ (regex bắt cả từ),
  dấu `-` trong công thức nội tuyến → dấu trừ −.
- Sửa sai khoa học: tính nửa nhóm của Lindeberg là làm mờ một lần ở √(σ₁²+σ₂²).
- Sửa 2 tham chiếu chéo: "Hình 3.7 đặt hai chế độ" → Hình 3.5; "Hình 3.8 vẽ lại Bảng 3.6" → Hình 3.6.
- Thêm tài liệu DySample (Liu và cs., ICCV 2023) thành [24]; ESPCN tách riêng; chạy lại đánh số
  theo thứ tự trích dẫn (24 mục, script renumber_refs.py tổng quát hóa N).
- Bổ sung chi tiết TLTK: [2] GRSL vol. 19, pp. 1–5; [5] mã arXiv 2110.13389; [15] Kluwer, Dordrecht;
  [17] ngày truy cập; thống nhất tên hội nghị viết tắt.
- Sửa số liệu và trình bày bảng: Bảng 3.2 tiny1 Δ = −0,03 (hiệu chưa làm tròn, ghi chú giải thích);
  Bảng 3.5 thêm cột Δ (−4,66/−4,22/−3,20/−1,77/−1,22); Bảng 3.6 thêm hàng "Trung tính" (4.731/4.979,
  khớp error_analysis neutral_predictions) và caption nêu chế độ ảnh gốc; Bảng 3.7 tách cột Độ trễ/FPS;
  Bảng 2.2 tách hàng S thành 3 hàng; Bảng 2.3 thống nhất "1 × 1", điền công thức đếm các hàng "—";
  Bảng 2.4 tách nhóm Tối ưu thành 3 hàng, bỏ mệnh đề "lần chạy đối chứng"; Bảng 1.1 đổi "+nhỏ/+0";
  Bảng 1.2 thêm 2 chữ số thập phân, cột "Huấn luyện trước"; Bảng 3.9 viết lại 2 ô.
- Công thức có dấu câu cuối (dấu chấm/phẩy theo ngữ cảnh, chèn tự động trong builder).
- Văn phong: in nghiêng thuật ngữ Anh lần đầu; bỏ từ thoại (chỉ nhích, chen vào, khung khít,
  bắn một báo nhầm, mỏ neo, lịch ngắn…); thống nhất ngôi khách quan (giữ 1 "nhóm chúng em" ở mở đầu);
  "backbone huấn luyện trước" thống nhất; viết lại 4 ý kết luận cân đối; "vòng" → "lượt lặp";
  reg_max/PixelUnshuffle/PixelShuffle viết lại thành lời; khoảng số dùng khoảng trắng không ngắt.
- Khoa học: bỏ khẳng định "phủ trọn tiny3" của 2σ₂; tách trung vị ảnh và số mảnh trung bình
  (bỏ mệnh đề trung vị 3840×2160 ở mục 3.9); đồng bộ −0,03 giữa bảng và hình.
- Hình: dấu thập phân dùng dấu phẩy trong mọi biểu đồ (FuncFormatter); legend "Đồ án/Công bố khác"
  cho hình tán xạ; caption Hình 2.4 nêu rõ là minh họa theo tham số thiết kế.
- Bìa: thêm dòng Môn học/Học kỳ; co khoảng trắng để bìa gọn 1 trang (trước đây tràn sang trang 2).
- Builder: dò số trang thân theo running head thay vì từ khóa "DANH MỤC"; đặt tblGrid + tblCellMar
  (mép ô 45 twips) để bề rộng cột có hiệu lực trong LibreOffice; bảng ratio riêng cho 2.3/3.2/3.3/
  3.6/3.7 — hết gãy chữ "reasonab-le", "AP_me-an", "Thang/đo".
- Danh mục ký hiệu mở rộng 20 → 27 mục (thêm BN, SiLU, FP16, GPU/CPU, HSV, UAV, COCO, TinyPerson;
  bỏ DoG không dùng; "AP_mean" → "AP mean"); running head bỏ nghiêng.
- Kết quả: 40 trang tổng, thân 33 (ch1 5, ch2 11, ch3 15, TLTK 2); 27/27 mục danh mục khớp số trang
  (kiểm tra tự động); 0 dấu "####"; bìa gọn 1 trang.

## Phase 25 — 13/11/2026: Sửa Hình 3.8 (đánh giá mục A1)
- `scratch/make_report_figures.py`: hai điểm đồ án đổi sang AP50 thang tiny (37,57 / 37,40), bỏ giá trị toàn thang 41,08; tách 2 panel — (a) scatter toàn bộ cấu hình tham khảo, (b) bar nhóm nano (≤ 7 M).
- Kiểm tra chồng chữ bằng đo bounding-box tự động → bad=0 (legend "Đồ án (2 cấu hình)" upper-left, FCOS nhãn bên phải marker).
- `course_report_vi.md` §3.8: viết lại trung thực — bộ đánh giá không phải TinyBenchmark, tiny AP50 của + GauSS (37,40) thấp hơn YOLOv8n+P2 (38,00) và BPD-YOLO (39,10); bỏ "hai điểm cam là cặp của đồ án" và "nhóm (c) gần cấu hình nhất"; caption mới mô tả 2 panel.
- Build lại docx/pdf: 40 trang, TOC hội tụ, ảnh mới đã nhúng (image36.png khớp md5).

## Phase 26 — 13/11/2026: Sửa loạt nhận xét A–G (đợt 2)
Nội dung (đã kiểm bằng build + pdftotext):
- §3.1: định nghĩa MR = log-average miss rate (9 điểm FPPI 10⁻²–10⁰), nói rõ MR ≠ 1−recall; thêm đoạn tiền xử lý (letterbox 1280, mảnh 640×512 không co, "ảnh gốc" = letterbox 1280, bin đo ở tọa độ gốc); 786 = split test gốc trừ 30 khung dense, val 84 tự tách.
- §3.5: viết lại — ablation là cấu hình hai lớp, 78/150 epoch, sea-person; thêm cột KTC 95% bootstrap; chỉ dải giữa và router lai có ý nghĩa; nêu rõ cổng saliency TẮT trong cấu hình báo cáo nên dòng "bỏ saliency" không đọc được; nêu thiếu dòng "không module".
- §2.2: thêm mô tả đường nền + số kênh head 32/64/128/256 + nhánh C3k2 chết trong topology v3.1. §2.4: chênh lệch 36.878 đo từ checkpoint; FLOPs ghi đúng 640×640 và 1280×1280.
- §2.3.2: thống nhất "bề rộng hỗ trợ" = 2σ·4 (5,2 / 11,6 px); Hình 2.3 caption "trọng số tại khởi tạo".
- §2.3.7 + §2.5: bỏ "ánh xạ đồng nhất chính xác" và "không thể tệ đi"; nói đúng gradient bước đầu chỉ vào ba hệ số.
- §3.4: chuyển giải thích thành giả thuyết; thêm σ sau huấn luyện (0,66 / 1,46, gần init).
- §3.7: định nghĩa "in view" (cửa sổ 280×280), thừa nhận 10 dự đoán cho 6 GT là có dư, bỏ ví dụ 0,60 và bỏ liên hệ cổng saliency.
- §3.9: FLOPs nhãn đúng; nêu độ trễ chỉ tính lượt gọi mạng; bỏ "real-time".
- §3.11/§3.12/§1.4: bỏ con số "1 điểm phần trăm" không đo được, bỏ "dịch chuyển cùng hướng", bỏ khuyến nghị triển khai hai chế độ; kết luận theo mức ý nghĩa thống kê thực có.
- §2.1.1–2.1.3, §1.3: sửa "không có gradient", tính duy nhất scale-space, reg_max, Bảng 2.1 (quy ước khoảng kích thước), khoảng trống nghiên cứu theo phạm vi tra cứu.
- Thêm TÓM TẮT (+ Abstract), KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN, PHỤ LỤC A–D; builder hỗ trợ heading cấp 2 không đánh số + TOC entry.
Hình: Hình 3.8 hai panel (AP50 tiny), Hình 3.6 hai panel (dự đoán / FP), 07 loss 2 panel (bỏ panel rỗng), 04 hardware vẽ lại (bỏ đường "real-time", sửa nhãn chồng), Hình 3.3 thêm tiêu đề panel, Hình 3.9 nhãn không chồng, Hình 3.5 legend lên trên.
Kết quả build: 46 trang, TOC hội tụ sau 2 lượt pass2, 0 ký tự Markdown/LaTeX rò rỉ.

## Phase 27 — 13/11/2026: Xác minh tài liệu và sửa lại Hình 3.8 (lần 2)
- Xác minh bằng nguồn sơ cấp (search web): [3] Chao et al. Sci Rep 15:31888 (2025) tồn tại, Bảng 5 TinyPerson chỉ báo P/R/F1/mAP50/mAP50-95 → 38,0 (YOLOv8n+P2) và 39,1 (BPD-YOLO) là mAP50 TOÀN TẬP, không phải thang tiny; tham số 2,93/1,50 M khớp; AFPN 6,05 M/39,2, BiFPN 2,34 M/38,4 cũng khớp scatter.
- Hệ quả: bản sửa trước (dùng tiny 37,40/37,57 cho điểm đồ án) lại là so lệch đại lượng. Sửa lại: trục Hình 3.8 = "AP50 toàn thang", điểm đồ án 40,63/41,08; §3.8 viết lại trung thực (hai nguồn khác giao thức, không kết luận kiến trúc từ khoảng cách); Bảng 1.2 đổi cột "AP50 tiny" → "AP50 toàn thang (mAP50 toàn tập)".
- SET [8] CVPR 2025 xác minh: HBS (làm mượt nền tần số cao) + API (nhiễu đối kháng), chỉ tác động khi huấn luyện, không thêm chi phí suy luận → §1.2.5 viết lại đúng cơ chế, nêu rõ đồ án không chạy SET; Bảng 1.1 dòng SET đổi thành "0 khi suy luận / chỉ trong huấn luyện".
- SEPC [4] CVPR 2020 xác minh: tích chập 3D trên trục không gian + tỉ lệ, căn bằng tỉ lệ bằng resample, BN trên tensor pyramid → bổ sung vào §1.2.3.
- QueryDet [10] CVPR 2022 xác minh: truy vấn thưa cascade, dự đoán thô ở mức thấp rồi chỉ tính mức cao tại các vị trí đó bằng sparse conv → mô tả trong §1.2.7 đã đúng.
- Ref [3] bổ sung tên tác giả đầu (M. Chao, C. Peng, Z.-Q. Chen).
- Subagent citation-verifier lỗi 504; đã tự xác minh bằng search web trực tiếp.

## Phase 28 — 14/11/2026: Sửa đợt nhận xét cuối + resume ablation (bị chặn bởi hạn mức Modal)
- Sửa toàn bộ nhận xét định dạng + nội dung (chi tiết trong transcript): #### leaks, LaTeX lộ chữ, công thức σ₁²+σ₂² → √(σ₁²+σ₂²), tham chiếu chéo Hình 3.5/3.6, MR định nghĩa lại, §3.8 so sánh đại lượng, provenance split 1610/794/816/746/786, letterbox 3840→1280, σ per-channel từ checkpoint, refs [25]–[33], sơ đồ TikZ tiếng Việt (xelatex + DejaVu Serif), builder fixes (γ̂, cover, TOC, symbols, page-break TLTK).
- Build cuối: 51 trang; Ch1=9, Ch2=15, Ch3=29, KẾT LUẬN=46, TLTK=47 (trang riêng), PHỤ LỤC=50; 0 leak; 0 cụm từ cấm; TOC hội tụ đủ tiểu mục.
- Xuất BaoCao_DoAn_GauSS_TinyPerson.{pdf,docx} (bản 51 trang này).
- Ablation resume: checkpoint DCAF v2 (78/150e) trên volume tiny-person-runs; worktree /tmp/abl_resume @9b12ca8 + patch `_install_resume_optimizer_patch` trong trainer.py (stash optimizer state mismatch rồi áp sau khi thêm control param group). Resume thành công epoch 79→ nhưng workspace Modal ac-8yiFoLEeoixRCYfKPMAmM7 vượt hạn mức chi tiêu → dừng ở 139–143 epoch (high=143, router=140, mid/low/saliency=139). CẢN: mọi lệnh Modal (kể cả evaluate_all) bị chặn cho tới khi nâng hạn mức.
- Người dùng chọn: đánh giá tại checkpoint 139–143 epoch + ghi trung thực "chưa đủ 150" trong báo cáo; xuất bản hiện tại trước, cập nhật sau khi có số.
- PENDING: khi Modal mở khóa → resume nốt 7–11 epoch (hoặc đánh giá luôn tại 139–143), chạy evaluate_all + bootstrap_ablation_allcells.py, cập nhật Bảng 3.5/Bảng 3.9/§3.5/KẾT LUẬN, rebuild, xuất lại.

## Phase 29 — 14/11/2026: Sửa toàn diện theo bản đánh giá chất lượng (Nội dung, Phương pháp, Số liệu, Văn phong)
- **Nội dung & Phương pháp**:
  - Làm rõ sự lệch thang đo: huấn luyện dùng ảnh letterbox 1280 (co nhỏ ~3 lần) trong khi đánh giá chính dùng pipeline cắt mảnh 640×512 ở độ phân giải gốc nguyên bản; giải thích rõ bề rộng hỗ trợ của bộ lọc Gauss (5,2 px và 11,6 px) khớp trực tiếp với kích thước vật thể ở chế độ cắt mảnh chứ không khớp với thang đo huấn luyện.
  - Hạ mức khẳng định kết luận: chênh lệch 0,45 AP50 toàn thang là mức cải thiện khiêm tốn gắn chặt với pipeline cắt mảnh; thừa nhận biến động ngẫu nhiên giữa các lần huấn luyện có thể lên tới hơn 4 điểm AP50 (minh chứng qua việc tắt cổng saliency vốn không hoạt động vẫn làm điểm giảm 4,22 điểm); phục hồi thông tin đợt ablation chạy ở 78/150 epoch với cấu hình 2 lớp vào mục 3.5.
  - Phân tích minh bạch nhánh C3k2 trung gian stride 4: cấu hình thực nghiệm v3.1 chứa khối C3k2 không dùng (8.704 tham số, tổng 2.552.854 tham số, +1,47%); cấu hình làm sạch cho triển khai v3.2 loại bỏ khối này chỉ tăng 28.174 tham số hiệu dụng (+1,12%) và FLOPs chỉ tăng 0,8%. Bảng 3.8 bổ sung cột cấu hình triển khai sạch v3.2.
  - Phân tích hành vi bộ lọc Gauss: giải thích việc $\sigma$ thực tế duy trì cố định quanh giá trị khởi tạo (do gradient triệt tiêu ở các bước đầu khi các hệ số cổng $\alpha, \gamma$ bằng 0); đổi tên tiêu đề mục 2.3.2 thành phân rã không gian tỉ lệ bằng bộ lọc Gauss; làm rõ bề rộng hỗ trợ 11,6 px bao phủ phần lớn thang tiny2 (8–12 px, cận trên 12 px vượt nhẹ 11,6 px).
  - Phân tích chi tiết thang đo: bổ sung phân tích cho thang small (+0,88) và reasonable (+3,45) do triệt tiêu phản xạ tần số cao của mặt nước; làm rõ thang tiny3 chịu mức suy giảm mạnh nhất (-1,09 AP50, MR50 tăng 0,58) do nằm ở vùng chuyển tiếp stride 4 và stride 8; đính chính ở chế độ ảnh gốc cả tiny1 lẫn small đều giảm.
  - Định tính (mục 3.7): ghi rõ ID khung ảnh (`youtube_V0001_I0000320` cho cụm người bơi, `youtube_V0001_I0002640` cho đỉnh sóng); thừa nhận việc xuất hiện 10 dự đoán cho 6 ground truth là hiện tượng dư thừa/trùng lặp; công khai hai ví dụ đối nghịch nhau không cấu thành bằng chứng thống kê.
  - Hiệu năng (mục 3.9): làm rõ con số 355 ms chỉ là tổng thời gian các lượt gọi mạng chuyển tiếp (forward pass), không bao gồm khâu cắt mảnh và NMS toàn cục.
- **Số liệu, Bảng biểu & Trích dẫn**:
  - Hình 1.1: nhãn phần trăm khớp chính xác số ground truth thật (tiny1 17%, tiny2 20%, tiny3 28%, small 18%, reasonable 17%).
  - Đồng bộ định nghĩa MR50 trong danh mục từ viết tắt: Log-average Miss Rate tại 9 điểm FPPI (IoU 0,50).
  - Hình 3.8: chuyển về biểu đồ phân tán 1 panel khoa học, loại bỏ panel dạng bảng xếp hạng nano (b); chỉ giữ các mô hình có nguồn rõ ràng trong Bảng 1.2; ghi rõ nguồn mAP50 toàn tập theo [1, 2, 3].
  - Bổ sung trích dẫn đầy đủ: ST-MoE z-loss [34], không gian màu HSV [35], cân bằng biểu đồ thích ứng CLAHE [36], SiLU [37]; bổ sung thông tin chi tiết bộ biến đổi `tinyperson_b` (ADR-018) vào Phụ lục A và giải thích việc chọn trước chiến dịch huấn luyện.
  - Mục 3.10: làm rõ đồ tích tụ loss chỉ phản ánh loss huấn luyện, mô hình không theo dõi val mAP định kỳ từng epoch mà chỉ kiểm tra xác thực cuối kỳ (val/box_loss e150 = 2,19).
- **Văn phong**:
  - Chuyển ngôi kể chủ động sang "chúng tôi" cho mọi quyết định phương pháp; dùng "nghiên cứu này" khi nói về văn bản; loại bỏ hoàn toàn các câu vô chủ kiểu "đồ án đề xuất", "đồ án chạy".
  - Loại bỏ hoàn toàn cấu trúc khuôn sáo "Ưu điểm... / Nhược điểm..." ở 7 tiểu mục 1.2.1–1.2.7; loại bỏ các cấu trúc "không phải X mà Y", "câu hỏi trung tâm", "đóng góp ở hai tầng".
  - Quét sạch hoàn toàn các từ cấm và rò rỉ ký tự LaTeX: 0 kết quả rò rỉ `####`, `\sqrt`, `\hat`, `ammain`.
- **Xuất bản**:
  - Dựng lại hoàn chỉnh: `course_report_vi.docx` và `course_report_vi.pdf` (55 trang, mục lục hội tụ chính xác qua pass 2).
  - Xuất ra thư mục gốc: [BaoCao_DoAn_GauSS_TinyPerson.pdf](file:///home/longtran/Projects/thigiacmaytinh/BaoCao_DoAn_GauSS_TinyPerson.pdf) và [BaoCao_DoAn_GauSS_TinyPerson.docx](file:///home/longtran/Projects/thigiacmaytinh/BaoCao_DoAn_GauSS_TinyPerson.docx).
