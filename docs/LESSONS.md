# Lessons

Bài học rút ra từ lỗi đã xảy ra. Mỗi mục có mã `L-###` để tham chiếu.

## L-001: Test "tự chèn sys.path" che mất lỗi import thật

- **Ngày:** 2026-11-09
- **Bối cảnh:** viết lại `launch_train_and_eval_60e.py`, xóa `sys.path.insert(... "modal")`
  nhưng vẫn `import app`. Khi verify, tôi tự chèn `sys.path` trong đoạn test rồi kết luận
  "module loads".
- **Hậu quả:** script không chạy được thật (`ModuleNotFoundError: No module named 'app'`);
  lỗi chỉ lộ ra ở lần rà soát sau.
- **Quy tắc:** verify đúng **đường thực thi thật** (import file như Modal/CLI làm), không
  dựng lại môi trường bằng tay trong test. Nếu test cần thêm bước chuẩn bị mà production
  không có, thì test đang che lỗi.

## L-002: Callback của Ultralytics không có try/except — mọi callback phải tự bảo vệ

- **Ngày:** 2026-11-09
- **Bối cảnh:** thêm `SystemMetricsCallback` / `GaussDiagnosticsCallback` vào
  `model.callbacks["on_fit_epoch_end"]`.
- **Phát hiện:** `BaseTrainer.run_callbacks` gọi `callback(self)` trực tiếp, không bắt
  exception. Một lỗi nhỏ (ví dụ `float()` trên numpy array 1 phần tử — numpy 2 raise
  `TypeError`) sẽ **làm chết training**.
- **Quy tắc:** mọi callback logging phải bọc `try/except` và tự disable
  (`self.enabled = False`) thay vì để exception lan ra. Coerce số bằng `.item()` khi giá
  trị có thể là numpy array.

## L-003: Ghi CSV append phải cố định fieldnames

- **Ngày:** 2026-11-09
- **Bối cảnh:** `csv.DictWriter` với `fieldnames` tính lại từ `rows` mỗi epoch.
- **Hậu quả:** nếu tập key thay đổi giữa các epoch (hoặc khi resume/two-stage tạo callback
  mới), cột dữ liệu lệch so với header đã ghi — số liệu sai âm thầm.
- **Quy tắc:** đọc header có sẵn từ file nếu file đã tồn tại; nếu chưa, chốt fieldnames ở
  lần ghi đầu và dùng `extrasaction="ignore"`.

## L-004: Đừng kết luận "đã verify" khi mới chỉ đọc code

- **Ngày:** 2026-11-09
- **Bối cảnh:** `fetch()` được viết dùng `Path("/mnt/runs")` — đường dẫn chỉ tồn tại trong
  container — và `Volume.read_file()` (async generator) trong vòng lặp đồng bộ.
- **Quy tắc:** với code chạy trên hạ tầng khác (Modal volume, container), phải kiểm tra
  **chữ ký hàm thật** (`inspect.signature`) và môi trường thật, hoặc ghi rõ vào mục
  "Chưa xác minh".

## L-005: Vector hóa từng detection vẫn chậm — phải batch theo ảnh

- **Ngày:** 2026-11-09
- **Bối cảnh:** đổi `_iou` scalar thành `iou()` numpy cho từng detection trong
  `_evaluate_size_bin`. Microbenchmark báo 7x, nhưng end-to-end chỉ 1.4x
  (34.6 s → 25.1 s trên profile 84 ảnh × 200 dets).
- **Nguyên nhân:** overhead gọi numpy (reshape, mask, `flatnonzero`) ~vài chục µs mỗi lần,
  nhân với ~350k lần gọi (84 ảnh × 200 dets × 3 thresholds × 7 buckets) — chi phí per-call
  lớn hơn phần tính toán được vector hóa.
- **Quy tắc:** khi vòng lặp có N lần gọi numpy trên mảng nhỏ, hãy tính **ma trận đầy đủ một
  lần** (`iou_matrix`, broadcasting) rồi để vòng lặp chỉ tra cứu hàng. Kiểm chứng bằng test
  tương đương (vectorized vs scalar reference) trước khi tin vào tốc độ.

## L-006: Callback wrap `validator.update_metrics` phải idempotent

- **Ngày:** 2026-11-09
- **Bối cảnh:** stress test gọi `on_train_start` hai lần trên cùng callback. Lần hai lấy
  `validator.update_metrics` (đã là wrapper của chính nó) làm "original" → đệ quy vô hạn
  (`RecursionError` sau ~1000 frame).
- **Hậu quả nếu vào production:** Ultralytics gọi `on_train_start` một lần mỗi `fit`, nên chưa
  xảy ra; nhưng bất kỳ thay đổi nào gọi lại (resume trong cùng trainer, test, callback dùng lại)
  sẽ treo training.
- **Quy tắc:** mọi callback wrap method của validator phải có guard
  `if self._original_update_metrics is not None: return` (đã áp dụng cho
  `SizeStratifiedValCallback` và `WandbMediaCallback`; `SystemMetricsCallback` đã có guard tương
  tự qua `_system_metrics_grad_hook`).

## L-007: Worker của `InterpreterPoolExecutor` phải là hàm module-level và tự import dependency

- **Ngày:** 2026-11-09 (khi đánh giá Python 3.14 features).
- **Bối cảnh:** viết benchmark cho `InterpreterPoolExecutor` (3.14). Ba lỗi gặp phải lần lượt:
  (1) worker là closure bên trong hàm → `NotShareableError: <function> does not support
  cross-interpreter data`; (2) worker định nghĩa qua `python -c` cũng fail tương tự; (3) worker
  module-level nhưng dùng `json` import ở đầu file `__main__` → `NameError: name 'json' is not
  defined` trong subinterpreter.
- **Nguyên nhân:** subinterpreter khởi động với `__main__` rỗng — không kế thừa import của
  tiến trình cha; hàm được truyền bằng tham chiếu `(module, qualname)` nên phải nằm ở
  module-level của một module import được.
- **Quy tắc:** worker của `InterpreterPoolExecutor` phải (a) là hàm module-level trong một
  module import được (không phải `__main__`, không phải closure/lambda), (b) tự
  `import` mọi dependency bên trong thân hàm hoặc ở module-level của module đó, (c) nhận/trả
  chỉ các kiểu shareable (str/bytes/int/float/tuple của các kiểu này) — dict/list lớn sẽ bị
  pickle qua lại và ăn mất toàn bộ lợi thế song song (đo thấy 2.76x → 0.99x khi trả full dict).
- **Bổ sung 2026-11-09 (cơ chế chính xác):** hàm worker được chia sẻ **by value** (code object),
  KHÔNG re-import module. Hệ quả: **mọi tham chiếu tới tên module-level — kể cả một hằng số
  `int` — đều gây `NotShareableError: args not shareable`**. Đã kiểm chứng bằng 3 biến thể:
  hàm chỉ dùng tham số → OK; hàm đọc `GLOBAL_NP`/`CHUNK_LINES` → fail; hàm `import numpy`
  trong thân → `ImportError` (numpy không load được trong subinterpreter). Vì vậy worker phải
  **hoàn toàn tự thân**: chỉ biến local + import stdlib bên trong thân hàm. Xem
  `src/tinydet/error_analysis/baseline_summary.py` làm mẫu.
- **Bổ sung 2026-11-09 (chia chunk):** với file JSONL có dòng rất lớn (`samples.jsonl`:
  786 dòng × ~59 KB), chia theo **số dòng** sẽ tạo 1 chunk duy nhất và mất hoàn toàn song song
  (đo được chỉ 1.5x thay vì 3.9x). Phải chia theo **byte** rồi nới tới ranh giới dòng.

## L-008: Patch đường loss phải được smoke-test bằng criterion thật trước khi launch campaign

- **Ngày:** 2026-11-09.
- **Bối cảnh:** patch tiny-floor trong `src/tinydet/models/ultralytics.py` (thêm ở commit
  `72f6fc4`) viết `is_tiny = (sqrt_area < 8.0) & fg_mask.unsqueeze(-1)`. `fg_mask` của
  TaskAlignedAssigner có shape `(b, h*w)` — giống hệt `sqrt_area` — nên unsqueeze tạo
  `(b, h*w, 1)` và broadcast với `(b, h*w)` lỗi
  `size of tensor a (8) must match the size of tensor b (136000)` ngay batch đầu tiên.
  Cả 2 config của campaign 60e chết sau ~20 s GPU, phải launch lại.
- **Nguyên nhân gốc:** patch mới trên đường loss **chưa từng được chạy qua criterion thật**
  trước khi launch. Test forward module (`gauss self-test`) không đi qua assigner.
- **Quy tắc:** trước khi launch campaign, chạy smoke test CPU:
  build model từ YAML thật → `model.init_criterion()` → xác nhận patch installed →
  `model.model.loss(batch)` với batch tổng hợp (b=2, imgsz nhỏ, có box tiny + lớn) →
  assert loss finite. Lưu ý: gọi `DetectionModel.loss` (qua `wrapper.model`), KHÔNG gọi
  `YOLO.loss` trên wrapper — method đó khởi động cả trainer (100 epoch trên coco8 mặc định).
  Test mẫu: `/tmp/deepbench/test_model_loss_cpu.py` (cần chuyển thành test chính thức nếu
  muốn giữ lâu dài).
- **Sửa:** `make_tiny_floor_forward` giờ broadcast `fg_mask` trực tiếp với `sqrt_area`
  rồi mới unsqueeze `is_tiny` lên chiều class; thêm `**kwargs` passthrough để wrapper sống
  sót khi assigner tương lai nhận thêm tham số (ví dụ `anc_strides`).


## L-009: Nâng RAM ceiling khi kéo dài schedule — SIGKILL 137 xuất hiện muộn, không phải ở epoch 1

- **Bối cảnh:** campaign 150e (app `ap-sTJpvV9GSDzzFYKOEq0PAW`, 2026-11-09) — cả 2 run
  baseline + gauss đều bị `Runner killed (SIGKILL), exit code: 137` ở ~epoch 131 với
  `MEMORY_LIMIT_MB = (12288, 24576)`. 60e batch 8 chạy cùng limit này không vấn đề.
- **Bài học:** RAM của training container **tăng dần theo số epoch** (per-epoch val metrics
  `val_predictions_latest.json`/`size_stratified.csv`, W&B media callback), nên smoke test
  vài epoch đầu không phát hiện được. Lỗi RAM OOM trên Modal là SIGKILL từ host, không có
  traceback Python — phải grep log cho "SIGKILL/exit code 137", và kiểm tra run dir thiếu
  `completed.flag` dù `system_metrics.csv` ghi đủ epochs.
- **Sửa:** nâng `MEMORY_LIMIT_MB` → `(24576, 49152)` trong `modal/app.py` (ADR-020).
  Trend RAM cũ: chết ở ~130 epoch với 24 GB ⇒ ~185 MB/epoch tăng trưởng; 48 GB đủ cho 150e.
- **Theo dõi:** nếu campaign sau (300e+) vẫn chết RAM ⇒ điều tra leak thật sự (BACKLOG BL-028).


## L-010: Augmentation có thể thay thế chức năng của module — bật augment cho cả hai model làm mờ lợi thế kiến trúc

- **Bối cảnh:** campaign 150e (ADR-019) bật `augmentations_preset: tinyperson_b` cho cả
  baseline và GauSS. Gap AP@0.50 tiled (GauSS − baseline) thu hẹp từ +7.3% (60e, không
  augment) xuống +1.1% (150e); bucket tiny từ +12.6% xuống −0.4%.
- **Bài học:** khi so sánh một module với baseline, **không được bật augmentation trùng
  chức năng với module cho cả hai nhánh** — augmentation có thể hấp thụ trọn vẹn lợi thế
  kiến trúc, làm kết quả so sánh vô nghĩa (hoặc tệ: kết luận sai rằng module vô dụng).
  Trước khi thiết kế so sánh, liệt kê chức năng của module (ở đây: scale invariance) và
  kiểm tra augmentation/recipe nào cung cấp cùng chức năng đó.
- **Cách tránh:** (a) chạy ablation có/không augment cho cả hai nhánh (ma trận 2×2) trước
  khi kết luận; (b) nếu augment là một phần recipe chính thức, báo cáo cả hai cột và nêu
  rõ tương tác; (c) phân tích đường cong hội tụ — module có thể thắng ở "hội tụ nhanh"
  ngay cả khi hoà ở "trần chất lượng" (đúng trường hợp này: GauSS đạt 90% fitness ở e61
  vs baseline e76).
- **Trạng thái:** giả thuyết chưa được chứng minh — cần BL-030. Đừng viết vào báo cáo
  "GauSS vô dụng với augment" trước khi có thí nghiệm kiểm soát.


## L-011: Chênh lệch AP 1 seed phải qua paired bootstrap trước khi báo cáo — CI chứa 0 nghĩa là "không khác biệt"

- **Bối cảnh:** campaign 150e cho gap AP@0.50 tiled +0.0046 (GauSS − baseline), tưởng là
  kết quả dương. Paired bootstrap (B=120, resample 786 ảnh test, tái dùng
  `evaluate_tinyperson` không đổi) cho 95% CI **[−0.0031, +0.0145]**, P(diff≤0)=0.175;
  bucket tiny: diff −0.0024, CI [−0.0132, +0.0104], P=0.767. CI chứa 0 ở cả hai nhóm.
- **Bài học:** gap nhỏ hơn ~2× độ lệch chuẩn bootstrap (±0.013–0.014) là nằm trong nhiễu
  sampling; val và test còn nói ngược chiều (baseline tốt hơn trên val). Một seed + một
  lần eval **không bao giờ** đủ để claim "module cải thiện AP".
- **Cách tránh:** (a) trước khi claim, chạy paired bootstrap trên ảnh test (resample có
  hoàn lại, giữ cặp base/gauss cùng ảnh để giữ tương quan); (b) quy ước ngưỡng: chỉ claim
  khi CI không chứa 0; (c) multi-seed (3–5) cho error bar giữa các lần train, vì bootstrap
  chỉ đo nhiễu sampling của eval, không đo nhiễu khởi tạo/train.
- **Trạng thái:** đã áp dụng cho 150e — kết luận ghi ở ADR-021; script bootstrap ở
  `/tmp/deepbench/bootstrap_ap.py` (chưa promote vào repo, xem BL-030).
