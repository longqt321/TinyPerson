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
