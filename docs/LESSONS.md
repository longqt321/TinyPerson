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


## L-012: "Identity at init" và "gradient khác 0 cho tham số cơ chế" loại trừ nhau — warm-start cổng

- **Bối cảnh:** GauSS v3.1 dùng ReZero-exact (`alpha = 0.5·tanh(raw)`, raw init 0) để
  `y_p2 == x0` bitwise ở epoch 0. Hệ quả không được xét: `y = x0 + alpha·z(σ) + gamma·b(σ)`
  với `x0` độc lập σ ⇒ `alpha = gamma = 0` thì `∂L/∂σ ≡ 0` **chính xác**, và output **hoàn
  toàn không phụ thuộc σ** (đo: perturb σ 0.65→0.77 cho Δoutput = 0.000e+00). Sau 150
  epoch thật, σ chỉ trôi 3.5–5% dải; router correction cũng chết gradient vì `∂L/∂a = 0`
  tại init; `high_band_share` đứng yên.
- **Bài học:** mọi "residual module khởi tạo bằng 0 để giữ baseline" đều có **dead-start**
  cho các tham số chỉ tác động qua nhánh residual. Trước khi chọn identity-at-init, phải
  chứng minh bằng test rằng `∂L/∂θ > 0` tại init cho **từng** tham số đặc trưng của module.
  Không có test đó thì "module không cải thiện" là kết luận vô nghĩa — module chưa từng
  được phép hoạt động.
- **Cách tránh:** (a) warm-start cổng ở giá trị nhỏ nhưng khác 0 (ở đây alpha=0.10,
  gamma=0.05 ⇒ residual norm 17% của x0) và ghi rõ đánh đổi; (b) thêm test hồi quy
  `assert max|∂L/∂σ| > 0` + `assert output sensitivity > 0` vào self-test của module;
  (c) với tham số có gradient bậc nhỏ hơn conv 1 bậc, tách param group với `lr_mult`
  (ở đây 3.0) thay vì tăng LR toàn cục.
- **Trạng thái:** đã sửa trong v3.2 (ADR-022); self-test PASS với `max|∂L/∂σ|@init =
  5.01e-05`, `σ drift/200 steps = 1.074`. Chưa train lại trên GPU.


## L-013: Đo "độ nhạy" của module phải cố định BN và dùng đúng mode của loss

- **Bối cảnh:** khi probe σ của GauSS bằng loss YOLO thật, tôi gặp hai cái bẫy liên tiếp:
  (1) forward ở `train()` + `no_grad()` vẫn **cập nhật BN running stats** ⇒ trạng thái model
  đổi giữa các điều kiện đo; (2) sửa bằng `eval()` thì **sai nặng hơn**: eval mode decode
  box (output 895k vs 853k phần tử) và độ nhạy với σ tụt từ 0.13 xuống 7e-5 (~2000×), vì
  loss được định nghĩa trên prediction thô của train mode.
- **Bài học:** với model có BN + head decode, muốn đo độ nhạy tham số bằng loss thì phải
  **train mode + snapshot/restore BN buffers** quanh mỗi lần đo. Không dùng eval mode để
  "cho sạch", và không tin rằng no_grad là đủ để đo là không đổi trạng thái.
- **Cách tránh:** viết một hàm `measure_losses()` duy nhất (snapshot buffers → đo → restore)
  và dùng cho mọi điều kiện; kiểm tra chéo bằng một can thiệp "chắc chắn phải đổi" (ở đây:
  gates→0) — nếu can thiệp đó cho delta ≈ 0 thì phép đo hỏng, không phải module bất động.
- **Trạng thái:** đã áp dụng trong `scripts/gauss_probe.py`; số liệu T3 trùng khớp với bản
  đo cũ nên kết luận không đổi.

## L-014: Retry mặc định của cloud = train lại từ đầu nếu run_id không tất định

- **Bối cảnh:** `_train` trên Modal bị hard-kill; Modal tự retry; vì `run_id` sinh từ
  timestamp, mỗi retry tạo thư mục mới ⇒ `last.pt` cũ không được dùng ⇒ train lại 150e.
- **Bài học:** với job dài và tốn tiền, **idempotency là yêu cầu về chi phí, không phải
  chuyện code đẹp**. Mọi thứ có thể retry phải có khoá tất định (commit+seed), phải resume
  được từ checkpoint, và phải trả về ngay nếu việc đã xong.
- **Cách tránh:** trước khi launch, tự hỏi "nếu hàm này chạy lại 3 lần thì tốn bao nhiêu?".
  Nếu câu trả lời là "3 lần full training" thì chưa được launch.

## L-015: Trả lời dài, nhiều bảng biểu khi người dùng đang mệt = không trả lời

- **Bối cảnh:** trong phiên nghiên cứu tiny object detection, tôi liên tục trả lời bằng
  báo cáo nhiều tầng (bảng số, thuật ngữ, chia mục Critical/Major). Người dùng phản hồi
  ba lần liên tiếp: "trình bày khó hiểu quá", "giải thích đơn giản hơn, tôi rất mệt rồi",
  và cuối cùng phải tự chốt "chọn cách 3" vì tôi không đưa ra được một khuyến nghị dứt
  khoát. Nội dung kỹ thuật không sai, nhưng **thông tin không đến được người đọc**.
- **Bài học:** độ chi tiết phải theo trạng thái người đọc, không theo độ phức tạp của
  công việc. Khi người dùng nói mệt/chưa hiểu: **một câu trả lời, một khuyến nghị, một
  lý do** — chi tiết để trong file, chỉ ra file nếu họ muốn đọc thêm.
- **Cách tránh:** trước khi gửi, tự hỏi "nếu chỉ được giữ 3 câu thì giữ câu nào?"; nếu
  câu trả lời cần nhiều hơn 3 câu để hiểu thì viết lại. Không mở đầu bằng bảng số liệu;
  không dùng thuật ngữ chưa định nghĩa trong cùng câu.
- **Trạng thái:** áp dụng từ cuối phiên (các câu trả lời sau đó đã rút xuống 5–10 dòng).

## L-016: Trích dẫn arXiv ID sai — sai một chữ số là bịa tài liệu

- **Bối cảnh:** trong `CRITERIA_PAIRSPLIT_PREREG.md`, `TINY_RESEARCH_LOG.md` (R23) và
  `TINY_RESEARCH_FINAL.md` §14, tôi trích CrowdDet là **arXiv:1906.04423**. Khi người dùng
  yêu cầu kiểm tra lại prior art, tra cứu trực tiếp mới thấy ID đúng là **arXiv:2003.09163**
  (Chu et al., CVPR 2020). ID cũ không tồn tại cho paper đó ⇒ về mặt học thuật, đó là
  **trích dẫn bịa** dù ý định không phải vậy.
- **Bài học:** arXiv ID / số trang / tên tác giả **không được viết từ trí nhớ**. Một ID sai
  làm mất toàn bộ độ tin cậy của báo cáo, kể cả những chỗ khác đúng. Quy tắc: mỗi citation
  phải được xác nhận bằng cách tra **title ↔ ID** trong cùng một lần tra cứu, rồi mới ghi.
- **Cách tránh:** khi viết tài liệu có trích dẫn, tạo bảng `title | authors | venue | arXiv ID`
  và tra từng dòng; nếu chưa tra được thì ghi `[CHƯA XÁC MINH]` thay vì điền ID suy đoán.
  Với PairSplit, bảng đó nằm ở `docs/PAIRSPLIT_PRIOR_ART.md` §2.
- **Trạng thái:** đã sửa cả 3 file; thêm `docs/PAIRSPLIT_PRIOR_ART.md` làm nguồn chuẩn cho
  phần prior art.

## L-017: Hook P2 gắn sai object — train "thành công" nhưng 0 step

- **Bối cảnh:** `scripts/train_gauss_split.py` gắn forward hook lấy feature P2 vào
  `net.model[19]` của model vừa build. Trên CPU, một lượt train 2 ảnh in ra
  `epoch 0 done, 0 steps total` nhưng vẫn `saved head` và exit code 0.
- **Nguyên nhân gốc:** `ultralytics.nn.autobackend.AutoBackend.__init__` có
  `if isinstance(model, nn.Module) and any(x.is_inference() ...): model = deepcopy(model)`.
  Vì ta gọi `.eval()` trước khi predict, predictor chạy trên một **bản deepcopy**, nên hook
  trên bản gốc không bao giờ fire ⇒ `feat is None` ⇒ mọi step `continue`.
  (Hook dạng lambda trong test thủ công vẫn "fire" vì deepcopy của function chia sẻ closure,
  nên thí nghiệm kiểm tra bằng lambda cho kết quả dương tính giả.)
- **Bài học:** với ultralytics, mọi hook/can thiệp vào module phải gắn lên
  `yolo.predictor.model.model` (object thực thi), sau khi predictor đã được khởi tạo.
  Và bất kỳ vòng train nào cũng phải có chốt `if step == 0: raise` — "0 step" không được
  coi là thành công.
- **Cách tránh:** (1) `P2Capture` nhận `yolo`, tự warmup một predict rồi resolve object thực
  thi; (2) guard `step == 0` trong `train()`; (3) khi smoke test, kiểm tra **số step > 0** và
  `pos_rate` khác 0, không chỉ nhìn exit code.
- **Trạng thái:** đã sửa; smoke trên Modal A10 chạy 5/5 step, `pos_rate` 0.67–0.91.

## L-018: Trộn số AP của hai checkpoint khác nhau trong cùng bảng

- **Bối cảnh:** bảng R6.a trong log nghiên cứu ghi "full-frame 0.1886 / tiled 0.3540" như
  một cặp so sánh. Khi tái lập pipeline ngày 2026-11-11, head-off tiled trên checkpoint
  150e cho 0.37556 — lệch 2.2pp so với 0.3540. Truy ngược: 0.1886 là AP của **model GauSS**,
  0.3540 là AP của **baseline checkpoint cũ trước 150e**; còn checkpoint 150e đúng là
  0.3002 (full) / 0.3757 (tiled) — đã có sẵn trong `reports/eval/*_150e_*.json`.
- **Nguyên nhân gốc:** lấy số từ các lần đo khác nhau (model khác, checkpoint khác) mà
  không ghi provenance trong chính bảng số; bảng chỉ có nhãn "full-frame/tiled" nên nhìn
  như cùng một model.
- **Bài học:** mỗi số AP trong log/report phải đi kèm (checkpoint, model, protocol) ngay
  trên dòng đó hoặc trong một bảng provenance riêng. Trước khi dùng một số cũ, chạy lại
  hoặc đối chiếu file `reports/eval/*.json` cùng checkpoint.
- **Cách tránh:** quy tắc "không số mồ côi" — số không khai báo nguồn (file + checkpoint)
  không được phép xuất hiện trong bảng kết luận.
- **Trạng thái:** đã sửa runbook §3 (cảnh báo), PAPER_OUTLINE §3, PREREG §3; tái lập
  0.37556 bằng pipeline mới.

## L-019: Viết lại thứ thư viện đã có (NMS, IoU) — vi phạm quy tắc ưu tiên dùng lại

- **Bối cảnh:** khi viết `scripts/train_gauss_split.py` và
  `src/tinydet/evaluation/tiled_inference.py`, tôi tự viết hàm greedy NMS 15 dòng, tự viết
  `box_iou` 12 dòng, và định nghĩa `count_parameters` **hai lần** ở hai module. Người dùng
  phát hiện và phản ứng mạnh: "ĐỪNG CÓ PHẠM LẠI LỖI TƯƠNG TỰ NỮA".
- **Nguyên nhân gốc:** tôi port nguyên văn từ `scripts/export_qualitative_figures_honest.py`
  (file cũ cũng viết tay NMS) để "đảm bảo bit-for-bit", mà không kiểm tra rằng
  `torchvision.ops.nms` — vốn **đã được import sẵn trong chính module đó** — cho kết quả
  y hệt. Tức là tôi ưu tiên "chắc chắn giống bản cũ" hơn quy tắc dùng lại codebase.
- **Bài học:** "giống bản cũ" là lý do hợp lệ để **kiểm chứng**, không phải lý do để
  **copy code**. Quy trình đúng: grep codebase + thư viện đã cài → dùng lại → viết test
  chứng minh tương đương với bản cũ. Không bao giờ làm ngược lại.
- **Cách tránh:** trước khi viết bất kỳ helper nào, bắt buộc chạy 3 lệnh:
  `rg -n "def <tên>" src/ scripts/`, `python -c "import <lib>; dir(<lib>)"`,
  và kiểm tra import sẵn có ở đầu file đang sửa. Ghi kết quả vào comment cạnh chỗ dùng.
- **Đã sửa:** `batched_nms` → `torchvision.ops.nms` (200/200 batch identical);
  `box_iou` → `ultralytics.utils.metrics.box_iou` (sai khác < 1e-6 trên 50 batch);
  `count_parameters` → một định nghĩa duy nhất ở `pair_split`, `gauss_split` re-export.
  39/39 test pass, self-check pass.
- **Trạng thái:** đã sửa; thêm `docs/banned-phrases.json` với 2 cụm từ cấm.

## L-020: Train 19.000 bước nhưng target nằm ngoài cửa sổ receptive của head

- **Bối cảnh:** chạy B0 (PairSplit, patch 3) 19.000 bước. Log: `reg` kẹt 0.3–0.9, loss
  1.13 → 0.82 (mức nhiễu), `cls` ≈ 0.5. Tôi định tăng epoch / đổi lr để "cứu".
- **Nguyên nhân gốc:** head chỉ đọc patch 3×3 ở stride 4 = **±6 px**, trong khi target cho
  phép láng giềng tới `1.5×max(w,h)` (trung vị 14.1 px, tối đa ~30 px). Đo lại: chỉ **14.1%**
  positive có láng giềng nằm trong ±6 px; ±12 px thì 46.0%. Tức ~86% nhãn dương **không thể
  nhìn thấy** bằng bất kỳ trọng số nào. Loss không giảm không phải vì thiếu capacity hay
  thiếu bước, mà vì bài toán không giải được với cửa sổ đó.
- **Bài học:** trước khi train, phải chứng minh **target nằm trong receptive field của
  model** bằng một phép đo trên dữ liệu thật (khoảng cách phân vị của target so với bán kính
  cửa sổ). Đây là kiểm tra bắt buộc, ngang với kiểm tra shape.
- **Cách tránh:** viết một script đo phân bố khoảng cách target (`scratch/measure_pair_targets.py`)
  và in tỉ lệ target nằm trong cửa sổ **trước** khi gửi lệnh train. Nếu < 90%, sửa cửa sổ
  hoặc siết bán kính target trước, đừng train.
- **Đã sửa:** bán kính target có trần `min(1.5×max(w,h), 10 px)`; patch 3 → 7 (±12 px);
  bottleneck 1×1 giữ ngân sách tham số. Đo lại: 6.055/30.798 positive (19.7%) và **mọi**
  target ≤ 10 px đều nằm trong cửa sổ. Kết quả B0 cũ bị coi là không hợp lệ.
- **Trạng thái:** đã sửa code + test (38/38 pass) + prereg §13; AP vẫn chưa có.

## L-021: Chèn layer trước `Detect` làm lệch toàn bộ key checkpoint, load "thành công" nhưng detector vẫn random

- **Bối cảnh:** bản tích hợp GaussSplit đầu tiên là một layer `GaussSplitLayer` đặt **sau**
  `Detect` trong graph YAML. Khi load checkpoint baseline 150e, log in `Transferred 582/910 items`
  và `gs_loss` luôn bằng 0. Tôi suýt coi đó là "head chưa học" thay vì lỗi nạp trọng số.
- **Nguyên nhân gốc:** ultralytics giả định `model.model[-1]` **chính là** head
  (`init_criterion`, `v8DetectionLoss.__init__`, `bias_init`, probe stride). Thêm layer sau
  `Detect` phá các giả định đó; thêm layer **trước** `Detect` đẩy index 29 → 30 và mọi key
  `model.29.*` của baseline thành `model.30.*` → `intersect_dicts` bỏ qua **toàn bộ** detector,
  chỉ còn phần backbone trùng tên. Không có exception nào được ném ra.
- **Bài học:** "load không lỗi" không có nghĩa là "load đúng". Với checkpoint, phải so **số item
  transferred** với **số key kỳ vọng**, và phải có test khẳng định state-dict key của model mới
  là **superset** của baseline (không đổi tên key cũ).
- **Cách tránh:** khi thêm module vào graph YOLO, giữ `model.model[-1]` là **subclass của
  `Detect`** (ở đây `DetectGaussSplit`), để head là các key phụ `model.29.head.*` mà
  `intersect_dicts` tự bỏ qua. Bắt buộc có test `test_state_dict_keys_match_baseline`.
- **Đã sửa:** `DetectGaussSplit` thay cho layer rời; checkpoint load `Transferred 902/910`
  (8 key thiếu = tensor của head mới), round-trip `910/910`, `match: True`.
- **Trạng thái:** đã sửa; 47/47 test pass.

## L-022: NMS lại toàn bộ hộp khi thêm hộp thứ hai làm giảm cả detection gốc

- **Bối cảnh:** viết test cho hook `DetectionValidator.postprocess`. Với ngưỡng
  `p(second)=1.1` (không nhận hộp nào) số hộp phải bằng baseline, nhưng đo được
  **400 hộp khi tắt head và 248 hộp khi bật** — tức bật head làm mất 152 detection gốc.
- **Nguyên nhân gốc:** `add_second_boxes` chạy `nms(merged, 0.5)` trên **toàn bộ** tập
  (hộp gốc + hộp mới). Validator/predictor dùng `iou=0.7`, nên NMS 0.5 chạy lại đã siết
  ngưỡng của chính detector — một thay đổi ngầm không có trong prereg.
- **Bài học:** khi thêm phần tử vào danh sách detection, chỉ được **lọc phần tử mới**;
  phần tử có sẵn phải giữ nguyên bit-for-bit. Luôn có test "identity khi tắt module"
  (`torch.equal`) cho mọi hook inference.
- **Đã sửa:** giữ toàn bộ index gốc, chỉ lọc các index của hộp thứ hai; thêm test
  `test_add_second_boxes_is_identity_below_threshold`. 53/53 pass (54/54 sau khi thêm test zero-detection).
- **Trạng thái:** đã sửa.

## L-023: Cache tensor trên module làm crash deepcopy và phình checkpoint

- **Bối cảnh:** `DetectGaussSplit.forward` cache feature map P2 vào `self.feat` để
  `aux_loss` và `add_second_boxes` dùng lại. Rà soát lại phát hiện 2 hậu quả:
  (1) checkpoint `best.pt` chứa **tensor feature cũ** của batch cuối (13 MB ở imgsz 1280,
  trạng thái rác trong file); (2) nếu `val: false`, feature còn mang `grad_fn` ⇒
  `deepcopy` trong `save_model`/`ModelEMA` **raise RuntimeError** — run chết ở epoch 1.
- **Nguyên nhân gốc:** tensor gắn vào `__dict__` của nn.Module sẽ được pickle/deepcopy
  theo module. Tensor có graph không phải leaf nên deepcopy của torch từ chối.
- **Bài học:** mọi cache tạm trên module phải (a) `detach()` ngay khi lưu, và (b) bị loại
  khỏi serialization bằng `__getstate__`. Kiểm tra bắt buộc: `deepcopy(model)` thành công
  ngay sau một forward có gradient, và checkpoint không chứa tensor cache.
- **Cách tránh:** trước khi thêm attribute không phải parameter/buffer vào một nn.Module
  nằm trong graph ultralytics, tự hỏi "ai deepcopy/pickle module này?" (save_model, EMA,
  AutoBackend, strip_optimizer) và viết test cho từng đường.
- **Đã sửa:** `self.feat = x[0].detach()` + `__getstate__` trả `feat=None`; smoke CPU với
  `val=false` chạy xong, checkpoint có đủ 8 key head và `feat is None`. 54/54 test pass.
- **Trạng thái:** đã sửa.

## L-024: Đưa lệnh `PYTHONPATH=$PWD/src modal run ...` mà không test bằng chính CLI đó

- **Bối cảnh:** người dùng hỏi "sao lệnh chạy này khác trước giờ? Trước giờ chỉ có
  `modal run ...` thôi mà?". Kiểm tra thật: `PYTHONPATH=$PWD/src modal run ...` **fail** với
  `ModuleNotFoundError: yaml`, còn `modal run ...` (venv activate) và
  `.venv/bin/python -m modal run ...` đều chạy được (`--help` in ra entrypoint).
- **Nguyên nhân gốc:** tôi viết lệnh từ giả định "modal CLI dùng Python của project venv".
  Thực tế `~/.local/bin/modal` là uv-tool riêng với Python thiếu `yaml`; `PYTHONPATH` chỉ
  thêm `src/`, không thêm site-packages nên không cứu được. Tôi đã "xác minh" bằng
  `python -c "import app"` với interpreter của project — không phải bằng CLI sẽ chạy.
- **Bài học:** lệnh đưa cho người dùng phải được test **bằng đúng binary + môi trường** mà
  người dùng sẽ gõ. `python -c import` thành công không chứng minh `modal run` thành công.
- **Cách tránh:** trước khi đưa lệnh, chạy chính lệnh đó với `--help` (hoặc `--dry-run` nếu
  có) trong shell sạch. Ghi rõ điều kiện tiền đề (venv activate hay không).
- **Đã sửa:** runbook §6 dùng `.venv/bin/python -m modal run ...` và ghi chú 3 biến thể.
- **Trạng thái:** đã sửa; lệnh đã kiểm chứng bằng `--help` thật.

## L-025: Preset augmentation hardcode `RandomSizedBBoxSafeCrop` 1280x1280 làm crash DataLoader khi imgsz != 1280

- **Bối cảnh:** smoke run (imgsz 320) chết giữa epoch với
  `RuntimeError: stack expects each tensor to be equal size, but got [3, 320, 320] ... [3, 1280, 1280]`
  trong `YOLODataset.collate_fn`.
- **Nguyên nhân gốc:** preset `tinyperson_b` (ADR-018) hardcode
  `RandomSizedBBoxSafeCrop(height=1280, width=1280)`. Transform này resize crop về
  (height, width), nên với ảnh letterbox không-vuông nó đẩy ~20% ảnh trong batch
  lên 1280x1280 trong khi phần còn lại là imgsz. Ở các run 150e imgsz 1280 khớp
  đúng nên không bao giờ lộ; imgsz 320 của smoke run làm nó bùng nổ.
- **Bài học:** mọi transform phụ thuộc kích thước phải được sinh từ `imgsz` của
  run, không hardcode. Lỗi chỉ hiện khi batch chứa ảnh kích thước khác nhau, tức
  là lỗi cấu hình phải được bắt lúc resolve config, không phải lúc train.
- **Cách tránh:** (1) preset là factory nhận `imgsz`; (2) `resolve_augmentations`
  từ chối list transform tường minh có height/width != imgsz; (3) test hồi quy
  chạy đúng pipeline `build_yolo_dataset` ở imgsz 320 và assert mọi ảnh cùng shape.
- **Đã sửa:** `src/tinydet/augment/albumentations_stack.py` (factory +
  validation), `trainer.py` truyền `int(config.train["imgsz"])`; test mới
  `tests/test_augment_presets.py` (8 test). 62/62 pass, ruff sạch.
- **Trạng thái:** đã sửa.

## L-026: Dùng `modal/app.py::train` (orchestration ở máy local) cho run dài → local chết là train chết

- **Bối cảnh:** B0 (40e) được launch 2 lần bằng
  `modal run [--detach] modal/app.py::train --config ...`. Cả hai lần app còn sống
  nhưng `Tasks 0`, container biến mất, **log của `_train` rỗng hoàn toàn** và
  không có thư mục run nào được tạo trên volume. Người dùng báo "train mất rồi,
  wandb cũng không còn".
- **Nguyên nhân gốc:** `modal/app.py::train` là **local entrypoint**: nó gọi
  `_audit_original_dataset.remote()`, `_train.remote()` rồi `_finalize_pipeline()`
  **trong tiến trình local**. Khi tiến trình local bị dừng (harness dọn background
  task), Modal stop app ⇒ container `_train` bị kill ngay cả khi đã spawn.
  `docs/agents/OPERATIONS.md` đã ghi rõ đường chuẩn là
  `modal run -d launch_campaign.py::main`, nơi `_orchestrate_campaign` chạy
  **bên trong Modal** và `_train` được `.spawn()` + `_collect()` handle.
- **Bài học:** trước khi launch run dài, kiểm tra tài liệu vận hành xem entrypoint
  đó chạy orchestration ở đâu (local hay remote). "Có `--detach`" không đủ: detach
  chỉ giữ *app*, không giữ *tiến trình local đang giữ orchestration*.
- **Cách tránh:** run > vài phút ⇒ luôn dùng `launch_campaign.py::main -d`
  (`_orchestrate_campaign` remote). Kiểm chứng ngay sau launch:
  `modal app list` phải thấy `Tasks >= 1` và `modal container list` phải có container;
  nếu `Tasks 0` thì run đã chết, đừng chờ.
- **Đã sửa:** thêm `--no-evaluate` cho `launch_campaign.py::main` /
  `_orchestrate_campaign(..., evaluate=True)` để dừng sau train+finalize (prereg
  yêu cầu test chỉ chấm 1 lần với threshold chọn trên val). B0 chạy lại bằng
  `modal run -d launch_campaign.py::main --configs ... --no-evaluate`.
- **Trạng thái:** đã sửa; app detached có 2 task (orchestrator + `_train`).

## L-027: Callback force BN eval chạy TRƯỚC `_model_train()` của Ultralytics → BN của detect head trôi thống kê

- **Bối cảnh:** B1 (tile 640, freeze 29) cho val tiny AP@0.50 = 0.3194 dù tắt hết
  box thứ hai (`head_threshold=1.0`), thua baseline 0.3559 — tức box thứ nhất
  của detector "frozen" đã bị hỏng. So state-dict baseline vs B1 trên volume
  (`scratch/modal_compare_ckpt.py`): 902/902 key load đủ, 0 lệch shape, nhưng
  **96 tensor khác nhau** — toàn bộ `running_mean/running_var` của BN trong
  `model.29` (detect head).
- **Nguyên nhân gốc:** `GaussSplitCallback.on_train_epoch_start` force BN eval,
  nhưng Ultralytics gọi `self._model_train()` **sau** callback
  (`ultralytics/engine/trainer.py` ~dòng 502-507), đặt lại toàn bộ model về
  train mode. `_model_train` chỉ giữ BN eval cho các layer trong
  `freeze_layer_names` = `model.0.`…`model.<freeze-1>.`, không gồm layer 29.
  Freeze bằng `requires_grad=False` không chặn BN cập nhật running stats, nên
  20 epoch train trên tile 640 làm thống kê BN của head trôi khỏi phân bố
  full-frame 1280 của baseline checkpoint → box thứ nhất mất ~3.6pp AP.
- **Bài học:** "frozen" nghĩa là *bit-identical*, không chỉ `requires_grad`.
  Callback của thư viện bên ngoài có thể chạy sai thứ tự so với hook của mình;
  phải kiểm chứng bằng so state-dict thực tế, không chỉ tin log "frozen: N".
- **Cách tránh:** khi freeze detector, force BN eval ở hook chắc chắn chạy
  **sau** `_model_train` và **trước** forward — `on_train_batch_start`.
- **Đã sửa:** thêm `GaussSplitCallback.on_train_batch_start` re-apply
  `_freeze_bn` mỗi batch; thêm test
  `test_bn_eval_survives_ultralytics_model_train` mô phỏng đúng thứ tự
  Ultralytics và assert running_mean không đổi sau forward. 63/63 pass.
- **Trạng thái:** đã sửa; B1 kết quả cũ vô hiệu (BN drift), relaunch thành
  `gausssplit_b1b_tiled` (app `ap-RD4N5y6vl21zYIaUSqcTSW`).

## L-028: Metadata trích dẫn bị bịa khi viết bib nhanh

- **Bối cảnh:** audit `docs/report/07_audit_report.md` phát hiện nhiều entry
  `references.bib` có metadata sai/bịa: tác giả AI-TOD bị chế ("Yang, Guo…"
  sai thứ tự), venue NWD ghi ICPR 2022 (thực tế arXiv 2021), title VisDrone
  sai hoàn toàn, journal/pages của Pintea sai, và 2 reference hoàn toàn không
  tồn tại (Yu/Chen/Wu "Tiny Person Detection in the Wild", BPD-YOLO tiêu đề cũ).
- **Nguyên nhân gốc:** viết bib từ trí nhớ khi đang gấp deadline, không verify
  từng entry với nguồn gốc (trang hội nghị, arXiv, DOI).
- **Bài học:** metadata citation không được "suy đoán hợp lý" — sai một field
  là citation giả. Trước khi compile, mọi entry mới phải đối chiếu nguồn gốc.
- **Cách tránh:** verify bằng web search từng entry (tác giả + năm + venue +
  trang/DOI); entry nào không verify được thì bỏ, không giữ "tạm".
- **Trạng thái:** đã sửa toàn bộ 12 entry lỗi; paper compile sạch 12 trang.

## L-029: Cấu hình train ghi trong báo cáo lệch với YAML thật

- **Bối cảnh:** cả hai bản báo cáo ghi batch 8, warmup 5 epochs; YAML thật
  (`configs/experiment/*_150e.yaml`) là batch 16, warmup 3.0.
- **Nguyên nhân gốc:** viết phần experimental setup từ bản nháp cũ, không mở
  lại config file khi viết số.
- **Bài học:** mọi hyperparameter trong báo cáo phải đọc lại từ config file
  đúng version đã chạy, không từ trí nhớ/nháp.
- **Cách tránh:** khi viết experimental setup, mở YAML song song và copy trực tiếp.
- **Trạng thái:** đã sửa cả hai bản (batch 16, warmup 3.0, epochs 150, seed 42).

## L-030: Phần Method mô tả kiến trúc của một revision khác với model đã train

- **Bối cảnh:** audit hình (`docs/report/08_figure_audit.md`) và đối chiếu code phát
  hiện paper mô tả kiến trúc v3 (upsample `Interp`, dilation {1,2,3}, gate
  `0.5·Sigmoid`, router 4 kênh học được, output `SiLU(X0 + αF̃)`, 34.666 tham số),
  trong khi checkpoint đã train (`a20f7b6`) dùng DySample, dilation {1,3,5}, gate
  `0.5·tanh`, hybrid energy router, output tuyến tính có hai mỏ neo, 36.878 tham số.
- **Nguyên nhân gốc:** viết Method từ bản nháp v3 và không đối chiếu lại với code
  tại đúng commit đã sinh ra checkpoint; số tham số lấy từ `profiling_results.json`
  (build v2.1) thay vì đếm từ checkpoint.
- **Bài học:** "kết quả dương" chỉ có nghĩa nếu Method mô tả đúng thứ đã chạy.
  Một báo cáo có Method lệch kiến trúc là không tái lập được, dù số liệu thật.
- **Cách tránh:** trước khi viết Method, `git show <commit-của-run>:src/...` và
  instantiate module để đếm tham số; mọi phương trình lấy từ `forward()` thật.
- **Trạng thái:** đã sửa cả paper và báo cáo VI; thêm mục "Optional (disabled)"
  vào bảng tham số để tổng khớp đúng 36.878.

## L-031: Hình vẽ thiết kế ở 7–10 inch nhưng chèn vào cột 3.28 inch

- **Bối cảnh:** đo effective font size của 8 hình: tất cả dưới 6 pt, có hình 2.6 pt
  (tiêu đề 9 pt × scale 0.32). Nguyên nhân: `figsize` 7–10 in trong khi cột WACV
  chỉ 236 pt.
- **Bài học:** hình phải được thiết kế đúng bề rộng sẽ in (3.3 in cho 1 cột,
  6.9 in cho `figure*`), không phải vẽ to rồi thu nhỏ.
- **Cách tránh:** đặt `figsize` theo bề rộng đích, kiểm tra `pdfinfo` sau khi xuất
  và tính scale = width_in / native_width; mọi text phải ≥5 pt sau scale.
- **Trạng thái:** đã vẽ lại 2 sơ đồ kiến trúc bằng TikZ, regenerate hình 19/04/07
  ở đúng cỡ cột, bỏ 4 hình không đọc được (13/14/15/08) và đưa vào `_archive/`.

## L-032: Hình vẽ bị chồng chữ và mũi tên dính chùm vì không kiểm tra hình học

- **Bối cảnh:** người dùng mở PDF và chỉ ra mọi hình đều "chữ dính chùm",
  sơ đồ TikZ "mũi tên dính chùm vào nhau". Nguyên nhân cụ thể: (a) nhãn trục x
  của hình 19 là 7 nhóm × 2 dòng ở 3.05 in nên chồng nhau; (b) nhãn giá trị
  hình 04 đè lên nhau vì cột quá hẹp; (c) sơ đồ module có 2 mũi tên cắt qua
  box (gauss→det cắt ds, p2→det cắt p3p) và 2 corridor chạy xuyên box.
- **Nguyên nhân gốc:** chỉ kiểm tra "compile được" mà không đo bounding box
  của node và không kiểm tra từng đoạn mũi tên có cắt box nào không; nhãn
  trục được thiết kế cho bề rộng lớn hơn bề rộng in.
- **Bài học:** TikZ/Matplotlib không báo lỗi khi chồng chữ hay mũi tên cắt
  node — phải kiểm tra bằng số: dump anchor của từng node, dựng polyline của
  từng connector, kiểm tra giao với mọi box; nhãn trục phải vừa trong khoảng
  cách giữa các nhóm.
- **Cách tránh:** (1) thiết kế hình đúng bề rộng in (3.28 in / 6.875 in);
  (2) nhãn nhóm quá dày thì rút gọn 1 dòng thay vì xoay; (3) với sơ đồ, chạy
  script kiểm tra hình học (dump anchor + segment-box intersection) trước khi
  chốt; (4) corridor dọc phải nằm ngoài x-range của box hai bên.
- **Trạng thái:** đã sửa — sơ đồ module vẽ lại theo lane, kiểm tra hình học
  0 giao; sơ đồ tổng thể sửa 2 mũi tên cắt box; hình 19/04/07 regenerate với
  nhãn 1 dòng và font ≥6.2 pt.

## L-033: Sửa Method theo code HEAD thay vì code tại commit đã train

- **Bối cảnh:** khi sửa Method cho khớp model đã train (L-030), tôi đối chiếu
  với `src/tinydet/modules/gauss.py` ở HEAD (v3.2) và đổi phương trình sigma
  sang dạng tanh-centered. Người dùng hỏi "đây là v3.1 chứ?" — kiểm tra lại
  code tại commit chạy thật (`a20f7b6`, trước v3.2) cho thấy v3.1 dùng
  **sigmoid-scaled** (`σ = σ_min + (σ_max−σ_min)·Sigmoid(σ̂)`), và công thức
  sigmoid gốc trong paper mới là đúng.
- **Nguyên nhân gốc:** sau khi phát hiện "HEAD ≠ model đã train", tôi vẫn lấy
  HEAD làm nguồn sự thật cho những phần chưa kiểm tra từng dòng; thực ra phải
  pin đúng commit của checkpoint cho TẤT CẢ chi tiết kiến trúc.
- **Bài học:** version của model trong báo cáo = version code tại commit sinh
  checkpoint, không phải version hiện tại của repo. Các thành phần đổi giữa
  hai version (tanh↔sigmoid sigma, gamma_mid) phải đối chiếu từng dòng với
  file tại commit đó, không suy diễn từ changelog.
- **Cách tránh:** extract code tại commit run (`git show <commit>:path`) thành
  file tham chiếu, mọi phương trình trong Method đối chiếu với file đó; ghi
  rõ version (v3.1) trong experimental setup.
- **Trạng thái:** đã revert phương trình sigma về dạng sigmoid ở cả paper và
  báo cáo VI; phần gate `0.5·tanh` giữ nguyên (v3.1 đúng dùng tanh cho gate).

## L-034 — Paper dài và thừa nội dung do không có budget từng phần

- **Hiện tượng:** người dùng phản hồi paper "còn nhiều phần đang bị thừa": cấu
  hình train nằm trong Abstract, công thức SiLU được viết lại dù chỉ cần trích
  dẫn, bảng notation chiếm 1/3 trang, mỗi phương trình lại có một câu diễn giải
  và cùng một con số lặp ở Abstract + Intro + Results + Discussion + Conclusion.
  Bản trước dài 12 trang (nội dung ~8.3 trang), vượt giới hạn 8 trang của WACV.
- **Nguyên nhân gốc:** viết theo kiểu "mỗi bước đều giải thích" mà không đặt
  budget từ khóa cho từng section, không phân biệt nội dung nào là tutorial
  (nên trích dẫn) và nội dung nào là đóng góp (mới cần phương trình).
- **Cách phòng tránh:** trước khi viết/sửa, đọc `docs/report/09_writing_guide.md`
  và đặt budget: Abstract 150–250 từ, Intro ~900, Related 450–900, Method
  1,600–2,000, Experiments 1,200–1,800, Limitations 200–400, Conclusion
  120–200. Mỗi con số headline xuất hiện tối đa 2–3 lần; bảng là nguồn chuẩn,
  prose chỉ tham chiếu "Table N". Phương trình chỉ giữ khi bản thân công thức
  là đóng góp; công thức phổ thông thì trích dẫn. Sau khi sửa, đếm lại từ và
  kiểm tra trang >8 chỉ chứa references.

## L-035
- **Lỗi:** Báo cáo đồ án bị phản hồi là chưa hoàn thiện: style chữ/bảng thiếu nhất quán, thiếu trích dẫn, thiếu bảng và sơ đồ.
- **Nguyên nhân gốc:** Lượt trước coi báo cáo là tài liệu thuần chữ, tái dùng hình của paper mà không có lớp style riêng cho báo cáo (khung bảng, cỡ chữ, caption theo chương), và không rà soát checklist thành phần bắt buộc (trích dẫn, bảng, hình, danh mục) trước khi báo xong.
- **Phòng tránh:** Với mỗi loại tài liệu (paper vs báo cáo đồ án), lập checklist thành phần và style riêng trước khi viết; sau khi ghép nội dung, chạy kiểm tra tự động (số bảng, số hình, caption, trích dẫn, số trang) trước khi báo hoàn thành.

## L-036
- **Lỗi:** Báo cáo chia mỗi chỉ số đánh giá thành một mục riêng, trình bày dài dòng
  (37 trang cho nội dung có thể gói trong ~20 trang).
- **Nguyên nhân gốc:** Coi "nhiều trang" là mục tiêu thay vì để bảng biểu mang khối
  lượng thông tin; mỗi chỉ số tách mục làm nhân bản cấu trúc thay vì nội dung.
- **Phòng tránh:** Gộp các chỉ số vào một mục "Kết quả", dùng bảng để trình bày số
  liệu; chỉ tạo mục con khi có một ý mới, không phải một số liệu mới.

## L-037
- **Lỗi:** Bản báo cáo sau khi tái cấu trúc vẫn bị đánh giá là "chưa đạt" cả hình thức lẫn nội dung.
- **Nguyên nhân gốc:** Cắt gọn bằng cách bỏ bớt hình/bảng và giữ khối giả mã, siêu tham số; chương
  tổng quan không có bằng chứng định lượng nên chỉ còn văn xuôi.
- **Phòng tránh:** Độ dày của báo cáo phải đến từ hình, bảng và lập luận thiết kế; không bao giờ đưa
  code/giả mã vào báo cáo; mỗi bảng số phải có cột hoặc câu nêu căn cứ.

## L-038
- **Lỗi:** Mục "các hướng tiếp cận đã có" bỏ sót các công trình trọng yếu của bài toán vật cực nhỏ
  (NWD, Dot Distance, RFLA, SET, SEPC), và báo cáo không nói rõ nguồn gốc các hình tự vẽ.
- **Nguyên nhân gốc:** Chỉ liệt kê công trình đã có sẵn trong repo thay vì rà lại toàn bộ tài liệu
  của lĩnh vực; coi hình tự vẽ là hiển nhiên nên không ghi xuất xứ.
- **Phòng tránh:** Với mỗi mục "tổng quan", lập danh sách công trình chuẩn của lĩnh vực (theo khảo sát
  và benchmark gốc) rồi đối chiếu; mọi hình tự vẽ phải có một câu nêu nguồn số liệu và thông lệ
  biểu đồ đang theo.

## L-039
- **Lỗi:** Chèn câu "mọi hình do nhóm vẽ, không sao chép hình từ công trình nào" vào mục 3.1.
- **Nguyên nhân gốc:** Phản ứng thái quá trước câu hỏi của người đọc, biến một câu trả lời riêng thành
  lời thanh minh trong văn bản học thuật.
- **Phòng tránh:** Chỉ đưa vào báo cáo thông tin kỹ thuật cần cho người đọc (ví dụ hình tính từ tham số
  thiết kế hay từ số liệu nào); không viết câu tự bào chữa về tính liêm chính của chính mình.

## L-040
- **Lỗi:** Bản PDF nộp được phát hiện 8 lỗi nặng: "####" sót lại do parser chỉ đọc heading tới
  cấp 3, chương không xuống trang, công thức không đánh số, bảng vỡ ô, bìa bỏ trống,
  danh mục cắt cụt, thiếu danh mục viết tắt, footer sai cỡ.
- **Nguyên nhân gốc:** Không đối chiếu từng dòng của biểu mẫu trường với sản phẩm trước khi nộp;
  chỉ kiểm tra những gì mình nghĩ là quan trọng.
- **Phòng tránh:** Trước khi nộp, chạy checklist đối chiếu biểu mẫu (lề, font, cỡ, đánh số,
  ngắt trang, bìa, danh mục) và quét ký hiệu Markdown sót trong PDF xuất ra.

## L-041
- **Lỗi:** Bản PDF hiển thị nguyên văn mã LaTeX (`\sqrtw · h`, `ĝammain`), công thức toán sai bản chất
  (nửa nhóm của Lindeberg viết thiếu dấu căn), tham chiếu chéo hình sai 2 chỗ, trích dẫn DySample
  trỏ nhầm sang ESPCN, và danh mục hình/bảng lệch toàn bộ số trang (offset sai do trang bìa tràn sang
  trang 2 khiến hàm dò body_start trả về trang 2).
- **Nguyên nhân gốc:** Bộ chuyển LaTeX→Unicode trong builder không xử lý `\sqrt` và `\hat\gamma` (regex
  bắt một ký tự thay vì cả từ); không kiểm tra chéo số tham chiếu hình với nội dung; số tham chiếu
  [20] dùng chung cho hai công trình khác nhau; hàm dò số trang dựa vào từ khóa "DANH MỤC" thay vì
  dấu hiệu ổn định (running head) và không phát hiện bìa tràn trang.
- **Phòng tránh:** Sau mỗi lần sửa builder, build thử một đoạn chứa mọi cấu trúc LaTeX đang dùng và
  quét PDF bằng pdftotext để bắt chuỗi rò rỉ; danh mục hình/bảng phải đối chiếu tự động với trang
  caption thật; thêm tham chiếu mới phải chạy lại script đánh số lại và grep từng lần trích dẫn.

## L-042 — Hình 3.8 so sánh hai đại lượng khác nhau
- **Lỗi:** Điểm "+ GauSS" trong biểu đồ tham số–AP50 dùng AP50 toàn thang (41,08) trong khi mọi điểm công bố và điểm mô hình nền dùng AP50 thang tiny → so sánh vô nghĩa, làm sai kết luận (mô hình đề xuất tưởng hơn các công trình nano).
- **Nguyên nhân gốc:** Lấy số tiện tay từ bảng tổng hợp không kiểm tra thang đo của từng cột; không có assert "mọi điểm trên cùng một trục phải cùng định nghĩa".
- **Phòng tránh:** Trước khi vẽ biểu đồ so sánh, ghi rõ định nghĩa từng đại lượng trong comment của script sinh hình; mọi số đưa vào cùng một trục phải cùng giao thức/thang đo; khi nghi ngờ, đối chiếu lại với JSON nguồn.

## L-043 — Sửa một lỗi so sánh lại tạo ra lỗi so sánh ngược chiều
- **Lỗi:** Khi sửa Hình 3.8, tôi đổi điểm của đồ án sang AP50 thang tiny (37,40/37,57) theo giả định của người nhận xét rằng các số công bố (38,0/39,1) là thang tiny. Kiểm tra nguồn sơ cấp cho thấy Bảng 5 của bài báo chỉ báo mAP50/mAP50-95 toàn tập, không có cột tiny → bản sửa đó lại là so lệch đại lượng.
- **Nguyên nhân gốc:** Tin vào mô tả trong nhận xét thay vì kiểm tra định nghĩa đại lượng trong bài báo gốc trước khi vẽ.
- **Phòng tránh:** Với mọi số lấy từ công trình khác, phải đọc đúng bảng/cột trong bài gốc và ghi lại định nghĩa đại lượng (thang đo, tập đánh giá, giao thức) trước khi đưa vào bảng hoặc biểu đồ; nếu bài báo không báo theo thang thì không được gán nhãn "thang tiny".

## L-034: Rà soát toàn diện văn phong AI và tính nhất quán phương pháp trong báo cáo khoa học

- **Thời điểm:** 2026-11-14
- **Lỗi:** Báo cáo xuất bản có dấu hiệu khuôn mẫu AI (cấu trúc câu 'không phải X mà Y', 'Ưu điểm/Nhược điểm' lặp lại, thiếu chủ ngữ 'chúng tôi', dùng từ thổi phồng/dàn dựng); công thức toán học bị rò rỉ cú pháp raw string (, ); phân tích thang đo bị mâu thuẫn giữa kích thước huấn luyện (1280 letterbox) và kích thước suy luận phân mảnh (640×512 nguyên bản); đánh giá ablation khẳng định quá mức dù tồn tại nhiễu huấn luyện lớn (saliency tắt làm giảm 4,22 điểm); thiếu phân định rõ ràng giữa nhánh topology chạy thử nghiệm (v3.1) và cấu trúc làm sạch cho triển khai (v3.2).
- **Nguyên nhân gốc:**
  1. Quá trình sinh văn bản tự động có xu hướng lặp lại các liên từ khuôn sáo và cấu trúc đối lập nhị phân.
  2. Việc trích xuất công thức từ Markdown sang DOCX qua script regex xử lý thiếu các ký tự thoát đặc biệt của Python string literal (, ).
  3. Quá trình huấn luyện sử dụng ảnh letterbox 1280 nhưng đánh giá chính lại chạy trên các mảnh cắt độ phân giải gốc mà không giải thích rõ ràng sự bất đối xứng về kích thước hình học.
  4. Đợt ablation ban đầu chạy ở 78 epoch và tồn tại biến động lớn giữa các lần huấn luyện, nhưng văn bản báo cáo vẫn dùng ngôn từ khẳng định 'có ý nghĩa thống kê'.
- **Cách khắc phục và phòng tránh:**
  1. Thống nhất chuyển toàn bộ ngôi nhân xưng hành động nghiên cứu sang 'chúng tôi', loại bỏ hoàn toàn các cặp câu khuôn mẫu 'Ưu điểm / Nhược điểm' ở mục 1.2, thay bằng phân tích kỹ thuật độc lập.
  2. Bổ sung trích dẫn đầy đủ cho các thành phần: SiLU [37], ST-MoE z-loss [34], không gian màu HSV [35], cân bằng biểu đồ thích ứng CLAHE [36], hồi quy khung DFL [30].
  3. Bóc tách minh bạch chi phí tham số: topology thử nghiệm v3.1 chứa khối C3k2 trung gian stride 4 không sử dụng (8.704 tham số), trong khi topology triển khai v3.2 loại bỏ khối này chỉ tăng 28.174 tham số hiệu dụng (+1,12%).
  4. Công khai rõ ràng trong mục 3.5 rằng mức giảm 4,22 điểm khi tắt cổng saliency vốn không hoạt động là bằng chứng cho thấy phương sai ngẫu nhiên giữa các lần huấn luyện lên tới hơn 4 điểm AP50; hạ mức khẳng định của ablation xuống mức quan sát xu hướng và yêu cầu huấn luyện đa seed để kiểm chứng.


## L-035: Báo cáo khắc phục thiếu bằng chứng kiểm tra đối chiếu trực tiếp trên sản phẩm xuất bản

- **Thời điểm:** 2026-11-15
- **Lỗi:** Khi thông báo hoàn thành chỉnh sửa các nhận xét của người dùng, chỉ liệt kê các thay đổi mã nguồn/script mà không trích xuất trực tiếp bằng chứng văn bản/hình ảnh từ file PDF và DOCX kết quả để người dùng đối chiếu.
- **Nguyên nhân gốc:** Báo cáo tiến độ mang tính tổng kết chung chung, không chỉ rõ số trang, đoạn văn bản cụ thể và giá trị đã được sửa trong file tài liệu cuối cùng, gây cảm giác chưa thực sự khắc phục hoặc sửa sót.
- **Cách khắc phục và phòng tránh:** Khi báo cáo kết quả khắc phục các mục nhận xét, luôn trích dẫn trực tiếp dòng/trang/hình cụ thể từ tệp PDF/DOCX xuất bản để người dùng có thể kiểm chứng ngay lập tức.

## L-044: Sửa báo cáo theo cẩm nang phải kiểm tra cả tầng sinh tài liệu, không chỉ tầng văn bản

- **Thời điểm:** 2026-11-15
- **Lỗi:** Sau khi rà báo cáo theo `docs/academic_report_handbook.md`, các lỗi về dấu câu và cấu trúc chỉ được sửa trong file `.md`, trong khi script sinh DOCX vẫn neo cứng ký tự em dash trong regex caption (`^\\*\\*Hình (\\d+\\.\\d+) — `). Nếu chỉ sửa `.md` thì caption mất nhận diện, hình và bảng mất tiêu đề trong bản in.
- **Nguyên nhân gốc:** Coi file Markdown là nguồn duy nhất, bỏ qua việc script build có bộ phân tích cú pháp riêng (regex caption, chuỗi tiêu đề, footer) cũng phải sửa đồng bộ.
- **Cách khắc phục và phòng tránh:** Khi thay đổi quy ước trình bày (dấu phân cách caption, ký tự đặc biệt, tiêu đề), phải rà toàn bộ chuỗi phụ thuộc trong script sinh tài liệu (`grep` theo ký tự đó), rồi build lại và kiểm tra bằng `pdftotext` trên chính file PDF xuất bản.

## L-045: Câu ngắn để tăng nhịp điệu không được là khẩu hiệu đối lập

- **Thời điểm:** 2026-11-15
- **Lỗi:** Khi sửa báo cáo theo cẩm nang (mục burstiness), tôi chèn các câu ngắn kiểu "Ngân sách tính toán, không phải độ sâu mạng, là nút thắt.", "Recall lên, precision xuống.", "Tham số rẻ, bộ nhớ thì không.", "Hai nhánh là điều kiện cần, chưa phải điều kiện đủ." Người dùng nhận ra ngay đây là văn AI.
- **Hậu quả:** Các câu này vừa mang giọng khẩu hiệu, vừa lặp ý câu liền trước, và câu "không phải độ sâu mạng" còn mâu thuẫn với lập luận của Chương 2 (vấn đề nằm ở độ phân giải/stride, không phải độ sâu mạng).
- **Nguyên nhân gốc:** Hiểu burstiness như việc "chèn câu ngắn" thay vì đa dạng hóa cấu trúc câu có nội dung; không kiểm tra câu mới có trùng ý câu trước hay mâu thuẫn với lập luận đã nêu hay không.
- **Cách khắc phục và phòng tránh:** Câu ngắn phải có chủ thể, vị ngữ và thông tin cụ thể (số liệu, đối tượng, điều kiện). Cấm câu khẩu hiệu đối lập dạng "X, không phải Y" đứng độc lập; cấm câu khẳng định không kèm bằng chứng. Trước khi thêm câu ngắn, đọc lại câu liền trước để chắc không lặp ý, và đối chiếu với lập luận của chương để chắc không mâu thuẫn.

## L-046: Overclaim và mâu thuẫn cấu trúc là lỗi reviewer bắt ngay, không phải lỗi văn phong bề mặt
- **Thời điểm:** 2026-11-18
- **Lỗi:** Bản paper `paper/main.tex` sạch về dấu hiệu AI bề mặt (không em dash, không từ khóa AI), nhưng vẫn chứa các lỗi nghiêm trọng hơn mà reviewer CVPR sẽ bắt ngay: (1) mâu thuẫn saliency gate — mục 3.3/3.4 ghi gate bị tắt trong cấu hình báo cáo nhưng kết luận và mục 5.1 vẫn kết luận "the clutter gate carry the maritime targets"; (2) overclaim "keeps real-time throughput" trong khi chế độ tiled (chế độ chính xác nhất) chỉ đạt 2.8–7.3 FPS end-to-end; (3) câu chuyện "gain tập trung ở 8–12 px" không khớp số liệu — bin tăng mạnh nhất là reasonable (+3.45) chứ không phải tiny2 (+0.75), tiny1 thì flat; (4) tham chiếu chéo sai Section 3.3 thay vì 3.2; (5) đóng góp trong Intro chọn lọc, bỏ qua tiny3 giảm 1.09; (6) suy diễn chưa kiểm chứng trình bày như kết luận ("points at memory bandwidth", "direct optimization target"); (7) dấu vết phiên bản nội bộ ("revision v3.1", "legacy", "no later variant").
- **Nguyên nhân gốc:** Tập trung sửa văn phong AI slop (em dash, từ khóa) mà quên kiểm tra tính nhất quán logic giữa các phần và tính trung thực của claim so với số liệu. Việc "sạch AI" chỉ là điều kiện cần, không phải điều kiện đủ để paper đạt chuẩn.
- **Cách khắc phục và phòng tránh:** Sau khi sửa văn phong, phải chạy một lượt kiểm tra riêng cho: (a) mọi claim trong abstract/conclusion có khớp số liệu bảng không; (b) mọi thành phần được nêu trong kết luận có thực sự nằm trong cấu hình báo cáo không; (c) mọi tham chiếu chéo Section có đúng mục không; (d) mọi suy diễn chưa có profiling có được hedge thành giả thuyết không; (e) mọi dấu vết phiên bản nội bộ có bị đưa vào bài không. Đánh giá bên ngoài bắt được phần lớn các lỗi này — nên luôn đối chiếu đánh giá ngoài với source trước khi sửa, nhưng không răm rắp: có 2 điểm đánh giá ngoài hơi quá (bold run-in header thực ra là chuẩn CVPR; "anchor" đa nghĩa chỉ là vấn đề nhỏ), cần phản biện nhẹ thay vì sửa máy móc.
