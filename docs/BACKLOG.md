# Backlog

Việc phát hiện thêm trong lúc làm task, chưa nằm trong phạm vi đã thống nhất.

## BL-001: Mâu thuẫn chính sách GPU giữa code và tài liệu

- **Phát hiện:** 2026-11-05
- **Hiện trạng:** `modal/app.py` chặn mọi GPU khác A10 ở hai chỗ:
  `train` (dòng 317-319) và `compare` (dòng 448), với thông điệp
  "Only A10 GPU is allowed by the resource cap". Nhưng
  `docs/agents/OPERATIONS.md` dòng 7 lại ghi "Full training and benchmark
  require Modal with an NVIDIA L4".
- **Ảnh hưởng:** không rõ GPU nào là chuẩn để tái lập kết quả. Cũng chưa rõ
  `_train_ablation` (gọi thẳng `base_app._train.with_options(gpu=...)`, bỏ qua
  local entrypoint) có vượt được guard hay không.
- **Đề xuất:** chọn một nguồn sự thật duy nhất, sửa bên còn lại, và ghi lý do
  vào `docs/DECISIONS.md`.

## BL-002: Chưa có số đo throughput của L4 và T4 cho workload này

- **Phát hiện:** 2026-11-05
- **Hiện trạng:** mọi so sánh chi phí giữa A10G / L4 / T4 trong repo đều phải
  giả định hệ số chậm (1.75× cho L4, 2.30× cho T4) suy từ bandwidth lý thuyết.
  Không có `results.csv` nào của L4 hay T4 trong repo.
- **Ảnh hưởng:** không thể kết luận chắc chắn GPU nào rẻ hơn **trên mỗi run**,
  chỉ kết luận được trên \$/giờ.
- **Đề xuất:** chạy 1 pilot 3 epoch trên L4 và T4 với đúng config, ghi
  s/epoch vào `reports/`. Chi phí ước tính dưới \$0.20.

## BL-003: `save_period` không ghi checkpoint định kỳ

- **Phát hiện:** 2026-11-05
- **Hiện trạng:** config đặt `save_period: -1` nên Ultralytics không ghi
  checkpoint theo chu kỳ; chỉ có `last.pt` và `best.pt` được ghi mỗi epoch.
  Yêu cầu ban đầu là "lưu checkpoint mỗi 10 epoch vào thư mục riêng theo tên run".
- **Ảnh hưởng:** không có snapshot trung gian để quay lại; chỉ resume được từ
  epoch gần nhất.
- **Đề xuất:** nếu cần snapshot định kỳ, đặt `save_period: 10` cho các run tiếp
  theo — nhưng phải ghi vào `docs/DECISIONS.md` vì nó làm tăng dung lượng volume
  và có thể ảnh hưởng thời gian epoch.

## BL-004: `scripts/validate_tiny_ablation.py` trỏ tới model YAML không tồn tại (đã đóng — xem BL-015)

- **Phát hiện:** 2026-11-06
- **Hiện trạng:** `DEFAULT_MODELS` trong `scripts/validate_tiny_ablation.py`
  liệt kê `yolo26n_tiny_A0_baseline.yaml`, `yolo26n_tiny_A1_no_sppf.yaml`,
  `yolo26n_tiny_A3_p2_no_sppf.yaml`. Cả ba file đã bị xóa khỏi `configs/model/`;
  chỉ còn `yolo26n_tiny_A2_p2.yaml`. Script cũng còn nhánh `_msr`/`_btrm`.
- **Ảnh hưởng:** chạy script không tham số sẽ `FileNotFoundError`. `docs/agents/CODE.md`
  đã được sửa để gọi với `configs/model/yolo26n_tiny_A2_p2.yaml`, nhưng bản thân
  script vẫn sai.
- **Đề xuất:** cập nhật `DEFAULT_MODELS` và bỏ nhánh `_msr`/`_btrm` khi có dịp
  sửa code (ngoài phạm vi task dọn dẹp này, vì yêu cầu là không đổi code).

## BL-005: `reports/experiment_results_report.md` là hồ sơ của hướng nghiên cứu đã dừng

- **Phát hiện:** 2026-11-06
- **Hiện trạng:** file này tổng hợp A0–A3, MSR, BTRM, SADR, QAC v1–v3, oracle AP75.
  Đây là hồ sơ lịch sử của các hướng đã dừng, không phải kết quả GauSS hiện tại.
  Các tài liệu QAC/QAR riêng lẻ đã được xóa; file này được giữ vì nó là nguồn
  duy nhất ghi lại số liệu A0–A3/MSR/BTRM/SADR.
- **Ảnh hưởng:** người đọc mới dễ nhầm đây là kết quả hiện hành.
- **Đề xuất:** thêm banner "lịch sử — đã ngừng" ở đầu file, hoặc chuyển vào
  `docs/archive/` khi không còn ai cần đối chiếu.


## BL-006: Evaluator TinyPerson bỏ toàn bộ dự đoán lớp thứ hai (đã sửa)

- **Phát hiện:** 2026-11-06
- **Hiện trạng:** `src/tinydet/evaluation/tinyperson_benchmark.py` lọc
  `if int(class_id) != 0: continue` trong khi `_load_gt` giữ ground truth của cả
  `sea_person` (category 1) và `earth_person` (category 2). Mọi bảng AP tuyệt
  đối trong bản thảo trước ngày 2026-11-06 đều bị ảnh hưởng.
- **Ảnh hưởng định lượng:** AP50_all của GauSS 12.94 → 29.97 (per-class) /
  35.39 (merged); Baseline 11.95 → 31.18 / 36.81. Kết luận "+0.98 pp" đảo thành
  −1.21 pp khi khớp đúng lớp.
- **Đã sửa:** thêm `class_mode` với ba chế độ; `per_class` là mặc định, `primary`
  giữ hành vi cũ để tái lập số đã công bố. Xem ADR-008.
- **Còn lại:** các bảng legacy trong `paper/main.tex` (Table 3, Table 5) vẫn
  dùng số cũ, đã gắn nhãn cảnh báo; cần quyết định có xoá hẳn hay không.

## BL-007: 5 run ablation seed-42 dừng ở epoch 78/150 và không có mAP trong log

- **Phát hiện:** 2026-11-06
- **Hiện trạng:** `backups/ablation_seed42/*/*/training/results.csv` chỉ có 78
  dòng (epoch 1–78), cột `metrics/mAP50(B)` bằng 0 ở **mọi** epoch của **cả 5**
  run. `backups/MANIFEST.json` ghi rõ: "interrupted at epoch 77/150 (Modal credit
  exhausted)". `metadata.json` của cả 5 run vẫn để `status: RUNNING`.
- **Nguyên nhân cột mAP = 0:** `config.yaml` đặt `train.val: false`, nên
  Ultralytics không chạy validation trong lúc train và để trống các cột metrics.
  Đây là hành vi đã biết, không phải lỗi dữ liệu.
- **Ảnh hưởng:** mọi so sánh ablation hiện tại là giữa các checkpoint 78 epoch,
  không phải 150 epoch như run tham chiếu. Không thể coi là ablation hoàn chỉnh.
- **Đề xuất:** hoặc train lại 5 run đủ 150 epoch (cần ngân sách Modal), hoặc ghi
  rõ trong bài là "78-epoch ablation, so với reference 150-epoch" và không so
  sánh trực tiếp với run tham chiếu.

## BL-008: Không thể truy vết cờ ablation từ artifact đã lưu

- **Phát hiện:** 2026-11-06
- **Hiện trạng:** không `metadata.json`, `config.yaml` hay `environment.txt` nào
  của 5 run ablation ghi lại giá trị `TINYDET_ABLATION` lúc train. Cờ chỉ được
  pickle vào checkpoint (`GauSS.ablation_flags`), nên chỉ đọc được khi load model.
- **Bằng chứng cờ có hiệu lực:** trọng số EMA khác nhau giữa 5 run
  (`gamma_h_raw` = −0.4055, −0.2607, −0.2739, −0.4915, −0.2030), nên các run
  thực sự phân kỳ chứ không phải cùng một model.
- **Đề xuất:** ghi cờ vào `metadata.json` ngay từ đầu trong lần chạy sau; thêm
  kiểm tra trong `modal/app.py` để từ chối ghi artifact khi thiếu trường này.

## BL-009: Derivative train thực tế có `val` trỏ vào chính tập train và 2 lớp

- **Phát hiện:** 2026-11-07
- **Hiện trạng:** snapshot `/mnt/data/tinyperson-yolo` (nay là `backups/data/tinyperson-yolo`)
  có `data.yaml` ghi `val: images/train`, `names: {0: sea_person, 1: earth_person}`.
  Mọi config experiment cũ trỏ vào derivative này với `train.val: false`.
- **Ảnh hưởng:** `best.pt` trùng `last.pt`; không có checkpoint selection thực sự. Train
  2 lớp trong khi benchmark chính thức gộp 2 category. Mọi kết luận "GauSS thắng/thua"
  trong `paper/main.tex` đứng trên nền protocol này.
- **Đã xử lý:** sinh lại derivative `tinyperson-yolo-v2` (1 lớp, val 84 ảnh tách theo
  nhóm video, test 786 ảnh riêng) và bật `val: true`, `patience: 20`. Cần train lại
  (chờ duyệt GPU) trước khi trích dẫn bất kỳ số nào.
- **Chưa xử lý:** chưa xác định được ai đã sửa `val: images/train` trên volume.

## BL-010: Derivative parquet local có `val` = chính tập test

- **Phát hiện:** 2026-11-07
- **Hiện trạng:** `datasets/derived/tinyperson-yolo` (sinh bởi `convert_parquet_to_yolo.py`
  phiên bản cũ) gán `val` = `test-*.parquet`, tức 786 ảnh test chính thức. Bật `val: true`
  với derivative này là leak test vào checkpoint selection.
- **Đã xử lý:** converter giờ sinh `train`/`val`/`test` riêng; `val` tách từ train theo
  nhóm filename, `test` không bao giờ dùng để chọn checkpoint.
- **Chưa xử lý:** derivative cũ vẫn còn trên đĩa; nên xóa hoặc đổi tên để tránh dùng nhầm.

## BL-011: Ảnh parquet khác ảnh erase ở vùng ignore

- **Phát hiện:** 2026-11-07
- **Hiện trạng:** ảnh trong `datasets/tiny-object-detection/tinyperson` là ảnh gốc, còn
  `erase_with_uncertain_dataset` đã tô vùng ignore/uncertain. So sánh pixel trên 3 ảnh mẫu
  cho khác biệt 1,1–11,9 %.
- **Ảnh hưởng:** hai nguồn không thể trộn trong cùng một derivative; người dùng cần biết
  mình đang train trên bản nào khi so sánh với con số công bố của TinyPerson.
- **Đề xuất:** coi erase là nguồn canonical (khuyến nghị chính thức của tác giả), ghi rõ
  trong paper; giữ parquet làm nguồn phụ có ghi chú.

## BL-012: `compute_gauss_analytical_macs` chưa có phiên bản chuẩn hóa cho P2Refine

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** Trong `src/tinydet/modules/gauss.py` có hàm giải tích tính MACs cho GauSS.
  Đối với `P2Refine`, MACs hiện được đo thực nghiệm độc lập bằng script `measure_p2refine.py`.
- **Ảnh hưởng:** Các script báo cáo hoặc profile tự động cần một hàm giải tích thống nhất
  để xuất bảng thống kê FLOPs chi tiết cho từng block mà không cần instantiate tensor.
- **Đề xuất:** Thêm hàm `compute_p2refine_analytical_macs` vào `src/tinydet/modules/p2_refine.py`
  và re-export trong `tinydet.modules`.

## BL-013: GauSS diagnostics chưa từng được lưu trong 21 lượt train trước

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** 21 thư mục run cũ trên Modal volume (`metadata.json`) không có bất kỳ log
  nào về router gate, band weight hay dynamic scale-space.
- **Đã xử lý:** Đã triển khai `P2RefineDiagnosticsCallback` gắn tự động qua trainer lifecycle
  để ghi lại `rho_mean`, `rho_max`, `cos_x0_delta`, `w_out_norm` cho P2Refine.
- **Chưa xử lý:** Không thể phục hồi dữ liệu động học (training dynamics) của 21 run GauSS cũ
  ngoại trừ các checkpoint `best.pt`/`last.pt` còn lưu.

## BL-014: `wandb` thiếu khỏi `pyproject.toml` nên `uv sync` không cài

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** `src/tinydet/training/trainer.py`, `iou_logging.py`, `module_diagnostics.py`
  và `wandb_media.py` đều `import wandb` trong `try/except` hoặc có guard, nhưng `wandb`
  không nằm trong `[project.dependencies]` và không có trong `uv.lock`. Trên Modal thì
  `modal/image.py:8` cài riêng nên không lộ; trên máy local hoặc Colab/Kaggle dùng
  `uv sync` thì W&B **im lặng tắt** — `_init_wandb` bắt `ImportError` và in
  "W&B disabled after init failure".
- **Ảnh hưởng:** mọi run chạy bằng `uv sync` đều không log W&B dù config có
  `wandb_project`. Không mất dữ liệu (CSV và metadata.json vẫn ghi), nhưng mất khả năng
  theo dõi trực tuyến.
- **Đã xử lý:** thêm `"wandb>=0.21,<1"` vào `[project.dependencies]`, chạy `uv lock`
  (thêm `wandb v0.31.0`). Kiểm chứng: `uv run --extra cpu python -c "import wandb"` →
  `0.31.0`.
- **Chưa xử lý:** `modal` vẫn là dependency bắt buộc (xem BL-016).

## BL-015: `scripts/validate_tiny_ablation.py` trỏ tới 3 model YAML đã xóa

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** `DEFAULT_MODELS` liệt kê `yolo26n_tiny_A0_baseline.yaml`,
  `yolo26n_tiny_A1_no_sppf.yaml`, `yolo26n_tiny_A3_p2_no_sppf.yaml`. Cả ba không còn
  trong `configs/model/`, nên chạy script không tham số sẽ `FileNotFoundError`.
- **Đã xử lý:** rút `DEFAULT_MODELS` còn 3 file thực có:
  `yolo26n_tiny_A2_p2.yaml`, `yolo26n_tiny_A2_p2_tinyperson.yaml`,
  `yolo26n_tiny_A2_p2_p2refine_tinyperson.yaml`. Chạy không tham số giờ pass cả 3.
- **Ghi chú:** mục này thay thế BL-004 (đã đóng).

## BL-016: `modal` vẫn là dependency bắt buộc của package

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** `pyproject.toml:17` khai báo `modal>=1,<2` trong `[project.dependencies]`.
  Trên Colab/Kaggle, `pip install -e .` kéo theo `modal` dù không dùng.
- **Ảnh hưởng:** tăng thời gian cài và dung lượng, nhưng không chặn gì. `src/tinydet/**`
  không import `modal` ở đâu.
- **Đề xuất:** chuyển `modal` sang optional-dependency (ví dụ `[project.optional-dependencies]
  modal = [...]`) và cập nhật `uv sync` trong tài liệu. Chưa làm vì sẽ đổi lệnh cài mặc
  định của mọi người và ảnh hưởng workflow Modal hiện tại.

## BL-017: Chưa có smoke test tự động cho `train.py`

- **Phát hiện:** 2026-11-08
- **Hiện trạng:** `train.py` được kiểm chứng thủ công bằng `--dry-run` trên CPU và bằng
  cách gọi `trainer.train()` để xác nhận cổng chặn CUDA. Repo chưa có test suite
  (`pytest` không nằm trong dependency).
- **Ảnh hưởng:** nếu ai sửa `_resolve_model_config` hoặc `_resolve_paths`, lỗi chỉ lộ ra
  khi chạy thật.
- **Đề xuất:** thêm `tests/test_train_cli.py` kiểm tra `--dry-run` exit 0 và không có
  `--dry-run` exit 1 trên CPU. Cần thêm `pytest` vào `[dependency-groups].dev` trước.

## BL-018: Checkpoint cũ được pickle với class `DCAF` không thể load sau khi đổi tên

- **Phát hiện:** 2026-11-09
- **Hiện trạng:** Toàn bộ mã nguồn đã thống nhất dùng tên `GauSS` (class `DCAF` bị xóa khỏi
  `tinydet.modules.gauss`). Các checkpoint cũ (v2/v2.1/v3) được lưu bằng pickle chứa tham chiếu
  class `tinydet.modules.gauss.DCAF`, nên `torch.load` sẽ lỗi `AttributeError` khi unpickle.
- **Ảnh hưởng:** không chặn train mới (v3.1 train từ đầu), nhưng không thể resume/eval trực tiếp
  trên checkpoint cũ.
- **Đề xuất:** nếu cần dùng lại checkpoint cũ, extract state-dict rồi chuyển bằng
  `remap_v3_to_v31_state_dict()` (hoặc shim tạm `tinydet.modules.gauss.DCAF = GauSS` chỉ trong
  script load). Chưa làm vì các run sắp tới đều train mới với v3.1.

## BL-019: Thư mục run trên Modal volume vẫn mang tên `dcaf_v2_1_*`

- **Phát hiện:** 2026-11-09
- **Hiện trạng:** các thư mục run cũ trên Modal volume (`/mnt/runs/dcaf_v2_1_tinyperson_wacv/...`)
  và các artifact cục bộ (`backups/ablation_seed42/.../args.yaml`, `local_eval/*.json`) vẫn chứa
  chuỗi `dcaf`. Repo không kiểm soát được tên thư mục trên volume; các file này là artifact lịch sử.
- **Ảnh hưởng:** các script eval cũ (`launch_eval_tinyperson.py`, `launch_tiled_eval.py`,
  `configs/experiment/gauss_v3_visdrone_finetune.yaml` với `pretrained`) giờ trỏ tới tên mới
  `gauss_*` trên volume, nên sẽ không tìm thấy thư mục cũ cho đến khi run mới với tên `gauss_*`
  được tạo ra.
- **Đề xuất:** sau khi train v3.1 xong, cập nhật `pretrained` trong config finetune sang run mới;
  không đổi tên thư mục cũ trên volume (giữ provenance).

## BL-020: Config finetune VisDrone trỏ `pretrained` tới checkpoint cũ, đã chết sau đổi tên

- **Phát hiện:** 2026-11-09
- **Hiện trạng:** `configs/experiment/gauss_v3_visdrone_finetune.yaml` trỏ
  `pretrained: /mnt/runs/gauss_v2_1_tinyperson_wacv/...` và
  `baseline_visdrone_finetune.yaml` trỏ `/mnt/runs/baseline_tinyperson_wacv/...`.
  Thư mục thật trên volume vẫn tên `dcaf_v2_1_*` (BL-019), nên đường dẫn mới không tồn tại.
  Ngay cả khi trỏ đúng, checkpoint cũ là kiến trúc GauSS v2.1 — load kiểu "intersect" của
  Ultralytics sẽ bỏ qua ngầm các tham số mới của v3.1 (`router_corr.*`, `dy_up*.offset.*`,
  `gamma_in/out_raw`), tạo partial-load không kiểm soát.
- **Ảnh hưởng:** track finetune VisDrone không thể chạy đúng cho tới khi có checkpoint v3.1 mới.
- **Đề xuất:** sau khi run TinyPerson v3.1 (60e) hoàn tất, cập nhật `pretrained` của cả hai
  config finetune sang `best.pt` của run mới tương ứng (baseline → baseline, gauss → gauss),
  cùng kiến trúc, cùng protocol. Không sửa tạm bằng shim.

## BL-021: Cấm dùng `VisDrone.yaml` tích hợp sẵn của Ultralytics trong Modal

- **Phát hiện:** 2026-11-09
- **Hiện trạng:** Ultralytics có sẵn `ultralytics/cfg/datasets/VisDrone.yaml` tự download
  dataset lúc runtime về thư mục cục bộ của container. Bản này khác cấu trúc với derivative
  `/mnt/data/visdrone-yolo` trên volume và không có signature SHA-256 cho preflight.
- **Ảnh hưởng:** nếu dùng, dữ liệu train không reproducible, preflight audit bị bỏ qua,
  vi phạm ADR-009 và `docs/agents/DATA_EVALUATION.md`.
- **Đề xuất:** mọi config VisDrone chỉ được trỏ `/mnt/data/visdrone-yolo/data.yaml`
  (derivative chuẩn trên volume, 6471/548/1610 ảnh train/val/test, 10 lớp). Đã kiểm tra
  volume ngày 2026-11-09.

## BL-022: Cache key của preflight audit không bao gồm nội dung images/labels

- **Phát hiện:** 2026-11-09
- **Hiện trạng:** `_audit_original_dataset` trong `modal/app.py` tạo cache key từ
  `sha256(dataset_yaml_path + imgsz + bytes(data.yaml))[:16]`. Nếu nội dung
  `images/` hoặc `labels/` trên volume thay đổi mà `data.yaml` không đổi (ví dụ upload lại
  derivative đã sửa), audit trả về kết quả stale từ cache mà không hề hay biết.
- **Ảnh hưởng:** signature SHA-256 ghi vào metadata run vẫn đúng (nó hash file stats), nhưng
  quyết định "passed" có thể dựa trên dữ liệu cũ.
- **Đề xuất:** đưa `manifest.json` (có sẵn trong derivative, chứa sha256 nguồn) vào cache key,
  hoặc hash danh sách `(basename, size, mtime)` của images/labels khi tạo key. Chưa làm vì
  cần cân nhắc chi phí hash ~1.5k file mỗi lần audit.

## BL-023: Size-conditioned routing metrics (M8 trong METRICS_EXPANSION_SPEC)

- **Phát hiện:** 2026-11-09
- **Hiện trạng:** để đo band weight `a_h/a_m/a_l` bên trong GT box (kiểm chứng giả thuyết
  "router chọn high-band cho vật thể tiny"), cần thêm buffer band-weight map vào
  `GauSS.forward` (flag `TINYDET_SAVE_BAND_MAP`, off mặc định). Đây là mục duy nhất trong
  spec metrics phải đụng forward của module, nên đã defer chờ review riêng.
- **Đề xuất:** khi có nhu cầu cho paper, thêm buffer + callback crop theo GT box stride-8,
  test so sánh memory overhead trước/sau.

## BL-024: MR^-2 (log-average miss rate over FPPI) chưa có trong evaluator

- **Phát hiện:** 2026-11-09
- **Hiện trạng:** TinyPerson protocol hiện chỉ tính AP và MR@IoU cố định (0.25/0.50/0.75).
  MR^-2 (chuẩn Caltech/TinyPerson literature) chưa được implement ở đâu trong repo.
- **Đề xuất:** thêm vào `tinyperson_benchmark.py` (post-train, không per-epoch) nếu bảng
  so sánh với literature trong paper cần số này.

## BL-025: `InterpreterPoolExecutor` (Python 3.14) bị chặn bởi numpy/torch/PIL

- **Phát hiện:** 2026-11-09, đo trực tiếp trên Python 3.14.9 của repo.
- **Hiện trạng:** `concurrent.futures.InterpreterPoolExecutor` chạy tốt với pure-Python
  (0.31 s so với 1.23 s của ThreadPoolExecutor cho 8×3M vòng lặp, ngang `ProcessPoolExecutor`
  fork 0.30 s), nhưng **import numpy 2.5.4, torch 2.14.0, PIL 13 đều thất bại** trong
  subinterpreter (`ImportError: ... does not support loading in subinterpreters`). Upstream
  numpy xác nhận chưa hỗ trợ và không nằm trong roadmap gần (cần chuyển sang multi-phase init
  + heap types). Vì mọi đường CPU-heavy của repo (IoU matrix, AP/MR, decode ảnh) đều dựa trên
  numpy/PIL/torch, không thể thay `ProcessPoolExecutor(fork)` bằng interpreter pool.
- **Đề xuất:** giữ nguyên fork-based `ProcessPoolExecutor` trong evaluator (đã đo 3.36x ở 84
  ảnh). Chỉ xem lại khi numpy công bố hỗ trợ subinterpreter; khi đó benchmark lại trước khi đổi.
- **Cập nhật 2026-11-09 (đào sâu):** đã xác định được **một pattern dùng được**: parse JSONL
  song song khi worker tự aggregate và chỉ trả kết quả nhỏ. Đo trên file thật
  `outputs/error_analysis/tiny_A2_p2-test/errors.jsonl` (56.5 MB, 140k dòng):
  - sequential full parse: 844 ms
  - interpreter(4) parse + aggregate trong worker (trả `Counter` nhỏ): **303 ms (2.79x)**
  - interpreter(4) parse rồi trả full list dict về main: **803 ms (0.99x — mất hết lợi thế)**
  - interpreter(4) chỉ parse, trả `int`: 287 ms (2.76x)
  - payload 0.83 MB (84 ảnh × 200 dets, mức per-epoch): 51 ms vs 23 ms → **chậm gấp 2.2x**
  - ngưỡng có lợi: ~5 MB (1.42x), 15 MB (1.97x), 25 MB (2.38x)
  - Module load được trong subinterpreter: json, csv, re, hashlib, zlib, lzma, sqlite3, pickle,
    decimal, yaml, **polars**. Không load được: numpy, scipy, sklearn, matplotlib, seaborn, cv2,
    torch, PIL.
  - Ứng viên trong repo: `_compare_samples` (`src/tinydet/error_analysis/core.py:703`) parse
    `samples.jsonl` (45 MB) để lấy summary theo image — đúng dạng aggregate, tiết kiệm ~0.6 s
    mỗi lần chạy error analysis. Chưa làm vì lợi ích tuyệt đối nhỏ so với tổng thời gian script
    và cần thêm một nhánh code song song; xem lại nếu error analysis chạy thường xuyên.
- **Trạng thái: ĐÃ LÀM (2026-11-09).** Tạo `src/tinydet/error_analysis/baseline_summary.py`
  (stdlib-only) với worker `summarize_chunk` + driver `load_baseline_summaries`;
  `_compare_samples` giờ dùng nó. Thiết kế: chia theo **byte** (1 MB, cắt tại ranh giới dòng —
  `samples.jsonl` chỉ có 786 dòng × ~59 KB nên chia theo số dòng sẽ thành 1 chunk), ngưỡng
  5 MB, 4 workers, worker tự aggregate và chỉ trả `{basename: (misses, fps)}`.
  Đo trên dữ liệu thật (46.5 MB, 786 dòng): old 762 ms → new 200–254 ms, **3.0–4.0x**,
  output `equal=True` với implementation cũ, provenance mismatch vẫn detect đúng.
  Phát hiện thêm (L-007): worker bị chia sẻ by value — **mọi tham chiếu global, kể cả hằng số
  module-level, đều gây `NotShareableError`**; worker phải chỉ dùng biến local + import bên
  trong thân hàm.


## BL-026: `compression.zstd` cho artifact lớn (prediction JSONL / eval dump)

- **Phát hiện:** 2026-11-09.
- **Hiện trạng:** Python 3.14 có `compression.zstd`; đo trên JSON 9.53 MB: zstd level 3 nén
  3 ms → 0.03 MB, gzip level 6 nén 19 ms → 0.07 MB (nhanh ~6x, nhỏ hơn ~2x). Hiện repo ghi
  prediction/eval dạng JSON/JSONL thuần (tới ~41 MB cho 2000 ảnh) và các script đọc trực tiếp.
- **Đề xuất:** chỉ áp dụng khi cần lưu trữ dài hạn hoặc truyền artifact qua Volume; phải cập
  nhật đồng thời writer + mọi reader (`evaluate_coco_small.py`, `fetch`, error analysis) và
  ghi format version. Không làm trước campaign 60e vì sẽ phá tương thích file hiện có.

## BL-027: Cap zoom của `RandomSizedBBoxSafeCrop` trong preset `tinyperson_b`

- **Phát hiện:** 2026-11-09, trong rà soát tiền-campaign 150e (ADR-019).
- **Hiện trạng:** preset B dùng `A.RandomSizedBBoxSafeCrop(1280, 1280, erosion_rate=0.0, p=0.20)`.
  Đo thực nghiệm bằng hook resize trên toàn bộ 609/662 ảnh train có bbox (53 ảnh background):
  zoom factor = 1280 / max(crop_h, crop_w) → median 1.26, mean 1.74, p90 3.10, p95 4.52,
  **max 20.3x**; 22.8% số lần crop zoom >2x, 6.7% >4x, 1.6% >8x. Với p=0.20 tức
  ~1.3% sample mỗi epoch bị zoom >4x (người tí hon phóng to tới 20x, texture gần như mất).
  `erosion_rate` **không** phải sàn kích thước crop: source `BBoxSafeRandomCrop.get_params_dependent_on_data`
  cho thấy nó erosion (thu nhỏ) union bbox, tức tăng erosion_rate làm zoom cực đoan hơn.
  Không có knob cap zoom thuần thư viện trong transform này.
- **Quyết định (user, 2026-11-09):** giữ nguyên p=0.20 cho vòng 150e; zoom-in crop vốn có lợi
  cho tiny object và exposure cực đoan rất nhỏ. Không churn preset trước campaign.
- **Đề xuất:** ablation vòng sau — (a) p=0.10, (b) thay bằng `A.Affine` zoom-in có kiểm soát
  (scale ≤2x), hoặc (c) bỏ crop; đo AP theo size-stratified (tiny1/tiny2) để xem zoom-in crop
  giúp hay hại. Cần đổi preset trong `src/tinydet/augment/albumentations_stack.py` và cập nhật ADR-018.

## BL-028: Điều tra tăng trưởng RAM theo epoch của training container

- **Phát hiện:** 2026-11-10, từ crash SIGKILL 137 của campaign 150e (L-009).
- **Hiện trạng:** RAM container tăng dần theo epoch; với batch 16 @1280 và `workers: 8`,
  ceiling 24 GB bị vượt ở ~epoch 131. Chưa xác định được nguồn tăng: ứng viên là
  `register_size_stratified_metrics` / `val_predictions_latest.json` (84 ảnh × 300 dets mỗi
  epoch) / `WandbMediaCallback` (log ảnh mỗi epoch) / cache của Ultralytics.
- **Đề xuất:** đo `psutil.Process().memory_info().rss` mỗi epoch và ghi vào
  `system_metrics.csv` (hiện chỉ có VRAM) để có bằng chứng; sau đó mới quyết định free
  object trong callback hay giảm tần suất log media. Không sửa trước campaign 150e.

## BL-029: `launch_campaign.py::fetch` lỗi `reload()` ngoài container

- **Phát hiện:** 2026-11-10, sau campaign 150e. `modal run launch_campaign.py::fetch` →
  `RuntimeError: reload() can only be called from within a running function`
  (`modal/volume.py:935`). Nguyên nhân: `runs_volume.reload()` gọi từ local entrypoint
  (không phải trong container).
- **Workaround đã dùng:** `modal volume get tiny-person-runs eval/<name>/summary.json reports/eval/<name>.json`
  cho từng dir — hoạt động tốt, chỉ thủ công.
- **Đề xuất:** bỏ `reload()` (Volume API `listdir`/`read_file_into_fileobj` từ client đã đọc
  trạng thái mới nhất), hoặc chuyển `fetch` thành một `@app.function` rồi `.remote()`.
  Cần kiểm tra lại với Modal 1.6.3 trước khi sửa.

## BL-030: Ablation 2×2 (augment × model) để phân lập nguyên nhân gap GauSS thu hẹp

- **Phát hiện:** 2026-11-10. Gap AP@0.50 tiled (GauSS − baseline) giảm từ +0.0221 (+7.3%)
  ở vòng 60e xuống +0.0046 (+1.1%) ở vòng 150e; tiny từ +0.0337 xuống −0.0016.
- **Giả thuyết chính:** `augmentations_preset: tinyperson_b` bật cho **cả hai** model đã
  "dạy" scale invariance bằng dữ liệu (`Downscale` 0.4–0.7 + `RandomSizedBBoxSafeCrop`
  zoom tới 20×), trùng chức năng GauSS ⇒ baseline không còn cần module.
- **Bằng chứng gián tiếp (không tốn GPU):**
  - Baseline tăng 60e→150e mạnh nhất ở tiny (+40.7%) và tiny1 (+33.6%), GauSS chỉ +24.4%/+17.7%;
    GauSS tăng mạnh nhất ở reasonable (+43.8%) nơi scale aug ít liên quan.
  - GauSS đạt 90% fitness cuối ở epoch 61, baseline ở epoch 76 ⇒ lợi thế GauSS ở 60e phần
    lớn là tốc độ hội tụ, không phải trần chất lượng.
  - Gap fitness theo epoch dao động ±0.01 quanh 0 (e20 −0.008, e40 +0.009, e60 −0.004).
- **Confound:** giữa 2 vòng thay đổi 5 biến cùng lúc (epochs 60→150, batch 8→16,
  patience 15→40, warmup 2→3, augment none→preset B) ⇒ chưa thể quy cho augment.
- **Đề xuất (chưa chạy — user quyết định hoãn):** hoàn tất ma trận 2×2 bằng 2 run
  **150e không augment** (baseline + gauss), giữ nguyên mọi siêu tham số khác. Chi phí
  ~2 h GPU ≈ $4. Dự đoán bác bỏ được: gap(no-augment) ≈ +0.02 ⇒ giả thuyết đúng;
  ≈ +0.005 ⇒ giả thuyết sai (nguyên nhân là schedule/batch).
- **Ý nghĩa:** nếu giả thuyết đúng, claim của GauSS phải chuyển từ "tăng AP cuối" sang
  "recall sớm (AP@0.25/MR@0.25) + hội tụ nhanh + score calibration"; đây là claim cơ chế
  mạnh hơn và trung thực hơn.

## BL-031: 16 lỗi ruff pre-existing ngoài `src/` — `ruff check .` đang đỏ

- **Phát hiện:** 2026-11-10 khi lint trước commit L-011. `uv run --extra cpu --group dev
  ruff check .` báo 16 lỗi (5 fixable), toàn bộ nằm ngoài `src/tinydet`:
  `launch_parallel_tinyperson.py` (3), `report-bundle/tools/{build,check_manuscript,
  guard,make_figures,md_lint,verify_refs}.py` (13). Chủ yếu I001 import sort và tương tự.
- **Tác động:** hook lint đỏ khiến mỗi commit phải kiểm tra thủ công lỗi nào là mới.
- **Đề xuất:** chạy `ruff check launch_parallel_tinyperson.py report-bundle --fix` một
  lần trong commit riêng, hoặc thêm `exclude` cho các script này trong `pyproject.toml`
  nếu chúng là tool một-lần.
