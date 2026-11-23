# Pre-registration — PairSplit (second-instance head)

**Written before any training run.** Committed to the repo so the criteria cannot
be moved after seeing results. If a criterion is later changed, the change and its
reason must be appended here with a date, and the original kept visible.

Date: 2026-11-11. Author: research loop (see `TINY_RESEARCH_LOG.md` R20–R23).

## 1. What is being tested

`PairSplit` (23,690 params, `src/tinydet/modules/pair_split.py`) is a post-hoc head
attached to a **frozen** YOLO26n-P2 checkpoint. It reads the 7×7 P2 neighbourhood
around each existing detection and predicts whether a second person is present,
plus its relative offset and size. It does **not** change the backbone, the Detect
head, the assigner, or the detector loss.

The claim under test is narrow and falsifiable:

> A single extra 23.7 k-parameter head, trained only on targets derived from the
> existing annotations, recovers a measurable part of the close-pair recall gap
> that survives 1:1 tiled inference.

It is **not** claimed that PairSplit beats tiling, replaces tiling, or improves
isolated objects. Tiling remains the resolution fix; PairSplit is the residual fix.

## 2. Motivation that must hold (already measured, not re-litigated)

| Evidence | Value | Source |
|---|---|---|
| Close-pair TP rate vs isolated (tiled) | 59.1% vs 76.0% | R20 |
| Gap, full-frame → tiled | +33.0 pp → +17.0 pp | R20 |
| Cleanest cell: s_eff 10–16 px, d_nn<6 vs ≥48 px | 52.5% vs 77.5% | R20 |
| Close-pair misses with a box credited to the neighbour | 3.7% (tiled) | R21 |
| Nearest detection distance, in GT sizes | 0.37 (median) | R22 |
| Missed GT with a detection at IoU ≥ 0.3 | 53.5% | R22 |
| Share of tiled GTs in close pairs | 19.0% | R20 |
| Implied AP ceiling for the module | +3–4 pp | derived |

If the training run shows the head learning nothing (criterion K3 below), the
R22 reading — "the model fires near the pair but emits one box for two people" —
is wrong, and that must be reported as a falsification of the motivation, not as a
tuning problem.

## 3. Protocol (fixed before the run)

- **Checkpoint:** the same frozen YOLO26n-P2 weights used for the R20–R22 probes.
  No fine-tuning of the detector in the primary experiment.
- **Inference protocol:** tiled 640×512, overlap 100, 1:1 (the protocol that
  produced the **0.3757** baseline for checkpoint 20261109-182052-a20f7b6-seed42;
  reproduced 2026-11-11 at 0.37556). Full-frame results (0.3002) are reported as a
  secondary column only.
- **Head training data:** train split only. Targets from `build_pair_targets`
  (IoU ≥ 0.5 match, radius `min(1.5 × max(w,h), 10 px)`; xem §13). No new annotation.
- **Threshold:** chọn trên **val** (84 ảnh, `scratch/ann_val_official.json`), sau đó chấm
  test **một lần**. `max_ratio = 3.0`, fixed. (Sửa 2026-11-11, xem §13.)
- **Evaluation:** official `evaluate_tinyperson` with `class_mode="merge"`,
  AP50 restricted to the tiny bins (`sqrt(w*h)`), the same evaluator that produced
  0.3757 / 0.3002 for this checkpoint. (0.1886 / 0.3540 in log R6.a are legacy and
  belong to a different model/checkpoint; do not mix.) In-house numbers are reported
  alongside but are not decisive.
- **Statistics:** clustered bootstrap over video groups, B = 2000, 95% CI on the
  paired AP difference (same images, same checkpoint, head on vs off).

## 4. Primary criterion

**P1.** AP50_tiny under the tiled protocol improves, and the 95% clustered
bootstrap CI lower bound on the paired difference is **> 0**.

Pass = P1. Anything else is a fail, regardless of how the other numbers look.

## 5. Guard criteria (all must hold; a violation voids P1)

**G1 — no harm to isolated objects.** TP rate on GTs with `d_nn ≥ 48 px` must not
drop by more than **1.0 pp**. The head must not be buying close-pair recall by
degrading the easy majority.

**G2 — the extra boxes are real.** Precision of the emitted second boxes at
threshold 0.5 must be **≥ 0.5** (an emitted box counts as correct at IoU ≥ 0.5
against a GT that is *not* the one the source detection already matched).

**G3 — no duplicate cloud.** The ratio of detections per GT must not increase by
more than **20%** relative to the head-off run. A head that emits a near-copy of
the source box is a failure mode, not a gain.

**G4 — the gain is in the close-pair stratum.** The AP gain must be concentrated
where the mechanism predicts: GTs with `d_nn < 10 px` must improve by **at least
2×** the improvement on `d_nn ≥ 48 px`. A uniform gain across strata means the
head is acting as a generic re-ranker and the mechanism story is wrong.

## 6. Kill criteria (stop, do not tune)

**K1.** P1 fails with CI lower bound ≤ 0 after one training run at the fixed
hyper-parameters (AdamW, lr 1e-3, 20 epochs, batch 64 patches). No hyper-parameter
sweep to rescue it.

**K2.** G2 fails (precision < 0.5): the head emits noise. Report as a negative
result for the *representation* claim.

**K3.** `p(second)` collapses to the init value (0.119 ± 0.02 mean over the val
set) and the regression term does not decrease over training: the head found no
signal. This falsifies the R22 interpretation and must be written up as such.

**K4.** G3 fails by more than 2× the allowed margin (ratio +40%): the head is
duplicating, not splitting. No post-hoc NMS rescue — that would change the
protocol after the fact.

## 7. What will be reported either way

- The paired AP difference with CI, per size bin, per `d_nn` stratum.
- The head's learned `p(second)` distribution and its calibration against the
  derived targets.
- The measured parameter and FLOP cost (23,690 params; 7×7×32 gather + 1×1 bottleneck
  32→6 + 2×64 MLP per detection).
- If it fails: the falsified hypothesis, stated as falsified.

## 8. Known limits, stated up front

- The ceiling is **+3–4 pp AP**, derived from 19.0% of tiled GTs being in close
  pairs at 59.1% vs 76.0% TP. A result near +1 pp is within the mechanism's
  budget; a result of +10 pp would be suspicious and must be audited for leakage
  (e.g. the head seeing the GT through the target builder).
- The head is trained on detections from the frozen checkpoint, so it inherits
  that checkpoint's failure modes. It cannot recover a person the detector never
  fires near (R22: ~24% of misses have no detection within 1 GT-size).
- CrowdDet (arXiv:2003.09163) is the closest prior art and predicts multiple
  instances per proposal. PairSplit differs in regime (tiny objects where the
  response width exceeds object spacing), in being anchor-free and single-class,
  in being post-hoc attachable to a frozen model, and in deriving targets from
  existing boxes with no EMD loss. This must be stated in the paper, not omitted.

---

# Phần 2 — GaussSplit (bản có Gaussian), viết trước khi chạy

Bổ sung 2026-11-11. Mọi mục ở Phần 1 (P1, G1–G4, K1–K4) **giữ nguyên** và áp dụng cho
GaussSplit. Phần này chỉ thêm những gì riêng cho họ module Gaussian.

## 9. Các biến thể được phép chạy (và chỉ những biến thể này)

| Mã | Head | Hình học | Số tham số |
|---|---|---|---|
| `B0` | PairSplit (patch đơn, L1) | L1 | 23,690 |
| `B1` | PairSplit | NWD | 23,690 |
| `G1` | GaussSplit, DoG tắt (chỉ fine patch) | NWD | 17,608 |
| `G2` | GaussSplit, DoG bật | L1 | 17,608 |
| `G3` | GaussSplit, DoG bật | NWD | 17,608 |
| `G4` | GaussSplit đầy đủ (DoG + NWD) | NWD | 17,608 |

Số tham số đã cập nhật sau amendment 2026-11-11 (§13); giá trị cũ 23,108 / 15,124
tương ứng patch 3 và không còn dùng.

`--head pair --geom l1` cho `B0`, `--head gauss --geom nwd` cho `G4`. Các biến thể còn lại
chỉ khác cờ trong `GaussSplit`/`gauss_split_loss`; nếu không dựng được bằng cờ thì ghi vào
BACKLOG chứ không sửa module sau khi đã nhìn kết quả.

## 10. Tiêu chí riêng cho họ Gaussian

**P2.** `G4` phải **≥ `B0`** trên AP50_tiny (tiled), CI paired không âm ở cận dưới. Nếu
`G4 < B0` thì kết luận là biểu diễn Gaussian không giúp trong chế độ này — báo cáo trung
thực, và module của paper là `B0`.

**P3 (đóng góp của từng thành phần).** Mỗi thành phần phải chứng minh đóng góp **dương**
hoặc bị loại khỏi module cuối:
- `G3 − G1` = đóng góp của DoG (G1: scale-space).
- `G3 − G2` = đóng góp của NWD (G2: hình học Gaussian).

Chỉ báo cáo bảng của module cuối + các biến thể đã chạy; thành phần không đóng góp thì
không nằm trong module cuối.

**G5 (đã huỷ 2026-11-11, xem §13).** Tiêu chí cũ yêu cầu `w_resp` tương quan dương với
độ rộng GT thực của cặp gần nhất. Feature `w_resp` đã bị **loại khỏi module** vì đo được
là không đơn điệu theo kích thước blob ở patch 7 (0.0038 → 4.19 → 3.79 với σ = 0.5/1.0/2.0),
tức là bão hoà theo cửa sổ chứ không mang thông tin. Không có `w_resp` thì G5 không áp dụng.

**G6 (mới).** σ học được của `LearnableGaussianScaleSpace` phải **dịch chuyển khỏi giá trị
khởi tạo** (|σ − σ_init| > 0.02 ở ≥ 25% kênh sau train). Nếu σ đứng yên, G1 không thực sự
học được thang đo — báo cáo và cân nhắc bỏ DoG.

## 11. Điều kiện dừng riêng

**K5.** Nếu `G4` và `B0` chênh nhau **< 0.5 pp** với CI chứa 0, thì kết luận "Gaussian không
tạo khác biệt đo được ở chế độ này" và **module của paper là bản đơn giản hơn** (`B0`). Không
được chọn bản phức tạp hơn khi chênh lệch nằm trong nhiễu.

**K6.** Nếu `G4` **thấp hơn** `B0` quá 1 pp, dừng nhánh Gaussian, không tinh chỉnh thêm.

## 12. Chi phí phải báo cáo

Số tham số (17,608), thời gian forward của head trên mỗi detection (đo trên cùng batch), và
số detection tăng thêm (guard G3 ở Phần 1). Không báo FPS end-to-end nếu chưa đo thật.

## 13. Amendment 2026-11-11 — sửa sau khi đọc log B0 (trước khi có bất kỳ AP nào)

**Bằng chứng dẫn tới sửa.** Log train B0 (19,000 step, thiết kế cũ patch 3) cho thấy
`reg` kẹt 0.3–0.9, loss đi từ 1.13 → 0.82 (mức nhiễu), `cls` ≈ 0.5. Nguyên nhân đo được:

- Cửa sổ receptive của head cũ = patch 3 × stride 4 = **±6 px**.
- Trên 9,330 cặp train, khoảng cách trung vị tới láng giềng gần nhất là **14.1 px**;
  chỉ 32% cặp ≤ 10 px.
- Tỉ lệ **nhìn thấy** láng giềng được supervise: ±6 px → 14.1%; ±12 px → 46.0%.
  ⇒ ~86% positive nằm ngoài cửa sổ của head. Đây là lỗi thiết kế, không phải lỗi tune.

**Ba thay đổi (giữ nguyên P1, G1–G4, K1–K4):**

1. **Bán kính target có trần 10 px**: `min(1.5 × max(w,h), 10 px)`. Lý do khoa học: chỉ
   supervise chế độ cặp sát mà detector thực sự thất bại; supervise cả cặp 20–30 px (detector
   đã xử lý được) chỉ thêm false positive. Đo lại trên pool thật (662 ảnh, **3,195 tile có GT**,
   30,798 GT): positive = **6,055 (19.7%)** với trần 10 px, so với 18,949 (61.5%) theo luật cũ.
2. **patch 7** (±12 px) ⇒ mọi target ≤ 10 px nằm trong cửa sổ. Để giữ ngân sách tham số,
   thêm bottleneck 1×1 `32→6` trước MLP thay vì flatten 32×49. Param: PairSplit **23,690**,
   GaussSplit **17,608**.
3. **Ngưỡng chọn trên val** (84 ảnh, `scratch/ann_val_official.json`, dựng bằng
   `subset_annotations_to_split` từ `tiny_set_train_with_dense.json`), test chấm **một lần**.
   Không leakage: val không nằm trong train, test không được dùng để chọn.
4. **Bỏ `w_resp`** (và tiêu chí G5): ở patch 7 thống kê này bão hoà theo cửa sổ và không đơn
   điệu theo kích thước blob. G4 = DoG + NWD.

**Hệ quả:** kết quả B0 cũ (patch 3) bị coi là không hợp lệ và phải train lại; không dùng
`head.pt` cũ. Mọi con số AP vẫn chưa tồn tại cho tới khi có run mới.

**Lưu ý về không gian pixel (ghi rõ để tránh nhầm khi đọc log):** các phép đo ở mục này
(14.1 px trung vị, 19.7% positive, ±6/±12 px) được đo trên **tile 640×512 ở tỉ lệ 1:1** —
cùng không gian với tiled eval. Training chạy **full-frame letterbox 1280** (hệ số ~0.667
với ảnh 1920 rộng), nên cùng một cặp có khoảng cách pixel nhỏ hơn ~1.5× trong không gian
train. Hệ quả: cửa sổ ±12 px (train) ≈ ±18 px (tile) — vẫn phủ trùm dải ≤10 px tile cần cho
eval, không có điểm mù; nhưng tỉ lệ positive trong log train **không** phải 19.7% và không
dùng để đối chiếu tiêu chí nào. Chỉ tiêu d_nn<10 px của G4 (mục 86) luôn được đo ở không
gian tile/eval.
