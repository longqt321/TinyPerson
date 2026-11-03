# YOLO26 trong repo này: những gì thực sự có trong code (và các hiểu lầm cần sửa)

> Mục đích: rà soát lại toàn bộ YOLO26 theo **code thực thi** (Ultralytics 8.4.164 đã cài trong `.venv`, repo `thigiacmaytinh`), sửa các hiểu lầm đã xuất hiện trong các tài liệu trước (QAC v3/v4, adaptive proposal), và nêu hệ quả cho thiết kế QAC.
> Phạm vi: architecture, head, loss, assigner, DFL, bbox loss, inference/postprocess, optimizer, và các khác biệt của repo (P2, reg_max, QAC hooks).

---

## 0. Tóm tắt các hiểu lầm đã sửa (đọc phần này trước)

| # | Hiểu lầm trước đây | Sự thật trong code |
|---|---|---|
| 1 | "IoU = 0 khi không overlap → không có gradient" | YOLO dùng **CIoU** (`bbox_iou(..., CIoU=True)`) ở **cả** bbox loss (`loss.py:134`) **và** assigner (`tal.py:246`), có center-distance penalty → gradient tồn tại khi không overlap. |
| 2 | "NWD giải quyết vấn đề gradient" | Sai. NWD (nếu dùng) chỉ là **làm mượt target** (variance reduction) cho tiny object, không phải gradient. |
| 3 | "QAC v3 patch NWD vào assigner cứu tiny off-grid" | Yếu hơn tuyên bố, vì CIoU base đã xử lý non-overlap. Cần ablation "bỏ patch" để xác nhận. |
| 4 | "YOLO26 có STAL / ProgLoss" | **Không có** trong code 8.4.164 (grep toàn package: không có STAL, không có ProgLoss). Đây là claim marketing/tài liệu, không có trong installed build. |
| 5 | "reg_max=16 là DFL chuẩn của YOLO26" | Vanilla YOLO26 dùng **reg_max=1** (không DFL). Repo **tự re-enable DFL** (reg_max=16 baseline, 8 cho SADR) trên backbone YOLO26. |
| 6 | "one2one head là bản sao đơn giản" | one2one branch **detach khỏi backbone** trong training (`x_detach = [xi.detach()...]`); backbone không nhận gradient từ nhánh này. |
| 7 | "quality head trên one2one được supervision tự nhiên" | Vì detach, quality head trên one2one **không** nhận gradient backbone; phải supervision riêng (QAC v3 đã làm đúng việc này). |
| 8 | "Top-k candidate selection là NMS" | end2end inference **không có NMS**; `postprocess` chỉ làm top-k (max_det=300) trên one2one output. |
| 9 | "IoU target cho quality head không ổn định vì gradient" | Sai về gradient; vấn đề thật là **target variance** (IoU dốc cho tiny), không phải gradient. |
| 10 | "target_scores_sum normalization bảo toàn gradient norm" | Đúng là chuẩn hóa, nhưng cần hiểu: cls loss chia cho `target_scores_sum` (tổng target TAL), dfl loss chia cho cùng tổng. QAC v3 "preserve base_target_scores_sum" là đúng cơ chế. |

---

## 1. YOLO26 là gì trong repo này

- **Version**: Ultralytics 8.4.164 (`.venv/.../ultralytics/__init__.py`).
- **Kiến trúc base**: `yolo26.yaml` — backbone Conv / C3k2 / SPPF / C2PSA; head P3-P5 (stride 8/16/32); Detect với `end2end: True`, **`reg_max: 1`** (không DFL).
- **Repo config** (`configs/model/yolo26n_tiny_A2_p2_qac_v3.yaml`): dựa trên YOLO26n nhưng **thêm P2** (stride 4, 4 levels), `reg_max: 16`, `end2end: True`, `tal_beta: 6.0`. Tức là repo **dùng backbone YOLO26 + DFL re-enabled + P2 + dual end2end head**. Đây không phải "YOLO26 vanilla".
- **QAC hooks** (`src/tinydet/models/ultralytics.py`): sau khi Ultralytics build `Detect`, hook thay `model[-1]` bằng `QACDetectV3` (from_detect, strict=False), và `init_criterion` trả về `QualityAwareE2ELossV3`.

### 1.1. Detect head (`ultralytics/nn/modules/head.py`)

- `cv2`: box branch, output `4 * reg_max` channels/level.
- `cv3`: cls branch, output `nc` channels/level (DWConv-based, không legacy).
- `dfl = DFL(reg_max)` nếu `reg_max > 1` else `nn.Identity()`.
- Nếu `end2end=True`: `one2one_cv2`, `one2one_cv3` là `deepcopy` của cv2/cv3.
- `forward`: train → trả dict `{"one2many": preds, "one2one": one2one}` (one2one trên `x.detach()`). Inference → dùng one2one nếu `end2end` (NMS-free) else one2many.
- `max_det = 300`.

### 1.2. Dual head detach (quan trọng)

```python
x_detach = [xi.detach() for xi in x] if self.training else x
one2one = self.forward_head(x_detach, **self.one2one)
```
→ Backbone **không** nhận gradient từ one2one branch. QAC quality head trên one2one (nếu có) chỉ nhận gradient từ loss của one2one branch, không từ backbone. Đây là lý do QAC v3 phải có QSH riêng cho one2one và supervision riêng.

### 1.3. `fuse()` (inference)

```python
def fuse(self):
    end2end = self.end2end
    for name in tuple(self._modules):
        if name.startswith("one2one_"):
            setattr(self, name[8:] if end2end else name, None)
```
→ Sau fuse, nếu end2end=True, `cv2/cv3` (one2many) bị set None, chỉ giữ one2one. **Module QAC gắn vào one2many branch sẽ bị loại bỏ khi fuse** nếu không fuse đúng. Đây là chi tiết deploy cần lưu ý.

---

## 2. Loss

### 2.1. `E2ELoss` (`ultralytics/utils/loss.py:1322`)

- `one2many = v8DetectionLoss(model, tal_topk=10)`
- `one2one = v8DetectionLoss(model, tal_topk=7, tal_topk2=1)`
- Tổng loss = `0.8 * loss_o2m + 0.2 * loss_o2o` ban đầu, decay theo epoch: `o2m` giảm từ 0.8 → 0.1 (final), `o2o` tăng từ 0.2 → 0.9. `decay(x) = max(1 - x/(epochs-1), 0)*(0.8-0.1) + 0.1`.
- Loss items trả về từ **one2one** branch (để logging).
- `update()` được gọi mỗi epoch (trainer).

### 2.2. `v8DetectionLoss` (`loss.py:347`)

- `hyp = model.args` (hyperparameters từ yaml: box, cls, dfl, epochs, ...).
- `assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=nc, alpha=0.5, beta=6.0, stride=strides, topk2=tal_topk2)`.
- `bbox_loss = BboxLoss(reg_max)`.
- `use_dfl = reg_max > 1`.
- `loss_names = ("box_loss", "cls_loss", "dfl_loss")` hoặc `("box_loss","cls_loss","l1_loss")` nếu reg_max=1.

### 2.3. `get_assigned_targets_and_loss` (cơ chế)

1. `pred_distri = preds["boxes"]` (B, 4*reg_max, N) → permute (B, N, 4*reg_max).
2. `pred_scores` (B, nc, N) → permute.
3. `anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)` → anchor centers và stride per anchor.
4. Targets preprocess: pad vào (B, max_gt, 5) [cls, xyxy pixel].
5. `pred_bboxes = bbox_decode(anchor_points, pred_distri)` (xyxy, normalized by stride).
6. Assigner → `target_bboxes, target_scores, fg_mask, target_gt_idx`.
7. `target_scores_sum = target_scores.sum().clamp_(min=1)`.
8. **Cls loss**: `BCE(pred_scores, target_scores)` sum / `target_scores_sum`.
9. **Bbox loss**: `BboxLoss(...)` (CIoU + DFL), weighted by `target_scores`, normalized by `target_scores_sum`.
10. Nhân gain: `loss[0]*=hyp.box`, `loss[1]*=hyp.cls`, `loss[2]*=hyp.dfl`.

**Điểm quan trọng**: target_scores là **soft** (TAL alignment metric `cls^alpha * IoU^beta`), không phải 0/1. Loss được chuẩn hóa theo tổng target_scores, nên adding more positives increases denominator — QAC v3 "preserve base_target_scores_sum" là để không pha loãng gradient khi peak-lift target.

### 2.4. `BboxLoss` (`loss.py:111`)

```python
iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
loss_iou = 1.0 - iou  # (N_fg,)
```
- Nếu `use_dfl` (reg_max>1): `dfl_loss = DFLoss(pred_dist, target)` — target là khoảng cách từ anchor đến box edge (continuous, in bin units), loss là cross-entropy giữa 2 bin kề nhau.
- Weighted by `target_scores[fg_mask]` (per-anchor soft weight).
- Normalized by `target_scores_sum`.

**CIoU** = IoU − ρ²(b,b_gt)/c² − αv (center distance penalty + aspect ratio). Có gradient ngay cả khi IoU=0 (vì center distance term). **Đây là lý do "IoU=0 → no gradient" là sai.**

### 2.5. `DFLoss` (`loss.py:89`)

```python
def __call__(self, pred_dist, target):
    # target: continuous distance in [0, reg_max-1]
    # pred_dist: (N, 4, reg_max) logits
    target = target.clamp_(0, reg_max - 1 - 0.01)  # avoid boundary
    tl = target.floor().long()  # left bin
    tr = tl + 1  # right bin
    wl = tr - target  # weight left
    wr = target - tl  # weight right
    # CE(pred, tl) * wl + CE(pred, tr) * wr
```
→ DFL là **cross-entropy giữa 2 bin kề nhau**, weighted by khoảng cách. Đây là lý do reg_max=16 "bóp cụt" ở P2/P3 cho box rộng: target distance có thể > reg_max-1 → clamp → mất gradient cho phần vượt. SADR8 (reg_max=8) giảm vấn đề này và cải thiện recall.

---

## 3. Assigner (`ultralytics/utils/tal.py`)

### 3.1. `TaskAlignedAssigner`

- `alpha=0.5` (cls exponent), `beta=6.0` (IoU exponent) trong metric `align = cls^alpha * IoU^beta`.
- `topk=10` (o2m) / `topk=7` (o2o), `topk2=1` (o2o: secondary filter).
- `stride_val = stride[1]` (stride P3, thường 8) hoặc stride[0] nếu chỉ 1 level.

### 3.2. `select_candidates_in_gts` — xử lý tiny object có sẵn

```python
wh_mask = gt_bboxes_xywh[..., 2:] < stride_val
gt_bboxes_xywh[..., 2:] = where(wh_mask, stride_val, original)
```
→ **GT box nhỏ hơn stride_val được phóng to đến stride_val** khi chọn candidate anchors. Đây là xử lý tiny object **có sẵn trong YOLO**. Điều này quan trọng: phần lớn "tiny rescue" đã được làm ở assigner; QAC/NWD patch có thể redundant.

### 3.3. `iou_calculation` — CIoU

```python
return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
```
→ CIoU, clamp ≥0. **Không phải IoU thuần túy.**

### 3.4. `topk2` (o2o)

Secondary filter: sau topk candidates, chỉ giữ top-1 theo align metric. Đây là lý do one2one head output ≤300 predictions và NMS-free.

---

## 4. Inference & postprocess (end2end)

- `forward` (inference): dùng one2one branch (nếu end2end) → `_inference` → decode boxes → `postprocess`.
- `postprocess`: split boxes/scores → `get_topk_index(scores, max_det=300)` → gather. **Không có NMS, không có conf threshold ở đây.** Conf threshold lọc sau ở predictor pipeline.
- `get_topk_index`: top-k theo score (grouped topk optimization). `max_det=300`.
- **Sau fuse()**: one2many branch bị xóa → model chỉ còn one2one. QAC module gắn vào one2many sẽ bị loại bỏ nếu không handle fuse.

---

## 5. Optimizer

- `optimizer: auto` (default.yaml) → trainer chọn: nếu `iterations > 10000` → MuSGD (lr 0.01, momentum 0.9) với Muon (matrix params) + SGD (scalar params); else AdamW (lr_fit, 0.9).
- MuSGD là optimizer mới trong YOLO26. Repo train với `optimizer: AdamW` (exp yaml) → không dùng MuSGD. Đây là một biến chưa khai thác.

---

## 6. Các claim "STAL / ProgLoss" — không có trong code

- Grep toàn package 8.4.164: không có "STAL", "small target aware", "ProgLoss", "progressive loss". 
- → Các claim này là marketing/docs, **không** có implementation trong installed build. Không nên dựa thiết kế QAC vào chúng. Nếu muốn dùng, phải tự implement.

---

## 7. Hệ quả cho QAC (đã điều chỉnh trong QAC_V4_DESIGN.md)

1. **IoU target cho QSH là đúng** (AP đo IoU), và không có vấn đề gradient vì CIoU. NWD auxiliary chỉ là smoothing/variance.
2. **QAC v3 NWD assigner patch** cần ablation "bỏ patch" để xác nhận giá trị.
3. **one2one branch detach** → QSH trên one2one phải supervision riêng và eval đúng branch sẽ deploy.
4. **Tiny handling có sẵn** (`select_candidates_in_gts` phóng to GT < stride_val) → QAC không nên duplicate; nên đo xem QAC thêm gì ngoài cái có sẵn.
5. **fuse()** → QAC module trên one2many bị loại bỏ khi fuse; cần ensure QAC cũng chạy trên one2one branch hoặc handle fuse.
6. **MuSGD** chưa khai thác — có thể thử trong ablation (optimizer change, không phải mechanism).
7. **DFL re-enabled** (reg_max=16/8) trên YOLO26 backbone → SADR8 (reg_max=8) là regression improvement có bằng chứng; giữ.

---

## 8. Tham chiếu code (file:line)

- `ultralytics/nn/modules/head.py:100-120` Detect __init__ (cv2/cv3/dfl/one2one)
- `ultralytics/nn/modules/head.py:122-135` forward (detach one2one)
- `ultralytics/nn/modules/head.py:249-270` postprocess / get_topk_index
- `ultralytics/nn/modules/head.py:219-228` fuse
- `ultralytics/utils/loss.py:89-110` DFLoss
- `ultralytics/utils/loss.py:111-158` BboxLoss (CIoU)
- `ultralytics/utils/loss.py:347-480` v8DetectionLoss
- `ultralytics/utils/loss.py:1322-1350` E2ELoss (o2m/o2o weights, decay)
- `ultralytics/utils/tal.py:1-60` TaskAlignedAssigner __init__ (alpha/beta/stride_val)
- `ultralytics/utils/tal.py:236-247` iou_calculation (CIoU)
- `ultralytics/utils/tal.py:317-345` select_candidates_in_gts (tiny GT enlargement)
- `ultralytics/engine/trainer.py:1160-1230` optimizer selection (MuSGD/AdamW)
- `src/tinydet/models/ultralytics.py` QAC hooks
- `src/tinydet/models/registry.py` build_model

---

## Kết luận

Tài liệu này sửa các hiểu lầm kỹ thuật đã xuất hiện trong QAC v3/v4/adaptive proposal. Từ đây, thiết kế QAC nên dựa trên **code thực thi** (CIoU, detach one2one, fuse, tiny handling có sẵn, DFL cross-entropy mechanics), không dựa trên claim marketing (STAL/ProgLoss) hay giả định sai về gradient. QAC_V4_DESIGN.md v2 đã được cập nhật theo correction về CIoU (§1.1, §2.2).
