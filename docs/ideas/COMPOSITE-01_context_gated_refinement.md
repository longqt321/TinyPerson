# COMPOSITE-01 — CGLR: Gate ngữ cảnh, tinh chỉnh có điều kiện và xếp hạng chất lượng

## 1. Câu hỏi nghiên cứu

MSR/BTRM đã cho thấy một tín hiệu quan trọng: tăng evidence trên P2 có thể tăng
recall nhưng đồng thời làm background FP tăng. CGLR không cố làm evidence mạnh
hơn. Nó đặt evidence vào một chuỗi có kiểm soát:

```text
P2/P3/P4 → Context-Evidence Gate → candidate có tín hiệu sạch
        → Candidate Refinement → box có hình học tốt hơn
        → Quality Ranking → confidence phản ánh IoU của box cuối
```

Mỗi module có một nhiệm vụ khác nhau. Module sau không nhận đầu ra thô của module
trước mà nhận cả **sản phẩm** và **độ tin cậy** của sản phẩm đó.

## 2. Bằng chứng và dự đoán trước khi thiết kế

- BTRM trên official test giảm localization FP `39.121 → 37.684`, nhưng tăng
  background FP `84.359 → 89.045` và làm mất 146 TP. Vì vậy evidence thô là tín
  hiệu có ích nhưng không có precision đủ cao.
- Validation có 866 GT ở best IoU `0.10–0.50`; đây là nhóm còn khả năng sửa box.
- Trong 436 GT đạt `IoU>=0.75`, có 179 box confidence `<0.25`; sau khi sửa box,
  ranking vẫn có thể là nút thắt độc lập.
- P2 stride 4 tại `imgsz=1280`: một ô tương ứng khoảng 4 input px. Box tiny1
  `sqrt(area)∈[1,8)` px sau resize có median khoảng 4,5 px, tức khoảng 1,1 ô;
  khoảng 1/3 số tiny1 nằm trong tối đa 1 ô. Không thể dùng một gate toàn ảnh
  hoặc một refinement chạy trên mọi ô mà không khuếch đại nền.

**Dự đoán:** gate sẽ giảm background FP trước khi precision bị đưa vào
refinement; refinement cải thiện nhóm `0.10–0.50` hơn nhóm đã đạt `>=0.75`;
ranking chủ yếu tăng AP75 qua thứ tự score, không nhất thiết tăng recall.

## 3. Setting duy nhất được train

Chỉ train một cấu hình CGLR đầy đủ, không train C1/C2/C3 riêng.

- Backbone/neck: A2-P2 hiện tại; P2/P3/P4/P5 giữ nguyên.
- Dữ liệu: 662 ảnh train / 84 ảnh validation; không đổi split.
- Input: `imgsz=1280`, batch 8, mosaic 0, multi-scale 0; giữ nguyên split và
  augmentation baseline.
- Warm-start từ A2 `best.pt` để giữ nguyên khởi tạo Detect head; control là chính
  A2 `best.pt` đánh giá cùng evaluator.
- Fine-tune 20 epoch, AdamW, `lr0=2e-4`, warmup và weight decay baseline.
- Seed ban đầu: 42; không mở thêm setting khác trong vòng đầu.
- Gate dùng context P3/P4; refinement chạy trên top 300 candidate trước NMS, sau
  đó vẫn giữ `max_det=200` khi đánh giá.
- Cấu hình cố định: P2 gate 32 kênh, context bottleneck 32 kênh, ROI sampling
  thích nghi theo kích thước (P2 `3×3` cho box `<8 input px`, `5×5` cho box
  `>=8 input px`; P3 `3×3`), hidden MLP 128, top candidate 300, gate
  temperature 1, các loss weight cố định từ đầu.

Nếu setting này không qua ngưỡng, không được chọn lại từng hệ số để tạo kết quả
đẹp; giả thuyết CGLR bị bác bỏ ở cấp hệ thống.

## 4. Module 1 — Context-Evidence Gate

### Đầu vào

- `P2: [B,32,H/4,W/4]`, stride 4 và độ phân giải cao nhất.
- `P3: [B,64,H/8,W/8]`, `P4: [B,128,H/16,W/16]`, cung cấp ngữ cảnh rộng hơn.
- Evidence residual từ một nhánh MSR/BTRM nhẹ, cùng kích thước P2.

P3/P4 được upsample về kích thước P2 bằng nearest-neighbor, sau đó chiếu về 32
kênh. Không dùng P5 vì receptive field quá rộng và tín hiệu semantic có thể làm
mất chi tiết tiny1.

### Kiến trúc và receptive field

Gate gồm `1×1 conv → BatchNorm2d → SiLU → 3×3 depthwise conv → BatchNorm2d →
SiLU → 1×1 conv → sigmoid`. Các MLP dùng `LayerNorm` trước SiLU; lớp output cuối
vẫn zero-init. Nhánh context chỉ dùng để **điều chế** evidence, không thay P2 gốc:

```text
g = sigmoid(G([P2, up(P3), up(P4), evidence_context]))
P2_gate = P2 + α · g · E
```

`3×3` trên P2 nhìn vùng 12×12 input px; một lớp 3×3 trên P3 sau upsample bổ
sung 24 input px, còn P4 bổ sung 48 input px. Đây là **incremental RF**, cộng
thêm vào RF tích lũy của backbone/neck. Cân bằng cần thiết là: tiny1 cần local
detail, nhưng background/texture cần context để quyết định evidence có đáng tin
không. Không được viết “RF của module là 12–48 px” như thể đó là effective RF.

### Initialization và gradient

- Nhánh evidence scale `α=0.01` cố định; không đặt `α=0` vì khi đó gate không
  nhận gradient. Gate bias khởi tạo khoảng `-2.5` để `mean(g)≈0.05–0.1`; model
  bắt đầu gần A2, tránh phá checkpoint/head từ epoch đầu.
- `P2` đi qua skip connection không bị gate; gradient detector vẫn chảy về
  backbone ngay cả khi gate đóng.
- Auxiliary gate loss chỉ tác động vào gate: positive P2 gần tâm GT là target,
  background là negative. Không dùng toàn bộ ô trong GT làm positive vì box nhỏ
  dễ làm tỷ lệ positive/negative mất cân bằng.
- Lấy positive trong bán kính `1.5×sqrt(area)/stride`, tối đa 4 ô/GT; các ô còn
  lại là negative hoặc ignore nếu gần GT khác. Như vậy gradient không bị hàng
  nghìn background ô lấn át.

### Tín hiệu mong muốn truyền sang module 2

Module 1 không cần làm box chính xác. Nó phải truyền:

- Đầu ra: `P2_gate [B,32,H/4,W/4]`, `gate_map [B,1,H/4,W/4]` và
  `context_map [B,32,H/4,W/4]`.
- `P2_gate`: đặc trưng local đã giảm response nền;
- `gate_map`: xác suất evidence đáng tin tại từng ô;
- `context_map`: ngữ cảnh dùng để phân biệt texture nền với người nhỏ.

Nếu `mean(g)` tăng nhanh lên gần 1 ở toàn ảnh, gate đang học shortcut “bật mọi
evidence”; nếu giảm về 0, evidence không có giá trị hoặc auxiliary target sai.

**Tín hiệu học được mong muốn:** gate mở ở vùng có evidence tương quan với tâm
GT và đóng ở nền phẳng; không học một ngưỡng toàn ảnh.

## 5. Module 2 — Candidate-Conditioned Refinement

### Đầu vào và cách lấy mẫu

Detect A2 tạo candidate từ `P2_gate/P3/P4/P5`. Chọn top 300 theo objectness ×
class score trước NMS. Với mỗi candidate, chuyển box về tọa độ feature và lấy
mẫu bilinear `grid_sample` trên P2 cùng `3×3` điểm trên P3. Trên P2 dùng `3×3`
cho box `<8 input px` và `5×5` cho box `>=8 input px`; đây là quy tắc hình học
cố định theo kích thước candidate, không phải một nhánh học thêm. Lý do: `5×5`
trên P2 phủ 20×20 input px, trong khi tiny1 sau resize chỉ khoảng 0,33–5,3 px,
nên một cửa sổ cố định 5×5 sẽ pha quá nhiều nền vào feature của tiny1.

Đầu vào MLP gồm:

- pooled P2/P3 feature;
- `cx,cy,w,h` chuẩn hóa theo ảnh;
- level/stride embedding;
- `mean(gate)` và `max(gate)` trong box;
- objectness/class score thô.

Không lấy ROI từ P4 vì tiny box ở P4 có thể nhỏ hơn một ô; P4 đã được dùng gián
tiếp qua context gate.

### Đầu ra

- `Δt=[Δcx,Δcy,Δlogw,Δlogh]`;
- `log_sigma`, một độ bất định hình học;
- `refined_box = decode(box_raw, Δt)`.

Residual được giới hạn bằng `tanh` và scale tối đa 0.25 box size. Đây là giới
hạn an toàn: module chỉ sửa near-miss, không được dịch một background box vào
người thật một cách tùy ý.

### Initialization và gradient

- Lớp cuối của residual khởi tạo zero, nên `refined_box=box_raw` ở bước đầu.
- `log_sigma` khởi tạo bằng log độ lệch chuẩn trung bình theo size bin train;
  không khởi tạo quá nhỏ vì tiny annotation có jitter.
- Loss regression gồm CIoU trên refined box và Huber residual chuẩn hóa theo
  `sqrt(area)+1`. Candidate IoU `<0.1` không backprop refinement mạnh; candidate
  `0.1–0.75` nhận gradient đầy đủ; candidate `>=0.75` nhận trọng số nhỏ để tránh
  phá box đã tốt.

Mục tiêu là dồn gradient vào nhóm có thể sửa, thay vì để background FP chiếm
phần lớn loss. Với 300 candidate/ảnh nhưng chỉ khoảng vài chục positive, dùng
hard negative mining tỷ lệ tối đa 1:3; negative còn lại chỉ dùng cho ranking.

### Tín hiệu mong muốn truyền sang module 3

- box cuối có IoU cao hơn box thô;
- uncertainty thấp khi residual ổn định, cao ở vùng mơ hồ;
- pooled feature sau refinement vẫn giữ evidence gate.

Nếu IoU tăng nhưng uncertainty không tương quan với lỗi, module 3 không được dùng
uncertainty như confidence; nếu residual lớn ở candidate `IoU>=0.75`, module đang
phá box tốt.

**Tín hiệu học được mong muốn:** residual tương quan với sai số box thật, không
học một phép dịch scale toàn ảnh; uncertainty cao ở near-miss, thấp ở box tốt.

## 6. Module 3 — Quality Ranking Head

### Đầu vào và đầu ra

Nhận pooled feature, refined box, residual norm, `log_sigma`, `gate_map` và
objectness. Xuất `q_logit`, biểu diễn `q≈IoU(refined_box, GT)`.

```text
q = sigmoid(q_logit)
score_final = score_cls × q
```

`log_sigma` là **feature đầu vào** của quality head, không nhân thêm vào score.
Lý do: nhân thêm `exp(-0.5σ)` sẽ đếm cùng một tín hiệu bất định hai lần, làm
score của box đúng nhưng khó vẫn bị phạt quá mạnh; quality head phải học cách
dùng sigma nếu sigma thực sự có ích. Chỉ thay score, không thay box. NMS và
`max_det=200` giữ nguyên để AP75 phản ánh chất lượng ranking chứ không phải thay
đổi hậu xử lý.

### Target, gradient và calibration

- Target là IoU thật của **refined box**, stop-gradient qua target và qua box.
- Dùng SmoothL1 ở miền logit hoặc BCE mềm; box `IoU>=0.75` được oversample trong
  quality loss vì đây là nhóm AP75 đang thiếu.
- Positive/negative ranking pair lấy cùng ảnh: một box IoU cao phải đứng trên
  box background/localization thấp. Không tạo pair giữa hai người cạnh nhau nếu
  cả hai đều match GT khác nhau.
- Quality head khởi tạo `q_logit=0` (`q≈0.5`), không để score cuối sụp về 0 ở
  epoch đầu; sau warmup, loss weight tăng tuyến tính trong 5 epoch.

**Tín hiệu học được mong muốn:** `q` xếp hạng box theo IoU thật, không theo
confidence thô của classification.

## 7. Tương tác tối ưu và dự đoán hành vi

AdamW baseline có thể làm gate, refinement và ranking tranh gradient. Vì vậy
gradient được tách theo nhánh:

- backbone/P2 nhận loss detection chính + gate loss nhỏ;
- refinement nhận CIoU/residual loss trên candidate;
- ranking nhận quality loss, không truyền gradient ngược để thay box.

Trong 5–10 epoch đầu, dự đoán: gate vẫn thấp, box gần A2, ranking chưa ổn định;
AP75 có thể chưa tăng. Từ giữa training, kỳ vọng background FP giảm trước, sau
đó nhóm IoU `0.50–0.75` tăng, cuối cùng confidence của box `>=0.75` được nâng.

Failure mode dễ xảy ra nhất là gate lọc mất tiny1. Dấu hiệu là recall tiny1 giảm
trước khi AP75 thay đổi; khi đó CGLR thất bại vì gate học precision thay vì
precision–recall cân bằng.

## 8. Tiêu chí quyết định cho đúng một lần train

Đánh giá `best.pt` trên validation, không tune lại:

- AP75 tăng ít nhất `+0.002` so với A2;
- background FP không tăng; localization FP giảm;
- recall tiny1 không giảm quá 1 điểm phần trăm;
- số box tốt `IoU>=0.75` có confidence `<0.25` giảm;
- AP50 và AP không giảm quá `0.002`;
- latency tăng không quá 20%, VRAM không quá 20%.

Nếu không đạt, kết luận là chuỗi gate–refine–rank chưa chứng minh được lợi ích,
không chọn một module con có kết quả đẹp để thay thế kết luận hệ thống.

## 9. Claim được phép viết

Chỉ khi qua ba seed paired mới được viết: “CGLR dùng evidence có điều kiện, tinh
chỉnh candidate và quality ranking theo thứ tự để tách ba lỗi background FP,
near-miss localization và mis-ranking.” Không được gọi MSR, refinement hay
IoU-aware ranking riêng lẻ là đóng góp mới.

## 10. Tham khảo tư duy thiết kế

- ATSS (Zhang et al., 2020): assignment theo thống kê từng GT, không dùng một
  ngưỡng IoU cố định cho mọi ảnh.
- FCOS (Tian et al., 2019) và GFL (Li et al., 2020): tách chất lượng localization
  khỏi classification; quality phải là tín hiệu học được, không phải confidence thô.
- IoU-Net (Jiang et al., 2018): dự đoán IoU như một tín hiệu ranking riêng.
- Deformable DETR (Zhu et al., 2021) và DCNv2 (Zhu et al., 2019): sampling thưa
  và offset học được giúp tập trung vào vùng quan trọng.
- Focal loss (Lin et al., 2017): xử lý mất cân bằng positive/negative; dùng cho
  gate heatmap thay vì BCE đều.

## 11. Đặc tả kỹ thuật chi tiết và pre-registration

Phần này khóa các quyết định trước khi train. Nếu một con số không được nêu ở
đây, nó không được tự do đổi trong lần chạy duy nhất.

### 11.1. Hợp đồng tensor toàn hệ thống

Ở `imgsz=1280`, ảnh vuông sau letterbox có P2 `320×320`, P3 `160×160`, P4
`80×80`, P5 `40×40`. Với ảnh chữ nhật, thay `320` bằng `H/4`, `W/4`.

| Tensor | Shape | Ý nghĩa | Bất biến |
|---|---|---|---|
| `P2` | `[B,32,H/4,W/4]` | feature stride 4 | kênh và độ phân giải không đổi |
| `P3` | `[B,64,H/8,W/8]` | context stride 8 | chỉ dùng qua phép chiếu 32 kênh |
| `P4` | `[B,128,H/16,W/16]` | context stride 16 | không thay P2 |
| `E` | `[B,32,H/4,W/4]` | evidence residual | cùng hệ tọa độ P2 |
| `g` | `[B,1,H/4,W/4]` | xác suất evidence đáng tin | `[0,1]`, sigmoid |
| `P2_gate` | `[B,32,H/4,W/4]` | P2 đã điều chế | gần P2 khi gate đóng |
| `Δbox` | `[B,N,4]` | residual `Δcx,Δcy,Δlogw,Δlogh` | bị chặn bởi `tanh` |
| `log_sigma` | `[B,N,1]` | bất định hình học | float32, clamp trước khi dùng |
| `q` | `[B,N,1]` | IoU dự đoán | `[0,1]`, sigmoid |

`N=300` candidate/ảnh. Tọa độ box trong loss là input px; khi báo cáo phải đổi
về source px. Không trộn hai đơn vị trong cùng một biểu thức.

### 11.2. Giới hạn thông tin và hình học lấy mẫu

- Train có 14.825 GT/662 ảnh, trung bình 22,39 GT/ảnh, median 6, max 176; 53 ảnh
  rỗng. Validation có 3.608 GT/84 ảnh, trung bình 42,95, median 18, max 177; 5
  ảnh rỗng. Validation đông hơn train khoảng 1,9 lần.
- Ảnh 1920×1080 co khoảng `2/3` khi vào mạng; ảnh 4K co khoảng `1/3`. Tiny1
  nguồn 1–8 px thành khoảng 0,33–5,3 input px. P2 stride 4, nên phần lớn tiny1
  chỉ chiếm khoảng **1–1,5 ô P2**; khoảng 1/3 nằm trong tối đa 1 ô.
- Vì vậy box head phải hồi quy liên tục; assignment chỉ là điểm neo. Không được
  thiết kế gate/refinement dựa trên giả định “mỗi người có một ô P2”.
- ROI `3×3` trên P2 phủ 12×12 input px; `5×5` phủ 20×20. Với tiny1, `5×5` có
  thể chứa >90% nền. Quy tắc chọn `3×3`/`5×5` theo kích thước là bắt buộc, không
  phải tùy chọn.
- P3 `3×3` phủ 24×24 input px. P4 không lấy trực tiếp vì một box tiny1 có thể
  nhỏ hơn một ô P4; P4 chỉ vào gate như context.

### 11.3. Module 1: tín hiệu, khởi tạo và gradient

**Tín hiệu học được.** Gate phải học “evidence này có tương quan với một người
thật ở gần đây không”, không học “ảnh này có người”. Đầu vào có P2 local, P3/P4
context và E; đầu ra là một bản đồ xác suất theo ô.

**Khởi tạo bằng số.**

```text
g = sigmoid(logit_g),  logit_g = b + residual_g
b = -2.5                => sigmoid(b) = 0.076
α = 0.01                => P2_gate = P2 + tối đa 1%·E khi g=1
```

Lớp `1×1` cuối của gate zero-init; bias `-2.5`. Ở step 0, gate gần như đóng,
`P2_gate≈P2`, nên checkpoint A2 không bị phá. Gradient tới các lớp trước của
gate bằng 0 ở step đầu, nhưng gradient tới lớp cuối khác 0; sau vài step lớp
cuối mở đường cho các lớp trước. Không dùng `α=0` vì sẽ triệt tiêu gradient của
toàn bộ nhánh evidence.

**Auxiliary loss cho gate.** Positive là tối đa 4 ô P2 gần tâm GT trong bán kính
`1.5×sqrt(area)/stride`; các ô gần GT khác được ignore; còn lại là negative.
Với 22,39 GT/ảnh trung bình, tối đa khoảng 90 positive trên 102.400 ô P2
(`0,09%`). Dùng focal-style loss, chuẩn hóa theo số positive, không dùng BCE
trung bình toàn ảnh. Nếu không, 99,9% negative sẽ dạy gate đóng vĩnh viễn.

**Gradient budget.**

```text
∂L/∂E  = α·g·∂L/∂P2_gate
∂L/∂g  = α·(E·∂L/∂P2_gate)
∂L/∂P2 = ∂L/∂P2_gate                 # skip connection
```

Với `α=0.01`, gradient từ detection loss vào E nhỏ hơn 100 lần so với đường
skip. Vì vậy gate phải có auxiliary loss; nếu chỉ dựa vào detection loss, gate
gần như không học trong 1.660 step.

**Chẩn đoán bắt buộc.** Log `mean(g)`, p10/p50/p90, tỷ lệ `g>0.9`, tỷ lệ
`g<0.1`, và AUPRC của `g` so với tâm GT. Kỳ vọng: `mean(g)` tăng từ khoảng 0,08
lên 0,2–0,5 ở vùng object, không tăng toàn ảnh. Nếu `mean(g)>0.5` sau epoch 3
hoặc `<0.01` sau epoch 5, gate thất bại.

### 11.4. Module 2: candidate, target và gradient

**Chọn candidate.** Top 300 theo `objectness × class_score` trước NMS. Với
22–43 GT/ảnh, tỷ lệ positive/negative xấp xỉ 1:7 đến 1:13 nếu chỉ lấy một
candidate/GT; hard-negative mining tối đa 1:3 giữ loss không bị nền chi phối.

**Ghép candidate với GT.** Mỗi candidate lấy GT có IoU lớn nhất. Candidate
`IoU<0.10` không nhận box loss mạnh; `0.10–0.75` nhận đầy đủ; `>=0.75` nhận
trọng số nhỏ. Không lấy trung bình nhiều GT cho một box; nếu hai GT chồng nhau,
chọn GT có IoU cao hơn và đánh dấu ambiguous để loại khỏi loss.

**Đầu ra refinement.** `Δbox` bị chặn bởi `tanh` và `0.25×box_size`. Lớp cuối
zero-init để `refined_box=box_raw` ở step 0. `log_sigma` khởi tạo bằng log độ
lệch chuẩn trung bình theo size bin **train**, không dùng validation để chọn.
Sigma dùng cho NLL/feature của quality head, không nhân trực tiếp vào score.

**Gradient.** Box loss chỉ chảy vào refinement và box head; quality loss không
được chảy ngược vào box. Nếu muốn box học từ quality, đó là một loss khác và phải
được khai báo riêng. ROI feature của quality head được detach khỏi box regression
nếu cần tránh hai mục tiêu kéo feature ngược nhau.

**Chẩn đoán.** Log phân bố IoU trước/sau refinement theo band `0.10–0.50`,
`0.50–0.75`, `>=0.75`; residual norm theo size bin; tỷ lệ box `>=0.75` bị dịch
quá 0.1 box size. Kỳ vọng: band `0.10–0.50` tăng IoU, band `>=0.75` gần như
không đổi, background box không được “refine” thành người.

### 11.5. Module 3: quality ranking và calibration

- `q_logit=0` lúc init, `q=0.5`; score bị giảm đều 2 lần nhưng thứ tự NMS không
  đổi. Khi q học, box tốt mới được đẩy lên.
- Target `q* = IoU(refined_box, GT)`; stop-gradient qua box và target.
- Loss SmoothL1 ở miền logit hoặc BCE mềm. Oversample box `IoU>=0.75` trong
  quality loss, nhưng không oversample quá 3 lần để tránh q chỉ học nhóm dễ.
- Không tạo cặp ranking giữa hai GT khác nhau. Chỉ tạo cặp positive-negative
  trong cùng ảnh khi negative là background/localization FP, không phải một
  người thật khác.
- Đo ECE/reliability diagram trên validation; q phải tăng đơn điệu theo IoU.
  Nếu q cao cho background hoặc q gần như hằng số, quality head vô dụng.

### 11.6. Optimizer, lịch train và số học

- AdamW, `lr0=2e-4` cho toàn bộ tham số, weight decay và warmup baseline; không
  chia parameter group riêng trong lần chạy duy nhất.
- 20 epoch × 83 iteration = 1.660 step. Warmup 2 epoch; gradient clipping theo
  baseline; EMA giữ baseline.
- Sigma/NLL và phép log/exp tính ở float32; clamp sigma và offset trước khi dùng;
  kiểm tra NaN/Inf trong 20 step đầu.
- Ngân sách tham số mới: gate khoảng 5–6k, refinement khoảng 180–200k, quality
  khoảng 4–5k; tổng dưới 10% baseline 2,52M. Nếu vượt, giảm hidden width trước
  khi tăng dữ liệu.
- Latency: ROI sampling là nút thắt chính. Ngưỡng chấp nhận tăng không quá 20%
  ở batch 1, 1280; nếu vượt, giảm top candidate hoặc số điểm sample, không tăng
  thêm module.

### 11.7. Dự đoán trước train và điều kiện bác bỏ

| Thời điểm | Dự đoán | Nếu sai |
|---|---|---|
| Step 0 | `P2_gate≈P2`, box gần A2, q=0,5 | output lệch >1% hoặc NaN |
| Epoch 1–5 | gate mean tăng chậm; refinement residual nhỏ | gate bão hòa hoặc box drift |
| Epoch 5–10 | background FP giảm; AP50 giữ | AP50 giảm >0.002 |
| Epoch 10–20 | near-miss `0.10–0.50` tăng IoU; AP75 tăng | chỉ recall tăng, AP75 không tăng |
| Kết thúc | AP75 +0.002; background FP không tăng; tiny1 recall giảm <1 điểm | bác bỏ toàn bộ CGLR |

Không được chọn riêng gate, refinement hay quality head để tạo một kết quả
dương tính sau khi hệ thống thất bại.
