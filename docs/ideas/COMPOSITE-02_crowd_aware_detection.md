# COMPOSITE-02 — CASS: Định tuyến theo mật độ, gán nhãn đám đông và chấm điểm chống trùng lặp

## 1. Câu hỏi nghiên cứu

TinyPerson có hai lỗi tách biệt: người rất nhỏ bị bỏ sót, và trong ảnh đông có
nhiều box cùng giải thích một người hoặc một box phải gánh nhiều người. CASS
không chỉ tăng recall; nó thay đổi cách hệ thống phân bổ **trách nhiệm** giữa
candidate và GT:

```text
P2/P3/P4 → Density Routing → candidate có context phù hợp mật độ
        → Crowd-Aware Soft Assignment → mỗi GT có candidate riêng
        → Duplicate-Aware Set Scoring → score phản ánh trách nhiệm, không chỉ objectness
```

## 2. Bằng chứng và dự đoán trước khi thiết kế

- 90,7% tổng miss nằm trong ảnh crowded/very-crowded; 627/786 ảnh chạm
  `max_det=200`.
- 133.941 FP trên official test; trong đó 10.461 là duplicate FP và 39.121 là
  localization FP. Duplicate không thể giải quyết chỉ bằng tăng ngưỡng NMS.
- Tiny1 recall rất thấp, đặc biệt trong ảnh rất đông; hard assignment dễ để một
  candidate đại diện cho nhiều người gần nhau.
- P2 stride 4 tại `imgsz=1280`; một cửa sổ 16×16 px tương đương 4×4 ô P2. Đây là
  đơn vị hợp lý để đo mật độ cục bộ, không dùng mật độ toàn ảnh.
- `max_det=200` và top-k candidate/GT tạo áp lực trực tiếp: nếu mỗi GT lấy 16
  candidate, ảnh có hơn 12 GT đã có thể chồng lấn trách nhiệm.

**Dự đoán:** routing sẽ giúp ích chủ yếu ở ảnh đông; soft assignment giảm miss
nhưng có thể tăng duplicate nếu không chuẩn hóa hai chiều; set scoring phải giảm
duplicate trước khi AP75 tăng. Nếu chỉ recall tăng mà AP75 không tăng, chuỗi này
thất bại.

## 3. Setting duy nhất được train

Chỉ train một cấu hình CASS đầy đủ; không train từng module riêng.

- Warm-start từ A2 `best.pt` để giữ nguyên khởi tạo Detect head và giảm chi phí;
  control là chính A2 `best.pt` đánh giá cùng evaluator.
- Dữ liệu: 662 ảnh train / 84 ảnh validation; không đổi split.
- Fine-tune 20 epoch, `lr0=2e-4`, AdamW, weight decay baseline, batch 8,
  `imgsz=1280`, mosaic 0, multi-scale 0, seed 42.
- Density head: 1 lớp 3×3 trên P3, upsample về P2, hidden 32.
- Routing: context từ P3 upsample, `β=0.1`, bias khởi tạo -4.
- Assignment: top-k 16, temperature 0.5, prior IoU–center–neighbor, MLP hidden 32.
- Set scoring: pairwise MLP hidden 32, chỉ xét cặp có IoU>0.3, bias duplicate -4.
- Giữ `max_det=200`, NMS và confidence inference như baseline.

Không có bảng ablation. Nếu setting đầy đủ không đạt, không thử một biến thể
khác của cùng ý tưởng trong cùng ngân sách.

## 4. Module 1 — Density-Conditioned Routing

### Đầu vào và target

- Đầu vào: P2 `[B,32,H/4,W/4]`, P3 `[B,64,H/8,W/8]`, P4 `[B,128,H/16,W/16]`.
- Density head chạy trên P3/P4 để có receptive field lớn hơn, rồi upsample về P2.
- Target mật độ tại mỗi ô P2: số tâm GT trong cửa sổ 4×4 ô P2 (16×16 px ở ảnh
  resized), chuẩn hóa `min(count,4)/4`. Đây là target mềm, không phải nhãn
  crowded toàn ảnh.
- Các ô không có GT và xa người là negative; ô gần biên ảnh hoặc gần GT khác
  được ignore để tránh target sai do letterbox.

### Kiến trúc, receptive field và gating

```text
d = sigmoid(MLP_density(up(P3), up(P4)))
C3 = Conv1x1(up(P3))                  # 64 → 32 kênh, cùng hệ tọa độ P2
F_route = P2 + β · d · (C3 - P2)
```

Mọi conv ẩn dùng `Conv2d → BatchNorm2d → SiLU`; lớp dự đoán cuối giữ zero-init.

- `d` là bản đồ mật độ `[B,1,H/4,W/4]`; `β=0.1` cố định để routing bắt đầu như
  một nhiễu loạn nhỏ quanh P2.
- Không dùng `d` để nhân toàn bộ feature; chỉ dùng để trộn context P3. Như vậy
  vùng thưa vẫn giữ P2 sắc nét, vùng đông nhận thêm context.
- Receptive field của nhánh density trên P3/P4 khoảng 24–96 px, đủ để phân biệt
  cụm người với một người đơn lẻ; P2 vẫn giữ vị trí chính xác.
- `C3` phải được chiếu về 32 kênh trước khi trừ P2; nếu trừ trực tiếp feature 64
  kênh với 32 kênh thì biểu thức không hợp lệ.
- `d` bias khởi tạo -4, nên lúc đầu `d≈0.018`, `F_route≈P2`. Gradient vào `d`
  không bị triệt tiêu vì `(C3-P2)` khác 0.

### Tín hiệu mong muốn truyền sang module 2

Module 1 phải truyền `F_route` và `d`:

- Đầu ra: `F_route [B,32,H/4,W/4]` và `d [B,1,H/4,W/4]`.
- `F_route` giữ chi tiết P2 nhưng có context ở vùng đông;
- `d` cho assignment biết vùng nào cần nhiều candidate hơn;
- không truyền một scalar mật độ toàn ảnh, vì sẽ làm mất khác biệt giữa người
  đứng riêng và cụm người trong cùng ảnh.

**Tín hiệu học được mong muốn:** `d` cao ở cụm người và thấp ở nền/vùng thưa;
không học “ảnh này crowded” thành một scalar toàn ảnh.

## 5. Module 2 — Crowd-Aware Soft Assignment

### Đầu vào

- `F_route` tại P2/P3;
- GT boxes và candidate boxes từ Detect;
- `d` tại tâm candidate;
- khoảng cách tới GT gần nhất và GT thứ hai.

Với mỗi GT, lấy top-k 16 candidate gần nhất trong bán kính
`max(2 ô, 2×sqrt(area)/stride)`. Không lấy toàn bộ candidate vì ảnh đông có thể
có 200 box và chi phí assignment sẽ tăng bậc hai.

### Cơ chế trách nhiệm

```text
prior_ij = IoU_ij - λ·(center_distance_ij / sqrt(area_i))
           - μ·neighbor_distance_ij
logit_ij = prior_ij + MLP(features_ij)
r_ij = softmax_j(logit_ij / τ)          # theo candidate cho từng GT
r_ij = Sinkhorn1(r_ij)                  # một vòng chuẩn hóa hai chiều
```

- `τ=0.5` làm phân bố sắc hơn; nếu để `τ=1`, responsibility gần uniform và box
  loss bị trung bình hóa quá mức.
- MLP cuối khởi tạo zero, nên lúc đầu assignment xấp xỉ prior hình học. Đây là
  điểm quan trọng: module học phần hiệu chỉnh, không học lại từ đầu.
- `Sinkhorn1` là một vòng chuẩn hóa xen kẽ: chuẩn hóa theo candidate để mỗi GT
  có tổng trách nhiệm 1, rồi chuẩn hóa theo GT để mỗi candidate không nhận quá
  nhiều GT. Không dùng `r/max(1,Σr)` vì phép chia đó không bảo toàn tổng của
  từng GT và có thể làm GT bị đói gradient.

### Initialization và gradient

- Density head bias khởi tạo -4, để routing bắt đầu gần P2; MLP cuối của
  assignment khởi tạo zero, để `r_ij` bắt đầu từ prior hình học thay vì ngẫu nhiên.
- Set-scoring bias khởi tạo -4, để duplicate penalty gần 0 ở epoch đầu.
- Gradient của density/routing bị warmup 2 epoch; gradient của assignment được
  normalize theo số GT để ảnh đông không chi phối batch.
- Không dùng stop-gradient giữa assignment và set scoring, vì module 3 cần học
  trực tiếp từ trách nhiệm; nhưng không cho set scoring sửa box.

### Loss và cân bằng dương/âm

- Classification dùng BCE mềm với target `r_ij`, cộng entropy nhẹ để tránh mọi
  candidate đều nhận 0.
- Box regression không lấy trung bình nhiều GT: mỗi candidate chọn GT có
  `r_ij` lớn nhất, rồi dùng CIoU + residual với trọng số `max_i r_ij`. Nếu lấy
  trung bình box của hai GT gần nhau, box target sẽ rơi vào khoảng trống giữa
  hai người và dạy model một vị trí sai.
- Mỗi GT được chuẩn hóa để tổng trọng số box bằng 1, tránh ảnh nhiều người chi
  phối loss.
- Hard negative: lấy tối đa 3 candidate background gần mỗi GT; không dùng toàn
  bộ background vì 200 candidate/ảnh sẽ làm positive/negative lệch mạnh.
- Ràng buộc: mỗi GT phải có ít nhất một candidate với `r>0.5`; nếu không, top-1
  candidate được gán cứng để tránh GT bị bỏ đói hoàn toàn.

### Tín hiệu mong muốn truyền sang module 3

- `r_ij` và candidate index: ai chịu trách nhiệm cho GT nào;
- `quality` sơ bộ từ IoU dự đoán;
- `d` tại candidate: mật độ cục bộ;
- feature identity để phân biệt hai người thật đứng cạnh nhau với hai box trùng.

Assignment không được truyền một danh sách candidate không có trách nhiệm; module
3 cần biết cặp nào có khả năng là duplicate thật.

Đầu ra của module 2: `r_ij [G,16]`, candidate index `[G,16]`, IoU/quality prior
`[G,16]` và `d` tại candidate.

**Tín hiệu học được mong muốn:** `r_ij` tập trung vào candidate gần đúng và phân
biệt được hai GT cạnh nhau; entropy giảm ở vùng đông nhưng không collapse về
one-hot cho mọi trường hợp.

## 6. Module 3 — Duplicate-Aware Set Scoring

### Đầu vào và đầu ra

Với mỗi cặp candidate có IoU>0.3, tạo feature:

- IoU, khoảng cách tâm, tỷ lệ diện tích;
- cosine similarity của ROI feature;
- overlap trách nhiệm `Σ_i min(r_i,a, r_i,b)`;
- `d` trung bình của hai candidate.

MLP hidden 32 xuất `duplicate_logit`. Score cuối:

```text
score_final = quality × sigmoid(-duplicate_logit)
```

Không xóa box, chỉ giảm score trước NMS; nhờ vậy box không bị mất hoàn toàn nếu
penalty sai.

### Target và gradient

- Positive duplicate: hai candidate cùng match một GT, hoặc cùng có IoU>0.5 với
  cùng GT và overlap responsibility cao.
- Negative: hai candidate match hai GT khác nhau, kể cả khi hai GT ở gần.
- Bias khởi tạo -4 để penalty gần 0 lúc đầu; module phải học từ dữ liệu.
- Không dùng gradient từ set scoring để sửa box; chỉ sửa score, tránh làm lệch
  hình học đã học ở module 2.

**Tín hiệu học được mong muốn:** duplicate logit cao cho hai box cùng giải thích
một GT và thấp cho hai người thật đứng gần nhau.

## 7. Tương tác tối ưu và dự đoán hành vi

- AdamW `lr0=2e-4` khi warm-start; nếu dùng `lr0=1e-3`, routing và assignment
  mới có thể phá đặc trưng A2 trong vài epoch đầu.
- Density head nhận gradient từ routing và assignment; cần warmup 2 epoch để
  tránh `d` bão hòa ở 0 hoặc 1.
- Assignment nhận gradient từ classification và box loss; normalize hai chiều
  giữ tổng gradient mỗi GT ổn định khi số GT thay đổi từ 1 đến hàng chục.
- Set scoring học chậm hơn vì chỉ có cặp duplicate; dùng positive pair hiếm và
  hard negative, không dùng toàn bộ cặp.

Dự đoán theo thời gian:

1. Density map tăng ở vùng đông, routing chuyển dần sang context P3.
2. Entropy assignment giảm; số GT không có candidate `r>0.5` giảm.
3. Duplicate FP giảm trước; sau đó recall crowded và AP75 mới tăng.
4. `max_det=200` saturation giảm nếu set scoring xếp box tốt lên trước.

## 8. Tiêu chí quyết định cho đúng một lần train

- AP75 tăng ít nhất `+0.002` so với A2;
- recall crowded/very-crowded tăng, recall ảnh thưa không giảm quá 0.5 điểm;
- duplicate FP giảm ít nhất 10%;
- background FP không tăng quá 2%;
- số ảnh chạm `max_det=200` giảm;
- AP50 không giảm quá `0.002`.

Nếu chỉ recall tăng hoặc chỉ duplicate giảm mà AP75 không tăng, CASS không giải
quyết được nút thắt AP75. Không chọn một module con để tuyên bố thành công.

## 9. Claim được phép viết

“CASS dùng mật độ cục bộ để định tuyến context, phân bổ trách nhiệm giữa candidate
và GT, rồi dùng trách nhiệm đó để chấm điểm duplicate; ba bước này cùng giải quyết
miss và duplicate trong cảnh đông.” Không claim soft assignment, density routing
hay set scoring là kỹ thuật mới riêng lẻ.

## 10. Tham khảo tư duy thiết kế

- ATSS (Zhang et al., 2020) và TOOD/TAL (Feng et al., 2021): assignment phải
  thích nghi theo từng GT và theo chất lượng căn chỉnh.
- DETR (Carion et al., 2020): bài toán detection là bài toán set prediction;
  trách nhiệm giữa candidate và GT không nên là quan hệ one-hot cứng.
- Soft-NMS (Bodla et al., 2017) và WBF (Solovyev et al., 2021): duplicate là
  quan hệ giữa các box, không chỉ là ngưỡng score.
- CSRNet (Li et al., 2018): mật độ cục bộ là tín hiệu có cấu trúc, không phải
  metadata toàn ảnh.
- Varifocal loss (Zhang et al., 2021): positive và negative cần trọng số bất đối
  xứng trong cảnh đông.

## 11. Đặc tả kỹ thuật chi tiết và pre-registration

### 11.1. Hợp đồng tensor

| Tensor | Shape | Ý nghĩa | Bất biến |
|---|---|---|---|
| `P2` | `[B,32,H/4,W/4]` | feature stride 4 | giữ độ phân giải |
| `P3` | `[B,64,H/8,W/8]` | context | chiếu về 32 kênh trước khi trộn |
| `P4` | `[B,128,H/16,W/16]` | context rộng | chỉ dùng cho density head |
| `d` | `[B,1,H/4,W/4]` | mật độ cục bộ | `[0,1]`, sigmoid |
| `F_route` | `[B,32,H/4,W/4]` | P2 có context theo mật độ | gần P2 khi `d` nhỏ |
| `r_ij` | `[G,16]` | trách nhiệm candidate–GT | tổng mỗi GT xấp xỉ 1 |
| `duplicate_logit` | `[P,1]` | logit cặp trùng | `P` là số cặp trong budget |

`G` là số GT trong ảnh; ảnh rỗng có `G=0` và chỉ dùng classification negative.
`P` bị giới hạn bởi pair budget, không phải toàn bộ `200×200` cặp.

### 11.2. Distribution shift và hình học mật độ

- Train: 662 ảnh, 14.825 GT, trung bình 22,39 GT/ảnh, median 6, max 176, 53 ảnh
  rỗng. Validation: 84 ảnh, 3.608 GT, trung bình 42,95, median 18, max 177, 5
  ảnh rỗng. Validation đông hơn train khoảng 1,9 lần.
- Vì vậy không được học một ngưỡng “crowded” toàn ảnh từ train. `d` phải là mật
  độ **cục bộ**, và ngưỡng quyết định phải được chuẩn hóa theo kích thước cửa sổ.
- Cửa sổ 4×4 ô P2 = 16×16 input px. Ở 320×320 P2 có 102.400 ô; một ảnh train
  trung bình 22,39 GT, nên mật độ trung bình cực thấp. Positive density chỉ tập
  trung ở cụm; loss phải dùng focal/pos-weight.
- Mục tiêu `min(count,4)/4` là tín hiệu “thưa/đông” chứ không phải đếm chính
  xác. Nếu cần phân biệt 4 và 20 người trong cùng cửa sổ, phải dùng target
  log-scale; nhưng thiết kế này chỉ cần routing, nên target bão hòa ở 4 là chấp
  nhận được và phải ghi rõ.
- Ô gần biên letterbox, GT uncertain hoặc GT chồng lấn được ignore; không đưa
  chúng vào negative vì target sẽ sai.

### 11.3. Density-Conditioned Routing

**Tín hiệu học được.** `d` học mật độ tâm người trong cửa sổ 16×16 input px,
không học “ảnh này đông”. `F_route` chỉ thay đổi mạnh ở vùng `d` cao; vùng thưa
giữ P2.

**Khởi tạo.** `d = sigmoid(MLP(up(P3), up(P4)))`; lớp cuối zero-init, bias `-4`
cho `d≈0,018`. `C3=Conv1x1(up(P3))`, `β=0,1`. Ở step 0,
`F_route≈P2 + 0,0018·(C3-P2)`, tức gần identity. Không dùng `β=1` vì context
P3 sẽ thay P2 ngay từ epoch đầu và phá checkpoint.

**Gradient.** `∂L/∂d = β·(C3-P2)·∂L/∂F_route`; `∂L/∂P2` vẫn có đường skip.
Detection loss truyền vào `d` yếu vì `β=0,1`, nên cần density auxiliary loss.
Density loss dùng focal BCE, chuẩn hóa theo số cửa sổ positive; nếu normalize
theo toàn ảnh, 99% negative sẽ làm `d→0`.

**Chẩn đoán.** Log `mean(d)` theo density bin, AUPRC của `d` với tâm GT, tỷ lệ
`d>0,5` ở vùng thưa và `d<0,1` ở vùng đông. Kỳ vọng: `d` cao ở cụm, thấp ở nền;
không tăng đều theo toàn ảnh. Nếu `mean(d)>0,5` sau epoch 3, routing đang bật
context toàn ảnh.

### 11.4. Crowd-Aware Soft Assignment

**Số mẫu.** Với top-k 16, ảnh train trung bình tạo 358 responsibility, ảnh
validation trung bình 688, trong khi chỉ có khoảng 200 candidate/ảnh. Nếu không
chuẩn hóa hai chiều, một candidate có thể nhận trách nhiệm của 3–4 GT và box
loss sẽ kéo box về giữa cụm.

**Chuẩn hóa.** Dùng một vòng Sinkhorn: chuẩn hóa theo candidate cho từng GT,
rồi chuẩn hóa theo GT cho từng candidate. Không dùng phép chia
`r/max(1,Σr)` vì nó không bảo toàn tổng trách nhiệm của GT và có thể làm GT
không còn candidate nào.

**Target box.** Classification dùng `r_ij` mềm. Box regression dùng
`j*=argmax_i r_ij` cho mỗi candidate, với trọng số `max_i r_ij`; không lấy trung
bình box của nhiều GT. Nếu hai GT chồng lấn, candidate được đánh dấu ambiguous
và bị loại khỏi box loss, nhưng vẫn có thể dùng cho classification.

**Cân bằng.** Hard negative tối đa 3 candidate background gần mỗi GT; positive
box loss chuẩn hóa theo số GT; ảnh rỗng chỉ đóng góp classification negative.
Mỗi GT phải có ít nhất một candidate `r>0,5`; nếu không, top-1 được gán cứng,
nhưng phải log số lần fallback. Fallback quá 10% GT là dấu hiệu top-k/prior sai.

**Chẩn đoán.** Log entropy `r_ij` theo density bin, số GT không có candidate
`r>0,5`, và tỷ lệ candidate nhận >1 GT. Kỳ vọng: entropy giảm ở ảnh đông nhưng
không collapse one-hot toàn bộ; một vòng Sinkhorn giữ tổng trách nhiệm mỗi
candidate xấp xỉ ≤1, không đảm bảo tuyệt đối.

### 11.5. Duplicate-Aware Set Scoring

**Cặp và budget.** Chỉ xét cặp có IoU>0,3; sắp theo `IoU × score` và giữ tối đa
4.096 cặp/ảnh. Budget này đủ cho 200 candidate trong ảnh đông nhưng không làm
pairwise MLP thành nút thắt bộ nhớ.

**Target.** Positive: hai candidate cùng match một GT với IoU>0,5 và overlap
responsibility cao. Negative: hai candidate match hai GT khác nhau, kể cả khi
hai GT ở gần. Cặp ambiguous (một candidate gần hai GT) bị ignore, không ép thành
negative.

**Khởi tạo và gradient.** `duplicate_logit` bias `-4` để penalty gần 0 lúc đầu.
Set scoring chỉ sửa score, không sửa box; gradient không đi ngược vào box head.
Score cuối `quality × sigmoid(-duplicate_logit)`; `quality` lấy từ head hiện có.

**Chẩn đoán.** Log duplicate FP trước/sau penalty, AUROC của `duplicate_logit`
trên cặp positive/negative, và số box bị giảm score quá 50%. Nếu penalty giảm
duplicate nhưng recall crowded giảm mạnh, module đang học “hai người gần nhau là
trùng”.

### 11.6. Optimizer, số học và dự đoán

- AdamW, `lr0=2e-4`, 20 epoch, batch 8, 1.660 step; warmup 2 epoch; không đổi
  optimizer giữa các nhánh.
- Density head, assignment MLP và set scoring đều zero-init hoặc bias âm để
  không phá A2 ở step 0.
- Sinkhorn, softmax và log tính ở float32; clamp `d` và `r_ij` trước khi dùng;
  kiểm tra NaN/Inf trong 20 step đầu.
- Tham số mới dự kiến dưới 30k, nhỏ hơn 2% baseline; nếu vượt, giảm hidden 32
  hoặc bỏ nhánh P4.

| Thời điểm | Dự đoán | Nếu sai |
|---|---|---|
| Step 0 | `F_route≈P2`, penalty duplicate gần 0 | feature lệch hoặc NaN |
| Epoch 1–5 | `d` tăng ở cụm, entropy assignment giảm | `d` bật toàn ảnh |
| Epoch 5–10 | duplicate FP giảm; recall crowded giữ | recall crowded giảm |
| Epoch 10–20 | miss crowded giảm, AP75 tăng | chỉ duplicate giảm, AP75 không tăng |
| Kết thúc | AP75 +0,002; duplicate FP −10%; recall ảnh thưa giảm <0,5 điểm | bác bỏ CASS |

Không được chọn riêng density, assignment hay set scoring để tạo kết quả dương
tính sau khi hệ thống thất bại.
