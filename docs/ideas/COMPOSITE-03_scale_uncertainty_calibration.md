# COMPOSITE-03 — SURE: Căn chỉnh theo kích thước, hồi quy có bất định và hiệu chỉnh IoU

## 1. Câu hỏi nghiên cứu

SADR8 cho thấy chỉ thêm bin phân phối không đủ: validation AP75 tăng nhưng
official-test AP75 giảm. SURE không coi DFL là module mới. Nó dùng DFL như một
biểu diễn tọa độ bên trong một chuỗi có ba tầng:

```text
P2/P3/P4 → Scale-Conditioned Alignment → feature đúng hình học hơn
        → Uncertainty Distributional Regression → box kỳ vọng + độ bất định
        → IoU-Calibrated Confidence → score phản ánh chất lượng box cuối
```

Câu hỏi là: khi feature đã được căn chỉnh theo scale, uncertainty có giúp phân
biệt near-miss với box đã tốt và giúp ranking đúng hơn không?

## 2. Bằng chứng và dự đoán trước khi thiết kế

- 866 GT có best IoU `0.10–0.50`; sai số tâm trung vị `3.81 px`.
- 1.157 GT có best IoU `0.50–0.75`; đây là nhóm cần cải thiện để AP75 tăng.
- Box `IoU>=0.75` có sai số tâm trung vị `0.94 px`, nên không cần module sửa tất
  cả box; chỉ cần sửa near-miss.
- `reg_max=1` hiện tại là scalar regression, rất nhạy với 1–2 px.
- P2 stride 4 tại `imgsz=1280`; box tiny1 1–8 px sau resize có median khoảng
  1,1 ô P2, khoảng 1/3 nằm trong tối đa 1 ô. P3/P4 không đủ độ phân giải để
  định vị tiny1 nhưng đủ để cung cấp context.

**Dự đoán:** alignment phải làm giảm sai số tâm trước; distributional head phải
làm giảm số GT ở nhóm `0.50–0.75`; calibration phải làm tăng confidence của box
`>=0.75`. Nếu AP75 tăng nhưng số GT `>=0.75` không tăng, đó là ranking artifact,
không phải localization.

## 3. Setting duy nhất được train

Chỉ train một cấu hình SURE đầy đủ; không train U1/U2/U3 riêng.

- Warm-start backbone/neck từ A2 `best.pt`; box head khởi tạo mới vì `reg_max`
  thay đổi từ 1 sang 8. Classification head giữ nguyên nếu state dict tương thích.
- Dữ liệu: 662 ảnh train / 84 ảnh validation; không đổi split.
- Fine-tune 20 epoch, `lr0=2e-4`, AdamW, weight decay baseline, batch 8,
  `imgsz=1280`, mosaic 0, multi-scale 0, seed 42.
- Distribution: `reg_max=8`, bin biểu diễn theo stride; P2 tương đương dải
  0–28 px, đủ cho tiny1/tiny2.
- Alignment: sampling `3×3` trên P2, context `3×3` trên P3; offset scale dự đoán
  theo từng ô.
- Sigma: dự đoán theo đơn vị stride, clamp `[0.25, 2.0]`; khởi tạo `0.25`.
- Quality head: hidden 32, score `cls × q`; `max_det=200`, NMS baseline.

Không có bảng ablation. Nếu SURE không đạt, không chọn riêng alignment hoặc
quality head để tạo một kết quả dương tính.

## 4. Module 1 — Scale-Conditioned Alignment

### Đầu vào và đầu ra

- Đầu vào P2 `[B,32,H/4,W/4]`, P3 `[B,64,H/8,W/8]`, P4 `[B,128,H/16,W/16]`.
- Scale map `s_map [B,1,H/4,W/4]` dự đoán log kích thước trung bình cục bộ.
- Offset `[B,2K,H/4,W/4]` cho `K=9` điểm sample; offset bị giới hạn bởi
  `tanh(offset) × 0.75 × stride` (tối đa 3 input px ở P2).
- Đầu ra `F_align [B,32,H/4,W/4]`, giữ nguyên kích thước P2.

### Receptive field và độ phân giải

- P2 là nguồn chính vì box 1–8 px cần độ phân giải stride 4; P3 chỉ cung cấp
  context, không thay thế P2.
- Với `3×3` sample quanh mỗi ô P2, vùng nhìn cơ bản khoảng 12×12 input px.
  Offset bị chặn ở `0,75×stride = 3 input px` theo mỗi chiều, đủ phủ median
  near-miss khoảng 2,5 input px nhưng không biến module thành biến dạng tùy ý.
  Không cho offset vượt quá 3 px vì với 662 ảnh train, offset lớn rất dễ học
  thành một phép dịch scale toàn ảnh.
- P3 được upsample về P2 để căn thẳng tọa độ; không sample P4 trực tiếp vì một
  box tiny1 có thể nằm gọn trong một ô P4.

### Initialization và gradient

- Các conv ẩn dùng `BatchNorm2d` trước SiLU; các MLP dùng `LayerNorm`; lớp output
  dự đoán cuối vẫn zero-init.
- Lớp cuối dự đoán offset khởi tạo zero; `s_map` khởi tạo để scale trung bình
  khoảng 4 px (một ô P2). Như vậy `F_align≈P2` ở bước đầu.
- Sampling dùng bilinear `grid_sample`; gradient phải chảy qua **cả feature và
  grid** thì offset predictor mới học được. Không detach grid, vì detach sẽ biến
  offset thành hằng số và module chỉ còn là một phép làm mờ. Để tránh bùng nổ,
  offset bị chặn bằng `tanh`, grid được clamp theo biên ảnh, và gradient của
  nhánh offset được clip theo baseline.
- Nếu offset học quá lớn, thêm phạt `mean(offset²)`; hệ số cố định trước khi
  train, không tune theo validation.

### Tín hiệu mong muốn truyền sang module 2

- `F_align` chứa feature đã căn theo scale;
- `s_map` cho biết vùng nào là tiny1/tiny2/tiny3;
- offset magnitude là tín hiệu phụ về độ bất định hình học.

Không truyền `s_map` như một nhãn cứng; nó chỉ là conditioning cho regression.

**Tín hiệu học được mong muốn:** offset học căn chỉnh theo biên/tâm, không học
một phép biến dạng tùy ý; `s_map` phản ánh scale cục bộ, không phải scale toàn ảnh.

## 5. Module 2 — Uncertainty Distributional Regression

### Đầu vào và đầu ra

- Đầu vào `F_align` và candidate/reference point.
- Mỗi cạnh có `reg_max=8` logits; box kỳ vọng:

```text
p_k = softmax(logits_k)
d_side = Σ_k p_k · bin_k
box_expected = decode(reference, d_side)
```

- Mỗi cạnh có `log_sigma`; `sigma = clamp(softplus(log_sigma), 0.25, 2.0) × stride`.
- Residual refiner nhận ROI feature, box kỳ vọng, sigma và objectness; xuất
  `Δbox` và cổng sửa `g_refine`.

### Loss và gradient

```text
L_DFL   = cross_entropy(logits, target_bin)
L_box   = CIoU(box_final, GT)
L_sigma = 0.5 · ((d_side - target_side) / sigma)² + log(sigma)
L_final = L_DFL + λ_box·L_box + λ_sigma·L_sigma
```

- `L_sigma` dùng beta-NLL hoặc Gaussian NLL có clamp; sigma không được nhận
  gradient từ CIoU để tránh nó học cách che lỗi box.
- `sigma` khởi tạo 0.25 stride: P2 tương đương 1 px, P3 tương đương 2 px. Đây là
  mức jitter hợp lý cho tiny annotation, không quá lớn để làm mất tín hiệu.
- Nếu sigma collapse về sàn, uncertainty không học được; nếu sigma chạm trần,
  NLL đang bị lạm dụng để giảm loss. Cả hai là failure mode phải báo.
- Residual refiner khởi tạo `Δbox=0`, `g_refine=0.5`; box cuối bắt đầu bằng box
  kỳ vọng. Loss residual là Huber chuẩn hóa theo `sqrt(area)+1`.

### Cân bằng dương/âm

Box loss chỉ áp dụng cho positive/assigned candidate. Với negative, classification
và objectness vẫn học như baseline. Quality head dùng positive + tối đa 3 hard
negative mỗi positive; không dùng toàn bộ background vì sẽ làm quality head học
“mọi thứ là 0”.

### Tín hiệu mong muốn truyền sang module 3

- box cuối và `sigma` theo từng cạnh;
- `s_map`/scale level;
- ROI feature đã align;
- `g_refine` và độ lớn residual.

Module 3 không được dùng `sigma` thay cho chất lượng box; nó phải học IoU thật
của box cuối.

**Tín hiệu học được mong muốn:** sigma cao ở near-miss/annotation mơ hồ và thấp
ở box đã tốt; không collapse về sàn hoặc trần.

## 6. Module 3 — IoU-Calibrated Confidence

### Đầu vào, đầu ra và target

- Đầu vào: ROI feature, box cuối, sigma, residual norm, scale level.
- Đầu ra: `q_logit`; `q=sigmoid(q_logit)` xấp xỉ IoU của box cuối với GT.
- Score cuối: `score_final = cls_prob × q`; chỉ sửa score, không sửa box.
- Target: IoU thật của box cuối; stop-gradient qua target.

### Initialization và gradient

- `q_logit` khởi tạo 0 (`q=0.5`) để không làm sụp score ở epoch đầu.
- Loss SmoothL1/BCE mềm; box `IoU>=0.75` được oversample vì đây là nhóm AP75 cần
  tăng.
- Không truyền gradient từ quality head về box; nếu muốn box học từ quality,
  phải tách một loss riêng và đó là thay đổi hệ thống khác.

**Tín hiệu học được mong muốn:** `q` tương quan với IoU cuối, không tương quan
chỉ với class confidence.

## 7. Tương tác tối ưu và dự đoán hành vi

- AdamW `lr0=2e-4` khi warm-start; box head mới có thể nhận gradient lớn hơn
  backbone. Dùng warmup 2 epoch và gradient clipping baseline.
- `reg_max=8` làm head lớn hơn scalar; VRAM tăng nhưng không đáng kể so với
  backbone.
- Sigma và box regression tương tác: nếu sigma tăng để giảm NLL, box loss có thể
  bị downweight. Vì vậy sigma chỉ dùng cho NLL, không dùng làm trọng số CIoU.

Dự đoán theo thời gian:

1. Alignment giảm sai số tâm ở nhóm `0.50–0.75` trước.
2. Distributional head tăng số GT `>=0.75`; sigma cao hơn ở near-miss.
3. Quality head tăng confidence của box tốt, AP75 tăng sau cùng.

Failure mode:

- AP50 giảm mạnh: box bị drift do alignment/refinement quá mạnh.
- AP75 không tăng dù số GT `>=0.75` tăng: ranking vẫn sai.
- Sigma uniform hoặc collapse: uncertainty không có tín hiệu.
- Train loss giảm nhưng validation AP75 giảm: overfit box head trên 662 ảnh.

## 8. Tiêu chí quyết định cho đúng một lần train

- AP75 tăng ít nhất `+0.002` so với A2;
- số GT `IoU>=0.75` tăng, số GT `0.50–0.75` giảm;
- AP50 không giảm quá `0.002`;
- calibration error của quality head giảm;
- sigma có tương quan dương với sai số box;
- latency/VRAM không vượt quá 25% so với A2.

Nếu không đạt, SURE bị bác bỏ ở cấp hệ thống. Không chọn alignment, DFL hay
quality head riêng lẻ để tạo kết quả dương tính.

## 9. Claim được phép viết

“SURE truyền thông tin scale và uncertainty xuyên suốt từ alignment sang
distributional regression và calibration, nhờ đó phân biệt near-miss với box đã
tốt và cải thiện AP75.” DFL, deformable sampling và IoU-aware confidence phải
được mô tả là thành phần đã biết, không phải đóng góp mới.

## 10. Tham khảo tư duy thiết kế

- DFL (Li et al., 2020) và GFL: phân phối rời rạc trên cạnh box là representation
  mạnh hơn scalar, nhưng không tự động giải quyết ranking.
- IoU-Net (Jiang et al., 2018): chất lượng localization là tín hiệu riêng, có
  thể dự đoán và dùng để xếp hạng.
- Deformable DETR (Zhu et al., 2021) và DCNv2 (Zhu et al., 2019): offset/sampling
  học được giúp căn chỉnh feature theo hình học.
- Beta-NLL (Seitzer et al., 2022): uncertainty cần regularization để không bị
  lạm dụng làm giảm loss.
- Varifocal loss (Zhang et al., 2021): quality-aware target và asymmetric
  weighting phù hợp với bài toán positive/negative lệch mạnh.

## 11. Đặc tả kỹ thuật chi tiết và pre-registration

### 11.1. Hợp đồng tensor

| Tensor | Shape | Ý nghĩa | Bất biến |
|---|---|---|---|
| `P2` | `[B,32,H/4,W/4]` | feature stride 4 | giữ độ phân giải gốc |
| `P3/P4` | `[B,64,H/8,W/8]`, `[B,128,H/16,W/16]` | context | chỉ dùng qua phép chiếu/upsample |
| `s_map` | `[B,1,H/4,W/4]` | log scale cục bộ | không dùng như nhãn cứng |
| `offset` | `[B,18,H/4,W/4]` | 9 điểm × 2 tọa độ | `tanh`, tối đa 0,75 stride |
| `F_align` | `[B,32,H/4,W/4]` | feature đã căn | cùng hệ tọa độ P2 |
| `d_side` | `[B,4,H/4,W/4]` | kỳ vọng khoảng cách 4 cạnh | đơn vị stride |
| `log_sigma` | `[B,4,H/4,W/4]` | log bất định 4 cạnh | float32, clamp |
| `q` | `[B,N,1]` | IoU dự đoán | `[0,1]` |

Không trộn `source px` với `stride unit`. Toàn bộ offset, DFL bin và sigma dùng
`stride unit`; khi báo cáo lỗi mới đổi về source px.

### 11.2. Hình học và giới hạn thông tin

- Train có 14.825 GT/662 ảnh, trung bình 22,39 GT/ảnh, median 6, max 176, 53
  ảnh rỗng. Validation có 3.608 GT/84 ảnh, trung bình 42,95, median 18, max
  177, 5 ảnh rỗng.
- Ảnh 1920×1080 co khoảng `2/3`; ảnh 4K co khoảng `1/3`. Tiny1 nguồn 1–8 px
  thành khoảng 0,33–5,3 input px, thường chỉ 1–1,5 ô P2 và khoảng 1/3 nằm
  trong tối đa 1 ô. Đây là lý do phải dùng tọa độ liên tục và offset dưới ô,
  không dựa vào cell assignment.
- Với `K=9` và offset tối đa 3 px, vùng sample P2 phủ khoảng 18×18 input px;
  tức khoảng 4,5×4,5 ô P2. Vùng này đủ chứa một tiny2/tiny3 nhưng vẫn có thể
  pha nền với tiny1; vì vậy 9 điểm sample dùng trọng số Gaussian cố định theo
  khoảng cách, không thêm một nhánh chọn cửa sổ học được.
- P3 `3×3` phủ 24×24 input px và chỉ cung cấp context; P4 không sample trực
  tiếp vì box tiny1 có thể nhỏ hơn một ô P4.

### 11.3. Scale-Conditioned Alignment

**Tín hiệu học được.** Offset phải học “sample ở đâu quanh ô này để feature
khớp biên/tâm người”, không học một phép dịch toàn ảnh. `s_map` học scale cục
bộ để conditioning cho regression; nó không phải nhãn size bin cứng.

**Khởi tạo.** Lớp cuối offset zero-init; `s_map` khởi tạo để scale trung bình
khoảng 4 input px. Ở step 0, `F_align≈P2`. Với lớp cuối zero-init, gradient tới
các lớp trước của offset predictor bằng 0 ở step đầu, nhưng gradient qua
`grid_sample` tại vị trí sample vẫn khác 0; sau step đầu offset predictor bắt
đầu học. Không detach grid.

**Gradient và ổn định.** `grid_sample` bilinear cho gradient liên tục theo grid,
nhưng đạo hàm có thể lớn khi feature thay đổi mạnh. Chặn offset bằng `tanh`,
clamp grid theo biên, và clip gradient của nhánh offset theo baseline. Thêm phạt
`mean(offset²)` với hệ số cố định trước train; nếu offset gần trần ở >30% ô,
module đang học biến dạng quá mạnh.

**Chẩn đoán.** Log phân vị offset theo size bin, tỷ lệ offset chạm trần, và
tương quan giữa offset và sai số tâm GT. Kỳ vọng: offset lớn hơn ở near-miss,
gần 0 ở box đã tốt; không có pattern dịch đều theo ảnh.

### 11.4. Uncertainty Distributional Regression

**DFL theo level.** `reg_max=8` nghĩa là 8 bin `0..7`, nhân với stride của level:
P2 tối đa 28 input px, P3 56, P4 112. Đây là lý do không dùng chung một thang
pixel cho mọi level. Với tiny1 sau resize 0,33–5,3 px, box chủ yếu rơi vào bin
0–1 của P2; DFL không tự tạo độ phân giải dưới 4 px, nó chỉ biểu diễn phân phối
quanh bin.

**Khởi tạo DFL.** Không để logits zero vì softmax đều 8 bin cho kỳ vọng
`3,5×stride = 14 px` ở P2, quá lớn cho tiny1. Khởi tạo logits bằng một phân
phối Gaussian rời rạc quanh median `ltrb` của positive train theo level, tính
trên train-only. Không dùng validation để chọn prior. Nếu không có prior tin
cậy, dùng trung vị `0,5×stride` và ghi rõ đây là prior yếu.

**Loss.** `L_DFL` là cross-entropy trên target bin; `L_box` là CIoU trên box
kỳ vọng/refined; `L_sigma` là Gaussian NLL có clamp. Sigma chỉ nhận gradient từ
NLL, không nhận từ CIoU. Không dùng sigma làm trọng số CIoU vì như vậy sigma có
thể học cách che lỗi box.

**Khởi tạo sigma.** `sigma = clamp(softplus(log_sigma), 0,25, 2,0) × stride`;
`log_sigma ≈ -1,24` cho `softplus=0,25`. P2 tương đương 1 input px, P3 2 px.
Đây là mức jitter hợp lý cho annotation tiny, nhưng không được để sigma chạm
sàn/trần quá sớm.

**Chẩn đoán.** Log phân bố sigma theo size bin, tương quan sigma với sai số
tâm, tỷ lệ sigma ở sàn/trần, và phân bố bin DFL. Kỳ vọng: sigma cao ở near-miss,
thấp ở box `>=0.75`; phân bố bin không collapse về một bin.

### 11.5. IoU-Calibrated Confidence

- `q_logit=0` lúc init, `q=0.5`; score bị giảm đều, không đổi thứ tự NMS.
- Target `q*=IoU(box_final, GT)`, stop-gradient qua box và target.
- SmoothL1 ở miền logit hoặc BCE mềm; oversample box `IoU>=0.75` tối đa 3 lần.
- Không truyền gradient từ q về box; q chỉ sửa score. Nếu muốn box học từ q,
  phải tách loss và khai báo là thay đổi hệ thống khác.
- Đo ECE/reliability diagram trên validation; q phải tăng theo IoU. Nếu q cao
  cho background hoặc q gần hằng số, calibration thất bại.

### 11.6. Optimizer, số học và ngân sách

- AdamW, `lr0=2e-4`, 20 epoch, batch 8, 1.660 step; warmup 2 epoch; EMA baseline.
- Backbone/neck/classification warm-start từ A2; box head mới vì `reg_max` đổi
  từ 1 sang 8; alignment/quality head mới.
- DFL, sigma/NLL và log/exp ở float32; clamp trước khi dùng; kiểm tra NaN/Inf
  trong 20 step đầu.
- Tham số mới dự kiến dưới 50k, khoảng 2% baseline; nếu vượt, giảm hidden hoặc
  bỏ nhánh P4.
- Latency: sample 9 điểm trên P2 cho mọi ô là chi phí lớn; chỉ sample quanh
  candidate/positive trong loss, không sample toàn bộ P2 ở inference nếu có thể.

### 11.7. Dự đoán trước train và điều kiện bác bỏ

| Thời điểm | Dự đoán | Nếu sai |
|---|---|---|
| Step 0 | `F_align≈P2`, box gần prior, q=0,5 | output lệch hoặc NaN |
| Epoch 1–5 | offset nhỏ, sigma chưa chạm trần | offset bão hòa, sigma collapse |
| Epoch 5–10 | sai số tâm nhóm `0.50–0.75` giảm | AP50 giảm >0,002 |
| Epoch 10–20 | số GT `>=0.75` tăng; q tăng theo IoU | AP75 không tăng dù IoU tăng |
| Kết thúc | AP75 +0,002; AP50 không giảm quá 0,002; sigma tương quan lỗi | bác bỏ SURE |

Không được chọn riêng alignment, DFL hay quality head để tạo kết quả dương tính
sau khi hệ thống thất bại. DFL và IoU-aware confidence là thành phần đã biết,
không phải đóng góp mới.
