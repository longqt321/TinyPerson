# Hướng dẫn thiết kế kỹ thuật trước khi train

Tài liệu này là khung chung cho ba file `COMPOSITE-*.md`. Mục tiêu không phải
viết code, mà là **dự đoán hành vi model trước khi tiêu tài nguyên train**.

## 1. Luật một setting

Mỗi ý tưởng chỉ được train đúng một treatment setting. Control là A2 baseline có
sẵn, không train thêm một ma trận ablation.

- Không train từng module riêng.
- Không thử 3–4 giá trị hyperparameter.
- Không chọn module con có kết quả đẹp để thay thế kết luận hệ thống.
- Nếu treatment không qua ngưỡng, giả thuyết hệ thống bị bác bỏ; muốn thử biến
  thể khác phải là một ý tưởng mới có lý do thiết kế mới.

## 2. Hằng số dữ liệu phải nhớ

- Train: 662 ảnh; validation: 84 ảnh; official test chỉ dùng sau freeze.
- `imgsz=1280`, batch 8, AdamW, `lr0=0.001`, mosaic 0, multi-scale 0.
- P2 stride 4; ở ảnh vuông 1280, P2 có khoảng 320×320 ô.
- TinyPerson size bin theo `sqrt(area)` px: tiny1 `[1,8)`, tiny2 `[8,12)`,
  tiny3 `[12,20)`, small `[20,32)`, reasonable `>=32`.
- Tiny1 1–8 px chỉ chiếm khoảng 0.25–2 ô P2 theo một chiều; sau resize, median
  tiny1 khoảng 4,5 input px, tức khoảng 1,1 ô P2, và khoảng 1/3 số tiny1 nằm
  trong tối đa 1 ô. Đây là lý do mọi module phải bảo vệ độ phân giải P2, không
  thay bằng P3/P4.
- Validation: 3.608 GT; 1.149 GT best IoU `<0.10`; 866 GT `0.10–0.50`;
  1.157 GT `0.50–0.75`; 436 GT `>=0.75`.
- 179/436 box `IoU>=0.75` có confidence `<0.25`.

### 2.1. Phân bố mẫu thực tế

Số liệu dưới đây tính trực tiếp từ `datasets/derived/tinyperson-yolo/manifest.json`
và label YOLO, không phải ước lượng:

| Tập | Ảnh | GT | GT/ảnh trung bình | Median | Max | Ảnh rỗng |
|---|---:|---:|---:|---:|---:|---:|
| Train | 662 | 14.825 | 22,39 | 6 | 176 | 53 |
| Validation | 84 | 3.608 | 42,95 | 18 | 177 | 5 |

- Validation **đông hơn train khoảng 1,9 lần** theo GT/ảnh. Đây là distribution
  shift thật, không phải lỗi chia split. Mọi module dùng mật độ hoặc top-k phải
  chứng minh nó không học ngưỡng mật độ của train rồi hỏng trên validation.
- Theo `sqrt(area)` nguồn: train có 3.110 tiny1, 3.495 tiny2, 3.841 tiny3,
  2.023 small, 2.356 reasonable. Validation có 805/916/1.106/553/228.
- 5 ảnh validation rỗng và 53 ảnh train rỗng vẫn phải tham gia classification
  negative; loss không được chia cho 0 GT.

### 2.2. Hình học sau resize và giới hạn thông tin

Ảnh nguồn không đồng nhất: 449/662 ảnh train là 1920×1080, 85 ảnh 1280×720,
32 ảnh 1920×1072, 28 ảnh 3840×2160. Với `imgsz=1280` và letterbox:

- ảnh 1920×1080 co khoảng `2/3`; ảnh 4K co khoảng `1/3`; ảnh 1280×720 giữ
  gần nguyên tỉ lệ.
- tiny1 nguồn `[1,8)` px trở thành khoảng `0,33–5,3` px sau resize; median toàn
  bộ GT train sau resize khoảng `8,9` px.
- P2 stride 4, nên một box tiny1 sau resize chỉ chiếm khoảng **1–1,5 ô P2**;
  khoảng 1/3 số tiny1 nằm trong tối đa 1 ô.
  Không có module nào có thể tái tạo chi tiết đã mất khi resize; đó là trần
  thông tin, phải báo riêng khỏi lỗi module.
- Vì vậy mọi module localization phải dùng tọa độ liên tục hoặc offset dưới ô,
  không được giả định mỗi GT có một ô P2 riêng.

### 2.3. Đơn vị đo phải nhất quán

Một lỗi thiết kế phổ biến là trộn pixel nguồn với pixel sau resize. Quy ước:

- **source px**: đơn vị của annotation và AP; dùng cho báo cáo lỗi.
- **input px**: ảnh sau letterbox, đơn vị model thực sự nhìn thấy.
- **stride unit**: `input px / stride`; dùng cho offset, sigma, DFL bin.

Median lệch tâm `3,81 source px` ở nhóm `IoU 0.10–0,50` tương đương khoảng
`2,5 input px` trên ảnh 1920×1080, tức khoảng `0,6` ô P2. Đây là lý do bài toán
không thể giải chỉ bằng thêm một ô positive rộng hơn.

## 3. Checklist thiết kế bắt buộc

### 3.1. Hợp đồng đầu vào/đầu ra

- Shape, stride, số kênh, dtype, phạm vi giá trị.
- Module nhận feature nào, ở level nào, có upsample hay không.
- Đầu ra là feature, box, score, uncertainty hay mask.
- Module sau được phép dùng thông tin gì; thông tin nào phải bị chặn để tránh
  rò rỉ hoặc học shortcut.

Trước khi vẽ kiến trúc, phải trả lời được năm câu:

1. **Tín hiệu nào module học?** Nó có nhìn thấy biến cần dự đoán trong vùng
   receptive field không, hay chỉ nhìn thấy metadata toàn ảnh?
2. **Tín hiệu đó có identifiability không?** Hai nguyên nhân khác nhau có tạo
   cùng một đầu vào không? Nếu có, module không thể phân biệt và cần thêm
   nguồn thông tin.
3. **Bất biến nào phải giữ?** Với translation, scale, border, letterbox, ảnh
   rỗng, ảnh cực đông, module có trả output đúng nghĩa không?
4. **Capacity có đủ nhưng không quá lớn?** Số tham số, số kênh và số điểm sample
   phải tương xứng với 662 ảnh train; module quá lớn sẽ overfit trước khi học
   được tín hiệu.
5. **Module gây nhiễu gì cho loss cũ?** Gradient mới có lấn át classification,
   CIoU hay DFL ở giai đoạn đầu không?

### 3.2. Feature map và receptive field

- P2 giữ vị trí; P3/P4 giữ ngữ cảnh. Không đổi P2 sang P3 nếu mục tiêu là tiny1.
- Tính receptive field theo stride và kernel; ghi rõ module nhìn được bao nhiêu
  pixel ở ảnh gốc resized.
- Nếu receptive field quá lớn, module có thể học ngữ cảnh nhưng mất tọa độ.
- Nếu quá nhỏ, module không phân biệt được người với texture nền.

Bảng nominal incremental RF cho các nhánh trực tiếp (chưa cộng RF tích lũy của
backbone/neck):

| Nhánh | Công thức | RF tăng thêm ở input px |
|---|---|---:|
| 1 lớp 3×3 trên P2 | `3 × 4` | 12 |
| 3 lớp 3×3 trên P2 | `(3 + 2×2) × 4` | 28 |
| 1 lớp 3×3 trên P3, upsample về P2 | `3 × 8` | 24 |
| 1 lớp 3×3 trên P4, upsample về P2 | `3 × 16` | 48 |
| 3 lớp 3×3 trên P3 | `(3 + 2×2) × 8` | 56 |

Đây là **incremental RF**, không phải effective RF của cả mạng. Phải dùng đúng
ngôn ngữ này trong bài báo để không claim sai.

### 3.3. Initialization

- Module mới nên bắt đầu gần identity: residual zero, gate bias âm, scale nhỏ.
- Không khởi tạo gate mở toàn ảnh; như vậy model bắt đầu bằng một model khác A2.
- Khởi tạo phải giữ gradient chảy: nếu nhân với scalar 0, nhánh đó không học.
- Ghi rõ bias, variance và lý do chọn; không để “mặc định PyTorch”.

Với gate sigmoid, phải ghi cả ba con số: bias `b`, `sigmoid(b)`, và hệ số residual
`α`. Ví dụ `b=-2,5` cho `mean(g)≈0,076`; `b=-4` cho `≈0,018`. Nếu `α=0,01` thì
perturbation tối đa ở bước 0 chỉ là 1% feature P2. Đây là cách kiểm soát
“identity lúc init” bằng số, không bằng cảm giác.

Với lớp cuối zero-init, gradient tới các lớp trước bằng 0 ở step đầu nhưng
gradient tới chính lớp cuối khác 0. Phải nói rõ điều này; nếu không, thiết kế
zero-init có thể làm module không học trong vài chục step.

### 3.4. Gradient

- Loss nào truyền về module nào; module nào bị stop-gradient.
- Gradient có bị khuếch đại ở box nhỏ không; có cần scale theo `sqrt(area)` không.
- Có nguy cơ gate bão hòa, sigma collapse, assignment uniform không.
- AMP có làm tràn số với loss chia cho sigma hoặc diện tích nhỏ không.
- Gradient clipping và warmup có cần không.

Viết gradient budget bằng công thức trước khi train. Ví dụ với
`P2' = P2 + α·g·E`:

```text
∂L/∂E   = α · g · ∂L/∂P2'
∂L/∂g   = α · (E · ∂L/∂P2')
∂L/∂P2  = ∂L/∂P2'  (skip connection)
```

Nếu `α=0,01`, gradient vào evidence nhỏ hơn 100 lần so với đường skip. Vì vậy
cần auxiliary loss cho gate hoặc một loss riêng cho evidence; không được kỳ vọng
detection loss đơn thuần huấn luyện được gate trong 1.660 step.

Mỗi loss mới phải có **normalizer**: theo số GT, theo số positive, theo số cặp
hoặc theo số ảnh. Không dùng tổng loss thô vì ảnh 176 GT sẽ chi phối ảnh 1 GT.

### 3.5. Tỷ lệ dương/âm và số mẫu

- Box nhỏ có ít anchor/positive; nếu lấy toàn bộ background, positive bị nhấn chìm.
- Ghi rõ số positive mỗi GT, top-k candidate, hard negative ratio.
- Với ảnh đông, số GT lớn; chuẩn hóa loss theo GT hoặc theo ảnh, không để ảnh
  đông chi phối toàn batch.
- Không dùng test để chọn ngưỡng positive/negative.

Ví dụ định lượng cho P2 ở 1280: 320×320 = 102.400 ô/ảnh. Nếu mỗi GT lấy tối đa
4 ô positive, ảnh train trung bình 22,39 GT cho tối đa khoảng 90 positive, tức
`0,09%` số ô. Ảnh 176 GT cho tối đa 704 positive, tức `0,69%`. Vì vậy gate/
density head phải dùng focal-style loss hoặc pos-weight, và báo cáo positive
fraction thực tế chứ không chỉ loss.

### 3.6. Optimizer và lịch train

- AdamW, learning rate, weight decay, warmup, epochs, batch, EMA.
- Warm-start từ A2 để kiểm soát initialization; nếu head thay đổi shape, nói rõ
  phần nào được load, phần nào khởi tạo mới.
- Learning rate quá cao có thể phá feature A2 trước khi module mới học xong.
- Learning rate quá thấp làm module mới không thoát khỏi identity trong 20 epoch.

Với 662 ảnh, batch 8, mỗi epoch có 83 iteration; 20 epoch là 1.660 optimizer
step. Đây là ngân sách rất nhỏ. Vì vậy:

- Không dùng learning-rate schedule phức tạp hoặc per-parameter group nếu chưa
  chứng minh cần; một `lr0` cố định giữ tính nhân quả của experiment.
- New module nên có zero-init hoặc bias hợp lý để không cần một “warmup riêng”
  dài hơn baseline.
- Nếu module chưa thoát identity sau 5 epoch, đó là failure của thiết kế
  initialization, không phải lý do mở thêm ablation.

### 3.7. Cơ chế gate

- Gate dùng sigmoid, softmax hay temperature; có bị bão hòa không.
- Gate có phạt sparsity/smoothness không; hệ số có cố định trước train không.
- Gate được giám sát bằng tín hiệu trực tiếp hay chỉ bằng detection loss.
- Nếu chỉ dùng detection loss, credit assignment cho gate thường nhiễu; cần
  auxiliary target hoặc stop-gradient hợp lý.

Phải log tối thiểu: `mean(g)`, phân vị 10/50/90 của `g`, tỷ lệ ô `g>0,9`, tỷ lệ
ô `g<0,1`, và AUPRC của `g` so với tâm GT (nếu có auxiliary target). Nếu
`mean(g)` đi từ 0,08 lên 0,9 trong 2 epoch, gate đang học “bật tất cả”, không
phải chọn evidence. Nếu `mean(g)<0,01` sau 5 epoch, gate đang chết.

### 3.8. Thông tin truyền cho lớp sau

- Chỉ truyền thông tin mà lớp sau thực sự dùng được.
- Không truyền một scalar toàn ảnh nếu lớp sau cần quyết định theo vùng.
- Không truyền uncertainty nếu lớp sau không có loss calibration.
- Ghi rõ invariant: shape, range, ý nghĩa, đơn vị.

Kiểm tra leakage: target auxiliary (tâm GT, mật độ GT, IoU GT) chỉ được xuất hiện
trong loss, không được đi vào inference path. Nếu một tensor chỉ có ý nghĩa khi
có GT, nó phải nằm trong nhánh loss và bị detach khỏi input của module kế tiếp.

### 3.9. Số học và AMP

- `sigmoid`, `softmax`, `log`, `exp` phải được clamp trước khi dùng.
- Sigma/NLL và DFL nên tính ở `float32`, kể cả khi backbone chạy AMP.
- Không chia cho `sqrt(area)` hoặc sigma mà không có `+eps`.
- Kiểm tra NaN/Inf ở gate, sigma, offset và box decode trong 20 step đầu.
- Gradient clipping phải là baseline hiện có; không thêm hệ số mới nếu chưa
  chứng minh cần.

### 3.10. Ngân sách tham số và latency

- Tổng tham số mới nên dưới 10% baseline 2,52M; vượt quá cần lý do từ dữ liệu,
  không chỉ từ kiến trúc.
- ROI sampling theo candidate thường là nút thắt latency, không phải phép nhân
  MLP. Phải đo latency ở batch 1, 1280 trước khi cam kết.
- Nếu latency tăng quá 20–25%, module khó được chấp nhận ngay cả khi AP75 tăng.

## 4. Cách dự đoán hành vi trước khi train

Trước khi train, viết ra các tín hiệu kỳ vọng và cách kiểm tra offline:

| Tín hiệu | Kỳ vọng nếu thiết kế đúng | Nếu sai thì sao |
|---|---|---|
| Gate mean | tăng chậm từ khoảng 0.05 lên 0.2–0.5 ở vùng object | gate bão hòa hoặc tắt |
| Refinement residual | trung vị <0.25 box size | box bị drift |
| Sigma | cao ở near-miss, thấp ở box tốt | uncertainty vô dụng |
| Assignment entropy | giảm ở ảnh đông | assignment uniform |
| Duplicate score | tương quan với duplicate FP | set scoring không học |
| Quality calibration | ECE giảm, q cao cho IoU>=0.75 | ranking vẫn sai |

Không cần train thêm để đo các tín hiệu này; dùng diagnostic trên validation và
log nội bộ trong lần train duy nhất.

### 4.1. Mẫu pre-registration

| Thời điểm | Kỳ vọng | Dấu hiệu thất bại |
|---|---|---|
| Step 0 | module gần identity; box gần A2 | output lệch >1% hoặc NaN |
| Epoch 1–5 | chỉ có head mới thay đổi; gate/density chưa bão hòa | gate mean >0,5 hoặc <0,01 |
| Epoch 5–10 | tín hiệu auxiliary tăng; box chưa drift | AP50 giảm >0,002 |
| Epoch 10–20 | lỗi mục tiêu giảm trước, AP75 tăng sau | AP75 không tăng nhưng recall tăng |
| Kết thúc | AP75 +0,002; AP50 không giảm quá 0,002 | chỉ tăng recall/duplicate giảm |

Không được đổi các ngưỡng này sau khi xem kết quả.

## 5. Tư duy thiết kế tham khảo

- **ATSS**: chọn positive theo thống kê IoU của từng GT thay vì một ngưỡng cố
  định. Bài học: assignment phải thích nghi theo phân bố dữ liệu.
- **FCOS/centerness**: chất lượng box có thể được dự đoán từ tín hiệu tâm; bài
  học: đừng dùng chung một score cho classification và localization.
- **GFL/Varifocal**: phân bố chất lượng và class nên được học cùng nhau nhưng
  không đồng nhất; bài học: quality target phải là IoU thật, không phải confidence thô.
- **IoU-Net**: dự đoán IoU là một tín hiệu ranking riêng; bài học: localization
  confidence có thể cải thiện AP mà không đổi box.
- **Deformable DETR/DCNv2**: sampling offset và attention thưa giúp tập trung
  vào vùng quan trọng; bài học: không cần xử lý mọi ô với cùng mức.
- **Soft-NMS/WBF**: duplicate là bài toán set, không chỉ là bài toán score;
  bài học: cần thông tin trách nhiệm giữa các box.
- **DFL**: phân phối rời rạc trên cạnh box giúp biểu diễn uncertainty; bài học:
  DFL là representation, không tự động là module localization mới.
- **Beta-NLL**: uncertainty có thể bị lạm dụng để giảm loss; bài học: phải clamp
  sigma, dùng beta-NLL hoặc regularization.
- **FPN và effective receptive field** (Lin et al., 2017; Luo et al., 2016):
  level quyết định độ phân giải và ngữ cảnh; phải phân biệt nominal với effective
  RF, không suy luận từ kernel size đơn lẻ.
- **Sinkhorn/optimal transport** (Cuturi, 2013): chuẩn hóa hai chiều là cách có
  cơ sở để tránh một candidate gánh quá nhiều GT trong ảnh đông.
- **Calibration** (Guo et al., 2017): confidence thô thường không calibration;
  quality head phải được đo bằng reliability diagram, không chỉ loss.

## 6. Quy tắc viết claim

- Không claim module đã biết là mới.
- Claim phải nêu rõ chuỗi module và lỗi nào được xử lý.
- Kết quả một seed chỉ là tín hiệu; muốn claim khoa học cần 3 seed paired.
- Kết quả âm phải được giữ và báo cáo.

### 6.1. Giới hạn bằng chứng của một setting

Một treatment setting chỉ trả lời được câu hỏi “hệ thống này có vượt control
không”, không trả lời được “module nào đóng góp bao nhiêu”. Vì vậy:

- Không viết “gate có lợi” nếu chỉ train hệ thống đầy đủ.
- Không viết “DFL cải thiện AP75” nếu DFL chỉ là một thành phần.
- Có thể viết “chuỗi X→Y→Z vượt control ở AP75”, kèm bằng chứng cơ chế từ
  diagnostic (gate mở đúng vùng, sigma tương quan lỗi, duplicate score giảm).
- Nếu kết quả âm, kết luận là chuỗi giả thuyết bị bác bỏ; không cứu bằng cách
  chọn module con đẹp nhất.
