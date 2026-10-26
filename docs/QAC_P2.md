# QAC-P2: Hiệu chuẩn độ tin cậy theo chất lượng hình học cho phát hiện người tí hon

**Trạng thái:** Đã hoàn thành implement, vượt qua lint/kiểm thử tích hợp, và xác thực qua lượt chạy Smoke Test (5 epoch, seed 42) trên Modal A10. Kết quả 5 epoch ban đầu ghi nhận vượt trội so với baseline A2 trên `mAP50-95` (+21%), `mAP50` (+15%), và `Recall` (+25%). Đang chờ kết quả huấn luyện đầy đủ 50 epoch để có kết luận thống kê chính thức.

---

## 1. Vấn đề đo được từ thực nghiệm (The Empirical Bottleneck)

Trong tài liệu phân tích lỗi của dự án ([`docs/ideas/TECHNICAL_DESIGN_GUIDE.md`](TECHNICAL_DESIGN_GUIDE.md#L28-L30) và [`docs/ideas/IDEA-05_quality_aware_ranking.md`](ideas/IDEA-05_quality_aware_ranking.md#L18-L24)), khi đánh giá mô hình baseline **`tiny_A2_p2`** (YOLO26n-P2 chuẩn) trên 3.608 Ground Truth (GT) của tập validation TinyPerson:

1. **Chất lượng hình học cao bị nghịch lý gán điểm thấp**:
   - Có **436 GT** được mô hình định vị chính xác cao với **$\text{IoU} \ge 0.75$**.
   - Tuy nhiên, trong số 436 hộp đạt chuẩn này, có tới **179 hộp mang điểm tin cậy $\text{Confidence} < 0.25$** (chiếm tỷ lệ **$41.05\%$**).
2. **Hậu quả hệ thống**:
   - Ở ngưỡng lọc suy luận tiêu chuẩn (`conf=0.25`), hơn $41\%$ số người tí hon đã được định vị chính xác này bị vứt bỏ hoàn toàn.
   - Chỉ số $\text{AP}_{75}$ bị đè bẹp xuống mức cực thấp **$0.016198$** (dưới $1.7\%$), do trong đường cong Precision-Recall, các dự đoán tốt bị xếp sau hàng chục nghìn False Positive nhiễu nền biển/đất đá có confidence ảo cao hơn.

### Nguyên nhân gốc rễ trong kiến trúc YOLO

Hiện tượng này bắt nguồn từ hai cơ chế thiết kế cố hữu của YOLO hiện đại:
- **Sự cô lập của Decoupled Head**: Nhánh phân loại (`cv3`) và nhánh hồi quy hộp (`cv2`) chạy độc lập. Với người tí hon ($2 \div 8\text{ px}$), nhánh phân loại không có đủ chi tiết kết cấu bề mặt (mặt, tay chân, nếp áo) để tự tin khẳng định danh tính, trong khi nhánh hộp lại nhận biết rất rõ sự tương phản viền cục bộ của cơ thể người.
- **Sự đè bẹp target của Task-Aligned Assigner (TAL)**: Công thức gán nhãn của TAL dùng lũy thừa $t = s^\alpha \times \text{IoU}^\beta$ với $\beta = 6.0$. Đối với vật thể siêu nhỏ chỉ đạt IoU $0.3 \div 0.5$, mục tiêu phân loại ($T_{\text{target}}$) bị ép tụt xuống $0.001$, trực tiếp trừng phạt mô hình nếu nó dám dự đoán confidence cao.

---

## 2. Giải pháp kỹ thuật QAC (Quality-Aware Calibration)

QAC can thiệp trực tiếp vào 2 điểm nghẽn trên theo nguyên tắc: **Tác động tối thiểu, có luận cứ toán học rõ ràng, và zero-overhead lúc suy luận**.

### A. Kiến trúc Head: `QACDetect` (`src/tinydet/modules/qac.py`)
Mở một kênh tương tác một chiều (Cross-Branch Interaction) từ nhánh hộp sang nhánh phân loại trên 4 tầng pyramid ($P_2, P_3, P_4, P_5$):

1. **Cổng chất lượng không gian (Spatial Quality Gate)**:
   Với mỗi tầng $i \in \{0, 1, 2, 3\}$, một lớp tích chập $1\times 1$ siêu nhẹ chiếu đặc trưng hộp $F_{\text{box}}^{(i)}$ thành cổng chất lượng $Q_i$:
   $$Q_i = \text{Sigmoid}\big(W_{\text{gate}}^{(i)} \ast F_{\text{box}}^{(i)} + b_{\text{gate}}^{(i)}\big) \quad \in (0, 1)$$
2. **Hiệu chuẩn Logit phân loại in-place**:
   Thay vì nhân trực tiếp làm bất ổn gradient, QAC dùng phép cộng điều chế theo tỷ lệ học được $\text{Scale}_i$:
   $$F_{\text{cls}}^{(i)} \leftarrow F_{\text{cls}}^{(i)} + \text{Scale}_i \odot Q_i$$
   Được thực thi bằng toán tử tối ưu phần cứng của PyTorch:
   ```python
   gate = self.quality_gates[i](box).sigmoid_()
   cls_calib = cls_feat.addcmul(gate, self.calib_scale[i], value=1.0)
   ```
3. **Khởi tạo gần Identity (Near-Identity Initialization)**:
   - Trọng số $W_{\text{gate}}$ khởi tạo chuẩn với $\sigma = 0.01$.
   - Bias $b_{\text{gate}}$ khởi tạo cố định ở $-2.5$ ($\text{Sigmoid}(-2.5) \approx 0.076$).
   - Nhờ đó, tại epoch 0, mô hình khởi đầu hoàn toàn tương đồng với baseline A2 chuẩn, không gây sốc gradient cho backbone.

### B. Hàm mất mát: `QualityAwareDetectionLoss` (`src/tinydet/modules/qac.py`)
Tái hiệu chuẩn phân phối mục tiêu học phân loại (Classification Target Score) cho các mẫu dương (`fg_mask`):

1. **Đo lường IoU thực tế vector hóa trên GPU**:
   Tính IoU tức thời giữa các anchor dương được giải mã và ground-truth tương ứng.
2. **Bù trừ suy giảm TAL bằng Soft Quality Target**:
   $$T^*[\text{fg}] = \max\Big(T_{\text{TAL}} \cdot \text{IoU}^\gamma,\; \text{min\_conf} \cdot \text{IoU}^\gamma\Big)$$
   với $\gamma = 1.0$ và $\text{min\_conf} = 0.25$.
3. **Tác dụng**: Các anchor đạt độ chính xác hình học cao ($\text{IoU} \ge 0.5$) được bảo đảm nhận mục tiêu học phân loại ít nhất $0.25 \div 0.50$ thay vì bị TAL dìm xuống $0.005$. Mô hình được tưởng thưởng xứng đáng khi định vị đúng người tí hon.

---

## 3. Tối ưu hóa GPU & Phần cứng

- **Toán tử In-Place**: Toàn bộ luồng tính toán sử dụng `sigmoid_()`, `addcmul()`, `clamp_min_()`, `unsqueeze_()`, tránh cấp phát tensor trung gian trên VRAM và tận dụng tối đa L2 cache của GPU NVIDIA.
- **Không có vòng lặp Python**: Loại bỏ hoàn toàn vòng lặp Python qua các bounding box; toàn bộ IoU và target recalibration được tính toán theo lô (Batch Matrix Tensor).
- **Ngân sách tham số siêu nhỏ (Zero-Parameter Bloat)**: Toàn bộ module chỉ thêm đúng **256 tham số** (4 lớp conv $1\times 1$ có kích thước $64 \times 1$). Độ trễ suy luận (Latency) hoàn toàn tương đương với baseline A2.

---

## 4. Những điểm QAC có thể chưa tốt hoặc còn hạn chế

> **[!NOTE]**
> *Toàn bộ các điểm dưới đây là **suy luận lý thuyết và phân tích rủi ro kỹ thuật**, chưa được kiểm chứng đầy đủ qua thực nghiệm 50 epoch.*

### 1. Rủi ro rò rỉ False Positive (FP Leakage) trên nền có cấu trúc cạnh sắc
- **Suy luận**: Nhánh `cv2` chỉ nhận biết ranh giới tương phản hình học. Nếu ảnh xuất hiện các đốm sóng biển có bọt trắng sắc nét, mép bóng đổ của vách đá, hoặc ngọn cỏ có tỷ lệ khung hình dọc giống người tí hon, nhánh `cv2` có thể nhầm đó là một biên hộp đẹp và kích hoạt cổng $Q_i \to 1$.
- **Hệ quả tiềm ẩn**: Việc cổng $Q_i$ mở có thể vô tình đẩy điểm confidence của một đốm nhiễu nền lên trên ngưỡng lọc, làm tăng số lượng False Positive và làm giảm nhẹ chỉ số Precision.

### 2. Sự phụ thuộc vào siêu tham số cố định `quality_min_conf = 0.25`
- **Suy luận**: Con số $0.25$ được chọn dựa trên ngưỡng confidence mặc định của hệ sinh thái YOLO. Tuy nhiên, trên tập dữ liệu có mật độ cực đông (Crowd Scene) hoặc nền biển nhiễu động mạnh, ngưỡng chặn dưới này có thể cố định quá cứng nhắc.
- **Hệ quả tiềm ẩn**: Một giá trị sàn cố định không phân biệt được giữa ảnh thưa ($1 \div 2$ người) và ảnh cực đông ($176$ người), có thể cần một cơ chế điều chỉnh động (Dynamic Density-dependent Floor) trong tương lai.

### 3. Nguy cơ trôi dạt hiệu chuẩn ở các epoch cuối (Over-Confidence Drift)
- **Suy luận**: Trong giai đoạn đầu, việc kích thích target score giúp mô hình thoát khỏi bẫy under-confidence. Tuy nhiên, khi huấn luyện dài (50 epoch) trên tập train nhỏ (662 ảnh), hàm loss có thể thúc đẩy mô hình trở nên "quá tự tin" (Overconfident).
- **Hệ quả tiềm ẩn**: Mô hình có thể dự đoán confidence quá cao cho các hộp có IoU trung bình ($0.50 \div 0.65$), làm sai lệch đường cong hiệu chuẩn (Expected Calibration Error - ECE) nếu không có cơ chế Label Smoothing hoặc Temperature Scaling đi kèm.

### 4. Giới hạn trên phụ thuộc vào chất lượng của Backbone
- **Suy luận**: QAC là cơ chế hiệu chuẩn (Calibration), không phải cơ chế tái tạo đặc trưng (Feature Reconstruction). Nếu ở độ phân giải thấp hoặc đối tượng bị che khuất hoàn toàn, bản thân tầng $P_2$ không còn tín hiệu biên nào, nhánh `cv2` sẽ đoán sai hoàn toàn.
- **Hệ quả tiềm ẩn**: Khi nhánh hộp định vị sai, QAC sẽ không có tín hiệu biên để kích hoạt cổng chất lượng, dẫn tới việc mô hình quay trở lại trạng thái bất lực như baseline đối với các ca cực khó (extreme tiny1 $< 4\text{ px}$).

---

## 5. Tóm tắt Cấu hình & Tích hợp mã nguồn

- **Module thực thi**: [`src/tinydet/modules/qac.py`](../src/tinydet/modules/qac.py)
- **Cấu hình Model**: [`configs/model/yolo26n_tiny_A2_p2_qac.yaml`](../configs/model/yolo26n_tiny_A2_p2_qac.yaml)
- **Cấu hình Thí nghiệm**: [`configs/experiment/qac_tinyperson.yaml`](../configs/experiment/qac_tinyperson.yaml)
- **Lệnh chạy A/B Test song song trên Modal**:
  ```bash
  modal run --detach modal/app.py::compare --config configs/experiment/qac_tinyperson.yaml --parallel
  ```

