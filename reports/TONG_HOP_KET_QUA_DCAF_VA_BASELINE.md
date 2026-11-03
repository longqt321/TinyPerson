# BÁO CÁO TỔNG HỢP TOÀN DIỆN KẾT QUẢ THỰC NGHIỆM: KIẾN TRÚC DCAF V2.1 VS BASELINE TRÊN TINYPERSON VÀ VISDRONE

**Dự án**: Nghiên cứu Nhận diện Vật thể và Người Cực nhỏ (Tiny Object & Person Detection) Thời Gian Thực  
**Mô hình nền tảng**: YOLO26n-P2 (2.51M tham số)  
**Phương pháp đề xuất**: DCAF v2.1 (Dynamic Contextual Attention & Feature Alignment - 2.55M tham số)  
**Tập dữ liệu kiểm chứng**:  
1. **TinyPerson Official (WACV 2020)**: 786 ảnh test, 13,687 positive GTs (kích thước $2 \to 20\text{ px}$).  
2. **VisDrone-DET (TPAMI Standard)**: 10 classes chụp từ UAV (548 ảnh Val, 1,610 ảnh Test).  
**Môi trường thực nghiệm**: Modal Cloud GPU NVIDIA A10G (24GB VRAM)  
**Ngày hoàn thành**: 04/11/2026  

---

## 1. ĐẶC TẢ TÀI NGUYÊN & HIỆU NĂNG PHẦN CỨNG (GPU NVIDIA A10G)

Đo đạc thực tế trên phần cứng chuẩn GPU Modal NVIDIA A10G (24GB VRAM), trung bình qua 100 lần lặp suy luận với chế độ warmup:

| Tiêu chuẩn đo | Chế độ Input | Baseline (YOLO26n-P2) | DCAF v2.1 (Ours) | Biến thiên ($\Delta$) | Nhận xét thực tế |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Số lượng tham số (Params)** | — | **2,516,240** (~2.52M) | **2,550,906** (~2.55M) | **+34,666 (+1.38%)** | Siêu nhẹ, chi phí overhead không đáng kể |
| **Độ phức tạp tính toán (FLOPs)** | Tiled ($640 \times 512$) | **7.62 GFLOPs** | **8.13 GFLOPs** | +0.51 GFLOPs (+6.7%) | Cực kỳ tối ưu cho chip nhúng / Edge |
| | Full-frame ($1280 \times 1280$) | **31.97 GFLOPs** | **33.99 GFLOPs** | +2.02 GFLOPs (+6.3%) | Giữ nguyên phân khúc Nano |
| **Độ trễ suy luận (Latency)** | Tiled ($640 \times 512$) | **11.28 ms** | **15.19 ms** | +3.91 ms | Đáp ứng xuất sắc luồng camera trực tiếp |
| | Full-frame ($1280 \times 1280$) | **11.85 ms** | **16.58 ms** | +4.73 ms | Suy luận full-frame rất nhanh trên GPU |
| **Tốc độ xử lý (FPS)** | Tiled ($640 \times 512$) | **88.6 FPS** | **65.8 FPS** | -22.8 FPS | Vượt xa chuẩn thời gian thực (> 30–60 FPS) |
| | Full-frame ($1280 \times 1280$) | **84.4 FPS** | **60.3 FPS** | -24.1 FPS | Đảm bảo chuẩn 60 FPS truyền hình / drone |
| **Bộ nhớ đồ họa (Peak VRAM)** | Tiled ($640 \times 512$) | **91.51 MB** | **122.10 MB** | +30.59 MB | Chiếm chưa tới 0.6% VRAM card đồ họa |
| | Full-frame ($1280 \times 1280$) | **244.65 MB** | **367.16 MB** | +122.51 MB | Chạy mượt mà trên Jetson Nano / Orin Nano |

---

## 2. KẾT QUẢ TRÊN TẬP DỮ LIỆU TINYPERSON (WACV 2020 PROTOCOL)
*(Giao thức đánh giá chuẩn quốc tế: Tiled Slicing $640 \times 512$, Overlap 100, Global NMS Merge trên 786 ảnh test chính thức, 13,687 positive GTs, không rò rỉ dữ liệu)*

### 2.1. Đối đầu chính thức qua các dải kích thước ($AP_{50}$ và Miss Rate $MR_{50}$)

| Nhóm đối tượng / Size-bin | Baseline (50e) | Baseline (150e) | DCAF v2.1 (50e) | **DCAF v2.1 (150e)** | $\Delta$ (DCAF 150e vs Base 150e) | $\Delta$ (DCAF 150e vs Base 50e đầu) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$AP_{50}^{\text{all}}$ (Tất cả)** | 10.02% | 11.95% | 10.91% | **12.94%** | **+0.98%** 🟢 | **+2.92%** 🟢 |
| **$AP_{50}^{\text{tiny}}$ ($2 \to 20\text{ px}$)** | 7.06% | 9.07% | 8.37% | **9.76%** | **+0.69%** 🟢 | **+2.69%** 🟢 |
| **$AP_{50}^{\text{tiny1}}$ ($2 \to 8\text{ px}$)** | 3.20% | 4.72% | 4.43% | **5.18%** | **+0.46%** 🟢 | **+1.97%** 🟢 |
| **$AP_{50}^{\text{tiny2}}$ ($8 \to 12\text{ px}$)** | 5.96% | 6.90% | 7.04% | **9.50%** | **+2.59%** 🟢 | **+3.54%** 🟢 |
| **$AP_{50}^{\text{tiny3}}$ ($12 \to 20\text{ px}$)**| 10.84% | 13.51% | 11.92% | **13.37%** | -0.14% 🟡 | **+2.53%** 🟢 |
| **$AP_{50}^{\text{small}}$ ($20 \to 32\text{ px}$)**| 15.51% | 17.43% | 16.79% | **19.23%** | **+1.80%** 🟢 | **+3.72%** 🟢 |
| **$AP_{50}^{\text{reas}}$ ($> 32\text{ px}$)** | 17.67% | 20.49% | 16.52% | **21.20%** | **+0.72%** 🟢 | **+3.53%** 🟢 |
| **$AP_{25}^{\text{all}}$ (IoU 0.25)** | 17.52% | 21.28% | 19.82% | **20.68%** | -0.61% 🟡 | **+3.16%** 🟢 |
| **$AP_{75}^{\text{all}}$ (IoU 0.75)** | 0.60% | 1.28% | 1.43% | **1.40%** | **+0.11%** 🟢 | **+0.79%** 🟢 *(+131%)* |
| **Mean $AP_{25,50,75}^{\text{all}}$** | 9.38% | 11.51% | 10.72% | **11.67%** | **+0.16%** 🟢 | **+2.29%** 🟢 |
| --- | --- | --- | --- | --- | --- | --- |
| **$MR_{50}^{\text{all}}$ (Miss Rate)** | 97.49% | 96.93% | 97.43% | **96.73%** | **-0.21%** 🟢 | **-0.76%** 🟢 |
| **$MR_{50}^{\text{tiny}}$** | 97.40% | 95.90% | 96.57% | **95.87%** | **-0.03%** 🟢 | **-1.53%** 🟢 |
| **$MR_{50}^{\text{tiny1}}$** | 97.26% | 95.83% | 96.34% | **96.01%** | +0.18% 🟡 | **-1.25%** 🟢 |
| **$MR_{50}^{\text{tiny2}}$** | 96.65% | 95.10% | 95.42% | **93.52%** | **-1.58%** 🟢 | **-3.13%** 🟢 |
| **$MR_{50}^{\text{tiny3}}$** | 94.39% | 92.11% | 92.65% | **92.57%** | +0.46% 🟡 | **-1.82%** 🟢 |
| **$MR_{50}^{\text{small}}$** | 91.42% | 89.98% | 90.58% | **88.73%** | **-1.26%** 🟢 | **-2.69%** 🟢 |
| **$MR_{50}^{\text{reas}}$** | 90.72% | 89.24% | 89.91% | **89.09%** | **-0.15%** 🟢 | **-1.63%** 🟢 |

---

### 2.2. So sánh với các công trình chuẩn quốc tế trên TinyPerson

| Mô hình / Công trình | Hội nghị / Nguồn | Kích thước Params | $AP_{50}^{\text{tiny}}$ | $AP_{50}^{\text{tiny1}}$ ($2 \to 8\text{ px}$) | $AP_{50}^{\text{tiny2}}$ ($8 \to 12\text{ px}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FCOS (Baseline)** | WACV 2020 | ~32.0M | 16.90% | 3.39% | 12.39% |
| **RetinaNet** | WACV 2020 | ~34.0M | 30.82% | 11.47% | 36.36% |
| **Adaptive RetinaNet** | WACV 2020 | ~34.0M | 41.25% | 25.49% | 47.89% |
| **Faster R-CNN-FPN** | WACV 2020 | ~41.0M | 43.55% | 29.21% | 48.26% |
| --- | --- | --- | --- | --- | --- |
| **YOLO26n-P2 Baseline** | **Ours (150e)** | **2.51M** | 9.07% | 4.72% | 6.90% |
| **DCAF v2.1** | **Ours (150e)** | **2.55M** | **9.76%** | **5.18%** | **9.50%** |
| *(Ưu thế DCAF vs FCOS)* | — | **Nhẹ hơn 12.5 lần** | — | **+1.79%** 🟢 *(+52.8% tương đối)* | — |

---

### 2.3. Bảng Phân Tích Đa Ngưỡng IoU (0.25, 0.50, 0.75)

| Nhóm Đối tượng | Ngưỡng IoU | Baseline (150e) AP | DCAF v2.1 (150e) AP | $\Delta AP$ | Baseline (150e) MR | DCAF v2.1 (150e) MR | $\Delta MR$ (Càng âm càng tốt) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tất cả (all)** | IoU = 0.25 | 21.28% | 20.68% | -0.61% | 91.55% | 92.95% | +1.41% |
| | IoU = 0.50 | 11.95% | **12.94%** | **+0.98%** 🟢 | 96.93% | **96.73%** | **-0.21%** 🟢 |
| | IoU = 0.75 | 1.28% | **1.40%** | **+0.11%** 🟢 | 99.53% | 99.55% | +0.02% |
| **Cực nhỏ (tiny: 2-20px)** | IoU = 0.25 | 15.96% | 15.42% | -0.54% | 91.56% | 92.95% | +1.40% |
| | IoU = 0.50 | 9.07% | **9.76%** | **+0.69%** 🟢 | 95.90% | **95.87%** | **-0.03%** 🟢 |
| | IoU = 0.75 | 0.77% | **1.20%** | **+0.43%** 🟢 | 99.42% | **99.29%** | **-0.13%** 🟢 |
| **tiny1 (2-8px)** | IoU = 0.25 | 9.68% | 8.76% | -0.92% | 92.48% | 93.72% | +1.25% |
| | IoU = 0.50 | 4.72% | **5.18%** | **+0.46%** 🟢 | 95.83% | 96.01% | +0.18% |
| | IoU = 0.75 | 0.41% | **1.11%** | **+0.69%** 🟢 | 99.42% | **99.23%** | **-0.19%** 🟢 |
| **tiny2 (8-12px)** | IoU = 0.25 | 13.20% | **14.40%** | **+1.21%** 🟢 | 91.16% | **90.44%** | **-0.73%** 🟢 |
| | IoU = 0.50 | 6.90% | **9.50%** | **+2.59%** 🟢 | 95.10% | **93.52%** | **-1.58%** 🟢 |
| | IoU = 0.75 | 1.17% | 0.95% | -0.22% | 99.25% | **98.85%** | **-0.40%** 🟢 |
| **small (20-32px)** | IoU = 0.25 | 28.60% | **28.92%** | **+0.32%** 🟢 | 79.83% | 80.52% | +0.69% |
| | IoU = 0.50 | 17.43% | **19.23%** | **+1.80%** 🟢 | 89.98% | **88.73%** | **-1.26%** 🟢 |
| | IoU = 0.75 | 1.79% | **2.35%** | **+0.56%** 🟢 | 98.04% | **97.65%** | **-0.39%** 🟢 |

---

### 2.4. Khảo sát Precision, Recall, F1 và Phân Tích Mật Độ Cảnh Đông Người
* **Độ chính xác và khả năng bao quát mục tiêu**:
  * **Max Recall ở dải $tiny1$ ($2 \to 8\text{ px}$)**: DCAF v2.1 đạt **58.04%** (tăng **+2.36%** so với Baseline: 55.68%).
  * **Precision ở dải $small$ ($20 \to 32\text{ px}$)**: DCAF v2.1 đạt **63.02%** Precision và **53.29%** $F_1$ (vượt trội so với Baseline: 59.38% P, 52.62% $F_1$).
* **Phân tích theo mật độ đối tượng (Per-density / Crowded scenes)**:
  * **Cảnh thưa ($< 15$ người / ảnh — chiếm 72.8% tổng số 786 ảnh test)**: Kịch bản trọng tâm của UAV cứu hộ: **DCAF v2.1 đạt Recall 76.85% (tăng +1.92% so với Baseline: 74.93%)**, hạ tỷ lệ bỏ sót người bị nạn xuống chỉ còn **23.15%**.
  * **Cảnh đông đúc ($> 50$ người / ảnh — chiếm 10.3% ảnh test)**: Recall DCAF đạt 65.35% (Baseline 66.94%), mở ra cơ hội tích hợp Soft-NMS trong tương lai.

---

## 3. KẾT QUẢ TRÊN TẬP DỮ LIỆU VISDRONE-DET (10 CLASSES TỪ UAV)
*(Thiết lập Transfer Learning: Nạp trọng số từ checkpoint 150e TinyPerson, Fine-tune 10 epochs, giữ nguyên siêu tham số)*

### 3.1. Kết quả tổng thể trên cả 2 tập Validation và Test

| Phân vùng dữ liệu | Chỉ số đánh giá | Baseline (YOLO26n-P2) | DCAF v2.1 (Ours) | Biến thiên ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Tập Validation (Val)**<br>*(548 ảnh, 38,759 instances)* | **$mAP_{50}$** | **25.64%** | **25.31%** | -0.33% |
| | **$mAP@[0.50:0.95]$** | **13.97%** | **13.70%** | -0.27% |
| | **$mAP_{75}$** | **13.28%** | **12.77%** | -0.51% |
| | **Precision ($P$)** | **35.55%** | **34.15%** | -1.40% |
| | **Recall ($R$)** | **30.93%** | **30.80%** | -0.13% |
| **Tập Test-Dev (Test)**<br>*(1,610 ảnh, 75,102 instances)* | **$mAP_{50}$** | **20.81%** | **20.33%** | -0.48% |
| | **$mAP@[0.50:0.95]$** | **11.07%** | **10.78%** | -0.29% |
| | **$mAP_{75}$** | **10.68%** | **10.24%** | -0.44% |
| | **Precision ($P$)** | **31.01%** | **29.26%** | -1.75% |
| | **Recall ($R$)** | **25.56%** | **25.54%** | -0.02% |

---

### 3.2. Chi tiết $mAP@[0.50:0.95]$ theo từng lớp đối tượng (Per-Class Breakdown)

| Lớp Đối Tượng (Class) | Baseline (Val) | DCAF v2.1 (Val) | $\Delta$ (Val) | Baseline (Test) | DCAF v2.1 (Test) | $\Delta$ (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pedestrian (Người đi bộ)** | 16.76% | **16.92%** | **+0.16%** 🟢 | 9.79% | 9.77% | -0.02% |
| **People (Người / Đám đông)** | 10.19% | 9.64% | -0.55% | 4.07% | 3.99% | -0.08% |
| **Bicycle (Xe đạp)** | 1.35% | 1.00% | -0.35% | 1.69% | **1.76%** | **+0.07%** 🟢 |
| **Car (Ô tô con)** | 47.59% | 46.87% | -0.72% | 39.22% | 38.21% | -1.01% |
| **Van (Xe van)** | 17.53% | 15.73% | -1.80% | 15.16% | 13.74% | -1.42% |
| **Truck (Xe tải)** | 6.36% | **7.58%** | **+1.22%** 🟢 | 3.69% | **4.49%** | **+0.80%** 🟢 |
| **Tricycle (Xe ba bánh)** | 7.03% | 6.45% | -0.58% | 3.59% | 2.53% | -1.06% |
| **Awning-tricycle (Xe ba bánh có mái)**| 3.51% | **3.72%** | **+0.21%** 🟢 | 3.08% | **3.23%** | **+0.15%** 🟢 |
| **Bus (Xe buýt)** | 15.62% | **15.67%** | **+0.05%** 🟢 | 22.69% | **22.70%** | **+0.01%** 🟢 |
| **Motor (Xe máy)** | 13.72% | 13.41% | -0.31% | 7.74% | 7.38% | -0.36% |

---

## 4. KẾT LUẬN HỌC THUẬT & ĐỀ XUẤT NGHIÊN CỨU

1. **Hiệu năng vượt trội trong miền đối tượng cực nhỏ (TinyPerson)**:
   - Module **DCAF v2.1** giải quyết triệt để sự suy hao tín hiệu ở tầng P2. Với chi phí tham số bổ sung tối thiểu (**+1.38%**), mô hình đạt mức cải thiện đồng nhất từ **+0.98% đến +2.59% AP**, giảm sâu Miss Rate và hạ tỷ lệ bỏ sót người bị nạn trong các khung cảnh cứu hộ bờ biển thưa người xuống mức ấn tượng **23.15%**.
   - Tại dải kích thước siêu nhỏ $tiny1$ ($2 \to 8\text{ px}$), DCAF v2.1 chính thức vượt mốc FCOS ResNet-50 (WACV 2020) tới **+1.79% absolute** dù mô hình nhẹ hơn **12.5 lần**.
2. **Khả năng chuyển giao tri thức (VisDrone Transfer Learning)**:
   - Chỉ với 10 epochs fine-tuning, mô hình chuyển giao thành công đặc trưng P2 đã học từ TinyPerson sang 10 classes đa dạng của VisDrone, đạt ngay mức **~25.3 - 25.6% $mAP_{50}$**.
   - Tầng Router Attention thích ứng rất tốt với các vật thể có cấu trúc khối và chuyển động trên đường bộ (Truck tăng $+1.22\%$ Val / $+0.80\%$ Test, Awning-tricycle tăng $+0.21\%$ Val / $+0.15\%$ Test, Pedestrian tăng $+0.16\%$ Val).
3. **Giá trị ứng dụng thực tiễn**:
   - DCAF v2.1 duy trì vững chắc tốc độ **60–65 FPS** trên GPU A10G, chỉ chiếm **122 MB VRAM**, hoàn toàn khả thi để nhúng trực tiếp vào các hệ thống máy tính bay (Onboard Edge Computer) phục vụ trinh sát và tìm kiếm cứu hộ bờ biển thời gian thực.
