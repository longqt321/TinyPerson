# HƯỚNG DẪN TRA CỨU & BẢN ĐỒ TÀI LIỆU THAM KHẢO (BIBLIOGRAPHY GUIDE)
## DỰ ÁN: GauSS (Gaussian Scale-Space Alignment & Refinement)

File cơ sở dữ liệu trích dẫn chuẩn BibTeX nằm tại [`paper/references.bib`](file:///home/longtran/Projects/thigiacmaytinh/paper/references.bib) (**71 bài báo kinh điển và SOTA**).

Dưới đây là bảng phân loại chi tiết theo từng chuyên đề giúp bạn dễ dàng tra cứu, hiểu rõ bối cảnh và trích dẫn chuẩn xác vào bài báo:

---

### NHÓM 1: CÁC BỘ DỮ LIỆU & CHUẨN ĐÁNH GIÁ (DATASETS & BENCHMARKS)
*Nhóm này dùng để trích dẫn trong phần **Introduction** và **Experimental Setup**.*

| Citation Key | Tên bài báo / Tác giả | Hội nghị / Tạp chí | Vai trò trong bài báo của chúng ta |
| :--- | :--- | :---: | :--- |
| `yu2020tinyperson` | *Scale Match for Tiny Person Detection* (Yu et al.) | WACV 2020 | **Bài báo gốc khai sinh TinyPerson**. Nguồn gốc phân chia các dải size-bin ($tiny1, tiny2, tiny3$). |
| `zhu2021visdrone` | *Vision Meets Drones: Past, Present and Future* (Zhu et al.) | IEEE TPAMI 2021 | **Chuẩn đánh giá VisDrone** 10 classes quay từ camera UAV. |
| `wang2020aitod` | *Tiny Object Detection in Aerial Images* (Wang et al.) | ICPR 2020 | Bộ dữ liệu đối tượng siêu nhỏ nhìn từ ảnh vệ tinh / hàng không. |
| `lin2014coco` | *Microsoft COCO: Common Objects in Context* (Lin et al.) | ECCV 2014 | Bộ dữ liệu gốc sinh ra chuẩn tính $mAP@[0.50:0.95]$. |

---

### NHÓM 2: CÁC KIẾN TRÚC DETECTOR NỀN TẢNG & MÔ HÌNH ĐỐI CHUẨN
*Nhóm này dùng để viết **Related Work** và so sánh trong **Bảng kết quả chính**.*

| Citation Key | Tên bài báo / Tác giả | Hội nghị / Tạp chí | Vai trò trong bài báo của chúng ta |
| :--- | :--- | :---: | :--- |
| `tian2019fcos` | *FCOS: Fully Convolutional One-Stage Object Detection* (Tian et al.) | ICCV 2019 | **Đối thủ đối chứng trực tiếp**: GauSS vượt FCOS ở dải $tiny1$ dù nhẹ hơn 12.5 lần. |
| `lin2017fpn` | *Feature Pyramid Networks for Object Detection* (Lin et al.) | CVPR 2017 | Khái niệm tháp đặc trưng đa tầng (P2–P5). |
| `liu2018panet` | *Path Aggregation Network for Instance Segmentation* (Liu et al.) | CVPR 2018 | Cấu trúc truyền đặc trưng ngược Bottom-up Path trong cổ mạng. |
| `lin2017focal` | *Focal Loss for Dense Object Detection / RetinaNet* (Lin et al.) | ICCV 2017 | Mô hình SOTA RetinaNet và hàm mất mát Focal Loss xử lý mất cân bằng mẫu. |
| `ren2015faster` | *Faster R-CNN: Towards Real-Time Object Detection* (Ren et al.) | NeurIPS 2015 | Mô hình 2 giai đoạn kinh điển dùng so sánh trong bài. |
| `cai2019cascade` | *Cascade R-CNN: High Quality Object Detection* (Cai et al.) | IEEE TPAMI 2019 | Kiến trúc phát hiện ngưỡng IoU cao nhiều tầng. |
| `redmon2016yolo` | *You Only Look Once: Unified, Real-Time Object Detection* (Redmon et al.) | CVPR 2016 | Nền tảng nguyên lý phát hiện thời gian thực một giai đoạn. |
| `bochkovskiy2020yolov4`| *YOLOv4: Optimal Speed and Accuracy* (Bochkovskiy et al.) | ArXiv 2020 | Đóng góp về Bag-of-Freebies và PANet cải tiến. |
| `wang2023yolov7` | *YOLOv7: Trainable bag-of-freebies* (Wang et al.) | CVPR 2023 | Kiến trúc E-ELAN và quy chuẩn real-time hiện đại. |
| `wang2024yolov10` | *YOLOv10: Real-Time End-to-End Object Detection* (Wang et al.) | ArXiv 2024 | Xu hướng loại bỏ NMS và tối ưu hóa latency cho dòng YOLO. |
| `ge2021yolox` | *YOLOX: Exceeding YOLO Series in 2021* (Ge et al.) | ArXiv 2021 | Kiến trúc Anchor-free Decoupled Head. |
| `carion2020detr` | *End-to-End Object Detection with Transformers* (Carion et al.) | ECCV 2020 | Kiến trúc Transformer trong phát hiện vật thể. |
| `zhu2020deformabledetr`| *Deformable DETR* (Zhu et al.) | ICLR 2021 | Chú ý phân rã cục bộ cho vật thể đa tỷ lệ. |

---

### NHÓM 3: LÝ THUYẾT KHÔNG GIAN TỶ LỆ GAUSS & TOÁN HỌC (SCALE-SPACE FOUNDATIONS)
*Nhóm này dùng để củng cố cơ sở lý thuyết toán học trong phần **Methodology** của GauSS.*

| Citation Key | Tên bài báo / Tác giả | Năm / Xuất bản | Cơ sở lý luận cho GauSS |
| :--- | :--- | :---: | :--- |
| `witkin1983scale` | *Scale-space filtering* (Andrew Witkin) | IJCAI 1983 | **Bài báo đặt nền móng lý thuyết Scale-Space**: Chứng minh rằng phép lọc Gauss là toán tử duy nhất không sinh thêm cấu trúc giả khi làm mịn ảnh. |
| `koenderink1984structure`| *The structure of images* (Jan Koenderink) | Biol. Cybern. 1984 | Chứng minh tương đương giữa không gian tỷ lệ Gauss và phương trình truyền nhiệt (Diffusion equation). |
| `lindeberg1994scale` | *Scale-space theory in computer vision* (Tony Lindeberg) | Springer 1994 | Chuyên khảo kinh điển về việc chọn lọc tỷ lệ đặc trưng tự động. |
| `lowe2004sift` | *Distinctive image features from scale-invariant keypoints* (David Lowe) | IJCV 2004 | Thuật toán SIFT khai thác hiệu hai phân phối Gauss (DoG) bắt điểm đặc trưng cực tiểu. |
| `mallat1989wavelet` | *A theory for multiresolution signal decomposition* (Stephane Mallat) | IEEE TPAMI 1989 | Cơ sở giải tích biến đổi đa phân giải (Multi-resolution analysis). |
| `villani2009optimal` | *Optimal Transport: Old and New* (Cédric Villani) | Springer 2009 | Cơ sở toán học cho việc tính khoảng cách giữa các phân phối Gauss (Wasserstein Metric). |

---

### NHÓM 4: KỸ THUẬT CHUYÊN BIỆT CHO VẬT THỂ CỰC NHỎ (TINY OBJECT TECHNIQUES)
*Nhóm này dùng để so sánh sâu trong phần **Related Work** và **Ablation Studies**.*

| Citation Key | Tên bài báo / Tác giả | Hội nghị / Năm | Giá trị tham chiếu |
| :--- | :--- | :---: | :--- |
| `akyon2022sahi` | *Slicing Aided Hyper Inference (SAHI)* (Akyon et al.) | ICIP 2022 | Cơ sở cho kỹ thuật Tiled Slicing crop ảnh nhỏ khi suy luận. |
| `duan2022sodnet` | *Small Object Detection in Aerial Images (SODNet)* (Duan et al.) | Remote Sens. 2022 | Mô hình SOTA TinyPerson đạt AP cao nhờ Scale Match. |
| `wang2021nwd` | *A Normalized Gaussian Wasserstein Distance (NWD)* (Wang et al.) | ICPR 2022 | Mô hình hóa bounding box cực nhỏ thành phân phối xác suất Gauss 2 chiều để tính IoU. |
| `chen2021rfla` | *RFLA: Gaussian Receptive Field based Label Assignment* (Xu et al.) | ECCV 2022 | Sử dụng receptive field Gauss để gán nhãn cho người cực nhỏ. |
| `kisantal2019augmentation`| *Augmentation for small object detection* (Kisantal et al.) | ArXiv 2019 | Kỹ thuật Copy-Paste tăng cường mẫu người nhỏ. |

---

### NHÓM 5: CƠ CHẾ CHÚ Ý & ĐỊNH TUYẾN ĐẶC TRƯNG (ATTENTION & ROUTING)
*Nhóm này giải thích cơ chế **Router Attention** của GauSS.*

| Citation Key | Tên bài báo / Tác giả | Hội nghị / Năm | Ứng dụng trong GauSS |
| :--- | :--- | :---: | :--- |
| `tan2020efficientdet` | *EfficientDet / BiFPN* (Tan et al.) | CVPR 2020 | Ý tưởng gán trọng số thích ứng có thể học được cho các tầng đặc trưng. |
| `dai2021dyhead` | *Dynamic Head* (Dai et al.) | CVPR 2021 | Chú ý động hợp nhất không gian, kênh và mức tỷ lệ. |
| `woo2018cbam` | *CBAM: Convolutional Block Attention Module* (Woo et al.) | ECCV 2018 | Chú ý kết hợp Channel + Spatial. |
| `hu2018senet` | *Squeeze-and-Excitation Networks* (Hu et al.) | CVPR 2018 | Khái niệm nén và tái kích hoạt kênh thông tin. |
| `wang2020eca` | *ECA-Net: Efficient Channel Attention* (Wang et al.) | CVPR 2020 | Chú ý kênh 1D siêu nhẹ không làm giảm số chiều. |
| `vaswani2017attention` | *Attention Is All You Need* (Vaswani et al.) | NeurIPS 2017 | Nền tảng cơ chế tính điểm tương đồng Scaled Dot-Product Attention. |
| `dai2017deformable` | *Deformable Convolutional Networks* (Dai et al.) | ICCV 2017 | Toán tử tích chập biến dạng theo hình dáng mục tiêu. |

---

### NHÓM 6: HÀM MẤT MÁT, GÁN NHÃN & HẬU XỬ LÝ (OPTIMIZATION & INFERENCE)
*Nhóm này dùng để chứng minh sự chặt chẽ của **Training & Inference Protocol**.*

| Citation Key | Tên bài báo / Tác giả | Hội nghị / Năm | Minh chứng trong thực nghiệm |
| :--- | :--- | :---: | :--- |
| `li2020gfl` / `li2021gflv2`| *Generalized Focal Loss (GFL v1 & v2)* (Li et al.) | NeurIPS 2020 / CVPR 2021 | Cơ sở cho hàm mất mát Distribution Focal Loss (DFL). Giải thích lý do giữ `reg_max=1` để cô lập biến số. |
| `feng2021tood` | *Task-aligned One-stage Object Detection* (Feng et al.) | ICCV 2021 | Cơ chế gán nhãn tối ưu hóa đồng thời phân loại và hồi quy hộp bao. |
| `zheng2020diou` | *Distance-IoU Loss* (Zheng et al.) | AAAI 2020 | Đo đạc khoảng cách tâm giữa hai hộp bao để tối ưu hóa hồi quy. |
| `loshchilov2017adamw` | *Decoupled Weight Decay Regularization (AdamW)* (Loshchilov et al.) | ICLR 2019 | Thuật toán tối ưu hóa mà chúng ta dùng trong suốt 150 epochs. |
| `loshchilov2016sgdr` | *SGDR: Cosine Annealing Learning Rate* (Loshchilov et al.) | ICLR 2017 | Lịch trình học Cosine Annealing đưa mô hình hội tụ ổn định. |
| `bodla2017softnms` | *Soft-NMS* (Bodla et al.) | ICCV 2017 | Kỹ thuật làm mềm điểm tự tin thay vì xóa cứng box trong cảnh đông người. |
| `bolya2020tide` | *TIDE: A General Toolbox for Identifying Errors* (Bolya et al.) | ECCV 2020 | Phương pháp luận phân rã lỗi False Positive / False Negative. |

---

### Cách sử dụng ngay:
- Khi viết bài bằng LaTeX, bạn chỉ cần đặt `\bibliography{references}` ở cuối bài và trích dẫn theo cú pháp `\cite{yu2020tinyperson}`, `\cite{tian2019fcos}`, `\cite{witkin1983scale}`.
- Không cần phải nhớ thông tin chi tiết của bài báo hay dùng Zotero, toàn bộ mã nguồn trích dẫn đã được kiểm tra tính chính xác và lưu trữ vĩnh viễn trong repo của bạn.
