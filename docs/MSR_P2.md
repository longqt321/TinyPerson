# MSR trên nhánh P2

## Mục tiêu

Kiểm tra liệu chuẩn hóa residual cục bộ đa tỷ lệ có cải thiện feature P2 cho phát hiện người rất nhỏ hay không. MSR là feature module; thí nghiệm đầu tiên không đổi backbone, neck trước P2, Detect, loss, label assignment hoặc dữ liệu.

## Vị trí trong YOLO26n

Trong A2/A3, layer 2 là feature backbone stride 4. Neck tiếp tục upsample và hợp nhất feature từ backbone stride 8, 16, 32. Layer 19 là output P2 sau hợp nhất. MSR nằm tại layer 29, nhận trực tiếp layer 19; Detect nhận output MSR làm P2 và giữ nguồn gốc cho P3/P4/P5.

```text
backbone stride 32 → upsample + layer 6 (stride 16) → layer 13
                 → upsample + layer 4 (stride 8)  → layer 16
                 → upsample + layer 2 (stride 4)  → layer 19 (P2)
                 → layer 29 (MSR) → Detect P2
```

YAML khai báo module thật, không dùng placeholder:

```yaml
- [19, 1, MSR, [32, [3, 5, 7], 0.000001, 0.001, 10.0, 4, 0.001]]
- [[29, 22, 25, 28], 1, Detect, [nc]]
```

Khi import `tinydet.modules.msr`, class `MSR` được thêm vào namespace `ultralytics.nn.tasks` mà parser Ultralytics tra cứu. YAML này dùng scale `n`, nên channel đầu vào của MSR là 32.

## Luồng tensor và toán học

Input và output có shape `[B,C,H,W]`; ở model nano, `C=32`. Với mỗi kernel $k\in\{3,5,7\}$, MSR tính local mean và variance theo từng channel:

\[
\mu_k=A_k(X),\quad v_k=\max(A_k(X^2)-\mu_k^2,0),\quad
\hat\sigma_k=\max(\sqrt{v_k+\epsilon},\sigma_{min})
\]

\[
Z_k=\operatorname{clip}((X-\mu_k)/\hat\sigma_k,-z_{max},z_{max})
\]

Thống kê chạy FP32; mean và $E[X^2]$ dùng cùng quy tắc padding, không tính padding ngoài ảnh. `eps=1e-6`, `sigma_min=1e-3`, `z_max=10`.

MSR tách $Z_k$ thành `ReLU(Z_k)` và `ReLU(-Z_k)`, nối thành `2C` channel. Một projection riêng cho mỗi kernel gồm Conv 1×1, BatchNorm và SiLU trả feature về C channel, gọi là $F_k$.

GAP và MLP cho mỗi $F_k$ tạo gate theo channel $\alpha_k$. MSR hợp nhất các tỷ lệ bằng tổng có chuẩn hóa:

\[
E=\frac{\sum_k \alpha_k\odot F_k}{\max(\sum_k\alpha_k,10^{-6})}
\]

Conv 1×1 + sigmoid tạo gate không gian-channel $G$. Sau đó depthwise Conv 3×3 refine feature đã điều chỉnh:

\[
Y=X+\gamma\,\operatorname{DWConv}_{3\times3}(X\odot G)
\]

$\gamma$ là scalar học được, khởi tạo `0.001`. Tách đáp ứng dương/âm và học gate theo tỷ lệ là giả thuyết thiết kế, không phải bảo đảm tăng thông tin hoặc tăng AP.

## Thiết kế đối chứng

Nguồn config: `configs/experiment/ablation_msr_p2.yaml`. Dùng cùng dataset, seed 42, 50 epochs, patience 15, imgsz 1280, batch 8, AdamW, lr 0.001, AMP và deterministic bật, augmentation nhẹ giống nhau.

| Run | Cấu hình | So sánh |
|---|---|---|
| E0 | A2, P2 + SPPF | Baseline |
| E1 | A2 + MSR | E1 với E0 đo MSR khi giữ SPPF |
| E2 | A3, P2, bỏ SPPF | Baseline phụ |
| E3 | A3 + MSR | E3 với E2 đo MSR khi bỏ SPPF |

Không so E1 với E2 để quy kết hiệu quả riêng cho MSR. Nếu hiệu quả nhỏ, chạy paired seeds 42/43/44 và báo cáo mean/std.

## Trạng thái evidence

Model YAML đã được parser nhận diện; CPU smoke test chạy criterion YOLO thật, backward, optimizer step, lưu/nạp checkpoint và predict. CUDA AMP chưa xác minh trong môi trường local CPU-only.

Biểu đồ W&B đã chia sẻ cho bốn run tới 50 epoch cho thấy các đường recall, precision và mAP gần nhau. Đây là quan sát sơ bộ từ hình, không thay thế số metric trong CSV/JSON. Chưa thể kết luận MSR có cải thiện. Hãy lấy metric của `best.pt`, so từng cặp và tính AP theo kích thước gốc trước khi viết kết luận.

Profiler Ultralytics ở imgsz 1280 ghi A2+MSR 2,525,313 parameters và 33.554 GFLOPs; A3+MSR 2,360,705 parameters và 33.024 GFLOPs. MSR thêm 9,337 parameters. FLOPs là ước lượng vì profiler có thể bỏ qua pooling và tensor arithmetic trong MSR.

## Chạy

Từ repo root, trong venv đã kích hoạt:

```bash
python scripts/validate_tiny_ablation.py \
  configs/model/yolo26n_tiny_A2_p2_msr.yaml \
  configs/model/yolo26n_tiny_A3_p2_no_sppf_msr.yaml --imgsz 640
python scripts/smoke_msr.py
modal run --detach modal/app.py::compare \
  --config configs/experiment/ablation_msr_p2.yaml --parallel
```

`--smoke` giới hạn mọi model còn 5 epoch; nó chỉ kiểm tra đường chạy, không dùng để kết luận metric. `--parallel` tối đa bốn training job. Modal `--detach` để job sống qua lần terminal disconnect, còn compare vẫn cần kết nối để hiển thị tiến độ và chờ toàn pipeline hoàn tất.

Full training cần Modal GPU và dataset trong volume; không chạy từ máy CPU. Chi tiết benchmark TinyPerson nằm trong [TINYPERSON_BENCHMARK.md](TINYPERSON_BENCHMARK.md).
