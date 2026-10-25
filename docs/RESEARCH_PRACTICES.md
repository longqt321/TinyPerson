# Research project practices

Tài liệu này chốt hướng refactor của repo: đây là **research project**, không phải
application project. Mục tiêu là trả lời câu hỏi khoa học bằng bằng chứng tái lập
được với chi phí compute hợp lý, không phải xây sản phẩm cho người dùng cuối.

## Khác biệt cốt lõi

| Hạng mục | Application project | Research project |
|---|---|---|
| Mục tiêu | Hành vi đúng cho người dùng/nghiệp vụ | Câu hỏi giả thuyết và kết luận có bằng chứng |
| Tiêu chí thành công | SLA, độ ổn định, bảo mật, accessibility, chi phí vận hành | Effect size, độ bất định, reproducibility, chi phí thí nghiệm |
| Đơn vị thay đổi | Feature, API, workflow | Giả thuyết, biến kiểm soát, baseline, ablation |
| Test chính | Unit/integration/E2E, regression | Kiểm tra graph, dữ liệu, metric, seed, artifact |
| Kết quả | Release, user outcome | Bảng kết quả, checkpoint, phân tích lỗi, kết luận |
| Kỹ thuật ưu tiên | Bảo mật, migration, rollback, contract | Tốc độ vòng lặp, provenance, so sánh paired, thống kê |
| Kỹ thuật không ưu tiên | Tốc độ thí nghiệm | Bảo mật ứng dụng, nghiệp vụ, UI, test pyramid đầy đủ |

Điểm quan trọng: research project vẫn cần kiểm tra tính hợp lệ của dữ liệu và
metric. Các kiểm tra đó bảo vệ kết luận khoa học, không phải yêu cầu bảo mật
ứng dụng.

## Invariant của repo

1. Mỗi experiment trả lời một câu hỏi falsifiable.
2. Chỉ thay một biến kiểm soát giữa control và treatment.
3. Mọi so sánh dùng cùng dataset, split, seed, imgsz, optimizer và augmentation.
4. Chọn checkpoint bằng validation. Không dùng official test để tune.
5. Metric phải có provenance: checkpoint nào, split nào, evaluator nào.
6. Kết quả cuối báo cáo paired delta và seed variance, không chỉ một run đẹp.
7. Kết quả âm tính được giữ. Không xóa run hoặc đổi metric để có kết luận đẹp.
8. Artifact là nguồn sự thật; W&B chỉ là log phụ.
9. Tối ưu tốc độ bằng reuse, chạy song song và tránh đánh giá lại không cần thiết.
10. Repo không hứa backward compatibility cho API ứng dụng; chỉ giữ tương thích
    checkpoint và protocol cần cho thí nghiệm.

## Áp dụng trong repo

- `configs/experiment/*.yaml` khai báo protocol, `seeds` và `comparisons`.
- `seeds: [42, 43, 44]` mở rộng một experiment thành nhiều run paired.
- `src/tinydet/research/results.py` gom artifact, chọn một primary run cho mỗi
  experiment/seed, tính mean/std và paired delta.
- `tools/list_runs.py --config ... --markdown ...` sinh bảng kết quả từ artifact.
- `docs/MSR_P2.md`, `docs/TINYPERSON_BENCHMARK.md` và `docs/agents/*` giữ quy tắc
  dữ liệu, metric và lifecycle.

## Ngoài phạm vi

- Bảo mật ứng dụng, authorization, multi-tenant, billing.
- Nghiệp vụ sản phẩm và backward compatibility cho public API.
- Test pyramid đầy đủ cho mọi đường ứng dụng.
- UI, accessibility và trải nghiệm người dùng cuối.

## Tham khảo

- Wilson et al. (2014), *Best Practices for Scientific Computing*, PLoS Biology.
- Sandve et al. (2013), *Ten Simple Rules for Reproducible Computational Research*,
  PLoS Computational Biology.
- Munafò et al. (2017), *A manifesto for reproducible science*, Nature Human
  Behaviour.
- Pineau et al. (2021), *Improving Reproducibility in Machine Learning Research*,
  JMLR 22(164).
- Sculley et al. (2015), *Hidden Technical Debt in Machine Learning Systems*,
  NeurIPS.
- Breck et al. (2017), *The ML Test Score*, IEEE Big Data.

Các citation trên là tài liệu nền cho reproducibility và ML research
engineering. Kiểm tra lại metadata citation trước khi dùng trong paper.
