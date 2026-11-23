# Style guide: viết như sinh viên thật, không "AI slop"

Mục tiêu: bài báo WACV (tiếng Anh) và báo cáo đồ án (tiếng Việt) phải đọc như do sinh viên thật viết, không như ChatGPT xuất ra. Nguyên tắc chung: **cụ thể thắng trừu tượng, số liệu thắng tính từ, câu ngắn dài đan xen, không từ đệm**.

---

## Part A — English paper checklist (WACV style)

### A1. Banned / near-banned vocabulary
Delete on sight; replace with a plain word or a number:

- **Grandiose verbs/nouns:** delve (into), underscore, showcase, leverage, harness, facilitate, illuminate, foster, navigate, realm, landscape, tapestry, testament, paradigm, synergy, ecosystem, pivotal, transformative, groundbreaking, multifaceted, intricate, comprehensive, robust (unless describing a robustness *experiment*), seamless, novel (overused), cutting-edge, state-of-the-art (as adjective filler — cite instead).
- **Hedging/filler scaffolding:** "It is worth noting that…", "It is important to note/emphasize…", "Notably,", "Crucially,", "In recent years,", "In today's rapidly evolving…", "plays a crucial/vital role", "has garnered significant attention", "remains an open challenge" (unless you then say *which* challenge).
- **Connective spam:** Moreover / Furthermore / Additionally / In addition more than once per section. "Overall," "In summary," at the start of a Conclusion that already summarizes.
- **Correlative crutches:** "not only … but also …", "both X and Y" when a plain list works; "aims to", "seeks to", "is designed to" repeated.

### A2. Structural tells (fix these)
1. **No triads.** Never three parallel adjectives/nouns ("fast, accurate, and robust"). One precise word. If you have three real items, number them.
2. **No em-dash drama.** Max ~1 em dash per page. Prefer commas, colons, parentheses, or two sentences.
3. **No bold-lead-in bullet lists** ("**Efficiency:** our model…"). Bullets only for genuinely enumerable data (hyperparameters, steps). Most content belongs in prose.
4. **Paragraph openings must vary.** Ban starting paragraphs with "Furthermore," / "Moreover," / "However," / "In this section," more than once each. Start with the claim or the fact.
5. **No over-signposting.** One roadmap sentence in the intro is enough; don't open every section with "In this section, we…" and close with "As shown above…".
6. **Conclusion ≠ abstract.** Do not restate the abstract; state what you learned, limits, and what you'd try next.
7. **Uniform paragraph length is a tell.** Let paragraphs be 2–8 sentences, uneven.
8. **No "In this paper, we propose X, which achieves Y, while also enabling Z."** One claim per sentence, backed by a table/figure reference.

### A3. Sentence-variety rules
- Mix lengths: some sentences under 10 words, some over 25. Read aloud; if every sentence has the same rhythm, rewrite.
- Prefer active voice with real agents: "We freeze the backbone" not "The backbone is frozen in our framework".
- Quantify instead of adjectives: not "significantly improves" but "improves AP by 2.1 points (26.4 → 28.5)".
- Report negative results plainly: "Both variants underperform the baseline; we analyze why in Sec. 5." No spin, no "encouragingly".
- Keep first person ("we") — it's normal in CV papers and reads human; avoid passive-everything.
- Citations do the hype work: "YOLO26n [ref] is fast" not "the cutting-edge YOLO26n".

### A4. Do instead
- Lead each subsection with its finding, not its topic.
- When a number is surprising, say so in one clause and move on.
- Admit specific limitations with causes ("our P2 head adds 1.8 ms/iter because…").
- Keep terminology consistent (GauSS, TinyPerson, AP, MR); never synonym-swap technical terms for variety — that's an AI tell too.

---

## Part B — Vietnamese checklist (báo cáo đồ án môn học)

### B1. Cụm từ cấm / hạn chế tối đa
- **Từ nối khuôn mẫu:** "Đầu tiên… Thứ hai… Cuối cùng…", "Hơn nữa", "Bên cạnh đó", "Thêm vào đó", "Ngoài ra" (dùng quá 1 lần/đoạn), "Đáng chú ý là", "Điều này cho thấy rằng" (khi ý đã rõ).
- **Tổng kết giáo điều:** "Tóm lại", "Nhìn chung", "Có thể thấy rằng", "Như đã trình bày ở trên", "Qua quá trình nghiên cứu, chúng ta có thể kết luận rằng…".
- **Mở bài vĩ mô:** "Trong bối cảnh [AI] đang phát triển không ngừng…", "Không thể phủ nhận rằng…", "Ngày nay, …", "đóng vai trò vô cùng quan trọng", "giá trị thiết thực".
- **Từ sáo chung chung:** toàn diện, cốt lõi, đột phá, tối ưu hóa (khi không có số liệu), nâng cao hiệu quả, giải quyết triệt để, đóng góp to lớn.
- **Kết cấu AI:** "không chỉ … mà còn …", "việc + động từ" dồn dập (việc áp dụng việc sử dụng việc đánh giá), "một cách" lặp lại (một cách hiệu quả, một cách chính xác), "dựa trên cơ sở", "trong khuôn khổ đề tài".

### B2. Xưng hô và văn phong
- Chọn một đại từ và giữ nguyên toàn bài: "tôi" (báo cáo cá nhân, như file hiện tại) hoặc "nhóm chúng em" (báo cáo nhóm). Không trộn "tôi / chúng em / chúng ta / người viết".
- Nội dung kỹ thuật (cơ sở lý thuyết, thực nghiệm): ưu tiên câu không ngôi hoặc chủ thể là sự vật — "Mô hình được huấn luyện 100 epoch…", "Bảng 3 cho thấy…". Đại từ chỉ xuất hiện ở lý do chọn đề tài, kết luận, bài học rút ra.
- Không kể công hay than vãn ("chúng em đã rất vất vả…"); viết sự việc + cách khắc phục.
- Không hô hào ở kết bài ("Tóm lại, đề tài đã đạt được những kết quả to lớn…"). Kết luận = những gì làm được (có số), cái chưa làm được, hướng tiếp theo.
- Giữ thuật ngữ nhất quán: GauSS, TinyPerson, stride 4, khung (bounding box) — không đổi từ để "cho đỡ lặp".

### B3. Cấu trúc và nhịp câu
- Mỗi đoạn mở đầu bằng ý chính, không mở đầu bằng từ nối.
- Câu ngắn dài xen kẽ; cho phép câu một vế ngắn khi nhấn mạnh ("Kết quả là âm.").
- Số liệu thay tính từ: không "cải thiện đáng kể" mà "tăng 2,1 điểm AP (26,4 → 28,5)".
- Ghi nhận kết quả âm thẳng thắn, kèm phân tích nguyên nhân — đây chính là thứ làm bài đọc "người thật".
- Bullet chỉ dùng cho danh sách thật (siêu tham số, các bước); phân tích viết thành văn xuôi.
- Không dùng gạch đầu dòng in đậm kiểu "**Ưu điểm:** …" — đó là dấu hiệu AI.

---

## Part C — Rewrite recipe (before → after)

**EN 1.**
- AI: "In today's rapidly evolving field of computer vision, detecting tiny objects remains a pivotal and challenging problem."
- Human: "Detecting a person that occupies 2–8 pixels in an aerial image is still unsolved; TinyPerson [Yu et al., 2020] showed that COCO-strong detectors collapse on this data."

**EN 2.**
- AI: "Our framework not only improves accuracy but also provides a robust and seamless solution."
- Human: "GauSS raises AP from 26.4 to 28.5 and adds 0.3 ms per iteration."

**EN 3.**
- AI: "It is worth noting that the proposed module leverages a multi-scale feature fusion strategy to effectively harness fine-grained information."
- Human: "The module fuses stride-4 features with stride-8 features by concatenation followed by a 1×1 convolution."

**EN 4.**
- AI: "In summary, this paper has presented a comprehensive and novel approach, underscoring its significance for the field."
- Human: "The main lesson is that scale alignment, not model capacity, was the bottleneck in our setting; the negative result on the split head supports this."

**EN 5.**
- AI: "Extensive experiments demonstrate that our method achieves state-of-the-art performance across multiple benchmarks."
- Human: "On TinyPerson (test), GauSS reaches AP 28.5 vs. 26.4 for the P2 baseline; on VisDrone the gain does not transfer (Table 4)."

**VI 1.**
- AI: "Trong bối cảnh thị giác máy tính đang phát triển không ngừng, bài toán phát hiện vật thể nhỏ đóng vai trò vô cùng quan trọng."
- Human: "Một người rơi xuống biển chỉ chiếm 2–8 pixel trong khung hình. Ở cỡ đó, detector huấn luyện trên COCO gần như mù."

**VI 2.**
- AI: "Đầu tiên, chúng em tiến hành khảo sát. Hơn nữa, bên cạnh đó, đề tài còn thực hiện việc đánh giá một cách toàn diện."
- Human: "Trước khi sửa mô hình, tôi đo xem đặc trưng stride 4 còn giữ lại bao nhiêu thông tin sau 5 lần downsample; kết quả nằm ở Hình 3."

**VI 3.**
- AI: "Có thể thấy rằng, module GauSS không chỉ giúp cải thiện độ chính xác một cách đáng kể mà còn tối ưu hóa hiệu năng một cách hiệu quả."
- Human: "GauSS tăng AP từ 26,4 lên 28,5, đổi lại chậm thêm 0,3 ms mỗi lần suy luận."

**VI 4.**
- AI: "Tóm lại, đề tài đã đạt được những kết quả to lớn và có giá trị thiết thực, góp phần thúc đẩy sự phát triển của lĩnh vực."
- Human: "Mục tiêu 3 thất bại: cả hai biến thể head đều dưới baseline. Phân tích ở chương 5 cho thấy nguyên nhân nhiều khả năng nằm ở anchor assignment, không nằm ở head."

**VI 5.**
- AI: "Đáng chú ý là, quá trình thực hiện đề tài đã giúp nhóm chúng em nâng cao kiến thức một cách toàn diện."
- Human: "Lỗi tôi mất một ngày để tìm ra: tôi quên đặt `model.fuse()` trước khi xuất ONNX, nên số liệu latency lệch 20%. Ghi lại ở đây để lần sau không lặp."

---

## Nguồn tham khảo
- Wikipedia: "Signs of AI writing" — https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- Harvard Guide to Using Sources / academic-integrity guidance on AI text (tổng hợp các dấu hiệu "AI slop" trong văn học thuật) — https://usingsources.fas.harvard.edu/ai-guidance
- The Register / Quartz: phân tích spike "delve" trong bài báo hậu-ChatGPT — https://qz.com/llm-favorite-word-delve-1851668839
- Brands Vietnam: "Dấu hiệu nhận biết bài viết do AI" (văn phong máy móc tiếng Việt) — https://www.brandsvietnam.com/
- Hướng dẫn văn phong báo cáo đồ án, xưng hô "chúng em" — https://thuedoan.vn/ và https://ai-hay.vn/
