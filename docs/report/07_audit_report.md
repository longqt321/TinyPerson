# 07 — Audit report: `paper/main.tex` + `docs/report/course_report_vi.md`

Audit date: 2026-11-12. Ground truth: `reports/eval/{baseline,gauss_v3}_tinyperson_wacv_150e_{tiled,full}.json`, `reports/eval/ablation/`, `reports/eval/profiling_results.json`. No source files were modified.

## A. NEGATIVE-RESULT / SECOND-MODULE LEAKS

1. **`paper/references.bib:618`** — comment `% Second-instance splitting, point-based counting, and frozen-BN practice` directly names the forbidden second-instance experiment. Any reader of the source sees it. **Fix:** delete the comment block (lines 617–619).
2. **`paper/figures/16_second_box_outcomes.pdf`, `17_second_instance_threshold_sweep.pdf`, `18_frozen_bn_guard.pdf`** — unused figure files with the forbidden experiment's names, shipped next to the paper. Not rendered, but a camera-ready bundle or figure script will expose them. **Fix:** move to `backups/` or delete before any submission packaging.
3. **Ablation reference provenance (see C-14)** — `reports/eval/ablation/bootstrap_headline.json` and `scratch/bootstrap_ablation_allcells.py:33-37` label the "Full GauSS (150e)" reference as the **`dcaf`** model (`local_eval/dcaf_predictions_tiled.json`, `reports/eval/dcaf_tiled_per_class.json` — now deleted). `reports/eval/b1_test_t070.json` and `g4_test_p030.json` point at `gausssplit_*` checkpoints. The paper text contains no leak, but if the 29.63 reference row comes from a GauSS-split/DCAF checkpoint rather than `gauss_v3`, the ablation table silently mixes a second architecture into the "one module" story. **Fix (mandatory before submission):** re-evaluate the 150-epoch `gauss_v3` checkpoint per-class (`sea_person`) and regenerate the ablation reference from that JSON, or document in the caption that the reference is a different checkpoint of the same module.
4. Otherwise **OK**: `main.tex` and `course_report_vi.md` contain no "we also tried", "second variant", "split head", "did not help" sentences. The sub-pixel-offset-head future work (main.tex:522, report:275) is framed as an extension, not a failed attempt — acceptable.

## B. CITATION AUDIT

All 50 `\cite` keys resolve; `main.bbl` has 50 entries; style is IEEE-numbered via `ieee.bst` — **OK**. 21 bib entries are never cited (`bodla2017softnms`, `bolya2020tide`, `carion2020detr`, `dai2017deformable`, `duan2022sodnet`, `ge2021yolox`, `girshick2015fastrcnn`, `he2016resnet`, `kisantal2019augmentation`, `lin2014coco`, `liu2020cbnet`, `lowe2004sift`, `mallat1989wavelet`, `redmon2016yolo`, `rezatofighi2019giou`, `vaswani2017attention`, `villani2009optimal`, `wang2023yolov7`, `wang2024yolov10`, `witkin1986space`, `zhu2020deformabledetr`) — harmless but should be pruned.

### B.1 Verified mismatches (paper/references.bib)

1. **`wang2020aitod`: authors** — current `Wang, Jinwang; Yang, Chang; Duan, Jie; Ni, Kang; Gao, Yuan; Ye, Qixiang` → correct `Wang, Jinwang; Yang, Wen; Guo, Haowen; Zhang, Ruixiang; Xia, Gui-Song` (ICPR 2020, pp. 3791–3798, year 2021 in proceedings). The current author list appears to be **invented** (source: semanticscholar.org / ICPR proceedings).
2. **`wang2021nwd`: venue** — current `ICPR 2022, pages 1–8` → correct: **arXiv preprint arXiv:2110.13389, 2021** (never published at ICPR; the ICPR entry is fabricated). Source: arxiv.org/abs/2110.13389.
3. **`zhu2021visdrone`: title + pages + year** — current `Vision Meets Drones: Past, Present and Future, TPAMI 44(11):7680–7700, 2021` → correct TPAMI version: **"Detection and Tracking Meet Drones Challenge", TPAMI 44(11):7380–7399, 2022, DOI 10.1109/TPAMI.2021.3119563**. Pages 7680–7700 do not exist for this paper. Source: computer.org / DOI.
4. **`cai2019cascade`: year** — current `year={2019}` → correct **2021** (TPAMI 43(5):1483–1498, DOI 10.1109/TPAMI.2019.2956516; online 2019, print 2021). Source: ieeexplore.
5. **`pintea2021resolution`: journal + pages + author** — current `TPAMI 43(7):2233–2245` → correct **IEEE Transactions on Image Processing, vol. 30, pp. 8342–8353, 2021**; author `Goes, F. Xavier` → **`Goes, Stanley F.`**. The TPAMI volume/pages are fabricated. Source: arXiv 2104.12659 / IEEE TIP.
6. **`jiang2021smplus`: author** — current `Peng, Xingpeng` → correct **`Peng, Xiaoke`** (Nan Jiang, Xuehui Yu, Xiaoke Peng, Yuqi Gong, Zhenjun Han). Venue/pages OK. Source: arXiv 2105.07585 / ICASSP 2021.
7. **`sosnovik2019scale`: pages** — `pages={1--14}` on an ICLR entry; ICLR has no page numbers. Drop the field. (Year 2020 is correct; key name is stale.)
8. **`worrall2019scale`: pages** — `pages={1--11}` on a NeurIPS entry; NeurIPS proceedings have no such pagination. Drop the field.
9. **`gong2021fusionfactor`: entry type** — `@article` with `journal={Proceedings of ... WACV}` → change to `@inproceedings, booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}`.
10. **`yu2020tinyperson`: pages** — `1257–1265` conflicts with IEEE Xplore listing `1246–1254`. Verify against IEEE Xplore and use one consistently.
11. **`ultralytics2026yolo`: year** — YOLO26 was announced by Ultralytics in Sept 2025; `year={2026}` is defensible only if tied to a 2026 release page. Cite the specific docs URL and add a `note` or use `year={2025}`.

### B.2 Verified OK (spot-checked metadata correct)
`lin2017fpn`, `liu2018panet`, `lin2017focal`, `ren2015faster`, `tan2020efficientdet`, `dai2021dyhead`, `akyon2022sahi`, `li2020gfl`, `li2021gflv2`, `feng2021tood`, `witkin1983scale`, `koenderink1984structure`, `lindeberg1994scale`, `burt1983laplacian`, `howard2017mobilenets`, `chollet2017xception`, `ioffe2015batch`, `elfwing2018silu`, `ramachandran2017swish`, `shi2016pixelshuffle`, `sunkara2022spdconv`, `deng2009imagenet`, `russakovsky2015imagenet`, `zheng2020diou`, `neubeck2006nms`, `loshchilov2016sgdr`, `loshchilov2017adamw`, `lu2019gridrcnn`, `pang2019librarcnn`, `ghiasi2019nasfpn`, `liu2021swin`, `hong2022sspnet` (GRSL 19:1–5, DOI correct), `chao2025bpdyolo` (Sci Rep 15:31888, DOI correct), `wang2020eca`, `chen2021rfla` (ECCV 2022; key name stale only).

### B.3 Table 3 literature numbers (spot-check)
- Group (a) vs Yu et al. 2020: Faster R-CNN 43.55 / RetinaNet 30.82 / FCOS 16.90 AP50^tiny — **confirmed** (thecvf.com openaccess WACV2020).
- Group (b) vs Hong et al. 2022: Cascade R-CNN 57.19 → +SSPNet 58.59 confirmed; Faster R-CNN+SSPNet ≈59.13 consistent with paper claims. Other cells (Swin-T 40.52, Grid R-CNN 47.14, Libra 44.68, NAS-FPN 37.75, tiny1 columns) not independently confirmed — verify against the SSPNet PDF table before submission.
- Group (c) vs Chao et al. 2025: BPD-YOLO 39.1 mAP50, 1.50M params — **confirmed** (researchgate/Sci Rep). YOLOv8n+P2 38.0/2.93M, BiFPN 38.4/2.34M, AFPN 39.2/6.05M not independently confirmed.
- **main.tex:389** — "Faster R-CNN $43.55\%$ vs.~$57.05\%$ with PAFPN-style pipelines": **57.05 appears nowhere in Table 5** and matches no SSPNet number I could verify (Cascade is 57.19). Likely a typo. Fix: use 57.19 or 59.13 and cite the exact row.

### B.4 Vietnamese report reference list (section 7)
1. **course_report_vi.md:284 — ref [1] is INVENTED.** "X. Yu, Z. Chen, and Y. Wu, 'Tiny Person Detection in the Wild,' WACV, 2020" — wrong authors, wrong title, no pages. Correct: `X. Yu, Y. Gong, N. Jiang, Q. Ye, Z. Han, "Scale Match for Tiny Person Detection," Proc. WACV, 2020, pp. 1257–1265.`
2. **course_report_vi.md:296 — ref [13] is INVENTED.** "Y. Chao et al., 'BPD-YOLO: A Lightweight Detector for Tiny Person Detection,' 2025" — wrong author initial (M.-X. Chao), invented title, no venue. Correct: `M.-X. Chao et al., "A lightweight small object detection model for UAV images based on deep semantic integration," Scientific Reports, vol. 15, art. 31888, 2025.`
3. **course_report_vi.md:285 — ref [2]** "arXiv, 2021" is actually closer to truth than the paper's bib (which fabricates ICPR 2022). Keep arXiv 2110.13389, 2021 in both.
4. **course_report_vi.md — missing reference for YOLO26/Ultralytics**, the base architecture of the whole report. Add a citation.
5. **course_report_vi.md:286 — ref [3] YOLOv7 (CVPR 2023)** is cited in the Vietnamese report but not used anywhere in the report body and not cited in the paper. Remove or cite it in text.
6. Refs [4]–[12], [14]–[18]: metadata consistent with the bib — OK.

## C. NUMBER AUDIT

| # | Claim | Document value | JSON value | Verdict |
|---|---|---|---|---|
| 1 | AP50 all tiled (Base→GauSS) | 40.63→41.08 | 40.628→41.085 | OK |
| 2 | AP50 tiny1/tiny2/tiny3/tiny | 22.30/36.92/48.14/37.57 → 22.26/37.67/47.05/37.40 | 22.297/36.924/48.144/37.568 → 22.264/37.674/47.049/37.404 | OK |
| 3 | AP50 small/reasonable | 51.63→52.51, 44.14→47.59 | 51.626→52.508, 44.143→47.593 | OK |
| 4 | MR50 deltas tiny2/small/reasonable | −0.55/−1.00/−0.61 | 0.544/1.006/0.612 | OK |
| 5 | AP25 all bins | 61.13→62.67; tiny1 39.55→43.54 (+3.99); small 71.66→73.80; reasonable 65.76→69.52; tiny3 70.22→69.00 | 61.131→62.671; 39.545→43.543; 71.657→73.795; 65.765→69.518; 70.222→69.001 | OK |
| 6 | Full-frame AP50/AP25 | 35.32→35.45; 60.96→61.08 | 35.323→35.453; 60.958→61.081 | OK |
| 7 | Params | 2,516,240 / +34,666 / 2.55M / +1.38% | 2516240 / 2550906−2516240=34,666 / 1.378% | OK |
| 8 | Latency/FPS tiled | 11.28/88.6 vs 15.19/65.8 | identical | OK |
| 9 | Latency/FPS full-frame | 11.85/84.4 vs 16.58/60.3 | identical | OK |
| 10 | GFLOPs | 7.62→8.13 (+0.51, +6.7%); 31.97→33.99 (+2.02, +6.3%) | identical | OK |
| 11 | Latency overhead % | +34.7% tiled, +39.9% full | 34.66%, 39.92% | OK |
| 12 | **VRAM** | "Peak VRAM is 122.1 MB tiled / 367.2 MB full-frame" | 122.1/367.16 are **GauSS** values; baseline is 91.51/244.65 MB (profiling_results.json) | **MISMATCH (ambiguity)** — paper does not say these are GauSS-only; state the model and ideally both models |
| 13 | **AP75 claim** | "AP75 ≈ 1.3–1.4%" (main.tex:522; report:257) | tiled all AP75 = 4.99/5.20; tiny1 = 1.99/0.85; full all = 3.74/3.23 | **WRONG** — no JSON supports 1.3–1.4%. Replace with actual numbers, e.g. "AP75 all is 4.99→5.20 tiled; on tiny1 it is 1.99→0.85". |
| 14 | **Ablation table** | 29.63 / 24.96 / 25.41 / 26.43 / 27.86 / 28.41 | per-class `sea_person` AP50: 24.964/25.409/26.429/27.857/28.409 match; reference 29.63 matches `bootstrap_headline.json` but that file's "GauSS" input was `dcaf_tiled_per_class.json` (deleted) — see A-3 | **Numbers OK, provenance UNVERIFIABLE** |
| 15 | Ablation deltas | −4.67 / −4.22 / −3.20 / −1.22 | 4.67/4.22/3.20/1.22 | OK |
| 16 | **Dataset split** | "1,610 frames ... partitions into 746 training and 786 evaluation" | 746+786=1532 ≠ 1610 (78 frames unaccounted; actual split is 662 train + 84 val + 786 test, docs/agents/DATA_EVALUATION.md:9-10) | **WRONG framing** — rewrite: "1,610 frames; we use 746 for training (662 train + 84 validation) and the official 786-frame test split; 78 frames are unused." |
| 17 | GT counts | 13,687 / 1,989 ignore / 2,832 uncertain; bins 2,333/2,778/3,767/8,878/2,485/2,324 | 13,687 positive ✓; 7,763+10,745=18,508 anns = 13,687+1,989+2,832 ✓; bin GTs match JSON | OK |
| 18 | End-to-end tiling | 23.4 tiles; ~355 ms/2.8 FPS; 9 tiles/137 ms/7.3 FPS; 8.3×; baseline 3.8–9.8 FPS | 23.4×15.19=355.4 ✓; 9×15.19=136.7 ✓; 137/16.58=8.3 ✓; 23.4×11.28=263.9→3.79 FPS ✓ | OK (arithmetic consistent; "median 3840×2160" not independently verifiable from JSONs) |
| 19 | Qualitative counts | 6 GT, 2 vs 10 detections ≥0.20, 12 vs 21 frame totals; FP 0.60; 95→71 detections | generated by `scripts/export_qualitative_figures_honest.py` (conf 0.20/0.25 defaults match) | Not independently verifiable from JSONs — OK with caveat |
| 20 | Table 3 group (a) params | ~41M/~34M/~32M | plausible ResNet-50 two-stage sizes | OK (approximate, labelled "~") |
| 21 | **"57.05%"** (main.tex:389) | 57.05 | not in Table 5, not in SSPNet | **UNSUPPORTED** — fix or delete |

## D. PRESENTATION & STRUCTURE

1. **Figure 3 (`fig:merge_bins`, main.tex:347) is never referenced in the text** — add a `\ref{fig:merge_bins}` in Sec. 5.1 or drop the figure.
2. Table numbering (1 notations, 2 params, 3 main, 4 ablation, 5 reference, 6 hardware) and all figure/table `\ref`s resolve — OK. Zero overfull boxes — OK.
3. **Ablation protocol inconsistency (coherence, not leak):** Table 3 uses merged-class AP50 (41.08) while Table 4 uses two-class `sea_person` (29.63) for the "same" full model. A reviewer will ask why the same model has two scores. Add one sentence in the Table 4 caption: the reference row is a per-class evaluation of the 150-epoch checkpoint, not comparable to Table 3.
4. **Profiling checkpoint ambiguity:** `profiling_results.json` labels the GauSS entry `gauss_v2_1`; the main results are `gauss_v3`. If v2_1 and v3 are architecturally identical, say so; otherwise the hardware table describes a different model than the accuracy tables.
5. Paper reads as one coherent single-module story — OK.
6. Vietnamese report structure: cover block, TOC, chapters 1–6, references (7), appendices A/B, acknowledgements — OK. TOC lists appendices with a stray `-` bullet (line 27) — minor formatting.
7. Vietnamese report has no figure/table numbering at all (tables are unnumbered markdown, "Bảng 4.4" is referenced in text at line 255 but no table carries that number). Number the tables (Bảng 4.1–4.3) and reference them consistently.
8. `main.tex:39` TODO comment "add the corresponding author's e-mail address here" — must be resolved before submission.

## E. STYLE / AI-SLOP (top items only; vocabulary scan found zero banned words in either document)

1. **20 bold-lead-in `\item \textbf{...}` bullets** (main.tex:438–441, 480–486, 519–526) — direct violation of guide A2.3. Rewrite as prose; keep bullets only for the hyperparameter list.
2. **main.tex:43 (abstract)** — 9 sentences, one paragraph, ends with a list-shaped sentence "We analyze X, report Y, examine Z, and discuss W" (triad+). Rewrite: "We report per-bin instance counts, end-to-end tiling latency, and an ablation of the module's components."
3. **main.tex:60–66 (contributions)** — four bold-lead bullets, the classic AI template. Consider two sentences of prose or keep at most three bullets without bold lead-ins.
4. **main.tex:357** — "A module that finds more of the 2–8 px targets but localizes them slightly worse shows exactly this signature" — reads like AI pattern-matching; rewrite plainly: "The AP25 gains with flat AP50 mean the module finds more 2–8 px targets without tightening their boxes."
5. **main.tex:520** — "This coherence between design intent and measurement is the main reason we trust the +0.45 pp aggregate" — over-narrated; shorten to "The gains are distributed across tiny2, small and reasonable rather than concentrated in one bin."
6. **main.tex:484** — 7-line bullet paragraph with three em-dash-free but very long sentences; split into 2–3 shorter items or prose.
7. **main.tex:43 abstract** — "strictly isolated splits" appears 3× in the paper (43, 58, 324); keep once.
8. **report:189** — "Cách đọc hợp lý là ... đúng như kỳ vọng với một module xử lý năng lượng tần số chứ không hồi quy biên. Với bài toán cứu hộ, tìm thấy còn quan trọng hơn khung khít, nên đây là tín hiệu tích cực." — last clause is spin ("tín hiệu tích cực"); per B1 guide, state the fact and stop.
9. **report:197** — "Nhóm chúng em không có lời giải thích chắc chắn cho dấu hiệu này" — good honesty; but the following hedge sentence is long; keep only the first sentence and the multi-seed point.
10. **report:269** — "Điều nhóm chúng em học được nhiều nhất không nằm ở con số cuối." — fine, but the paragraph then re-explains; trim to two sentences.
11. Both documents keep terminology consistent (GauSS, TinyPerson, AP, MR) — OK. Em-dash count in paper: 1 — OK.

## F. LENGTH

- Compiled PDF: **13 pages** (main.log). References begin on **page 11**; content = **pages 1–10**. WACV limit is **8 pages of content excluding references** → **~2 pages over**.
- Cut candidates (ranked least loss first):
  1. Table 2 (notations, main.tex:93–117) — half of it repeats the equations; cut rows already defined inline (~0.4 page).
  2. Table 5 group (b) — keep 4 representative rows (Faster R-CNN-FPN, Cascade R-CNN, SSPNet best, one scale-matched), drop the full zoo (~0.4 page).
  3. Sec. 5.4 qualitative bullets → compress the four items into one paragraph + two figures instead of four figure panels (~0.4 page).
  4. Merge Sec. 4.3 (loss/head) into Sec. 3.4 (~0.2 page).
  5. Discussion items 1 and 5 repeat Sec. 5.1/5.5 verbatim in places; merge (~0.3 page).
  6. Appendix figure (p.13) is outside the 8-page content budget — fine as-is.
- Vietnamese report has no page limit constraint stated — OK.

## Priority order of blocking issues
1. C-13/AP75 claim wrong in both documents.
2. A-3/C-14 ablation reference provenance (dcaf/gausssplit) unresolved.
3. B.4-1, B.4-2 invented Vietnamese references.
4. B.1-1, B.1-2, B.1-3, B.1-5 fabricated bib metadata (AI-TOD authors, NWD venue, VisDrone title/pages, Pintea venue).
5. F: 10 content pages vs 8-page limit.
6. C-16 dataset split arithmetic (1610 vs 746+786).
7. main.tex:389 unsupported "57.05%".
