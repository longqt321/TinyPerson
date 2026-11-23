# Writing Guide: Concise 8-Page WACV/CVPR-Style Paper (Evidence-Based)

Scope: hard rules from official author/reviewer guidelines + writing advice from established researchers (Freeman, Peyton Jones, Dreyer, Zobel, Whitesides). Every number below is either an official rule or a community-standard estimate; estimates are labeled.

## 1. Page/word budget per section

Official constraints:
- WACV: main content ≤ 8 pages, references unlimited but pages > 8 must contain ONLY references (thecvf.com WACV author guidelines).
- CVPR: same 8-page rule, references excluded; reviewers are NOT required to read supplementary material (CVPR author/reviewer guidelines), so everything essential must fit in 8 pages.
- Layout: two-column IEEE/CVF, 10pt Times. A text-heavy page ≈ 800–1,000 words; a typical 8-page paper ≈ 5,000–7,000 words excluding references (community estimate).
- Abstract: no hard CVPR word limit, but it must fit the top of column 1; practical target 150–250 words (community norm).

Recommended budget (fractions of 8 content pages; word counts assume ~900 w/page minus figure space):

| Section | Page fraction | ≈ Words | Notes |
|---|---|---|---|
| Abstract | 0.2 col | 150–250 | one paragraph |
| 1. Introduction | 1.0–1.25 pp | 900–1,100 | incl. teaser figure + contribution bullets |
| 2. Related Work | 0.5–1.0 pp | 450–900 | may also sit at end (Peyton Jones) |
| 3. Method | 2.0–2.5 pp | 1,600–2,000 | incl. main architecture figure |
| 4. Experiments | 2.0–2.5 pp | 1,200–1,800 text + tables | setup ≤ 0.5 page |
| 5. Discussion/Limitations | 0.25–0.5 pp | 200–400 | reviewers are told missing limitations shouldn't drive rejection, but including them builds trust (Freeman) |
| 6. Conclusion | 0.15–0.25 pp | 120–200 | no new claims |

## 2. Internal structure rules per section

General paragraph rules (Zobel, "Writing for Computer Science"; Whitesides, "Whitesides' Group: Writing a Paper"):
- One idea per paragraph; one topic per section. Start each paragraph with a topic sentence; middle sentences support only that idea; stop when the idea is done.
- Text should be "taut": every sentence necessary. If a paragraph drifts into a second point, split or cut.

Abstract: what problem, why it matters, what you do (one sentence), headline result with a number. No citations, no section structure, no implementation details — those belong in experimental setup.

Introduction (Freeman "How to write a good CVPR submission"; Dreyer "How to Write Papers So People Can Read Them"; Peyton Jones):
- Order: (a) problem + why the reader should care, (b) why existing solutions fall short, (c) your approach in one paragraph, (d) explicit bullet list of contributions.
- Contributions list is what reviewers check off — deliver on every promise; forward-reference each claim to the section/table that proves it (Dreyer).
- Do NOT narrate failed approaches or heavy related work in the intro (Dreyer); no "The rest of this paper is structured as follows" paragraph (Dreyer).
- Related work: group thematically, state the specific gap, pivot to your method; keep it brief if placed early; Peyton Jones recommends placing full related work near the end.

Method / equations (Freeman):
- Equations must be part of the sentence (proper punctuation), every variable defined at first use, and each equation must earn its place: use an equation only when it carries information a citation or one-line intuition cannot (e.g., your own formulation, a non-obvious loss). Do NOT restate well-known formulas (softmax, IoU, cross-entropy) — cite instead.
- No tutorial-style background of textbook material; jump to what is novel.

Experiments: setup (datasets, metrics, implementation details, augmentation) compressed into one subsection/table; every claim in intro must map to a table or figure; report ablations that directly support each contribution claim.

Captions: every caption is a mini-abstract — what is shown, how to read it (colors, markers, boxes), and the takeaway; include key numbers in qualitative comparison captions; never write "Figure 3: Overview of our method." (CVPR reviewer guidance / community norms).

Number repetition rule: a headline number may appear at most 2–3 times (abstract, intro/results, table). The table is the canonical source; prose should reference ("Table 2") rather than re-quote numbers. Never repeat the same number in abstract + intro + results + conclusion.

## 3. Common reviewer complaints (and fixes)

From CVPR reviewer guidelines and recurring reviewer feedback:
- Overselling / mismatch between intro promises and evidence — fix: every intro claim has a forward reference.
- Wall-of-text intro without contributions in the first ~1.5 pages — fix: bullet contributions by end of page 2.
- Tutorial-style background and restated well-known formulas — fix: cite, don't derive; assume the CVPR reader.
- Unreadable figures (micro-fonts, spaghetti diagrams), lazy captions, figures far from their discussion — fix: self-contained captions, single-column readable fonts, figures placed on the same page.
- Buried methodological details in appendix/supplementary — reviewers are not required to read it; keep the core method in the main 8 pages.
- Overlong abstracts (>250 words) that restate the paper — fix: 150–250 words, one paragraph.
- Repeated numbers and redundant restatement of results across sections — fix: canonical table + at most 2–3 mentions.

## 4. Checklist: trim 8.3 → ≤ 8 content pages without losing substance

1. Delete every paragraph that only restates another paragraph (search for repeated claims; keep the first occurrence).
2. Remove all restated well-known equations; replace with citations (typically saves 0.2–0.5 page).
3. Compress background/tutorial text to ≤ 3 sentences per concept + citation.
4. Cut the "paper is organized as follows" paragraph (Dreyer).
5. Cut numbers from prose that already appear in a referenced table; keep only headline numbers (abstract, intro, conclusion).
6. Move implementation minutiae (hyperparameter sweeps, per-layer settings) to supplementary; keep one setup paragraph + a config table.
7. Shrink Related Work to thematic groups of 1–2 sentences each; cut papers you don't actually compare against.
8. Shorten captions to 2–3 sentences that still stand alone.
9. Merge single-plot figures into one multi-panel figure; ensure every figure/table is referenced and earns its space — cut any that isn't discussed.
10. Convert verbose limitations/discussion paragraphs to a 4–6 item bullet list.
11. Tighten every paragraph to one idea; delete topic sentences that merely announce ("In this section we describe...").
12. Re-check: after trimming, verify contributions list ↔ tables/figures still match 1:1, and pages > 8 contain only references (WACV rule).

## Sources
- WACV author guidelines (thecvf.com/WACV): 8-page main content limit, references-only beyond page 8.
- CVPR author guidelines (thecvf.com): 8 pages + unlimited references; supplementary not guaranteed to be read; LLM disclosure excluded from limit.
- CVPR reviewer guidelines (thecvf.com): soundness/novelty/clarity criteria; weaknesses must be evidence-based; missing limitations not a rejection driver.
- Bill Freeman, "How to write a good CVPR submission" (MIT): problem→gap→solution intro, honest limitations, equations in-sentence, no tutorials.
- Simon Peyton Jones, "How to write a great research paper": key idea, story structure, contribution list, related work at end, reader-first writing.
- Derek Dreyer, "How to Write Papers So People Can Read Them" (PLMW): intro = expanded abstract, no failed-approach narration, forward-referenced contributions, no outline paragraph.
- Justin Zobel, "Writing for Computer Science": one idea per paragraph, taut prose.
- George Whitesides, "Whitesides' Group: Writing a Paper" (Adv. Mater.): outline-first, one finding per paragraph.
- Community norms (arXiv CVPR-style papers, writing blogs): ~800–1,000 words/page, 150–250-word abstracts.
