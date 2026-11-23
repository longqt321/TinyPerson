# 08 — Figure / Layout Audit (WACV submission)

Audit of `paper/main.tex` (12 pp total: content ends mid-p.9, references p.9–10,
appendix p.11–12) and `paper/figures/`. Read-only audit: no `.tex` and no figure
was modified.

Reference geometry (from `paper/wacv.sty`): `\textwidth` = 6.875 in = **495 pt**,
`\columnwidth` = **236.25 pt**, `0.48\textwidth` = **237.6 pt**.
Effective font pt = native font pt × (slot width / native PDF width).
Threshold used: **< 6 pt effective = unreadable** (WACV body is 10 pt).

---

## A. Inventory

| # | tex line | file | float | width spec | slot (pt) | native (pt) | scale | caption words | label | first `\ref` (line) | verdict |
|---|----------|------|-------|-----------|-----------|-------------|-------|---------------|-------|---------------------|---------|
| 1 | 73 | `11_overall_gauss_pipeline.pdf` | `figure*` | `\linewidth` | 495 | 628.2 × 263.9 | 0.788 | 43 | `fig:pipeline_overview` | 82 (same page, p.3) | **FAIL text**: box labels 5.0 pt → **3.9 pt**, section titles 4.3 → **3.4 pt** |
| 2 | 106 | `10_gauss_router_architecture.pdf` | `figure*` | `\linewidth` | 495 | 860.0 × 413.6 | 0.576 | 33 | `fig:gauss_router` | 84 (p.3; float lands p.4) | **FAIL text**: box labels 8.0 → **4.6 pt**; also content-wrong (see B) |
| 3 | 330 | `19_merge_protocol_scale_bins.pdf` | `figure*` | `\linewidth` | 495 | 712.6 × 223.1 | 0.695 | 69 | `fig:merge_bins` | 340 (p.7, same page) | **FAIL text**: value labels 6.5 → **4.5 pt**, ticks 7.5 → 5.2 pt; caption 69 w > 60 |
| 4 | 418/422 | `12_qualitative_tiny_detections.pdf` | `figure*` (2×`minipage` 0.48) | `\linewidth` in minipage | 237.6 | 648.0 × 244.8 | 0.367 | 87 | `fig:qualitative_detections` | 416 (p.7; float lands p.9) | **FAIL**: panel title 9 → **3.3 pt**, conf text 7 → **2.6 pt**; 2-page drift; caption 87 w |
| 5 | 426/428 | `13_sea_clutter_suppression.pdf` | `figure*` (minipage 0.48) | `\linewidth` | 237.6 | 547.2 × 288.0 | 0.434 | 62 | `fig:sea_clutter` | 416 (range only) | **FAIL**: title 8.5 → **3.7 pt**, conf 8 → 3.5 pt |
| 6 | 434/438 | `14_p2_feature_activation_energy.pdf` | `figure*` (minipage 0.48) | `\linewidth` | 237.6 | 714.9 × 233.6 | 0.332 | 26 | `fig:p2_activation` | 416 (range only) | **FAIL worst**: title 9 → **3.0 pt**; stale export (script figsize now 6.6 in, PDF still 9.93 in) |
| 7 | 442/444 | `15_gaussian_band_decomposition.pdf` | `figure*` (minipage 0.48) | `\linewidth` | 237.6 | 715.0 × 195.3 | 0.332 | 27 | `fig:band_decomp` | 416 (range only) | **FAIL worst**: title 9 → **3.0 pt**; stale export (script figsize now 7.0 in, PDF 9.93 in) |
| 8 | 478 | `04_hardware_efficiency.pdf` | `figure` | `\linewidth` | 236.25 | 444.5 × 188.4 | 0.532 | 43 | `fig:hardware` | 453 (p.8; float lands p.9) | **FAIL**: ticks 8.5 → **4.5 pt**, value labels 7.2 → 3.8 pt; redundant with Table 6 |
| 9 | 514/518 | `07_training_loss_convergence.pdf` | `figure*` (minipage 0.48) | `\linewidth` | 237.6 | 514.0 × 167.7 | 0.462 | 18 | `fig:train_loss` | 487 (p.8; float lands p.12) | **FAIL**: ticks 8.5 → **3.9 pt**, legend 7.2 → 3.3 pt |
| 10 | 522/524 | `08_training_total_loss_lr.pdf` | `figure*` (minipage 0.48) | `\linewidth` | 237.6 | 399.1 × 196.6 | 0.595 | 15 | `fig:train_lr` | 487 | **FAIL**: ticks 8.5 → **5.1 pt**, legend 7.2 → 4.3 pt |

**Every one of the 10 included figures renders text below 6 pt.** Root cause: every
figure is authored at ≈1.5–3× the width of its slot (all matplotlib `figsize` are
7–12 in wide, while a two-column slot is 3.28 in and a half-width panel is 3.30 in).
`docs/report/00_state.md` already records this for Figs. 14/15 ("~2.6 pt, đã sửa
`figsize` nhưng **chưa sinh lại hình**") — the stale PDFs are still the ones in
`paper/figures/`.

Unused (not included, keep out of the build): `01`, `02`, `03`, `05`, `06`, `09`,
`10_gauss_module_architecture.pdf`, `10b_gauss_residual_refinement.*`, plus
`_archive/`.

Caption length: 4 captions exceed ~60 words — `fig:qualitative_detections` (87),
`fig:merge_bins` (69), `fig:sea_clutter` (62); tables `tab:ablation` (84) and
`tab:intl_comparison` (81) are also long but tables are out of scope here.

Referencing: `fig:sea_clutter`, `fig:p2_activation`, `fig:band_decomp` are only
reached through the range `Figures~\ref{fig:qualitative_detections}--\ref{fig:band_decomp}`
(L416); none has its own sentence. `fig:merge_bins` is referenced twice (L340, L493) — fine.

`docs/report/course_report_vi.md` contains **no** figure references (no `![...]`,
no `figures/` path) — nothing to audit there.

---

## B. Architecture-diagram mismatches vs. code

Evidence base: `src/tinydet/modules/gauss.py` (current tree; verified against the
run commit `a20f7b6`), `configs/model/yolo26n_tiny_A2_p2_gauss_tinyperson.yaml`,
and `tab:param_breakdown` (`paper/main.tex:241–263`).

**B1. Upsampling is DySample, not nearest-neighbour `Interp`.**
Fig. 10 draws `Interp(·)` for both `S3↑` and `S4↑`; paper L137 states
"`Interp(·)` denotes nearest-neighbor upsampling". Code: `self.dy_up3 = DySampleUp(d, scale=2, groups=4)`
and `dy_up4` (`gauss.py:475–476`), used at `gauss.py:613` and `:628`. DySample
carries **1,056 learnable params each (2,112 total)** (offset conv `32→32`, k=1).
`tab:param_breakdown` has no row for them.

**B2. Dilation rates are {1, 3, 5}, not {1, 2, 3}.**
`gauss.py:408` `dilations: Sequence[int] = (1, 3, 5)`; the YAML passes only `[32]`
(`yolo26n_..._gauss_tinyperson.yaml:41`), so the default applies. Paper L172 and
`tab:param_breakdown` row "Dilated Context … $r \in \{1,2,3\}$" (L253) are wrong.
Fig. 10 label `r_i ∈ {1, 2, 3}` is wrong.

**B3. The router shown in Fig. 10 is the legacy 4-channel learned router, which is
not the trained path.** Fig. 10 shows `Z = Conv(DWConv(U))`, `a_bands ∈ R³`,
`m_gate ∈ R¹`; paper L183–196 (Eqs. 16–18) describes the same. Code default
(`gauss.py:695–704`): `corr = router_corr(u)` (**Conv1×1 d→3, 99 params**),
`a = softmax(log_e + corr)`, `m_gate = a_h + a_m` (derived, **no** sigmoid logit).
The `route_dw/route_bn/route_out` stack (484 params) is only executed under the
`learned_router` flag (`gauss.py:673–681`) and is listed as *unused* in the default
mode (`gauss.py:153–160`). So `tab:param_breakdown` row "Spatial Router … 484"
counts dead weights and omits the 99 live ones.

**B4. Gate parameterisation is `0.5·tanh`, not `Sigmoid`.**
Paper Eq. 6 (L146) `σ_k = σ_min + (σ_max−σ_min)·Sigmoid(σ̂_k)`; code
`gauss.py:204–207` `s = center + half*tanh(raw)`. Paper L203
`γ_h = 0.5·Sigmoid(γ_h,raw)` bounded to `(0, 0.5)`; code `gauss.py:561–575`
`0.5*tanh(raw)` bounded to **(−0.5, 0.5)** — sign can flip.

**B5. High-band shortcut is wired to the wrong node; two extra output anchors are
missing from the diagram.** Paper Eq. 20 (L200) `F_band = Σ a_i R_i + γ_h R_high`.
Code `gauss.py:716` `f_band = Σ a_i r_i` and `:720`
`x_d = f_band + c_ctx + gamma_in * r_h` — `γ_in·R_high` enters the **detail-branch
input together with `C_ctx`**, not `F_band`. Fig. 10 draws `Fband = ∑aiRi + γhRhigh`.

**B6. Output is linear and has two extra band anchors.**
Paper Eq. 25 (L233) `P2 = SiLU(X0 + α·F̃)`; Fig. 10 right-hand label
`= SiLU(X0 + αF̃)`. Code `gauss.py:756`:
`y_p2 = x0 + alpha*z + gamma_out*b_h_eff + gamma_mid*b_m_eff` — **no SiLU**
(comment `gauss.py:752`: "Residual addition WITHOUT activation") and two direct
stride-4 band anchors (`γ_out·B_high`, `γ_mid·B_mid`) that appear nowhere in the
paper or the figure.

**B7. The saliency gate is drawn and described as active, but it is off by default.**
Fig. 10 shows `R_low^att = R_low ⊙ (1 − 0.5·M_sal)`; paper Eqs. 21–22 (L205–210)
present it as part of the forward path, and §5.2/Table 3 report "w/o saliency
−4.22 pp". Code `gauss.py:725–729`: the branch executes only under the
`use_saliency` flag; default is `r_l_attenuated = r_l_eff`. `gauss.py:104`
comment: "saliency is off by default", and `:723–724` records that it *degraded*
`earth_person` by −2.38 pp. The five ablation runs are
`no_high_band, no_mid_band, no_low_band, no_router, no_saliency`
(`modal/ablation_runner.py:50–56`) — with saliency off in the reference, the
`no_saliency` run is a no-op and its −4.22 pp cannot come from that flag alone.
This is the single most serious content inconsistency in the paper.

**B8. Parameter total is stale: 34,666 vs. 36,879.**
Instantiated on the current tree: `GauSS(c1=(64,64,128), c2=32)` →
`sum(p.numel()) = 36,879` (verified by running the module). Difference from
34,666 = **2,112 (DySample) + 99 (router_corr) + 2 (γ_out, γ_mid)**. The 34,666
figure comes from the v2.1 profiling artifact
(`reports/eval/profiling_results.json` → `"gauss_v2_1": {"params": 2550906}`,
2,550,906 − 2,516,240 = 34,666). Therefore `tab:param_breakdown` (L241–263),
abstract ("+34,666, +1.38%"), L239, L503 and `tab:hardware_bench` (2.55M) all need
re-deriving: **+36,879 = +1.465 %**, total 2,553,119. (Caller's figure 36,878 is
off by one; the module reports 36,879.)
Note also `profiling_results.json` is labelled `gauss_v2_1`, so the latency/VRAM
numbers in Fig. 8 / Table 6 were measured on a **different module revision** than
the one described in §3.

**B9. Fig. 11 understates the module's fan-out.**
Fig. 11 shows GauSS feeding only "Head P2". In the trained topology GauSS output
(layer 17) is also downsampled (`Conv[128,3,2]`, layer 18, `32→32` ch) and
concatenated into the P3 path (layer 19 `18+16 → 96` ch → C3k2 layer 20), which
then feeds layers 21–26 and therefore Heads P3/P4/P5
(`configs/model/…_gauss_tinyperson.yaml:41–48`). The diagram implies a P2-only
insertion; the code inserts GauSS into the whole top-down path. Detect consumes
`[17, 20, 23, 26]` = `[P2_GauSS, P3, P4, P5]`.
Otherwise Fig. 11's channel labels are correct (C2 64/s4, C3 128/s8, C4 128/s16,
C5 256/s32, P2 32, P3 64, P4 128, P5 256 — width factor 0.25 applied).

**B10. Minor.** Fig. 10 labels `σ1 ≈ 0.65, σ2 ≈ 1.45` — correct as *initialisation*
values, but after training they are learned per channel; the figure should say
"init". Paper L149 correctly gives the physical mapping (σ1,img ≈ 2.6 px,
σ2,img ≈ 5.8 px) but Fig. 10 does not.

---

## C. Data-figure issues

Data correctness (checked against JSON): **all plotted numbers match**.
`19_merge_protocol_scale_bins` reads
`reports/eval/{baseline,gauss_v3}_tinyperson_wacv_150e_tiled.json` and reproduces
Table 2 exactly (all 40.63/41.08, tiny1 22.30/22.26, tiny2 36.92/37.67, small
51.63/52.51, reasonable 44.14/47.59; AP25 tiny1 39.55/43.54). `04_hardware_efficiency`
matches `reports/eval/profiling_results.json` (88.6/65.8, 84.4/60.3 FPS;
91.51/122.1, 244.65/367.16 MB). No fabricated values found.

**C1. Font family clash.** `scripts/plot_merge_protocol_figures.py` sets **no**
`rcParams` at all → matplotlib default **DejaVu Sans**. The paper body is Times.
Fig. 3 is therefore the only sans-serif figure in a serif paper. Same class of
issue for the two architecture diagrams (Fig. 1 uses DejaVu Sans throughout).
`plot_academic_figures.py`, `plot_training_wandb_figures.py`,
`modal/extract_figures.py` do set `font.family = serif` — good.

**C2. Colour-blind safety.** `plot_academic_figures.py` uses
`C_BASE = #2C5D88` (blue) vs `C_GAUSS = #C0392B` (red) and a third
`C_GREEN = #27AE60`. Red/green is the classic deuteranopia failure pair; Fig. 5
(`13_sea_clutter_suppression`) additionally prints **red** confidence text on the
baseline panel and **green** on the GauSS panel — the two panels are distinguished
only by that hue. Fig. 3 (`#8A97A8` grey-blue vs `#E8834A` orange) is safe.
Recommendation: single palette (blue `#2C5D88` vs orange `#E8834A`) everywhere,
plus hatch or marker shape for the second series.

**C3. Axis labels / units.** Fig. 3: `$AP_{50}$ (%)` ✓, `$AP_{25}$ (%)` ✓.
Fig. 8: `Throughput (FPS)` ✓, `Peak VRAM (MB)` ✓ — but the left panel plots FPS
while the caption quotes **latency in ms**; the reader must mentally invert. Add a
secondary ms axis or quote FPS in the caption. Figs. 9/10: `Loss Value` (unitless,
acceptable), `Epoch` ✓, LR panel needs `Learning Rate` (log scale?) verified —
not readable at 3.9 pt regardless.

**C4. Missing value labels.** Fig. 8 prints FPS/VRAM values ✓. Fig. 3 prints AP
values ✓ (but at 4.5 pt). Figs. 9/10 have none (line plots — acceptable). Fig. 4/5
print confidences at source-image scale, which becomes 2.6–3.5 pt on the page —
they are effectively invisible, i.e. the caption's numeric claims cannot be
verified by the reader from the figure.

**C5. Stale exports.** `14` and `15` were generated before the `figsize` fix in
`modal/extract_figures.py` (now `(6.6, 2.6)` and `(7.0, 2.2)`), but the PDFs on
disk are still 714.9/715.0 pt wide (old `(10, 3.8)`/wide layout). Regenerating them
alone lifts effective text from 3.0 pt to ≈5.6–6.0 pt — still marginal, so also
raise base font sizes.

**C6. Fig. 4/5 caption vs. figure content.** Fig. 4's caption states "6 tiny
ground-truth swimmers … baseline keeps 2 … GauSS keeps 10" and Fig. 5's states
"baseline fires one 0.60 false alarm … GauSS reports none". The scripts
(`scripts/export_qualitative_figures_honest.py:163–245`) compute the counts from
the checkpoint predictions, so the numbers are generated, not typed — good. But
both captions are 87 and 62 words and duplicate numbers already in §5.4 (L416);
trim to ≤40 words and let the text carry the numbers.

---

## D. Proposed layout plan

Constraint: content must end by p.8 (currently mid-p.9). Figure area is the budget.

| Fig | New float | Width | Placement | Change |
|-----|-----------|-------|-----------|--------|
| 1 (pipeline) | `figure*[t]` | `0.92\linewidth` | p.2 top (first ref L82 is p.3 — one page early is acceptable for a `figure*`) | regenerate at `figsize=(7.0, 2.9)`, base font 8 pt → effective ≈7.4 pt |
| 2 (router) | **merge into Fig. 1** as panel (b) | `figure*`, two stacked panels at `\linewidth` | p.3 | one caption, one float, saves a full caption block + `\abovecaptionskip`; regenerate at `figsize=(7.0, 5.2)` total |
| 3 (merge bins) | `figure*[t]` | `0.95\linewidth` | p.6, immediately above Table 2 | regenerate `figsize=(7.0, 2.2)`; drop the redundant `all`/`tiny` bars (keep tiny1/tiny2/tiny3/small/reasonable) → narrower |
| 4+5 (qualitative) | **single `figure*`**, 2 panels | each `0.48\linewidth` | p.7 | already one float — keep, but regenerate each at `figsize=(3.3, 2.2)` with 7 pt fonts → effective ≈7 pt |
| 6+7 (feature maps) | **single `figure*`**, 4 panels (input, baseline, GauSS, bands) | each `0.24\linewidth` | p.7 or p.8 | merge the two `figure*` blocks into one row; saves one caption + ~0.35 in |
| 8 (hardware) | **delete** (Table 6 already carries every number) or demote to `figure*` bottom-of-column | — | — | deleting reclaims ≈0.9 in² + caption; if kept, `0.48\linewidth` and regenerate at `figsize=(3.3,2.4)` |
| 9+10 (training) | keep `figure*` in Appendix A | each `0.48\linewidth` | p.11 | regenerate at `figsize=(3.3, 2.2)`; consider merging into one 3-panel row |

Concrete rules to apply:
- **No figure may be authored wider than its slot.** Target `figsize` width (in)
  = slot width (in) × 1.0–1.15, never more. For a `0.48\textwidth` panel that is
  **3.3 in**, not 9 in.
- Base font ≥ 8 pt when the author width equals the slot width; ≥ 9 pt if the
  figure is authored 1.15× the slot.
- Use `\begin{figure*}[t]` only; `[htbp]` on a full-width float is a no-op that
  invites drift (currently `[t]` everywhere — correct, keep it).
- Put each `figure*` **immediately after** the paragraph that first references it,
  so the float lands on the same or the next page.

---

## E. Space budget

Measured figure area (width × rendered height, at current widths):

| Fig | area (in²) | % of 61 in² text block |
|-----|-----------|------------------------|
| 1 pipeline | 6.88 × 2.89 = 19.9 | 33 % |
| 2 router | 6.88 × 3.31 = 22.7 | 37 % |
| 3 merge bins | 6.88 × 2.16 = 14.8 | 24 % |
| 4+5 qualitative block | 6.88 × 1.74 = 12.0 | 20 % |
| 6+7 feature block | 6.88 × 1.08 = 7.4 | 12 % |
| 8 hardware | 3.28 × 1.39 = 4.6 | 7.5 % |
| **total (body)** | **81.4** | **1.33 pages** |
| 9+10 appendix | 6.88 × 1.62 = 11.2 | (appendix) |

Reclaim plan:

| Action | Saving |
|--------|--------|
| Merge Fig. 1+2 into one two-panel `figure*` (one caption, one float, tighter layout) | ≈ 0.30 page |
| Fig. 2 regenerated at 0.92 width instead of 1.0 | ≈ 0.06 page |
| Merge Fig. 6+7 into one 4-panel row (one caption instead of two) | ≈ 0.10 page |
| Delete Fig. 8 (Table 6 already reports FPS and VRAM) | ≈ 0.08 page |
| Trim Fig. 3 caption 69→35 words, Fig. 4 87→40, Fig. 5 62→35 (≈ 3 lines each) | ≈ 0.10 page |
| **Total from figures** | **≈ 0.64 page** |

Remaining ≈ 0.4 page must come from text (the ablation-discussion paragraph
L489–491 and the repeated numbers in §5.4 L416 duplicate captions almost verbatim;
also `tab:ablation` caption is 84 words). Combined this closes the p.9 overflow
with margin.

---

## F. Priority list

**P0 — must fix before submission**
1. **B7** saliency gate: either enable `use_saliency` for the reported model or
   remove the gate from the paper, from Fig. 10, and from Table 3. As written the
   paper describes a forward pass the released code does not run.
2. **B8** parameter count: 34,666 → 36,879 (+1.465 %), total 2.55M → 2.553M;
   re-derive `tab:param_breakdown` with a DySample row (2,112) and a router row of
   99 (drop the 484 dead row), and fix abstract/L239/L503/Table 6.
3. **A / C5** regenerate **all** figures at slot width; Figs. 6/7 are at 3.0 pt,
   Figs. 4/5 at 2.6–3.7 pt. This is the difference between "figure" and "grey mush".
4. **B1/B3/B4/B5/B6** Fig. 10 + Eqs. 6/16–20/25: DySample not `Interp`, dilations
   {1,3,5}, `0.5·tanh` not `Sigmoid`, `γ_in·R_high` into `x_d`, linear output with
   `γ_out·B_high + γ_mid·B_mid`, no `SiLU`.
5. **D** page budget: content currently ends mid-p.9 against an 8-page limit.

**P1 — should fix**
6. **B9** Fig. 11: show the GauSS→downsample→P3 concat path (fan-out to P3/P4/P5).
7. **C1** serif fonts in Fig. 3 (and Fig. 1) to match the paper.
8. **C2** colour-blind-safe palette; remove red/green panel distinction in Fig. 5.
9. Float drift: Figs. 4–7 land on p.9 while referenced on p.7; move the
   `figure*` blocks directly after the §5.4 paragraph.
10. Caption trimming (69/87/62 words) and individual `\ref`s for
    `fig:sea_clutter`, `fig:p2_activation`, `fig:band_decomp`.

**P2 — nice to have**
11. Delete Fig. 8 (redundant with Table 6) and reclaim the space.
12. Add ms axis (or FPS in caption) to Fig. 8 if it survives.
13. Note in Fig. 10 that σ1/σ2 are initialisation values.
14. Move `10_gauss_module_architecture.pdf`, `10b_*`, `01/02/03/05/06/09` into
    `paper/figures/_archive/` so a future build cannot pick them up by accident.
15. `profiling_results.json` is keyed `gauss_v2_1` — re-profile the shipped
    revision or state the revision explicitly in Table 6's caption.
