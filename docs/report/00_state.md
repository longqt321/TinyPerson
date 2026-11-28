# Research Report State Ledger (`GauCon`)

- **Updated**: 2026-11-28T23:55+07:00
- **Active Method Name**: **GauCon** (*Local Gaussian Consensus Head* + *Decoupled Wasserstein Task-Aligned Learning*, `DW-TAL`), implemented in `src/tinydet/modules/lgt_detect.py` and `src/tinydet/models/ultralytics.py` (commit `ac79514`).
- **Deliverables**:
  1. English WACV camera-ready paper (`paper/main.tex` -> `paper/main.pdf`, clean hypothesis-driven framing without exploratory prototypes, exact `136,000` grid cells, pre-fuse `+0.086` GFLOPs disclosure, `Base + Soft-NMS` post-hoc control `43.37%`, `reasonable` sub-bin breakdown, and `1,000`-resample paired image-level bootstrap `95%` CIs, zero code-artifact leaks).
  2. Vietnamese Course Report (`docs/report/course_report_vi.md` -> `docs/report/course_report_vi.pdf` & `.docx`, DUT QĐ 209/QĐ-ĐHBK format, zero code/file-path leaks, zero inline backticks, standard mathematical notation $\mathcal{H}_{1:1}$ / $\mathcal{H}_{1:M}$ and pure-math `Thuật toán 2.1`, Appendix A architectural table `Bảng 3.7` & full Appendix B `Bảng 3.8`–`3.9`, no Appendix C, zero `v1` history, standard terminology avoiding `"hạt nhân ngẫu nhiên"`).
- **Phase**: SOURCE REVISED; PDF/DOCX rebuild and visual QA pending.

## Verified Experimental Campaign (`tp80_*`, 80e COCO-pretrained, `batch=16`, `seed=42`, NVIDIA L4)

| Run ID (`outputs/tp80_*`) | Status | Val `mAP@0.50` | Full-1280 `AP@0.50` (`all` / `tiny`) | Tiled-640x512 `AP@0.50` (`all` / `tiny`) | Fused Params / FPS@1280 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `tp80_yolo26n_lgt/colab-seed42` (**GauCon**) | **COMPLETED** | **`0.4224`** | **`0.4413`** / `0.3606` (`AP75=0.0478`, `Dupe=799`) | **`0.4853`** / **`0.4399`** (`COCO AP=0.1696`, `AP75=0.0737`) | `2,402,374` (`+714`) / `73.95 FPS` |
| `tp80_yolo26n_base/colab-seed42` (**Base**) | **COMPLETED** | `0.3996` | `0.4308` / `0.3628` (`AP75=0.0444`, `Dupe=1027`) | `0.4558` / `0.4259` (`COCO AP=0.1555`, `AP75=0.0640`) | `2,401,660` / `69.51 FPS` |
| `tp80_yolo26n_nwd/colab-seed42` (**+NWD**) | **COMPLETED** | `0.3967` | `0.4324` / `0.3574` (`AP75=0.0413`, `Dupe=858`) | `0.4845` / `0.4531` (`COCO AP=0.1651`, `AP75=0.0609`) | `2,401,660` / **`74.36 FPS`** |
| `tp80_yolo11n_nms/colab-seed42` (**YOLO11n-P2 NMS**) | Ep 35/80 (`val=0.3397@29`) | — | — | — | — |
| `tp80_abl_nodwtal/colab-seed42` (**w/o DW-TAL**) | Queued | — | — | — | — |
| `tp80_abl_nogate/colab-seed42` (**w/o FG-Gate & Distill**) | Queued | — | — | — | — |

## Historical Exploratory Campaign (`tp_*`, 150e from scratch, recorded for Course Report evolution discussion)
- `tp_yolo26n_base`: Full-1280 `AP50 = 0.3767`, `tiny1 = 0.1715`, `reasonable = 0.4859`, `FP_Bkg = 9,277`, `FP_Dupe = 1,039`.
- `tp_yolo26n_lgt` (`v1` prior to `DW-TAL` & `FG-Gate`): Full-1280 `AP50 = 0.3699`, `tiny1 = 0.1900` (`+1.85 pp`), `reasonable = 0.4503` (`-3.56 pp`), `FP_Bkg = 14,489` (`+56.2%` soft-label inflation), `FP_Dupe = 956`. Directly motivated `DW-TAL`, `max_active_levels=2`, and `FG-Gated No-BN MLP`.
- `tp_yolo26n_nwd`: Full-1280 `AP50 = 0.3386`.
- `tp_yolo11n_nms`: Full-1280 `AP50 = 0.3958`.

## Six-Layer Literature Verification Log
- **L0 Direct**: `yu2020tinyperson` (WACV 2020), `jiang2021smplus` (ICASSP 2021), `gong2021fusionfactor` (WACV 2021), `hong2022sspnet` (IEEE GRSL 2022), `chao2025bpdyolo` (Sci. Rep. 2025), `wang2020aitod` (ICPR 2021), `zhu2021visdrone` (TPAMI 2022).
- **L1 Core Technique**: `wang2024yolov10` (NeurIPS 2024), `ultralytics2026yolo` (2025), `ultralytics2024yolo11` (2024), `sun2021onenet` (ICML 2021), `zhao2024rtdetr` (CVPR 2024), `carion2020detr` (ECCV 2020), `wang2021nwd` (arXiv 2021), `xu2022nwd_isprs` (ISPRS JPRS 2022), `yang2021gwd` (ICML 2021), `chen2021rfla` (ECCV 2022), `feng2021tood` (ICCV 2021).
- **L2 Mathematical Foundation**: `agueh2011barycenters` (SIAM J. Math. Anal. 2011, 43(2):904–924), `villani2009optimal` (Springer 2009).
- **L3 Seminal**: `lin2014coco`, `ren2015faster`, `lin2017fpn`, `lin2017focal`, `liu2018panet`, `tian2019fcos`, `cai2019cascade`, `zheng2020diou`, `li2020gfl`, `loshchilov2017adamw`, `neubeck2006nms`, `bodla2017softnms`, `ioffe2015batch`, `dai2021dyhead`.
- **L4 Adjacent**: `solovyev2021wbf` (IVC 2021), `hosang2017learning` (CVPR 2017), `zhang2021varifocalnet` (CVPR 2021).
- **L5 Critiques & Diagnostics**: `bolya2020tide` (ECCV 2020), `akyon2022sahi` (ICIP 2022).


## Editorial revision (2026-11-29)

- Updated `docs/report/course_report_vi.md` to remove source-code identifiers, prompt-like wording, unsupported implementation claims, excessive bold formatting, and inaccurate parameter explanations.
- Reconciled the reported TAL exponent, GauCon initialization, NWD comparison settings, bootstrap interpretation, and the NWD scale-shift example against the checked-in configurations and implementation.
- Updated `paper/main.tex` to align terminology and claims with the report, remove unsupported FLOP and transformer-complexity claims, and qualify causal and latency interpretations.
- Updated `scripts/build_dut_report.py` so plain figure/table captions remain recognized, and removed unverified cover metadata.
- Static source checks passed. The committed `course_report_vi.docx` and `course_report_vi.pdf` have not yet been regenerated from this revision or visually inspected. `paper/main.pdf` is not present in the repository at this commit.
