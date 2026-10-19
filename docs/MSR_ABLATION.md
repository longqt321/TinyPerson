# Multi-scale Standardized Residual (MSR): full neck-fusion design

## Research motivation

The first MSR prototype was a single-map response enhancer. This revision treats MSR as a detector feature-processing system rather than a scalar residual plug-in. The design has three pieces:

1. **Standardized Local Evidence (SLE):** positive center-vs-ring residual divided by local standard deviation.
2. **Multi-scale Evidence Aggregation (MEA):** fixed max pooling or response-adaptive softmax over dilations.
3. **Evidence-guided Cross-level Fusion (ECF):** a low-level evidence map estimates a channel/spatial attention map that modulates the high-level feature in a learned fusion operator.

This is an architecture hypothesis; implementation and equations do not establish that accuracy improves on TinyPerson.

## Local evidence

For feature map \(F\), channel \(c\), location \(p\), and dilation \(d\), \(\mathcal{N}_d(p)\) is the eight-position ring at offsets in \(\{-d,0,d\}^2\), excluding the center:

\[
\mu_{c,d}(p)=\frac{1}{8}\sum_{q\in\mathcal{N}_d(p)}F_c(q)
\]

\[
v_{c,d}(p)=\max\left(
\frac{1}{8}\sum_{q\in\mathcal{N}_d(p)}F_c(q)^2-\mu_{c,d}(p)^2,0
\right)
\]

\[
E_{c,d}(p)=\operatorname{ReLU}
\left(\frac{F_c(p)-\mu_{c,d}(p)}{\sqrt{v_{c,d}(p)+\epsilon}}\right)
\]

Ring means and variances are computed in FP32, using replicate padding and depth-wise convolution. This operates on feature maps; it does not assume that raw RGB intensity contrast directly identifies a person.

## Scale aggregation

The MLC control uses max aggregation, analogous to ALCNet's scale max-pooling:

\[
E_c(p)=\max_{d\in\mathcal{D}}E_{c,d}(p)
\]

The adaptive variant instead uses response-derived local scale scores and trainable scale biases:

\[
s_d(p)=\frac{1}{C}\sum_c E_{c,d}(p)+b_d,\qquad
\alpha_d(p)=\operatorname{softmax}_d(s_d(p)/T)
\]

\[
E_c(p)=\sum_d\alpha_d(p)E_{c,d}(p)
\]

This form has spatially varying scale weights while keeping the gate light. Its key ablation is full-max versus full-adaptive; a gain must be demonstrated experimentally rather than assumed.

## Cross-level fusion

Each of the six Identity slots in configs/model/yolo26n-msr-p2.yaml follows a YOLO neck concatenation. The tensor is split into its high-level and lateral low-level branches. The MSR evidence \(E(X)\) is computed from the low-level feature \(X\), then drives a bottleneck point-wise attention network:

\[
A(X)=\sigma\bigl(\operatorname{BN}(
W_2 * \delta(\operatorname{BN}(W_1 * [X,E(X)])))\bigr)
\]

Because YOLO's neighbouring pyramid stages have different channel counts, \(P(E(X))\) projects evidence into the high-level channel space. The full fusion operation is:

\[
Z=P(E(X))+A(X)\odot Y
\]

The gate concatenates \(Z\) with the unmodified \(X\), preserving the expected input channel count of the next C3k2 stage. It retains a separate shallow detail path while letting evidence decide where high-level semantics should contribute. This is a learned cross-level fusion, not a global scalar-gain residual.

The six slots cover top-down P5→P4, P4→P3 and P3→P2, plus bottom-up P2→P3, P3→P4 and P4→P5. Channel splits and graph routing are checked by the installer, which is deliberately restricted to the nano scale used by these experiments.

## Controlled ablation

configs/experiment/ablation_msr.yaml defines four runs with the same model YAML, dataset, seed, optimizer, augmentation, image size, epoch count, and hardware:

| Variant | Evidence aggregation | Cross-level ECF | Purpose |
|---|---|---|---|
| A0 msr_a0_identity | None | None | Baseline with the same six Identity slots |
| A1 msr_a1_local_max | Max over scales | None; refine low-level branch only | Test local evidence/attention |
| A2 msr_a2_full_max | Max over scales | Enabled | Test cross-level fusion |
| A3 msr_a3_full_adaptive | Adaptive weighted scales | Enabled | Test adaptive scale selection in the full system |

Primary contrasts: A1−A0, A2−A1, and A3−A2. Collect overall AP, size-wise AP/recall, fixed-confidence recall, false positives per image, parameter count, FLOPs, actual latency, and peak VRAM. Screen with 5 epochs first, then run the full schedule; promising results should be repeated across seeds and evaluated on a held-out test split.

## Original-image preflight

The dataset preflight reads images directly from disk with Pillow before YOLO preprocessing. It reports original width/height/aspect-ratio distributions, file formats, colour modes, estimated bit-depth category, EXIF orientation, file size, corrupt files, labels, native object-size distribution and a theoretical square-letterbox simulation at the selected imgsz. Label dimensions are reconstructed from normalized YOLO boxes or polygons, so the report can estimate how small objects become at a candidate input size.

This simulation is deliberately labelled theoretical: mosaic, random scaling, crop/augmentation, and EXIF orientation transforms are not applied by the audit. The full report stores one metadata record per image in JSON. When original TinyPerson COCO JSON is present, the audit compares declared source dimensions with on-disk rasters by image basename. The report marks source metadata as `verified`, `partial`, `mismatch`, `unmatched`, or `not_found`. Use `--require-source-metadata` to block training unless every relevant source metadata image matches; use `--strict-resolution` to block when at least half of parsed objects have square-root-area below 4 pixels after simulated resize. The 4-pixel threshold is a diagnostic policy, not a claim that all objects below 4 px are impossible to detect.

Run the audit without starting training:

    uv run --extra cpu python scripts/audit_dataset.py --data /path/to/data.yaml --imgsz 1280
    uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/ablation_msr.yaml --preflight-only
    uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/ablation_msr.yaml --preflight-only --require-source-metadata

## Commands

Local CPU integration tests:

    uv sync --extra cpu --group dev
    uv run --extra cpu --group dev pytest tests/test_msr.py tests/test_data_preflight.py tests/test_shapes.py -q
    uv run --extra cpu ruff check .

Five-epoch screen of every ablation, with four-way parallel training:

    uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/ablation_msr.yaml --smoke --parallel

Full training after reviewing the preflight report:

    uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/ablation_msr.yaml --parallel

Add --strict-resolution to stop before GPU training when the severe resize-risk criterion is met. The full audit is stored in the Modal run volume under preflight/. Each trained run also stores the condensed audit summary in diagnostics/dataset_preflight_summary.json.

## Relation to ALCNet and scope

ALCNet uses directional difference-product local contrast (DLC), max-pools contrast responses over dilation scales as MLC, and uses BLAM attention derived from low-level features to modulate high-level features. It was designed for infrared small-target segmentation, not RGB person detection. This implementation adapts the architectural idea, not the original equations or training recipe: the local evidence is variance-standardized, adaptive scale aggregation is an experimental extension, and the cross-level fusion is integrated at YOLO26's existing neck concatenations. See the [ALCNet paper](https://arxiv.org/abs/2012.08573) and the [original TinyPerson paper](https://openaccess.thecvf.com/content_WACV_2020/html/Yu_Scale_Match_for_Tiny_Person_Detection_WACV_2020_paper.html) for their distinct tasks and datasets.
