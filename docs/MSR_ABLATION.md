# Multi-scale Standardized Residual (MSR)

## Research hypothesis

A local feature is more informative when its positive deviation is measured relative to the variation of its surrounding background, not only by absolute contrast. The module tests this hypothesis on the P2/4 feature map of YOLO26n-P2. It is a research hypothesis, not an established accuracy improvement.

## Formula

For feature map \(F\), channel \(c\), spatial location \(p\), and dilation \(d\), let \(\mathcal{N}_d(p)\) be the eight surrounding positions at offsets \((\Delta x,\Delta y)\in\{-d,0,d\}^2\setminus\{(0,0)\}\).

\[
\mu_{c,d}(p)=\frac{1}{8}\sum_{q\in\mathcal{N}_d(p)}F_c(q)
\]

\[
v_{c,d}(p)=\max\left(\frac{1}{8}\sum_{q\in\mathcal{N}_d(p)}F_c(q)^2-\mu_{c,d}(p)^2,0\right)
\]

The raw-residual control is \(R_{c,d}(p)=\operatorname{ReLU}(F_c(p)-\mu_{c,d}(p))\). MSR standardizes that deviation by local variation:

\[
R_{c,d}(p)=\operatorname{ReLU}\left(\frac{F_c(p)-\mu_{c,d}(p)}{\sqrt{v_{c,d}(p)+\epsilon}}\right)
\]

Combine scales using a per-element maximum, then apply a bounded, learnable residual gain:

\[
R(p)=\max_{d\in\mathcal{D}}R_d(p),\qquad
Y=F+gR,\qquad g=g_{\max}\tanh(a)
\]

The gain parameter \(a\) starts at zero, making the module an exact identity at initialization. The response is clipped for numerical robustness. Ring statistics are computed in FP32 under AMP; replication padding avoids the artificial zero-valued background that ordinary zero padding would introduce at image boundaries.

A dense \(1\times1\) channel projection is deliberately omitted: on the high-resolution P2 map it would add avoidable compute. The single bounded gain is a low-cost first implementation of the standardized-residual hypothesis.

## Placement and routing

The model YAML inserts an nn.Identity at layer 20, the P2 detection feature. The P2 Detect branch consumes layer 20. The next downsampling layer explicitly consumes layer 19 instead, so P3/P4/P5 continue to receive the ungated P2 feature. This keeps the intervention confined to the highest-resolution prediction branch.

The model is designed for the YOLO26-nano/P2 experiment. The installation code validates the layer indices and routing instead of silently applying the module to an incompatible model.

## Controlled screening ablation

The experiment config runs four variants with identical dataset, architecture YAML, seed, optimizer, augmentation, training horizon and hardware:

| Variant | Local response | Dilation(s) | What it tests |
|---|---|---|---|
| A0 identity | Identity | — | Baseline with the same graph and gate slot |
| A1 raw multi-scale | Raw positive residual | 1, 2, 4 | Multi-scale residual without variance normalization |
| A2 standardized single | Standardized residual | 1 | Normalization without multi-scale dilation |
| A3 standardized multi-scale | Standardized residual | 1, 2, 4 | Proposed MSR module |

Primary contrasts: A3−A0 (overall module), A3−A1 (normalization), and A3−A2 (multi-scale effect). Use recall at a fixed confidence, AP by object size, precision/false positives, and actual latency/VRAM. The experiment has one seed and is only a screening stage; promising results should be repeated across seeds and verified on an untouched test set.

## Commands

    uv sync --extra cpu --group dev
    uv run --extra cpu --group dev pytest tests/test_msr.py tests/test_shapes.py -q
    uv run --extra cpu modal run modal/app.py::compare --config configs/experiment/ablation_msr.yaml

GPU training is intentionally not launched as part of this implementation/PR. No accuracy or latency gain is claimed.
