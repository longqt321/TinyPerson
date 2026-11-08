# GauSS report outline

- **Title/Abstract:** GauSS v2.1, TinyPerson, single-seed scope and principal tiled AP@0.50 result.
- **Introduction:** tiny-person detection challenge; GauSS scope; contributions; no unsupported novelty or SOTA language.
- **Related work:** categories for tiny-object detection, high-resolution/multiscale features, dilated context, frequency decomposition, background suppression. Bibliography deferred until primary sources verified.
- **Method:** architecture from GauSS cycle 2 source; training setup from paired YAMLs; evaluator protocol from JSON artifacts.
- **Experiments:** tiled main table (Table 1), full-frame comparison (Table 2), validation separated from test; discuss regressions and single-seed limit.
- **Limitations:** one seed, nondeterministic training, last checkpoint, standalone evaluator, dense-image handling, missing component ablations.
- **Conclusion:** descriptive evidence only; no significance or official-protocol claim.

Figures deferred until architecture source and raw profiling are checked. Result tables can be regenerated from JSON artifacts; current draft values cross-checked manually against those files.
