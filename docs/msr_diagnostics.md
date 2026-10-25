# MSR training diagnostics

Diagnostics are opt-in. Ordinary experiments have no MSR hooks, statistics, files, or W&B calls. `configs/experiment/ablation_msr_p2.yaml` enables diagnostics every 100 training batches. Baseline models in that comparison have no MSR module, so the callback does not create a diagnostics file for them.

## Configuration

```yaml
diagnostics:
  enabled: true
  collect_interval_steps: 100
  terminal_interval_steps: 100
  file_interval_steps: 100
  wandb_interval_epochs: 1
  gradient_interval_steps: 100
  epoch_summary: true
  file_logging: true
  gate_metrics: true
  feature_metrics: true
  residual_metrics: true
  gradient_metrics: true
  numerical_health: true
  performance_metrics: false
  cuda_timing: false
  sampled_elements: 4096
  flush_records: 10
```

Step intervals must be multiples of `collect_interval_steps`. `sampled_elements` uses a deterministic strided sample of flattened feature tensors. The sample is an approximation for feature statistics, min/max, zero fraction, and finite fraction. Compact gate vectors and selected parameter tensors are summarized in full. No activation or graph is saved.

## Metrics

| Group / namespace | Definition | Collection and epoch aggregation | Interpretation |
|---|---|---|---|
| Adaptive raw gates: `fusion/scale_k/gate_raw_*` | Mean, population std, min, max, finite fraction, and fractions `g <= 0.05` / `g >= 0.95` over `[B,C]` gate values | Full gate vector at collection steps; epoch means weighted by nominal batch size, extrema use most extreme sampled batch extrema | Shows gate range and saturation. Does not show detection benefit. |
| Fusion weights: `fusion/scale_k/weight_*` | Mean, population std, min, max, finite fraction of normalized `alpha_k` | Adaptive and SK weights use actual computed weights. Mean mode reports its fixed `1/K` coefficients. Max mode reports no weights. | Shows contribution assigned by fusion. Not a quality score. |
| Entropy: `fusion/entropy`, `fusion/normalized_entropy` | `H=-sum(alpha*log(max(alpha,1e-8)))`; normalized by `log(K)` | Actual normalized weights for adaptive/SK; fixed mean has `H=log(K)`; absent for max | Near zero indicates concentrated scale weighting; near one indicates distributed weights. |
| Evidence: `evidence/scale_k/{mean,std,rms,abs_mean,min,max,finite_frac,zero_frac}` | Statistics of projected `F_k` over deterministic strided sample, population std and RMS | Every collection step; epoch means weighted by nominal batch size, min/max take extrema across sampled batch summaries | Characterizes each scale's evidence without retaining feature maps. |
| Modulation: `modulation/gate_*`, `near_zero_frac`, `near_one_frac` | Statistics of `G`; near-zero means `G <= 0.05`, near-one means `G >= 0.95` | Deterministic strided sample at collection steps; same epoch rule as evidence | Describes mask values, not whether suppression is useful. |
| Modulated feature: `modulation/modulated_{rms,abs_mean}` | RMS and absolute mean of `M=X*G` | Deterministic strided sample at collection steps; batch-size-weighted epoch mean | Shows feature magnitude after gating. |
| Residual: `residual/*` | `gamma`; mean/std/RMS/finite fraction of `R`; RMS of `gamma*R`, `X`, `Y`; update/input ratio `RMS(gamma*R)/(RMS(X)+1e-12)` | Deterministic strided sample at collection steps; batch-size-weighted epoch mean | Measures correction scale. Ratio is diagnostic, not an objective. |
| Numerical health: `output/finite_frac`, `gamma_finite`, gate/weight/parameter finite fractions | Fraction of sampled tensor values finite; selected scalar/tensor finite checks | Sampled feature checks at collection steps; selected parameters at epoch end | Detects non-finite values. No automatic repair is applied. |
| Parameters: `grad/<group>_parameter_{norm,rms,finite_frac}` | L2 norm, RMS, and finite fraction of selected weights | Full selected tensors, once at epoch end; snapshot, not epoch average | Tracks parameter scale and numerical health. |
| Gradients: `grad/<group>_grad_{norm,rms,finite_frac,missing_frac,to_parameter_ratio}` | Norm/RMS and finite status of sampled microbatch gradient contributions; GradScaler scale is divided out | Gradient hook at configured collection batches, before Ultralytics unscale/clip/optimizer step; means weighted by nominal batch size | Gradient snapshot is pre-clipping and represents the sampled microbatch contribution, not accumulated optimizer gradient. Missing is explicit. |
| Performance, opt-in: `performance/*` | CUDA event or CPU wall-time estimate for MSR forward; host duration for collection and sink calls | Forward timings are averaged at epoch end; event synchronization occurs once at epoch boundary. Sink times appear in the next record. | CUDA events include queued GPU work. W&B time measures host enqueue duration, not network completion. |

Epoch records include `aggregation_rules`, `epoch`, `global_step`, `module`, `fusion_mode`, and sampling metadata. Global step counts training batches, not optimizer updates. Feature and scalar summaries are means of periodic batch summaries weighted by nominal batch size. Min/max are extrema of sampled batch min/max. Parameter metrics are epoch-end snapshots. Gradient metrics are averages of sampled pre-clipping microbatch gradients.

## Output and isolation

- W&B namespace: `msr/layer_29/...` (one batched `wandb.log` call per configured epoch, `commit=False`, reusing the active run).
- Local append-only records: `<run>/diagnostics/msr_metrics.jsonl`; flush every `flush_records` entries, close on normal completion and trainer cleanup.
- Terminal: one concise line per configured sample and epoch summary.
- No detector precision, recall, AP, mAP, images, histograms, CAM, or attribution metrics are emitted.
- W&B, terminal, and file errors disable only the failing sink. Collector errors disable diagnostics; model forward, loss, backward, and optimizer exceptions retain normal trainer behavior.
- Gradient hooks return the original gradient and retain only detached scalar summaries. Forward hooks are installed only for selected diagnostic batches and removed at batch end.

Host collection and sink call durations are measured only when `performance_metrics` is enabled. W&B timing measures host enqueue duration, not network completion; sink timings appear in the next record. CUDA forward timing additionally requires `cuda_timing: true`; it adds events and one synchronization at epoch boundaries. Do not enable it for ordinary throughput measurements.
