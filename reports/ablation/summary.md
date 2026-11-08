# Component-removal ablation summary

Protocol: frozen TOV tiled evaluation (tile 640x512, overlap 100, global NMS IoU 0.50, top-2000 pre-NMS, max 1000 detections/image, IoD > 0.50 ignore/uncertain).

Seed: 42 for every run, including the reference. Single seed only: no variance estimate is available.

Reference: `gauss_v2_1_tinyperson_wacv` (full GauSS, seed 42), trained by the paper pipeline with a config identical to the ablation configs apart from `name`/`seed`.

## Absolute metrics (AP/MR in %)

| Run | Ablation | AP50_all | AP50_tiny1 | AP50_tiny2 | AP50_tiny3 | AP50_small | MR50_tiny2 | AP75_all |
|---|---|---|---|---|---|---|---|---|
| gauss_v2_1_tinyperson_wacv | none (full GauSS, paper run) | 12.94 | 5.18 | 9.50 | 13.37 | 19.23 | 93.52 | 1.40 |
| abl_gauss_no_high_band_seed42 | no_high_band | - | - | - | - | - | - | - |
| abl_gauss_no_mid_band_seed42 | no_mid_band | - | - | - | - | - | - | - |
| abl_gauss_no_low_band_seed42 | no_low_band | - | - | - | - | - | - | - |
| abl_gauss_no_router_seed42 | no_router | - | - | - | - | - | - | - |
| abl_gauss_no_saliency_seed42 | no_saliency | - | - | - | - | - | - | - |

## Delta vs full GauSS (percentage points)

Negative = the removed component helped. `MR50_tiny2` is a miss rate, so a positive delta there means worse.

| Ablation | AP50_all | AP50_tiny1 | AP50_tiny2 | AP50_tiny3 | AP50_small | MR50_tiny2 | AP75_all |
|---|---|---|---|---|---|---|---|
| no_high_band | incomplete |---|---|---|---|---|---|---|
| no_mid_band | incomplete |---|---|---|---|---|---|---|
| no_low_band | incomplete |---|---|---|---|---|---|---|
| no_router | incomplete |---|---|---|---|---|---|---|
| no_saliency | incomplete |---|---|---|---|---|---|---|

## Incomplete / not-run

- `abl_gauss_no_high_band_seed42`: INCOMPLETE
- `abl_gauss_no_mid_band_seed42`: INCOMPLETE
- `abl_gauss_no_low_band_seed42`: INCOMPLETE
- `abl_gauss_no_router_seed42`: INCOMPLETE
- `abl_gauss_no_saliency_seed42`: INCOMPLETE

## Scope and limitations

- Only the 5 component-removal ablations were trained, at a single seed. No baseline model and no multi-seed runs were trained.
- Because n = 1 per configuration, no confidence interval can be reported. Differences below run-to-run seed noise cannot be distinguished from real effects.
- The reference run was produced by the paper pipeline; it was not retrained for this study. Its config matches the ablation configs apart from `name` and `seed`.

