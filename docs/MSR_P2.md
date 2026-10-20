# Isolated P2 MSR experiment

## Architecture

The inspected YOLO26n YAMLs confirm A0/A1 use Detect strides `[8, 16, 32]`; A2/A3 use `[4, 8, 16, 32]`. A0/A2 retain `SPPF`; A1/A3 replace it with `Identity`. E1/E3 name `MSR` directly at layer 29, after the P2 neck branch, and route its output to Detect. Importing `tinydet.modules.msr` registers `MSR` in Ultralytics' YAML parser namespace. P3/P4/P5 remain on their original graph paths.

For each odd kernel in `{3,5,7}`, MSR computes valid-pixel local mean and variance in FP32, clips signed standardized residuals, projects concatenated positive/negative responses, predicts per-channel scale gates, normalizes the weighted sum, and applies a sigmoid feature gate plus depthwise 3x3 refinement. A learnable residual coefficient starts at `1e-3`. This signed split and scale gating are hypotheses; the four-run study tests the full module, while later ablations should isolate those choices.

## Dataset and training pipeline

The configured dataset is `/mnt/data/tinyperson_official_v1/data.yaml`, which is in the Modal volume and not mounted in this local workspace. The latest available preflight artifact (`outputs/tiny_A2_p2/20261019-194809-8326468-seed42/diagnostics/dataset_preflight_summary.json`) reports 746 images total: 662 train and 84 validation; 14,825 and 3,608 objects respectively; zero corrupt images, missing label files, or out-of-range boxes; source COCO dimensions verified for all 746 images. At `imgsz=1280`, train/validation median square-root bbox areas are 8.94/10.04 px; 42.9%/34.1% are below 8 px. The official test set is not part of this derivative. The dataset preparation code combines the two TinyPerson categories as YOLO class 0 (`person`). The report does not include a duplicate-file audit.

Preparation stores YOLO-normalized boxes using original raster dimensions. Ultralytics' normal detection dataloader then applies its native letterbox/geometry transform; the experiment leaves that loader and all configured augmentations unchanged. `train()` calls `set_seed(seed, deterministic)` and forwards the full train mapping to `model.train()`, including `patience`, `amp`, `close_mosaic`, and the augmentation values. No external pretrained checkpoint is configured in this comparison; same seed and same source architecture initialize the paired runs from the same random baseline weights. The optional `model.pretrained` path supports A2/A3 checkpoint transfer; the P2 MSR loader remaps legacy Detect layer 29 to layer 30 and reports matched, remapped, and skipped tensors.

## Run

Local integration smoke: one synthetic batch through the actual YOLO26 end-to-end criterion, backward, optimizer update, state-dict save/reload, and prediction. Use a Modal GPU to check CUDA AMP and the real dataset pipeline.

```bash
uv run --extra cpu --group dev pytest tests/test_msr.py -q
uv run --extra cpu python scripts/smoke_msr.py
uv run --extra cpu python scripts/validate_tiny_ablation.py \
  configs/model/yolo26n_tiny_A2_p2_msr.yaml \
  configs/model/yolo26n_tiny_A3_p2_no_sppf_msr.yaml --imgsz 1280
uv run --extra cpu modal run modal/app.py::compare \
  --config configs/experiment/ablation_msr_p2.yaml --smoke --parallel
uv run --extra cpu modal run modal/app.py::compare \
  --config configs/experiment/ablation_msr_p2.yaml --parallel
```

`--smoke` limits all four jobs to five epochs. Full runs use 50 epochs and patience 15. Each run has a distinct output directory under `/mnt/runs`; the Modal workflow saves config, environment, metadata, train metrics/plot, validation metrics, and `best.pt`/`last.pt`. W&B remains enabled but local artifacts do not depend on W&B syncing.

## Evaluation

Compare E1 against E0 and E3 against E2 only. Native Ultralytics validation records AP@[.50:.95], AP50, AP75, precision, and recall on the same 84-image validation split. TinyPerson-specific size AP can be computed with the existing `scripts/predict_tinyperson_test.py` and `scripts/evaluate_tinyperson_benchmark.py`, which use original-image boxes and TinyPerson's size bins; keep its settings fixed across runs and do not tune on the official test split.

The separate `scripts/evaluate_coco_small.py` computes COCO bbox AP_small using the official train annotation JSON, the prepared `manifest.json` validation membership, and cached native-validator predictions. It maps both person categories to one class and uses original-coordinate bbox area. It requires optional `pycocotools` installed in the evaluation environment. It has not been run here because dataset files and `pycocotools` are unavailable locally. Report validation results as validation, not test.

After a run completes, download its native validation predictions and the dataset metadata, install `pycocotools`, then evaluate:

```fish
modal volume get tiny-person-runs /MODEL/RUN_ID/evaluation-val outputs/MODEL/RUN_ID/evaluation-val
modal volume get tiny-person-data /tinyperson_official_v1/annotations/tiny_set_train.json data/tiny_set_train.json
modal volume get tiny-person-data /tinyperson_official_v1/manifest.json data/manifest.json
uv run --extra cpu --with pycocotools python scripts/evaluate_coco_small.py \
  --run outputs/MODEL/RUN_ID \
  --annotations data/tiny_set_train.json --manifest data/manifest.json
```

Alternatively, add `evaluation` to `tools/download_results.py --include` to download `evaluation-val` with other run artifacts.

## Current verification boundary

At `imgsz=1280`, the Ultralytics profiler reports 2,525,313 parameters / 33.554 GFLOPs for A2+MSR and 2,360,705 / 33.024 GFLOPs for A3+MSR. MSR adds 9,337 parameters. The FLOP profiler sees convolution/linear work but may omit pooling and tensor arithmetic inside MSR, so treat GFLOPs as an estimate.

No full training run was launched. This workspace has Python 3.12.13 in `.venv`, PyTorch 2.14.0 CPU-only, Ultralytics 8.4.164, and no CUDA device. Therefore CUDA AMP, a real-dataset criterion batch, GPU memory, latency, and full-run metrics remain unverified. The existing completed A0–A3 runs are not MSR results.
