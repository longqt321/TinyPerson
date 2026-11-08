import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path("modal").resolve()))
import app as modal_app_module

from tinydet.utils.config import load_config
from tinydet.utils.experiment import git_info

app = modal_app_module.app

@app.local_entrypoint()
def main():
    configs = [
        "configs/experiment/baseline_tinyperson_wacv.yaml",
        "configs/experiment/gauss_v3_tinyperson_wacv.yaml",
    ]

    print("=" * 70, flush=True)
    print("PHASE 1: STARTING 100-EPOCH PARALLEL TRAINING ON MODAL (NVIDIA A10)", flush=True)
    print("Run 1: Baseline YOLO26n-P2 (100 epochs)", flush=True)
    print("Run 2: GauSS v3 YOLO26n-P2 (100 epochs)", flush=True)
    print("=" * 70, flush=True)

    # Preflight audit
    exp0 = load_config(configs[0])
    audit = modal_app_module._audit_original_dataset.remote(
        str(exp0.data["config"]), int(exp0.train["imgsz"]), uuid.uuid4().hex[:12]
    )
    if not audit["passed"]:
        raise ValueError("Dataset audit failed!")

    commit, dirty = git_info()

    # 1. Spawn parallel training runs
    train_handles = []
    for cfg_path in configs:
        exp = load_config(cfg_path)
        exp.data["preflight_signature"] = audit["dataset_signature_metadata_sha256"]
        config_text = yaml.safe_dump(asdict(exp), sort_keys=False)
        gpu = exp.hardware.get("gpu", "A10")
        options = {"gpu": gpu}
        if exp.output.get("wandb_project"):
            options["secrets"] = [modal_app_module.wandb_secret]

        print(f"--> Spawning 100e training for: {cfg_path}...", flush=True)
        handle = modal_app_module._train.with_options(**options).spawn(
            cfg_path, commit, dirty, config_text=config_text, dataset_audit=audit
        )
        train_handles.append((cfg_path, handle))

    # Wait for training
    run_paths = {}
    for cfg_path, handle in train_handles:
        run_full_path = handle.get()
        print(f"[TRAIN COMPLETED] {cfg_path} -> {run_full_path}", flush=True)
        rel_run = str(Path(run_full_path).relative_to(modal_app_module.RUN_ROOT))
        modal_app_module._finalize_pipeline(rel_run)
        run_paths[cfg_path] = rel_run

    b_run = run_paths[configs[0]]
    d_run = run_paths[configs[1]]

    b_ckpt = f"{b_run}/training/weights/last.pt"
    d_ckpt = f"{d_run}/training/weights/last.pt"

    print("\n" + "=" * 70, flush=True)
    print("PHASE 2: RUNNING STANDARDIZED TILED EVALUATION (640x512, TOV PROTOCOL)", flush=True)
    print("=" * 70, flush=True)

    tiled_targets = [
        ("baseline_tinyperson_100e_tiled", b_ckpt),
        ("gauss_v2_1_tinyperson_100e_tiled", d_ckpt),
    ]

    tiled_handles = []
    for name, ckpt in tiled_targets:
        print(f"--> Spawning tiled evaluation for {name} ({ckpt})...", flush=True)
        handle = modal_app_module._run_tiled_checkpoint_evaluation.spawn(
            relative_checkpoint=ckpt,
            name=name,
        )
        tiled_handles.append((name, handle))

    tiled_results = {}
    for name, handle in tiled_handles:
        res = handle.get()
        tiled_results[name] = res
        output_file = Path("reports/eval") / f"{name}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(res, indent=2) + "\n")
        print(f"[EVAL DONE] Finished tiled evaluation for {name}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("PHASE 3: RUNNING FULL-FRAME TEST EVALUATION (1280px)", flush=True)
    print("=" * 70, flush=True)

    full_targets = [
        ("baseline_tinyperson_100e_full", b_ckpt),
        ("gauss_v2_1_tinyperson_100e_full", d_ckpt),
    ]

    full_handles = []
    for name, ckpt in full_targets:
        print(f"--> Spawning full-frame evaluation for {name} ({ckpt})...", flush=True)
        handle = modal_app_module._run_checkpoint_evaluation.spawn(
            relative_checkpoint=ckpt,
            name=name,
            split="test",
        )
        full_handles.append((name, handle))

    full_results = {}
    for name, handle in full_handles:
        res = handle.get()
        full_results[name] = res
        output_file = Path("reports/eval") / f"{name}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(res, indent=2) + "\n")
        print(f"[EVAL DONE] Finished full-frame evaluation for {name}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ALL 100-EPOCH TRAINING AND EVALUATIONS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
