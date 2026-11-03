import json
import sys
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
        "configs/experiment/baseline_visdrone_finetune.yaml",
        "configs/experiment/dcaf_v2_1_visdrone_finetune.yaml",
    ]

    print("=" * 70, flush=True)
    print("STARTING PARALLEL VISDRONE FINE-TUNING ON MODAL (NVIDIA A10)", flush=True)
    print("Epochs: 10")
    print("Pretrained weights: TinyPerson 150e checkpoints")
    print("Run 1: Baseline YOLO26n-P2 VisDrone Fine-tune", flush=True)
    print("Run 2: DCAF v2.1 YOLO26n-P2 VisDrone Fine-tune", flush=True)
    print("=" * 70, flush=True)

    # 1. Preflight audit on VisDrone dataset
    exp0 = load_config(configs[0])
    audit = modal_app_module._audit_original_dataset.remote(
        str(exp0.data["config"]), int(exp0.train["imgsz"]), "visdrone-ft-audit"
    )
    print(f"Preflight status: passed={audit['passed']}", flush=True)
    if not audit["passed"]:
        raise ValueError("Dataset preflight audit failed!")

    commit, dirty = git_info()

    # 2. Spawn both training runs asynchronously
    train_handles = []
    for cfg_path in configs:
        exp = load_config(cfg_path)
        exp.data["preflight_signature"] = audit["dataset_signature_metadata_sha256"]
        config_text = yaml.safe_dump(asdict(exp), sort_keys=False)
        gpu = exp.hardware.get("gpu", "A10")
        options = {"gpu": gpu}
        if exp.output.get("wandb_project"):
            options["secrets"] = [modal_app_module.wandb_secret]

        print(f"--> Spawning remote A10 container for: {cfg_path}...", flush=True)
        handle = modal_app_module._train.with_options(**options).spawn(
            cfg_path, commit, dirty, config_text=config_text, dataset_audit=audit
        )
        train_handles.append((cfg_path, handle))

    print("\n[SUCCESS] Both VisDrone training runs spawned simultaneously in parallel!", flush=True)
    print("Waiting for training runs to complete...", flush=True)

    # 3. Gather results
    completed_runs = []
    for cfg_path, handle in train_handles:
        run_res = handle.get()
        print(f"\n[COMPLETED] Training run for {cfg_path}: {run_res}", flush=True)
        completed_runs.append((cfg_path, run_res))

    print("\n" + "=" * 70, flush=True)
    print("STARTING VISDRONE VALIDATION & TEST EVALUATION...", flush=True)
    print("=" * 70, flush=True)

    eval_handles = []
    for cfg_path, run_dir in completed_runs:
        rel_name = Path(cfg_path).stem
        # Check actual checkpoint folder
        for split in ["val", "test"]:
            eval_name = f"{rel_name}_{split}_10e"
            print(f"--> Spawning evaluation for {eval_name} on {run_dir}...", flush=True)
            ckpt_path = f"{Path(run_dir).name}/training/weights/last.pt"
            h = modal_app_module._run_visdrone_val.spawn(
                relative_checkpoint=ckpt_path,
                name=eval_name,
                split=split,
            )
            eval_handles.append((eval_name, h))

    eval_results = {}
    for eval_name, h in eval_handles:
        print(f"Waiting for eval: {eval_name}...", flush=True)
        res = h.get()
        print(f"✓ {eval_name} completed: mAP50={res['map50']:.4f}, mAP50-95={res['map50_95']:.4f}", flush=True)
        eval_results[eval_name] = res

    Path("reports/eval").mkdir(parents=True, exist_ok=True)
    with open("reports/eval/visdrone_finetune_10e_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    print("\nSaved all VisDrone results to reports/eval/visdrone_finetune_10e_results.json", flush=True)
