import sys
import uuid
import yaml
from pathlib import Path
from dataclasses import asdict

import modal

sys.path.insert(0, str(Path("modal").resolve()))
import app as modal_app_module

from tinydet.utils.config import load_config
from tinydet.utils.experiment import git_info

app = modal_app_module.app

@app.local_entrypoint()
def main():
    configs = [
        "configs/experiment/baseline_tinyperson_wacv.yaml",
        "configs/experiment/dcaf_v2_1_tinyperson_wacv.yaml",
    ]

    print("=" * 70, flush=True)
    print("STARTING PARALLEL TRAINING RUNS ON MODAL (NVIDIA A10)", flush=True)
    print("Run 1: Baseline YOLO26n-P2 (TinyPerson WACV)", flush=True)
    print("Run 2: DCAF v2.1 YOLO26n-P2 (TinyPerson WACV)", flush=True)
    print("=" * 70, flush=True)

    # 1. Preflight audit
    exp0 = load_config(configs[0])
    audit = modal_app_module._audit_original_dataset.remote(
        str(exp0.data["config"]), int(exp0.train["imgsz"]), uuid.uuid4().hex[:12]
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

    print("\n[SUCCESS] Both training runs spawned simultaneously in parallel!", flush=True)
    print("Waiting for training runs to complete...", flush=True)

    # 3. Gather results
    for cfg_path, handle in train_handles:
        run = handle.get()
        print(f"\n[COMPLETED] Run for {cfg_path}: {run}", flush=True)
        try:
            rel_path = str(Path(run).relative_to(modal_app_module.RUN_ROOT))
            modal_app_module._finalize_pipeline(rel_path)
            print(f"Finalized pipeline for {rel_path}", flush=True)
        except Exception as e:
            print(f"Warning during finalize: {e}", flush=True)

    print("\nALL PARALLEL RUNS FINISHED SUCCESSFULLY!", flush=True)
