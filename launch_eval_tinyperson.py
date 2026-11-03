import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("modal").resolve()))
import app as modal_app_module

app = modal_app_module.app

@app.local_entrypoint()
def main():
    checkpoints = [
        {
            "name": "baseline_tinyperson_wacv",
            "checkpoint": "baseline_tinyperson_wacv/20261102-073649-c626816-seed42/training/weights/last.pt",
        },
        {
            "name": "dcaf_v2_1_tinyperson_wacv",
            "checkpoint": "dcaf_v2_1_tinyperson_wacv/20261102-073650-c626816-seed42/training/weights/last.pt",
        },
    ]

    print("=" * 70, flush=True)
    print("STARTING TEST EVALUATION ON TINYPERSON (786 TEST IMAGES)", flush=True)
    print("=" * 70, flush=True)

    handles = []
    for item in checkpoints:
        name = item["name"]
        ckpt = item["checkpoint"]
        print(f"--> Spawning evaluation for {name} ({ckpt}) on split='test'...", flush=True)
        handle = modal_app_module._run_checkpoint_evaluation.spawn(
            relative_checkpoint=ckpt,
            name=name,
            split="test",
        )
        handles.append((name, handle))

    results = {}
    for name, handle in handles:
        res = handle.get()
        results[name] = res
        output_file = Path("reports/eval") / f"{name}_test.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(res, indent=2) + "\n")
        print(f"\n[DONE] Finished test evaluation for {name}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ALL EVALUATIONS COMPLETED. SUMMARY:", flush=True)
    for name, res in results.items():
        bench = res["summary"].get("benchmark", {})
        print(f"\n--- {name} ---", flush=True)
        print(f"AP50 (all):     {bench.get('all', {}).get('AP@0.50')}", flush=True)
        print(f"AP50 (tiny):    {bench.get('tiny', {}).get('AP@0.50')}", flush=True)
        print(f"AP50 (tiny1):   {bench.get('tiny1', {}).get('AP@0.50')}", flush=True)
        print(f"AP50 (tiny2):   {bench.get('tiny2', {}).get('AP@0.50')}", flush=True)
        print(f"AP50 (tiny3):   {bench.get('tiny3', {}).get('AP@0.50')}", flush=True)
        print(f"AP50 (small):   {bench.get('small', {}).get('AP@0.50')}", flush=True)
        print(f"AP50 (reas):    {bench.get('reasonable', {}).get('AP@0.50')}", flush=True)
        print(f"Mean IoU (all): {bench.get('all', {}).get('AP_mean_IoU_0.25_0.50_0.75')}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
