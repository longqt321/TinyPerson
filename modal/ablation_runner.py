"""Parallel ablation runner for the GauSS paper (component-removal ablations).

Scope (user decision, 2026-11-05): ONLY the 5 component-removal ablations, all at
seed 42, trained in PARALLEL on separate Modal A10 GPUs. No baseline runs and no
multi-seed runs are trained, even if budget remains.

The full-GauSS reference at seed 42 is NOT retrained: the paper run
`gauss_v2_1_tinyperson_wacv` (seed 42) already exists on the runs volume and its
config is byte-identical to the ablation configs apart from `name`/`seed`. Its
frozen TOV tiled evaluation is reused as the comparison point.

Usage:
    modal run modal/ablation_runner.py::status          # run status listing
    modal run modal/ablation_runner.py::run_all         # train all 5 in parallel
    modal run modal/ablation_runner.py::run_all_wait    # same, blocking locally
    modal run modal/ablation_runner.py::evaluate_all    # frozen TOV tiled eval
    modal run modal/ablation_runner.py::pilot           # timing/VRAM pilot
    modal run modal/ablation_runner.py::pilot_report    # per-epoch timings
    modal run modal/ablation_runner.py::cleanup_pilots  # remove pilot dirs

Budget note: the 10 h GPU budget applies to total GPU-hours, not wall clock.
Running 5 containers in parallel costs the same GPU-hours as running them
sequentially; only wall clock shrinks (~8.7 h -> ~1.8 h).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from datetime import UTC

import app as base_app

app = base_app.app
runs = base_app.runs
image = base_app.image
CPU_LIMIT = base_app.CPU_LIMIT
MEMORY_LIMIT_MB = base_app.MEMORY_LIMIT_MB
wandb_secret = base_app.wandb_secret

RUN_ROOT = base_app.RUN_ROOT

SEED = 42
ABLATIONS: tuple[str, ...] = (
    "no_high_band",
    "no_mid_band",
    "no_low_band",
    "no_router",
    "no_saliency",
)

# The five runs in this study. `ablation` is passed as a function argument to the
# training container (never via the runner's environment, which does not cross
# the container boundary).
PLAN: list[dict] = [
    {"config": f"configs/experiment/ablation/abl_gauss_{name}_seed{SEED}.yaml",
     "ablation": name}
    for name in ABLATIONS
]

# Full-GauSS seed-42 reference: trained by the paper pipeline, evaluated with the
# same frozen TOV tiled protocol. Reused instead of retrained.
REFERENCE = {
    "name": "gauss_v2_1_tinyperson_wacv",
    "run_dir": f"{RUN_ROOT}/gauss_v2_1_tinyperson_wacv/20261102-092728-c626816-seed42",
    "eval_file": "reports/eval/gauss_v2_1_tinyperson_150e_tiled.json",
}

# Measured on one Modal A10 with the real configs (5-epoch GauSS pilot):
# per-epoch deltas [198.6, 197.8, 34.4, 33.6, 239.6], training_seconds 1198.4.
# Steady state is the median of the middle epochs; the one-time cost covers the
# two warmup epochs, the final validation, W&B sync and container startup.
MEASURED_S_PER_EPOCH = 34.0
MEASURED_ONE_TIME_SECONDS = 1055.0
ESTIMATED_RUN_SECONDS = 150 * MEASURED_S_PER_EPOCH + MEASURED_ONE_TIME_SECONDS


def _run_dir(config_name: str) -> Path:
    return RUN_ROOT / config_name


def _latest_run(config_name: str) -> Path | None:
    """Return the run directory for a config.

    Prefers the deterministic run_id written by _train_ablation; falls back to the
    most recent timestamped directory for runs created before that change.
    """
    root = _run_dir(config_name)
    if not root.is_dir():
        return None
    exact = root / config_name
    if exact.is_dir():
        return exact
    candidates = sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.endswith("-pilot")
    )
    return candidates[-1] if candidates else None


@app.function(
    image=image,
    gpu="A10",
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    cpu=CPU_LIMIT,
    memory=MEMORY_LIMIT_MB,
    # Parallel on separate GPUs: one container per ablation, five ablations.
    max_containers=len(ABLATIONS),
    volumes=base_app.mounts,
    timeout=21600,
)
def _train_ablation(config: str, ablation: str, config_text: str,
                    epochs_override: int | None = None):
    """Train one run with an optional TINYDET_ABLATION env flag.

    Reuses the repository trainer (seed, resume-from-last.pt, checkpointing)
    unchanged; only the env flag and the config text differ per run.

    A deterministic run_id is passed so that a retried or interrupted run lands in
    the same directory and the trainer's resume-from-last.pt path can engage.
    """
    import yaml

    from tinydet.utils.experiment import git_info

    os.chdir("/root")
    runs.reload()

    raw = yaml.safe_load(config_text)
    if epochs_override is not None:
        raw["train"]["epochs"] = int(epochs_override)
        config_text = yaml.safe_dump(raw, sort_keys=False)

    commit, dirty = git_info()
    run_id = Path(config).stem if epochs_override is None else f"{Path(config).stem}-pilot"
    # base_app._train caps at max_containers=4; this study needs one container per
    # ablation so all five train simultaneously on separate A10s.
    options: dict = {"gpu": "A10", "max_containers": len(ABLATIONS)}
    if raw["output"].get("wandb_project"):
        options["secrets"] = [wandb_secret]
    started = time.monotonic()
    run = base_app._train.with_options(**options).remote(
        config, commit, dirty, config_text=config_text,
        wandb_group="ablation-study-seed42", run_id=run_id, ablation=ablation,
    )
    elapsed = time.monotonic() - started
    return {"run": run, "wall_seconds": elapsed, "ablation": ablation}


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1,
               volumes={str(RUN_ROOT): runs}, timeout=600)
def _read_run_status(config_name: str):
    """Read metadata.json for a config and report its training status."""
    runs.reload()
    latest = _latest_run(config_name)
    if latest is None:
        return {"name": config_name, "status": "NOT_STARTED"}
    meta_path = latest / "metadata.json"
    if not meta_path.is_file():
        return {"name": config_name, "status": "RUNNING", "run": str(latest)}
    meta = json.loads(meta_path.read_text())
    return {
        "name": config_name,
        "run": str(latest),
        "status": meta.get("status"),
        "training_seconds": meta.get("training_seconds"),
        "peak_vram_gb": meta.get("peak_vram_gb"),
        "error": meta.get("error"),
        "seed": meta.get("seed"),
        "ablation": meta.get("ablation"),
        "ablation_flags": meta.get("ablation_flags"),
    }


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1,
               volumes={str(RUN_ROOT): runs}, timeout=600)
def _read_metadata(run_dir: str):
    """Read metadata.json for an explicit run directory (pilot dirs included)."""
    runs.reload()
    path = Path(run_dir) / "metadata.json"
    if not path.is_file():
        return {"run_dir": run_dir, "error": "metadata.json missing"}
    meta = json.loads(path.read_text())
    return {
        "run_dir": run_dir,
        "status": meta.get("status"),
        "ablation": meta.get("ablation"),
        "ablation_flags": meta.get("ablation_flags"),
        "seed": meta.get("seed"),
        "training_seconds": meta.get("training_seconds"),
    }


@app.local_entrypoint()
def show_metadata(run_dir: str):
    """Print metadata.json for an explicit run directory."""
    print(json.dumps(_read_metadata.remote(run_dir), indent=2), flush=True)


@app.local_entrypoint()
def verify_ablation(ablation: str = "no_high_band"):
    """Train 1 epoch with an ablation flag and prove the flag reached the model.

    Guards against the failure mode where the flag is set in the runner container
    but never reaches the training container, which would silently produce N
    identical runs. Checks the metadata written inside the training container.
    """
    config = f"configs/experiment/ablation/abl_gauss_{ablation}_seed{SEED}.yaml"
    text = Path(config).read_text()
    print(f"VERIFY: 1 epoch with ablation='{ablation}' seed={SEED}", flush=True)
    result = _train_ablation.remote(config, ablation, text, epochs_override=1)
    meta = _read_metadata.remote(result["run"])
    print(json.dumps(meta, indent=2), flush=True)
    recorded = meta.get("ablation")
    if recorded != ablation:
        print(f"VERIFY_FAILED: expected ablation={ablation!r}, metadata has {recorded!r}",
              flush=True)
        return
    print(f"VERIFY_OK: training container recorded ablation={recorded!r}", flush=True)


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1,
               volumes={str(RUN_ROOT): runs}, timeout=600)
def _delete_pilot_dirs():
    """Delete leftover *-pilot run directories so resume cannot stop early."""
    import shutil

    runs.reload()
    removed = []
    for root in RUN_ROOT.iterdir():
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and child.name.endswith("-pilot"):
                shutil.rmtree(child)
                removed.append(str(child))
    runs.commit()
    return {"removed": removed}


@app.local_entrypoint()
def cleanup_pilots():
    """Delete leftover pilot run directories and report what was removed."""
    print(json.dumps(_delete_pilot_dirs.remote(), indent=2), flush=True)


@app.local_entrypoint()
def pilot(epochs: int = 5):
    """Short GauSS pilot: measure steady-state s/epoch and VRAM, then the estimate.

    Uses >=3 epochs on purpose: Ultralytics only validates on the final epoch when
    val=False, so the final epoch carries a one-time validation cost. The middle
    epochs are the steady-state cost that actually drives a 150-epoch run.
    """
    config = f"configs/experiment/ablation/abl_gauss_no_high_band_seed{SEED}.yaml"
    text = Path(config).read_text()
    print("=" * 70, flush=True)
    print(f"PILOT: {epochs} epochs on A10 — measuring s/epoch and VRAM", flush=True)
    print("=" * 70, flush=True)
    result = _train_ablation.remote(config, "", text, epochs_override=epochs)
    status = _inspect_run.remote(Path(config).stem)
    wall = result["wall_seconds"]
    deltas = status.get("per_epoch_seconds") or []
    steady_epochs = deltas[2:-1] if len(deltas) > 3 else deltas[1:-1]
    steady = round(statistics.median(steady_epochs), 1) if steady_epochs else None
    print(json.dumps({
        "pilot_wall_seconds": round(wall),
        "training_seconds": round(status.get("training_seconds") or 0, 1),
        "per_epoch_seconds": deltas,
        "steady_epochs_used": steady_epochs,
        "steady_state_s_per_epoch": steady,
        "peak_vram_gb": status.get("peak_vram_gb"),
        "run_dir": status.get("run"),
    }, indent=2), flush=True)
    print("NOTE: delete the pilot run directory before the real run", flush=True)
    print("      (resume would otherwise stop at the pilot epoch count).", flush=True)


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1,
               volumes={str(RUN_ROOT): runs}, timeout=600)
def _inspect_run(config_name: str, delete: bool = False):
    """Read per-epoch wall time from training/results.csv and optionally delete the run.

    Used to separate one-time costs (startup, final validation, W&B sync) from the
    steady-state s/epoch that drives the 150-epoch budget.
    """
    import csv
    import shutil

    runs.reload()
    latest = _latest_run(config_name)
    if latest is None:
        return {"name": config_name, "status": "NOT_STARTED"}

    csv_path = latest / "training" / "results.csv"
    epochs = []
    if csv_path.is_file():
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row = {k.strip(): v for k, v in row.items() if k}
                try:
                    epochs.append({
                        "epoch": int(float(row["epoch"])),
                        "cumulative_s": float(row["time"]),
                    })
                except (KeyError, ValueError):
                    continue
    deltas = []
    previous = 0.0
    for item in epochs:
        deltas.append(round(item["cumulative_s"] - previous, 1))
        previous = item["cumulative_s"]

    meta_path = latest / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    payload = {
        "name": config_name,
        "run": str(latest),
        "status": meta.get("status"),
        "training_seconds": meta.get("training_seconds"),
        "peak_vram_gb": meta.get("peak_vram_gb"),
        "per_epoch_seconds": deltas,
        "steady_state_s_per_epoch": deltas[-1] if deltas else None,
    }
    if delete:
        shutil.rmtree(latest)
        runs.commit()
        payload["deleted"] = str(latest)
    return payload


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1,
               volumes={str(RUN_ROOT): runs}, timeout=600)
def _checkpoint_state():
    """Report which checkpoint files exist per run and how stale they are.

    Answers two operational questions: can an interrupted run resume, and is the
    newest checkpoint recent enough to prove the container is still writing.
    """
    from datetime import datetime

    runs.reload()
    now = datetime.now(UTC)
    out = []
    for item in PLAN:
        name = Path(item["config"]).stem
        latest = _latest_run(name)
        if latest is None:
            out.append({"name": name, "status": "NOT_STARTED"})
            continue
        weights = latest / "training" / "weights"
        files = []
        if weights.is_dir():
            for f in sorted(weights.iterdir()):
                stat = f.stat()
                age = (now - datetime.fromtimestamp(stat.st_mtime, UTC))
                files.append({
                    "file": f.name,
                    "size_mb": round(stat.st_size / 1e6, 1),
                    "age_minutes": round(age.total_seconds() / 60, 1),
                })
        out.append({
            "name": name,
            "run": str(latest),
            "checkpoints": files,
            "resumable": any(f["file"] == "last.pt" for f in files),
        })
    return out


@app.local_entrypoint()
def checkpoints():
    """Print checkpoint state for every run (resume capability + freshness)."""
    for payload in _checkpoint_state.remote():
        print(json.dumps(payload), flush=True)


@app.local_entrypoint()
def timings():
    """Print per-epoch wall time for every ablation run.

    Used to measure whether 5-way concurrency slows each run versus the
    single-run 34.0 s/epoch baseline.
    """
    for item in PLAN:
        name = Path(item["config"]).stem
        payload = _inspect_run.remote(name)
        deltas = payload.get("per_epoch_seconds") or []
        steady = None
        if len(deltas) > 3:
            steady = round(statistics.median(deltas[2:-1]), 1)
        print(json.dumps({
            "name": name,
            "status": payload.get("status"),
            "epochs_done": len(deltas),
            "per_epoch_seconds": deltas,
            "steady_state_s_per_epoch": steady,
            "training_seconds": payload.get("training_seconds"),
            "peak_vram_gb": payload.get("peak_vram_gb"),
        }), flush=True)


@app.local_entrypoint()
def pilot_report(delete: bool = False):
    """Print hard per-epoch timings from the pilot run and optionally delete it."""
    payload = _inspect_run.remote(f"abl_gauss_no_high_band_seed{SEED}", delete)
    print(json.dumps(payload, indent=2), flush=True)


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    max_containers=1,
    volumes={str(RUN_ROOT): runs},
    timeout=21600,  # 6 h ceiling: 5 runs in parallel, ~1.8 h each
)
def _orchestrate_parallel():
    """Train all five ablations in parallel, then report per-run outcomes.

    Kept remote on purpose: a local entrypoint loop dies with the local process.
    Each ablation gets its own A10 container; the loop only spawns and gathers.
    """
    handles = []
    for item in PLAN:
        name = Path(item["config"]).stem
        status = _read_run_status.remote(name)
        if status.get("status") == "TRAINED":
            print(f"[SKIP] {name} already TRAINED ({status.get('run')})", flush=True)
            continue
        print(f"[SPAWN] {name} (ablation={item['ablation']})", flush=True)
        text = Path(item["config"]).read_text()
        handle = _train_ablation.spawn(item["config"], item["ablation"], text)
        handles.append((name, handle))

    print(f"[PARALLEL] {len(handles)} runs in flight", flush=True)

    results = []
    for name, handle in handles:
        try:
            r = handle.get()
            st = _read_run_status.remote(name)
            results.append({**st, "wall_seconds": r["wall_seconds"]})
            print(f"[DONE] {name}: {st.get('status')} in {r['wall_seconds']:.0f}s",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - log and continue per plan
            print(f"[ERROR] {name}: {exc}", flush=True)
            results.append({"name": name, "status": "FAILED", "error": str(exc)[:500]})

    print("\n===== RUN SUMMARY =====", flush=True)
    for r in results:
        print(json.dumps(r), flush=True)
    return results


@app.local_entrypoint()
def run_all():
    """Launch the parallel plan remotely and return immediately.

    Use `modal run --detach ...::run_all` so the plan survives a local disconnect.
    """
    handle = _orchestrate_parallel.spawn()
    print(f"orchestrator spawned: {handle.object_id}", flush=True)
    print("poll with: modal run modal/ablation_runner.py::status", flush=True)


@app.local_entrypoint()
def run_all_wait():
    """Same plan but block locally and print the summary (foreground use only)."""
    results = _orchestrate_parallel.remote()
    print(json.dumps(results, indent=2), flush=True)


@app.local_entrypoint()
def status():
    """Print the status of the five ablation runs."""
    for item in PLAN:
        st = _read_run_status.remote(Path(item["config"]).stem)
        print(json.dumps(st), flush=True)


@app.function(image=image, cpu=1.0, memory=1024, max_containers=1,
               volumes={str(RUN_ROOT): runs}, timeout=600)
def _read_losses(config_name: str):
    """Read the first few epochs of loss columns from training/results.csv.

    Used as a live check that the five ablation containers really train different
    models: identical loss values across runs would mean the env flag never
    reached the model.
    """
    import csv

    runs.reload()
    latest = _latest_run(config_name)
    if latest is None:
        return {"name": config_name, "status": "NOT_STARTED"}
    csv_path = latest / "training" / "results.csv"
    if not csv_path.is_file():
        return {"name": config_name, "status": "NO_RESULTS_CSV"}
    rows = []
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({k.strip(): v.strip() for k, v in row.items() if k})
    keys = [k for k in rows[0] if k.endswith("loss")] if rows else []
    return {
        "name": config_name,
        "epochs": len(rows),
        "loss_keys": keys,
        "first": [{k: rows[0][k] for k in keys}] if rows else [],
        "last": [{k: rows[-1][k] for k in keys}] if rows else [],
    }


@app.local_entrypoint()
def compare_losses():
    """Print first/last-epoch losses for every ablation run.

    Decisive live check for ablation-flag propagation: five runs whose losses are
    bit-identical are five runs of the same model.
    """
    payloads = []
    for item in PLAN:
        payloads.append(_read_losses.remote(Path(item["config"]).stem))
    for payload in payloads:
        print(json.dumps(payload), flush=True)

    complete = [p for p in payloads if p.get("first")]
    if len(complete) < 2:
        print("[CHECK] not enough runs with results.csv yet", flush=True)
        return
    keys = complete[0]["loss_keys"]
    signature = {p["name"]: tuple(p["first"][0][k] for k in keys) for p in complete}
    unique = set(signature.values())
    print(f"[CHECK] {len(complete)} runs, {len(unique)} distinct epoch-1 loss vectors",
          flush=True)
    if len(unique) == 1:
        print("[CHECK_FAILED] all runs have identical epoch-1 losses: "
              "ablation flags did NOT reach the model", flush=True)
    else:
        print("[CHECK_OK] runs differ at epoch 1: ablation flags reached the model",
              flush=True)


@app.function(
    image=image,
    gpu="A10",
    min_containers=0,
    buffer_containers=0,
    scaledown_window=2,
    cpu=CPU_LIMIT,
    memory=MEMORY_LIMIT_MB,
    max_containers=len(ABLATIONS),
    volumes=base_app.mounts,
    timeout=10800,
)
def _eval_tiled(relative_checkpoint: str, name: str):
    """Same TOV tiled evaluation as the paper; parameters are frozen."""
    return base_app._run_tiled_checkpoint_evaluation.remote(relative_checkpoint, name)


@app.local_entrypoint()
def evaluate_all():
    """Run the frozen TOV tiled evaluation for every TRAINED ablation run.

    Writes reports/eval/ablation/{name}.json (eval payload) and
    reports/eval/ablation/{name}.train.json (run metadata) for
    scripts/aggregate_ablation.py to consume.
    """
    out_dir = Path("reports/eval/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    statuses = {}
    for item in PLAN:
        name = Path(item["config"]).stem
        st = _read_run_status.remote(name)
        statuses[name] = st
        if st.get("status") != "TRAINED":
            print(f"[SKIP-EVAL] {name}: status={st.get('status')}", flush=True)
            continue
        run_dir = st["run"]
        ckpt = f"{run_dir.replace(str(RUN_ROOT) + '/', '')}/training/weights/last.pt"
        print(f"[EVAL] {name} <- {ckpt}", flush=True)
        handle = _eval_tiled.spawn(ckpt, f"ablation/{name}")
        handles.append((name, handle))

    for name, handle in handles:
        res = handle.get()
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=2))
        (out_dir / f"{name}.train.json").write_text(
            json.dumps(statuses.get(name, {}), indent=2)
        )
        bench = res["summary"]["benchmark"]
        print(f"\n--- {name} ---", flush=True)
        for bin_name in ("all", "tiny1", "tiny2", "tiny3", "small"):
            b = bench.get(bin_name, {})
            print(f"AP50 {bin_name}: {b.get('AP@0.50')}", flush=True)
        print(f"MR50 tiny2: {bench.get('tiny2', {}).get('MR@0.50')}", flush=True)
        print(f"AP75 all:  {bench.get('all', {}).get('AP@0.75')}", flush=True)

    for item in PLAN:
        name = Path(item["config"]).stem
        if name not in {n for n, _ in handles}:
            (out_dir / f"{name}.train.json").write_text(
                json.dumps(statuses.get(name, {}), indent=2)
            )
    print(f"[OK] wrote eval payloads to {out_dir}/", flush=True)
