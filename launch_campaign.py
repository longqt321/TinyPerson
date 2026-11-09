import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve() / "modal"))
import app as modal_app_module  # modal/ is not a package; path added above

app = modal_app_module.app

# Default campaign: the 150-epoch TinyPerson round (ADR-019). Override on the
# command line with `--configs a.yaml,b.yaml`; nothing is hardcoded downstream.
DEFAULT_CONFIGS = [
    "configs/experiment/baseline_tinyperson_wacv_150e.yaml",
    "configs/experiment/gauss_v3_tinyperson_wacv_150e.yaml",
]


def _parse_configs(configs: str) -> list[str]:
    """Split a comma-separated config list, falling back to the default campaign."""
    selected = [item.strip() for item in configs.split(",") if item.strip()]
    return selected or list(DEFAULT_CONFIGS)


@app.local_entrypoint()
def main(configs: str = "", detach: bool = True, wait: bool = False):
    """Launch a training campaign detached by default.

    The whole chain (audit -> parallel training -> finalize -> tiled + full
    eval) runs INSIDE Modal via `_orchestrate_campaign`, so a dropped local
    connection cannot kill training. Checkpoints, metrics and logs are
    committed to the runs Volume every epoch by `_train`.

    IMPORTANT: run with the `-d` flag so the app survives the local process
    exiting:

        modal run -d launch_campaign.py::main
        modal run -d launch_campaign.py::main --configs a.yaml,b.yaml

    Then monitor with:
        modal app logs <app-id>
    and fetch results with:
        modal run launch_campaign.py::fetch
    """
    from tinydet.utils.experiment import git_info

    selected = _parse_configs(configs)
    print(f"Campaign configs: {selected}", flush=True)
    commit, dirty = git_info()
    if dirty:
        print("[WARN] Working tree is dirty; commit/dirty will be recorded in run metadata.", flush=True)

    if detach and not wait:
        handle = modal_app_module._orchestrate_campaign.spawn(selected, commit, dirty)
        print("=" * 70, flush=True)
        print("DETACHED LAUNCH — the campaign now runs inside Modal.", flush=True)
        print(f"  Call ID : {handle.object_id}", flush=True)
        print("  Monitor : modal app logs   (app id shown above / `modal app list`)", flush=True)
        print("  Fetch   : modal run launch_campaign.py::fetch", flush=True)
        print("  NOTE    : this run used `modal run -d`; without -d the app stops", flush=True)
        print("            when this local process exits.", flush=True)
        print("=" * 70, flush=True)
        return

    # Attached mode: still executes remotely; a local drop only loses the
    # final printout, never the training itself.
    summary = modal_app_module._orchestrate_campaign.remote(selected, commit, dirty)
    print(json.dumps(summary, indent=2), flush=True)


@app.local_entrypoint()
def fetch():
    """Copy eval summaries + orchestration summary from the runs Volume to reports/.

    Uses the Modal Volume API (listdir/read_file): `/mnt/runs` only exists
    inside containers, never on the local machine.
    """
    runs_volume = modal_app_module.runs
    runs_volume.reload()

    def pull(volume_path: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            runs_volume.read_file_into_fileobj(volume_path, file)
        print(f"[SAVED] {target}", flush=True)

    copied = 0
    try:
        eval_entries = runs_volume.listdir("/eval")
    except Exception:  # noqa: BLE001 - /eval does not exist before the first eval
        eval_entries = []
    for entry in eval_entries:
        if entry.type.name != "DIRECTORY":
            continue
        children = {child.name for child in runs_volume.listdir(entry.path)}
        if "summary.json" in children:
            pull(f"{entry.path}/summary.json", Path("reports/eval") / f"{entry.name}.json")
            copied += 1
    print(f"Copied {copied} eval summary file(s) to reports/eval/", flush=True)

    try:
        orch_entries = [f for f in runs_volume.listdir("/orchestration") if f.name.startswith("summary-")]
    except Exception:  # noqa: BLE001 - no orchestration summary yet
        orch_entries = []
    for entry in sorted(orch_entries, key=lambda f: f.name):
        pull(entry.path, Path("reports/orchestration") / entry.name)


if __name__ == "__main__":
    main()
