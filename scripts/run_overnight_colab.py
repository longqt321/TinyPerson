"""Unattended overnight Google Colab / GPU campaign runner with HF Hub resume & OOM self-healing.

Executes the full publication matrix in scientific priority order:
- Phase A (TinyPerson Headline & Controls): reuses existing ``lgt_tinyperson_150e/colab-seed42``
  checkpoint if present on HF Hub (running only missing post-hoc evals), then trains fair
  ``tp_yolo26n_base``, ``tp_yolo26n_nwd``, and ``tp_yolo11n_nms``.
- Phase B (VisDrone2019-DET 10-Class Generalization): ``vd_yolo26n_base``, ``vd_yolo26n_lgt``,
  ``vd_yolo26n_nwd``, ``vd_yolo11n_nms``.
- Phase C (Ablations, Cross-Generation YOLOv10n-P2, Scale-Up YOLO26s-P2, Multi-Seed 43/44).

Every job runs in an isolated child Python process so CUDA memory, W&B runs, and Ultralytics
hooks never leak across runs. If a GPU has smaller VRAM (e.g., L4 24GB instead of A100 40GB)
and raises CUDA OOM, the runner automatically halves ``--batch`` and retries cleanly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import torch

from tinydet.utils.runtime import (
    DEFAULT_HF_DATASET_REPO,
    default_drive_backup_root,
    default_hf_runs_repo,
    default_output_root,
    detect_host,
    ensure_hf_auth,
    stable_run_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ordered 3-Phase Publication Matrix: (phase, exp_yaml, seed, preferred_batch_40gb)
CAMPAIGN_JOBS: list[tuple[str, str, int, int]] = [
    # Phase A: TinyPerson Headline (seed 42) — ~2.0h on A100 (tp_yolo26n_lgt seed42 reuses colab-seed42!)
    ("A", "configs/experiment/paper/tp_yolo26n_lgt.yaml", 42, 32),
    ("A", "configs/experiment/paper/tp_yolo26n_base.yaml", 42, 32),
    ("A", "configs/experiment/paper/tp_yolo26n_nwd.yaml", 42, 32),
    ("A", "configs/experiment/paper/tp_yolo11n_nms.yaml", 42, 32),
    # Phase B: VisDrone2019-DET 10-Class Generalization — ~9.4h on A100
    ("B", "configs/experiment/paper/vd_yolo26n_base.yaml", 42, 16),
    ("B", "configs/experiment/paper/vd_yolo26n_lgt.yaml", 42, 16),
    ("B", "configs/experiment/paper/vd_yolo26n_nwd.yaml", 42, 16),
    ("B", "configs/experiment/paper/vd_yolo11n_nms.yaml", 42, 16),
    # Phase C1: TinyPerson Component-Wise Ablation Ladder — ~2.0h on A100
    ("C", "configs/experiment/paper/tp_abl_wtal.yaml", 42, 32),
    ("C", "configs/experiment/paper/tp_abl_nodistill.yaml", 42, 32),
    ("C", "configs/experiment/paper/tp_abl_nocross.yaml", 42, 32),
    # Phase C2: Cross-Generation (YOLOv10n-P2 DFL reg_max=16) & Scale-Up (YOLO26s-P2) — ~3.6h on A100
    ("C", "configs/experiment/paper/tp_yolov10n_base.yaml", 42, 16),
    ("C", "configs/experiment/paper/tp_yolov10n_lgt.yaml", 42, 16),
    ("C", "configs/experiment/paper/tp_yolo26s_base.yaml", 42, 16),
    ("C", "configs/experiment/paper/tp_yolo26s_lgt.yaml", 42, 16),
    # Phase C3: Multi-Seed Statistical Significance (Seeds 43 & 44 on Headline Pair) — ~2.7h on A100
    ("C", "configs/experiment/paper/tp_yolo26n_base.yaml", 43, 32),
    ("C", "configs/experiment/paper/tp_yolo26n_lgt.yaml", 43, 32),
    ("C", "configs/experiment/paper/tp_yolo26n_base.yaml", 44, 32),
    ("C", "configs/experiment/paper/tp_yolo26n_lgt.yaml", 44, 32),
]


def _seed_legacy_lgt_run_if_available(
    output_root: Path,
    run_id: str,
    hf_runs_repo: str | None,
    hf_token: str | None,
) -> None:
    """Migrate verified ``lgt_tinyperson_150e/colab-seed42`` into ``tp_yolo26n_lgt/<run_id>`` if absent."""
    target_dir = output_root / "tp_yolo26n_lgt" / run_id
    if (target_dir / "training" / "completed.flag").is_file():
        return
    if not hf_runs_repo:
        return
    with suppress(Exception):
        from huggingface_hub import snapshot_download

        legacy_rel = "lgt_tinyperson_150e/colab-seed42"
        tmp_dl = output_root / "_legacy_seed_check"
        snapshot_download(
            repo_id=hf_runs_repo,
            repo_type="model",
            token=hf_token,
            allow_patterns=[f"{legacy_rel}/**"],
            ignore_patterns=["**/last.pt"],
            local_dir=str(tmp_dl),
        )
        src = tmp_dl / legacy_rel
        if (src / "training" / "completed.flag").is_file() and (
            src / "training" / "weights" / "best.pt"
        ).is_file():
            print(
                f"[campaign] Reusing completed 150-epoch run from hf://{hf_runs_repo}/{legacy_rel} "
                f"-> {target_dir} (saves ~42 min A100 compute!)",
                flush=True,
            )
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, target_dir, dirs_exist_ok=True)
        shutil.rmtree(tmp_dl, ignore_errors=True)


def _initial_batch_for_vram(preferred_batch: int, vram_gb: float) -> int:
    if vram_gb >= 35.0:
        return preferred_batch
    if vram_gb >= 20.0:
        return min(preferred_batch, 16)
    return min(preferred_batch, 8)


def _sync_campaign_manifest(
    manifest_path: Path,
    payload: dict[str, Any],
    hf_runs_repo: str | None,
    hf_token: str | None,
    drive_root: Path | None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    if drive_root is not None:
        with suppress(Exception):
            drive_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(manifest_path, drive_root / "campaign_status.json")
    if hf_runs_repo:
        with suppress(Exception):
            from huggingface_hub import HfApi

            HfApi(token=hf_token).upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo="campaign_status.json",
                repo_id=hf_runs_repo,
                repo_type="model",
                commit_message="Update overnight campaign status",
            )


def _run_job_with_oom_fallback(
    config_rel: str,
    exp_name: str,
    seed: int,
    run_id: str,
    start_batch: int,
    hf_dataset: str,
    hf_runs_repo: str | None,
    wandb_group: str,
    output_root: Path,
) -> tuple[bool, int, str]:
    batch = start_batch
    while batch >= 4:
        cmd = [
            sys.executable,
            "-u",
            str(REPO_ROOT / "train.py"),
            "--config",
            str(REPO_ROOT / config_rel),
            "--name",
            exp_name,
            "--seed",
            str(seed),
            "--run-id",
            run_id,
            "--batch",
            str(batch),
            "--hf-dataset",
            hf_dataset,
            "--output-root",
            str(output_root),
            "--wandb-group",
            wandb_group,
        ]
        if hf_runs_repo:
            cmd.extend(["--hf-runs-repo", hf_runs_repo])

        print(f"\n[campaign] >>> Executing: {' '.join(cmd)}", flush=True)
        tail_lines: list[str] = []
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail_lines.append(line)
            if len(tail_lines) > 80:
                tail_lines.pop(0)
        rc = proc.wait()
        if rc == 0:
            return True, batch, "completed"

        joined_tail = "".join(tail_lines).lower()
        is_oom = "out of memory" in joined_tail or "cuda error: out of memory" in joined_tail
        if is_oom and batch > 4:
            next_batch = max(4, batch // 2)
            print(
                f"[campaign] CUDA OOM detected at batch={batch}; auto-scaling batch -> {next_batch} and retrying...",
                flush=True,
            )
            batch = next_batch
            time.sleep(3.0)
            continue
        return False, batch, f"exit_code={rc}"
    return False, batch, "exhausted_oom_retries"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases",
        default="A,B,C",
        help="Comma-separated phases to run (default: A,B,C runs the entire paper suite).",
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET_REPO)
    parser.add_argument("--hf-runs-repo", default=None)
    parser.add_argument("--wandb-group", default="lgt-paper-v1")
    args = parser.parse_args()

    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    host = detect_host()
    output_root = default_output_root(host, REPO_ROOT)
    drive_root = default_drive_backup_root(host)
    hf_token, hf_user = ensure_hf_auth()
    hf_runs_repo = args.hf_runs_repo or default_hf_runs_repo(hf_user)

    if not torch.cuda.is_available():
        print("[campaign] ERROR: No CUDA GPU available!", file=sys.stderr)
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    wanted_phases = {p.strip().upper() for p in args.phases.split(",") if p.strip()}
    selected_jobs = [job for job in CAMPAIGN_JOBS if job[0] in wanted_phases]

    print("=" * 78, flush=True)
    print("[campaign] UNATTENDED OVERNIGHT PAPER CAMPAIGN STARTED", flush=True)
    print(f"[campaign] Host: {host} | GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)", flush=True)
    print(f"[campaign] HF Dataset Repo : {args.hf_dataset}", flush=True)
    print(f"[campaign] HF Runs Backup  : {hf_runs_repo or 'DISABLED (WARNING!)'}", flush=True)
    print(f"[campaign] Drive Backup    : {drive_root or 'not mounted'}", flush=True)
    print(f"[campaign] Selected Phases : {sorted(wanted_phases)} ({len(selected_jobs)} jobs)", flush=True)
    print("=" * 78, flush=True)

    # Seed tp_yolo26n_lgt seed 42 from already-completed colab-seed42 if present
    _seed_legacy_lgt_run_if_available(
        output_root, stable_run_id(42, host), hf_runs_repo, hf_token
    )

    status_payload: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "host": host,
        "gpu": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "hf_runs_repo": hf_runs_repo,
        "jobs": [],
    }
    manifest_path = output_root / "campaign_status.json"

    failures = 0
    for idx, (phase, config_rel, seed, pref_batch) in enumerate(selected_jobs, start=1):
        exp_name = Path(config_rel).stem
        run_id = stable_run_id(seed, host)
        run_dir = output_root / exp_name / run_id
        start_batch = _initial_batch_for_vram(pref_batch, vram_gb)

        # Check remote HF Hub first so restarting a disconnected Colab skips finished runs immediately
        if hf_runs_repo and not (run_dir / "metrics" / "eval_complete.flag").is_file():
            from tinydet.utils.runtime import restore_run_from_hf

            restore_run_from_hf(run_dir, hf_runs_repo, hf_token)

        if (run_dir / "training" / "completed.flag").is_file() and (
            run_dir / "metrics" / "eval_complete.flag"
        ).is_file():
            print(
                f"\n[campaign] [{idx}/{len(selected_jobs)}] SKIP {exp_name}/{run_id} "
                "(training + full paper evaluation already completed!)",
                flush=True,
            )
            status_payload["jobs"].append({
                "index": idx,
                "phase": phase,
                "experiment": exp_name,
                "seed": seed,
                "run_id": run_id,
                "status": "skipped_already_complete",
            })
            _sync_campaign_manifest(manifest_path, status_payload, hf_runs_repo, hf_token, drive_root)
            continue

        t0 = time.monotonic()
        print(
            f"\n[campaign] [{idx}/{len(selected_jobs)}] START Phase {phase}: "
            f"{exp_name} (seed={seed}, batch={start_batch})",
            flush=True,
        )
        ok, used_batch, note = _run_job_with_oom_fallback(
            config_rel=config_rel,
            exp_name=exp_name,
            seed=seed,
            run_id=run_id,
            start_batch=start_batch,
            hf_dataset=args.hf_dataset,
            hf_runs_repo=hf_runs_repo,
            wandb_group=args.wandb_group,
            output_root=output_root,
        )
        elapsed_min = round((time.monotonic() - t0) / 60.0, 2)
        if not ok:
            failures += 1
            print(
                f"[campaign] [{idx}/{len(selected_jobs)}] FAILED {exp_name} after {elapsed_min} min ({note}); "
                "continuing remaining overnight jobs!",
                flush=True,
            )
        else:
            print(
                f"[campaign] [{idx}/{len(selected_jobs)}] DONE {exp_name} in {elapsed_min} min!",
                flush=True,
            )

        status_payload["jobs"].append({
            "index": idx,
            "phase": phase,
            "experiment": exp_name,
            "seed": seed,
            "run_id": run_id,
            "batch": used_batch,
            "elapsed_minutes": elapsed_min,
            "status": "completed" if ok else "failed",
            "note": note,
            "finished_at_utc": datetime.now(UTC).isoformat(),
        })
        _sync_campaign_manifest(manifest_path, status_payload, hf_runs_repo, hf_token, drive_root)

    status_payload["completed_at_utc"] = datetime.now(UTC).isoformat()
    status_payload["failures"] = failures
    _sync_campaign_manifest(manifest_path, status_payload, hf_runs_repo, hf_token, drive_root)

    print("=" * 78, flush=True)
    print(
        f"[campaign] OVERNIGHT CAMPAIGN FINISHED: {len(selected_jobs) - failures}/{len(selected_jobs)} succeeded.",
        flush=True,
    )
    print("=" * 78, flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
