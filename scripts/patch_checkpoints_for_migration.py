"""Patch stale Modal volume paths inside backed-up checkpoints.

Why this is needed
------------------
Training checkpoints store the full `train_args` namespace, including
`save_dir`, which on Modal resolves to an account-specific volume path:

    /__modal/volumes/vo-1mSI8DTRRmOJaOvJIUm2al/<run>/<run>/training

`ultralytics.cfg.get_save_dir` returns `args.save_dir` verbatim when it is set, and
`BaseTrainer.check_resume` replaces `self.args` wholesale with the checkpoint's
stored args. So resuming on a *different* Modal account would try to write weights
into the old account's volume id, which does not exist there. The run would either
fail or write into container-local storage and lose the weights.

Fix
---
Clear `save_dir` in each checkpoint's `train_args`. With `save_dir` unset,
`get_save_dir` falls back to `project` / `name`, which the trainer passes
explicitly on every run, so the resumed run writes to the correct new path.

`data` is left alone: `check_resume` re-applies the caller's `data` argument when
one is passed, and `_train` always passes a freshly resolved dataset YAML.

Usage
-----
    # dry run, report what would change
    uv run --extra cpu --group dev python scripts/patch_checkpoints_for_migration.py \
        backups/ablation_seed42 --dry-run

    # apply in place
    uv run --extra cpu --group dev python scripts/patch_checkpoints_for_migration.py \
        backups/ablation_seed42
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

STALE_PREFIXES = ("/__modal/volumes/", "/mnt/runs/", "/mnt/data/")


def _args_to_dict(args) -> dict:
    if args is None:
        return {}
    return args if isinstance(args, dict) else dict(vars(args))


def _patch_file(path: Path, dry_run: bool) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = _args_to_dict(ckpt.get("train_args"))
    save_dir = args.get("save_dir")
    project = args.get("project")
    report = {
        "file": str(path),
        "epoch": ckpt.get("epoch"),
        "old_save_dir": save_dir,
        "project": project,
        "name": args.get("name"),
        "patched": False,
    }
    if not save_dir:
        report["action"] = "no save_dir; nothing to do"
        return report
    if not str(save_dir).startswith(STALE_PREFIXES):
        report["action"] = "save_dir is not a Modal volume path; left alone"
        return report

    report["action"] = "clear save_dir so project/name resolve at run time"
    if dry_run:
        return report

    args["save_dir"] = None
    if isinstance(ckpt.get("train_args"), dict):
        ckpt["train_args"] = args
    else:
        ckpt["train_args"] = args
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.is_file():
        shutil.copy2(path, backup)
    torch.save(ckpt, path)
    report["patched"] = True
    report["backup"] = str(backup)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing backed-up runs")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()

    targets = sorted(args.root.rglob("last.pt")) + sorted(args.root.rglob("best.pt"))
    if not targets:
        raise SystemExit(f"No last.pt/best.pt found under {args.root}")

    print(f"found {len(targets)} checkpoint(s) under {args.root}")
    patched = 0
    for path in targets:
        report = _patch_file(path, args.dry_run)
        print(
            f"  {path.relative_to(args.root)}: epoch={report['epoch']} "
            f"save_dir={report['old_save_dir']!r} -> {report['action']}"
        )
        patched += int(report["patched"])

    verb = "would patch" if args.dry_run else "patched"
    print(f"\n{verb} {patched if not args.dry_run else len(targets)} file(s)")
    if args.dry_run:
        print("re-run without --dry-run to apply")


if __name__ == "__main__":
    main()
