#!/usr/bin/env python3
"""Launch separate Kitty terminals (tabs or windows) for parallel Modal experiments with fish shell."""

import argparse
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

# Add src to sys.path so tinydet utilities can be imported
workspace_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(workspace_root / "src"))

from tinydet.utils.comparison import load_comparison


def launch_kitty_terminals(
    config: str,
    mode: str = "auto",
    smoke: bool = False,
    two_stage: bool = False,
) -> None:
    kitty_bin = shutil.which("kitty")
    fish_bin = shutil.which("fish")

    if not kitty_bin:
        raise RuntimeError("kitty executable not found in PATH")
    if not fish_bin:
        raise RuntimeError("fish executable not found in PATH")

    config_path = Path(config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config}")

    experiments = load_comparison(config_path)
    to_train = [exp for exp in experiments if "run" not in exp]

    if not to_train:
        print("No models require training in this comparison config.")
        return

    active_venv = os.environ.get("VIRTUAL_ENV", "")
    session_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    wandb_group = f"compare-{uuid.uuid4().hex[:8]}"

    print("=" * 70)
    print(f"Launching {len(to_train)} Kitty terminals running fish for parallel experiments:")
    for exp in to_train:
        print(f"  - Model: {exp['name']} (GPU: {exp['hardware'].get('gpu', 'A10')})")
    print(f"Shared W&B Group: {wandb_group}")
    print(f"Shared Session TS: {session_ts}")
    if active_venv:
        print(f"Active Virtualenv: {active_venv}")
    print("=" * 70)

    tmp_dir = Path("/tmp/tinydet_kitty")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    flags = []
    if smoke:
        flags.append("--smoke")
    if two_stage:
        flags.append("--two-stage")
    flags_str = " ".join(flags)

    launched = []
    env = os.environ.copy()

    local_modal = workspace_root / ".venv" / "bin" / "modal"
    modal_bin = str(local_modal.resolve()) if local_modal.is_file() else (shutil.which("modal") or "modal")

    for exp in to_train:
        model_name = exp["name"]
        script_file = tmp_dir / f"run_{model_name}.fish"

        script_content = f"""#!/usr/bin/env fish
# Auto-generated runner for TinyDet: {model_name}
echo -ne "\\033]0;[TinyDet] {model_name}\\007"
echo "======================================================================"
echo "  [TinyDet Experiment] {model_name}"
echo "  Config: {config_path.resolve()}"
echo "  W&B Group: {wandb_group}"
echo "  Session TS: {session_ts}"
echo "======================================================================"
echo ""

cd "{workspace_root}"

# Activate Python virtual environment so modal and dependencies (yaml) are available
if test -n "{active_venv}" -a -f "{active_venv}/bin/activate.fish"
    source "{active_venv}/bin/activate.fish"
else if test -f "{workspace_root}/.venv/bin/activate.fish"
    source "{workspace_root}/.venv/bin/activate.fish"
end

"{modal_bin}" run modal/app.py::train_comparison_model \\
    --config "{config_path.resolve()}" \\
    --model-name "{model_name}" \\
    --wandb-group "{wandb_group}" \\
    --session-ts "{session_ts}" \\
    {flags_str}

set -l exit_code $status
echo ""
if test $exit_code -eq 0
    echo "======================================================================"
    echo "  [SUCCESS] Finished {model_name} successfully!"
    echo "======================================================================"
else
    echo "======================================================================"
    echo "  [ERROR] {model_name} exited with code $exit_code"
    echo "======================================================================"
end
echo ""
echo "Terminal remains active in fish shell. Type 'exit' to close."
exec fish
"""
        script_file.write_text(script_content)
        script_file.chmod(0o755)

        title = f"[TinyDet] {model_name}"
        success = False

        # Attempt to launch as Kitty tab if requested or in auto mode
        if mode in ("auto", "tab"):
            try:
                res = subprocess.run(
                    [
                        kitty_bin,
                        "@",
                        "launch",
                        "--type=tab",
                        f"--tab-title={title}",
                        f"--cwd={workspace_root}",
                        str(script_file),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                    env=env,
                )
                if res.returncode == 0:
                    success = True
                    launched.append((model_name, "tab"))
                    print(f"✓ Opened Kitty TAB for {model_name}")
            except (subprocess.SubprocessError, OSError):
                success = False

        # Fallback or explicit window mode: open separate OS window
        if not success:
            proc = subprocess.Popen(
                [
                    kitty_bin,
                    "--title",
                    title,
                    f"--directory={workspace_root}",
                    str(script_file),
                ],
                start_new_session=True,
                env=env,
            )
            launched.append((model_name, f"window (PID {proc.pid})"))
            print(f"✓ Opened Kitty WINDOW for {model_name} (PID: {proc.pid})")

    print("\nAll terminals spawned successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Launch individual Kitty terminals (running fish) for parallel TinyDet experiments."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/experiment/qac_v3_tinyperson.yaml",
        help="Path to comparison experiment config YAML",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "tab", "window"],
        default="auto",
        help="Open as Kitty tabs or separate OS windows (default: auto)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode (caps epochs to 5 for quick check)",
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Chain two-stage training (50ep + 50ep)",
    )

    args = parser.parse_args()
    launch_kitty_terminals(
        config=args.config,
        mode=args.mode,
        smoke=args.smoke,
        two_stage=args.two_stage,
    )


if __name__ == "__main__":
    main()
