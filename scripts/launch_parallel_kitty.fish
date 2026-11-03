#!/usr/bin/env fish
# Launch parallel experiments in dedicated Kitty terminals running fish shell.
# Usage:
#   ./scripts/launch_parallel_kitty.fish [config_yaml] [--mode auto|tab|window] [--smoke] [--two-stage]

set -l script_dir (status dirname)
python3 "$script_dir/launch_parallel_kitty.py" $argv

