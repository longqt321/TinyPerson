#!/usr/bin/env bash
# One-command bootstrap & training runner for Google Colab, Kaggle, and any CUDA host.
#
# Usage (after `git clone`):
#   bash scripts/run_gpu.sh
#   bash scripts/run_gpu.sh --config configs/experiment/gauss_v3_tinyperson_wacv_150e.yaml
#   bash scripts/run_gpu.sh --config configs/experiment/baseline_tinyperson_wacv_150e.yaml --wandb-group exp1
#   bash scripts/run_gpu.sh --dry-run
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Stream Python/Ultralytics logs directly to Colab/Kaggle notebook cells without buffering
export PYTHONUNBUFFERED=1

# 1. Ensure uv is installed and on PATH (standard Astral installer)
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 2. Detect --dry-run to allow CPU-only validation without CUDA
IS_DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    IS_DRY_RUN=1
    break
  fi
done

if [[ "$IS_DRY_RUN" -eq 1 ]]; then
  EXTRA="cpu"
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[error] nvidia-smi not found. Enable a GPU runtime in Colab/Kaggle or pass --dry-run." >&2
    exit 1
  fi
  echo "[setup] GPU detected:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  EXTRA="gpu"
fi

# 3. Provision pinned Python 3.14 + PyTorch wheel from uv.lock (isolated from Colab/Kaggle system Python)
echo "[setup] Syncing environment with uv (--extra ${EXTRA} --frozen)..."
uv sync --extra "$EXTRA" --frozen

# 4. Run full preflight -> train -> finalize pipeline via train.py
echo "[run] Launching train.py..."
exec uv run --extra "$EXTRA" python train.py "$@"
