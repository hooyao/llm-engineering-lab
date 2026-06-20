#!/usr/bin/env bash
# A2 — full-parameter SFT on Llama-3.2-1B, inside the NVIDIA PyTorch 26.04
# container. Mirrors experiments/smoke-test/run-26.04.sh but:
#   - runs experiments/a02-sft-1b/train.py (no LoRA, full-parameter)
#   - mounts ~/runs -> /runs so the trained model persists on the host
#     (A2 deliverable: save to ~/runs/a02-sft-1b/)
#
# Usage:
#   bash experiments/a02-sft-1b/run.sh                 # default: 500 ex, 1 epoch
#   bash experiments/a02-sft-1b/run.sh --seq-len 8192  # deliberately OOM (read trace)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TRAIN_LOG="$HERE/train.log"

mkdir -p "$HOME/runs"

echo "=================================================================="
echo "A2 FULL-PARAM SFT — nvcr.io/nvidia/pytorch:26.04-py3"
echo "  repo root : $REPO_ROOT"
echo "  runs dir  : $HOME/runs (-> /runs in container)"
echo "  train log : $TRAIN_LOG"
echo "  extra args: ${*:-<none>}"
echo "=================================================================="

nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw \
           --format=csv 2>/dev/null || true

docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HOME/models:/models:ro" \
    -v "$HOME/runs:/runs" \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:26.04-py3 \
    bash -c "
        set -e
        echo '[setup] installing transformers/datasets/accelerate (no pin)...'
        pip install -q --root-user-action=ignore transformers datasets accelerate
        echo '[setup] done.'
        python experiments/a02-sft-1b/train.py $*
    " 2>&1 | tee "$TRAIN_LOG"

echo
echo "=================================================================="
echo "DONE. Trained model (if it completed) -> $HOME/runs/a02-sft-1b/"
echo "=================================================================="
