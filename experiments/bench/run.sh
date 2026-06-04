#!/usr/bin/env bash
# Launch the BF16 GEMM benchmark inside the NVIDIA PyTorch container.
# No pip installs — uses only torch, which the container ships natively.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# pass-through args to bf16_peak.py:
#   bash experiments/bench/run.sh                          # quick default
#   bash experiments/bench/run.sh --duration 30            # sustained 30s/size
#   bash experiments/bench/run.sh --sizes 4096,8192,16384  # smaller sweep

echo "=================================================================="
echo "GB10 BF16 GEMM benchmark"
echo "  repo root : $REPO_ROOT"
echo "  passthru  : $*"
echo "=================================================================="

exec docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    python experiments/bench/bf16_peak.py "$@"
