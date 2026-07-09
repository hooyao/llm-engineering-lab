#!/usr/bin/env bash
# Run the unified-memory copy-bandwidth benchmark inside the NVIDIA PyTorch
# 26.04 container. No pip installs — uses only torch.
#
# Companion to run-26.04.sh (BF16 GEMM). Measures CPU<->CUDA copy paths and the
# single-tensor capacity check on the GB10 unified pool.
#
#   bash experiments/bench/run-copy-bw-26.04.sh
#   bash experiments/bench/run-copy-bw-26.04.sh --mb 1024 --iters 50 --cap-gb 0
#
# --ulimit memlock=-1 matters here: pinned (page-locked) host allocations in the
# benchmark need it.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

echo "=================================================================="
echo "GB10 copy-bandwidth benchmark — nvcr.io/nvidia/pytorch:26.04-py3"
echo "  repo root : $REPO_ROOT"
echo "  passthru  : $*"
echo "=================================================================="

exec docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:26.04-py3 \
    python experiments/bench/copy_bandwidth.py "$@"
