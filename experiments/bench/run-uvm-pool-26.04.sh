#!/usr/bin/env bash
# Run the UVM allocator backport self-check inside the NVIDIA PyTorch 26.04
# container. No pip installs, no nightly wheel — uvm_pool.py backports the
# upstream torch.cuda._use_uvm() using only what the container already ships
# (MemPool, use_mem_pool, torch._C._cuda_customAllocator, cuda-python).
#
#   bash experiments/bench/run-uvm-pool-26.04.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

echo "=================================================================="
echo "UVM allocator backport self-check — nvcr.io/nvidia/pytorch:26.04-py3"
echo "  repo root : $REPO_ROOT"
echo "=================================================================="

exec docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:26.04-py3 \
    python experiments/bench/uvm_pool.py
