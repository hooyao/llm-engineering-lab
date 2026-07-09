#!/usr/bin/env bash
# Compile and run uvm_probe.cu inside the CUDA devel container (nvcc), which is
# separate from the PyTorch container because the runtime PyTorch image does not
# ship nvcc. -arch=sm_121 targets the GB10 (Blackwell, compute capability 12.1).
#
#   bash experiments/bench/run-uvm-probe.sh
#
# The compiled binary is written to a tmpfs inside the container and discarded
# on exit; nothing lands in the repo. Uses the base CUDA 13.0.1 devel image
# already pulled on this box (see notes/progress.md container table).

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
IMAGE="${CUDA_DEVEL_IMAGE:-nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04}"

echo "=================================================================="
echo "GB10 UVM probe — compile + run in $IMAGE"
echo "  repo root : $REPO_ROOT"
echo "=================================================================="

exec docker run --rm --gpus all \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    "$IMAGE" \
    bash -lc "nvcc -O3 -arch=sm_121 experiments/bench/uvm_probe.cu -o /tmp/uvm_probe && /tmp/uvm_probe"
