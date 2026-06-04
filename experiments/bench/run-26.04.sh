#!/usr/bin/env bash
# Launch the BF16 GEMM benchmark inside the NVIDIA PyTorch 26.04 container.
# No pip installs needed — uses only torch, which the container ships natively.
#
# Difference vs run.sh (which targets 25.11-py3):
#   - Image tag is 26.04-py3 (CUDA 13.2, PyTorch 2.12.0a0)
#   - No script change required; bench/bf16_peak.py only uses torch.matmul.
#
# Verified 2026-06-05: both containers produce same TFLOPS within noise margin.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

echo "=================================================================="
echo "GB10 BF16 GEMM benchmark — nvcr.io/nvidia/pytorch:26.04-py3"
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
    python experiments/bench/bf16_peak.py "$@"
