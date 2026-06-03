#!/usr/bin/env bash
# Launch the NVIDIA PyTorch container with the flags every real workload needs.
#
# Usage:
#   bash tools/launch_pytorch.sh                       # interactive shell
#   bash tools/launch_pytorch.sh python train.py ...   # run a command and exit
#
# Mounts:
#   ~/.cache/huggingface -> /root/.cache/huggingface   (HF cache, persistent)
#   ~/models             -> /models  (read-only)       (pre-downloaded models)
#   $PWD                 -> /workspace                 (your code; cwd inside container)
#
# Flags explained:
#   --gpus all             expose the GB10 to the container
#   --ipc=host             share /dev/shm with host (DataLoader num_workers>0 needs this)
#   --ulimit memlock=-1    unlimited pinned memory (CUDA async copies depend on it)
#   --ulimit stack=64M     larger stack for big CUDA kernel compiles
#   --rm                   delete container on exit (state lives in mounted volumes)
#   --network=host         simpler networking; useful when running Jupyter / TensorBoard on a port
#
# Override the image with PYTORCH_IMAGE env var if you upgrade tags later.

set -euo pipefail

IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

mkdir -p "$HF_CACHE"

# Optional: bind-mount $HOME/models only if it exists, so a fresh box doesn't error.
mount_models=()
if [[ -d "$MODELS_DIR" ]]; then
    mount_models=(-v "$MODELS_DIR:/models:ro")
fi

# If we got args, run them non-interactively. Otherwise drop into a shell.
if [[ $# -gt 0 ]]; then
    interactive_flags=(--rm)
    cmd=("$@")
else
    interactive_flags=(-it --rm)
    cmd=()
fi

exec docker run "${interactive_flags[@]}" \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --network=host \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    "${mount_models[@]}" \
    -v "$PWD:/workspace" \
    -w /workspace \
    "$IMAGE" "${cmd[@]}"
