#!/usr/bin/env bash
# A3 (A2 payoff) — generate from base vs A2-tuned 1B side by side, inside the
# 26.04 container. Mounts ~/runs so it can read the A2 checkpoint and write the
# results markdown back to the host.
#
# Usage:
#   bash experiments/a03-eval-1b/run.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
LOG="$HERE/eval.log"

docker run --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$HOME/models:/models:ro" \
    -v "$HOME/runs:/runs" \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:26.04-py3 \
    bash -c "
        set -e
        pip install -q --root-user-action=ignore transformers accelerate
        python experiments/a03-eval-1b/compare.py
    " 2>&1 | tee "$LOG"

echo
echo "Side-by-side also saved to ~/runs/a03-eval/results.md"
