#!/usr/bin/env bash
# A4 — gradient accumulation sweep, inside NVIDIA PyTorch 26.04.
#
# Runs the SAME effective batch (16) three ways and collects the payoff table:
#   micro=1 accum=16  (least memory, slowest)
#   micro=4 accum=4   (more memory, faster)
#   micro=8 accum=2   (most memory, fastest)
# Expectation: final_loss ~identical across all three (algorithm sees eff batch 16),
# but peak_mem and step_time differ. That contrast IS the A4 deliverable.
#
# Each config runs as its OWN python process so peak-memory is measured cleanly.
#
# Usage:
#   bash experiments/a04-grad-accum/run.sh                 # 3B, 30 opt-steps each
#   bash experiments/a04-grad-accum/run.sh --opt-steps 50

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TRAIN_LOG="$HERE/train.log"
RESULT_JSON="/runs/a04-grad-accum/results.jsonl"

mkdir -p "$HOME/runs/a04-grad-accum"
rm -f "$HOME/runs/a04-grad-accum/results.jsonl"   # fresh sweep

echo "=================================================================="
echo "A4 GRAD ACCUMULATION SWEEP — nvcr.io/nvidia/pytorch:26.04-py3"
echo "  repo root : $REPO_ROOT"
echo "  results   : $HOME/runs/a04-grad-accum/results.jsonl"
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
        for cfg in '1 16' '4 4' '8 2'; do
            set -- \$cfg
            echo
            echo '##################################################################'
            echo \"## CONFIG micro=\$1 accum=\$2\"
            echo '##################################################################'
            python experiments/a04-grad-accum/train.py \
                --micro-batch \$1 --accum-steps \$2 \
                --result-json $RESULT_JSON $* || echo \"[warn] config micro=\$1 accum=\$2 failed (OOM?)\"
        done

        echo
        echo '=================================================================='
        echo 'A4 PAYOFF TABLE  (same effective batch 16, three splits)'
        echo '=================================================================='
        python - <<'PYEOF'
import json
rows = [json.loads(l) for l in open('$RESULT_JSON')]
print(f\"{'micro':>5} {'accum':>5} {'eff':>4} | {'peak_mem':>9} {'step_time':>10} {'final_loss':>11}\")
print('-'*55)
for r in rows:
    print(f\"{r['micro']:>5} {r['accum']:>5} {r['eff']:>4} | \"
          f\"{r['peak_gb']:>7.2f}GB {r['step_ms']:>8.1f}ms {r['final_loss']:>11.4f}\")
print('-'*55)
if rows:
    losses = [r['final_loss'] for r in rows]
    print(f\"loss spread (max-min): {max(losses)-min(losses):.4f}  \"
          f\"(near 0 => same effective batch really gives the same training)\")
PYEOF
    " 2>&1 | tee "$TRAIN_LOG"

echo
echo "=================================================================="
echo "DONE. Results -> $HOME/runs/a04-grad-accum/results.jsonl"
echo "=================================================================="
