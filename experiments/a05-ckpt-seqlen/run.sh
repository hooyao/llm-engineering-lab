#!/usr/bin/env bash
# A5 — activation checkpointing vs seq_len sweep, inside NVIDIA PyTorch 26.04.
#
# 8 configs = seq_len {512,1024,2048,4096} x checkpointing {off,on}.
# 3B + LoRA r=16, batch=2. Each config its own python process (clean peak-mem).
#
# Predictions to check (the learner's):
#   P1: off-row activation ~8x from 512->4096, but peak_mem total < 8x (fixed part dilutes)
#   P2: off->on save% rises with seq_len toward k (~30-50%); step_time rises (extra forward)
#
# Usage:
#   bash experiments/a05-ckpt-seqlen/run.sh                 # 20 steps/config
#   bash experiments/a05-ckpt-seqlen/run.sh --opt-steps 30

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TRAIN_LOG="$HERE/train.log"
RESULT_JSON="/runs/a05-ckpt-seqlen/results.jsonl"

mkdir -p "$HOME/runs/a05-ckpt-seqlen"
rm -f "$HOME/runs/a05-ckpt-seqlen/results.jsonl"

echo "=================================================================="
echo "A5 CKPT x SEQ_LEN SWEEP — nvcr.io/nvidia/pytorch:26.04-py3"
echo "  results : $HOME/runs/a05-ckpt-seqlen/results.jsonl"
echo "  extra   : ${*:-<none>}"
echo "=================================================================="
nvidia-smi --query-gpu=name,temperature.gpu,power.draw --format=csv 2>/dev/null || true

docker run --rm --gpus all \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$HOME/models:/models:ro" \
    -v "$HOME/runs:/runs" \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:26.04-py3 \
    bash -c "
        set -e
        echo '[setup] installing transformers/datasets/accelerate/peft (no pin)...'
        pip install -q --root-user-action=ignore transformers datasets accelerate peft
        echo '[setup] done.'
        for seq in 512 1024 2048 4096; do
          for ck in off on; do
            echo
            echo '##################################################################'
            echo \"## seq_len=\$seq  ckpt=\$ck\"
            echo '##################################################################'
            python experiments/a05-ckpt-seqlen/train.py \
                --seq-len \$seq --checkpointing \$ck \
                --result-json $RESULT_JSON $* || echo \"[warn] seq=\$seq ckpt=\$ck failed\"
          done
        done

        echo
        echo '=================================================================='
        echo 'A5 2D TABLE  (peak_mem GB / step_time ms)'
        echo '=================================================================='
        python - <<'PYEOF'
import json
rows = [json.loads(l) for l in open('$RESULT_JSON')]
by = {(r['seq_len'], r['ckpt']): r for r in rows}
seqs = sorted({r['seq_len'] for r in rows})
def cell(r):
    if r is None: return '    --    '
    if r.get('oom'): return '   OOM    '
    return f\"{r['peak_gb']:>5.1f}G/{r['step_ms']:>5.0f}ms\"
print(f\"{'seq_len':>8} | {'ckpt OFF':>14} | {'ckpt ON':>14} | {'mem saved':>9} | {'time cost':>9}\")
print('-'*68)
for s in seqs:
    off, on = by.get((s,'off')), by.get((s,'on'))
    saved = cost = '   -'
    if off and on and not off.get('oom') and not on.get('oom'):
        saved = f\"{100*(off['peak_gb']-on['peak_gb'])/off['peak_gb']:>6.1f}%\"
        cost  = f\"{100*(on['step_ms']-off['step_ms'])/off['step_ms']:>6.1f}%\"
    print(f\"{s:>8} | {cell(off):>14} | {cell(on):>14} | {saved:>9} | {cost:>9}\")
print('-'*68)
print('P1: OFF peak_mem rises with seq_len but < 8x (fixed part dilutes).')
print('P2: mem-saved % should RISE with seq_len (activation share grows); time-cost > 0.')
PYEOF
    " 2>&1 | tee "$TRAIN_LOG"

echo
echo "DONE -> $HOME/runs/a05-ckpt-seqlen/results.jsonl"
