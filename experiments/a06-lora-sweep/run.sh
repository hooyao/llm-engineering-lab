#!/usr/bin/env bash
# A6 — LoRA sweep (4 configs) on Llama-3.1-8B-Instruct, inside NVIDIA PyTorch 26.04.
#
# Configs (alpha/r held = 2, so the sweep isolates capacity r + target coverage):
#   (1) r=8   alpha=16   attn       -> predict   6.82M params
#   (2) r=16  alpha=32   attn       -> predict  13.63M params
#   (3) r=16  alpha=32   attn+mlp   -> predict  41.94M params
#   (4) r=64  alpha=128  attn+mlp   -> predict 167.77M params
#
# Each config = its own python process (clean per-config peak_mem + its own adapter dir).
# What to read off the run:
#   - trainable params == predicted?  (CERTAIN column — pure arithmetic, must PASS)
#   - adapter bytes/param ~4 (fp32) or ~2 (bf16)?  (resolves the serialize-dtype bet)
#   - does final_loss fall as capacity grows (1)->(4)?
#
# Usage:
#   bash experiments/a06-lora-sweep/run.sh                  # 120 steps/config (default)
#   bash experiments/a06-lora-sweep/run.sh --opt-steps 60   # quicker

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TRAIN_LOG="$HERE/train.log"
RESULT_JSON="/runs/a06-lora-sweep/results.jsonl"

mkdir -p "$HOME/runs/a06-lora-sweep/adapters"
rm -f "$HOME/runs/a06-lora-sweep/results.jsonl"

echo "=================================================================="
echo "A6 LoRA SWEEP — nvcr.io/nvidia/pytorch:26.04-py3"
echo "  base    : Llama-3.1-8B-Instruct"
echo "  data    : allenai/tulu-3-sft-mixture"
echo "  results : $HOME/runs/a06-lora-sweep/results.jsonl"
echo "  extra   : ${*:-<none>}"
echo "=================================================================="
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,memory.used --format=csv 2>/dev/null || true

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

        # config = 'r target alpha'
        for cfg in '8 attn 16' '16 attn 32' '16 attn+mlp 32' '64 attn+mlp 128'; do
          set -- \$cfg; R=\$1; TGT=\$2; ALPHA=\$3
          SAFE_TGT=\$(echo \$TGT | tr '+' '_')
          echo
          echo '##################################################################'
          echo \"## config: r=\$R  alpha=\$ALPHA  target=\$TGT\"
          echo '##################################################################'
          python experiments/a06-lora-sweep/train.py \
              --r \$R --alpha \$ALPHA --target \$TGT \
              --out-dir /runs/a06-lora-sweep/adapters/r\${R}_\${SAFE_TGT} \
              --result-json $RESULT_JSON $* || echo \"[warn] config r=\$R target=\$TGT failed\"
        done

        echo
        echo '=================================================================='
        echo 'A6 SWEEP TABLE'
        echo '=================================================================='
        python - <<'PYEOF'
import json
rows = [json.loads(l) for l in open('$RESULT_JSON')]
print(f\"{'config':>16} | {'params (M)':>12} | {'match':>5} | {'peak GB':>7} | {'loss':>7} | {'B/param':>7} | {'dtype':>5}\")
print('-'*80)
for r in rows:
    print(f\"{r['config']:>16} | {r['trainable_params']/1e6:>10.2f}M | \"
          f\"{'PASS' if r['params_match'] else 'FAIL':>5} | {r['peak_gb']:>7.2f} | \"
          f\"{r['final_loss']:>7.4f} | {r['bytes_per_param']:>7.3f} | {r['serialize_dtype']:>5}\")
print('-'*80)
print('predicted params: 6.82M / 13.63M / 41.94M / 167.77M  (must all PASS)')
print('bytes/param ~4 => PEFT wrote fp32 (learner wins) ; ~2 => bf16 (tutor wins)')
PYEOF
    " 2>&1 | tee "$TRAIN_LOG"

echo
echo "DONE -> $HOME/runs/a06-lora-sweep/results.jsonl"
