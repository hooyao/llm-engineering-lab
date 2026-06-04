#!/usr/bin/env bash
# Run the GB10 smoke test inside the NVIDIA PyTorch container, with
# nvidia-smi dmon logging power / temperature / utilization in parallel.
#
# Usage:
#   bash experiments/smoke-test/run.sh
#
# Outputs (under experiments/smoke-test/):
#   gpu-dmon.log    one line per second: pwr, temp, gpu%, mem%, ...
#   train.log       Trainer stdout (loss curve)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
OUT_DIR="$HERE"

GPU_DMON_LOG="$OUT_DIR/gpu-dmon.log"
TRAIN_LOG="$OUT_DIR/train.log"

echo "=================================================================="
echo "GB10 SMOKE TEST"
echo "  repo root  : $REPO_ROOT"
echo "  out dir    : $OUT_DIR"
echo "  gpu log    : $GPU_DMON_LOG"
echo "  train log  : $TRAIN_LOG"
echo "=================================================================="

# Start GPU telemetry collector in the background (host-side, not container).
# `nvidia-smi dmon` columns:
#   gpu pwr gtemp mtemp sm  mem enc dec  mclk pclk
#   #     W    C     C   %   %   %   %   MHz  MHz
echo "[1/3] starting nvidia-smi dmon..."
nvidia-smi dmon -s pucmt -o DT -d 1 > "$GPU_DMON_LOG" 2>&1 &
DMON_PID=$!
trap "echo '[cleanup] stopping nvidia-smi dmon (pid=$DMON_PID)'; kill $DMON_PID 2>/dev/null || true" EXIT

# Pre-train GPU snapshot.
echo "[2/3] pre-train GPU state:"
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw,utilization.gpu \
           --format=csv

# Launch the training container with mandatory flags + bind mounts.
echo "[3/3] launching training (this will take ~5-10 min)..."
echo "  ↪ watch:   tail -f $TRAIN_LOG"
echo "  ↪ telemetry: tail -f $GPU_DMON_LOG"
echo

docker run --rm --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HOME/models:/models:ro" \
    -v "$REPO_ROOT:/workspace" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.11-py3 \
    bash -c '
        set -e
        echo "[setup] installing transformers/peft/datasets/accelerate..."
        pip install -q --root-user-action=ignore \
            transformers peft datasets accelerate
        echo "[setup] done."
        python experiments/smoke-test/train_lora_3b.py
    ' 2>&1 | tee "$TRAIN_LOG"

echo
echo "=================================================================="
echo "DONE — analyzing telemetry..."
echo "=================================================================="

# Summarize telemetry. Skip the header (first 2 lines from dmon).
python3 - <<'PY'
import os, sys
log = os.path.expanduser(os.environ.get("GPU_DMON_LOG", "experiments/smoke-test/gpu-dmon.log"))
if not os.path.exists(log):
    print("  (no telemetry log found at", log, ")")
    sys.exit(0)

pwr, temp, sm = [], [], []
with open(log) as f:
    for line in f:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # dmon -o DT prefixes lines with date + time; columns shift.
        # Typical: 2026-06-05 22:35:01   0   45  52  52  95  10 ...
        parts = s.split()
        try:
            # find first numeric token after date+time prefix
            # date is 2 tokens, then gpu index, then values
            vals = [p for p in parts if p.replace(".", "", 1).lstrip("-").isdigit()]
            # vals layout: [gpu, pwr, gtemp, mtemp, sm, mem, enc, dec, mclk, pclk]
            if len(vals) < 7:
                continue
            pwr.append(float(vals[1]))
            temp.append(float(vals[2]))
            sm.append(float(vals[4]))
        except Exception:
            continue

if not pwr:
    print("  (no data points parsed)")
    sys.exit(0)

def stats(xs, unit):
    return f"min={min(xs):.0f}{unit}  avg={sum(xs)/len(xs):.0f}{unit}  max={max(xs):.0f}{unit}"

print(f"  samples       : {len(pwr)} (1Hz)")
print(f"  power         : {stats(pwr, 'W')}")
print(f"  gpu temp      : {stats(temp, '°C')}")
print(f"  sm utilization: {stats(sm, '%')}")
print()
print("  raw log:", log)
PY
