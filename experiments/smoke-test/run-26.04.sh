#!/usr/bin/env bash
# Run the GB10 smoke test inside the NVIDIA PyTorch 26.04 container, with
# nvidia-smi dmon logging power / temperature / utilization in parallel.
#
# Difference vs run.sh (which targets 25.11-py3):
#   - Image tag is 26.04-py3 (CUDA 13.2, PyTorch 2.12.0a0, torchao 0.17.0+git)
#   - No version pins on transformers/peft/datasets/accelerate. The 25.11
#     container ships torchao 0.14, which made peft >= 0.18 refuse to load
#     (peft asserts torchao >= 0.16). 26.04 ships torchao 0.17, so latest
#     peft works out of the box. Less drift exposure, no pin maintenance.
#
# Verified 2026-06-05 on this unit: 5-step training succeeds at 0.91 steps/s,
# identical performance to 25.11 (same hardware, different software stack).
#
# Usage:
#   bash experiments/smoke-test/run-26.04.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
OUT_DIR="$HERE"

GPU_DMON_LOG="$OUT_DIR/gpu-dmon.log"
TRAIN_LOG="$OUT_DIR/train.log"

echo "=================================================================="
echo "GB10 SMOKE TEST — nvcr.io/nvidia/pytorch:26.04-py3"
echo "  repo root  : $REPO_ROOT"
echo "  out dir    : $OUT_DIR"
echo "  gpu log    : $GPU_DMON_LOG"
echo "  train log  : $TRAIN_LOG"
echo "=================================================================="

echo "[1/3] starting nvidia-smi dmon..."
nvidia-smi dmon -s pucmt -o DT -d 1 > "$GPU_DMON_LOG" 2>&1 &
DMON_PID=$!
trap "echo '[cleanup] stopping nvidia-smi dmon (pid=$DMON_PID)'; kill $DMON_PID 2>/dev/null || true" EXIT

echo "[2/3] pre-train GPU state:"
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,power.draw,utilization.gpu \
           --format=csv

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
    nvcr.io/nvidia/pytorch:26.04-py3 \
    bash -c '
        set -e
        echo "[setup] installing transformers/peft/datasets/accelerate (no pin)..."
        # 26.04 ships torchao 0.17.0+git, so modern peft / transformers
        # work without version pins. Last verified 2026-06-05 with
        # transformers 5.10.1 + peft 0.19.1 + datasets 4.8.4 + accelerate 1.13.0.
        pip install -q --root-user-action=ignore \
            transformers peft datasets accelerate
        echo "[setup] done."
        python experiments/smoke-test/train_lora_3b.py
    ' 2>&1 | tee "$TRAIN_LOG"

echo
echo "=================================================================="
echo "DONE — analyzing telemetry..."
echo "=================================================================="

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
        parts = s.split()
        try:
            vals = [p for p in parts if p.replace(".", "", 1).lstrip("-").isdigit()]
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
