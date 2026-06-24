# A5 — activation checkpointing vs seq_len: results & payoff

**Day:** A5 (`notes/curriculum-v2-execution.md` § A5). **Model:** Llama-3.2-3B-Instruct,
**LoRA r=16** (attn+mlp), BF16, batch=2. **Sweep:** seq_len {512,1024,2048,4096} ×
checkpointing {off,on} = 8 configs. Each config its own process (clean peak-mem).

## What A5 demonstrates

Activation checkpointing trades **recompute (time)** for **activation memory (space)**.
The learner derived the whole trade (notes Seg 0): backward needs each layer's forward
input `x` (`dloss/dw = x * dloss/dz`); checkpointing drops most x's in forward and
recomputes them from block-boundary checkpoints during backward — less memory, one extra
partial forward.

LoRA r=16 is used so the FIXED part (frozen base + tiny adapter grad/optimizer) is small
and the **activation part dominates** — which makes the checkpointing contrast legible.

## 2-D table (peak_mem GB / step_time ms)

<!-- FILLED FROM THE GX10 SWEEP -->
```
 seq_len |   ckpt OFF      |    ckpt ON      | mem saved | time cost
--------------------------------------------------------------------
     512 |  13.7G /  592ms |   8.2G /  796ms |   39.9%   |  +34.3%
    1024 |  21.0G / 1354ms |  10.1G / 1826ms |   52.0%   |  +34.8%
    2048 |  36.0G / 3063ms |  13.9G / 4085ms |   61.4%   |  +33.4%
    4096 |  66.5G / 7273ms |  21.5G / 9634ms |   67.7%   |  +32.5%
```

## Predictions (the learner's, made before the run)

- **P1:** OFF-row activation ~8× from 512→4096 (activation ∝ seq_len), but peak_mem TOTAL
  rises **less than 8×** because the fixed part (frozen 3B base ~6.5 GB + LoRA grad/opt)
  doesn't scale with seq_len.
- **P2 (derived algebraically by the learner):** off→on save% =
  `k / (F/(c·seq_len) + 1)` → rises monotonically with seq_len toward `k` (the drop
  fraction, ~30–50%), because activation's share of total memory grows with seq_len.
  Plus: step_time rises (the extra forward).

## Verdict

Both predictions verified on metal:

- **P2 confirmed (the headline):** save% rose **monotonically** 39.9 → 52.0 → 61.4 →
  67.7% across seq_len 512→4096 — exactly the `k/(F/(c·seq_len)+1)` shape. time-cost
  stayed ~+33% (the recompute is one fixed extra forward, constant share). The save
  exceeds the day-plan's quoted 30–50% at large seq_len because the activation share
  keeps growing.
- **P1 confirmed:** OFF peak rose 13.65 → 66.51 GB over seq ×8 = only **4.87×**, not 8×,
  because the fixed part doesn't scale. Solving the two-point model `peak = F + c·seq_len`
  gives **F ≈ 6.2–7.0 GB** = the frozen 3B base (3.2B × 2 bytes ≈ 6.4 GB) — the learner's
  algebra reverse-solves the physical fixed cost from the data.

**The bigger practical point (beyond the day-plan):** at seq=4096, OFF needs **66.5 GB** —
it would OOM on a 24 GB or 40 GB card. ON needs only **21.5 GB** — fits. So checkpointing
isn't "save a bit of memory," it's **the switch that makes an otherwise-impossible config
runnable**. It pushes the "what can this card train" boundary (talk slide 9) wide open,
especially for long-context training. On GX10's 128 GB unified pool 66.5 GB still fits, but
on a normal GPU checkpointing is the difference between "can train at this seq_len" and "can't."

Trade summary the learner reasoned out, now measured: checkpointing trades a steady ~+33%
step-time for a memory saving that grows with seq_len (40%→68% here), and the saving is
what converts OOM into runnable.
