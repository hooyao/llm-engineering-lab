# A2 — first full-parameter SFT (Llama-3.2-1B)

**Deliverable for A2** (`notes/curriculum-v2-execution.md` § A2). Code: `train.py`,
launcher: `run.sh`. This is the smoke-test pipeline with the LoRA wrapper removed —
the first time every parameter is trained.

## What changed vs the smoke-test (LoRA-3B)

| | smoke-test | A2 |
|---|---|---|
| model | Llama-3.2-3B + LoRA r=16 | Llama-3.2-1B, **full-parameter** |
| trainable | ~24M (1%) | **1.24B (100%)** |
| training | 200 fixed steps | 1 epoch over 500 examples (~125 steps) |
| learning rate | 2e-4 (LoRA can be aggressive) | **2e-5** (full-param would diverge at 2e-4) |
| optimizer state | only on the adapter | on all 1.24B params (16 B/param, A1) |
| output | adapter discarded | full model saved to `~/runs/a02-sft-1b/` |

## Run

```bash
# on GX10:
cd ~/dgx-spark-playground
bash experiments/a02-sft-1b/run.sh                 # 500 ex, 1 epoch, seq=1024
bash experiments/a02-sft-1b/run.sh --seq-len 8192  # deliberately OOM, read the trace
```

## Results

Run 2026-06-17 on GX10, 26.04 container. 500 alpaca-cleaned examples, 1 epoch,
batch=4, seq=1024, bf16, lr=2e-5 cosine.

- **steps / wall-clock:** 125 steps in 82.9 s (1.51 steps/s, 6.03 samples/s)
- **loss:** first logged 1.717 -> final train_loss 1.478 (last step 1.347)
- **peak GPU memory:** 13.84 GB
- **saved model:** `~/runs/a02-sft-1b/model.safetensors` = 2.47 GB
  (1.236B params x 2 bytes bf16 = 2.47 GB, exactly as expected)

Note: loss started at ~1.7, not the ~2.5 the day-plan guessed, because
Llama-3.2-1B-**Instruct** is already instruction-tuned — it isn't cold-starting on
the alpaca format the way a base model would. The drop is real but shallow (small
data, 1 epoch); the point of A2 is the loop, not a converged model.

### Loss curve

```
  5: 1.717   30: 1.587   55: 1.398   80: 1.431   105: 1.314
 10: 1.592   35: 1.374   60: 1.360   85: 1.332   110: 1.204
 15: 1.609   40: 1.530   65: 1.559   90: 1.533   115: 1.435
 20: 1.548   45: 1.557   70: 1.565   95: 1.294   120: 1.473
 25: 1.510   50: 1.597   75: 1.709  100: 1.384   125: 1.347
```

Downward but noisy — expected with 500 examples and micro-batch 4. The cosine
schedule pulls lr to ~3e-9 by the end, so the last ~20 steps barely move.

### The memory finding (A1 -> A2 closed loop) — IMPORTANT

A1 predicted full-param 1B SFT at **16 B/param** (mixed-precision Adam, with an
fp32 master weight) = ~18.4 GB before activations. **Measured peak was 13.84 GB —
lower than predicted.** Reconciling the three candidate recipes:

| recipe | B/param | predicted (weights+opt only) |
|---|---|---|
| mixed-precision Adam + fp32 master | 16 | 18.42 GB |
| fp32 master, no separate bf16 weight | 14 | 16.12 GB |
| **pure bf16 (w2 + g2 + m4 + v4, NO master)** | **12** | **13.81 GB** |

Measured 13.84 GB matches the **12 B/param** prediction to within 0.2%. Conclusion:

> **HuggingFace `Trainer` with `bf16=True` keeps NO fp32 master weight by default.**
> It trains in pure bf16: bf16 weight (2) + bf16 grad (2) + fp32 Adam m,v (8) = 12.
> The 16 B/param figure (with fp32 master) applies to **DeepSpeed / FSDP
> mixed-precision** or an explicitly mixed-precision optimizer, not naive
> `Trainer(bf16=True)`.

This corrects the A1 default assumption. `budget.py` already supports it:
`training_state_bytes(..., master_dtype=None)` gives the 12 B/param recipe. A1's
teaching-notes and notes were updated with this correction. Activations here are
tiny (~tens-hundreds of MB at seq=1024 batch=4 on a 1B model), which is why the
measured peak is essentially just the 12 B/param weights+optimizer total.

### Caveat: saved files are root-owned

The container runs as root, so `~/runs/a02-sft-1b/*` is `root:root`. Harmless for
reading, but `chown` (with sudo) if you need to modify/delete as `hooyao`.

## The OOM exercise (`--seq-len 8192`)

Not run yet — optional. With seq=8192 (8x the tokens), the activation term grows
~8x and the attention scores grow ~64x (quadratic in seq_len), which is what tips
a 1B full-SFT run over. Run `bash experiments/a02-sft-1b/run.sh --seq-len 8192` to
see the `torch.cuda.OutOfMemoryError` trace; the failed allocation will be an
activation/attention buffer, not the weights (those fit fine at 13.8 GB).

