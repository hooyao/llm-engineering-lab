# A1 — memory budget calculator: notes & verdicts

**Deliverable for A1** (`notes/curriculum-v2-execution.md` § A1). Code: `budget.py`.
Concept companions in this folder: `teaching-notes.md` (what a parameter is, where
the bytes go), `backprop-primer.md` (why training stores gradients + activations).

## Run it

```bash
python budget.py          # comparison table: 1B/3B/8B x full-SFT/LoRA x ckpt on/off
python budget.py --test   # asserts against notes/curriculum.md worked examples
```

Pure stdlib (argparse + dataclasses), no GPU needed — runs anywhere with Python 3.
`--test` was verified by hand against curriculum.md (GX10 was offline when written;
re-run on the box to confirm: all four checks land within 20%).

## The one number that matters: 12 vs 16 B/param — and which one you actually get

The teaching intuition (2 weight + 2 grad + 4 m + 4 v = 12) is the *pure-bf16*
recipe. A *mixed-precision* recipe adds a **fp32 master weight** (+4) for 16:

```
12 (pure bf16):   bf16 weight 2 + bf16 grad 2 +                 Adam m 4 + Adam v 4
16 (mixed-prec):  bf16 weight 2 + bf16 grad 2 + fp32 master 4 + Adam m 4 + Adam v 4
```

The fp32 master exists because bf16 has ~7 bits of mantissa, so many `lr * grad`
updates are too small to survive being added to a bf16 weight; you keep an fp32
copy, accumulate there, cast back to bf16 for the next forward.

> **MEASURED (A2, 2026-06-17):** which one you get depends on the trainer.
> **HuggingFace `Trainer(bf16=True)` uses 12 B/param — NO fp32 master.** A2's full
> SFT of Llama-3.2-1B peaked at 13.84 GB, matching the 12 B/param prediction
> (13.81 GB) to 0.2%, not the 16 B/param one (18.42 GB). The 16 B/param figure
> applies to **DeepSpeed / FSDP mixed-precision** or an explicitly mixed-precision
> optimizer. `curriculum.md` quotes 16 as the headline (conservative, DeepSpeed-era);
> the naive `Trainer` path is cheaper. Treat 16 as the *upper bound* and 12 as the
> HF-`Trainer` default. `budget.py` supports both: `master_dtype="fp32"` (16) vs
> `master_dtype=None` (12).

## Per-param byte cheat sheet (aligned with curriculum.md)

| method | B/param (weight+grad+state) | why |
|---|---|---|
| Full SFT, mixed-precision Adam | 16 | 2+2 + fp32 master 4 + m 4 + v 4 |
| Full SFT, 8-bit Adam | ~10 | moments quantized to 1 B each |
| LoRA (base frozen, bf16) | 2 (base) + ~0 | only ~1% adapter carries opt state |
| QLoRA (base NF4) | ~0.5 + ~0 | base in 4-bit |
| FP8 base + LoRA | ~1 + ~0 | native Blackwell path on sm_121 |

Activations add on top: `seq x batch x hidden x layers x dtype x mult`, mult ~6 no
checkpointing / ~1 with it (A5 makes this trade concrete).

## Verdicts on this unit (116 GB usable, seq=2048 batch=4, AdamW)

Three buckets: **comfortable** (<70% usable), **marginal** (70-100%), **WILL OOM** (>100%).

- **Comfortable, no thought needed:** Llama-1B/3B full SFT (any ckpt); 1B/3B/8B LoRA
  (any ckpt). The 8B LoRA case sits in tens-of-GB — huge headroom for batch/seq.
- **Marginal — full SFT 8B:** ~128 GB of weights+state alone at 16 B/param, before
  activations. Over budget as plain Adam. Fix per curriculum.md: **8-bit Adam (~80 GB)
  + activation checkpointing**, then it fits. This is the budget edge for full SFT.
- **WILL OOM as full SFT (need LoRA/QLoRA):** anything >=14B full-parameter. 14B full
  Adam ~= 14.8e9 x 16 ~= 237 GB — not close. These are LoRA/QLoRA/FP8-only on one box.
- **The GX10 selling point:** 14B BF16 LoRA ~= 30 GB, 32B FP8 LoRA ~= 33 GB — both
  roomy. QLoRA NF4 would drop the base ~4x again *if* a bitsandbytes aarch64+sm_121
  wheel works (unverified; Tier 3 ships FP8 to sidestep this — see curriculum.md).

## Caveats / honesty

- `target_frac=0.01` approximates the adapter size; a real run should compute the
  exact count from `rank x (in+out) x num_modules`. ~1% is right for the verdict,
  not for a precise checkpoint-size claim (that's A6's job).
- The activation `x6` multiplier is an estimate; real factor depends on arch/impl
  and attention kernel (Flash-Attention stores far less). A1 only needs ~20% and
  the right *direction* of the checkpointing trade. Verify with
  `torch.cuda.memory._record_memory_history()` on an actual run.
- Table uses binary GiB (1024^3, matches nvidia-smi); `--test` uses decimal GB
  (1e9, matches curriculum.md's headline arithmetic). Same bytes, different divisor.
