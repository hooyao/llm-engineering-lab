# A6 — prediction table (derived offline; MEASURED column waits for GX10)

> Built tonight (2026-06-25) with the learner, fully offline (GX10 unreachable this
> session). The **trainable-params** column is pure arithmetic → **certain**. The
> **adapter-on-disk** column is a **prediction** that depends on the dtype PEFT serializes
> the adapter in — to be reverse-engineered from the measured file size on the box,
> exactly like A2 reverse-engineered `12 B/param` from the measured peak. Fill the
> MEASURED columns when the sweep runs (Task #3).

## The 4 configs (only `r` and target modules vary; α/r held = 2)

```
config              r     α     α/r   target modules
─────────────────────────────────────────────────────────────
① r8  / attn        8     16    2     attn-only  (q,k,v,o)
② r16 / attn        16    32    2     attn-only
③ r16 / attn+mlp    16    32    2     attn+mlp   (q,k,v,o + gate,up,down)
④ r64 / attn+mlp    64    128   2     attn+mlp
```

Adjacent pairs isolate one variable each: ①→② = r×2 (target fixed); ②→③ = +mlp (r
fixed); ③→④ = r×4 (target fixed).

## Base model shape (Llama-3.1-8B-Instruct)

```
d_model = 4096,  d_ff (MLP intermediate) = 14336,  num layers = 32
per-layer linear layers (W = [d_out, d_in]):
  attn: q_proj [4096,4096]  k_proj [1024,4096]  v_proj [1024,4096]  o_proj [4096,4096]
  mlp : gate_proj [14336,4096]  up_proj [14336,4096]  down_proj [4096,14336]
```

## Per-layer LoRA params (learner-derived formula)

LoRA params for one W = `r × (d_in + d_out)` — the **perimeter** (sum of the two dims),
NOT the **area** (d_out×d_in) that full fine-tuning pays. This is why LoRA makes the wide
mlp matrices (one dim = 14336) affordable: 14336 becomes an addend, not a multiplier.

```
attn-only per layer = (4096+4096 + 1024+4096 + 1024+4096 + 4096+4096) × r = 26,624 × r
attn+mlp  per layer = 26,624×r + (14336+4096)×3×r = 26,624×r + 55,296×r = 81,920 × r
```

× 32 layers (every layer adapted identically within a config):

```
attn-only whole model = 26,624 × 32 × r =   851,968 × r
attn+mlp  whole model = 81,920 × 32 × r = 2,621,440 × r
```

## Column 1 — trainable params (CERTAIN — pure arithmetic)

```
config              formula              trainable params      MEASURED (fill on box)
──────────────────────────────────────────────────────────────────────────────────────
① r8  / attn        851,968 × 8            6,815,744  ≈ 6.82M    __________
② r16 / attn        851,968 × 16          13,631,488  ≈ 13.63M   __________
③ r16 / attn+mlp    2,621,440 × 16        41,943,040  ≈ 41.94M   __________
④ r64 / attn+mlp    2,621,440 × 64       167,772,160  ≈ 167.77M  __________
```

Two clean regularities the configs were arranged to expose:
```
① → ②   r×2  (8→16)         params ×2.00   ← r is LINEAR: double r, double params
② → ③   +mlp (r=16 fixed)   params ×3.08   ← adding mlp ≈ ×3 (perimeter, not the ×3.5 area)
③ → ④   r×4  (16→64)        params ×4.00   ← r linear again
```

## Column 2 — adapter on disk (PREDICTION — depends on serialize dtype)

`adapter_bytes = trainable_params × bytes_per_param`. Two hypotheses:

```
                       fp32 (4 B/param)              bf16 (2 B/param)
config        params   bytes         MiB    MB       bytes         MiB    MB
────────────────────────────────────────────────────────────────────────────
① r8/attn     6.82M    27,262,976    26.0   27.3     13,631,488    13.0   13.6
② r16/attn    13.63M   54,525,952    52.0   54.5     27,262,976    26.0   27.3
③ r16/a+mlp   41.94M  167,772,160   160.0  167.8     83,886,080    80.0   83.9
④ r64/a+mlp   167.77M 671,088,640   640.0  671.1    335,544,320   320.0  335.5
```

(MiB = ÷1,048,576, what `ls -lh` shows; MB = ÷1e6. Measured file will be a hair larger
than pure params×bytes due to the safetensors header — a few KB, negligible.)

## The bet (resolve on GX10, A2-style)

- **Learner's bet: fp32.** Reasoning: adapter is only a few hundred MB so there's no
  pressure to save space, and fp32 "carries more information."
- **Tutor's bet: bf16.** Reasoning: the A/B matrices are **weights** (same class as the
  base W). The base is stored bf16; at inference `(α/r)·B·A·x` is added to a bf16 `W·x`,
  so any fp32 precision in the adapter is truncated on that add — it's *unusable*
  information. The real driver isn't space-saving (learner is right that nobody cares
  about the few hundred MB) but **dtype-alignment with the base**, so adapter and base
  don't mismatch at inference. So bf16 is the likely default.
- **BUT** PEFT's actual default is version-dependent and must NOT be quoted from memory.
  Some versions do write fp32. So this is genuinely open — the measured file size decides.

### How to resolve on the box

```
# after each config trains and saves its adapter dir:
ls -l <adapter_dir>/adapter_model.safetensors          # raw bytes
# divide bytes by the trainable_params above:
#   ≈ 4  → PEFT serialized fp32   (learner wins)
#   ≈ 2  → PEFT serialized bf16   (tutor wins)
# also cross-check trainable_params: the training log prints
#   "trainable params: X || all params: Y || trainable%: Z"
#   X must match column 1 (6.82M / 13.63M / 41.94M / 167.77M).
```
