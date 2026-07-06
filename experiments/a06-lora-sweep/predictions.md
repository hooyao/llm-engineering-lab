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
config              formula              trainable params      MEASURED (2026-07-01 GX10)
──────────────────────────────────────────────────────────────────────────────────────
① r8  / attn        851,968 × 8            6,815,744  ≈ 6.82M    6,815,744    PASS (exact)
② r16 / attn        851,968 × 16          13,631,488  ≈ 13.63M   13,631,488   PASS (exact)
③ r16 / attn+mlp    2,621,440 × 16        41,943,040  ≈ 41.94M   41,943,040   PASS (exact)
④ r64 / attn+mlp    2,621,440 × 64       167,772,160  ≈ 167.77M  167,772,160  PASS (exact)
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

### MEASURED (2026-07-01 GX10) — the fp32 column is the winner

```
config        params   adapter bytes   MiB     bytes/param   -> dtype   matches fp32 pred?
──────────────────────────────────────────────────────────────────────────────────────
① r8/attn     6.82M     27,297,032    26.0    4.005         fp32       27.30M vs 27.26M ✓ (+34 KB header)
② r16/attn    13.63M    54,560,368    52.0    4.003         fp32       54.56M vs 54.53M ✓ (+34 KB header)
③ r16/a+mlp   41.94M   167,832,240   160.1    4.001         fp32      167.83M vs 167.77M ✓ (+60 KB header)
④ r64/a+mlp   167.77M  671,149,168   640.1    4.000         fp32      671.15M vs 671.09M ✓ (+60 KB header)
```

Every file lands on the **fp32** prediction (bytes/param → 4.000–4.005; the small excess
over exactly 4 is the safetensors header, 34–60 KB, exactly as predicted). The bf16 column
is refuted across the board.

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

### RESOLVED (2026-07-01): learner won — PEFT wrote fp32 (all 4 configs, bytes/param = 4.00)

The learner's **conclusion** was right; the **reason** was not the one that decides it.
- Right call, wrong mechanism: fp32 is not chosen "to carry more information." It's chosen
  because the trainable A/B are held in fp32 during training (that IS the master copy the
  AdamW m/v update — recall A2 Seg-6d: accumulated quantities need fp32). `save_pretrained`
  just serializes the live training tensors as-is → fp32 on disk. So the on-disk dtype is a
  **spillover of the training-state dtype**, not a deliberate precision choice.
- The tutor's bf16 reasoning was about the *inference* add (`(α/r)·B·A·x` onto bf16 `W·x`)
  — true as far as it goes, but it governs how you'd *load/merge* the adapter, not how PEFT
  *serializes* it. I applied an inference-time argument to a save-time decision. That's the
  actual error, and it's worth keeping: "which dtype survives to disk" is set by the
  training state, not by what inference will later tolerate.
- Bonus confirmation: the excess over exactly 4 B/param (34–60 KB) is the safetensors
  header, as predicted — clean, no surprise bytes.

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

---

## Column 3 — the one column with NO prior prediction: final loss + peak memory

Params and adapter-bytes were both derivable offline. Loss and peak memory were NOT — they
depend on the data + the run. Measured (Llama-3.1-8B-Instruct, tulu-3-sft-mixture ~484
samples, batch=4, seq=1024, lr=2e-4, 120 opt-steps, no checkpointing):

```
config          params    peak_mem   final_loss   step_ms   adapter (fp32)
────────────────────────────────────────────────────────────────────────────
① r8/attn       6.82M      49.24 GB    0.9120      3600      26.0 MiB
② r16/attn      13.63M     49.34 GB    0.9084      3615      52.0 MiB
③ r16/attn+mlp  41.94M     60.67 GB    0.9018      4921     160.1 MiB
④ r64/attn+mlp  167.77M    62.48 GB    0.9697      5028     640.1 MiB
```

Reading this column (nothing here was pre-derivable — this is the genuinely new data):

1. **More capacity did NOT monotonically lower loss.** ①→②→③ nudges down
   (0.9120 → 0.9084 → 0.9018 — a 0.011 total drop, tiny), then ④ (r=64) **rises** to 0.9697.
   The best final loss is ③, not the biggest ④. Two forces: (a) on ~484 samples / 120 steps
   there is very little to learn, so extra adapter capacity buys almost nothing; (b) r=64
   is 168M trainable params on a tiny dataset — it overfits / trains less stably at the same
   lr, so its *training* loss at step 120 is actually worse. This is exactly the
   underfit↔overfit trade the A6 plan said the sweep would expose (learning-notes Seg 2).
   Caveat: this is **train** loss on a tiny run, not a held-out eval — don't over-read the
   0.011 spread among ①②③; the honest signal is "capacity past ~r16+mlp did not help here,
   and r64 clearly hurt stability."

2. **peak_mem barely moves with adapter size, jumps with target coverage.** ①→② adds 6.8M
   params but peak is flat (49.24 → 49.34 GB): the LoRA state is a rounding error against the
   ~16 GB frozen base + activations. ②→③ jumps +11 GB (49.34 → 60.67) — NOT from the adapter
   (0.1 GB) but because **adapting mlp means the mlp activations now need grad**, so the
   backward graph retains far more activation memory. ③→④ adds another 126M params but only
   +1.8 GB (60.67 → 62.48). Lesson: **which modules you adapt drives memory (via activations),
   the rank barely does.** Same shape as A5's activation story.

3. **step_ms tracks the same split:** attn-only ~3.6 s, attn+mlp ~5.0 s — adapting the wide
   mlp matrices adds compute per step; rank (③→④) adds almost none.

## Qualitative gen (same 3 prompts, greedy, all 4 configs)

The point of the gen check: **general SFT on tulu-3 changes nothing factual, and this
prompt set can't distinguish the 4 configs** — expected, and itself the lesson.

- **"Explain LoRA to a systems engineer"** — all 4 answer "LoRA = Long Range (radio), an
  IoT wireless tech." The base model's world-knowledge meaning of the token "LoRA"; tulu-3
  is generic instruction data and never teaches the ML meaning, so no config learns it.
  A clean demonstration of A6 Seg-0: LoRA fits *style/format*, it does not *install facts*.
- **"Capital of France"** — all 4 say Paris (byte-similar). Knowledge unchanged by SFT, same
  as A3's finding on the 1B.
- **"nth Fibonacci"** — ①②③ emit a bare `def fibonacci(n):` stub; ④ (r=64) preambles with
  "Here is a Python function that returns...". The only visible cross-config difference, and
  it's a formatting nuance, not a capability gap.

Conclusion the learner reads: on this small generic-SFT run the four adapters are
behaviorally near-identical; the sweep's real payoff is the **params/memory/loss structure**
above (formula PASS ×4, dtype = fp32, memory driven by target-modules not rank, capacity
past r16+mlp stops helping), not a dramatic generation diff. To *see* a behavioral gap you'd
need task-specific data where the adapted capability is actually exercised (a Track-A8/A10
follow-up), not a 3-prompt smoke set.

## Which config would ship

**③ r16 / attn+mlp.** Best final loss (0.9018), adapting mlp (the memory/quality lever that
matters), adapter only 160 MiB, peak 60.7 GB (comfortable on the 128 GB unified pool). ④'s
4× params bought worse loss and more memory — no reason to pay it here. ①/② (attn-only) leave
the mlp frozen, which caps how much the adapter can re-express. So: **rank 16 is enough,
adapting mlp is worth the +11 GB, rank 64 is over-provisioning for a dataset this size.**
