# A4 — learning notes (gradient accumulation, learner-paced)

> **What this file is — same per-learner format as A2's `a02-sft-1b/learning-notes.md`.**
> Three rules carried over (honor all three):
> 1. **Complete coverage** of A4 — standalone, relearn the whole day from this file.
> 2. **Depth set by THIS learner's familiarity** — unfamiliar things in full;
>    things they already know (systems/.NET, the A2 four-beat loop they've learned)
>    one line and move on.
> 3. **Phrased the way THIS learner understands best** — discovered through the
>    dialogue. Anchor in real numbers / systems trade-offs; state direction and
>    scale explicitly (their #1/#2 weak spots); math terminal-safe (`y[i]`, loops,
>    no rendered subscripts).
>
> Pacing: one small segment -> learner clarifies in place -> fold Q&A back here ->
> next segment. **Learner wants to READ the code before running it** — do not rush
> to a GX10 run.
>
> A4 also pays off an owed debt: **A2's `train.py` hidden loop gets unrolled here.**
> `trainer.train()` hid forward/loss/backward/optimizer; gradient accumulation forces
> an explicit loop, so each line maps back to the four beats from A2 Seg 6.

---

## What A4 teaches (the whole point, in one line)

**Decouple "how big a batch the algorithm sees" (effective batch) from "how big a
batch the GPU holds at once" (micro-batch)** — the bridge between them is gradient
accumulation. Plus: the explicit training loop A2 hid.

---

<!-- segments appended here as the lesson proceeds -->

## Segment 0 — calibration (where the learner started)

Two priming questions before any mechanism:

- **Q1: what does a bigger batch buy the training itself?** Learner: "makes the
  computed gradient closer to the real data." Correct; sharpened to the load-bearing
  word: a batch's gradient is the **average** of the per-sample gradients. Bigger N
  -> averaging more samples -> less noise -> closer to the true full-dataset gradient.
  (The word "average" is what A4's whole mechanism rests on.)

- **Q2: why not just keep making the batch bigger?** Learner gave TWO reasons,
  correctly, and they are two independent axes:
  1. "memory can't hold it" — YES, this is the exact wall gradient accumulation
     removes (activation memory grows linearly with batch, per A1).
  2. "vague impression that too-big a batch hurts convergence" — also a REAL ML
     effect (large-batch generalization gap / sharp minima / lr must scale with
     batch). **Key boundary: gradient accumulation does NOT fix axis 2.** It only
     knocks down axis 1 (memory). The big effective batch it hands you still carries
     every convergence side-effect. So A4 is not "open the batch infinitely" — it's
     "decouple what-the-algorithm-sees from what-the-GPU-holds."

This learner's systems instinct framed memory as the wall immediately — lean on that.

## Segment 1 — gradient is per-sample AND per-parameter (clearing the "16 numbers" confusion)

The learner was right about gradient and I was sloppy. Correction recorded because it
hit weak spot #2 (fused scales): I said "16 numbers" without pinning the scale.

- **What the learner already had (correct, from A2 Seg 6e):** gradient is
  **per-parameter** — each `w`, each `b` has its own gradient number.
- **The axis I'd left implicit:** gradient is ALSO **per-sample**. Each single sample,
  fed alone, produces a WHOLE set of gradients (one per parameter). This is what batch
  introduces.

Pinned with a 5-param neuron (`w1,w2,w3,w4,b`) and a 3-sample batch as a TABLE — two
explicit axes (the framing that worked):

```
              w1      w2      w3      w4      b
sample 1:    +0.2    -0.1    +0.3    +0.0    -0.2     <- one full gradient row per sample
sample 2:    +0.4    -0.3    +0.1    -0.1    -0.4
sample 3:    +0.0    +0.1    -0.1    +0.2    -0.3
            ----    ----    ----    ----    ----
batch grad:  +0.2    -0.1    +0.1    +0.03   -0.3     <- average DOWN each column
```

- horizontal axis = WHICH PARAMETER (`w1..b`), the per-parameter axis they knew.
- vertical axis = WHICH SAMPLE (the batch), the axis I'd glossed.
- **"batch gradient" = per-sample gradients averaged column-wise (per parameter).**
  `trainer.train()` (A2's black box) does exactly this average, then steps the optimizer.
- So "16 numbers' average" = for ONE parameter (say w1), the average of the 16 samples'
  gradients for w1 — one COLUMN of 16 numbers, not 16 parameters. My phrasing, fixed.

## Segment 2 — the learner derived gradient accumulation themselves

Asked "to average, don't you just keep ONE row, accumulate onto it, divide by the count
at the end?" — that IS gradient accumulation, complete. Confirmed and tied to memory:

```
accumulator: [0,0,0,0,0]                 <- the one row they keep (size = #params)
feed sample1 -> its row -> add; DROP sample1's activation
feed sample2 -> its row -> add; DROP sample2's activation
feed sample3 -> its row -> add; DROP sample3's activation
divide by 3                              <- identical, bit-for-bit, to "3 rows averaged at once"
```

**The two algorithms are mathematically identical** — same gradient to the optimizer.
The ONLY difference is memory:

| thing | grows with | in the accumulate scheme |
|---|---|---|
| accumulator (the gradient row) | #params only | CONSTANT — 3 samples or 300, same size |
| each sample's activation (A1) | # samples in flight at once | computed then DROPPED -> only 1 in memory |

The trick in one line: **the accumulator doesn't grow with batch (it's param-sized, and
gradient is stored anyway — A1's 12-byte item #2). What blows up memory is activation,
which is use-then-discard, so you only hold ONE sample's activation at a time.** Want
effective batch 16 but GPU holds only 1 sample's activation? Feed 1, accumulate, drop;
x16; divide; step. Algorithm sees the 16-sample average; GPU never holds >1 sample's
activation. The memory wall (learner's Q2 axis 1) is gone.

**The three A4 terms, now precise:**
- **micro-batch** = samples the GPU holds at once (= the activation-memory number).
- **effective batch** = samples the algorithm sees per optimizer step (= how many you
  accumulate before stepping).
- bridge: `effective_batch = micro_batch * accumulation_steps`.

The day-plan's three configs, all effective batch 16, just split differently:
```
micro=1  accum=16  -> eff 16   (least memory, slowest)
micro=4  accum=4   -> eff 16   (more memory, faster)
micro=8  accum=2   -> eff 16   (most memory, fastest, near-OOM)
```
Payoff shape (final form co-designed on the GX10): same effective batch -> ~identical
final loss (direct evidence of the math identity), different peak_mem / step_time. You
trade memory for speed WITHOUT changing what the algorithm sees.

## Segment 3 — why `loss / ACCUM_STEPS` makes every gradient divide by ACCUM_STEPS

Learner asked the precise question: why does scaling the scalar `loss` make each
parameter's gradient divide by the same factor? Answer = gradient is LINEAR in loss.

Proof for one param `w`, using the constant-multiple rule `d(c*f)/dw = c*d(f)/dw`:
```
new_gradient = d(loss/16)/dw = (1/16)*d(loss)/dw = gradient/16
```
Holds for EVERY param (the rule doesn't pick params), and PyTorch differentiates each
param separately -> all 16 micro-batches, every param, divided by 16. In the
accumulator: each g_k arrives as g_k/16, so the sum = (g1+...+g16)/16 = average — the
true batch-of-16 gradient. (Distributive law: dividing each term = dividing the sum.)

Two engineering points: (1) divide loss BEFORE backward (one scalar op) not the 1.2B
grad tensors after — same math, far cheaper. (2) This is the SAME linearity as A2 Seg
6d loss scaling, opposite direction: there multiply-up to dodge fp16 underflow, here
divide-down to recover the average. One property, two uses.

### Seg 3 follow-up — the product-rule slip (derivative of a constant = 0, not 1)

Learner recalled the product rule correctly `(uv)' = u'v + uv'`, set u=1/16, v=loss,
but wrote `u' = 1`. The slip: **derivative of a CONSTANT is 0, not 1.**
```
u  = 1/16   -> u' = d(1/16)/dw = 0    (1/16 is fixed, doesn't change as w changes)
v  = loss   -> v' = gradient
(uv)' = u'v + uv' = 0*v + (1/16)*gradient = gradient/16     <- matches the const-mult route
```
With u'=0 the `u'v` term vanishes and the product rule collapses to the constant-multiple
rule — they're not two rules, the const-mult rule IS the product rule when u is constant.
Root of the slip (weak spot #2, fusing adjacent things): confused **u's VALUE (1/16)**
with **u' (u's RATE OF CHANGE, 0)**, likely crossed with `d(x)/dx = 1` (x is a VARIABLE,
rate 1; 1/16 is a CONSTANT, rate 0). Teaching fix used: separate "value" from "rate of
change" explicitly.

## Segment 4 — learner predicted the payoff table before running it

Before the sweep, learner predicted all three columns (same effective batch 16,
configs micro=1/accum=16, micro=4/accum=4, micro=8/accum=2). Scored:

- **final_loss: "basically the same"** — CORRECT, and the right reason: effective
  batch 16 -> algorithm sees the same averaged gradient -> same training. This is the
  central thing A4 verifies.
- **step_time: "micro=8 fastest, it's 8 samples computed in parallel, only 2 passes
  vs micro=1's 16 sequential passes"** — CORRECT mechanism. Sharpened: not just
  "parallel" but a bigger GEMM fills the Tensor Cores (small 1-sample GEMMs leave
  compute idle) + 16->2 kernel launches. Core instinct (bigger micro -> bigger GEMM
  -> higher GPU util -> faster step) is right.
- **peak_mem decomposition: `fixed(params) + activation*micro`, micro=8 largest** —
  STRUCTURE PERFECT (fixed part shared across configs, activation scales with micro,
  so micro=8 biggest). But used **12 B/param** (carried from A2). Caught by the smoke
  test: micro=1 measured 30.09 GB total, yet 3.21B x 12 = 38.5 GB for the fixed part
  ALONE > total -> impossible (activation can't be negative). So 12 is wrong here.

### The 12-vs-8 catch (a real implementation detail, ties back to Seg 6d)

A2 used HF Trainer (mixed precision) -> m/v in **fp32** -> 12 B/param. A4 uses **raw
torch.optim.AdamW on bf16 params**. torch.optim.AdamW creates m/v via zeros_like(p),
so on a bf16 param the states are **bf16** -> 8 B/param:
```
weight 2 + grad 2 + m 2 + v 2 = 8 B/param      (this A4 script)
3.21B x 8 = 25.7 GB fixed + ~4.4 GB activation(micro=1) ~= 30.1 GB  == measured 30.09 ✓
```
Two takeaways the learner gets to SEE:
1. This A4 script is USING the exact Seg-6d footgun the learner warned about: bf16 m/v.
   Fine for a 30-step demo; a real long run would have small m/v increments eaten by
   "big eats small" -> use fp32 optimizer states. (The mem-check diagnostic prints the
   actual dtype to prove it's bf16, not my assertion.)
2. The constant 12->8 isn't a learner error in reasoning — the DECOMPOSITION was right;
   the constant is set by an implementation detail (who decides optimizer-state dtype),
   which is exactly the "why this design" the learner likes.

## Segment 5 — the payoff table, and the learner's best catch of the day

Ran the 3-config sweep on the GX10. Authoritative results (isolated re-run confirmed
micro=8 bit-for-bit after a contaminated first read — see notes.md data-integrity note):
```
            peak_mem    step_time   final_loss
micro=1     30.16 GB    4363 ms     1.3526
micro=4     35.82 GB    3378 ms     1.2469   <- sweet spot
micro=8     46.96 GB    3834 ms     1.2489   <- slower AND more memory
loss spread (max-min): 0.1057
```

**Prediction scorecard (Seg 4):** final_loss ~same — CONFIRMED (0.11 band << 985ms
step spread). peak_mem rises with micro, micro=8 largest — CONFIRMED. step_time:
learner said micro=8 fastest — DIRECTIONALLY right (bigger micro IS faster up to a
point) but the data showed a TWIST the learner then caught himself.

**The learner's catch ("doesn't seem much faster"):** step_time is NON-MONOTONIC.
```
micro 1->4:  -985 ms (-23%)   big speedup (filling an under-fed GPU)
micro 4->8:  +456 ms (+13%)   SLOWER
```
This is the day's best observation. The mechanism (two layers, both the learner's
systems turf):
1. Diminishing returns: micro=4 already saturates the GPU (big enough GEMM to fill the
   Tensor Cores; 16->4 kernel launches). micro=8 finds no idle compute to fill.
2. Negative returns: micro=8's 47 GB peak + bigger activation tensors push the kernel
   toward MEMORY-BANDWIDTH-BOUND on GB10's shared 273 GB/s LPDDR5x. More data to move,
   same compute -> slower. So past the saturation point, bigger micro = pay memory for
   negative speed.

**The corrected practical rule (stronger than the day-plan's):** micro-batch has an
OPTIMUM ("just saturate the GPU" — micro=4 here). Below it, waste GPU on tiny GEMMs;
above it, lose on BOTH memory and speed. Not "bigger is always faster." This is exactly
the learner's Q2-axis-2 boundary instinct ("too big hurts") landing quantitatively on
the throughput axis.

**Two ground-truth findings the learner gets to SEE:**
- m/v dtype printed as `torch.bfloat16` -> confirms 8 B/param fixed (not A2's 12); the
  Seg-6d bf16-m/v footgun is literally in use (fine for 30 steps, wrong for a long run).
- mem-check showed ~6 B/param resident between steps: `zero_grad(set_to_none=True)` frees
  grad entirely between steps (grad's 2 B is transient, only in the peak). Literal proof
  of Seg-2's "gradient is use-then-discard."

**Process catch:** trusting the first (polluted) Monitor read would have reported
"micro=8 fastest" — the opposite of the truth. The learner's skepticism about the feel +
an isolated re-run on an idle GPU is what saved the conclusion. Measurement hygiene:
peak_mem/step_time are only valid on an otherwise-idle GPU.

## Segment 6 — when are m/v fp32 vs bf16? (the rule, and why A4 differs from A2)

Learner asked the sharp question: the A4 mem-check printed m/v = bf16, but A2 taught m/v
= fp32 (the 8-of-the-12-bytes). Which is it? Answer: BOTH are possible — it depends on
WHO creates the optimizer state and what dtype the param is.

**The one rule:** `torch.optim.AdamW` creates m and v via `torch.zeros_like(param)`.
`zeros_like` copies the param's shape AND dtype. So **m/v dtype always follows the param's
dtype**:
```
param fp32  ->  m/v fp32  (4+4 = 8 bytes)
param bf16  ->  m/v bf16  (2+2 = 4 bytes)   <- what A4 hit, with raw AdamW on a bf16 model
```
A4's loop loaded the model in bf16 and used a bare `torch.optim.AdamW`, so m/v came out
bf16 — the Seg-6d footgun, used silently (no error, no warning).

**Why A2 was fp32 even though its param was also bf16:** A2 used HF `Trainer` + mixed
precision, which keeps an EXTRA fp32 **master copy** of the weights. The optimizer updates
the master (fp32), so m/v follow the master, not the bf16 compute copy:
```
A2 (HF Trainer, mixed precision):
   bf16 compute copy of weights   (forward/backward — saves memory + bandwidth)
   fp32 master copy of weights    (what the optimizer actually steps)
   m/v = fp32                     (follow the master)
```

So the real determinant of "32 vs 16 bit m/v" = **does the framework keep an fp32 master
weight?**

| scenario | compute param | fp32 master? | m/v | B/param |
|---|---|---|---|---|
| A2 (HF Trainer, mixed precision) | bf16 | YES | **fp32** | 12 |
| A4 (raw torch.optim.AdamW, bf16 param) | bf16 | no | **bf16** | 8 |
| pure fp32 training (small models / old way) | fp32 | n/a (param is fp32) | fp32 | 16 |
| production standard (bf16 + fp32 master) | bf16 | YES | **fp32** | 12 |

**Which SHOULD you use? fp32 m/v for any real run (the 12 B/param tier).** This is exactly
the learner's own Seg-6d reasoning: m/v are LONG-ACCUMULATED quantities (m = 0.9*m + 0.1*g
over thousands of steps); bf16's 2-3 significant digits get the small increments eaten by
"big eats small" late in training -> optimizer silently stalls. So bf16 m/v (raw AdamW on
bf16 params) is a memory-saving but DANGEROUS config, fine only for a 30-step demo like A4.
A real run uses mixed precision (HF Trainer / accelerate auto-creates fp32 master + fp32
m/v) or manually forces fp32 optimizer state.

**One-line takeaway:** m/v dtype = the dtype of the tensor that created it (`zeros_like(
param)` for raw AdamW). bf16 param + bare optimizer -> bf16 m/v (8 B, risky); mixed
precision with an fp32 master -> fp32 m/v (12 B, standard). Use the latter for real runs.
