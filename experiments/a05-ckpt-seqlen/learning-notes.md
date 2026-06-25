# A5 — learning notes (activation checkpointing + seq_len, learner-paced)

> Same per-learner format as A2/A4. Honor all three: complete coverage; depth set by
> THIS learner's familiarity (compress what they know, expand what they don't); phrased
> the way that worked in dialogue (linear-regression / systems anchors, explicit
> direction+scale, math terminal-safe). Pacing: small segment -> clarify -> fold Q&A ->
> next. Read top to bottom to replay the lesson.

---

## What A5 teaches (one line)

**Activation checkpointing trades RECOMPUTE (time) for ACTIVATION MEMORY (space)** — and
the saving scales with seq_len. Plus the two prerequisites this learner needed: what
seq_len is, and the minimum transformer picture (input is a 2-D matrix of token vectors).

---

## Segment 0 — calibration: the learner derived the whole trade themselves

Three priming questions, all answered correctly:
- **Q1: why does activation cost so much memory — why not drop each layer's activation
  after computing it?** Learner: "gradient needs the previous layer's activation." Correct,
  sharpened: backward for layer L needs that layer's FORWARD INPUT x (`dloss/dw = x *
  dloss/dz`), and forward runs front-to-back while backward runs back-to-front — so layer
  1's x is computed early but consumed late, forcing ALL layers' activations to stay
  resident through the whole forward. That pile = activation memory.
- **Q2: if you could re-compute activations in backward instead of storing them, what's
  the trade?** Learner: "time for space." Exactly — the one-line essence of checkpointing.
- **Q3: why not drop ALL activations and recompute everything?** Learner: "compute cost
  too high." Correct — so real checkpointing is a compromise: keep checkpoints at block
  boundaries, drop the rest, recompute on demand. Save fraction tops out at k (~30-50%),
  never 100%.

## Segment 1 — the gradient formula (what data a single w's gradient needs)

Learner asked to review: for one weight w, what does computing its gradient need? Derived
on `z = w*x + b`:
```
dloss/dw = dloss/dz * dz/dw = dloss/dz * x
           ^^^^^^^^^         ^
           upstream grad      x = THIS layer's forward INPUT activation
           (from behind)
```
Two factors: (1) `x`, the forward-input activation (stored in forward); (2) `dloss/dz`,
the gradient flowing back. **The `x` factor is exactly the activation checkpointing drops
and must recompute** — without it, `dloss/dw = x * dloss/dz` can't be evaluated. (Also the
ReLU mask needs z, another stored forward value — "backward needs forward intermediates"
is general, not just x.)

### Seg 1 follow-up — `dloss/dw` vs `dloss/dz`: same operator, different quantities

Learner asked the sharp question: are `dloss/dw` and `dloss/dz` the same thing (same
symbol, same meaning)? Answer: same OPERATOR ("loss's sensitivity to ___"), DIFFERENT
quantities — different variable, value, and use.

| | `dloss/dw` | `dloss/dz` |
|---|---|---|
| differentiate wrt | parameter w (you train it) | activation z (an intermediate) |
| value | `dloss/dz * x` | the upstream grad passed back |
| use | UPDATE w (`w -= lr*dloss/dw`) | a CHAIN-RULE relay, passed one layer further back |
| after backward | kept (feeds optimizer) | discarded (use-then-throw) |

Picture (ties to A2 Seg 6b "gradient flows like water"): `dloss/dz` = the flow PASSING
THROUGH point z; `dloss/dw` = the branch SPLITTING OFF to parameter w. Same stream,
different points, different values.

### Seg 1 follow-up 2 — the per-parameter gradient structure, stated cleanly

Learner restated: "w's gradient = previous layer's output activation * next layer's
gradient." Right multiplicative SHAPE (one activation * one passed-back gradient), but two
words mislocated — corrected to attribute both factors to w's OWN layer:
```
WRONG: prev layer's output activation * next layer's gradient
RIGHT: THIS layer's INPUT activation * THIS layer's OUTPUT gradient
       dloss/dw = x * dloss/dz
```
(The input x does equal the previous layer's output numerically, but anchor it as "what w
multiplies in z=w*x+b", which stays unambiguous in branched nets.) Takeaway: forward info
arrives from the front (x), gradient info from behind (dloss/dz), they MEET at w and
multiply. That meeting is the whole picture of backprop.

## Segment 2 — seq_len (the learner had no prior contact with it)

Gap surfaced: learner's NN background = "input is one value / one vector." seq_len didn't
fit. Filled:
- **seq_len = number of tokens processed at once** = the length of the token sequence
  (A2 Seg 4's text->token chain, but how MANY tokens).
- Orthogonal to batch: batch = how many samples (rows); seq_len = how long each sample
  (columns). One forward processes a `batch x seq_len` block of tokens.
- It's the `--seq-len 1024` in the A2/A4 scripts (every sample truncated/padded to that).
- Activation `∝ batch * seq_len * hidden * layers` — seq_len scales activation memory
  linearly, which is why A5 sweeps it.
- Real-world anchor: the "context window" (128k) when using Claude/GPT IS the seq_len
  ceiling; long context is expensive because activation (training) and KV cache
  (inference) both grow with seq_len.

## Segment 3 — minimum transformer (input is a 2-D matrix of token vectors)

Learner correctly sensed their old model ("input is one vector -> one output") didn't fit
"a sequence of vectors" and asked for it. Built from their existing model:
- **One token = one vector** (embedding: token ID -> a `hidden`-dim vector, e.g. 3072 =
  the 3B hidden_size).
- **A sentence = a row of vectors** = a `[seq_len, hidden]` 2-D matrix. The old model is
  the special case seq_len=1.
- **Why a whole row at once:** attention lets each token "look at" all others to resolve
  meaning ("bank" looks at "river"). That REQUIRES the whole sequence resident at once —
  it's a mechanism requirement, not an optimization. (The old single-vector model can't do
  this; tokens never communicate.)
- **A transformer layer = attention (tokens exchange info) + MLP (the learner's OLD network,
  applied per-vector).** So their prior NN experience IS half a transformer (the MLP half);
  attention is the new half.
- Learner's own confirmation: "input is a 2-D matrix, fed in all at once?" -> YES. Nuance
  flagged: TRAINING feeds the whole `[batch, seq_len, hidden]` tensor at once; INFERENCE
  generates one token at a time but the whole existing prefix stays resident for attention
  (= the KV cache). Learning training now -> "all at once" is literally true.
- Decision: this is the "just-enough" transformer; the full attention build is Track B4.
  Did NOT rabbit-hole.

## Segment 4 — P1/P2 predictions before the run (the learner nailed P2 with algebra)

- **P1: OFF row, seq 512->4096 (8x), what happens to activation memory?** Learner: "~8x."
  Correct (activation ∝ seq_len). Sharpened: the 8x is on the ACTIVATION part only;
  peak_mem TOTAL rises less than 8x because the fixed part (frozen 3B base ~6.5GB +
  grad/optimizer on the tiny LoRA params) doesn't scale with seq_len.
- **P2: why does checkpointing's SAVE % rise with seq_len?** Learner produced the algebra
  himself:
  ```
  save% = (k * A) / (F + A)         F = fixed part, A = activation = c*seq_len, k = drop fraction
        = k / ( F/(c*seq_len) + 1 )
  seq_len -> small: F/(c*seq_len) large -> save% small (fixed part dominates)
  seq_len -> large: F/(c*seq_len) -> 0  -> save% -> k (activation dominates)
  ```
  This is the precise algebraic form of the day-plan's "the save scales with seq_len" —
  the learner derived it as a monotone function with limit k. (k<1 because block-boundary
  checkpoints stay — his own Q3 prediction.)

<!-- Segment 5 (the measured 2-D table + verdict) appended after the sweep -->

## Segment 5 — the measured 2-D table, both predictions confirmed

```
 seq_len |   ckpt OFF      |    ckpt ON      | mem saved | time cost
--------------------------------------------------------------------
     512 |  13.7G /  592ms |   8.2G /  796ms |   39.9%   |  +34.3%
    1024 |  21.0G / 1354ms |  10.1G / 1826ms |   52.0%   |  +34.8%
    2048 |  36.0G / 3063ms |  13.9G / 4085ms |   61.4%   |  +33.4%
    4096 |  66.5G / 7273ms |  21.5G / 9634ms |   67.7%   |  +32.5%
```

**P2 (the headline) — CONFIRMED, beautifully.** save% rose MONOTONICALLY
39.9 → 52.0 → 61.4 → 67.7% — exactly the `k/(F/(c·seq_len)+1)` curve the learner derived.
time-cost held ~+33% (recompute = one fixed extra forward, constant share). Save exceeds
the day-plan's 30–50% at large seq_len because activation's share keeps growing toward k.

**P1 — CONFIRMED.** OFF peak 13.65 → 66.51 GB over seq ×8 = only 4.87×, not 8×, because
the fixed part doesn't scale. Two-point solve of `peak = F + c·seq_len` gives F ≈ 6.2–7.0
GB ≈ the frozen 3B base (3.2B×2 ≈ 6.4 GB). The learner's algebra reverse-solves the
physical fixed cost from the measured data — model wasn't just directionally right, its
parameters map to real bytes.

**Beyond the prediction — the practical "aha":** at seq=4096 OFF needs 66.5 GB (OOM on a
24/40 GB card); ON needs 21.5 GB (fits). Checkpointing isn't "save a little memory" — it's
the switch that turns an impossible config into a runnable one, pushing the "what can this
card train" boundary open (ties to talk slide 9). On GX10's 128 GB it still fits; on a
normal GPU it's the difference between can and can't.

**Scorecard:** every prediction the learner made (P1 magnitude + dilution, P2 monotone rise
+ limit, the constant time-cost) matched the metal. The algebra he wrote unprompted (`save%
= k/(F/(c·seq_len)+1)`) is the exact closed form of the day-plan's one-line "save scales
with seq_len."
