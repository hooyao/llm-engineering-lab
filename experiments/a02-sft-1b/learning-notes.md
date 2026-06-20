# A2 — learning notes (one small piece at a time)

> **What this file is — a teaching note CUSTOMIZED for this one learner.**
> Not the traditional "same explanation for everyone" note. Three things make it
> different, and any session continuing it must honor all three:
>
> 1. **Complete coverage.** It contains ALL of A2's knowledge — a standalone review
>    document, not just scattered Q&A snippets. Someone could relearn A2 from this
>    file alone.
> 2. **Depth set by THIS learner's familiarity.** Things the learner doesn't know
>    well (e.g. tokens, CLM, how loss is computed) are explained in detail. Things
>    they already know (their .NET/systems background, the linear regression they've
>    written) are stated in one line and moved past. The goal: on review, attention
>    lands on the unfamiliar, without slogging through what's already understood.
> 3. **Explained in the way THIS learner understands best.** Not a generic textbook
>    explanation — phrased for their knowledge structure and mental model. *What
>    that best way is gets discovered through our teaching dialogue itself* — how
>    they ask, which framing makes it click, where they get stuck. As the lesson
>    proceeds, write each concept the way that worked for them in conversation.
>
> Pacing (how it's built): one small segment → learner clarifies in place → fold
> the explanation + their Q&A back in here → next segment. The `notes.md` in this
> folder holds A2's results/payoff; `../a01-mem-budget/teaching-notes.md` holds the
> general gap-concepts. THIS file is the per-learner, dialogue-shaped A2 lesson.
>
> Read top to bottom to replay the lesson at the right depth for this learner.

---

## What A2 teaches (the whole point, in one line)

The end-to-end **SFT (supervised fine-tuning) training loop** — from one
"instruction + answer" text to the model's weights being updated one step — every
link in between.

We build up to that one small piece at a time below.

---

<!-- segments appended here as the lesson proceeds -->

## Segment 1 — what A2 does, and what "fine-tuning" means

A2 does exactly one thing, repeated: take one "instruction + answer" text, feed it
to the model, and nudge the model's `parameter`s one step toward "answering this
kind of thing better." Repeat many times (125 times in our run) = training.

**Fine-tuning is NOT training from zero.** Llama-3.2-1B is *already* trained — it
already speaks English and knows Paris is the capital of France. Fine-tuning =
*adjust an already-capable model* with our own data. The move is:
`existing model + our data → adjusted model`. Not "build a new model."

> (This learner has written linear regression, so "adjust parameters to make the
> output better" is already familiar — fine-tuning is that same act, applied to a
> model that already knows a lot.)

## Segment 2 — parameter is the umbrella term; weight & bias are kinds of it

This is the segment that cleared up the central confusion. The relationship is
**not** "a parameter contains a weight and a bias." It's the other way around:

> **A `weight` is one kind of `parameter`. A `bias` is another kind. Each single
> weight is itself one parameter; each single bias is itself one parameter.**

`parameter` is the umbrella term; `weight` and `bias` are kinds under it — like
"fruit" is the umbrella and "apple"/"orange" are kinds. You wouldn't say "an apple
contains fruit"; you say "an apple is a kind of fruit." Same here: not "a parameter
contains a weight" but "a weight *is* a parameter."

Counted on the familiar linear regression `y = w1*x1 + w2*x2 + b`:

| the number | its kind | how many parameters |
|---|---|---|
| `w1` | a weight | 1 parameter |
| `w2` | a weight | 1 parameter |
| `b`  | a bias   | 1 parameter |

→ **3 parameters total**: 2 of kind weight, 1 of kind bias.

So **"full-parameter" is the *precise* phrasing**, more precise than "all weights":
- `full-parameter` SFT = *every trainable number* updates (weights AND biases).
- "all weights update" = imprecise, drops the biases (and any other parameter kind).

**full-parameter vs LoRA:** full-parameter = all parameters are trainable and may
update. LoRA = freeze the original parameters, train only a tiny added slice. A2
removed the LoRA wrapper, so all 1.236B parameters are trainable (the log confirmed
`1.236B / 1.236B (100%)`).

### Q&A folded in

- **"full-parameter SFT means all weights may update?"** — Yes. And not only
  weights: biases too. Every parameter (weight + bias + other kinds) may update.
- **"Is 'full-parameter' imprecise — shouldn't I say weights?"** — Reversed:
  `parameter` is the *most* precise word, because it's the umbrella that already
  includes weight and bias. "All weights" is the imprecise one.

## Segment 3 — the "12 bytes" is a SEPARATE axis from weight-vs-bias

Two different questions were getting stacked. Keep them apart:

- **Axis A — what KIND is this parameter?** weight or bias. (Segment 2.)
- **Axis B — to TRAIN one parameter, how many bytes of memory does it cost?**
  This is A1's "12 bytes." (Note: **bytes**, not bits — 1 byte = 8 bits.)

The 12 bytes is **not** "12 bytes of different things packed inside one parameter."
It's: to train, every parameter — *whether weight or bias, treated identically* —
needs 4 things held in memory:

| stored per parameter (fp16/bf16 recipe) | bytes |
|---|---|
| ① the parameter's own value (BF16) | 2 |
| ② its gradient (BF16) | 2 |
| ③ AdamW `m` (FP32) | 4 |
| ④ AdamW `v` (FP32) | 4 |
| **total** | **12** |

**12 is not fixed — it depends on the precision** (the learner correctly insisted
on the "fp16 case" qualifier). Rows ① and ② are 2 bytes *because* the value and
gradient are stored in half precision; store them in fp32 and they become 4 bytes
each → 16 bytes total. That's why A1 has both 12 and 16 (and why A2's measured
13.84 GB matched the 12-byte recipe — HF `Trainer(bf16=True)` uses it).

Putting both axes together:
- `w1` is a **weight**, which is a **parameter**, which costs **12 bytes** to train.
- `b` is a **bias**, which is also a **parameter**, which also costs **12 bytes**.

So the linear regression (3 parameters), full-parameter trained, = `3 × 12 = 36
bytes`. Llama-3B's 3.2e9 parameters (mostly weights, some biases) = `3.2e9 × 12`.

> **"12 bytes" answers "how expensive is one parameter to store"; "weight/bias"
> answers "which kind is this parameter." Two independent questions — don't stack
> them.**

### Q&A folded in

- **"Does a parameter (the 12-byte thing) contain weight, bias, and other stuff?"**
  — No. Direction confused. The 12 bytes is storage cost; weight/bias is the kind.
  A single bias is one parameter and also costs 12 bytes — same as a weight.
- **"bias is a parameter, and in fp16 it's 12 bytes?"** — Correct, and the "in
  fp16" qualifier is exactly right: 12 is the half-precision recipe; fp32 → 16.

## Segment 4 — the data pipeline: text → token → ID → logits → softmax → probabilities

This learner already had the high level (c-level): a tokenizer cuts text into
chunks (tokens), a transformer emits one token at a time (autoregressive), tokens
are the unit during training. All correct — stated, not re-taught. What needed
detail was the chain *underneath* "emit a token," and three reversed intuitions got
corrected along the way (those corrections are the valuable part to review).

**The full chain (this is A2's spine — recognize it in any training code):**

```
text          →  tokens          →  token IDs   →  [model] →  logits        →  softmax →  probabilities
"Hello world." → ["Hello"," world","."] → [9906,1917,13]        [2.1,0.5,...]            [0.08,0.02,...]
                  (chunks)          (integers,           (~120k-long       (~120k-long, each 0–1,
                                     via vocab table)     vector)           sums to 1)
```

Step by step:

1. **text → tokens**: tokenizer splits into chunks. (Known.)
2. **tokens → token IDs**: each token chunk is looked up in a fixed table
   (vocabulary) and replaced by an **integer ID**. The model never sees text; it
   processes the integer sequence `[9906, 1917, 13]`. ("Encoding" = this lookup.)
3. **model → logits**: the model's last layer emits a **real ~120k-long vector**,
   one number per token in the vocabulary. Name: **logits**. Raw scores, can be
   negative, don't sum to anything nice. This length is real, not a convenience.
4. **logits → softmax → probabilities**: softmax keeps the length (~120k) and just
   *tidies* the numbers: makes each one 0–1 and makes them sum to 1. Now readable as
   "probability that the next token is this one." Output is still a ~120k vector,
   **not an integer**.
5. (generation only) **argmax**: pick the highest-probability entry → one integer
   ID → that's the emitted token. **Training does NOT do this step** (see below).

### The three reversed intuitions that got corrected (most useful to review)

This learner kept landing on the right pieces but with the direction flipped. Each
correction is a keeper:

- **one-hot is NOT a probability / NOT the model's output.** one-hot
  (`0 0 0 1 0 0`) is the **true correct answer** (100% certain, from the training
  data). The model's output is the **probability distribution** (`0.05 0.1 0.7 …`,
  a guess). They are the two opposite ends that `loss` compares. They have the
  *same length* (~120k), which is why they're comparable — and why this learner
  associated them.
- **Which end is the long vector vs the integer.** The MODEL OUTPUT is the real,
  uncompressed ~120k vector (logits/probabilities). The TRUE ANSWER is
  conceptually a ~120k one-hot but is stored in code as a single **integer ID**
  (the index of the correct token) to save memory. The learner had these swapped:
  "model outputs a long vector" is right for the output end; "compressed to an
  integer for convenience" is right for the answer end — they'd attached each to
  the wrong side.
- **softmax does NOT produce an integer.** softmax: ~120k logits → ~120k
  probabilities (same length, values tidied to 0–1 summing to 1). Producing an
  integer is a *separate later* step (**argmax**), which the learner had merged
  into softmax. softmax tidies; argmax picks the winner.

### Why training stops at the probability distribution (bridges to loss)

- **Training:** logits → softmax → probability distribution → compare against the
  one-hot true answer to get `loss`. **No argmax** — picking would throw away the
  information loss needs (how much probability the model put on the *correct*
  answer, not just which it preferred).
- **Inference/generation:** logits → softmax → probabilities → **argmax** → emit
  one token. ("Transformer emits one token" = this path.)

So "emit one token" is the generation path; training halts one step earlier, at the
full distribution, precisely so it can be scored.

### Q&A folded in

- **"Is my understanding right — tokenizer chunks text, transformer emits one token,
  token is the training unit?"** — Yes, all three. Just made the under-the-hood
  chain explicit: token → integer ID → the model works on integers.
- **"I thought of one-hot — it ends up like `00010000`, isn't that a probability?"**
  — one-hot is the *true answer* (certain), not the model's probabilistic guess;
  they're the two ends loss compares, same length.
- **"Does the model internally have a one-hot? Is the output a 120k one-hot
  compressed to an integer for engineering convenience?"** — Output is a real 120k
  vector (logits→probabilities), NOT one-hot, NOT compressed. The
  conceptually-one-hot-stored-as-integer thing is the *true answer* end, not the
  model output end.
- **"softmax turns logits into an integer?"** — No. softmax → a 120k probability
  vector (same length, sums to 1). The integer comes from argmax, a separate step,
  and training doesn't even do it.
- **Confirmed:** "softmax output is a 120k-long vector whose entries sum to 1." ✓

## Segment 5 — loss = cross-entropy, and why it collapses to one `-log(p)`

> **Notation note (rendering, not comprehension):** the learner reads `Σ` fine —
> the issue is the Claude Code terminal does NOT render LaTeX sub/superscripts, so
> `Σ` with an under-index and `y_i` with a subscript come out as garbled bare
> letters. So: write math in plain-text-safe form — `y[i]` not `y_i`, "sum over i"
> or a `for` loop instead of Σ-with-subscript, inside code blocks. This is a
> terminal-rendering constraint, not a math-literacy one. They also have a 2015
> Andrew Ng Coursera ML background that's gone fuzzy, so this is *re-awakening*, not
> first-contact: connect to "the complicated loss formula you half-remember."

`loss` just measures how far the model's guess is from the true answer — the two
vectors Segment 4 already lined up:

```
model guess (softmax):  [0.05, 0.1, 0.05, 0.7, 0.03, ...]   (probabilities)
true answer  (one-hot):  [0,    0,   0,    1,   0,    ...]   (correct = index 3)
```

### The "complicated formula" (full cross-entropy) — written as a loop

The learner remembered loss as a scary formula. It is real; it just collapses.
Written the way that reads for them (a loop, not `Σ`):

```python
loss = 0
for i in range(120000):            # over every vocab position
    loss = loss + y[i] * log(p[i])
loss = -loss
```

- `y[i]` = true-answer vector at position i (one-hot: 1 at the correct token, else 0)
- `p[i]` = model's softmax probability at position i

(Correction the learner needed: cross-entropy is `y[i] * log(p[i])` summed —
*element-wise multiply then add* — NOT "truth vector minus model vector." The
subtraction one is MSE, used for regression. They'd merged the two.)

### Why it collapses to a single term

Substitute the one-hot `y` (correct = index 3, all others 0) into the loop:

```python
# i=0: y[0]=0 -> 0*log(p[0]) = 0      adds nothing
# i=1: y[1]=0 -> 0                    adds nothing
# i=2: y[2]=0 -> 0                    adds nothing
# i=3: y[3]=1 -> 1*log(p[3]) = log(p[3])   <-- the ONLY term that adds anything
# i=4..119999: y[i]=0 -> 0           adds nothing
loss = -log(p[3])
```

120 000 iterations, but every one except the `y[i]=1` term multiplies by 0. So the
whole sum is just:

```
loss = -log( model's probability on the correct token )
```

**The full ~120k-term formula collapses to one `-log(p)` because the true answer is
one-hot — 99.999% of the terms are multiplied by 0.** This is the "magical
simplification" Track C4 names.

### Why `-log(p)` is the right penalty

| p on correct token | -log(p) = loss | meaning |
|---|---|---|
| 1.0 | 0     | perfect, no penalty |
| 0.9 | 0.105 | tiny |
| 0.7 | 0.357 | small |
| 0.1 | 2.30  | large |
| 0.01| 4.61  | huge |
| →0  | →∞    | catastrophic |

Higher probability on the correct token → smaller loss. To shrink loss the model is
forced to put more probability on the correct next token — exactly what we want it
to learn.

### Back to A2's real numbers

loss 1.72 → 1.35 now means something concrete. Inverting `-log(p)`:
- loss 1.72 → p ≈ 0.18 (model gave the correct token ~18% on average)
- loss 1.35 → p ≈ 0.26 (~26% after training)

Training raised the model's average confidence in the *correct* next token from
~18% to ~26%. (A full sentence has many token positions; each contributes one
`-log(p)` and they're averaged — but each position is just this one log.)

### The learner's own insight (keeper) — three things are one thing

Unprompted, the learner said: "no point wasting compute on the 0 terms — just do
`-log(p[3])`." That IS how real frameworks work, and it ties three earlier points
into one fact:

| learned point | segment | |
|---|---|---|
| true answer stored as one integer ID | Seg 4 | because |
| cross-entropy collapses to the one correct term | Seg 5 | so |
| don't spend compute on the 0 terms | learner's insight | therefore |

PyTorch's `cross_entropy` takes the true answer as an **integer** (e.g. `3`), not a
vector: it indexes `p[3]` directly and computes `-log(p[3])`, touching zero of the
0-terms. One-hot collapses the math → only the correct index is needed → store one
integer, compute one term. The math simplification, the storage trick, and the
compute optimization are the same fact seen three ways. (The learner derived the
compute optimization themselves — systems instinct doing real work.)

### Q&A folded in

- **"Can you derive `-log(p)`? I remember a complicated formula, truth vector minus
  model vector."** — Full cross-entropy is `sum of y[i]*log(p[i])`, negated (multiply
  then sum, NOT subtract — subtraction is MSE). One-hot kills all but the correct
  term → `-log(p_correct)`. The "complicated" formula was right; it just collapses.
- **"I can't read the formula you rendered."** — Not a Σ-comprehension issue: the
  Claude Code terminal doesn't render LaTeX sub/superscripts, so `Σ`-with-index and
  `y_i` came out garbled. Re-expressed in plain-text-safe form (a `for` loop /
  `y[i]`). (Going forward: write math terminal-safe — `y[i]`, "sum over i", code
  blocks — never rely on rendered sub/superscripts.)
- **"Why only one term survives?"** — Because y[i]=0 for all but the correct
  position, and `0 * log(p[i]) = 0`; only the `y=1` term contributes. ✓ (learner's
  own words, plus the compute-saving insight above.)

