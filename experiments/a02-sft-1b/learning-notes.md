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

## Segment 4b — activation functions (the gap: we'd never named sigmoid/ReLU)

The learner noticed we had never once covered the activation function. Real gap,
and it sits right before backward (Seg 6) because the backward pass has to travel
through it. Filled here as a Seg 4 supplement (it belongs to the forward path).

**First, two things both called "activation" — don't conflate:**
1. **activation** (the value) — the intermediate result each layer produces in the
   forward pass (the per-layer vectors before `logits` in Seg 4; the activation-
   memory term in A1).
2. **activation function** (sigmoid/ReLU) — the *function* that produces it.
The function (2) is applied to produce the value (1).

**Why activation functions exist (the core why):** linear regression
`y = w1*x1 + w2*x2 + b` is purely linear. Stacking linear layers collapses back to
a single linear layer:

```
layer 1:  h = W1 * x + b1
layer 2:  y = W2 * h + b2
sub in:   y = W2*(W1*x + b1) + b2 = (W2*W1)*x + (W2*b1+b2)  ==  one linear layer
```

100 stacked linear layers still equal one `y = matrix*x + number`. So depth would
be pointless — the net could only ever represent straight lines/planes. An
activation function inserts a **non-linear kink** between layers and breaks that
collapse:

```
layer 1:  h = ReLU(W1 * x + b1)   <- the kink; no longer purely linear
layer 2:  y = W2 * h + b2
```

Now stacking does NOT collapse, and the net can learn non-linear patterns (language,
images). **That is the entire reason activation functions exist: inject
non-linearity so depth means something.** Without it: collapses to one linear layer,
learns only lines.

**The two most common, as code (terminal-safe):**

```python
def relu(x):     return max(0, x)            # neg -> 0, pos -> unchanged (a folded line)
def sigmoid(x):  return 1 / (1 + exp(-x))    # squashes any real into (0,1), an S-curve
```

| | ReLU | sigmoid |
|---|---|---|
| shape | folded line (neg flattened, pos at 45°) | smooth S |
| output range | 0 .. +inf | 0 .. 1 |
| used much now? | yes, default in deep nets | mostly legacy / output layers |
| why ReLU won | fast, avoids vanishing gradient | gradient gets tiny when deep |

**Connections to A2 and backward:**
- A2's Llama doesn't use these two — it uses **SwiGLU** (a ReLU-family gated variant,
  taught in Track B13). Same principle: a non-linear kink between layers. ReLU is the
  simplest member; SwiGLU is "fancier ReLU."
- **Backward travels through it.** When the gradient flows back and hits a ReLU it
  multiplies by ReLU's slope: slope = 1 where input was positive (gradient passes
  through), 0 where input was negative (gradient is killed). This is the source of
  the `mask = [0, 1]` step in `../a01-mem-budget/backprop-primer.md` Step C.

### Q&A folded in

- **"We never covered the activation function — sigmoid or ReLU."** — Correct, real
  gap. Filled: activation functions inject non-linearity; without one, stacked
  linear layers collapse to a single linear layer and the net can only learn lines.
  ReLU = `max(0,x)` (the modern default), sigmoid = the 2015 S-curve. A2's Llama
  uses SwiGLU (a ReLU-family variant, Track B13).
- **Confirmed:** "activation function injects non-linearity so the network can learn
  non-linear things." ✓ (plus the "without it → collapses to one linear layer" half).

### Seg 4b follow-up — "why not y = w1*x^2 + w2*x + b? that's non-linear too"

A genuinely good design question (why this design, not that one). The learner's
intuition is *correct*: `x^2` IS non-linear, it does break the linear collapse. The
issue isn't "does it work" but "is it general / trainable." Three layers of answer:

1. **It does inject non-linearity.** `x^2` makes a parabola, not a line. As "break
   the collapse," it succeeds. So the question is how it compares to ReLU, not
   whether it's wrong.

2. **A fixed `x^2` is a hard-coded kink — it can only ever bend into ONE shape.**
   With `y = w1*x^2 + w2*x + b`, the non-linearity lives in the *fixed* function
   `x^2`; training only scales/shifts the parabola, never changes that it's a
   parabola. ReLU's non-linearity is a trivial kink `max(0,x)`, but **stacking many
   ReLUs composes into ANY shape.** Analogy: `x^2` is a fixed-shape ruler (parabolas
   only); ReLU is Lego (one boring fold per piece, but thousands compose into any
   curve). Formally: the **universal approximation theorem** — enough ReLU/sigmoid
   units approximate *any* continuous function; `x^2` can only ever be `x^2`.

3. **Why the "dumbest" kink (ReLU) wins over a smarter `x^2`:**
   - **Speed.** `max(0,x)` is one compare; `x^2` is a multiply. At 1.2B params and
     trillions of ops/s, that's real money (the learner's systems instinct).
   - **Stable gradient.** Backward passes through the activation. ReLU's slope is
     1 or 0 (constant — gradient can't explode or vanish). `x^2`'s slope is `2x`,
     which blows up when x is large → exploding gradients → training diverges.
   - **Net learns WHERE to bend.** ReLU's kink is fixed at 0, but the preceding
     `W*x+b` shifts/scales the input so the network *learns* where the kink lands
     and how steep. `x^2` is rigidly symmetric about the origin — less flexible.

> Core takeaway (learner's own words): `x^2` injects a **hard-coded shape** (always
> a parabola); a pile of ReLUs injects non-linearity that, **with enough units,
> approximates any continuous function**. This is deep learning's central design
> philosophy: **many simple units + depth, not a few complex units.**

- **Q: "Why not use `x^2` as the non-linearity?"** — It works but is a fixed shape
  (parabola only); ReLUs compose to any continuous function (universal
  approximation), are faster (one compare vs a multiply), have a stable 0/1 slope
  vs `x^2`'s exploding `2x` slope, and let the network learn where to bend. Simple
  units × depth beats one complex unit.

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


## Segment 6 — from loss to updated parameters: backward + optimizer

We have `loss` (one number, how wrong this step was). Seg 6 is the last piece: how
that number changes all 1.236B parameters so next time is less wrong. Two beats:
**6a/6b backward** (compute how each parameter should change) and **6c optimizer**
(actually change them).

### Seg 6a — backward gives each parameter a gradient ("which way to nudge it")

Start with ONE parameter, not 1.2B. Say `w = 0.5`, this step's `loss = 2.0`.
Question: do I make `w` bigger or smaller to lower loss? Backward answers it by
computing a **gradient** `dloss/dw` ("loss's gradient w.r.t. w"). Plain meaning:

> gradient = "if I increase w a tiny bit, does loss go up or down, and how fast?"

- gradient **positive** (+3): "w up -> loss up" -> so go the OTHER way, make w smaller
- gradient **negative** (-3): "w up -> loss down" -> go with it, make w bigger

Loss wants to fall, so you move OPPOSITE the gradient — that's the minus sign in the
update rule (Seg 6c uses it):

```
new_w = old_w - learning_rate * gradient
```

This is exactly 3Blue1Brown Ch3/4: `dloss/dw` is "this weight's influence on loss";
the hill picture = gradient points uphill, we step downhill (opposite).

- **Confirmed:** gradient = -5 (negative) means "w up -> loss down", so to lower
  loss, make w **bigger**. ✓

### Seg 6b — backward across all layers = chain rule, flowing loss -> input

Now: the net has many layers and 1.2B parameters. How does backward get a gradient
for every one? One word: **chain rule** (what you saw in 3B1B Ch4). Minimal example:

```
x  ->  [layer1: h = ReLU(W1*x + b1)]  ->  [layer2: y = W2*h + b2]  ->  loss
```

`W1` is several steps from loss (through h, y). Chain rule multiplies local slopes
back from loss:

```
W1's gradient = (loss's influence on y) * (y's influence on h) * (h's influence on W1)
                |---- step back one layer at a time, multiply local slopes ----|
```

The "back" insight: compute the layer nearest loss FIRST (layer2's gradient), then
pass that result back to the previous layer (layer1) to reuse — don't recompute.
Gradient flows like water from the loss end toward the input end. At each layer it:
1. uses the incoming gradient + that layer's saved activation to compute THIS
   layer's parameter gradient (stored for the optimizer);
2. passes a gradient one more layer back.
Crossing a ReLU: positive-input positions pass the gradient through, negative-input
positions kill it to 0 (the `mask` from Seg 4b / backprop-primer Step C).

**Why it MUST go loss -> input (not input -> loss):** you need `loss` before you can
know "loss's influence on" any layer, and loss is at the far end. Standing at the
input end you don't yet know what loss is, let alone its influence on anyone — so
the only possible direction is loss-end -> input-end, backward. That's the "back".

- **Confirmed:** "must compute loss first to know its influence on each layer, so
  compute from the loss end back toward the input." ✓ — the root of "back"prop.

## Segment 6c — the optimizer (AdamW) actually updates the parameters

Backward gave every parameter a gradient. The optimizer's whole job: use those
gradients to actually update the parameters.

**Simplest optimizer — SGD (already understood from 6a):**
```
new_w = old_w - learning_rate * gradient        # step opposite the gradient
```

**AdamW = SGD plus two improvements — and they ARE the m and v from A1's memory bill.**
Now we know what those 8 bytes/param are for:

- **m (momentum)** — a running average of recent gradients. Like a ball rolling
  downhill with inertia: consistent directions accelerate, jittery gradients cancel
  out -> smoother, faster. (Andrew Ng C2W2L06.)
- **v (adaptive step size)** — a running average of gradient *squared*, used to
  scale the step per-parameter. Consistently large gradients -> smaller steps
  (sensitive, do not overshoot); consistently small -> bigger steps. Each parameter
  gets its OWN step size, not one global lr. (Ng C2W2L07 / RMSProp.)
- **Adam = momentum (m) + RMSprop (v)** combined (Ng C2W2L08).
- **The W in AdamW** = weight decay: each step also pulls parameters slightly toward
  0 (regularization, fights overfitting). "Adam + shrink weights a touch each step."

**Closes the A1 loop.** The 12 bytes/param now fully explained, each with a job:
```
(1) parameter value  2 B   <- itself (needed for inference too)
(2) gradient         2 B   <- from backward (Seg 6a/b), train only
(3) m (momentum)     4 B   <- AdamW, running avg of gradient, train only
(4) v (adaptive)     4 B   <- AdamW, running avg of gradient^2, train only
                    ----
                    12 B
```
AdamW must persistently store m and v (the 8 bytes) because they are per-parameter
running averages updated every step — that is the origin of A1's "AdamW = 8
bytes/param optimizer state."

### The whole A2 training step, one picture (all the pieces)

One step (repeat 125x = A2):
```
1. forward:   token -> layers (W*x+b, ReLU) -> logits -> softmax -> probabilities  [Seg 4, 4b]
2. loss:      probabilities vs one-hot true answer -> cross-entropy -> -log(p)      [Seg 5]
3. backward:  flow back from loss, chain rule -> a gradient per parameter           [Seg 6a, 6b]
4. optimizer: AdamW uses gradient + m + v -> update each parameter; zero grads      [Seg 6c]
```
From one "instruction + answer" to all 1.236B parameters nudged one step.

- **Confirmed:** training 3B costs ~36 GB (3e9 x 12); inference costs ~6 GB
  (3e9 x 2, value only); the extra 30 GB is gradient + m + v. (learner's own
  derivation) ✓

## Segment 6d — floating-point formats (the gap under "why fp32 for m/v")

The learner asked why m/v need fp32, then correctly insisted we first explain how
fp16 actually works ("it is completely different from float"). LLM-engineering
bedrock — all of A1's byte counts and mixed precision rest on it.

**Big idea first: floats are NOT evenly spaced on the number line. Dense near 0,
sparse for large values.** With `double` precision is so high you never feel this;
fp16's low precision makes it glaring.

A float = sign + **exponent** (magnitude, the 2^e) + **mantissa** (significant
digits, the 1.xxx). Like scientific notation +/- 1.xxx x 2^e:

| format | C# name | bits | exponent | mantissa | sig. decimal digits | max value | used for |
|---|---|---|---|---|---|---|---|
| fp64 | `double` | 64 | 11 | 52 | ~15-16 | 1e308 | sci-computing; ML ~never |
| fp32 | `float`  | 32 | 8  | 23 | ~7     | 1e38  | ML high-precision tier: m, v, master |
| fp16 | (none)   | 16 | 5  | 10 | ~3-4   | 65504 | half precision; narrow both ends |
| bf16 | (none)   | 16 | 8  | 7  | ~2-3   | 1e38  | the half precision A2 actually uses |

**Anchor:** C#'s `float` IS fp32; `double` IS fp64. ML just uses the fp32 naming.

**Precision is relative, not absolute** (mantissa = fixed # of significant digits):
```
near 1:     resolves to ~0.001     (1.000, 1.001, ...)
near 1000:  resolves to ~1         (1024, 1025, ...)
near 65000: resolves to ~32        (adjacent representable values differ by 32!)
```
Bigger number -> wider gap between representable values. This is "big eats small":
`1024 + 0.5 = 1024` in fp16 because 0.5 falls in a gap fp16 cannot represent — not a
bug, fp16 simply has no fine ruler at that magnitude.

**fp16's other two quirks (both ends narrow, exponent only 5 bits):**
- max ~65504 -> large gradients overflow to `inf` -> loss becomes NaN -> training dies.
- min positive ~1e-4 -> smaller gradients underflow to 0 -> that update vanishes.

**fp16 vs bf16 — same 16 bits, OPPOSITE split (key):**
```
fp16:  exponent 5 (narrow range) + mantissa 10 (decent precision)
bf16:  exponent 8 (full fp32 range) + mantissa 7 (worse precision)
```
bf16 keeps fp32's 8-bit exponent -> range to 1e38, ~never overflows; pays with fewer
significant digits. In training, "enough range to not overflow" beats "one more
digit," so modern LLM training (A2 included) uses **bf16**, not fp16. (bf16 = "brain
float", Google's DL design — that is the B. Explains `bf16=True` in the A2 script.)

### Why m/v need fp32 — and the learner's two research-grade questions

**Why m/v use fp32:** not because the *values* need precision, but because m/v do
**long-running accumulation**: m_new = 0.9*m_old + 0.1*gradient, every step, for
thousands of steps. Once m is sizable, each step's tiny increment gets eaten by "big
eats small" in fp16/bf16 (2-4 sig digits) -> the update silently vanishes -> training
quietly fails. fp32's 7 sig digits keep the small increments. gradient (2) can be
bf16 because it is used-then-discarded, never accumulated. parameter value (1) is
bf16 in the 12 B recipe, or gets an fp32 **master copy** in the 16 B recipe — exactly
A2's measured 12-vs-16 mystery, now closed: the fp32 master stops small updates being
eaten by a bf16 weight.

**The judge for "can this quantity go low precision?" = does it accumulate?** The
learner nailed this. Used-then-discarded (gradient, activation) -> low precision fine,
any scaling error is one-shot. Long-accumulated (m, v, master) -> needs fp32 or the
error snowballs.

**Learner's two independent insights (research-grade — keep):**

1. **"Compress optimizer state to save memory."** This IS 8-bit Adam (bitsandbytes
   `adamw_8bit`, the `2 bytes/param` row in A1) — block-wise quantization of m/v to
   8 bits. Correction the learner needed: memory is saved by **fewer bits**, NOT by
   "limiting the range." A fp32 number is 4 bytes whether its value is 3 or 3000;
   clipping the range saves nothing. normalize/scaling is the *enabler* that makes
   low bit-width accurate (256 levels over [-1,1] instead of [-1000,1000]), not the
   saving itself. Caveat for this box: bitsandbytes 8-bit optimizer may lack an
   aarch64+sm_121 wheel (A7 risk).

2. **"Keep m/v in fp16's comfort zone so low precision suffices."** This IS the core
   idea of **loss scaling / mixed precision** — control the magnitude so a low-precision
   format is accurate. Real, standard NVIDIA practice. BUT applied to **gradient and
   activation** (used-then-discarded -> scale once, low risk, big win), NOT to m/v:
   m/v accumulate AND their magnitude is dynamic (gradient magnitude swings orders of
   magnitude across steps/layers), so pinning them in a comfort zone needs complex
   per-block dynamic scaling whose cost exceeds saving 2 bytes — fp32 is simpler/safer.
   So the instinct was right and IS used by industry — just where it pays
   (gradient/activation), not where it does not (m/v).

- **Confirmed:** training chose bf16 over fp16 for the larger range (avoids overflow). ✓
- **Confirmed:** the magnitude-control trick is used on gradient (used-then-discarded,
  scaling error does not accumulate), not on m/v (long accumulation -> error drifts). ✓

## Segment 6e — what a single LLM neuron looks like (deepens Seg 2)

Triggered by the learner writing `y = swiglu(Wx+b)` for a neuron and asking: "if w
and b are both parameters, why do they each have their OWN gradient and m/v?" The
question exposed a fused mental image of "one neuron" vs "a whole layer." Fixing the
picture makes the gradient question answer itself.

**Correction 1 — the activation function is NOT inside a neuron.** A single neuron
is just a dot product plus a bias, purely linear:
```
one neuron:        z = w1*x1 + w2*x2 + ... + wn*xn + b      (no non-linearity here)
activation:        applied to the WHOLE layer's output afterwards -> ReLU/swiglu(layer output)
```
swiglu/ReLU acts on the whole layer's output vector, not per-neuron. (SwiGLU
internally uses several W matrices for gating — more complex than ReLU, but that's
Track B13. For now: activation wraps the layer output, not the neuron.)

**Correction 2 — a neuron is n weights + 1 bias, each an independent parameter.**
A neuron taking 4 inputs:
```
z = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b
```
| parameter | kind | count |
|---|---|---|
| w1, w2, w3, w4 | weight | 4 |
| b | bias | 1 |
| **one neuron total** | | **5 parameters** |

A neuron is NOT "1 w and 1 b" — it's a *string* of w (as many as inputs) + 1 b, and
every single `w_i` and the `b` is its own independent parameter.

**Why each w, each b has its own gradient/m/v — gradient is PER-PARAMETER.** A
gradient is by definition "loss's partial derivative w.r.t. THIS one number" (Seg
6a). Each parameter affects loss differently (w1 multiplies x1, w2 multiplies x2 —
different influence paths), so each gets its own gradient number. m and v are running
averages of the gradient, so they're per-parameter too:
```
parameter   value    gradient       m       v       bytes (12B recipe)
w1          0.5      dloss/dw1      m_w1    v_w1     12
w2         -0.3      dloss/dw2      m_w2    v_w2     12
w3          0.8      dloss/dw3      m_w3    v_w3     12
w4          0.1      dloss/dw4      m_w4    v_w4     12
b           0.05     dloss/db       m_b     v_b      12
                                                    ----
one neuron, 5 parameters, training cost            60 bytes
```
Every row is an independent parameter with its own complete (value, gradient, m, v).
Nothing is shared — that's why A1's memory is `param_count × 12`: the 12 bytes is
per-parameter, w or b alike.

**Scale to an LLM:** a layer has many neurons (e.g. 3072), each a string-of-w + one
b. Stack all neurons' w as rows -> the matrix `W` from Seg 4b; stack the b's -> the
bias vector. Llama-3B's 3.2B parameters are neurons stacked into layers, but the
atomic unit is always "one number = one parameter, each with its own (value,
gradient, m, v)."

- **Confirmed:** the 4-input neuron has 5 parameters -> 5 independent gradients,
  5 m's, 5 v's. ✓

---

## A2 lesson complete — learner diagnostic (strengths, weak spots, how to teach)

This learner worked through A2 by asking questions one segment at a time. Reviewing
ALL their questions reveals a clear cognitive profile. Recorded so future sessions
teach to it.

### Strengths (lean on these)

- **Systems/precision instinct is excellent and transferable.** They independently
  re-derived 8-bit Adam ("compress optimizer state") and loss scaling ("keep m/v in
  fp16's comfort zone"), and produced the general rule "low precision is fine for
  used-then-discarded quantities, not for accumulated ones." This is graduate-level
  reasoning arriving from a systems background, not an ML one. Frame new ML memory/
  precision topics as systems trade-offs and they move fast.
- **Asks "why this design, not that one"** (the `x^2`-instead-of-ReLU question, the
  "limit the range" question). They don't accept a mechanism until they've probed
  the alternative. Reward this — give the alternative honest treatment, then the
  real reason it lost.
- **Self-corrects with the right anchor.** Once given linear regression as the
  anchor, they reason correctly from it. Concrete worked examples with real numbers
  land; abstractions don't.

### Weak spots (the recurring pattern — address these proactively)

1. **Umbrella-vs-kind / direction-of-containment confusion (the #1 recurring bug).**
   It showed up THREE times: (a) "does a parameter contain weight+bias?" (reversed —
   weight/bias are kinds OF parameter); (b) "model outputs a one-hot compressed to an
   integer?" (reversed — the long vector is the model output, the integer is the
   *answer* side); (c) "softmax outputs an integer?" (conflated softmax with the
   later argmax). **Pattern: they tend to invert which thing contains/produces which,
   and to fuse two adjacent stages into one.** Teaching fix: always state direction
   explicitly ("A is a KIND of B", "X is produced BY Y, not the reverse") and keep
   adjacent stages visibly separate.

2. **Fused mental images of nested scales.** "one neuron" got fused with "one layer"
   (`y=swiglu(Wx+b)` for a neuron); the activation function got fused into the neuron.
   Teaching fix: always pin the scale — neuron vs layer vs network — with an explicit
   count ("this ONE neuron has 5 parameters").

3. **Concepts present but fuzzy from a 10-year gap (2015 Andrew Ng).** one-hot,
   softmax, cross-entropy, Adam's m/v — all half-remembered, several half-swapped
   (one-hot<->softmax; cross-entropy<->MSE "truth minus prediction"). These need
   *re-awakening with the right name attached*, not building from zero. Connect to
   "the formula you half-remember" and name it.

4. **Notation/environment, not math:** can read `Σ` fine, but the terminal doesn't
   render LaTeX sub/superscripts — write math as code (`y[i]`, for-loops), never as
   rendered subscripts.

### How to teach this learner (the working method, proven over A2)

- One small segment, then stop and let them clarify in place. Big dumps lose them.
- Anchor every new idea in linear regression / real numbers / a tiny worked example.
- When they propose an alternative design, take it seriously, confirm what's right
  about it, then give the honest reason the standard choice won — they learn most
  from this.
- State containment direction and scale explicitly to pre-empt weak spot #1 and #2.
- Frame memory/precision as systems trade-offs (their home turf).
- Their extension questions are higher-value than the core lesson — budget time for
  them; that's where the deep understanding (and the reward) actually happens.
