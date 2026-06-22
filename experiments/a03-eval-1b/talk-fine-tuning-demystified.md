# Talk: "Fine-tuning isn't magic — anyone can do it"

**Audience:** software engineers, no ML background (or only school-level ML).
**Goal / thesis:** fine-tuning is not mysterious; with a desktop box and an afternoon,
you can do it. Demystify, don't over-promise. Show real numbers from a real run.
**Tone:** peer engineers. Lead with bytes/numbers/demos, not theory. No backprop math.

> This file is the slide OUTLINE + talking points + the real numbers to show. It is
> the blueprint to turn into slides (the talk is delivered in 中文; this file is the
> English working material per repo convention). Every number here is from the
> learner's own A1/A2/A3 runs on the GX10 — that authenticity is the whole point.
> Keep ML terms in English on the slides too (weight/bias/gradient/parameter/...).

---

## Slide 0 — Title

**"微调没那么神秘 — 一台桌面设备 + 一个下午就能玩"**
(Fine-tuning isn't magic — a desktop box and an afternoon)

One line under it: *"I fine-tuned a 1B model in 83 seconds on a box on my desk. Let
me show you."*

---

## Slide 1 — What is fine-tuning (one slide, no jargon)

- A pretrained model (Llama, Qwen, ...) already knows language and facts. Someone
  spent millions of GPU-hours on that.
- **Fine-tuning = take that finished model + your own data → nudge its behavior.**
  NOT training from zero. You're adjusting, not building.
- Analogy-free framing: it's `existing model + your data → adjusted model`.
- What it changes: **style / format / how it responds.** What it (mostly) does NOT
  change: the underlying knowledge. (We'll SEE this in the demo.)

Talking point: "the scary part — pretraining — is already done and you didn't pay
for it. Fine-tuning is the cheap, accessible part."

---

## Slide 2 — THE DEMO (lead with this if you can — it's the hook)

Before vs after, same prompts, base 1B-Instruct vs the learner's 500-example SFT.
Pull 3-4 pairs from `experiments/a03-eval-1b/results.md`. Suggested picks:

- **#6 "make this more polite"** — BEFORE: preamble + 3 variants + closing paragraph;
  AFTER: one direct sentence. → fine-tuning made it *direct*.
- **#1 "three tips"** — BEFORE: "Here are three tips…" + wordy; AFTER: straight to
  "1. 2. 3." → dropped the preamble.
- **#5 "capital of France"** — BEFORE and AFTER **byte-identical** ("Paris"). →
  knowledge unchanged. THIS is the slide that proves "fine-tuning changes style, not
  facts."

Talking point: "I trained it on 500 examples for 83 seconds and you can already see
the personality shift. That's it. That's fine-tuning."

**Honesty slide-half (this builds credibility with engineers):** it also has a cost —
the fine-tuned model got *more concise but sometimes dropped content and followed
instructions less strictly* (show #10: lost the recipe steps). Fine-tuning is a
trade-off, not a free upgrade. (Engineers trust you more when you show the downside.)

---

## Slide 3 — The two ways: full fine-tuning vs LoRA

One comparison table, that's the slide:

| | Full fine-tuning | LoRA |
|---|---|---|
| What trains | ALL parameters | freeze the model, train a tiny added slice (~1%) |
| Memory | high (see next slide) | much lower |
| Result | a whole new model copy | a small "adapter" file (MBs) |
| When | small models, max quality | big models, limited memory, many variants |

Talking point: "Full = retrain everything, heavy. LoRA = bolt on a small trainable
piece, freeze the rest — same idea, fraction of the cost. LoRA is why people fine-tune
70B models on one GPU."

(Note for you: you've DONE full fine-tuning (A2); LoRA you know conceptually, run it
yourself in A6 before Friday if you want a live LoRA number — optional, not required.)

---

## Slide 4 — Memory math (THE engineer-bait slide — they'll love this)

The whole pitch: you can predict whether a model fits BEFORE you run anything. Just
arithmetic.

**The rule:** `memory ≈ number_of_parameters × bytes_per_parameter`

**Per parameter, training with Adam needs 4 things:**
```
(1) the parameter value      2 bytes   (bf16)
(2) its gradient             2 bytes   (bf16)
(3) Adam m (momentum)        4 bytes   (fp32)  <- optimizer state
(4) Adam v (adaptive step)   4 bytes   (fp32)  <- optimizer state
                            ---------
                            12 bytes per parameter
```

**Worked examples (show these as a table):**

| model | full fine-tune (×12) | inference only (×2) |
|---|---|---|
| 1B  | 12 GB  | 2 GB |
| 3B  | 36 GB  | 6 GB |
| 8B  | 96 GB  | 16 GB |
| 70B | 840 GB | 140 GB |

Talking points:
- "Training costs ~6× inference. The model itself is the small part — the optimizer
  state is the hog."
- "This is why 70B full fine-tuning is impossible on one box (840 GB), and why LoRA
  exists — kill items (2)(3)(4) for 99% of params."
- **Real number to flex:** "My actual 1B run peaked at 13.84 GB — I predicted 12 GB
  from this formula before running it. The arithmetic works." (from A2)

---

## Slide 5 — Why m/v need 32-bit but everything else is fine at 16-bit

(The learner specifically wanted this — it's a great "aha" for engineers who know
floats.)

**Setup — fp16/bf16 has a "big eats small" problem:**
```
in fp16:  1024 + 0.5  =  1024     <- the 0.5 is gone
```
Why: 16-bit floats only have ~3-4 significant digits. Up at 1024, the gap between
representable numbers is bigger than 0.5, so it rounds away. (Show fp32 = ~7 digits,
fp16/bf16 = ~3-4.)

**The key idea — does the number ACCUMULATE?**

| quantity | accumulates over many steps? | precision needed |
|---|---|---|
| gradient | no — computed, used once, discarded | 16-bit fine |
| activation | no — used then discarded | 16-bit fine |
| **m, v (optimizer)** | **YES — running averages over thousands of steps** | **32-bit needed** |

- m/v add a tiny increment every step for thousands of steps. In 16-bit, once m grows,
  each tiny update gets "eaten" → training silently stalls.
- gradient is used-then-thrown-away each step, so a little rounding doesn't pile up →
  16-bit is fine.

Talking point: "the rule is dead simple — if a number is accumulated over time, it
needs the extra precision; if it's used once and discarded, 16-bit is enough. That
single rule explains the whole mixed-precision memory layout."

(Optional flex: bf16 vs fp16 — both 16-bit, but bf16 trades precision for a bigger
range so gradients don't overflow to infinity. That's why training uses bf16.)

---

## Slide 6 — What a "neuron" actually is (demystify the black box)

(The learner wanted this — it makes "parameters" concrete instead of mystical.)

**A neuron is just your linear regression.** If you've written `y = w1*x1 + w2*x2 + b`,
you've written a neuron.

A neuron taking 4 inputs:
```
z = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b
```
| parameter | kind | count |
|---|---|---|
| w1..w4 | weight | 4 |
| b | bias | 1 |
| **one neuron** | | **5 parameters** |

Key points for the audience:
- A "parameter" is just **one number** — a weight or a bias. That's all.
- "8 billion parameters" = a bag of 8 billion such numbers, arranged in layers.
- Each number has its own value, and during training its own gradient + m + v —
  nothing is shared. That's literally why the memory is `param_count × 12`.
- A whole layer = many neurons stacked → the weight matrix `W`. A model = layers
  stacked. **It's linear-regression-times-a-lot, plus a non-linear function between
  layers** (so depth means something).

Talking point: "there's no magic inside. It's millions of `w*x+b`. The mystery is the
scale, not the mechanism — and the mechanism is something you learned in school."

---

## Slide 7 — What hardware do YOU need? (audience has no GX10 — land it on real GPUs)

The audience does NOT have a desktop AI box. This slide makes "you can do it" concrete
on hardware they actually have or can rent. One table = the slide.

**The picking rule (one line):** training memory ≈ `params × bytes_per_param`. Full =
~12 B/param, LoRA = ~2 B/param (base only), QLoRA = ~0.6 B/param. Match that to a GPU.

| You have / rent | VRAM | What you can fine-tune | Method |
|---|---|---|---|
| Consumer card (RTX 4060/4070) | 8–12 GB | 0.5B–1B full; up to ~7B | QLoRA |
| **RTX 4090 / 3090** (common) | **24 GB** | 1–3B full; 7–13B LoRA; up to ~33B | QLoRA |
| Rent 1× A100 | 40–80 GB | 7B full; 13–34B LoRA; 70B | QLoRA |
| Rent 1× H100 | 80 GB | same, faster | LoRA/QLoRA |
| Desktop AI box (my GX10) | 128 GB unified | 8B full; 70B | QLoRA |

**Talking points:**
- "A **16 GB consumer card** already fine-tunes a 7B model with QLoRA. That's a gaming
  GPU." ← the headline for this audience.
- "Don't have one? **Rent an A100 for ~$1–2/hour** on RunPod / Lambda / Vast. A small
  fine-tune is minutes — so it costs you the price of a coffee."
- "The only thing that changes across all of these is the **method** (full → LoRA →
  QLoRA), not the idea. Pick the method that fits your VRAM, using the same arithmetic
  from slide 4."

> HONESTY MARKER for you (the presenter): the ONLY number you measured yourself is
> **1B full SFT = 13.84 GB on the GX10**. Everything else in the table above is
> *estimated from the param×bytes arithmetic* (and matches widely-reported community
> numbers), NOT personally verified. If someone asks "did you test the 4090 row?",
> say "no — that's the formula's prediction; my measured point is the 1B run, which
> hit its prediction within 0.2%, so I trust the arithmetic." Don't present estimates
> as measurements. (QLoRA's ~0.6 B/param and the bf16-base LoRA ~2 B/param are the
> figures behind the table; cross-check against `experiments/a01-mem-budget/notes.md`.)

---

## Slide 8 — "So can I do this?" (the close)

- Yes. My actual run: **Llama-3.2-1B, 500 examples, 1 epoch, 83 seconds, 13.84 GB.**
  But (slide 7) a **16 GB gaming GPU** or a **$1/hr rented A100** gets you there too.
- Tools are all open: HuggingFace `transformers` + `Trainer`, a public dataset, a
  Docker container. ~100 lines of Python.
- Bigger model than your card holds? LoRA/QLoRA drops the memory enough to fine-tune
  14B–70B by renting one GPU for an hour.
- **Thesis restated:** the hard, expensive part (pretraining) is done for you.
  Fine-tuning is the accessible part — arithmetic you can predict, code you can read,
  on hardware you already own or can rent for coffee money. Go try it.

---

## Delivery notes / what to prep before Friday

- **Must-have:** Slides 2 (demo) + 4 (memory math) + 7 (what GPU YOU need) — these
  three are what make it land for a no-GX10 audience.
- **Pull the demo text** from `experiments/a03-eval-1b/results.md` (pick pairs #6, #1,
  #5; optionally #10 for the honesty point).
- **Slide 7 honesty:** only the 1B=13.84 GB number is measured; the GPU table is
  arithmetic-estimated. Present it as "the formula predicts" not "I tested" (see the
  honesty marker on that slide). Cross-check the per-method bytes against
  `experiments/a01-mem-budget/notes.md` before Friday.
- **Optional live LoRA number:** run A6 before Friday if you want a real LoRA
  memory/adapter-size figure for Slide 3. Not required — concept is enough.
- **Time budget:** 8 slides ≈ 15–20 min talk + demo. Fits a team share comfortably.
- **Don't** go into backprop, chain rule, or the cross-entropy derivation — wrong
  depth for this audience. Slides 5 and 6 are the deepest you should go, and both are
  framed as "things you already know (floats / linear regression), just at scale."
