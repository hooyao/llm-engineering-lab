# Study resources — videos that back the A2 lesson (backprop + AdamW)

Curated for THIS learner: has written linear regression, watched Andrew Ng's
Coursera ML in 2015 (now fuzzy), strong systems/perf background, needs
backpropagation + AdamW at a level they can actually follow. All links verified to
resolve to the stated video (2026-06-17). The point of these is to *re-awaken* a
2015 foundation and to give the backward/optimizer half of A2 a visual + mechanical
backing before we teach it.

## Watch order (the efficient path)

```
Step 1 — backprop, ~20 min (watchable today):
  3Blue1Brown Ch3 (intuition) -> Ch4 (calculus)        <- backprop in one sitting

Step 2 — AdamW, ~25 min:
  Andrew Ng C2W2L06 momentum -> L07 RMSProp -> L08 Adam <- re-awaken the Adam memory

Optional deepening (2.5 h, not now):
  Karpathy micrograd  <- when you want to write it yourself; this IS Track B1
```

## Backpropagation

### Primary: 3Blue1Brown (two episodes, watch both)

1. **Backpropagation, intuitively** — Deep Learning Chapter 3
   https://www.youtube.com/watch?v=Ilg3gGewQ5U
   Pure geometric intuition: what a gradient *means* for nudging each weight. No
   chain-rule machinery yet.

2. **Backpropagation calculus** — Deep Learning Chapter 4
   https://www.youtube.com/watch?v=tIeHLnjs5U8
   (~10 min) Unfolds the chain rule — maps directly onto how the `-log(p)` loss in
   `learning-notes.md` propagates back layer by layer. These two are the consensus
   best visualization of backprop.

### Hands-on version: Karpathy (later, not now)

3. **The spelled-out intro to neural networks and backpropagation: building micrograd**
   https://www.youtube.com/watch?v=VMj-3S1tku0
   2h25m, hand-writes an autograd engine from scratch. **This is literally Track B1**,
   so watching it now = doing B1 early. Deeper/slower than 3B1B. Do the 3B1B pair
   first for intuition; treat this as the "write it myself" deepening.

## AdamW (the optimizer)

### Primary: Andrew Ng (the exact lecturer this learner already knows)

Ng splits Adam into two simpler prerequisites — watch in this order, Adam is just
the two combined:

4. **Gradient Descent With Momentum (C2W2L06)**
   https://www.youtube.com/watch?v=k8fTYJPd3_I  — explains Adam's `m`
5. **RMSProp (C2W2L07)**
   https://www.youtube.com/watch?v=_e-LFe_igno  — explains Adam's `v`
6. **Adam Optimization Algorithm (C2W2L08)**
   https://www.youtube.com/watch?v=JXQT_vxqwIs  — m + v combined = Adam

> AdamW's **W** (decoupled weight decay) is a small correction on top of Adam that
> didn't exist as a separate thing in the 2015 lectures. Understanding Adam is
> enough; the W difference is a one-line verbal addendum (covered when we teach the
> optimizer step).

## Should I watch the whole 3Blue1Brown Deep Learning series in parallel?

Yes — high value for this learner, with two guardrails.

**Why it's worth it:** the series is pure geometric intuition, ~zero code/engineering
— exactly the layer this learner is weakest on and the notes are filling. It doesn't
teach new material; it gives already-half-known concepts a "oh, *that's* what it
looks like" picture. And it's in sync with our pace (gradient descent, backprop =
notes Seg 4/5 and the next segment), so watching in parallel is dual-coding: same
thing taught once in code/their terms, once as a picture. Sticks harder.

**Guardrail 1 — it's not a course replacement.** Only ~4-5 episodes, narrow scope
(NN basics + backprop; no transformer, no training engineering, no LLM). Finishing
it is *foundation visualization*, not "done with deep learning." Don't get the
"I've finished" illusion.

**Guardrail 2 — don't run ahead of our pace.** Watch only through the backprop
episode now (in sync with A2). The series' attention/transformer/GPT episodes are
real and good, but **save them for Track B4** — watched now they're castles in the
air with no felt context.

**Concrete plan:**

```
Watch in parallel now (in sync with A2, ~30-40 min):
  Ch1 What is a neural network   <- re-awaken; you can probably run this at 1.5x
  Ch2 Gradient descent           <- pairs with the loss segment in learning-notes
  Ch3 Backpropagation intuition  <- the next segment we'll teach
  Ch4 Backpropagation calculus

Do NOT watch yet:
  the attention / transformer / GPT episodes  <- save for Track B4
```

Ch1/Ch2 may feel like review — good, run them at 1.5x as a refresher. The real
value is Ch3/Ch4, which mesh exactly with the backward + optimizer piece (A2's last
puzzle piece).

## Division of labor between the two sets

- **3Blue1Brown** — the *picture* of what backprop is doing (the backward pass).
- **Andrew Ng C2W2L06-08** — the *mechanism* of how Adam's m/v are computed (the
  optimizer step).

They don't overlap; watch both. 3B1B covers the `loss.backward()` beat, Ng covers
the `optimizer.step()` beat — exactly the two halves of A2's final puzzle piece.
