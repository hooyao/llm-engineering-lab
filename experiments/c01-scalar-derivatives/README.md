# Track C — Day 1: Scalar derivatives, sanity-check loop

**Time budget:** ~1.5h evening session.

**Why this exists:** Many "matrix calculus" ML formulas are scalar derivatives in
disguise (sigmoid, softmax temperature, single-element loss). Before tensors,
make sure the basics still work in your fingers.

**Anchor reading:** Parr & Howard §1-2. https://explained.ai/matrix-calculus/

---

## Plan

Three pieces, each ~25 minutes:

1. **Read** Parr & Howard §1-2. Goal is to recognize notation, not memorize.
2. **Derive 10 derivatives by hand** (the list below). Pencil + paper.
3. **Verify each** with `experiments/c01-scalar-derivatives/check.py` —
   either via numerical finite differences or via `sympy`.

## The 10 derivatives to do by hand

For each, find `df/dx`:

| # | f(x) | Notes |
|---|---|---|
| 1 | `f(x) = x^3 - 5x + 7`                       | warmup polynomial |
| 2 | `f(x) = (2x + 1)^4`                           | chain rule |
| 3 | `f(x) = e^(2x)`                               | exp + chain |
| 4 | `f(x) = ln(x^2 + 1)`                          | log + chain |
| 5 | `f(x) = x * ln(x)`                            | product rule |
| 6 | `f(x) = 1 / (1 + e^(-x))` (sigmoid σ(x))      | the one ML uses; result has the famous σ(x)(1-σ(x)) form |
| 7 | `f(x) = tanh(x)`                              | useful identity: 1 - tanh²(x) |
| 8 | `f(x) = max(0, x)` (ReLU)                     | piecewise; derivative undefined at 0 (subgradient choice) |
| 9 | `f(x) = -y*ln(p) - (1-y)*ln(1-p)` w.r.t. p, where y is a constant in {0,1} | binary cross-entropy — write the result and compare it to σ(z)-y after composing with sigmoid (preview of next day) |
| 10 | `f(x) = exp(x_i) / sum_j exp(x_j)` w.r.t. x_i (one element of softmax) | the diagonal of the softmax Jacobian; you'll do the full Jacobian on day C4 |

**Tip:** Write each step. Once you arrive at the answer, look at the result line
and see if you recognize a *named* function. Sigmoid's derivative being
`σ(x)(1-σ(x))` is one of the most reused identities in deep learning.

## Verification

Use `check.py` (committed alongside this file). Two methods supported:

- **Numerical:** central finite difference, `(f(x+h) - f(x-h)) / (2h)` with `h = 1e-5`.
  Should agree with your symbolic answer to ~7 decimal places.
- **Symbolic:** `sympy.diff(f, x)` for an algebraic match.

Both are in `check.py`. You don't need to write them — just edit your hand-derived
expression into the lookup table at the top of the file and run it.

## Deliverable

By the end of the session, this directory should contain:

- `check.py`               (provided — fill in your answers)
- `notes.md`               (your own — 5 lines: which derivative tripped you up, if any; what notation P&H uses that you didn't see before)
- Optionally, a photo of your scratch pencil work, committed as `scratch.jpg`

That's it. No model, no GPU, no docker. Pure math warmup.

## What this prepares you for

- **C2 tomorrow:** partial derivatives → gradient vector. Same exercises but with multiple inputs.
- **C3:** chain rule with shared subexpressions (this is what makes backprop efficient).
- **C6 (later):** you'll derive backprop through a 2-layer MLP by hand. The
  sigmoid and softmax derivatives from tonight are the building blocks.

## If you finish early

Sebastian Raschka *Build a Large Language Model (From Scratch)* — read **only
chapter 1** (overview). Don't start chapter 2 yet; we'll thread chapter 2 (data
loading) into Track A or B at the right moment so you have GPU context to actually
run it.
