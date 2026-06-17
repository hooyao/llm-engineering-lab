# Backpropagation, from scratch — offline study note

> A Chinese companion (`backprop-primer.zh.md`, same folder) exists by explicit user
> request — same content, Chinese connective prose with ML/systems terms kept in
> English. This English version is the canonical one.

**Who this is for:** you have written linear regression, you have a systems/backend
background, and you have forgotten every bit of neural-network internals. This note
is self-contained. Read it alongside whatever YouTube lecture you pick (Karpathy's
"The spelled-out intro to backpropagation" is the canonical one — that's Track B1
later anyway). It exists so the video has a written companion pitched at exactly
your level.

**Where it sits:** this is the "why" under `experiments/a01-mem-budget/`. The A1
calculator needs you to know that training stores, per parameter, a *gradient*
(2 bytes) plus *activations* whose size depends on the network. Both of those come
straight out of backprop. `teaching-notes.md` (same folder) covers what a parameter
is and how a forward pass runs — read that first; this picks up from there.

---

## 0. The one-sentence summary

Training a model is just this loop, repeated:

```
1. forward pass:  feed an input through the network, get an output, compute a loss
2. backward pass: compute, for every weight, how much it contributed to the loss
3. update:        nudge every weight a small step in the direction that lowers loss
```

**Backpropagation is step 2** — the algorithm that computes "how much each weight
contributed to the loss," efficiently, for all weights at once. That quantity is
the *gradient* `∂L/∂w`. Everything below is unpacking that.

---

## 1. What we are actually trying to compute

The training update rule is **gradient descent**:

```
w  ←  w  -  lr · ∂L/∂w
```

For each weight `w`, take a small step (size `lr`, the *learning rate*) in the
direction that decreases the loss `L`. The sign of `∂L/∂w` tells you which way to
move; its magnitude tells you how steeply the loss responds.

So the *only* thing training needs that we don't already have is, for every weight:

```
∂L/∂w   =   "if I nudge this weight up by a tiny amount,
             how much does the loss change, and in which direction?"
```

`∂L/∂w` is read "partial derivative of L with respect to w." If calculus is rusty:
a derivative is just a *slope* — output-change divided by input-change for a tiny
input nudge. Nothing more exotic than that is needed here.

Backprop is the procedure that produces all of these `∂L/∂w` values.

---

## 2. The obstacle: the loss is many layers away from the weight

Take a tiny 2-layer network (the same toy used in `teaching-notes.md`):

```
x  ──[W1, b1]──►  z1  ──ReLU──►  h  ──[W2, b2]──►  y  ──loss──►  L
```

The weight `W1` does not touch `L` directly. It affects `z1`, which affects `h`,
which affects `y`, which affects `L`. Four hops. You cannot write `∂L/∂W1` in one
shot — there's a chain of functions in the way.

The tool for "derivative through a chain of functions" is the **chain rule**:

```
∂L/∂W1  =  ∂L/∂y · ∂y/∂h · ∂h/∂z1 · ∂z1/∂W1
            └─────────────┬──────────────┘
              a product of per-hop slopes, multiplied together
```

Read it right-to-left: a tiny change in `W1` changes `z1` (slope `∂z1/∂W1`), which
changes `h` (slope `∂h/∂z1`), which changes `y`, which changes `L`. Multiply the
per-hop slopes and you get the end-to-end slope `∂L/∂W1`.

**The key efficiency insight** (this is *why* it's called back-propagation): the
left part of that product, `∂L/∂y · ∂y/∂h · ...`, is *shared* by every weight that
sits behind that point in the network. So instead of recomputing the whole chain
for each weight, you compute it **once, starting from the loss end, and carry the
running product backwards layer by layer.** Each layer reuses the result handed to
it by the layer behind it. That carried-backwards quantity is the "propagation."

---

## 3. Why the forward pass must save its intermediate values

Look again at the chain-rule factors: `∂y/∂h`, `∂h/∂z1`, `∂z1/∂W1`. When you
actually evaluate these slopes, they turn out to depend on the **activations
computed during the forward pass** — `h`, `z1`, `x`. (You'll see exactly where in
§4.) That is the entire reason a training forward pass holds onto its intermediate
results and an inference forward pass doesn't: the backward pass needs them as
ingredients. This is the source of the "activation memory" term in the A1 budget.

---

## 4. One full pass, with real numbers

### Forward

```
x  = [1.0, 2.0]                         # input activation to layer 1
W1 = [[0.5, -0.3],
      [0.2,  0.8]]
b1 = [0.1, -0.1]
z1 = W1 @ x + b1 = [0.0, 1.7]           # pre-activation
h  = ReLU(z1)    = [0.0, 1.7]           # input activation to layer 2
W2 = [0.7, -0.4]
b2 = [0.05]
y  = W2 @ h + b2 = -0.63                 # output

target t = 0
L  = 0.5·(y - t)² = 0.5·(-0.63)² = 0.198 # squared-error loss
```

(`@` is matrix/vector multiply. `ReLU(z) = max(0, z)`, applied elementwise.)

### Backward — start at the loss, walk back to the input

**Step 0 — gradient of the loss w.r.t. the output `y`.** This is the seed of the
whole backward chain. For `L = 0.5·(y - t)²`, the derivative is just `(y - t)`:

```
g_y = ∂L/∂y = (y - t) = -0.63
```

**Step A — gradient w.r.t. `W2`** (one of the things we're after). Since
`y = W2 @ h + b2`, the slope of `y` w.r.t. `W2` is `h`, so by the chain rule:

```
∂L/∂W2 = g_y · hᵀ = -0.63 × [0.0, 1.7] = [0.0, -1.071]   ← uses h
∂L/∂b2 = g_y                          = -0.63
```

**Step B — push the gradient back onto `h`** (so we can continue toward layer 1).
The slope of `y` w.r.t. `h` is `W2`, so:

```
g_h = W2ᵀ · g_y = [0.7, -0.4] × (-0.63) = [-0.441, 0.252]
```

**Step C — push back through the ReLU, onto `z1`.** ReLU's slope is 1 where its
input was positive, 0 where it was negative (it blocks gradient on dead units):

```
z1 = [0.0, 1.7]  →  mask = [0, 1]
g_z1 = g_h ⊙ mask = [-0.441×0, 0.252×1] = [0.0, 0.252]
```

(`⊙` is elementwise multiply.)

**Step D — gradient w.r.t. `W1`** (the other thing we're after). Since
`z1 = W1 @ x + b1`, the slope w.r.t. `W1` is `x`:

```
∂L/∂W1 = g_z1 · xᵀ = [0.0, 0.252]ᵀ ⊗ [1.0, 2.0]
       = [[0.0,   0.0  ],
          [0.252, 0.504]]                            ← uses x
∂L/∂b1 = g_z1 = [0.0, 0.252]
```

**Step E — update every weight** (gradient descent, say `lr = 0.1`):

```
W2 ← W2 - 0.1·∂L/∂W2 = [0.7, -0.4] - 0.1·[0.0, -1.071] = [0.7, -0.293]
W1 ← W1 - 0.1·∂L/∂W1 = [[0.5, -0.3],[0.2,0.8]] - 0.1·[[0,0],[0.252,0.504]]
b2 ← b2 - 0.1·∂L/∂b2 ; b1 ← b1 - 0.1·∂L/∂b1
```

That is **one training step**. The loss on this example is now slightly lower than
0.198. Repeat across many examples, many times — that loop *is* training.

---

## 5. The distinction that trips everyone up: two kinds of gradient

The backward pass moves **two different things**, and confusing them is the #1
source of "I don't get backprop":

| symbol | what it is | purpose | kept after the step? |
|---|---|---|---|
| `g_y`, `g_h`, `g_z1` | gradient of loss w.r.t. an **activation** | *intermediate messenger* — its only job is to be passed to the previous layer | thrown away |
| `∂L/∂W2`, `∂L/∂W1`, `∂L/∂b` | gradient of loss w.r.t. a **weight/bias** | *the actual goal* — used to update the parameter | this is the "gradient = 2 bytes/param" in the memory budget |

Every layer does exactly two jobs:
- **(a)** use the incoming activation-gradient + its own saved input activation to
  compute *its own weight gradient* (Steps A, D);
- **(b)** pass an activation-gradient further back to the previous layer (Steps B, C).

Think of `g` as the messenger flowing backward, and `∂L/∂W` as the cargo each layer
drops off along the way.

---

## 6. Why every layer needs *its own* saved activation

Notice: Step A used `h` to get `∂L/∂W2`; Step D used `x` to get `∂L/∂W1`. Different
layers' weight-gradients lock onto *different* activations, and they are **not
interchangeable** — `h` is useless for `W1`, `x` is useless for `W2`. The general
rule for layer `k`:

```
∂L/∂W_k  =  (activation-gradient arriving at layer k)  ·  (input activation of layer k)ᵀ
```

A 28-layer network therefore keeps 28 activations alive (`x = a₀, a₁, ..., a₂₇`).
The backward pass walks from layer 28 down to layer 1, and as it reaches layer `k`
it consumes activation `a_{k-1}`. That is precisely why a forward pass in training
can't discard intermediates after each layer the way inference does: **each one is
owed to a future backward step.**

This is the answer to "why store every layer's output, not just the last one?" —
it's not the last `h` that's needed; it's one input activation *per layer*, because
the backward pass will come back for each of them in turn.

---

## 7. How this connects back to A1's memory budget

Now the two training-only memory terms in `budget.py` have a reason:

- **gradient, 2 bytes/param** — the `∂L/∂W` cargo from §5. One gradient value per
  weight, same dtype as the weight (BF16). You must hold it long enough to do the
  update in Step E.
- **activations** — the per-layer saved inputs from §6. Their total size scales
  with `seq_len × batch × hidden × layers`, because more tokens / bigger batch =
  more intermediate values to keep alive until the backward pass collects them.
- **activation checkpointing** (Track A5) — the optimization that this whole note
  makes legible: *don't* save most activations during forward; when the backward
  pass needs `a_{k-1}`, recompute it by re-running that slice of the forward pass.
  Trade compute for memory. The "×6 vs ×1" factor in A1's activation formula is
  exactly the difference between keeping all of a transformer block's intermediates
  alive for backward versus recomputing them on demand.

That's the full loop: forward saves activations → backward walks from the loss,
using the chain rule layer by layer, producing weight gradients → gradient descent
applies them. Activations cost memory because they must survive from forward all
the way to the backward step that consumes them.

---

## 8. A 30-line numpy version to read after the video

Once the video + this note click, this is the entire algorithm with no framework.
Reading it is the fastest way to confirm you actually have it (you'll hand-write
something like this in Track C6 and Track B1):

```python
import numpy as np

# one 2-layer MLP, one training step, no autograd
x  = np.array([1.0, 2.0]); t = np.array([0.0])
W1 = np.array([[0.5, -0.3], [0.2, 0.8]]); b1 = np.array([0.1, -0.1])
W2 = np.array([[0.7, -0.4]]);             b2 = np.array([0.05])

# ---- forward (save z1, h, x for the backward pass) ----
z1 = W1 @ x + b1
h  = np.maximum(0, z1)            # ReLU
y  = W2 @ h + b2
L  = 0.5 * np.sum((y - t) ** 2)

# ---- backward ----
g_y  = (y - t)                   # Step 0
gW2  = np.outer(g_y, h)          # Step A   (uses h)
gb2  = g_y
g_h  = W2.T @ g_y                # Step B
g_z1 = g_h * (z1 > 0)            # Step C   (ReLU mask)
gW1  = np.outer(g_z1, x)         # Step D   (uses x)
gb1  = g_z1

# ---- update ----
lr = 0.1
W2 -= lr * gW2; b2 -= lr * gb2
W1 -= lr * gW1; b1 -= lr * gb1
```

Map each line to a step above. When `gW1 = np.outer(g_z1, x)` makes obvious sense —
"of course `W1`'s gradient needs `x`" — you've got backprop back.

---

## Suggested videos (pick one, this note is the companion)

- **Karpathy, "The spelled-out intro to neural networks and backpropagation"**
  (YouTube, ~2h25m) — builds micrograd, scalar-by-scalar. This is Track B1; doing
  it now just means B1 is review. Best single match for this note.
- **3Blue1Brown, "Backpropagation calculus"** (Deep Learning ch.4, ~10m) — the
  visual/intuitive version. Shorter; good first pass before Karpathy.

Either works. The numbers and steps here are written to line up with how both of
them present it.
