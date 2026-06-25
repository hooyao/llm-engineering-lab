# A1 teaching notes — what a "parameter" actually is, and where the bytes go

**Audience:** these notes were written for a learner with strong systems/backend
background but **no prior neural-network internals**. They capture the conceptual
prerequisites the curriculum (`notes/curriculum-v2-execution.md` § A1) assumes but
does not teach: what a model parameter physically is, how a forward pass runs, and
why that turns into the `bytes/param` arithmetic A1's calculator is built on.

This is the "why" behind `budget.py`. Read this before reading the code.

---

## 1. A parameter is just a number. A model is a bag of numbers.

If you have written linear regression, you already have the whole idea:

```
y = w1*x1 + w2*x2 + b
```

Here `w1, w2, b` are **3 parameters** — three numbers you tune so the line fits.
The weight is a *vector* `[w1, w2]`. That is exactly **one neuron**: a dot product
plus a bias.

A neural network is **not** a new idea on top of this. It is the same thing,
expanded along two axes only:

1. **Put many neurons in one layer** → the weight stops being a vector and becomes
   a **matrix** (one row per neuron).
2. **Stack layers** → you now have several weight matrices, each layer feeding the
   next.

That's it. "A large model has 8 billion parameters" literally means: there is a
bag of 8 billion numbers, arranged into a few hundred matrices and vectors.

---

## 2. A concrete 9-parameter model

Input 2 numbers → one hidden layer of 2 neurons → output 1 number:

```
x1 ─┐
    ├─→ [neuron h1] ─┐
x2 ─┤               ├─→ [neuron y] ─→ output
    ├─→ [neuron h2] ─┘
   (input)     (hidden: 2)      (output: 1)
```

Its **entire parameter set** is these 9 numbers (values chosen arbitrarily):

| name             | shape        | the actual numbers              | count |
|------------------|--------------|---------------------------------|-------|
| `W1` (layer-1 weight) | 2×2 matrix | `[[0.5, -0.3], [0.2, 0.8]]`     | 4     |
| `b1` (layer-1 bias)   | length-2   | `[0.1, -0.1]`                   | 2     |
| `W2` (layer-2 weight) | 1×2 matrix | `[0.7, -0.4]`                   | 2     |
| `b2` (layer-2 bias)   | length-1   | `[0.05]`                        | 1     |
|                  |              | **total**                       | **9** |

**Key insight:** each *row* of `W1` is one linear regression. Row `[0.5, -0.3]` is
neuron h1's two weights; row `[0.2, 0.8]` is h2's. Stacking two linear regressions'
weight vectors on top of each other *is* the weight matrix. That is why "the weight
is a matrix" — a layer holds many neurons, one per row.

---

## 3. Forward propagation, with numbers

Input `x = [1.0, 2.0]`.

**Layer 1** — each neuron does the familiar dot-product + bias, then passes through
an activation function ReLU (`max(0, ·)`):

```
h1 pre-activation = 0.5*1.0 + (-0.3)*2.0 + 0.1  = 0.0
h2 pre-activation = 0.2*1.0 +  0.8 *2.0 + (-0.1) = 1.7

after ReLU:  h1 = max(0, 0.0) = 0.0
             h2 = max(0, 1.7) = 1.7
```

Hidden output `h = [0.0, 1.7]`.

**Layer 2** — the output neuron takes `h`, does another dot-product + bias:

```
y = 0.7*0.0 + (-0.4)*1.7 + 0.05 = -0.63
```

Output `-0.63`. The whole forward pass is just:
**dot-product + bias → activation → dot-product + bias → activation → ...**, each
layer feeding the next. No magic.

### Matrix form (this is what the GPU actually computes)

Both layer-1 dot products at once = **matrix × vector**:

```
h_pre = W1 @ x + b1
      = [[0.5, -0.3],  @  [1.0]  +  [ 0.1]   =  [0.0]
         [0.2,  0.8]]     [2.0]     [-0.1]      [1.7]
```

`@` is matrix multiply. **The entire point of "weights are matrices" is this step:**
one matrix multiply = every neuron's dot product in a layer, computed together. The
GPU's Tensor Cores exist to do exactly this `W @ x` (a GEMM, general matrix multiply).
The "~93 TFLOPS BF16 sustained" number in `CLAUDE.md` / `notes/hardware-gx10.md`
measures how many of these multiply-adds the GB10 does per second.

---

## 4. Scaling up to an LLM

Identical machinery, just bigger numbers:

| | the 9-param toy | Llama-3.2-3B |
|---|---|---|
| size of one `W` matrix | 2×2 | ~3072×3072 ≈ 9.4M numbers |
| number of layers | 2 | 28 (each with several attention `W`s too) |
| total parameters | 9 | 3.2×10⁹ |
| forward pass | dot-product + activation, 2 steps | the same dot-product + activation, just a few hundred matrix multiplies chained |

An LLM is a few hundred large matrices plus some vectors; a forward pass runs them
one after another. Attention adds a little structure (Q/K/V — three weight matrices
— plus a softmax), but the substrate is still "a bag of numbers arranged as
matrices, multiplied in sequence."

---

## 5. From "bag of numbers" to the memory bill

Now "parameter" is concrete: it is **one number in the bag** (9 of them in the toy,
3.2 billion in Llama-3B). The memory question is simply: **how many copies of each
number must we hold in memory to train?**

Per parameter, during BF16 training with AdamW:

| what | bytes/param | why |
|---|---|---|
| the weight itself (BF16) | 2 | the number, stored at 16-bit precision |
| its gradient (BF16) | 2 | which direction to nudge it this step |
| AdamW moment `m` (FP32) | 4 | running average of the gradient |
| AdamW moment `v` (FP32) | 4 | running average of the gradient *squared* |
| **total** | **12** | |

So a 3B model in full fine-tuning, before activations:

```
3 × 10⁹ params × 12 bytes = 36 GB
```

**The counter-intuitive part:** the optimizer state (`m + v` = 24 GB) is **4× larger
than the model itself** (6 GB). A "small" 3B model costs 36 GB to fully fine-tune —
and that still excludes activations (the intermediate `h` values from §3; see §6).

> What `m` and `v` *do* — the actual AdamW update equations — is derived in Track C7.
> For sizing memory you only need to know that AdamW keeps two FP32 books (`m`, `v`)
> per parameter, 4 bytes each.

> **12 vs 16 — read this once.** The 12 B/param above is the *pure-bf16* recipe and
> is the right number to build the intuition on. **Production full SFT actually uses
> 16 B/param** (mixed-precision Adam): it adds a **fp32 master weight** (+4 B) on top
> of the bf16 weight. Reason: bf16 has only ~7 bits of mantissa, so many `lr·grad`
> updates are too small to survive being added to a bf16 weight — keep an fp32 master
> copy, accumulate the update there, cast back to bf16 for the next forward. So the
> real bill is `2 weight + 2 grad + 4 master + 4 m + 4 v = 16`, and a 3B full SFT is
> `3.2e9 × 16 ≈ 51 GB`, not 36. `budget.py` and `notes/curriculum.md` use 16; this
> note teaches with 12 then corrects to 16 here so both numbers make sense.

> **Update — MEASURED in A2 (2026-06-17):** the *default* is actually 12, not 16.
> HuggingFace `Trainer(bf16=True)` keeps **no fp32 master weight**, so it runs the
> 12 B/param recipe; A2's 1B full SFT peaked at 13.84 GB, matching 12 B/param to
> 0.2%. The 16 B/param (fp32 master) recipe is what **DeepSpeed / FSDP
> mixed-precision** uses. So: 12 = HF-`Trainer` default, 16 = DeepSpeed/FSDP upper
> bound. See `../a02-sft-1b/notes.md` for the full reconciliation.

### dtype is the per-number cost factor

| dtype | bytes/number | used for |
|---|---|---|
| FP32 (single) | 4 | old full precision; AdamW moments |
| **BF16 / FP16 (half)** | **2** | default training precision today |
| FP8 | 1 | |
| NF4 (4-bit) | 0.5 | QLoRA base weights |

`param_bytes = num_params × dtype_bytes`. Everything else in the calculator is a
variation on this one multiply.

---

## 6. Why this is the whole motivation for Track A

That single multiplication is the entire "can I train this model, and how" decision
on the GX10's 128 GB unified pool:

| model, full fine-tune | params × 12 bytes | verdict on GX10 |
|---|---|---|
| 3B | 36 GB | fits |
| 8B | 96 GB | fits but tight once activations are added |
| 70B | 840 GB | off by an order of magnitude — impossible |

Every later Track A method is an attack on this bill:

- **LoRA** (A6) — train only ~1% of the parameters, so the 12 bytes/param only
  applies to a tiny slice. Optimizer state drops from tens of GB to a few hundred MB.
- **QLoRA** (A7) — store even the frozen base weights in NF4 (0.5 bytes), cutting the
  "weight itself" term ~4×. This is how a 70B model becomes trainable on one box.

The **activations** term — the intermediate `h` values produced at every layer during
the forward pass (§3) — is the third piece of the calculator (`activation_bytes`). It
scales with `seq_len × batch × hidden × layers`, and activation checkpointing (A5)
trades recompute for memory by *not* storing most of them. The rough `×6`
no-checkpointing factor in the A1 formula accounts for the several intermediate
tensors each transformer block keeps alive for the backward pass.

> **Why activations must be kept alive at all** — i.e. why a *training* forward pass
> can't discard each layer's output the way *inference* can — is the backward pass.
> Each layer's weight gradient needs that layer's own input activation, so every
> activation must survive from forward until the backward step that consumes it.
> Full derivation with worked numbers: `backprop-primer.md` (same folder), §§3–7.

---

## 7. What you can now do (the A1 deliverable)

With the three terms — `param_bytes`, `optimizer_bytes`, `activation_bytes` —
`budget.py` lets you size **any** model + method combo on this box without trial
and error: full SFT vs LoRA, BF16 vs NF4, with/without checkpointing, and read off
whether each lands in "comfortable / marginal / will OOM."

That is the skill A1 buys: look at a HuggingFace model card, do the arithmetic, and
know the answer before launching a run.
