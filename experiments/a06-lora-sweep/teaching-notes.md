# A6 — teaching notes: the structure LoRA attaches to (clean review reference)

> A clean, self-contained reference for REVIEW — written 2026-06-25 at the learner's
> request ("write the attn/mlp architecture down so I can review it"). The dialogue-shaped
> record is in `learning-notes.md` Seg 4-7; THIS file is the de-dialogued version: read it
> top to bottom to relearn "what is LoRA actually bolted onto."
>
> **Why this file exists / the mistake it fixes:** while teaching, I drew the frozen weight
> as a square `W d×d` block WITHOUT saying I was silently using `q_proj`/`o_proj` (which
> happen to be square) as the example. A general weight matrix is NOT square. That silent
> special-case caused real confusion. This file uses the **general form `[d_out, d_in]`
> everywhere** and flags the square case explicitly when it appears.

---

## 0. The frame: what question this answers

LoRA has three hyperparameters: `r` (rank), `α` (alpha), and **target modules**. The third
one — "which weight matrices get an adapter" — can't be understood without knowing what
weight matrices a transformer even has. This file builds that picture, then the three
hyperparameters sit on it cleanly.

---

## 1. One weight matrix: shape = `[d_out, d_in]`

A linear layer computes `y = W · x`.

- **`x`** is the **input activation** — the vector the previous layer emitted, fed into
  this layer.
- **`W`** is this layer's weight matrix.
- **`y`** is the output activation, fed to the next layer.

Shapes, with ONE token (drop batch/seq_len for now):

```
x       = [d_in]            input activation vector
W       = [d_out, d_in]     the matrix
y = W·x = [d_out]           output activation vector
```

**Why W must be `[d_out, d_in]` (forced, not chosen):** its whole job is to turn a
`d_in`-dimensional vector into a `d_out`-dimensional one. To do that:
- columns must = `d_in` (so x can multiply in),
- rows must = `d_out` (so it emits a `d_out` vector).

Dimension accounting (the two `d_in`'s cancel):

```
   W            ·    x       =    y
[d_out, d_in]      [d_in]        [d_out]
        └─────┬──────┘
         these must match and cancel, leaving d_out
```

### The square special case (the thing that confused you)

If `d_out == d_in`, the matrix looks square — `[4096, 4096]`. **That is a special case.**
`q_proj` and `o_proj` happen to be square in Llama-3.1-8B, which is what I was silently
drawing when I wrote `W d×d`. Most weight matrices are NOT square. **Always think
`[d_out, d_in]`; treat `d×d` as the coincidence it is.**

### Batch and seq_len don't change W

Real runs feed a `batch × seq_len` block of tokens at once. x gains two dims; **W is
unchanged** (it's the layer's fixed parameter, independent of how many tokens flow):

```
x = [batch, seq_len, d_in]    W = [d_out, d_in]    y = [batch, seq_len, d_out]
```

W's size is `d_out × d_in` only — no batch, no seq_len. (Contrast: activation size carries
`batch × seq_len` — that was the A5 sweep axis. Weights and activations scale on different
things; don't fuse them.)

---

## 2. The 7 weight matrices in one transformer layer

Real numbers, Llama-3.1-8B-Instruct: `d_model = 4096`, MLP intermediate `d_ff = 14336`.

```
                 W = [d_out, d_in]     square?   group
─────────────────────────────────────────────────────────
q_proj           [4096,  4096]         yes       attn
k_proj           [1024,  4096]         no (GQA)  attn
v_proj           [1024,  4096]         no (GQA)  attn
o_proj           [4096,  4096]         yes       attn
─────────────────────────────────────────────────────────
gate_proj        [14336, 4096]         no        mlp   ← widens 4096 → 14336
up_proj          [14336, 4096]         no        mlp   ← widens 4096 → 14336
down_proj        [4096,  14336]        no        mlp   ← narrows 14336 → 4096
```

- **k_proj / v_proj are narrower** (`d_out = 1024`, not 4096). That's grouped-query
  attention (GQA) — K and V are shared across head groups. Why → Track B4. For now just
  note: they're `1024+4096` in any formula, NOT `4096+4096`.
- **The mlp matrices have a `14336` dim** — 3.5× wider than 4096. This is what makes them
  "expensive" under full fine-tuning, and what LoRA tames (see §6).
- "**attn**" and "**mlp**" are just **group names for these 7 matrices**. attn = the 4 that
  attention uses; mlp = the 3 the feed-forward network uses (the MLP is the same kind of
  net you trained in A2). What attention does internally → B4; not needed here.

---

## 3. A transformer is a STACK of these layers

Direction matters (this was a confusion point): **a transformer CONTAINS many weight
matrices; a weight matrix does not contain a transformer.** Scale, with explicit counts:

```
1 model (Llama-3.1-8B)
  └─ 32 transformer layers, stacked
       each layer = 7 weight matrices:
         attn group: q_proj, k_proj, v_proj, o_proj   (4)
         mlp  group: gate_proj, up_proj, down_proj     (3)
  → 32 × 7 = 224 weight matrices total that LoRA could attach to
```

---

## 4. What a LoRA adapter is, and "target modules"

LoRA never touches `W`. Beside a chosen `W` it adds a **parallel branch** of two small
matrices `A` and `B` (the Seg-1 picture). 

**One LoRA adapter = the `(A, B)` pair attached to ONE weight matrix.** One W → one
adapter. So the number of adapters = how many W's you choose to attach to.

**target modules = which W's get an adapter.** This is the 3rd hyperparameter. The two
choices the A6 sweep uses:

```
attn-only:  attach to the 4 attn W's per layer   → 32 × 4 = 128 adapters
            (gate/up/down stay frozen: used in forward, never trained)

attn+mlp :  attach to all 7 W's per layer        → 32 × 7 = 224 adapters
```

Un-adapted W's stay **frozen** — they still run in the forward pass, they just don't learn.

---

## 5. Where the LoRA branch sits, and where r / α/r live

One linear layer with its LoRA branch (general shapes, no `d×d`):

```
  x ──┬───────────────► [ W  [d_out,d_in]  frozen ] ───────────► W·x ──┐
(d_in)│                                                         (d_out) │
      │                                                                 ▼
      │   ┌ A [r,d_in] ┐   ┌ B [d_out,r] ┐   ┌ scalar ┐             ( + ) ──► output
      └──►│   down     │──►│     up      │──►│ ×(α/r) │──────────────►       (d_out)
          └────────────┘   └─────────────┘   └────────┘
            A·x: [r]          B·A·x: [d_out]     [d_out]
              ▲                                     ▲
         r = bottleneck                        α/r = how hard the
         (the "waist")                         branch is added in

  forward:  output = W·x  +  (α/r) · B · (A · x)
```

- **A** squashes `d_in → r`. **B** lifts `r → d_out`. So `B·A` has the same shape as W
  (`[d_out, d_in]`) but is built from only `r×(d_in+d_out)` numbers — never formed in full.
- **`r`** is the dim of the squashed vector between A and B — a visible position (the waist).
- **`α/r`** is the scalar multiplying the branch output before the add — a visible position.
- **`α` alone has NO position.** It's just one of the two numbers you divide to get the
  `α/r` scalar (`α/r = α ÷ r`). On the diagram you can point at `r` and at `α/r`, never at
  a bare `α`.
- **B is initialized to 0**, so `B·A = 0` at step 0 → the adapter starts as a no-op and the
  model departs smoothly from the base.

The three hyperparameters, one line each:

```
r              bottleneck width inside each adapter   → how big one adapter is
α (via α/r)    strength of each adapter's branch       → how hard one adapter is added in
target modules which W's get an adapter               → how many adapters there are
```

---

## 6. Why LoRA makes the wide mlp matrices affordable: area vs perimeter

For ONE weight matrix:

```
full fine-tune pays:  d_out × d_in           ← AREA   (product of the two dims)
LoRA pays:            r × (d_in + d_out)      ← PERIMETER (sum of the dims) × r
```

Concrete, `down_proj [4096, 14336]`, `r = 16`:

```
full W:   4096 × 14336        = 58,720,256   ≈ 58.7M
LoRA:     16 × (4096 + 14336) =    294,912   ≈  0.29M     → 0.5%
```

Compare mlp vs attn for a single W:

```
                full FT (area)            LoRA (perimeter, r=16)
attn W (q)      4096×4096  = 16.8M        16×(4096+4096)   = 131K
mlp  W (down)   14336×4096 = 58.7M        16×(14336+4096)  = 294K
mlp / attn      3.5×                      2.25×
```

The point: under **full fine-tuning** the 14336 dim is a *multiplier*, so mlp is 3.5× an
attn matrix and you can't afford to tune it. Under **LoRA** the 14336 becomes an *addend*,
so mlp is only 2.25× — affordable. That's why `attn+mlp` (adapt everything) is a common
default with LoRA even though full-FT would never touch mlp lightly.

---

## 7. Per-layer formula → whole model → the 4 configs

Per-layer LoRA params (each W contributes `r × (d_in + d_out)`):

```
attn-only per layer = (4096+4096 + 1024+4096 + 1024+4096 + 4096+4096) × r = 26,624 × r
attn+mlp  per layer = 26,624×r + (14336+4096)×3×r                        = 81,920 × r
```

× 32 layers (every layer adapted identically within one config):

```
attn-only whole model =   851,968 × r
attn+mlp  whole model = 2,621,440 × r
```

The 4 sweep configs (only `r` and target modules vary; α set to 2r so α/r = 2 throughout):

```
config              r     target      trainable params
──────────────────────────────────────────────────────
① r8  / attn        8     attn-only     6,815,744  ≈ 6.82M
② r16 / attn        16    attn-only    13,631,488  ≈ 13.63M
③ r16 / attn+mlp    16    attn+mlp     41,943,040  ≈ 41.94M
④ r64 / attn+mlp    64    attn+mlp    167,772,160  ≈ 167.77M

①→② r×2  → params ×2.00   (r is linear)
②→③ +mlp → params ×3.08   (perimeter, not the full-FT ×3.5 area)
③→④ r×4  → params ×4.00   (r linear again)
```

These params are **certain** (pure arithmetic). The adapter-on-disk sizes are a prediction
that depends on the serialize dtype — see `predictions.md` (fp32 vs bf16 bet, resolved on
GX10 from the measured file size, A2-style).

---

## One-screen summary (the whole thing)

```
model → 32 layers → 7 W per layer (attn: q k v o | mlp: gate up down)
W shape = [d_out, d_in]   (square only when d_out==d_in, e.g. q/o — a special case)
LoRA bolts an (A,B) branch beside chosen W's:  output = W·x + (α/r)·B·A·x
  r       = waist of the branch         (how big an adapter)
  α/r     = scalar on the branch        (how hard it's added; α alone has no position)
  targets = which W's get a branch       (how many adapters: attn-only=128, attn+mlp=224)
cost per W:  full = d_out×d_in (area)  |  LoRA = r×(d_in+d_out) (perimeter)
```
