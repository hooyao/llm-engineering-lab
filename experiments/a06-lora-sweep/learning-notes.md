# A6 — learning notes (LoRA: rank / alpha / target modules) — THEORY DONE, sweep pending

> Same per-learner format as A2/A4/A5. **STATUS (2026-06-25): all three hyperparameters
> taught (rank, alpha, target modules) AND the full prediction table hand-derived offline.
> The only thing LEFT is running the 4-config sweep on GX10 to fill the MEASURED column.**
> GX10 was unreachable this session, so by the learner's call we finished 100% of the
> theory offline first; the metal run is a separate machine + a later session (Task #3).
> A new session continuing A6 should: read this file, then `predictions.md` (the table to
> verify), then `notes/curriculum-v2-execution.md` § A6 for the sweep spec. The learner
> already USED LoRA all through A5 (3B + LoRA r=16); A6 opened it up.
>
> Pacing rule for this learner (proven A2→A5): one small segment, let them clarify in
> place, fold Q&A back, anchor in linear-regression / systems trade-offs, state
> direction+scale explicitly, math terminal-safe. They derive well from algebra and
> ask "why this design not that" — reward that.

---

## What A6 teaches (one line)

LoRA's two hyperparameters — **rank `r`** (how much capacity the update has) and **alpha `α`**
(how strongly the update is applied) — plus **target modules** (which weight matrices
get an adapter). All three now taught + the prediction table hand-derived. The sweep
(r=8/16/64, attn-only vs attn+mlp) is the pending metal run (`predictions.md` Task #3).

---

## Segment 0 — calibration: the learner derived LoRA's bet AND its boundary

Two priming questions before any mechanism:

- **Q1: why is fine-tuning's update ΔW "low information" (vs training from zero)?**
  Learner: "the base model already learned almost all the world's knowledge; the
  fine-tuning data is just another expression of that knowledge, so ΔW carries far less
  information than W itself." Correct — that IS LoRA's bet (the update is a small
  re-alignment, not a rebuild).
  - **The learner's own extension (deep, unprompted):** "if the fine-tuning data
    REVERSED world knowledge — e.g. 'the capital of France is London' — it wouldn't
    converge." This is exactly LoRA's BOUNDARY: LoRA assumes ΔW is LOW-RANK (a small
    correction). Overwriting deeply-entrenched pretraining knowledge needs a
    high-information ΔW that two skinny matrices can't express. So the learner derived,
    without being told, that LoRA fits small corrections (style, format, eliciting
    existing ability) and struggles against deep entrenchment — the same thing as talk
    slide 4 ("fine-tune changes behavior easily, installs facts poorly"), made STRICTER
    by the low-rank constraint. Keep this — it's the conceptual core.

- **Q2: ΔW ≈ B(d×r)·A(r×d) with r=16 on a 4096×4096 W — how many numbers, vs the
  original?** Learner: "4096×2×16" = 131,072. Correct. vs 4096×4096 = 16.78M → ~0.78%,
  saving ~99.2%. Tied back: A5's log showed LoRA trainable = 24.3M / 3.237B = 0.75% —
  the learner's 0.78% IS the origin of that 0.75% they'd been running all through A5.

## Segment 1 — where B and A actually come from (the learner's key confusion)

The learner asked: "how do the two skinny matrices B/A come about?" Corrected a
misleading framing (mine): it is NOT "compute a big ΔW then factor it into B·A." It is
"never form a big ΔW at all — drop two small trainable matrices into the layer and let
gradient descent learn them."

How it wires in. A layer is `output = W·x` (W = frozen base weight). LoRA adds a PARALLEL
branch:
```
         ┌──────────────────────┐
  x ──┬──┤  W  (frozen)          ├────────► W·x
      │  └──────────────────────┘            │
      │                                      + ──► output
      │  ┌─────┐   ┌─────┐                    │
      └──┤  A  ├───┤  B  ├─────────────────────┘
         └─────┘   └─────┘
          r×d       d×r
        output = W·x + B·(A·x)
```
- **W frozen** — used in forward, but no gradient, no update → the 6.4 GB fixed part /
  0.75% trainable seen in A5.
- **A, B are new trainable params**, randomly init (B init = 0 so B·A = 0 at step 0 → LoRA
  starts as a no-op, smooth departure from the base model).
- forward: x goes BOTH ways, `W·x + B·(A·x)`, summed.
- backward: gradient flows ONLY into A and B (W frozen). Optimizer + m/v only for those
  → A5's 24M trainable.

**So "ΔW ≈ B·A" is the RESULT, not the method.** The big 4096×4096 ΔW is never formed in
memory — only its two skinny factors. THAT is why LoRA saves memory: the large update
never exists; only the low-rank factors do. `r` (the waist) caps how complex the
correction can be.

- **Learner confirmed** the parallel-branch picture (W·x + B·A·x, B/A trained from scratch).

## Segment 2 — hyperparameter 1: rank `r`

`r` = the "waist" of the two skinny matrices:
```
ΔW ≈ B(d×r)·A(r×d)    r small = thin waist = few params, low-complexity correction
                      r large = thick waist = more params, richer correction
```
- r is yours to set: r=8 → fewer params, smaller adapter, but more limited ΔW; r=64 →
  more params, richer, approaches full fine-tune, can overfit small data.
- The A6 sweep tries r=8/16/64 to see this trade (underfit ↔ overfit/diminishing return).

## Segment 3 — hyperparameter 2: alpha `α`, and why it feels like learning rate

LoRA actually uses `ΔW = (α/r)·B·A` — a scaling coefficient on the whole LoRA branch.

Learner's question: "I can't see the relation between α/r and learning rate." Resolved by
splitting SAME-vs-DIFFERENT:

**Same (the learner's instinct is right):** both are a multiplicative scale; bigger →
the correction hits harder. There's a well-known result that **tuning α and tuning the
LoRA learning rate are largely redundant** — doubling α ≈ doubling the LoRA lr. The
learner's "they feel alike" caught a real overlap.

**Different (the key):**
| | learning rate | α/r |
|---|---|---|
| acts on | every param's update | only the LoRA branch's output |
| when | each training step (a dynamic process) | baked into the forward pass, always multiplying |
| changing it affects | how fast you converge | how much the LoRA correction ultimately weighs |
| still there after training? | no (training-only) | YES — inference still computes `W·x+(α/r)·B·A·x` |

So lr is a hyperparameter acting on the training PROCESS (gone after training); α/r acts on the model
STRUCTURE (welded into forward, active at inference).

**Why a separate α/r instead of just lr:** the `/r` makes the branch's overall scale
INDEPENDENT of r — bump r 16→64 (B·A grows in magnitude) and the `/r` rescales it back,
so you can tune r (capacity) and α (strength) relatively independently without re-searching
lr each time you change rank. (LoRA paper's deliberate design.)

- **CONFIRMED (2026-06-25):** the α/r-vs-lr distinction landed. Learner: "learning rate
  没有了，α/r 是 lora 的结构...参数，肯定还在啊" — got both the direction (lr gone, α/r
  survives) AND the reason (α/r is structural / welded into the forward pass, lr is
  training-only). ✓
- **One refinement added (size vs strength — pre-empts weak spot #2, scale-fusion):** the
  learner called α/r a "结构大小参数". "结构" is right (survives to inference); "大小" fuses
  the two hyperparameters. Disambiguated: `r` = size/capacity (hyperparameter 1: rank),
  `α` = strength/scale of the branch (hyperparameter 2: alpha, entering as α/r). Two
  different hyperparameters. Load-bearing because **all 4 sweep configs have
  α=2r → α/r=2 held CONSTANT**; the sweep pins strength and varies only capacity (r) +
  target modules. Flagged this so the learner reads the sweep as isolating capacity, not
  strength.

---

## Segment 4 — where r / α / α/r physically sit in the architecture (learner asked for a diagram)

The learner could not picture the three hyperparameters on the actual data flow. The
diagram that landed (one linear layer). **NOTE — fix to my original drawing:** I first drew
the frozen block as a SQUARE `W d×d` without saying I was silently using q_proj/o_proj (the
square special case) as the example — this confused the learner (a general W is NOT square).
Corrected to the general `[d_out, d_in]` form below; see `teaching-notes.md` §1 for the
square-is-a-special-case point in full.

```
  x ──┬───────────────► [ W  [d_out,d_in]  frozen ] ───────────► W·x ──┐
(d_in)│                                                         (d_out) │
      │                                                                 ▼
      │   ┌ A [r,d_in] ┐   ┌ B [d_out,r] ┐   ┌ scalar ┐             ( + ) ──► output
      └──►│   down     │──►│     up      │──►│ ×(α/r) │──────────────►       (d_out)
          └────────────┘   └─────────────┘   └────────┘
            A·x: [r]          B·A·x: [d_out]     [d_out]
              ▲                                     ▲
        r = bottleneck width             α/r = how hard the whole branch is added
```

The three quantities, located:
- **`r`** = the dim of the squashed vector BETWEEN A and B (the "waist"). A: d→r, B: r→d.
  It has a visible position on the diagram.
- **`α/r`** = the scalar multiply box on the branch output, before the add. Also a visible
  position.
- **`α` itself has NO independent position.** This was the key fix: α is not a circuit
  location — it's just one of the two numbers you divide to GET the α/r scalar
  (`α/r = your α ÷ your r`). On the architecture you can only point at **r (the waist)** and
  **α/r (the scalar box)**; α is an input to dialing α/r, not a place.
- Tie-back: this is why the sweep sets α=2r everywhere → the scalar that actually enters
  forward, α/r, is pinned to 2.

## Segment 5 — W shape = [d_out, d_in], and x is the input activation (learner's "shape?" question)

The learner asked why I'd drawn `W d×d` and what x's shape is. Two fixes:

**My sloppiness corrected:** `W d×d` was lazy. W is generally **not square** — it's
`[d_out, d_in]`. I used 4096×4096 because q/o happen to be square; that's a special case,
not the rule.

**x is the input activation** (the vector the previous layer emitted, fed into this linear
layer). Shapes, pinned with one token first (batch/seq_len dropped):
```
x       = [d_in]          input activation vector
W       = [d_out, d_in]   matrix
y = W·x = [d_out]         output activation vector
```
Why W must be `[d_out, d_in]`: its job is to turn a d_in-vector into a d_out-vector, so
columns = d_in (to multiply in), rows = d_out (to emit). The shape is forced by "how many
in, how many out", not chosen. Dim accounting: `[d_out,d_in]·[d_in] → [d_out]` (the two
d_in's cancel).

Real Llama-3.1-8B shapes (d_model=4096, d_ff=14336) — and this is where the learner's own
instinct "a transformer isn't a single matrix" pays off:
```
q_proj/o_proj  [4096,4096]    square  ← the special case I'd lazily drawn
k_proj/v_proj  [1024,4096]    not square (GQA — narrower; deferred to B4)
gate/up_proj   [14336,4096]   not square — MLP widens 4096→14336
down_proj      [4096,14336]   not square — MLP narrows back
```
The learner spotted k/v are `1024` not `4096` and wrote the formula with `1024+4096`
correctly — did NOT autopilot to `4096+4096`. (Strength: reads the actual numbers.)

With batch+seq_len added back, **W is unchanged** (W is the layer's fixed parameter,
independent of how many tokens flow):
```
x = [batch, seq_len, d_in]    W = [d_out, d_in]    y = [batch, seq_len, d_out]
```
W's size is `d_out×d_in` only — no batch/seq_len. (Contrast: activation size DOES carry
batch×seq_len — that was the A5 sweep axis. The two scale on different things.)

## Segment 6 — transformer = a STACK of linear layers; LoRA fits into each W (the learner's real question)

The learner's question, verbatim concern: "a transformer isn't a single matrix — how does
LoRA fit in?" Correct instinct. The direction fix: **a transformer CONTAINS many W's; a W
does not contain a transformer.**

Scale pinned with explicit counts (pre-empts weak-spot #2, scale-fusion):
```
1 model (8B)
  └─ ~32 transformer layers stacked
       each layer has 7 linear layers (each holds 1 W):
         attn group (4): q_proj, k_proj, v_proj, o_proj
         mlp  group (3): gate_proj, up_proj, down_proj   (mlp = the old net from A2)
```
"attn" and "mlp" are just **group names for these 7 matrices** — nothing more (what
attention does → B4). LoRA doesn't understand the transformer; it just finds the W's and
drops the Seg-4 branch beside the chosen ones.

**A LoRA adapter = the (A,B) pair attached to one W.** One W → one adapter. So the number
of adapters = how many W's you choose to attach to.

**target modules = which W's get an adapter** — the **3rd hyperparameter**:
```
attn-only:  adapt only the 4 attn W's per layer  → 32×4 = 128 adapters; mlp stays frozen
attn+mlp :  adapt all 7 W's per layer            → 32×7 = 224 adapters
```
Un-adapted W's stay frozen: used in forward, never updated.

Three hyperparameters, one line each:
```
r              bottleneck width inside each adapter   → how big one adapter is
α (→ α/r)      strength of each adapter's branch       → how hard one adapter is added
target modules which W's get an adapter               → how many adapters
```

## Segment 7 — area vs perimeter: why LoRA makes the expensive mlp matrices affordable

The crux the learner derived. For ONE W:
```
full fine-tune pays:  d_out × d_in           ← AREA (product of the two dims)
LoRA pays:            r × (d_in + d_out)      ← PERIMETER (sum of the two dims) × r
```

The learner first guessed (from the full-FT view) that mlp's down_proj is "3× and then
some" bigger than q_proj — **right for full FT**: area ratio `(14336×4096)/(4096×4096) =
3.5`. But under LoRA the same comparison is only `294K/131K = 2.25`, because 14336 drops
from a **multiplier** to an **addend**. That collapse is the whole point: full FT can't
afford to touch mlp (area 14336×4096 = 58.7M per W); LoRA can (perimeter r×(14336+4096)).

The learner then wrote the per-layer formula himself, including the narrow k/v:
```
attn-only per layer = (4096+4096 + 1024+4096 + 1024+4096 + 4096+4096) × r = 26,624 × r
attn+mlp  per layer = 26,624×r + (14336+4096)×3×r                        = 81,920 × r
```
×32 layers (identical within a config):
```
attn-only whole model =   851,968 × r
attn+mlp  whole model = 2,621,440 × r
```

The 4 configs and the resulting **certain** param counts (full table + adapter-size
prediction in `predictions.md`):
```
① r8/attn      851,968×8     =   6.82M
② r16/attn     851,968×16    =  13.63M
③ r16/attn+mlp 2,621,440×16  =  41.94M
④ r64/attn+mlp 2,621,440×64  = 167.77M

①→② r×2  → ×2.00   (r is linear)
②→③ +mlp → ×3.08   (perimeter, not the ×3.5 area)
③→④ r×4  → ×4.00   (r linear again)
```

**The adapter-dtype bet (resolve on GX10, A2-style).** adapter MB = params × bytes/param.
- Learner bet **fp32**: few hundred MB so no pressure to save, and "fp32 carries more info."
- Tutor bet **bf16**: A/B are *weights* (same class as base W); base is bf16, so at
  inference `(α/r)·B·A·x` adds onto a bf16 `W·x` and any fp32 precision is truncated on that
  add — unusable. Real driver = dtype-alignment with base, not space. Likely bf16 default.
- The learner's "space doesn't matter" reasoning is CORRECT; it just doesn't decide the
  dtype (alignment does). Open because PEFT's default is version-dependent — must not be
  quoted from memory; the measured file size on the box decides. Both predictions in
  `predictions.md`, MEASURED column waits for Task #3.

---

## What's LEFT for A6 (the unfinished payoff — do this next session)

Per `notes/curriculum-v2-execution.md` § A6: an 8B base, `tulu-3-sft-mixture` ~5000
samples, **4 configs**:
- r=8,  α=16, attn-only
- r=16, α=32, attn-only
- r=16, α=32, attn+mlp
- r=64, α=128, attn+mlp
For each: trainable params, adapter checkpoint size on disk, final loss, qualitative gen
on ~5 prompts. **Deliverable + payoff:** a table comparing the 4, and "which would you
ship and why." The payoff form is co-designed with the learner ON THE DAY (per CLAUDE.md)
— likely "see adapter size + gen quality move as r/targets change," shown as something the
learner reads themselves. Knob 3 (**target modules**: attn-only vs attn+mlp) gets taught
during the sweep — it's the third axis, not yet covered.

Note: A6 spec says 8B; the box has Llama-3.1-8B-Instruct and Qwen3-8B. Confirm dataset
`tulu-3-sft-mixture` availability on the box (may need download) before the run.
