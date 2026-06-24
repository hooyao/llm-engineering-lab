# A6 — learning notes (LoRA: rank / alpha / target modules) — IN PROGRESS (front half taught)

> Same per-learner format as A2/A4/A5. **STATUS: front half taught (the LoRA mechanism
> + the two knobs rank & alpha). The hands-on sweep (4 configs, adapter sizes, gen
> quality) is NOT done yet.** A new session continuing A6 should: read this file, then
> read `notes/curriculum-v2-execution.md` § A6 for the sweep spec, and run the 4-config
> sweep as the payoff. The learner already USED LoRA all through A5 (3B + LoRA r=16);
> A6 opens it up.
>
> Pacing rule for this learner (proven A2→A5): one small segment, let them clarify in
> place, fold Q&A back, anchor in linear-regression / systems trade-offs, state
> direction+scale explicitly, math terminal-safe. They derive well from algebra and
> ask "why this design not that" — reward that.

---

## What A6 teaches (one line)

LoRA's two knobs — **rank `r`** (how much capacity the update has) and **alpha `α`**
(how strongly the update is applied) — plus **target modules** (which weight matrices
get an adapter). Today covered the mechanism + r + α. The sweep (r=8/16/64, attn-only
vs attn+mlp) is the unfinished payoff.

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

## Segment 2 — knob 1: rank `r`

`r` = the "waist" of the two skinny matrices:
```
ΔW ≈ B(d×r)·A(r×d)    r small = thin waist = few params, low-complexity correction
                      r large = thick waist = more params, richer correction
```
- r is yours to set: r=8 → fewer params, smaller adapter, but more limited ΔW; r=64 →
  more params, richer, approaches full fine-tune, can overfit small data.
- The A6 sweep tries r=8/16/64 to see this trade (underfit ↔ overfit/diminishing return).

## Segment 3 — knob 2: alpha `α`, and why it feels like learning rate

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

So lr is a knob on the training PROCESS (gone after training); α/r is a knob on the model
STRUCTURE (welded into forward, active at inference).

**Why a separate α/r instead of just lr:** the `/r` makes the branch's overall scale
INDEPENDENT of r — bump r 16→64 (B·A grows in magnitude) and the `/r` rescales it back,
so you can tune r (capacity) and α (strength) relatively independently without re-searching
lr each time you change rank. (LoRA paper's deliberate design.)

- **Open for the new session — learner did NOT yet explicitly confirm** the α/r vs lr
  distinction landed (it was the last thing before they went to sleep). Re-check it lands,
  then proceed to the sweep.

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
