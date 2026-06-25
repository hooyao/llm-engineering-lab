# Talk: "Fine-tuning isn't magic" — built around the two things that block engineers

**Audience:** experienced software engineers who are **AI power users** — they've tried
every application-layer trick (prompting, RAG, agents, tool use) and at work reach for
frontier models (Opus 4.8) by default. They are **AI-fluent but model-internals-naive**:
expert at using these models from *outside* the API, new to what happens *below* it. They
have NOT fine-tuned by hand / locally. They respect **mechanism over conclusion** — for
every claim, show *why it's true*, not just *that it's true*.

> **This is a pure KNOWLEDGE-SHARING talk. There is no ROI / cost case for the audience.**
> Their tokens are company-paid and they'll keep using Opus 4.8 — nobody here switches to
> a fine-tuned 7B to save money, and we do NOT suggest they should. The payoff is
> *understanding*, not a call to action: by the end they grasp the mechanism well enough
> to see *why some companies in the market do choose to fine-tune* — the engineering
> trade-off behind it — even though it's not a choice this room needs to make. Do not
> frame any slide as "you should fine-tune to save cost."

> **Altitude rule (critical — this audience is NOT AI-naive):** do NOT explain what RAG,
> prompting, or agents ARE. They use these daily, better than most. Explaining them is
> condescending and burns your credibility in the first 5 minutes. The novel content for
> this audience is strictly **below the API line**: (a) the *mechanism* — why the tools
> they already use behave as they do, at the weight level; (b) the *memory math* — which
> they've never needed because they've only ever called an API; (c) the *insight* — the
> mechanism explains a thing they've heard but maybe didn't believe: a fine-tuned small
> model really can rival a frontier model *in a narrow vertical*. That's an "aha," not
> advice. Frame everything as "you know the outside; here's the inside."

**Length:** 30 min, not counting Q&A.
**Thesis:** fine-tuning is not mysterious. Two questions stop people from trying it.
Answer those two — *from first principles* — and the wall is gone.

> This file is the talk's BLUEPRINT — narrative spine + per-slide idea + the MECHANISM
> for each + transitions + the real numbers to show. It is NOT the slides yet; it's
> what the slides get built from. Talk is delivered in 中文; this working file is
> English per repo convention. Keep ML terms in English on the slides too
> (weight / bias / gradient / parameter / logit).
>
> **Depth rule (per the audience):** every slide carries a `原理 (mechanism)` beat —
> the *why* an experienced engineer would want. The talking points state the
> conclusion; the mechanism beat earns it. If a slide has a conclusion but no
> mechanism, it's not finished.
>
> Every measured number here is from the learner's OWN A1/A2/A3 runs on the GX10.
> Estimates are marked as estimates (see honesty markers); never present an estimate
> as a measurement.

---

## The spine — the ONE thing that makes this talk coherent

Do not organize this as "8 topics about fine-tuning." Organize it as **two questions a
colleague has, answered from first principles, in order.** Every slide earns its place
by serving one of these two. If a slide serves neither, it is cut.

```
The two things blocking a colleague who hasn't tried fine-tuning:

  BLOCKER 1   "What does fine-tuning even turn my model into?
               Do I actually need it — or can a better prompt / RAG do the same?"

  BLOCKER 2   "For a given model, how much VRAM does it take, and what GPU do I need?"
               ← this is the CORE of the talk. Memory math is the centerpiece.

The whole talk is: answer 1 (with the mechanism), answer 2 (with the mechanism),
then "so go try it."
```

**The arc, with the transition between every beat written out** (this is the talk):

```
HOOK     "fine-tuning is the same process as pretraining, just less of it" → two
          questions stop everyone. Today = these two.

MAP      "Fine-tuning is ONE box on a bigger map." Two orthogonal axes:           ~1.5 min
  1  Axis 1 = which STAGE (pretrain → SFT / preference / distillation);
     Axis 2 = how MUCH (full / LoRA / QLoRA). "SFT"≠"LoRA" — different questions.
       原理: de-fuse the words; distillation = where answers come from (black-box=SFT,
             white-box=match logits). Locates today = SFT × (full→LoRA→QLoRA).
                                                          → "into that one box. what does it change?"

ACT 1 — DO I NEED IT?  (Blocker 1)                                       ~11 min
  2  Demo: before vs after. It changed STYLE.            → "did it learn anything NEW?"
  3  Pivot: #5 "Paris" byte-identical. Facts did NOT move.
       原理: signal vs entrenchment — gradient only moves what you push on.
                                                          → "so how do I steer it?"
  4  You DEMONSTRATE behavior, you don't program it (the data + loss masking).
       原理: fine-tuning reweights a distribution; it doesn't add a capability.
                                                          → "then when is it the wrong tool?"
  5  Decision: new facts→RAG | behavior→fine-tune | small model rivals big one on a vertical.
       原理: WHY weights are a bad knowledge store, WHY RAG works, + the specialist "aha".
                                                          → "OK, can my machine run it?"

BRIDGE   "That last question is the real wall. Good news: it's multiplication."  ~0.5 min

ACT 2 — HOW MUCH VRAM / WHAT CARD?  (Blocker 2 — CENTERPIECE)             ~14 min
  6  The unit: a parameter is ONE number. (neuron = w*x+b.)
       原理: why depth needs the nonlinearity; where the billions come from.
                                                          → "how many bytes per number?"
  7  Inference: 2 bytes/param. 70B → 140 GB to load.
       原理: what's in the 2 bytes; the KV cache is the other inference cost.
                                                          → "training needs more. why?"
  8  Training: 12 bytes/param. param + gradient + optimizer (m, v).   ← CORE
       原理: what gradient / m / v actually ARE, and why optimizer state dominates.
                                                          → "70B = 840 GB. impossible. now what?"
  9  Shrink the multiplier: full 12 → LoRA 2 → QLoRA 0.6.
       原理: the weight UPDATE is low-rank — train two skinny matrices, freeze the rest.
                                                          → "which fits a card I can get?"
  10 Map to real GPUs. 16 GB gaming card does 7B QLoRA; rent an A100 for $1/hr.
       原理: every cell is derived — VRAM / bytes-per-param. not memorized.
                                                          → "does the arithmetic actually hold?"
  11 Proof: predicted 12 GB, measured 13.84.
       原理: the gap = activations, the one term ×12 omits (workload-dependent).

CLOSE    Both blockers answered, from principle. Open tools, ~100 lines. Go try it.  ~2 min
```

Two optional deep-dives live OFF this spine, pulled in ONLY if time allows or a
question demands — they must never interrupt the arc:
- **(D1) Why optimizer state needs fp32 while everything else is fine at 16-bit.** A
  detail of slide 8. First thing cut if long; ideal Q&A answer.
- **(D2) The fp32 master-weight copy (12 vs 16 bytes/param).** Be ready for the
  engineer who's read about mixed precision — see slide 8's honesty note.

---

# HOOK

**On slide:** *"微调没那么神秘 — 一台桌面设备 + 一个下午就能玩"*

> **Hook line removed (2026-06-25, learner's call):** the old subtitle "I fine-tuned a 1B
> model in 83 seconds on a box on my desk" / "83 seconds · 13.84 GB · a box on my desk"
> was cut as self-congratulatory and content-free. The hook page now carries only the
> title + the two-line subtitle (what fine-tuning is + "same process as pretraining, just
> less of it"). The 13.84 GB measurement is NOT lost — it stays where it does real work, as
> the proof point on slide 11 (predicted 12 vs measured 13.84). Do not reintroduce the
> stat-drop on the title slide.

A pretrained model (Llama, Qwen) already knows language and facts — someone spent
millions of GPU-hours on that, and you didn't pay for it. **Fine-tuning takes that
finished model + your own data → nudges its behavior.** Not training from zero.

**原理 (demystify up front):** fine-tuning is *the exact same process* as pretraining —
predict the next token, measure error, nudge the weights — just run on a tiny amount of
*your* data instead of the whole internet. There is no second, mysterious mechanism.
Same gradient descent, fewer steps, your examples. That's the whole reason it's cheap
and accessible: you're not building the engine, you're tapping the brakes on one that's
already running.

**Open by naming who they are (don't condescend — leverage their expertise):**
> "Everyone here is an AI power user. You've built with RAG, agents, every prompting
> trick. But all of that is from *outside* the API — you call the model, you never open
> it. Today we go one layer down: the same model, but running on your own box, with you
> changing its weights. Two questions are the only things standing between you and doing
> that — and you already half-know the answers from the outside; I'm going to show you
> the inside."

**Then plant the spine explicitly:**
> "Question one: *what does fine-tuning actually change — and given everything you can
> already do with a prompt and RAG, when is it even the right tool?* Question two: *how
> much GPU does it take — can the machine on your desk run it at all?* That's the whole
> talk. I answer both from first principles, and you walk out able to try this yourself."

Transition → "But 'fine-tuning' is just *one box* on a bigger map. Before the two
questions, let me put the map up — and clear up a few words this room uses
interchangeably — so you know exactly where today sits."

---

# THE MAP (orientation — before the two questions)

## Slide 1 — "Fine-tuning" is one box on a bigger map

**← from hook:** "you keep hearing SFT, LoRA, RLHF, distillation thrown around. They're
not competing options on one list — they live on TWO independent axes. Here's the map."

This is an **orientation slide, not a topics list.** It earns its place (the spine cuts
anything that serves neither blocker) by doing one job: it *locates today* on the map and
de-fuses words the audience mixes up, so Act 1 ("which stage = SFT") and Act 2 ("how much
= full/LoRA/QLoRA") each land as a named axis instead of a new topic. It is NOT a survey
of training methods — keep it to the two axes and move on.

**The one idea: two ORTHOGONAL axes.**
```
Axis 1 — WHICH STAGE / what signal it learns from:
   Pretraining     next-token on the whole internet → the base model (someone else paid)
     ↓ post-training, on YOUR data:
   SFT             imitate (instruction → answer) pairs            ← TODAY
   Preference      RLHF / DPO: learn from chosen-vs-rejected
   Distillation    copy a stronger TEACHER model

Axis 2 — HOW MUCH you touch (applies to ANY stage above):
   Full            every parameter      ×12 bytes
   LoRA            ~1% added slice       ×2
   QLoRA           LoRA + 4-bit base     ×0.6                       ← Act 2
```

**原理 (the de-fusing — this is what the audience actually gets):**
- **The two axes are orthogonal.** Picking on Axis 1 doesn't pick on Axis 2:
  `full-SFT`, `LoRA-SFT`, `LoRA-DPO` are all real combinations. "Is this LoRA or SFT?"
  is a category error — like "is this car a sedan or a diesel?" Two different questions.
- **"SFT" answers *which stage*; "LoRA" answers *how much*.** Engineers conflate them
  because they hear both as "ways to train." Naming the two axes is the whole payoff.
- **Preference / RLHF / DPO (presenter ammo — name the difference from SFT in one line,
  do NOT expand; today's spine is SFT).** The essential difference: SFT gives **one gold
  answer to imitate**; preference gives **two answers and only says which is better** — it
  learns a *relative* preference, not imitation of a single answer. Used where there is no
  single correct answer, only "this is more helpful/safer than that" (aligning tone,
  helpfulness, harmlessness). Two implementations:
  - **RLHF** (Reinforcement Learning from Human Feedback) = the classic 3-step pipeline:
    SFT first → train a **reward model** on human chosen-vs-rejected labels (it learns to
    *score* an answer) → use RL (PPO) to push the model to *maximize that reward*. Three
    models, heavy pipeline.
  - **DPO** (Direct Preference Optimization) = **skips the reward model and the RL**;
    folds the preference straight into one classification-style loss trained on the
    chosen-vs-rejected pairs in a single step. The 2024-on default — simpler, more stable,
    quality close to RLHF. One-liner: "RLHF = train a scorer then RL against it; DPO =
    drop the scorer, train directly on the pairs — that's why almost everyone uses DPO now."
  - (This is curriculum A9 / B9–B11 hands-on; today only name it. Keep it off the main path.)
- **Data shape + where "supervised" comes from (a question this audience WILL have —
  "isn't all training data Q&A?").** Pretrain data is NOT instruction→answer — it's
  **raw continuous text** (scraped web / books / code), trained on plain next-token
  ("The capital of France is ___" appeared millions of times → Paris gets carved in; it
  was never *taught* as a Q&A item). The "answer key" is where the supervision label
  comes from, and that's literally what the words mean:
  - **Pretrain = self-supervised** — the label is the text's *own next token*, free, no
    human annotation.
  - **SFT = supervised** — each row's `answer` half is a **human- (or teacher-) written
    correct answer** the model aligns to; that written answer IS the supervision. *That's
    the "S" in SFT.* (Preference/DPO is supervised too, but by chosen-vs-rejected pairs,
    not one gold answer.)
  - Mechanism is the SAME next-token throughout (ties back to the hook's "same process");
    what changes across Axis 1 is the **data shape** (unstructured text → Q&A pairs) and
    **where the loss is computed** (all tokens → answer-half only = loss masking, slide 4).
- **Distillation is NOT a second mechanism — it only sets where the training answers
  COME FROM.** This is the one the learner asked about; answer it precisely, two kinds:
  - **black-box (response-based):** the teacher *generates text*; you take
    `(prompt, teacher's answer)` and run **ordinary SFT** on it. **This is exactly the
    "synthetic-data SFT" that's hyped online** — Alpaca (GPT-3.5 generated 52k rows to
    train LLaMA) is this. So "isn't distillation just using another model's output to SFT
    your own?" → **yes, that's black-box distillation; it IS SFT, just with model-written
    `output` instead of human-written.**
  - **white-box (logit-based — Hinton 2015, the original meaning of the word):** the
    student matches not the teacher's *single chosen token* but the teacher's *entire
    ~120k-long probability distribution* (logits / soft targets), via KL + a temperature.
    The soft info ("2nd-likeliest 20%, 3rd 5%") carries far more than one hard token.
    **Constraint:** you need the teacher's logits — distilling GPT-4 can only be black-box
    (the API won't give full logits); white-box needs a teacher whose internals you own.
  - (This is exactly what curriculum B16 builds hands-on, both kinds. Today: name it, move on.)

**Altitude check:** do NOT explain what RLHF/RAG ARE here — the audience uses them. This
slide only *arranges* the words they already know (below-the-API tidying), it doesn't
teach the words. If it starts feeling like a survey, you've overstayed — the payoff is
the two-axes framing plus the distillation de-fuse, nothing more.

Say: "'SFT' tells you *which stage*; 'LoRA' tells you *how much of the model you touch* —
different axes, stop fusing them. Today we zoom into one box: **SFT**, and walk it down
the second axis — full → LoRA → QLoRA — for the memory math."

Transition → "So we're inside one box now — SFT — and we'll read its memory cost down
that second axis. First question of the two: what does it even change? Let me show you."

---

# ACT 1 — DO I NEED IT?  (Blocker 1)

## Slide 2 — The demo: before vs after (the hook made real)

**← from hook:** "what does fine-tuning turn the model into? Look:"

Same prompts, base Llama-3.2-1B-Instruct vs the learner's 500-example fine-tune. Show
2–3 pairs verbatim from `experiments/a03-eval-1b/results.md`:

- **#6 "make this more polite"** — BEFORE: preamble + 3 variants + closing paragraph.
  AFTER: one direct sentence. → fine-tuning made it *direct*.
- **#1 "three tips"** — BEFORE: "Here are three tips…" + wordy. AFTER: straight to
  "1. 2. 3."

**原理 (why STYLE is the thing that moved):** every one of the 500 training examples is
terse and preamble-free. So "drop the preamble, list directly" is a pattern reinforced
by *all 500* — a strong, consistent gradient pushing in one direction. Style is carried
by lots of tokens and is low-specificity, so it shifts easily. Hold that thought —
it's exactly why the *next* thing did NOT move.

Transition → "Sure, the *style* changed. But the question an engineer should ask next:
did it actually learn anything new? Did it get smarter?"

## Slide 3 — The pivot: it changed style, NOT facts  ← mechanism slide of Act 1

**← from demo:** "watch what did NOT change."

- **#5 "capital of France"** — BEFORE and AFTER **byte-identical**: "The capital of
  France is Paris."

**原理 (this is the load-bearing mechanism of the whole first half — signal vs
entrenchment):**
```
· "Paris is the capital of France" appeared thousands of times in pretraining
   → that association is carved deep, redundantly, across many weights. Heavily entrenched.
· The 500 fine-tuning examples contain NOTHING about France's capital
   → the gradient pushing on that association is ~zero.
· Gradient descent moves a weight in proportion to (how hard you push) vs (how
   entrenched it already is). Zero push against a deeply-carved value = no movement
   → the relevant weights don't move, so the output stays byte-for-byte the same.
· (Only if asked "how do you know it really didn't move?") The comparison uses
   deterministic decoding, so two identical outputs are themselves the proof that the
   relevant logits didn't change. Don't teach decoding on this slide — it dilutes the
   three-step mechanism.
```
Say: "Most important slide in the first half. Style changed completely; facts didn't
budge — not one character. The model only moves what its gradient pushes on. 500 terse
examples pushed hard on *style* and not at all on *the capital of France* — so style
moved and Paris didn't. That single rule decides whether you should fine-tune at all."

> Honesty beat (engineers trust you more for it): conciseness has a price. Show **#10** —
> the tuned model dropped the recipe steps. Fine-tuning is a trade-off: more direct, but
> it can follow instructions less strictly and drop content. (learner's own A3 read.)

Transition → "If fine-tuning only moves what you push on, then *how you push* — the data
— is the entire lever. That's the most hands-on slide today."

## Slide 4 — How you steer it: you demonstrate, you don't program

**← from pivot:** "you just saw it moves only what you push on — here's how you push."

- Fine-tuning data is just **(instruction → the answer you wish it gave)** pairs. You
  hand it examples of the behavior you want. No code that says "be concise."
- The A2 run used **`yahma/alpaca-cleaned`**, 500 of its 51,760 rows.

**Show one real row** (most concrete slide of Act 1):
> **[[FILL: one real alpaca-cleaned row — instruction / (input) / output. You'll grab
> this.]]**

**原理 (two mechanisms an engineer will want):**
- **Loss masking — the model is graded ONLY on the `output` tokens, not the
  instruction.** Why: you want it to learn `P(good answer | instruction)`, not to learn
  to generate instructions. The instruction is context fed in; the loss (the error
  signal) is computed only on the answer half. So you're training "given a request like
  this, produce an answer like that."
- **Fine-tuning REWEIGHTS a distribution; it doesn't ADD a capability.** The base model
  can *already* produce terse, direct text — pretraining saw plenty. Fine-tuning just
  raises the probability of the terse continuation given an instruction. This is the
  deep reason Act 1 hangs together: reweighting toward a style the model already has
  (lots of signal, capability present) is easy; installing a fact it never saw (no
  signal, capability absent) basically doesn't happen — which is the next slide.

Say: "You don't *program* the behavior, you *demonstrate* it — and all you're doing is
shifting the odds toward outputs the model could already produce. Want your house
style? Show it 500 answers in your house style. You're reweighting, not rebuilding."

Transition → "Which tells you precisely when fine-tuning is the WRONG tool. If the
lever is 'reweight toward examples,' then a fact it's never seen has no example to
reweight toward…"

## Slide 5 — The decision: when is fine-tuning the right tool?  ← the conceptual payoff of Act 1

**← from steering:** "…so fine-tuning can teach a *behavior to imitate*, but it's the
wrong tool for a *fact it's never reliably seen*. Here's the decision, with the why:"

| The goal | Right tool | Why (mechanism) |
|---|---|---|
| Add **new facts / private knowledge** | **RAG** | facts live in editable data read at inference, not baked in weights |
| Change **style / format / behavior** | **fine-tune** | reweights toward demonstrated behavior — what the demo did |
| Make a **small model rival a big one** on one vertical | **fine-tune** | concentrates the model onto that task — the "aha" below |

> Note: "try a better prompt first" isn't on this table — you do that in your sleep. The
> table is about the cases where prompting *isn't* the lever, and people still pick the
> wrong one of the remaining two (reach for fine-tuning to add knowledge; *don't* reach
> for it when a specialist small model is exactly the point).

**原理 — WHY fine-tuning is the wrong tool for knowledge (state this carefully, it's
the part they'll push on — they've all *wondered* why people say "don't fine-tune for
facts"):**
```
First, kill the myth honestly: it's NOT that fine-tuning *can't* change a fact —
full fine-tuning updates every weight, you CAN overwrite facts if you push hard.
It's that it's the wrong TOOL, for four mechanical reasons:

1. SIGNAL vs ENTRENCHMENT (slide 3): a few examples can't overpower an association
   that thousands of pretraining examples carved in. Weak push, entrenched value, no move.

2. PUSH HARD ENOUGH AND YOU BREAK THINGS: to force a handful of new facts in, you train
   many epochs on them → catastrophic forgetting (unrelated capabilities get overwritten)
   and overfitting (it memorizes the exact phrasing, not the fact).

3. WEIGHTS ARE A TERRIBLE DATABASE: even when a fact sticks, you can't edit one entry,
   delete it, see where it came from, or scope who can read it. A fact in weights is
   write-mostly and un-auditable. The fact changes next quarter → retrain the model?

4. THE COUNTERINTUITIVE ONE: fine-tuning on facts the model didn't already know
   teaches it to HALLUCINATE. It learns the *form* of a confident answer without the
   content being reliably installed → confidently wrong.
   (Source: Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage
    Hallucinations?", EMNLP 2024, arXiv:2405.05904. Their controlled finding: new-fact
    examples are learned much SLOWER than facts the model already knows, and as they
    finally get learned, they LINEARLY increase the model's hallucination rate. Their
    framing: factual knowledge is mostly acquired in pretraining; fine-tuning teaches
    the model to USE it, not to absorb new facts.)
```
**原理 — WHY RAG works instead (they USE RAG — give them the weight-level reason it's
the right call, which they've never been told):** RAG never touches the weights. It puts
the fact in the *context* at inference and lets the model *read* it. Reading/
comprehension is the deeply-entrenched, reliable capability (massively reinforced in
pretraining); memorizing-a-new-fact-from-a-few-examples is the unreliable one. RAG leans
on the reliable half and keeps volatile knowledge in data you control — editable,
deletable, auditable, access-controlled. "You already reach for RAG over fine-tuning for
knowledge; *this* is the mechanism that makes that instinct correct."

**原理 — the INSIGHT this unlocks (the real payoff of the slide — an "aha," NOT advice):**

*First, reconcile the apparent contradiction head-on — someone WILL ask "if fine-tuning
can't inject knowledge, how does a fine-tuned 7B rival a big model?":*
```
No contradiction — the two claims govern different things:
  · "can't inject knowledge"  → governs WHAT the model KNOWS (world facts: Paris)
  · "7B can rival a big model" → governs HOW it USES what it has (skill / behavior)

Those vertical tasks (classify, extract, reformat, discriminate) don't test KNOWLEDGE —
they test BEHAVIOR: "will it follow your exact spec and format?" The 7B already HAS the
underlying capability from pretraining; it just doesn't emit it your way by default.
Fine-tuning ELICITS / ALIGNS that latent capability (= slide 4's reweight-an-existing-
distribution), it does NOT install a new fact. Knowledge stayed put (slide 3); SKILL got
concentrated.

This is literally the Gekhman paper's own framing: "factual knowledge is mostly acquired
in pretraining; fine-tuning teaches the model to USE it more efficiently." The same paper
backs BOTH claims — that's why it's the strongest citation here.

Boundary (state it, or you over-promise again): this only holds when the task is within
the 7B's EXISTING capability and only alignment is missing. If the vertical genuinely
needs knowledge the 7B never saw in pretraining, fine-tuning won't save it either.
```

Then the mechanism: once you see fine-tuning as *concentrating the distribution*, a
surprising market fact stops being surprising: **a fine-tuned 7B can go toe-to-toe with a
frontier model — but only inside one narrow vertical.** A frontier model spends its
capacity being good at *everything*; on your single task it's using a sliver of that
capacity, and it was never aligned to *your* exact spec. Fine-tuning a small model pours
*all* of its (smaller) capacity into that one task — it gives up being a generalist
(useless at everything else) to become a specialist at one thing. Enough capacity for one
narrow task is a low bar, so the 7B catches up *there*.
> Evidence to cite (this is the slide where someone says "prove it"): **LoRA Land**
> (Zhao et al., Predibase, arXiv:2405.00732, 2024) — 310 LoRA-fine-tuned models across
> 31 narrow tasks. 4-bit LoRA on Mistral-7B (exactly the QLoRA recipe from Act 2)
> **outperformed GPT-4 by ~10 points on average across those tasks.** That's the
> concrete form of "a 7B rivals a frontier model on a vertical."
> Be honest about the scope: this holds for *narrow, well-defined* tasks (classify these
> tickets, extract these fields, output our exact format), NOT open-ended reasoning. A
> 7B will not rival a frontier model in general — only on the one vertical you
> concentrated it on. The LoRA Land win is precisely *per-task specialists*, not one
> model beating GPT-4 at everything.
> Presenter hook (true, and it lands): "I didn't believe a 7B could rival a frontier
> model either — until you see it's not competing on *everything*, just on one task it's
> been pointed entirely at. That's why some companies ship fine-tuned small models in
> production, even though *we* here just call Opus — different problem, different tool."

Transition (bridge into Act 2) → "Say you've ruled out prompt and RAG — you genuinely
need to change behavior, so you're fine-tuning. The very next question is the one that
actually stops people from starting: *can my machine even run it?* Good news — that's
not a mystery, it's multiplication. And I can show you the multiplication."

---

# ACT 2 — HOW MUCH VRAM / WHAT CARD?  (Blocker 2 — THE CENTERPIECE)

> ~half the talk. Build slowly, one step at a time — it's what the audience can use on
> Monday. Every slide answers one question, **"given a model, will it fit on my card?"**
> and the answer is always `param_count × bytes_per_param`, then match to a GPU. Each
> slide adds one factor and explains *why* that factor is what it is.

## Slide 6 — The unit: a parameter is just one number

**← from bridge:** "before counting bytes, agree on what we count. What's a 'parameter'?"

**A neuron is just linear regression.** If you've written `y = w1*x1 + w2*x2 + b`,
you've written a neuron.

```
z = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b
```
| parameter | kind | count |
|---|---|---|
| w1..w4 | weight | 4 |
| b | bias | 1 |
| **one neuron** | | **5 parameters** |

A "parameter" is **one number** — a weight or a bias. "8 billion parameters" = a bag of
8 billion such numbers, arranged in layers.

**原理 (two things an engineer will actually wonder):**
- **Why stack layers — why isn't this just one big linear regression?** Because between
  layers there's a nonlinear function. Without it, stacked linear layers collapse: the
  composition of linear maps is itself linear, so 80 layers would equal 1 layer and
  depth would be pointless. The nonlinearity is what lets depth represent complex
  functions. *That's* the difference between a neuron and a network.
- **Where do the billions come from?** The weights live in *matrices*. One layer mapping
  a 4096-dim vector to 4096-dim is a 4096×4096 matrix = ~16.8M numbers — in a single
  matrix. Stack ~80 layers, several such matrices each → billions. The count we'll
  multiply against is just the total size of all these matrices.

Say: "No magic inside — it's millions of `w*x+b`, the linear regression you did in
school, stacked deep with a nonlinearity between layers so depth means something. Once
it's just 'a count of numbers,' the memory is a napkin multiplication."

Transition → "So how many bytes does one number cost? Start simple — you're not even
training yet, you just want to RUN the model."

## Slide 7 — Just running it (inference): 2 bytes per parameter

**← from unit:** "each parameter is a number. how big is a number?"

- A parameter in **bf16 = 2 bytes** (a 16-bit float). **To run: `param_count × 2`.**

| model | inference memory (×2) |
|---|---|
| 1B  | 2 GB  |
| 8B  | 16 GB |
| 70B | 140 GB |

**原理 (what's in the 2 bytes, and the cost people forget):**
- **bf16 = 1 sign + 8 exponent + 7 mantissa bits.** It keeps fp32's *range* (same 8
  exponent bits, so values don't overflow) but throws away precision (7 mantissa bits ≈
  2–3 significant digits). For a forward pass that's plenty — you're reading the weights
  once, errors don't accumulate across training steps, only flow through layers (which
  are normalized). So inference is happy at 2 bytes.
- **The other inference cost: the KV cache.** Serving long contexts, each token caches
  its key/value vectors so you don't recompute them — that grows with
  `batch × sequence_length × layers × hidden`, *independent* of param count. For a long
  conversation it can rival the model's own size. (Mention only — flag it so the
  serving-curious engineers know `×2` is the floor, not the whole story.)

Say: "A 70B model needs ~140 GB just to load, before doing anything — the model *is* its
parameters, running it costs one copy. Training costs more per parameter. Here's the
part worth memorizing."

Transition → "Why is training more expensive than running? Because for *every single
parameter*, you now store more than just the number. Three more things, in fact."

## Slide 8 — Training: 12 bytes per parameter, not 2  ← THE CORE SLIDE

**← from inference:** "running = 1 copy of each number. training = the number plus three
more, per parameter."

```
RUNNING (inference), per parameter:
  (1) the parameter value      2 bytes  (bf16)              → param_count × 2

TRAINING, per parameter — you ALSO keep, for EVERY number:
  (1) the parameter value      2 bytes  (bf16)
  (2) its gradient             2 bytes  (which way to nudge it)
  (3) optimizer state  m       4 bytes  (fp32)
  (4) optimizer state  v       4 bytes  (fp32)
                              --------
                              12 bytes                       → param_count × 12
```

**原理 (what these four things actually ARE — don't just name them):**
- **gradient** = the partial derivative of the loss with respect to that one weight: the
  direction and magnitude that would most reduce the error. Computed fresh each step,
  used once, discarded. (Used-once → 2 bytes is fine.)
- **m (first moment)** = a running average of the gradient — *momentum*. Smooths out
  noisy step-to-step gradients so training doesn't zig-zag.
- **v (second moment)** = a running average of the gradient *squared* — tracks how large
  this particular weight's gradients have been, so each weight gets its OWN step size
  (weights with big, jumpy gradients take smaller steps; stable ones take bigger). This
  *per-parameter adaptive step size* is the whole point of Adam over plain SGD.
- **Why optimizer state dominates (the 6× story):** items (3)+(4) are 8 of the 12 bytes
  — two-thirds of training memory is the optimizer's running averages, not the model.
  That is *exactly* what LoRA will delete for 99% of params on the next slide.

**The takeaway, say it as the line of the talk:**
> **Training ≈ `param_count × 12 bytes`. Inference ≈ `× 2`. Training is ~6×.**

| model | full fine-tune (×12) | inference (×2) |
|---|---|---|
| 1B  | 12 GB  | 2 GB  |
| 8B  | 96 GB  | 16 GB |
| 70B | 840 GB | 140 GB |

Say: "You can now predict whether a model fits on your card *before downloading a byte*.
Training ≈ param count × 12. The model itself is the small part; the optimizer state is
the hog — and that's the part the next trick throws away."

> (D1) OPTIONAL deep-dive, time/Q&A only: "why are m, v fp32 (4 bytes) when the weight
> is fine at 2?" → m and v *accumulate* over thousands of steps; in 16-bit, once they
> grow, each tiny update gets rounded away ("big eats small": `1024 + 0.5 = 1024` in
> fp16) and training silently stalls. The rule: a number *accumulated over time* needs
> the precision; a number *used once and discarded* (gradient, activation) is fine at
> 16-bit. One rule explains the whole mixed-precision layout. Keep OFF the main path.

> (D2) HONESTY / be-ready note: a fully-correct mixed-precision setup often also keeps a
> 4-byte fp32 *master copy* of the weights → 16 bytes/param, not 12. My 1B run measured
> ~13.84 GB, consistent with the 12-byte regime (+ activations), so 12 is the honest
> teaching number *for my run* — but if someone asks "isn't there an fp32 master copy?"
> the answer is "yes, some setups add 4 more bytes for it; mine measured ~12+activations.
> Verify against your config." Do NOT volunteer this on the main path; it muddies the
> clean ×12. (Cross-check the A2 config before Friday.)

Transition → "Look at the 70B row: 840 GB. No single box has that. So is 70B fine-tuning
just impossible for us? No — and the fix is the only other concept you need."

## Slide 9 — Shrink the multiplier: full → LoRA → QLoRA

**← from the core:** "840 GB is impossible. So we stop paying 12 bytes for 99% of the
parameters. Here's the mechanism that lets us."

| Method | What trains | bytes/param | The idea |
|---|---|---|---|
| **Full** | ALL parameters | **~12** | retrain everything (slide 8) |
| **LoRA** | a tiny ~1% added slice; freeze the rest | **~2** | only the slice gets gradient + optimizer |
| **QLoRA** | LoRA + 4-bit frozen base | **~0.6** | also compress the part you're not training |

**Then the same GB table as slide 8, one row of multiplication per method** (this is the
form the learner asked for — full was already `param × 12`; give LoRA/QLoRA the same
ready-to-read GB numbers, not just the coefficient):

```
          full ×12    LoRA ×2    QLoRA ×0.6
  1B       12 GB        2 GB       0.6 GB
  8B       96 GB       16 GB       4.8 GB
  70B      840 GB     140 GB        42 GB
```
Same multiplication as slide 8 — only bytes/param shrank. **Floor only; activations add on
top (slide 11).** The headline line: "70B full = 840 GB, nothing holds that; the same 70B
under QLoRA = 42 GB, one card. Same arithmetic, smaller coefficient."

**原理 — how LoRA actually works (this is the slide that rewards the audience most):**
```
Full fine-tuning learns a full update ΔW for each weight matrix W — same size as W,
millions of numbers per matrix.

LoRA's bet: that UPDATE is low-rank — it can be written as the product of two skinny
matrices, because the adaptation needed to shift behavior lives in a low-dimensional
subspace (you're nudging, not relearning).

   ΔW  (4096 x 4096  = 16.8M numbers)   ≈   B (4096 x r)  ·  A (r x 4096)
   at rank r = 16:   A and B together  =  2 x 4096 x 16  =  131K numbers  (~0.8% of 16.8M)

You FREEZE W — no gradient, no optimizer state, it just sits there at 2 bytes — and
train only A and B. So the 12-byte tax falls on ~1% of the parameters; the other 99%
cost 2 bytes (frozen weight only). Effective average ≈ ~2 bytes/param.
```
**原理 — QLoRA in one more step:** compress the frozen base from bf16 (2 bytes) to 4-bit
(NF4, 0.5 bytes), dequantizing on the fly during the forward/backward pass; keep the
small A/B adapters in bf16. Now even the frozen 99% costs 0.5 bytes → effective ≈ 0.6
bytes/param. **This is why people fine-tune 70B on a single 24–48 GB GPU.**

Say: "Same move every time — stop paying the full 12 bytes for parameters you're not
training. The arithmetic from slide 8 is unchanged; you're just dropping
bytes-per-param, because the *update* turns out to be low-rank, so you train two tiny
matrices instead of the whole thing."

> Honesty: the learner has DONE full fine-tuning (the 1B run). The LoRA/QLoRA
> bytes-per-param here are the standard per-method figures, not personally measured yet.
> (Cross-check ~0.6 / ~2 against `experiments/a01-mem-budget/notes.md` before Friday.
> Optional: run A6 for a live LoRA adapter-size number — not required, concept is enough.)

Transition → "So you've got three numbers — 12, 2, 0.6 bytes/param. Last step is the one
you actually want: which fits on a card you can get your hands on?"

## Slide 10 — Map it to real GPUs: what card do YOU need

**← from shrink:** "pick the method that fits your VRAM. This table is the literal
answer to 'what card do I need' — and every cell is *derived*, not memorized."

| You have / rent | VRAM | What you can fine-tune | Method |
|---|---|---|---|
| Consumer card (RTX 4060/4070) | 8–12 GB | 0.5–1B full; up to ~7B | QLoRA |
| **RTX 4090 / 3090** (common) | **24 GB** | 1–3B full; 7–13B; up to ~33B | LoRA / QLoRA |
| Rent 1× A100 | 40–80 GB | 7B full; 13–34B; 70B | LoRA / QLoRA |
| Desktop AI box (my GX10) | 128 GB unified | 8B full; 70B | QLoRA |

**原理 (so they see it's not a lookup table to trust on faith):** every cell is
`max_param_count = VRAM / bytes_per_param`. A 24 GB card, full fine-tune (12 B/param):
24 GB / 12 = 2B params → "1–3B full." Same card, QLoRA (0.6 B/param): 24 / 0.6 = 40B →
"up to ~33B" once you leave headroom for activations. You don't memorize this table —
you *derive* any row of it with the two numbers from slides 8 and 9.

Say:
- "A **16 GB gaming GPU** already fine-tunes a 7B model with QLoRA — a card lots of you
  have at home." ← the headline.
- "No card? **Rent an A100 for ~$1–2/hour** (RunPod / Lambda / Vast). A small fine-tune
  is minutes — costs you a coffee."
- "The only thing changing down this table is the *method* — full → LoRA → QLoRA — not
  the idea. Pick the method that fits your VRAM, using the arithmetic from slide 8."

> HONESTY MARKER (hold this line): the ONLY measured number is **1B full SFT = 13.84 GB
> on the GX10**. Every other cell is *estimated from param × bytes* (matches widely
> reported community numbers), NOT personally tested. If asked "did you test the 4090
> row?" → "No. That's the formula's prediction. My one measured point is the 1B run,
> which hit its prediction within 0.2%, so I trust the arithmetic." Never present an
> estimate as a measurement.

Transition → "And 'I trust the arithmetic' deserves proof — everything I just showed is
a prediction. Did it hold when I actually ran it?"

## Slide 11 — The proof: prediction vs reality

**← from the GPU table:** "all arithmetic. here's the one place I checked it against a
real run."

```
Predicted (slide 8):  1B × 12 bytes  ≈  12 GB
Measured  (A2 run):                    13.84 GB peak
```

**原理 (what the ~1.8 GB gap IS — and why it's the honest closer):** the ×12 formula
counts param + gradient + optimizer. It deliberately omits **activations** — the
intermediate forward-pass values stashed to compute the backward pass. Activations scale
with `batch_size × sequence_length × hidden × layers`, *not* with param count — so
they're the **workload-dependent** term you control with batch size and a trick called
gradient checkpointing, not a fixed per-param cost. Plus a few hundred MB of CUDA/
framework overhead. So: `×12` gives you the **floor**; activations are the variable part
on top.

Say: "Before running anything, the formula said ~12 GB. Real peak was 13.84 — the extra
~1.8 is activations and overhead, which you can also estimate. The point holds: I
predicted whether it would fit *before downloading the model*, and it did. **That's the
whole promise — you don't guess whether a model fits, you compute it**, floor first,
then the workload term on top."

---

# CLOSE

**← from proof:** "so — both walls are down, and you saw *why* each answer is true, not
just that it is."

- **Blocker 1 (do I need it?):** fine-tuning *reweights behavior*, it doesn't *install
  facts* (signal vs entrenchment). Style/format → fine-tune. New facts → RAG. Quick fix
  → prompt first.
- **Blocker 2 (what hardware?):** `param_count × bytes_per_param`. Full 12, LoRA 2,
  QLoRA 0.6, because the update is low-rank so you train ~1% and freeze the rest. Match
  to a card. A 16 GB gaming GPU or a $1/hr A100 gets you there.
- **Do it:** open tools — HuggingFace `transformers` + `Trainer`, a public dataset, a
  Docker container, ~100 lines. My run: Llama-3.2-1B, 500 examples, 1 epoch, **83
  seconds, 13.84 GB.**

Last line: "The hard, expensive part — pretraining — is done for you. What's left is
arithmetic you can predict and a mechanism you can reason about, on hardware you own or
rent for coffee money. Go try it."

---

## Delivery notes / what to prep before Friday

**Time budget (≈ 30 min, no Q&A):**

| Segment | Slides | Target |
|---|---|---|
| Hook | — | 1.5 min |
| Map (orientation — two axes) | 1 | ~1.5 min |
| Act 1 — do I need it? | 2–5 | ~11 min |
| Bridge | — | 0.5 min |
| **Act 2 — VRAM / card (centerpiece)** | **6–11** | **~14 min** |
| Close | — | ~2 min |
| Buffer / overflow into Q&A | D1, D2 | — |

If running long, cut order: D1 (fp32 detail) → trim Act 1 demo to 2 pairs → shorten
slide 9's QLoRA paragraph. Do NOT cut the `原理` beats on slides 3, 5, 8, 9 — those ARE
the "principle over conclusion" the audience came for, and the answer to both blockers.

**Mechanisms YOU must be solid on before presenting (this audience WILL probe — they
use these models more than you do, just from outside):**
- Slide 3/5: signal vs entrenchment; why "fine-tuning can't add facts" is too strong
  (it's the wrong *tool*, not impossible) — see the four-reason breakdown.
- Slide 5: the specialist "aha" — why a fine-tuned 7B can rival a frontier model *on one
  narrow vertical* (capacity concentration), and why that's an *insight about the market*,
  NOT a recommendation for this room (we keep using Opus; tokens are company-paid). Be
  ready to defend the scope limit: narrow well-defined tasks only, not general reasoning.
  Backing evidence verified and ready: LoRA Land (arXiv:2405.00732) for the 7B-rivals-GPT-4
  claim; Gekhman et al. (arXiv:2405.05904) for the fine-tuning-on-new-facts → hallucination
  claim. See the References section below — know these two before presenting.
- Slide 8: what m and v actually are (momentum / per-param adaptive step), why 8 of 12
  bytes are optimizer state.
- Slide 9: LoRA = the weight *update* is low-rank → train two skinny matrices B·A.
- Slide 11: the gap is activations (workload-dependent), not a formula error.
  (If any of these is shaky, walk through it before Friday — don't present a mechanism
  you can't defend to a room that will push back.)

**Placeholders to fill (you):**
- Slide 4: one real `alpaca-cleaned` row (instruction / input / output).
- (Optional) Slide 8: A2 loss step 1 vs final, if you want to show loss sliding down.

**Numbers to cross-check before Friday:**
- Slide 9 per-method bytes (~0.6 QLoRA, ~2 LoRA) against
  `experiments/a01-mem-budget/notes.md`.
- Slide 8: confirm whether the A2 run used an fp32 master copy (12 vs 16 bytes) — see
  the D2 honesty note; confirm the 13.84 GB / 83 s figures and base model path.

**Demo text:** pull pairs #6, #1, #5 (and #10 for the honesty beat) verbatim from
`experiments/a03-eval-1b/results.md`.

**Depth guardrail:** the `原理` beats go to *mechanism* (low-rank updates, momentum,
entrenchment) — that's the right depth for this audience. Do NOT cross into the *math*
of those mechanisms: no backprop derivation, no chain rule, no cross-entropy formula, no
SVD proof of low-rank. Mechanism = "what it does and why"; math = "derive it." Stay on
the first side.

---

## References (verified this session — cite if pushed)

The two claims most likely to draw "prove it / which paper?" are both backed and checked:

- **Slide 5 — "fine-tuning to add facts encourages hallucination."**
  Gekhman, Yona, Aharoni, Eyal, Feder, Reichart, Herzig — *"Does Fine-Tuning LLMs on New
  Knowledge Encourage Hallucinations?"*, EMNLP 2024. https://arxiv.org/abs/2405.05904
  Verified findings to quote: examples introducing *new* facts are learned significantly
  *slower* than examples consistent with the model's existing knowledge; as those
  new-knowledge examples are eventually learned, they *linearly increase* the model's
  tendency to hallucinate. Their framing: factual knowledge is mostly acquired in
  pretraining; fine-tuning teaches the model to *use* it more efficiently, not to absorb
  new facts. (Directly supports both the "entrenchment" mechanism and reason #4.)

- **Slide 5 — "a fine-tuned 7B can rival a frontier model on a narrow vertical."**
  Zhao, Wang, Abid, Angus, Garg, Kinnison, Sherstinsky, Molino, Addair, Rishi (Predibase)
  — *"LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report"*, 2024.
  https://arxiv.org/abs/2405.00732
  Verified findings to quote: 310 models = 10 base models × 31 tasks, trained with 4-bit
  LoRA (same QLoRA recipe as Act 2). The fine-tuned models *outperform their base by ~34
  points and GPT-4 by ~10 points on average* across these tasks. Scope honesty: this is
  *per-task specialists* on narrow well-defined tasks, NOT one model beating GPT-4 at
  everything. The demo served 25 fine-tuned Mistral-7B adapters on a single 80 GB A100.

Background concepts referenced (well-known, cite only if asked for the original):
- **LoRA** (slide 9): Hu et al., *"LoRA: Low-Rank Adaptation of Large Language Models"*,
  ICLR 2022. https://arxiv.org/abs/2106.09685 — the "weight update is low-rank" claim.
- **QLoRA / NF4 4-bit** (slide 9): Dettmers et al., *"QLoRA: Efficient Finetuning of
  Quantized LLMs"*, NeurIPS 2023. https://arxiv.org/abs/2305.14314 — the ~0.6 B/param path.

> Sourcing note: these four were confirmed via direct arXiv fetch this session. The
> WebSearch tool is broken in this environment (see CLAUDE.md); links above were fetched
> by URL. If you add more claims, verify them the same way before Friday — do not present
> an unsourced "studies show" to a room that will ask which study.
