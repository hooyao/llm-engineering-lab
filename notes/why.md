# Why — what this 8-week investment is actually buying you

> Read this when motivation flags. The point is not the skills, it's the outcomes
> the skills unlock. If a week's work doesn't ladder up to one of the outcomes
> below, something is wrong with the plan, not with you.

---

## The honest 2026 LLM landscape

By June 2026 the floor is high:

- **Frontier closed models** (Claude Opus 4.8, GPT-5, Gemini 3 Ultra) handle most
  knowledge tasks zero-shot at acceptable quality.
- **Open weights have caught up**: DeepSeek V4, Qwen3, Llama 4 are good enough for
  90% of business use cases at <$0.50/M tokens self-hosted.
- **Prompting + RAG + tool use** (the "Claude Code grep" pattern) covers the bulk
  of what people previously thought required fine-tuning.

So why are you still doing this?

Because **the value of fine-tuning has not gone to zero — it has consolidated into
5 specific niches**, and those niches are *exactly* where the durable
ML-engineering jobs and the highest-leverage personal projects live in 2026.

---

## The 5 use cases that survived

Every day's work in this curriculum trains for one or more of these. The
mapping is explicit in each track.

### 1. Style / format / behavior conformity (SFT primary value)

Make a model **always** output in a particular tone, schema, or behavior pattern.
Prompts can ask for it; only fine-tuning makes it the default distribution.

- Customer service tone, regulated industry disclaimers, JSON schema adherence,
  specific reasoning format (e.g. always emit a `<plan>` block first)
- Why prompts fail: 5-30% violation rate is too high for production
- Why RAG fails: RAG injects facts, doesn't change priors

**Track A** (especially A2-A3, A6, A9) builds the muscle to do this.

### 2. Domain vocabulary + implicit knowledge

Internalize an industry's specific terminology and unwritten conventions that
RAG can't fully provide because they're *implicit* in the way experts write.

- Legal contract phrasing, medical ICD coding, semiconductor process jargon,
  financial derivative naming conventions
- Why RAG fails: domain ambiguity (`CDR` = contract default rate, not carbon
  disclosure report) requires the model to have *seen patterns*, not just be
  handed a glossary
- Bar to entry: you need >1B tokens of in-domain corpus to justify it

**Track A8** (scaling fine-tunes to 14B+ models) and **Track B** (understand
what "internalize a vocabulary" means at the gradient level) cover this.

### 3. Distillation: compress big-model behavior into small-model latency

The single largest fine-tuning market in 2026. Take a frontier model's
outputs, train an open-source 3B-8B model on them, deploy that.

- Cuts API costs 50-200×; cuts latency 5-20×
- Enables on-device / edge deployment (cars, robots, regulated environments)
- Lets you run 24/7 high-QPS workloads without burning OpenAI / Anthropic
  budget

**Track A1 + A6 + A7 + A10** is exactly the distillation pipeline (memory
budget → LoRA → QLoRA → serve). You're not training a 405B model from
scratch; you're learning to take what's already trained and shape it for
your deployment envelope.

### 4. Alignment / safety with weight-level guarantees

Bake "never give medical advice" or "always cite a source" into the weights,
not the prompt. System prompts can be jailbroken, injected, or forgotten;
weight-baked behavior cannot.

- Healthcare, legal, financial — anywhere wrong output is a liability event
- Required for any deployment going through enterprise security review

**Track A9** (DPO) + **Track B9-B11** (RM + PPO + DPO) cover this end-to-end.
You will literally hand-label preferences and watch a model's behavior shift —
the most concrete way to understand what RLHF actually does.

### 5. New tools / new modalities / new reasoning modes

Teach a model to use *your* API schema, *your* internal DSL, *your* function
calling conventions. This is procedural knowledge (knowing how to do
something), distinct from factual knowledge (knowing that something).
RAG handles factual; procedural needs gradient updates.

- Custom agent loops, in-house tool use, code generation against private
  codebase patterns
- Growing market as 2026 sees the agent-as-a-product wave

**Track A6** (LoRA mechanics) + a future extension (function-calling SFT
on your own data) covers this.

---

## When RAG / prompting wins (be honest about it)

You should leave this curriculum knowing **when not to fine-tune**, not just
how to do it. RAG dominates fine-tuning when:

- Facts update frequently (stock prices, inventory, current docs)
- Citation / traceability is required (compliance, legal)
- Per-user or per-tenant data isolation matters
- Document volume is large but pattern repetition is low
- Iteration speed matters more than runtime efficiency

**Knowing this is itself a skill** — it stops you from over-engineering. The
fact that Claude Code does most of its work via `grep` + `read` rather than a
fine-tuned coding model is the *correct* engineering decision for that
context. You will be able to make the same call for your own systems.

---

## What this curriculum specifically gives you

**Hard skills (the surface):**
- SFT, LoRA, QLoRA, DPO end-to-end on a real GPU
- Memory arithmetic that lets you size any model for any hardware
- Inference serving (vLLM) with throughput / latency tradeoffs
- From-scratch transformer / attention / tokenizer implementation
- Classic + modern RLHF pipelines (PPO and DPO)
- Math fluency to read any 2026 ML paper without skipping equations

**Career capital (the underneath):**
- You join the small fraction of engineers who can train LLMs, not just call
  their APIs. In Microsoft / FAANG performance bands, this is the
  differentiator between "ML-aware engineer" and "ML engineer."
- You understand the entire stack from byte-pair encoding to RLHF — most
  practitioners have a 2-3 layer gap they paper over with libraries.
- You can debug *why* a model is bad, not just try different prompts.
- You can build personal projects (custom agents, distilled local models,
  evaluation tooling) without paying anyone for inference.

**Optionality (the strategic):**
- AI infrastructure / model ops roles are paying $400K-$700K TC at FAANG in
  2026 and the supply is thin. You will be eligible after this curriculum.
- If you ever leave Microsoft to do a startup, "I can run my own LLM stack"
  removes the largest infrastructure dependency.
- You will be able to evaluate AI products and partnerships at work with
  technical depth most peers lack — a leverage multiplier on whatever
  systems work you're already doing.

---

## What this curriculum does NOT give you

To set expectations:

- You will not become a research scientist. That requires 2+ years of
  paper-writing and isn't this curriculum's goal.
- You will not be able to train competitive frontier models. Single GX10 cannot
  produce a GPT-4-class model regardless of how good you get; that's a
  $10M-$100M compute problem.
- You will not learn distributed training (FSDP / DeepSpeed cross-node) —
  single-box GX10 doesn't need it. If you change jobs and need it, it's
  ~2 weeks of additional learning on top of this foundation.
- You will not become a prompt engineer (good — that "skill" is mostly
  evaporating as models get better).

---

## Track-by-track value map

| Track | Day | What you do | Why it matters (2026) |
|---|---|---|---|
| **A0** | PyTorch basics | One-time setup | Enables everything below |
| **A1** | Memory arithmetic | Calculate fit before trying | Distinguishes you from "throw it at GPU and see" engineers; the #1 skill recruiters test for in ML-infra interviews |
| **A2-A5** | Full SFT mechanics | Train 1B-8B end to end | Use case #1 (style/format) muscle memory |
| **A6** | LoRA properly | Sweep rank/alpha/modules | Use case #3 (distillation) core, use case #5 (custom adapters) |
| **A7-A8** | QLoRA + scale | Get 32B trainable on one box | Use case #2 + #3 at production scale |
| **A9** | DPO | Preference fine-tune | Use case #4 (alignment) |
| **A10** | vLLM serve | Inference economics | Without this, training is academic |
| **B1-B4** | Karpathy zero-to-hero | Write everything from scratch | Removes magic — you understand every line of every framework you ever touch after this |
| **B5-B7** | TinyStories pretrain | End-to-end pretrain on GX10 | Concrete answer to "what does pretraining actually do" — most engineers will never have this |
| **B8-B11** | SFT + RM + PPO + DPO | RLHF full pipeline | Use case #4 deeply; also resume-level credibility |
| **B12** | Read OLMo | Production pretrain reference | You map your toy code to real-world scale |
| **C1-C6** | Calculus + backprop | Hand-derive everything | You stop being scared of `loss.backward()` |
| **C7-C9** | AdamW + KL + DPO loss | Derive on paper | You can debug optimizer / loss issues, not just file a github issue |
| **C10** | Paper readthrough | One paper, every equation | Validates you're now equation-literate |

---

## How to use this file

- Re-read this whenever a week feels like "why am I doing this"
- After each track checkpoint (A10, B12, C10), come back and re-read — your
  understanding of what each section meant will deepen
- Add a personal note at the bottom of this file at week 4 + week 8:
  what concretely changed about how you think and what you can do
- After 8 weeks, this file becomes evidence: you can show it to a hiring
  manager, a coworker, or future-you to demonstrate the trajectory you took

---

## Personal notes (append over time)

<!-- e.g.
### Week 1 reflection (2026-06-12)
- Realized memory arithmetic is the bottleneck of EVERY discussion. Stopped
  reading PRs about model training without first computing what fits.
- Distillation idea I want to test: take Sonnet 4.6 outputs on internal docs,
  fine-tune Qwen3-8B for our team's QA bot.

### Week 4 reflection (TBD)
### Week 8 reflection (TBD)
-->

(empty — fill in as you go)
