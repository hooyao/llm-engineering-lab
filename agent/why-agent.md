# Why — what the agent engineering path is actually buying you

> Read this when motivation flags. Companion to `notes/why.md` (which covers the
> model side). If a week's work on Track D doesn't ladder up to one of the
> outcomes below, something is wrong with the plan, not with you.

---

## This path's place in the bigger goal

This repo is a **career-transition workspace**, not a fine-tuning hobby. Three
legs, one goal — move from "Microsoft L64 senior cloud engineer" to
"model-facing AI engineer":

1. **Model side** (`notes/curriculum-v2-execution.md`) — you can train, adapt,
   and serve models.
2. **Agentic side** (this folder) — you can build the systems that *use* models.
3. **Career transition** (`notes/career-transition-research.md`) — where it
   lands: roles, locations, leveling, visa, comp.

The career research (`notes/career-transition-research.md`) is blunt about a
tradeoff: choosing model-facing work means **giving up your strongest
differentiator** (ML-infra / distributed-systems-under-production-constraints is
chronically undersupplied and is exactly your moat). Track D is the **one
model-facing direction that does NOT make you give up that moat** — because
agent engineering *is* distributed-systems-and-reliability work pointed at LLMs.

## Why agent engineering is the highest-ROI leg for *you*

From the career research, §2, the four model-facing roles ranked by reachability
for your profile:

1. **AI/LLM Agent Engineer — most reachable.** Verbatim from the research:
   *"essentially product eng + orchestration (tool-calling, RAG, MCP/A2A,
   production reliability), needs no ML training, directly eats your AKS /
   distributed / reliability background."*
2. Evals / model-behavior — reachable (statistics + systems thinking, no PhD).
3. Applied fine-tuning — needs a portfolio, but no PhD.
4. Distillation / frontier-lab RS — least reachable (PhD-gated).

Track D trains **#1 and #2 directly**, and produces the portfolio artifact that
unlocks #3. It is the single best use of your existing 10+ years of systems
experience inside the model-facing world.

### The specific job reqs this path targets

The career research names concrete internal-transfer destinations where your
"internal transfer + model-facing" wishes both come true. The agent path maps to
them directly:

- **Microsoft Applied AI Engineer II** — the research notes its JD says
  *"cloud/AKS background directly relevant"* and it's about agentic / RAG /
  evals. **This is Track D's bullseye.** (Bachelor's + 2y, no PhD.)
- **Microsoft Copilot Tuning** (Dublin/Copenhagen) — applied model-facing.
- **Dublin "Agent Cloud" Senior AI SWE** — explicitly agent work.
- External fallback: OpenAI Dublin FDE, Google Cloud GenAI FDE.

Every one of these is an agent-engineering job, not a training-research job. The
portfolio they want to see is exactly what Track D produces.

---

## The 5 things this path makes you able to do

Each maps to a portfolio claim a hiring manager can verify.

### 1. Build an agent core from first principles

Not "I called an agent framework" but "I wrote the `while(true)` query loop, the
streaming tool-call parser, the concurrent-read/serial-write orchestrator, and
the four-tier compaction system — here's the code in Astra and here's how it
maps to Claude Code's architecture." This is the difference between an engineer
who *uses* agents and one who can *build and debug* them.

### 2. Reason about context as a budget, not a black box

You will internalize **context engineering** (the 2026 core discipline) and
**context rot** (recall degrades with token count before the hard limit). You'll
be able to say *why* an agent got worse at turn 40, quantify the token cost of a
design, and choose between compaction / clearing / external memory with
arithmetic — the same way the model-side curriculum makes you size VRAM with
arithmetic instead of trial and error.

### 3. Build agentic RAG and *evaluate* it

The career research lists "agent + evals" as Phase-0 portfolio item 3. You'll
implement a retrieve-then-reason loop, a CRAG-style retrieval grader with
corrective routing, and then **measure it** with RAGAS metrics (faithfulness,
context precision/recall). Crucially you'll know *when not to use RAG* — the same
judgment Claude Code encodes by using `grep` instead. Knowing when a vector index
is the wrong tool is itself a senior signal.

### 4. Evaluate agents like a professional, not by eyeballing

End-state outcome evaluation, LLM-as-judge with calibration against humans
(Cohen's kappa), capability-vs-regression eval suites, the OpenTelemetry GenAI
tracing conventions. "How do you know your agent is good?" is the question that
separates people who shipped agents from people who demoed them. You'll have a
real answer with a number attached.

### 5. Speak the 2026 interop stack fluently

MCP (agent↔tool) and A2A (agent↔agent) — what they are, how they compose, where
they're weak (A2A AgentCards are self-reported and unsigned). Plus the
"too many tools" problem and the code-execution-with-MCP fix (150k→2k tokens).
This is the vocabulary in every current agent JD.

---

## Career capital (the underneath)

- You join the small set of engineers who can **build** an agent harness, not
  just wire one. In Microsoft/FAANG performance-band terms, this is the
  difference between "uses AI tools" and "builds AI systems."
- Astra becomes a **public, inspectable portfolio** — a from-scratch agent
  framework in C# that an interviewer can read. The career research notes
  Anthropic and most product JDs explicitly accept "equivalent experience" in
  place of a degree; a working agent framework *is* that equivalent experience.
- It compounds with the model side: an engineer who can both fine-tune a model
  *and* build the agent that serves it is rare and is exactly what
  "Applied AI Engineer" reqs ask for.

## What this path does NOT give you

To set expectations honestly (mirroring `notes/why.md`):

- It won't make you a frontier-lab research scientist. That's PhD-gated and is
  not this path's goal (and per the career research, that tier is closed to this
  profile anyway).
- A from-scratch C# framework will not out-feature LangGraph/CrewAI on day one.
  The point isn't feature parity — it's that you *understand every line*, which
  is what makes you employable, and what lets Astra eventually be better *by
  design* rather than by accretion.
- It won't replace the model side. The strongest portfolio (per career research
  Phase 0) is fine-tuning **and** FSDP/DeepSpeed **and** agent/evals — the three
  together keep your infra moat while adding the model-facing entry ticket.

---

## How to use this file

- Re-read when a Track D evening feels like busywork. The throughline is: every
  hand-written subsystem is a sentence you can say in an interview and back with
  code.
- After the Track D checkpoint (D16 capstone), come back and re-read — then go
  update `notes/career-transition-research.md` Phase 0 with the portfolio item
  you just finished.
- The honest framing from the career research applies here too: this is a
  **12–18 month** part-time path to genuine mid-level competitiveness, not a
  3–6 month bootcamp promise. Track D is one of the three things that fills that
  window with verifiable artifacts.

---

## Personal notes (append over time)

<!-- e.g.
### Track D week 1 reflection (TBD)
- Wrote the agent loop. The "dumb loop, smart model" idea finally clicked when
  I saw query.ts is genuinely just a while-loop — all the intelligence is in
  the model + tool results, not the scaffold.

### After D16 capstone (TBD)
-->

(empty — fill in as you go)
