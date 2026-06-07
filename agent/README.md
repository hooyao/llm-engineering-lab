# Agent Engineering Path (Track D)

> **Where this sits in the project.** This repo is no longer a pure fine-tuning
> project — it is the workspace for a career transition into model-facing AI
> work. It has **three legs**:
>
> 1. **Model side** — `notes/curriculum-v2-execution.md` (Tracks A/B/C:
>    fine-tuning, pretrain+RLHF, math).
> 2. **Agentic side** — this folder (`agent/`, Track D: building a Claude
>    Code-style agent core by hand).
> 3. **Career transition** — `notes/career-transition-research.md` (where the
>    skills land: target roles, locations, leveling, visa, comp).
>
> Track D is not a side quest. Per the career research (§2), **"AI/LLM Agent
> Engineer" is the single most reachable model-facing role** for this profile —
> it is product engineering + orchestration (tool-calling, RAG, MCP/A2A,
> production reliability), needs no ML PhD, and directly reuses the AKS /
> distributed-systems / reliability background. This path is the highest-ROI
> bridge from "senior cloud engineer" to "model-facing engineer."

---

## What this path builds

A working, hand-written agent harness in **Astra** (your C# framework) that
reaches parity with — and is meant to eventually exceed — the Claude Code agent
core, plus the half Claude Code deliberately doesn't have (RAG, agent eval,
long-term memory, interop). At the end you can read any production agent
codebase and re-implement any piece of it from first principles.

The deliverable is not "I used LangGraph." It's **"I wrote the agent loop, the
tool orchestrator, the compaction tiers, the permission pipeline, an agentic-RAG
retrieval loop, and an eval harness myself, and here's the eval score and the
token-cost budget."** That is the portfolio artifact the career research calls
for (Phase 0, item 3).

## The two halves (read this before starting)

This path rests on two facts about its source material.

**Half 1 — the agent core (the *how*).** Claude Code's restored source
(`refs/claude-code-sourcemap/restored-src/src/`) is the source of truth for the
while-loop, tool dispatch, context assembly, compaction, permissions, and
multi-agent coordination. Reading it teaches you *how* a production agent core is
built. This half is well-covered by the source — study it, then re-implement in
Astra.

**Half 2 — the frontier (the *why-not*, and everything Claude Code lacks).**
Claude Code is a single-agent **coding** harness — sample size 1, optimized for
one vertical. It deliberately has **no RAG** (it uses `grep`+`read` instead),
deliberately **downplays multi-agent** (the Cognition "don't build multi-agents"
camp), and has no RAGAS-style eval or long-term semantic memory. The things you
named — RAG, agent eval, memory, interop — are **absent from the source**. They
exist only in primary engineering blogs and papers. This half is covered by
`research/2026-agent-patterns.md`.

> Studying only the source code would teach you one team's choices for one
> vertical and present them as the whole truth of agentic engineering. Studying
> only blogs would leave you unable to actually build the core. You need both.

## Source-tiering rule (how to judge a reference)

Most "AI agent" content online is shallow — LangChain/CrewAI quickstart rewrites
by authors who never ran an agent in production. They're easy to spot: toy
examples (check the weather, book a flight), no discussion of token cost, no
context degradation, no failure/retry/idempotency, no eval. **Skip those.**

The filter is **not the domain name** — company engineering blogs live on
`medium.com`. The test is the content:

> Does it discuss token cost, latency budget, context rot, evaluation, or
> failure modes? → **read it.** Is it a happy-path toy demo? → **skip it.**

Tiering used throughout this path:

| Tier | Source | Trust |
|---|---|---|
| **T1** | First-party engineering blogs: Anthropic Engineering, Cognition, Chip Huyen, Hamel Husain | Load-bearing |
| **T2** | Papers + specs: ReAct, Reflexion, Self-RAG, CRAG, RAGAS, MCP spec, A2A spec, OTel GenAI semconv | Load-bearing for *what a technique is* |
| **T3** | Practitioner write-ups that demonstrably ran it in prod | Directional |
| **Source** | `refs/claude-code-sourcemap` (the *how*) | Ground truth for the core |

## The three submodules and how they work together

```
agent/refs/
  claude-reviews-claude/     ← TEACHING layer: 17-chapter architecture analysis
                                of Claude Code. Read the relevant chapter first
                                to build the mental model. (EP01..EP17)
  claude-code-sourcemap/     ← SOURCE layer (source of truth): the restored
                                TypeScript source of Claude Code v2.1.88. The
                                "how does it ACTUALLY do X" answer.
                                (restored-src/src/QueryEngine.ts, Tool.ts,
                                 query.ts, context.ts, coordinator/, memdir/, ...)
  Astra/                     ← IMPLEMENTATION layer (yours): C#/.NET 10 agent
                                framework. Each day's deliverable lands here.
                                Re-implement, don't copy. Goal: exceed LangGraph
                                by learning from Claude Code's design.
```

Daily flow: **read the chapter (teaching) → check the source (how) → write it in
Astra (implementation) → for the frontier half, the source is silent, so read
`research/2026-agent-patterns.md` instead of the source.**

Astra is a **read-write submodule** — develop in its working tree, commit there,
then bump the pinned commit from this repo (same pattern as
`dgx-spark-playbooks`). The two Claude Code repos are read-only references.

## Non-negotiable scope constraint

This path studies **agentic logic**, not commodity infrastructure. Per your
explicit direction:

- **Build by hand:** the agent loop, tool orchestration, context/compaction,
  permissions, multi-agent coordination, the *agentic part* of RAG (when to
  retrieve, query rewriting, CRAG-style grading/correction, retrieval-as-tool),
  the eval harness, the judge.
- **Use off-the-shelf (do NOT reinvent):** vector DB, embedding models, the
  tracing backend (Jaeger/OTel collector). RAG's agentic decisions are in
  scope; the vector store underneath is not.

The line: *RAG is agentic; a vector DB is not.*

## Files in this folder

| File | What |
|---|---|
| `README.md` | This file. |
| `why-agent.md` | Motivation — why agent engineering is the 2026 leverage point and how it ladders to the job search. (Companion to `notes/why.md`.) |
| `curriculum-agent.md` | Track D day-by-day execution plan (D1–D16, two phases). |
| `research/2026-agent-patterns.md` | Cited, verified research — the source of truth for the frontier half. |
| `refs/` | The three submodules. |
| `experiments/` | Per-day deliverables (`d01-agent-loop/` … `d16-capstone/`). Code lands in Astra; notes/analysis land here. |

## Where to start

Read `why-agent.md` (motivation) → `curriculum-agent.md` Day D1 → skim
`research/2026-agent-patterns.md` Part A (you'll reference it all the way
through). Then open `refs/claude-reviews-claude/architecture/01-query-engine.md`
and `refs/claude-code-sourcemap/restored-src/src/query.ts` side by side, and
start writing the loop in Astra.
