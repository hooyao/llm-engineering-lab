# Agent Engineering Path (Track D)

> **Where this sits in the project.** This repo is no longer a pure fine-tuning
> project — it is the workspace for a career transition into model-facing AI
> work. It has **three legs**:
>
> 1. **Model side** — `notes/curriculum-v2-execution.md` (Tracks A/B/C:
>    fine-tuning, pretrain+RLHF, math).
> 2. **Agentic side** — this folder (`agent/`, Track D: building a Manus-style
>    autonomous core whose coding specialization is benchmarked against Claude
>    Code and Codex).
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

A working, hand-written autonomous-agent runtime in **Astra** plus a separate
set of applied-agent interview labs. These are deliberately different outputs:

1. **Astra is an independent product.** Its north star is a Manus-style general
   autonomous agent core. Coding is its first specialization and the benchmark
   used to compare it with Claude Code and Codex. Production code enters Astra
   only when a real task, regression, or benchmark justifies the runtime
   contract.
2. **The learning repo owns interview breadth.** Intent routing, ReAct
   comparisons, generic workflow orchestration, production RAG, and timed
   interview exercises may be worth learning without becoming Astra features.
   Those live in `curriculum-agent-interview.md` and use Astra only through its
   public surface.

The portfolio claim is therefore precise: **"I built and measured an autonomous
agent runtime, then used it and off-the-shelf application infrastructure to
solve and evaluate realistic agent problems."** It is not "I put every adjacent
AI technique into one framework."

## Product north star and reference boundaries

**General-agent north star — Manus.** Manus's official context-engineering
write-up describes the product boundary Astra is aiming at: an iterative
action/environment/observation loop, a stable action space, sandbox execution,
file-backed recoverable context, long-horizon task focus, and error recovery.
See [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus).

**Coding specialization — Claude Code and Codex.** Claude Code's restored source
is the implementation reference for the coding-agent loop, tools, context,
permissions, compaction, and coordination. Claude Code and Codex are the
comparison targets for coding-task correctness, recovery, latency, token cost,
safety, and extensibility. Feature count is not a success metric.

**Applied/interview breadth — papers, engineering reports, and application
frameworks.** `research/2026-agent-patterns.md` informs interview labs and can
suggest product hypotheses, but it is not an Astra backlog. A pattern enters
Astra only after passing Astra's feature-admission gate.

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
| **T1** | First-party engineering reports: Manus, Anthropic Engineering, Cognition, Chip Huyen, Hamel Husain | Load-bearing within the system and date studied |
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
                                runtime. Product-track deliverables land here
                                only after passing its feature-admission gate.
                                Goal: a Manus-style core whose coding
                                specialization surpasses Claude Code/Codex on
                                measured outcomes.
```

Product-track flow: **name a real failure → read the relevant production source
and engineering evidence → define a measurable invariant → implement the
smallest Astra contract → run the payoff.** Claude Code source is authoritative
for what Claude Code does; Manus's published engineering explains the general
agent boundary. Neither source is a feature checklist.

Interview-track flow: **study the pattern → implement or integrate it in a
standalone lab → evaluate it → practice explaining the tradeoff.** Interview
labs do not modify Astra unless an independently demonstrated product failure
passes Astra's admission gate.

Astra is a **read-write submodule** — develop in its working tree, commit there,
then bump the pinned commit from this repo (same pattern as
`dgx-spark-playbooks`). The two Claude Code repos are read-only references.

## Non-negotiable scope constraint

The two curricula have different ownership rules.

**Astra product track:** build only reusable autonomous-runtime behavior:
agent/action/observation execution, tool and action-space control, context and
recoverable task state, environment/sandbox boundaries, permission and trust,
failure recovery, worker isolation, measurement, and coding-agent
specialization. Every subsystem needs a concrete failing task or benchmark.

**Applied interview track:** learn and demonstrate intent routing, ReAct and
other orchestration patterns, generic workflows, production RAG, online eval,
multi-tenant design, and framework translation. Use off-the-shelf vector stores,
embedding models, workflow engines, and tracing backends. These labs may consume
Astra but cannot force abstractions into it.

**Always outside Astra Core:** model training/fine-tuning, a generic workflow
engine, vector DB/embedding/document-ingestion infrastructure,
application-specific intent taxonomies, and SaaS control-plane concerns. Astra
must compose with them, not own them.

## Files in this folder

| File | What |
|---|---|
| `README.md` | This file. |
| `why-agent.md` | Motivation — why agent engineering is the 2026 leverage point and how it ladders to the job search. (Companion to `notes/why.md`.) |
| `curriculum-agent.md` | Astra product-engineering track: the Manus-style autonomous core and coding specialization. |
| `curriculum-agent-interview.md` | Applied-agent interview track; standalone labs and mocks that do not define Astra's backlog. |
| `research/2026-agent-patterns.md` | Cited research for product hypotheses and interview labs, not an automatic Astra roadmap. |
| `refs/` | The three submodules. |
| `experiments/` | Product-track notes and payoffs. Astra code changes still require independent product justification. |
| `interview/` | Applied-agent labs, system-design exercises, and timed mocks. |

## Where to start

For Astra work, read its `CLAUDE.md` feature-admission gate, then continue the
next unfinished day in `curriculum-agent.md`. For interview preparation, use
`curriculum-agent-interview.md`; do not open an Astra change merely because an
interview topic appears there.
