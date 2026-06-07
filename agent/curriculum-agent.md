# Track D — Agent Engineering (16 evenings, two phases)

> **Why are you doing this?** See `agent/why-agent.md`. Re-read when motivation
> flags. **Where it lands:** `notes/career-transition-research.md` §2 ranks
> "AI/LLM Agent Engineer" as the most reachable model-facing role for your
> profile. Every day below is a sentence you can say in an interview, backed by
> code in Astra.

This is the **execution plan** for the agentic leg of the project. It mirrors
the structure of `notes/curriculum-v2-execution.md` (Tracks A/B/C): one concept
per day, each day produces a runnable artifact, single recommended resource per
day, an end-of-track checkpoint.

## What this track assumes

- 10+ years software architecture (C#/.NET deep): `git`, async, streaming,
  concurrency, DI — zero introduction. Astra is already C#/.NET 10 with an
  agent-loop skeleton.
- Agentic ML: **none assumed**. Every agent term is defined the first time it
  appears (see each day's **Term** block).
- You've read `agent/README.md` (the two-halves framing + source-tiering rule).

## The daily flow (three layers, one deliverable)

```
1. TEACHING   read the chapter:   refs/claude-reviews-claude/architecture/NN-*.md
2. SOURCE     check the how:       refs/claude-code-sourcemap/restored-src/src/*
3. IMPLEMENT  write it in C#:      Astra/src/Astra.Core/...  (the deliverable)
```

For **Phase D-II** (the frontier half), the source layer is silent — Claude Code
has no RAG / RAGAS / semantic memory. There, swap step 2 for
`agent/research/2026-agent-patterns.md`.

## Conventions

- **Each day produces a runnable artifact** in Astra (a class + a passing test,
  or a working CLI behavior). Notes/analysis land in
  `agent/experiments/dNN-<slug>/`.
- **Re-implement, don't copy.** Astra's own CLAUDE.md says it: *"Learn from
  Claude Code, don't copy it."* The source is a reference for design decisions,
  not a paste buffer.
- **Quantify.** Every agent design has a token cost. State it (tokens/turn,
  tokens/tool-result, context-window headroom) the same way the model side
  quantifies VRAM. "Uses a lot of context" is not an answer.
- **Commodity infra is off-the-shelf.** Vector DB, embeddings, tracing backend:
  use existing ones. The agentic logic on top is what you write.

## Track layout

- **Phase D-I — Agent Core (D1–D8):** the *how*. Re-implement the Claude Code
  agent core in Astra. Source-backed.
- **Phase D-II — The Frontier Half (D9–D16):** the half Claude Code lacks. RAG,
  eval, memory, interop. Blog/paper-backed.

---

# Phase D-I — Agent Core (8 evenings)

**Goal at the end:** Astra has a working agent core you wrote line by line —
loop, tools, orchestration, streaming, permissions, context, compaction,
multi-agent. You can point at any part of Claude Code's architecture and say "I
built that."

## D1 — The dumb loop: `while(true)` + the augmented LLM

**Why:** The whole field rests on one anticlimactic idea: *intelligence lives in
the model; the harness is just a loop.* Internalize this before adding anything.

**Do:**

- Read `refs/claude-reviews-claude/architecture/01-query-engine.md` (the brain).
- Read the source: `refs/claude-code-sourcemap/restored-src/src/query.ts` and
  `QueryEngine.ts`. Confirm for yourself that `query()` is an async generator
  that loops: send messages → get response → if `tool_use`, execute → push
  result → repeat.
- In Astra: implement `AgentLoop.SubmitAsync()` returning
  `IAsyncEnumerable<AgentEvent>` — a single-turn loop with **no tools yet**, just
  call the model and yield events. Astra's README already sketches this shape.

**Deliverable:** Astra runs a one-turn conversation end to end, streaming
`AgentEvent`s. A test asserts the loop terminates on `end_turn`.

**Term defined:** *agent loop*, *augmented LLM* (model + tools + retrieval +
memory), *`stop_reason`* (`end_turn` vs `tool_use`), *async generator as the
agent's event protocol*.

**Resource:** Anthropic, "Building Effective Agents" (T1) — the augmented-LLM
section + the autonomous-loop section. https://www.anthropic.com/research/building-effective-agents

## D2 — The tool contract: behavioral flags over inheritance

**Why:** A tool is *"a contract between deterministic systems and
non-deterministic agents."* The design choice that matters: behavior is
**input-dependent**, not type-dependent.

**Do:**

- Read `architecture/02-tool-system.md` + source `Tool.ts` and a couple of
  `tools/*/` implementations (e.g. `GrepTool`, `FileEditTool`).
- In Astra: define `ITool<TInput,TOutput>` with `IsReadOnly(input)`,
  `IsConcurrencySafe(input)`, `IsDestructive(input)`, `InputSchema`, and
  `CallAsync`. Implement two tools: one read-only (`echo`/`read`) and one write
  (`write`). Note `IsReadOnly("ls")==true` but `IsReadOnly("rm -rf")==false` —
  the point is it's a function of the *input*.

**Deliverable:** two registered tools, dispatched by the D1 loop; a test shows
the read tool flagged read-only and the write tool not.

**Term:** *tool*, *behavioral flags*, *input schema*, *fail-closed defaults*.

**Resource:** Anthropic, "Writing tools for agents" (T1) — the five principles.
https://www.anthropic.com/engineering/writing-tools-for-agents

## D3 — Tool orchestration: concurrent reads, serial writes, partition-sort

**Why:** Within one turn the model may request several tools. Reads can run in
parallel; writes must not race. And tool ordering must stay cache-stable.

**Do:**

- Read `architecture/02-tool-system.md` (orchestration section).
- In Astra: partition the turn's tool calls — run `IsConcurrencySafe` tools in
  parallel, run the rest serially in a write-exclusive batch. Implement the
  assembly pipeline (built-in tools as a contiguous prefix, MCP tools as suffix)
  for prompt-cache stability.

**Deliverable:** a turn with 3 read calls + 1 write executes reads concurrently
and the write alone; a test asserts ordering and that reads overlapped.

**Term:** *tool orchestration*, *write-exclusive batch*, *partition-sort for
prompt-cache stability*, *concurrency safety*.

**Resource:** the source — how `query.ts` groups tool calls.

## D4 — Streaming + the API layer: stream reassembly and tool_use

**Why:** Real agents stream. You must reassemble a streamed response, detect
`tool_use` blocks mid-stream, and surface partial output as events.

**Do:**

- Read `architecture/15-services-api-layer.md` (stream reassembly).
- In Astra: wire the provider through `Microsoft.Extensions.AI` `IChatClient`
  streaming. Parse the stream into `AgentEvent`s; detect `tool_use`, accumulate
  arguments, fire tool execution when the block completes.

**Deliverable:** streaming tokens render live in the Astra CLI; a `tool_use`
block mid-stream triggers a real tool call.

**Term:** *streaming*, *stream reassembly*, *tool_use block*, *prefill vs decode*
(define; reuse the model-side definition from Track A10).

**Resource:** the source — the API client in `services/`.

## D5 — Permission pipeline: layered, fail-closed

**Why:** An agent that can run `rm -rf` needs defense in depth. Claude Code uses
a 7-layer pipeline; you'll implement the load-bearing 2–3 layers and understand
where the rest fit.

**Do:**

- Read `architecture/07-permission-pipeline.md` + source `hooks/` and the
  permission flow.
- In Astra: implement Layer 1 (schema/input validation) + Layer 2 (permission
  rule matching: policy → user → project → session) + one interactive
  confirmation hook (Layer 5). Fail-closed: unknown → deny.

**Deliverable:** the write tool from D2 prompts for confirmation; a deny rule
short-circuits before execution. Test covers allow / deny / prompt paths.

**Term:** *permission pipeline*, *fail-closed*, *rule precedence*,
*defense-in-depth*, *hook* (PreToolUse).

**Resource:** Astra's CLAUDE.md already specs the 7 layers — implement the first
few, cite the rest.

## D6 — Context assembly: the three-layer cache strategy

**Why:** What goes into the model each turn (system prompt, user context,
per-turn attachments) has different lifetimes and cache strategies. Getting this
wrong wrecks prompt-cache hit rate.

**Do:**

- Read `architecture/10-context-assembly.md` + source `context.ts`.
- In Astra: assemble context in three layers — system prompt (static prefix,
  session-cached), user context (memoized once), attachments (recomputed
  per-turn with a timeout). Keep the static prefix byte-stable for caching.

**Deliverable:** context assembled with a stable prefix; log the prompt-cache
breakpoints; a test asserts the system-prompt prefix is identical across turns.

**Term:** *context assembly*, *static prefix*, *prompt cache*, *attachment*,
*memoization*.

**Resource:** Anthropic, "Effective context engineering" (T1) — read Part A of
`research/2026-agent-patterns.md` alongside.

## D7 — Compaction: the four tiers

**Why:** Long sessions overflow the window. Compaction is the survival mechanism.
You'll build the cheap tiers yourself and a sub-agent summarizer for the
expensive one.

**Do:**

- Read `architecture/11-compact-system.md` + source the compaction flow.
- In Astra: implement (1) **microcompact** (time-decay old tool results, keep a
  recent window) and (3) **full compact** (an LLM-summary sub-agent). Tune the
  summary "maximize recall first, then precision" (from the research). Note in
  your log that compaction loses verbatim detail (research A5: high-level 3/3,
  obscure 0/3) — which sets up D9's memory tool.

**Deliverable:** a session that would overflow instead compacts and continues;
log token count before/after each tier.

**Term:** *compaction*, *microcompact*, *reactive compact*, *recall vs precision
in summarization*, *`prompt_too_long`*.

**Resource:** research Part A (A4, A5) + the cookbook it cites.

## D8 — Multi-agent coordinator: orchestrator-worker + context isolation

**Why:** The most-debated pattern in 2026. You'll build it *and* learn when it
hurts — both camps' views (Anthropic pro-for-research, Cognition anti-for-coding).

**Do:**

- Read `architecture/03-coordinator.md` + `08-agent-swarms.md` + source
  `coordinator/`.
- In Astra: spawn a worker with a **clean, isolated context window** (it cannot
  see the coordinator's conversation), have it return a condensed
  1,000–2,000-token summary via the XML `task-notification` protocol. Keep
  **writes single-threaded** (the cross-camp consensus).
- Read both `research/2026-agent-patterns.md` Part C and the two Cognition posts.
  Write 5 lines on when you'd reach for multi-agent vs a single agent.

**Deliverable:** coordinator dispatches 2 isolated workers in parallel, collects
their summaries, synthesizes. Log the token multiple vs a single agent (expect
the ~15× ballpark).

**Term:** *orchestrator-worker*, *context isolation*, *worker summary*,
*single-threaded writes*, *token multiple*.

**Resource:** Anthropic multi-agent post + Cognition "Don't Build Multi-Agents"
+ "Multi-Agents: What's Actually Working" (all T1, all in research Part C).

### Phase D-I checkpoint

Astra now has a hand-written agent core: loop, tools, orchestration, streaming,
permissions, context, compaction, multi-agent. You can open any chapter of
`claude-reviews-claude` and point to your Astra implementation of it. Write a
1-page recap in `agent/experiments/track-d1-recap.md`. **This alone is a
portfolio piece** — a from-scratch agent core in C#.

---

# Phase D-II — The Frontier Half (8 evenings)

**Goal at the end:** Astra does what Claude Code deliberately doesn't — agentic
RAG, agent evaluation, long-term memory, standardized tracing, MCP/A2A interop.
The source layer is silent here; `research/2026-agent-patterns.md` is your guide.

## D9 — Context rot + just-in-time retrieval + the memory tool

**Why:** This is the hinge between the two halves. Claude Code uses `grep`
instead of RAG *on purpose* — because of context rot and the just-in-time
principle. You'll build the file-based memory primitive that makes it work, and
understand *why* before you build RAG in D10.

**Do:**

- Read research Part A (A2 context rot, A3 just-in-time, A4 the three primitives).
- In Astra: implement a **file-based memory tool** (`memory_*`-style): the agent
  stores lightweight identifiers + learned facts to disk and pulls them back on
  demand. This is the only context primitive that survives a new session — it
  complements D7's compaction (which loses verbatim detail).

**Deliverable:** an agent that writes a fact to memory in session 1 and recalls
it in session 2 (empty window). Show that compaction alone could not do this.

**Term:** *context rot*, *attention budget*, *just-in-time retrieval*, *external
memory tool*, *episodic vs semantic vs procedural memory* (define all three).

**Resource:** research Part A. Note the model-side analogue: this is the same
"read state first" discipline `notes/progress.md` uses for *this* repo.

## D10 — Agentic RAG I: retrieval-as-tool + query rewriting/decomposition

**Why:** RAG, done right in 2026, is *retrieval wrapped in a decision loop* — the
agent decides when to retrieve, rewrites the query, and reasons over results.
The vector DB underneath is commodity; the loop is what you write.

**Do:**

- Read research Part D (D1, D2). Skim the Agentic RAG survey (arXiv:2501.09136).
- In Astra: expose **retrieval as a tool** (back it with any off-the-shelf vector
  store + embeddings — do NOT build those). Implement a **retrieve-then-reason
  loop** with **query rewriting/decomposition**: the agent reformulates the user
  question into one or more retrieval queries before searching.

**Deliverable:** a multi-hop question answered by an agent that issued ≥2
rewritten retrieval queries. Log each query and why it was issued.

**Term:** *agentic RAG*, *retrieval-as-tool*, *query rewriting*, *query
decomposition*, *multi-hop*.

**Resource:** research Part D + Self-RAG (arXiv:2310.11511).

## D11 — Agentic RAG II: CRAG — retrieval grading + corrective routing

**Why:** Naive RAG generates from whatever was retrieved, including garbage.
**Corrective RAG** grades retrieved docs and takes corrective action (refine the
query, or fall back to web search) *before* generating. This is the
single-highest-value RAG upgrade and adds one classification layer, not a
re-architecture.

**Do:**

- Read research Part D (D2 CRAG) + the CRAG paper (arXiv:2401.15884).
- In Astra: add a **retrieval evaluator** that scores each retrieved chunk;
  below-threshold results trigger **corrective routing** — query refinement and
  a web-search fallback — before the generation step.

**Deliverable:** a query where internal retrieval scores low and the agent falls
back to web search, then answers. Log the grade and the route taken.

**Term:** *corrective RAG (CRAG)*, *retrieval evaluator*, *relevance threshold*,
*corrective routing*, *web-search fallback*.

**Resource:** research Part D + CRAG paper.

## D12 — RAG evaluation: RAGAS metrics

**Why:** You can't claim "my RAG is good" without numbers. RAGAS gives the
standard ones. This is the eval discipline a hiring manager will probe.

**Do:**

- Read research Part D (D3 — the metric definitions).
- In Astra (or a thin Python harness — eval tooling is commodity, your *agentic
  logic* stays in C#): wrap your D10/D11 loop and compute **faithfulness**
  (supported claims / total claims), **context precision**, **context recall**,
  **response relevancy** on a small eval set.

**Deliverable:** a RAGAS scorecard for your D10 vs D11 agent (CRAG should improve
faithfulness). 5-line interpretation.

**Term:** *faithfulness*, *context precision*, *context recall*, *response
relevancy*, *ground-truth reference*.

**Resource:** RAGAS docs (research Part D source ledger).

## D13 — Agent evaluation: end-state grading + LLM-as-judge

**Why:** RAG eval is one slice; whole-agent eval is the bigger skill. Grade the
*outcome*, not the trajectory — with the caveats.

**Do:**

- Read research Part B (B1 end-state, B2 escape hatch + calibration, B3 the two
  suites).
- In Astra: build an **LLM-as-judge** — a single call returning a 0.0–1.0 score +
  pass/fail across a rubric, with an **"Unknown" escape hatch**. Split your evals
  into a **capability suite** (low pass rate, a hill to climb) and a **regression
  suite** (near-100%). For your capstone agent, *add* a lightweight trajectory
  check (the B1 2-1 nuance: numeric scores alone are unstable).

**Deliverable:** a judge that scores agent runs; a capability set and a regression
set with the expected opposite pass-rate profiles.

**Term:** *end-state evaluation*, *LLM-as-judge*, *rubric*, *calibration /
Cohen's kappa*, *capability vs regression eval*.

**Resource:** Anthropic "Demystifying evals for AI agents" (T1, research Part B).

## D14 — Observability: OpenTelemetry GenAI tracing

**Why:** In production you debug agents from traces, not print statements. The
OTel GenAI semantic conventions are the emerging standard — but still unstable,
so you'll build an adapter, not hard-code it.

**Do:**

- Read research Part B (B4 the four span ops, B5 the instability warning).
- In Astra: instrument the loop with spans for the four operations —
  `create_agent`, `invoke_agent`, `invoke_workflow`, `execute_tool` — with the
  required attributes (`gen_ai.operation.name`, `gen_ai.provider.name`). Export
  to **any off-the-shelf backend** (Jaeger). Wrap the attribute names behind a
  thin adapter (they will break — semconv is in Development).

**Deliverable:** a Jaeger trace of a multi-tool agent run showing the four span
types nested correctly.

**Term:** *span*, *GenAI semantic conventions*, *span kind (CLIENT/INTERNAL)*,
*trace*, *cost/latency/token attributes*.

**Resource:** OTel GenAI agent-spans spec (T2, research Part B). Mind the date.

## D15 — Interop: MCP client + deferred tool loading + A2A

**Why:** MCP (agent↔tool) and A2A (agent↔agent) are the 2026 interop stack. And
the "too many tools" problem is real — loading every tool definition upfront can
cost hundreds of thousands of tokens.

**Do:**

- Read research Part E (E1 tool design, E2 code-execution-with-MCP, E3 MCP+A2A).
- In Astra: implement an **MCP client** (stdio transport) so external MCP tools
  appear identical to built-in tools. Add **deferred / dynamic tool loading**
  (Claude Code's `ToolSearch` pattern, or the file-tree + progressive-disclosure
  approach) so tool definitions load on demand instead of all upfront.
- Read the A2A spec enough to write 5 lines: how AgentCard/Task/Message would let
  Astra talk to another agent, and why AgentCards being self-reported/unsigned is
  a real weakness.

**Deliverable:** Astra connects to a real MCP server (e.g. a filesystem MCP) and
calls its tools; deferred loading keeps the upfront tool-definition token cost
flat as tool count grows. Log the token savings.

**Term:** *MCP*, *A2A*, *AgentCard*, *deferred tool loading*, *progressive
disclosure*, *code execution with MCP*.

**Resource:** Anthropic "Code execution with MCP" + MCP/A2A specs (research Part E).

## D16 — Capstone: a research agent that uses everything

**Why:** Integration is the test. Build one agent that exercises the whole stack
and *evaluates itself*.

**Do:**

- In Astra, build a **research agent** that: (a) **orchestrates workers** (D8) for
  breadth-first sub-questions; (b) each worker does **agentic RAG with CRAG**
  (D10–D11) over a corpus; (c) uses the **memory tool** (D9) for cross-session
  state; (d) is fully **traced** via OTel (D14); (e) is **scored by your judge**
  (D13) with a capability + regression split.
- Run it on a real multi-part question. Produce the trace, the judge score, and a
  token-cost budget.

**Deliverable:** end-to-end run with: the answer, the Jaeger trace, the RAGAS +
judge scorecard, and a one-paragraph token-cost analysis (the ~15× multi-agent
multiple should be visible and *justified by the task value* — the C2 lesson).

**Term:** integration of all prior terms; *the agent is the product, the eval is
the proof*.

**Resource:** everything. This is the portfolio centerpiece.

### Phase D-II checkpoint

Astra now does what Claude Code doesn't: agentic RAG with corrective grading,
RAGAS + LLM-judge evaluation, cross-session memory, OTel tracing, MCP interop.
Write a 1-page recap in `agent/experiments/track-d2-recap.md`, then update
`notes/career-transition-research.md` Phase 0 — you've completed portfolio item 3
("agent + evals"), and Astra is now a public, inspectable from-scratch agent
framework.

---

# Track D checkpoint (both phases)

You can read any production agent codebase and re-implement any piece. You have a
hand-written agent framework in C# that reaches Claude Code's core *and* the
frontier half it omits. You can quantify the token cost of any agent design, know
when *not* to use RAG or multi-agent, and prove an agent works with numbers, not
vibes.

This maps directly to the career target: **Microsoft Applied AI Engineer II**
(agentic / RAG / evals, "AKS background directly relevant"), Dublin "Agent Cloud"
Senior AI SWE, Copilot Tuning, or external OpenAI/Google FDE roles. Astra is the
"equivalent experience" those JDs accept in place of a degree.

Write the final 1-page recap in `agent/experiments/track-d-recap.md` and revisit
`agent/why-agent.md`'s personal-notes section.

---

# What Track D does NOT cover

Deliberately out of scope (mirrors the model side's exclusions):

- **Vector DB / embedding internals.** Commodity. Use off-the-shelf; the agentic
  logic on top is the lesson.
- **Tracing backend internals.** Use Jaeger/an OTel collector; you build the
  instrumentation, not the storage engine.
- **The Claude Code product shell.** Ink/React TUI (EP14), Bridge/remote-control
  (EP13), telemetry/infra plumbing (EP16/EP17) — these are product packaging, not
  agentic core. Read the chapters if curious, but they aren't learning days.
- **Distributed multi-box agent serving.** A scaling/ops problem, separate from
  learning the agentic patterns.

---

# Combined cadence with the model side

Track D slots into the existing patterns in `notes/curriculum-v2-execution.md`.
One working approach for all four tracks:

| Mon | Tue | Wed | Thu | Fri |
|---|---|---|---|---|
| Track A (model) | Track D (agent) | Track C (math) | Track B (pretrain) | Track D (agent) |

Track D Phase D-I (core) is self-contained C# work — good for evenings when you
want to build, not read papers. Phase D-II pairs naturally with Track A's eval/
serving days (A9/A10) since both touch evaluation and inference. Do Phase D-I
before D-II; within each phase, days are mostly sequential (D10→D11→D12 build on
each other; D13–D15 are more independent).
