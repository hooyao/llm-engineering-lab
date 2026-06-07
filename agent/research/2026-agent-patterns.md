# 2026 Agent Engineering — verified patterns and primary sources

> This file is the **source of truth for the "frontier half"** of Track D (the
> half that Claude Code's source code does not teach: RAG, agent eval, memory,
> interop). It is the synthesized output of a fan-out deep-research pass (106
> sub-agents, 24 sources fetched, 120 claims extracted, 25 adversarially
> verified at 3 votes each, 0 refuted) plus a targeted supplemental pass that
> filled the two gaps the funnel missed (Agentic RAG internals, MCP/A2A interop).
>
> Every claim below is tagged with a **source tier** (see `../README.md` for the
> tiering rule). Treat Tier-1/primary as load-bearing; treat blog/secondary as
> directional. Where practitioners disagree, the disagreement is stated, not
> smoothed over.

---

## How to read the source tiers

| Tier | Meaning | Trust |
|---|---|---|
| **T1 — primary engineering** | Anthropic Engineering, Cognition, the vendor that built the thing | Load-bearing |
| **T2 — papers / specs** | arXiv, MCP/A2A spec, OpenTelemetry semconv | Load-bearing for *what the technique is*; check the date |
| **T3 — secondary** | Practitioner write-ups that demonstrably ran the thing in prod (talk about cost/latency/eval/failure modes) | Directional / corroborating |
| **—  filtered out** | Toy-demo blogs (weather/flight-booking, no cost/eval/failure discussion) | Not cited |

The judging rule for T3 is **not** the domain name. Company engineering blogs
live on `medium.com`; that doesn't make them shallow. The test is whether the
author discusses token cost, latency budget, context degradation, evaluation, or
failure modes. If yes → citable. If it's a happy-path toy demo → dropped.

---

## Part A — Context engineering & memory (strongest evidence in the corpus)

### A1. Context engineering is the core 2026 discipline, distinct from prompt engineering
**T1 · verified 3-0.** Prompt engineering = writing the instructions/system
prompt. Context engineering = "managing the entire context state (system
instructions, tools, MCP, external data, message history)" across multiple
inference turns. The guiding principle, verbatim: *"finding the smallest
possible set of high-signal tokens that maximize the likelihood of some desired
outcome."* Recommended posture given the field's pace: *"do the simplest thing
that works."*
→ https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

### A2. Context rot is real and measured
**T1 + T2 · verified 3-0.** As token count grows, a model's ability to
accurately recall information from context **declines across all models**
(severity varies) — so the agent gets less value per token *before* hitting the
hard context limit. Anthropic frames the mechanism as the transformer's n²
pairwise attention relationships depleting a limited "attention budget,"
producing *"a performance gradient rather than a hard cliff."*
**Caveat:** the single-root-cause (n²) framing is Anthropic's; the phenomenon is
independently grounded in Chroma's "Context Rot" study (18 frontier models) and
Liu et al. "Lost in the Middle" (2023), but Chroma is a vector-DB vendor (COI)
and is itself cautious — training-data length distribution and RoPE drift are
complementary causes. The *phenomenon* is solid; the *explanation* is not settled.
→ https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
→ https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools

### A3. Just-in-time retrieval beats pre-loading (this is *why Claude Code uses grep, not RAG*)
**T1 · verified 3-0.** Agents should maintain lightweight identifiers (file
paths, stored queries, web links) and **load data at runtime via tools** rather
than pre-loading everything. Claude Code uses a hybrid: `CLAUDE.md` is dropped
into context up front, while `glob`/`grep` retrieve files just-in-time —
**bypassing stale indexing entirely.** The tradeoff is stated honestly: runtime
exploration is slower than pre-computed retrieval.
This is the single most important finding for your curriculum's framing: the
"is RAG dead" debate, from Anthropic's side, is *"prefer agentic search over a
maintained vector index when the corpus is navigable."* It is a position, not a
universal law — see Part D for when a real RAG pipeline still wins.
→ https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

### A4. Three composable context-management primitives
**T1 · verified 3-0 (4 merged claims).** Anthropic ships three first-party-API
primitives, and they compose:
1. **Compaction** (`compact_20260112`) — summarize a near-full window and
   reinitialize. Tuning order, verbatim: *"Start by maximizing recall... then
   iterate to improve precision."*
2. **Tool-result clearing / context trimming** (`clear_tool_uses_20250919`) —
   drop stale, re-fetchable data or old thinking blocks *inside* the window.
3. **File-based memory tool** (`memory_20250818`) — move information *out* of the
   window so it survives across sessions. **The only primitive that survives a
   new session with an empty window.**
Known failure mode (named in the Managed Agents post): *"irreversible decisions
to selectively retain or discard context can lead to failures."*
→ https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools
→ https://www.anthropic.com/engineering/managed-agents

### A5. Compaction preserves substance, loses verbatim detail — so it is *insufficient alone*
**T1 · verified 3-0.** In Anthropic's own probe, compaction preserved high-level
task-central facts **3/3** but obscure appendix-table specifics **0/3**
(preserved: C. elegans lifespan, killifish, ~60% Drosophila ortholog rate; lost:
Table A5 I², Table A2 effect size, Table A7 epigenetic ratio). Conclusion:
compaction *"keeps the substance in compressed form but loses verbatim detail,"*
so it must be **paired with an external memory tool** for facts that may matter
later. (N=3/3 is illustrative of the pattern, not a statistical claim.)
→ https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools

### A6. Long-running agents need an initializer + coding-agent two-prompt harness
**T1 · verified 3-0.** Compaction alone is not enough for long-horizon work:
*"even a frontier coding model like Opus 4.5 running on the Claude Agent SDK in a
loop... will fall short of building a production-quality web app if only given a
high-level prompt."* Two named failure modes: **one-shotting** (runs out of
context mid-implementation) and **premature completion** (declares the job done
on seeing existing progress). The fix is a two-part architecture with an
*identical harness/tools/system-prompt*, differing only in the initial user prompt:
- **Initializer agent** (runs once): sets up `init.sh`, a `claude-progress.txt`
  log, a feature-list file, and an initial git commit.
- **Coding agent** (every subsequent session): make incremental progress, leave
  structured updates.
→ https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

### A7. Cold-start orientation steps
**T1 · verified 3-0.** Each fresh session must recover state from an empty
window *before* doing new work: run `pwd`; read git logs + progress files; read
the feature list and pick the highest-priority unfinished feature; restart the
dev server and run a basic end-to-end test to detect a broken state first. This
is exactly the pattern `notes/progress.md` already encodes for *this* repo — the
agent version is the same idea, automated.
→ https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

### A8. Decouple the brain from the hands; the session log is the durable component
**T1 · verified 3-0.** Architecture, verbatim: *"decouple the brain (Claude and
its harness) from both the hands (sandboxes and tools) and the session (the log
of session events). Each became an interface that made few assumptions about the
others, and each could fail or be replaced independently."* Execution containers
are **cattle, not pets**. Because the session log sits *outside* the harness,
nothing in the harness needs to survive a crash — recover via
`wake(sessionId)` + `getSession(id)`. Concrete interfaces given:
`execute(name, input) -> string`, `provision({resources})`.
→ https://www.anthropic.com/engineering/managed-agents

### A9. Harnesses encode model-generation assumptions that rot
**T1 · verified 3-0.** Context-reset logic added to handle Sonnet 4.5's *"context
anxiety"* (wrapping up prematurely near its limit) *"became dead weight"* on Opus
4.5, which no longer had the behavior. Independently corroborated by Cognition's
"Rebuilding Devin for Claude Sonnet 4.5" (a third party, same premature-
termination observation, capped a 1M beta at 200k to mitigate). **Lesson for a
hand-built harness: don't over-fit workarounds to one model generation.**
→ https://www.anthropic.com/engineering/managed-agents

---

## Part B — Agent evaluation & observability

### B1. Grade the end-state, not the trajectory (with a nuance)
**T1 · verified 3-0 / 2-1.** *"It's often better to grade what the agent
produced, not the path it took,"* because *"agents regularly find valid
approaches that eval designers didn't anticipate."* The most consistent judge
Anthropic found: **a single LLM call** outputting a 0.0–1.0 score plus a
pass/fail grade across a fixed rubric (factual accuracy, citation accuracy,
completeness, source quality, tool efficiency).
**The 2-1 nuance (important):** independent 2026 work (Lau "Same Input, Different
Scores" arXiv:2603.04417; TrustJudge ICLR 2026; CORE arXiv:2509.20998) shows
single *numeric* judge scores are unstable in **absolute** terms, and high-stakes
agents benefit from **adding** trajectory/path checks. Anthropic itself still
tracks `tool_calls`/`turns`/`tokens` for debugging. So: end-state grading is a
strong **default**, not a universal rule. For a coding agent, add path checks.
→ https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
→ https://www.anthropic.com/engineering/multi-agent-research-system

### B2. Give the judge an escape hatch and calibrate it against humans
**T1 · verified 3-0.** To avoid hallucinated grades, *"give the LLM a way out,
like... return Unknown when it doesn't have enough information,"* and *"LLM-as-
judge graders should be closely calibrated with human experts"* (track Cohen's
kappa; practitioner consensus is refine until kappa > 0.8 before trusting it).
→ https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

### B3. Two eval types with opposite pass-rate targets
**T1 · verified 3-0.** **Capability evals** should *start at a low pass rate* — a
"hill to climb," deliberately targeting tasks the agent struggles with.
**Regression evals** should have a *near-100% pass rate* to catch backsliding.
Tasks **graduate** from the capability suite into the regression suite once
solved. (Corroborated by arXiv:2602.18029.)
→ https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents

### B4. OpenTelemetry GenAI semantic conventions standardize agent tracing
**T2 · verified 3-0.** Four span operations, each identified by
`gen_ai.operation.name`: **`create_agent`, `invoke_agent`, `invoke_workflow`,
`execute_tool`**. Span kind is CLIENT for hosted services (OpenAI Assistants
API, AWS Bedrock Agents) and INTERNAL for in-process frameworks (LangChain,
CrewAI). Attribute schema covers agent identity (`gen_ai.agent.id/name/
description/version`, `gen_ai.workflow.name`, `gen_ai.conversation.id`);
`gen_ai.operation.name` and `gen_ai.provider.name` are **Required**, identity
attrs are Conditionally Required.
→ https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/

### B5. ...but the GenAI semconv is NOT stable yet
**T2 · verified 3-0.** Overall status is **Development**; only `error.type`,
`server.address`, `server.port` are Stable. The newest experimental version is
gated behind `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`. As of
May 2026 there is **no stabilization timeline** (semconv v1.41.x). Expect
breaking changes — build a thin adapter, don't hard-code the attribute names.
→ https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/

---

## Part C — Multi-agent orchestration (the loudest disagreement)

### C1. Orchestrator-worker beat single-agent by 90.2% on breadth-first research
**T1 · verified 3-0.** Anthropic's multi-agent research system (Opus 4 lead
orchestrating Sonnet 4 subagents, each in a *clean context window*, returning
condensed **1,000–2,000-token** summaries) outperformed single-agent Opus 4 by
**90.2%** on their internal research eval, especially for breadth-first queries.
**Caveats (do not drop these):** the eval is internal, self-reported,
unreproduced; ~80% of the performance variance is attributable to token usage —
so the gain is partly a compute-spend confound. Scoped to breadth-first
parallelizable research, **not** "multi-agent always wins."
→ https://www.anthropic.com/engineering/multi-agent-research-system

### C2. Multi-agent is ~15× token cost
**T1 · verified 3-0.** Verbatim: *"agents typically use about 4x more tokens than
chat interactions, and multi-agent systems use about 15x more tokens than
chats"*; *"token usage by itself explains 80% of the variance"* in BrowseComp;
multi-agent *"requires tasks where the value of the task is high enough to pay
for the increased performance."*
→ https://www.anthropic.com/engineering/multi-agent-research-system

### C3. Multi-agent hurts on tightly-coupled tasks — the two-camp consensus
**T1 · verified 3-0.** Anthropic: *"some domains that require all agents to share
the same context or involve many dependencies between agents are not a good
fit"*; *"most coding tasks involve fewer truly parallelizable tasks than
research."* Cognition ("Don't Build Multi-Agents," Walden Yan, June 2025):
*"running multiple agents in collaboration only results in fragile systems...
context isn't able to be shared thoroughly enough."* Academic grounding:
CodeCRDT (arXiv:2510.18893, up to 39.4% slowdown on coupled tasks); Amdahl's-Law
analysis (arXiv:2503.15703). **Both camps agree: not universal, and working
patterns keep writes single-threaded.**
Cognition's April 2026 follow-up ("Multi-Agents: What's Actually Working")
partially reverses the June 2025 stance but confirms the original observations
"still hold today for parallel-writer swarms."
→ https://www.anthropic.com/engineering/multi-agent-research-system
→ https://cognition.ai/blog/dont-build-multi-agents
→ https://cognition.ai/blog/multi-agents-working

### C4. The six "Building Effective Agents" patterns (the canonical vocabulary)
**T1 · supplemental fetch.** The named patterns, with the tradeoff for each:

| Pattern | What it is | When | Tradeoff |
|---|---|---|---|
| **Prompt chaining** | decompose into a fixed sequence, each call processes the previous output; optional programmatic "gate" | task cleanly decomposes into fixed subtasks | trades latency for accuracy |
| **Routing** | classify input, direct to a specialized followup | distinct categories better handled separately | needs accurate classification |
| **Parallelization** (sectioning / voting) | run subtasks (or the same task) concurrently, aggregate | speed, or multiple perspectives for confidence | added calls/cost |
| **Orchestrator-workers** | central LLM dynamically breaks down + delegates + synthesizes | *can't predict the subtasks needed* | subtasks aren't pre-defined |
| **Evaluator-optimizer** | one LLM generates, another critiques, in a loop | clear eval criteria + iterative refinement helps | loop cost |
| **Autonomous agent loop** | LLMs using tools on environmental feedback in a loop | open-ended, unpredictable step count | compounding errors, higher cost |

Core thesis: **workflows** = LLMs orchestrated through predefined code paths;
**agents** = LLMs dynamically directing their own process. The "simplest
solution" principle: *"add complexity only when it demonstrably improves
outcomes."* Foundational unit = the **augmented LLM** (model + retrieval + tools
+ memory). Three principles: simplicity, transparency (show planning steps),
careful agent-computer interface. (Note: this post does **not** mention ReAct by
name — ReAct is the academic prior, arXiv:2210.03629.)
→ https://www.anthropic.com/research/building-effective-agents

---

## Part D — Agentic RAG (the half Claude Code's source does not cover)

> Claude Code deliberately has no RAG (Part A3). So the source layer
> (`refs/claude-code-sourcemap`) is silent here — this entire part rests on T2
> papers and T3 prod write-ups. This is the clearest example of *why source code
> alone is half the picture.*

### D1. Agentic RAG = retrieval wrapped in a decision loop
**T2 · supplemental.** The 2025 survey (Singh et al., arXiv:2501.09136,
"Agentic Retrieval-Augmented Generation") frames the shift: the LLM stops being
a passive consumer of retrieved chunks and becomes an agent that *plans what to
retrieve, decides how, critiques what it gets, and adapts.* The pipeline becomes
a **loop, not a line.** Retrieval is exposed to the agent **as a tool it invokes
as needed** during multi-step reasoning — which is exactly the same primitive as
Part A3, just pointed at a vector index instead of a filesystem.
→ https://arxiv.org/html/2501.09136v4

### D2. The five named production patterns (2026)
**T2/T3 · supplemental.**
- **Self-RAG** — the model emits *reflection tokens* deciding when to retrieve
  and whether retrieved content is relevant/supported (arXiv:2310.11511).
- **Corrective RAG (CRAG)** — a *lightweight retrieval evaluator* scores each
  retrieved doc; below-threshold results trigger **corrective routing**
  (query refinement, or a web-search fallback to supplement the internal index)
  *before* generation (arXiv:2401.15884). A 2025 MDPI benchmark reported CRAG at
  Precision@5 = 0.69, 10.5% hallucination, 240 ms latency — *"often the most
  practical first step into Agentic RAG: it adds one classification layer rather
  than a full re-architecture."*
- **Adaptive RAG** — a classifier picks pipeline depth per query (cheap path for
  easy queries, multi-hop for hard ones).
- **ReAct-over-documents** — reason-act loop using retrieval tools.
- **Multi-hop query decomposition** — break a complex query into sub-queries,
  retrieve each, recompose.
→ https://arxiv.org/html/2501.09136v4 · https://humanloop.com/blog/rag-architectures

### D3. How to evaluate RAG — RAGAS metrics (concrete definitions)
**T2 · supplemental.** The metrics you will implement against your own retrieval
loop:
- **Faithfulness** = (claims in the response supported by retrieved context) ÷
  (total claims in the response). Decompose answer into atomic claims, verify
  each against context. Inputs: `user_input`, `response`, `retrieved_contexts`.
  Range 0–1. *This is the anti-hallucination metric.*
- **Context Precision** — of the retrieved chunks, how many are actually
  relevant (signal vs noise in retrieval).
- **Context Recall** — of the chunks needed to answer, how many were retrieved
  (did retrieval miss anything). Needs ground-truth reference.
- **Response Relevancy** — does the answer actually address the question.
- **Tool Call Accuracy** — *"how accurately an LLM agent invokes tools compared
  to expected tool calls"* (sequence + arguments). Inputs: `user_input`,
  `reference_tool_calls`. Strict- or flexible-order.
- **Agent Goal Accuracy** — binary: did the agent achieve the user's goal
  (with-reference compares end state to an expected outcome; without-reference
  infers goal + outcome from the conversation).
- **Topic Adherence** — does the agent stay within predefined domains
  (precision/recall/F1 against `reference_topics`).
→ https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

---

## Part E — Tool use, MCP & interop

### E1. Tool design principles (designing *for agents*, not for developers)
**T1 · supplemental.** A tool is *"a contract between deterministic systems and
non-deterministic agents."* Five named principles:
1. **Choose the right tools** — more tools isn't better. *Consolidate* multi-step
   workflows into one tool (`schedule_event` over `list_users`+`list_events`+
   `create_event`; `get_customer_context` over three separate fetches). Each tool
   gets a clear, distinct purpose; overlapping tools distract the agent.
2. **Namespace** — group related tools by prefix (`asana_search`, `jira_search`).
   Prefix- vs suffix-namespacing has non-trivial, model-dependent effects — test it.
3. **Return meaningful context** — only high-signal info. Resolve cryptic UUIDs
   to semantic names (*"significantly improves Claude's precision... by reducing
   hallucinations"*). Offer a `response_format` enum (`concise` ≈ ⅓ the tokens of
   `detailed`).
4. **Token efficiency** — pagination/filtering/truncation with sensible defaults
   (Claude Code caps tool responses at 25,000 tokens). **Prompt-engineer your
   error messages** to be specific and actionable, not opaque tracebacks.
5. **Prompt-engineer tool descriptions** — *"one of the most effective methods."*
   Write as if for a new hire; name params unambiguously (`user_id` not `user`).
   Even small refinements yield measurable SWE-bench gains.
Evaluation: build realistic task evals, collect runtime/tool-calls/tokens/errors
(not just accuracy), read raw transcripts (*"what agents omit... can be more
important than what they include"*), don't over-specify the expected tool path
(multiple valid paths exist).
→ https://www.anthropic.com/engineering/writing-tools-for-agents

### E2. Code execution with MCP — the "too many tools" fix
**T1 · supplemental.** As tool counts grow, loading all MCP tool definitions
upfront can cost *"hundreds of thousands of tokens before reading a request,"*
and intermediate results flow through the model twice. Fix: **present MCP servers
as code APIs** the agent calls via code execution, with a **file-tree of tools**
(one file per tool under `./servers/<server>/`) and **progressive disclosure** —
the agent lists directories and reads only the tool files it needs (or a
`search_tools` tool with a detail-level parameter). Filter results in the
execution environment (*"the agent sees five rows instead of 10,000"*). Headline
claim: **150,000 → 2,000 tokens, 98.7% reduction.** Caveat: needs a sandboxed
execution environment. (Cloudflare reached the same conclusion: "Code Mode.")
This is the same idea as Claude Code's deferred `ToolSearch` loading.
→ https://www.anthropic.com/engineering/code-execution-with-mcp

### E3. MCP and A2A are complementary standards (the 2026 interop stack)
**T2/T3 · supplemental.** **MCP** (Anthropic, 2024; under the Linux Foundation's
Agentic AI Foundation by late 2025; adopted by OpenAI and Google) = the
agent↔tool interface. **A2A** (Agent2Agent; Google, April 2025; donated to the
Linux Foundation June 2025; **v1.0 in 2026**, Apache 2.0) = the agent↔agent
interface. As of April 2026: **150+ organizations**, 22k+ GitHub stars, SDKs in
5 languages (Python/JS/Java/Go/.NET), in production inside Azure AI Foundry and
Amazon Bedrock AgentCore. Primitives: **AgentCard** (capability discovery),
**Task**, **Message**, **Artifact**. Known weakness: AgentCard capability claims
are *self-reported and (until v0.3) unsigned* — there is no identity/attestation
layer yet. MCP's own maintainers describe A2A as complementary, not competing.
→ https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1108
→ (A2A v1.0 / Linux Foundation / 150-org adoption corroborated across multiple
2026 protocol-map write-ups; treat the exact counts as directional T3.)

---

## What the corpus does NOT settle (open questions)

- **Single numeric judge scores are unstable in absolute terms** (B1). For your
  capstone, plan to *add* a lightweight trajectory check, not rely on one float.
- **Context-rot's mechanism** (A2) is contested even by its own measurer. Trust
  the phenomenon, not the n² story.
- **OTel GenAI semconv will break** (B5). Adapter, not hard-coding.
- **A2A identity/attestation is unsolved** (E3). AgentCards are self-reported.
- **The Cognition reversal** (C3): June-2025 "don't" → April-2026 "here's what
  works" — the decision rule is task coupling + single-threaded writes, not a
  blanket yes/no. Read both posts before building multi-agent into Astra.

---

## Source ledger (primary + load-bearing)

**Anthropic Engineering (T1):**
- effective-context-engineering-for-ai-agents
- effective-harnesses-for-long-running-agents
- managed-agents (Scaling Managed Agents: decoupling brain from hands)
- demystifying-evals-for-ai-agents
- multi-agent-research-system
- writing-tools-for-agents
- code-execution-with-mcp
- research/building-effective-agents
- platform.claude.com cookbook: context-engineering tools

**Cognition (T1):** dont-build-multi-agents · multi-agents-working

**Specs / papers (T2):**
- opentelemetry.io GenAI agent-spans semconv
- arXiv:2501.09136 (Agentic RAG survey) · 2310.11511 (Self-RAG) ·
  2401.15884 (CRAG) · 2210.03629 (ReAct) · 2603.04417 (judge instability) ·
  docs.ragas.io (RAGAS metrics)

**MCP / A2A (T2):** modelcontextprotocol spec + discussion #1108 · A2A v1.0
(Linux Foundation)

_Research run: deep-research workflow, 106 sub-agents, 25/25 verified claims
confirmed at 3 votes, 0 refuted; plus a targeted brave-search / Search-MCP
supplemental pass for the RAG and MCP/A2A gaps the funnel missed._
