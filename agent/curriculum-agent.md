# Track D — Astra Product Engineering (20 evenings, two phases)

> **Why are you doing this?** See `agent/why-agent.md`. Re-read when motivation
> flags. **Where it lands:** `notes/career-transition-research.md` §2 ranks
> "AI/LLM Agent Engineer" as the most reachable model-facing role for your
> profile. This file is the product-engineering curriculum for Astra, not a list
> of everything that may appear in an agent interview. Interview-only breadth
> lives in `agent/curriculum-agent-interview.md`.

This is the **execution plan** for the independent Astra product. It mirrors the
structure of `notes/curriculum-v2-execution.md` (Tracks A/B/C): one product
problem per day, a runnable artifact, a measured payoff, and an end-of-track
checkpoint. A curriculum slot is not permission to add a subsystem: every
production change must first pass Astra's feature-admission gate.

## What this track assumes

- 10+ years software architecture (C#/.NET deep): `git`, async, streaming,
  concurrency, DI — zero introduction. Astra is already C#/.NET 10 with an
  agent-loop skeleton.
- Agentic ML: **none assumed**. Every agent term is defined the first time it
  appears (see each day's **Term** block).
- You've read `agent/README.md` (the product-vs-interview boundary and
  source-tiering rule).

## The daily flow (evidence before abstraction)

```
1. FAILURE    reproduce a real autonomous/coding-agent failure
2. SOURCE     read Claude Code source, Manus engineering, or another primary source
3. CONTRACT   state the invariant and why Astra owns it
4. IMPLEMENT  add the smallest compatible contract
5. PAYOFF     rerun the task and measure the improvement
```

Claude Code source is authoritative for what Claude Code does. Manus's official
context-engineering write-up defines a useful general-agent boundary. Neither
is a feature checklist. `agent/research/2026-agent-patterns.md` can suggest a
hypothesis, but it cannot by itself justify an Astra subsystem.

## Conventions

- **Each day produces a runnable artifact.** Product code lands in Astra only
  after the failure and ownership case are demonstrated. Notes/analysis land
  in `agent/experiments/dNN-<slug>/`.
- **Re-implement, don't copy.** Astra's own CLAUDE.md says it: *"Learn from
  Claude Code, don't copy it."* The source is a reference for design decisions,
  not a paste buffer.
- **Quantify.** Every agent design has a token cost. State it (tokens/turn,
  tokens/tool-result, context-window headroom) the same way the model side
  quantifies VRAM. "Uses a lot of context" is not an answer.
- **Commodity infra is off-the-shelf.** Vector DB, embeddings, tracing backend:
  use existing ones. The agentic logic on top is what you write.
- **Interview demand is not product demand.** Intent taxonomies, generic
  workflows, production RAG, and timed interview mocks belong to
  `curriculum-agent-interview.md` unless a separate Astra failure proves a core
  requirement.

## Track layout

- **Phase D-I — Agent Core (D1–D8):** the *how*. Re-implement the Claude Code
  agent core in Astra. Source-backed.
- **Phase D-II — Autonomous Runtime Hardening (D9–D20):** move from a working
  coding loop to a Manus-style general autonomous core, while using coding tasks
  as the first benchmark. Product-evidence-backed.

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

## D2 — The tool contract: behavioral classification and lazy execution

**Why:** A tool is *"a contract between deterministic systems and
non-deterministic agents."* The design choice that matters: behavior is
**input-dependent**, not type-dependent.

**Do:**

- Read `architecture/02-tool-system.md` + source `Tool.ts` and a couple of
  `tools/*/` implementations (e.g. `GrepTool`, `FileEditTool`).
- In Astra: separate immutable `ToolDefinition` metadata/classification from
  invocation-time `IToolExecutor`. `ToolDefinition.Classify(input)` returns the
  behavioral category (`Read`, `Write`, `Execute`, or fail-closed `Other`)
  without constructing an executor. Implement a read tool and a write tool;
  note that the Bash definition classifies `"ls"` as `Read` and `"rm -rf"` as
  `Execute` from the input rather than the implementation type.
- Advertise definitions to the model, then activate a keyed transient executor
  only after lookup and permission. Unused and denied tools must not construct
  executor instances.

**Deliverable:** two registered tool definitions and lazy executors dispatched
by the D1 loop; tests prove input-dependent classification, zero activation for
an unused/denied tool, and a fresh executor per admitted invocation.

**Term:** *tool definition*, *tool executor*, *behavioral classification*,
*input schema*, *fail-closed defaults*, *invocation-time activation*.

**Resource:** Anthropic, "Writing tools for agents" (T1) — the five principles.
https://www.anthropic.com/engineering/writing-tools-for-agents

## D3 — Tool orchestration: concurrent reads, serial barriers

**Why:** Within one turn the model may request several tools. Independent reads
can overlap; writes and execution effects are ordering barriers. Results still
have to return to the model in its original call order.

**Do:**

- Read `architecture/02-tool-system.md` (orchestration section).
- In Astra: use the D2 behavioral classification to coalesce adjacent `Read`
  calls into bounded concurrent batches. Treat `Write`, `Execute`, and `Other`
  as individual serial barriers. Preserve emission order when appending results
  even when reads finish out of order.
- Do not sort calls across a barrier. Capability-catalog ordering and deferred
  loading belong to D12/D18, when a real large catalog exists.

**Deliverable:** a turn with 3 read calls + 1 write executes reads concurrently
and the write alone; a test asserts ordering and that reads overlapped.

**Term:** *tool orchestration*, *stable partition*, *serial barrier*,
*concurrency safety*, *program order*.

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
  a global single-writer lane, but do not expose write-capable workers until
  D9's strict file-version and atomic MultiEdit invariants pass.
- Read both `research/2026-agent-patterns.md` Part C and the two Cognition posts.
  Write 5 lines on when you'd reach for multi-agent vs a single agent.

**Deliverable:** coordinator dispatches 2 isolated workers in parallel, collects
their summaries, synthesizes, and logs the token and wall-time multiple against
the same single-agent task. Treat the measured result as the answer; do not use
the published ~15× research-system token multiple as an expectation.

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

# Phase D-II — Autonomous Runtime Hardening (12 evenings)

**Goal at the end:** Astra is a Manus-style general autonomous-agent core whose
coding specialization can be compared honestly with Claude Code and Codex. It
owns execution, environment, context, recoverable task state, safety, recovery,
measurement, and capability integration. It does not become a generic workflow
or RAG platform.

## D9 — Strict file versions + atomic MultiEdit

**Why:** D8 can schedule a write worker, but exposing it is unsafe while a worker
can act on stale file content or while two edits from one model response can
partially apply. This is a demonstrated coding-agent correctness failure, not a
speculative abstraction.

**Do:**

- Have bounded reads return a stable file-version token for the exact bytes the
  model observed.
- Require an expected version for edits to an existing file. A mismatch returns
  a conflict and requires a fresh read; do not auto-rebase model edits.
- Normalize all edits to one file from the same model response into one ordered,
  validated transaction. Apply all or none through an atomic replace.
- Cover delete/create/replace races, BOM and line-ending preservation, duplicate
  matches, cancellation, and permission ordering.
- Enable write-capable workers only after the stale-write and atomicity tests
  pass.

**Deliverable:** reproduce two edits derived from the same old `hello world`
bytes, prove the second stale edit cannot silently apply, and prove a multi-edit
batch leaves the original file unchanged if any member fails.

**Term:** *optimistic concurrency*, *version token*, *stale read*, *atomic
transaction*, *conflict*, *no automatic rebase*.

**Resource:** the existing D8 stale-write teaching note, Claude Code file-edit
source, and the target OS atomic-replace contract.

## D10 — Durable session log + crash-safe resume

**Why:** An autonomous task can outlive one process or UI connection. The
conversation list in memory is not durable ownership of actions, observations,
artifacts, compaction references, and task status.

**Do:**

- Define an append-only, versioned session record for accepted user inputs,
  model outputs, tool calls/results, compaction replacements, task state, and
  artifact references.
- Rebuild a session deterministically without replaying completed side effects.
- Distinguish durable truth from UI events and from provider-specific payloads.
- Crash at controlled points: before a tool starts, after the side effect but
  before result persistence, and after persistence.
- Keep the persistence implementation replaceable; Astra owns the session
  semantics, not a database engine.

**Deliverable:** terminate a real multi-step task mid-run, restart Astra, resume
from the durable log, and prove a completed side effect is not duplicated.

**Term:** *event log*, *resume*, *replay*, *write-ahead state*, *artifact
ownership*, *side-effect boundary*.

**Resource:** Manus's file-system-as-context principle, Anthropic's long-running
agent harness guidance, and the existing Astra event protocol.

## D11 — Execution environment + sandbox lifecycle

**Why:** A Manus-style agent acts in an environment. Directly opening local
files and processes from tool classes prevents isolated execution, remote
workers, recoverable artifacts, and one enforceable lifecycle boundary.

**Do:**

- Define the smallest task-scoped execution-environment contract for paths,
  processes, artifacts, environment identity, and cancellation/disposal.
- Keep file, shell, browser/computer, and future remote capabilities behind the
  environment boundary without forcing them into one inheritance hierarchy.
- Provide a local implementation and a deterministic test implementation.
- Integrate an existing OS/container sandbox rather than implementing an
  isolation kernel.
- Prove that task cancellation tears down all environment-owned processes and
  that one worker cannot access another worker's private environment state.

**Deliverable:** run the same task in the local and isolated implementations,
collect an artifact, cancel midway, and verify complete environment cleanup.

**Term:** *execution environment*, *sandbox lifecycle*, *artifact*, *task
isolation*, *capability boundary*.

**Resource:** Manus's virtual-machine environment model and Claude Code/Codex
sandbox behavior, reconciled against Astra's existing file/process lifecycle.

## D12 — Stable action space + state-aware tool policy

**Why:** Dynamically adding and removing tool definitions damages prefix-cache
stability and leaves prior observations referring to missing actions. A large
unchanged action space, however, increases wrong-tool selection. Manus resolves
this with state-aware action constraints rather than arbitrary schema churn.

**Do:**

- Keep the advertised capability catalog deterministically ordered and stable
  across an agent run.
- Add provider-neutral `Auto`, `Required`, and `SpecifiedSubset` action policies.
- Use provider constrained decoding where available; retain a fail-closed
  dispatch/permission check when it is not.
- Make task state select an allowed action subset without granting new
  authority or mutating historical definitions.
- Measure prefix stability, invalid/wrong action rate, and tokens when the
  catalog grows.

**Deliverable:** drive one task through at least three action states, prove the
serialized prefix stays stable, and show an invalid action is rejected even
when the provider cannot enforce the subset.

**Term:** *action space*, *constrained decoding*, *tool-choice policy*, *state
machine*, *stable capability catalog*.

**Resource:** Manus "Mask, Don't Remove," provider tool-choice contracts, and
Claude Code deferred-tool behavior.

## D13 — Long-horizon plan state + file-backed working memory

**Why:** A long task drifts as its original objective moves away from the recent
context. Manus externalizes task state into files and repeatedly brings the
current plan back into recent attention. This is working state for the active
task, not a generic semantic-memory product.

**Do:**

- Define explicit goal, step, status, blocker, and artifact references for one
  active task.
- Persist the plan through the D10 session mechanism and expose controlled plan
  updates as an agent capability.
- Reinsert a bounded current-plan view near the tail without rewriting old
  action/observation history.
- Keep large observations recoverable by durable path/URL/artifact references,
  and distinguish restorable data from irrecoverable evidence.
- Compare long-horizon completion with and without plan recitation under the
  same model and tool budget.

**Deliverable:** a task requiring dozens of tool calls resumes after a restart,
retains its done criteria, and shows a measured reduction in goal drift or
repeated work.

**Term:** *working memory*, *plan recitation*, *goal drift*, *recoverable
reference*, *task state*.

**Resource:** Manus "Use the File System as Context" and "Manipulate Attention
Through Recitation," plus Astra's D7 compaction invariants.

## D14 — Failure evidence + retry/idempotency semantics

**Why:** Failed actions, stack traces, and environment errors are observations
the model can use to recover. Hiding them can make the agent repeat the same
mistake. Blind retries are also unsafe once a tool may have produced a side
effect.

**Do:**

- Preserve bounded failed actions and observations in the agent history and
  durable log.
- Classify provider failures, transport failures, policy denials, tool failures,
  ambiguous side-effect outcomes, and terminal task failures.
- Retry only where the operation and persistence point make it safe; add
  backoff, jitter, attempt/time/token budgets, and cancellation.
- Define idempotency semantics for side-effecting tools and resume recovery for
  the "effect happened, result was not persisted" case.
- Wire the deferred reactive-compaction retry through the same bounded policy.

**Deliverable:** inject deterministic failures before, during, and after a tool
side effect; prove Astra either recovers once or stops with an explicit
ambiguous outcome, never silently duplicates the effect.

**Term:** *failure evidence*, *retryable*, *idempotency key*, *ambiguous
outcome*, *backoff with jitter*, *retry budget*.

**Resource:** Manus "Keep the Wrong Stuff In," .NET resilience primitives, and
the D4 process-tree cancellation contract.

## D15 — Trust provenance + agent security boundaries

**Why:** System instructions, user requests, retrieved pages, repository files,
and tool output do not carry equal authority. A general agent must preserve
where content came from so untrusted observations cannot grant permission or
silently rewrite policy.

**Do:**

- Attach origin and trust metadata to system, user, attachment, tool, worker,
  and external-content observations without destabilizing provider prefixes.
- Keep authorization decisions outside model-generated text and enforce them at
  the capability boundary.
- Test direct and indirect prompt injection, tool-output injection, worker-report
  injection, path escape, secret leakage into logs, and malicious project
  instructions.
- Integrate an existing sandbox and host policy for OS enforcement; do not claim
  prompt filtering is a sandbox.
- Keep application PII classification, tenant RBAC/ABAC, compliance, and content
  moderation outside Astra Core while preserving the hooks they require.

**Deliverable:** a red-team suite where untrusted file/web/tool content asks for
new authority and Astra consistently denies the escalation while still making
the content available as evidence.

**Term:** *provenance*, *trust boundary*, *prompt injection*, *authority*,
*data exfiltration*, *defense in depth*.

**Resource:** Claude Code's permission/workspace-trust source, Codex sandbox
behavior, and current prompt-injection threat guidance.

## D16 — OpenTelemetry tracing + operator-visible state

**Why:** A long-running autonomous task cannot be debugged from final text or
console output. Operators need one trace that connects model calls, tool
attempts, workers, environment lifecycle, compaction, retries, and artifacts.

**Do:**

- Add an adapter over the evolving OpenTelemetry GenAI semantic conventions.
- Correlate session/task/worker IDs and emit spans for model invocation, tool
  execution, compaction, environment operations, and worker coordination.
- Record latency, token/cache counts, retry attempts, outcomes, and bounded
  error metadata without leaking prompts, secrets, or private worker history.
- Export to an off-the-shelf collector/backend and define a stable internal
  telemetry contract independent of that backend.

**Deliverable:** diagnose an injected intermittent worker/tool failure from one
trace and identify its exact retry, environment, and completion path.

**Term:** *trace*, *span*, *correlation*, *semantic convention*, *redaction*,
*cardinality*.

**Resource:** OpenTelemetry GenAI semantic conventions and Astra's typed event
and completion protocols.

## D17 — Agent evaluation harness

**Why:** "Surpass Claude Code/Codex" is meaningless without repeatable tasks and
outcome measures. Evaluation is a product measurement surface, not model logic
inside `AgentLoop`.

**Do:**

- Build an eval harness outside the runtime Core that can launch an agent,
  provision an environment, capture the final state and trace, and grade it.
- Separate capability suites from near-100% regression suites.
- Prefer deterministic end-state checks: tests, file content, repository state,
  artifacts, and explicit task invariants. Add bounded trajectory checks only
  where the outcome cannot expose the failure.
- Support an LLM judge with `Unknown` and calibration against human labels, but
  never make it the only grader.
- Record success, latency, input/output/cached tokens, tool calls, retries, and
  cost under a fixed model/environment configuration.

**Deliverable:** one capability suite and one regression suite that compare a
baseline loop with current Astra and catch a deliberately introduced
regression.

**Term:** *end-state evaluation*, *capability suite*, *regression suite*,
*trajectory check*, *judge calibration*, *reproducibility*.

**Resource:** Anthropic "Demystifying evals for AI agents" and the coding-agent
benchmark methodology selected for the task set.

## D18 — MCP capability transport

**Why:** A general core cannot compile every future capability into the runtime.
MCP is the first concrete external capability transport, but remote tools still
have to obey Astra's action-space, permission, cancellation, telemetry, and
lifecycle contracts.

**Do:**

- Implement one real MCP transport and normalize remote tools into Astra's
  definition/executor and permission contracts.
- Reconcile server discovery and reconnects with D12's stable action-space and
  prefix-cache requirements; do not silently mutate the catalog mid-action.
- Bound schemas, results, connection failures, and server-controlled names and
  descriptions as untrusted external input.
- Measure startup/tool-definition tokens and invocation latency with a large
  capability catalog.

**Deliverable:** connect a real MCP server and prove its tools pass the same
permission, cancellation, trust, result-bounding, and telemetry tests as a
built-in tool.

**Term:** *MCP*, *transport lifecycle*, *remote capability*, *capability
catalog*, *extension boundary*.

**Resource:** MCP specification, Claude Code/Codex MCP behavior, and Manus's
stable-action-space findings.

## D19 — Skills, hooks, and progressive capability disclosure

**Why:** Instructions and lifecycle customization are different from executable
remote tools. They need explicit formats and ownership so capability discovery
does not become arbitrary in-process plugin execution or invisible mutation.

**Do:**

- Load skills as bounded instruction/resource packages through progressive
  disclosure; advertise metadata before loading full content.
- Add pre/post lifecycle hooks through a language-neutral, bounded protocol.
- Ensure skills and hooks cannot grant permission, mutate durable history
  invisibly, or escape task/environment ownership.
- Preserve deterministic ordering, prefix stability, cancellation, and Native
  AOT.
- Keep A2A and arbitrary dynamic assembly/plugin loading outside the product
  until a concrete task demonstrates a requirement that workers, MCP, skills,
  and hooks cannot meet.

**Deliverable:** load one skill on demand and execute one lifecycle hook while
proving both remain visible in task state and cannot bypass permission or trust
boundaries.

**Term:** *skill*, *hook*, *progressive disclosure*, *lifecycle interception*,
*instruction package*.

**Resource:** Claude Code/Codex skill and hook behavior, reconciled with Astra's
feature-admission and security rules.

## D20 — Capstone: general autonomy + coding benchmark

**Why:** Integration is the test, and the product claim has two parts: a general
autonomous core and a coding specialization competitive with Claude Code and
Codex.

**Do:**

- Select at least one long-horizon general task that uses an isolated
  environment, files/artifacts, planning, recovery, and resume.
- Select at least one repository coding task with deterministic tests and a
  stale-write or injected-failure path.
- Run fixed-model/fixed-environment Astra baselines. Where practical, run the
  same coding task through Claude Code and Codex, documenting capability and
  environment differences instead of pretending the runs are identical.
- Grade end state, recovery, latency, tokens/cache, tool calls, retries, safety,
  and operator-visible trace quality.

**Deliverable:** a reproducible benchmark report with artifacts the learner runs
and inspects personally. Claims are limited to the measured tasks; no global
"best agent" conclusion.

**Term:** *benchmark validity*, *controlled comparison*, *task success*,
*recovery rate*, *cost/latency frontier*.

**Resource:** all product-track evidence. This is the Astra portfolio
centerpiece.

### Phase D-II checkpoint

Astra now has the runtime boundaries required by a long-running general agent:
strict writes, durable resume, task-scoped environments, controlled action
space, file-backed working state, explicit recovery semantics, trust
provenance, traces, evals, and bounded capability integrations. Its coding
specialization has a reproducible comparison suite rather than a feature-count
claim. Write a 1-page recap in `agent/experiments/track-d2-recap.md`.

---

# Track D checkpoint (both phases)

You can read a production autonomous-agent runtime and reason about its loop,
action space, environment, context, task state, side effects, recovery, trust,
measurement, and extension boundaries. You have a hand-written Manus-style core
in C# and a coding specialization measured against concrete Claude Code/Codex
tasks. You can explain not only what Astra implements, but why adjacent features
were deliberately excluded.

This maps directly to the career target: **Microsoft Applied AI Engineer II**
(agentic / RAG / evals, "AKS background directly relevant"), Dublin "Agent Cloud"
Senior AI SWE, Copilot Tuning, or external OpenAI/Google FDE roles. Astra is the
"equivalent experience" those JDs accept in place of a degree.

Write the final 1-page recap in `agent/experiments/track-d-recap.md`, continue
the separate `agent/curriculum-agent-interview.md` labs as needed, and revisit
`agent/why-agent.md`'s personal-notes section.

---

# What Track D does NOT cover

Deliberately outside the Astra product track:

- **Model training/fine-tuning.** Tracks A/B own model adaptation; Astra consumes
  model endpoints.
- **Generic workflow execution.** Use an existing workflow/durable-task engine
  and invoke Astra as a step. Astra owns autonomous task execution, not business
  DAGs.
- **Application RAG infrastructure.** Vector DBs, embeddings, document parsing,
  chunking, reranking, and application retrieval policy belong in application
  labs and integrations.
- **Application intent taxonomies and routers.** These are product policy, not a
  universal runtime contract.
- **Multi-tenant SaaS control plane.** Auth, billing, tenancy policy, compliance,
  and serving-fleet operations belong to the host product.
- **Tracing/eval storage backends.** Use off-the-shelf systems; Astra owns emitted
  semantics and the harness boundary.
- **Interview feature parity.** `curriculum-agent-interview.md` covers the topic
  without changing Astra.

---

# Combined cadence with the model side

Track D slots into the existing patterns in `notes/curriculum-v2-execution.md`.
One working approach for all four tracks:

| Mon | Tue | Wed | Thu | Fri |
|---|---|---|---|---|
| Track A (model) | Track D (agent) | Track C (math) | Track B (pretrain) | Track D (agent) |

Track D Phase D-I is the working core. Phase D-II hardens it into a general
autonomous runtime; most days are sequential because file transactions, durable
state, environments, recovery, and measurement establish each other's
correctness boundaries. The separate interview track can run between product
days without changing Astra's backlog.
