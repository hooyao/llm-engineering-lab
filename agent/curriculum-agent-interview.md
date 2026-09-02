# Applied Agent Interview Track (I1-I8)

This companion track prepares for Applied AI, Agent Engineer, Forward Deployed
Engineer, and agent-system-design interviews. It is deliberately separate from
the Astra product roadmap.

An interview topic does not justify an Astra feature. Labs in this track use
Astra through its public API, use an off-the-shelf application framework, or
remain system-design exercises. A change may move into Astra only when an
independent product failure passes the feature-admission gate in Astra's
`CLAUDE.md`.

Model training and fine-tuning remain in Tracks A/B. They are adjacent Applied
AI competencies, not agent-runtime responsibilities.

## Evidence behind this track

- Sierra's 2026 engineering process uses a Plan/Build/Review onsite: a two-hour
  AI-assisted build, product and code review, production discussion, a system
  design screen, and a medium-codebase debugging exercise.
- Public candidate reports repeatedly ask how to evaluate an agent or RAG
  system, protect sensitive retrieval data, debug intermittent tool failures,
  and reason about stateful workflows.
- Public interview corpora consistently include prompt/tool reliability,
  retrieval quality, online/offline evaluation, safety, production operations,
  and scenario-based tradeoffs.

These sources establish interview breadth. They do not define Astra's product
scope and are not treated as statistically representative question-frequency
data.

Source set:

- [Sierra: The AI-native interview](https://sierra.ai/blog/the-ai-native-interview)
- [LangChain candidate-report guide with source links](https://github.com/landedjobs/ai-interview-guides/blob/main/guides/langchain.md)
- [LangGraph/RAG candidate discussion referenced by that guide](https://www.reddit.com/r/LangChain/comments/1k662xc/got_grilled_in_an_ml_interview_today_for_my/)
- [A junior GenAI take-home and interview report](https://kaysnotes.medium.com/my-generative-ai-engineer-interview-experience-got-hired-6b3f1affc4e9)
- [Broad public AI-engineering interview corpus](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions)

## Ownership rules

- Deliverables live under `agent/interview/iNN-<slug>/`.
- Do not add an `Astra.Core` abstraction to make an interview lab convenient.
- Prefer an adapter or sample when exercising Astra with application
  infrastructure.
- Use existing workflow engines, vector stores, embedding models, document
  parsers, and tracing backends.
- Every lab ends with a visible payoff: a score, a before/after comparison, a
  runnable demo, or a timed mock with a rubric.
- Every system-design answer must name failure modes, measurement, security,
  latency, and token/cost boundaries.

## I1 - Agent-pattern selection: ReAct is one option, not the definition

**Goal:** choose the simplest execution pattern that fits the task rather than
defaulting every LLM application to an autonomous loop.

Cover:

- direct model call;
- prompt chaining;
- deterministic workflow;
- routing;
- parallel sectioning and voting;
- evaluator-optimizer;
- ReAct-style action/observation iteration;
- plan-and-execute;
- autonomous agent;
- orchestrator-worker.

Implement the same bounded task with a deterministic workflow and with an
agent loop. Compare task success, model calls, tool calls, tokens, and wall
time. Record observable actions and observations; do not persist raw
chain-of-thought.

**Deliverable:** `agent/interview/i01-pattern-selection/` with the decision
matrix and measured comparison.

## I2 - Intent classification and routing

**Goal:** route requests explicitly when product policy requires stable and
auditable behavior.

Build a standalone structured-output router with:

- a small, versioned intent taxonomy owned by the sample application;
- `Unknown`/abstain and out-of-domain behavior;
- multi-label cases;
- schema validation and repair/failure handling;
- a deterministic policy router baseline;
- an LLM router;
- optional model routing by quality, latency, and cost.

Evaluate accuracy, macro-F1, per-class precision/recall, abstention quality,
latency, and token cost. Do not add the sample taxonomy or router to Astra.

**Deliverable:** `agent/interview/i02-intent-routing/` with a labeled set,
confusion matrix, and policy-vs-LLM comparison.

## I3 - Stateful workflow orchestration

**Goal:** understand the production boundary between predefined workflows and
model-directed agents.

Use an off-the-shelf C# workflow/orchestration library. Exercise:

- typed workflow state;
- sequential and conditional nodes;
- parallel fan-out/fan-in;
- bounded cycles;
- checkpoint/resume;
- human-in-the-loop suspend/resume;
- retry and compensation boundaries;
- invoking an Astra agent as one workflow step.

Astra must remain a callable component; this lab must not turn Astra into a
generic workflow engine.

**Deliverable:** `agent/interview/i03-workflow-orchestration/` with a crash,
resume, and human-approval payoff.

## I4 - Production RAG, not a vector-database tutorial

**Goal:** design and debug the complete retrieval path expected in Applied AI
interviews while keeping commodity infrastructure off-the-shelf.

Build one small document corpus and compare:

- PDF/HTML parsing and provenance preservation;
- fixed, recursive, and semantic chunking;
- metadata filtering;
- dense, lexical, and hybrid retrieval;
- reranking;
- citation/attribution;
- unanswerable-query detection;
- prompt-injection-bearing retrieved content.

An Astra agent may consume retrieval as a tool. The parser, embedding model,
index, and reranker remain application infrastructure.

**Deliverable:** `agent/interview/i04-production-rag/` with a retrieval
ablation table and a query where the unanswerable path correctly refuses.

## I5 - Evaluation lifecycle

**Goal:** answer "How do you know the agent works?" with a reproducible system,
not one aggregate score.

Cover three levels:

1. Retrieval: Recall@k, MRR or nDCG, context precision/recall.
2. Generation and action: faithfulness, citation accuracy, answer relevance,
   tool-name/argument correctness, invalid-call rate.
3. End-to-end agent: task success, final-state invariants, bounded trajectory
   checks, latency, tokens, and cost.

Build golden datasets from hand-authored cases and curated production-like
traces. Calibrate an LLM judge against human labels, retain an `Unknown` escape
hatch, and separate capability from regression suites. Exercise offline
evaluation, CI gating, and a simulated online A/B or canary decision.

**Deliverable:** `agent/interview/i05-evaluation-lifecycle/` with a scorecard,
judge calibration result, and a deliberately failing regression gate.

## I6 - Production safety, reliability, and multi-tenancy

**Goal:** design the operational boundary around an agent without pretending a
single guardrail solves it.

Produce a system design covering:

- direct and indirect prompt injection;
- untrusted tool and retrieval observations;
- data exfiltration and secret handling;
- PII classification and retention;
- tenant context isolation and RBAC/ABAC;
- retries with backoff and jitter;
- rate limits and provider overload;
- idempotency for side-effecting tools;
- partial failure, replay, and dead-letter handling;
- provider/model fallback and quality degradation;
- per-request token, latency, and cost budgets;
- audit trails and incident response.

Reuse Astra's real permission, cancellation, and worker-isolation mechanisms as
evidence, while keeping application auth, compliance, and tenancy policy outside
the runtime.

**Deliverable:** `agent/interview/i06-production-boundaries/` with a threat
model, failure matrix, and one fault-injection demo.

## I7 - Framework translation and two-hour build

**Goal:** prove that first-principles Astra knowledge transfers to the
framework named in a job description.

Implement one bounded application twice: once with Astra's public API and once
with one relevant off-the-shelf framework. Map loop, state, tools, interrupts,
checkpointing, observability, and eval concepts explicitly. Do not chase
feature parity.

Then repeat as a two-hour Plan/Build/Review exercise. Scope down deliberately,
ship the unique behavior, demo it, and explain the path to production.

**Deliverable:** `agent/interview/i07-framework-build/` with the concept map,
two implementations, timing log, and review rubric.

## I8 - Senior interview loop

Run three independent mocks:

1. **System design (60 minutes):** clarify requirements, select workflow vs
   agent, design state/context/tools/eval/security/operations, quantify budgets,
   and enumerate failure modes.
2. **Debugging and review:** inspect a medium-sized unfamiliar repository and a
   cross-cutting draft PR, reproduce behavior, identify correctness and design
   issues, and improve it with tests.
3. **Project deep dive:** defend one Astra design using measured evidence,
   explain a rejected alternative, and state what remains unimplemented.

**Deliverable:** `agent/interview/i08-senior-loop/` with recordings or complete
notes, interviewer rubrics, misses, and the next-practice list.

## Completion criterion

This track is complete only when the learner can both build and defend the
system under time pressure. Memorized framework vocabulary and unmeasured demos
do not count. Completion does not add any implied requirement to Astra's
roadmap.
