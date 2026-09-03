# Track D Phase I Recap — Agent Core (D1-D8)

Phase D-I produced a working C# agent core in Astra and, more importantly, a
set of measured correctness boundaries. The implementation is not a collection
of framework APIs: every day closed one concrete failure in iterative agent
execution.

## What exists

| Day | Runtime capability | Load-bearing invariant |
|---|---|---|
| D1 | Iterative `AgentLoop` over `IChatClient` streaming | A tool result returns to the model; the loop ends only when no tool call remains |
| D2 | Immutable `ToolDefinition` + invocation-time `IToolExecutor` | Classification depends on arguments; unknown behavior fails closed; unused/denied executors are never constructed |
| D3 | Stable tool-call batching | Adjacent reads may overlap; write/execute/unknown calls are serial barriers; results return in model emission order |
| D4 | Process and cancellation control | Cancellation kills and reaps the complete process tree rather than merely stopping output consumption |
| D5 | Permission pipeline | Lookup/classification/permission precede executor activation and every side effect |
| D6 | Three-lifetime context assembly | Static/session context remains byte-stable; per-turn attachments are bounded by one deadline |
| D7 | Micro/full compaction with explicit result states | Only a complete `Applied` candidate replaces history; failure leaves original history authoritative |
| D8 | Isolated worker coordination | Every worker owns a DI scope, model client, loop, history, telemetry, and cancellation lifetime; the coordinator receives only bounded reports |

The post-D7 coding specialization also has bounded `Read`, `Glob`, and `Grep`,
atomic `Write`/exact `Edit`, a dedicated PowerShell tool, optional multi-root
file boundaries, and interactive permission for side effects.

## Architectural conclusions

### The model proposes; the runtime owns authority

The model may choose a tool and arguments. Astra still owns lookup,
classification, permission, ordering, path policy, cancellation, result
bounding, and durable runtime facts. A model-authored worker report cannot
claim trusted usage, identity, status, or file changes; those come from the
harness envelope.

### Context is an execution resource

System/session/turn lifetimes, deterministic serialization, prompt-cache
prefixes, tool-result size, compaction thresholds, and recent-tail retention
all affect correctness, latency, and cost. Compaction is transactional because
an incomplete summary cannot safely replace the only history copy.

### Lifetime ownership is explicit

The coordinator owns scheduling, not a shared executable worker. Each admitted
worker receives an independent async DI scope and a durable
`Task<WorkerCompletion>`. Disposal performs cancel, join, and then scope
release. Queued workers allocate no model client or loop.

### Recoverable tool failure is typed

A tool exception produces operator-visible `ToolFailure`, a model-visible error
`ToolResult`, and another model round. Terminal `Error` is a separate event.
The regression sequence is:

```text
ToolUse -> ToolFailure -> ToolResult -> TextDelta
```

This distinction was discovered by the learner payoff: the first demo consumer
incorrectly aborted on a recoverable missing-file read.

## Verification evidence

- 113/113 deterministic tests pass.
- Formatter, zero-warning Release build, and Native AOT publish pass.
- Real local `gpt-5.6-sol` runs exercised compaction, file tools, permissions,
  DI-scoped workers, completion batching, and context isolation.
- Large-file editing preserved 29,999 LF terminators while replacing the unique
  target in a 30,000-line file.
- D7 deterministic and real-provider compaction retained the required
  `RETENTION-CODE-7429` fact.
- D8 worker reports never contained the coordinator-only isolation marker.

## Learner payoff: when multi-agent hurts

The learner ran the same repository audit with one agent and with a coordinator
plus two workers:

| Path | Tokens | Wall time | Model calls | Tool calls |
|---|---:|---:|---:|---:|
| Single agent | 33,130 | 27.8 s | 4 | 12 |
| Coordinator + workers | 88,778 | 41.0 s | 13 | 28 |

The workers achieved real overlap: 28.5 seconds maximum duration versus a 53.3
second sum, equivalent to 1.87x worker-phase speedup. End to end, however, the
multi-agent path consumed 2.68x tokens and was 1.47x slower. The task was too
narrow: duplicated investigation plus dispatch and synthesis cost exceeded the
parallel work saved.

The decision rule is evidence-based: use isolated workers for independent
breadth or slow external work only when the expected parallel work dominates
startup, duplicated context, and synthesis. Multi-agent is not a default agent
architecture.

## Product boundary after Phase I

Astra's north star is a Manus-style general autonomous-agent core. Coding is
the first specialization and Claude Code/Codex are measured coding benchmarks.
Feature count is not the goal. Generic workflows, application RAG, intent
taxonomies, model training, and SaaS control planes remain external systems and
interview labs unless an independent Astra failure passes the feature-admission
gate.

## Open correctness boundary and next step

Write-capable workers remain intentionally unavailable. A global writer lane
prevents simultaneous writes but cannot prove that a worker reasoned from the
current file version. D9 therefore implements strict file-version conflicts and
atomic same-response MultiEdit:

- no automatic rebase of stale model reasoning;
- all eligible same-file edits in one response commit or fail together;
- a later edit requires a new model-visible `Read`;
- permission and file-version validation remain separate gates.

Phase I is complete. D9 begins only from this demonstrated stale-write failure,
not from a generic workflow or interview feature list.
