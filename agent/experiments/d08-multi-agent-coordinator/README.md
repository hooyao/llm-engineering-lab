# D8 — Multi-Agent Coordinator

Status: implementation and assistant-side verification complete; learner-run
payoff pending.

Current decision: D8 v1 uses strict file-version conflicts and never
automatically rebases a stale worker edit. Eligible adjacent same-file edits in
one assistant response become one internal all-or-nothing transaction; later
edits require another `Read`. Workers return only a bounded condensed report to
the coordinator; their full private histories never enter coordinator context.

## Implementation

- `src/Astra.Core/Coordination/WorkerContracts.cs` — trusted completion envelope
  and bounded model-authored report.
- `DependencyInjectionWorkerSessionFactory` — creates one async scope per
  admitted worker and resolves its scoped runtime graph.
- `AgentLoopWorker` — runs the scope-owned `AgentLoop`, parses the
  source-generated JSON report, and reads scope-owned provider telemetry.
- `WorkerCoordinator` — owns the coordinator-session registry, targeted
  cancellation, bounded worker slots, completion fan-in, batching, and a global
  single-writer lane; it never shares a worker instance.
- Tool activation is also scope-correct: `AgentLoop` advertises immutable
  `ToolDefinition` metadata, then resolves one keyed transient `IToolExecutor`
  only after an invocation passes classification and permission.
- CLI configuration uses strongly typed `IOptions<T>` with constructor
  injection. Compaction enablement is an explicit no-op policy rather than an
  optional service lookup, and `Program.cs` contains no section-key access or
  manually assembled options object.
- `AgentTool` — starts clean-context read-only workers and returns immediately.
- `WorkerCompletionXml` — emits escaped user-role task notifications.
- `Astra.Cli` — registers `Agent`, waits outside the main loop for the active
  worker group, batches completions, and performs one synthesis turn.

Write-capable workers are intentionally not exposed through the CLI yet. The
coordinator's write lane is implemented and tested, but the chosen strict
file-version/MultiEdit transaction must exist before an LLM can use that lane.

Verification: 112/112 tests, formatter clean, zero-warning Release build, and
Native AOT publish successful.

## Notes

- [Coordination correctness boundaries](teaching-notes.md)
- [Measured payoff](results.md)

## Payoff — learner run required

```powershell
dotnet run --project agent\refs\Astra\samples\MultiAgentDemo -c Release -- --real --root agent\refs\Astra
```

The demo runs the same repository audit once with a single agent and once with
a coordinator plus two isolated workers. It prints both answers, wall time,
provider-reported tokens, model/tool calls, token multiple, and an isolation
sentinel check. D8 is not complete until the learner runs this and interprets
the comparison.
