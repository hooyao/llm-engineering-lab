# D9 — File Freshness and Same-Path Serialization

Status: implementation and assistant-side deterministic verification complete;
learner-run payoff pending.

## Product failure

Two workers can read the same file bytes and then propose incompatible edits.
Ordering their worker lifetimes globally wastes parallelism, while exact
`old_string` matching alone misses an unrelated intervening change. Astra needs
to tell the stale worker to read once and decide again.

This is not a filesystem-transaction problem. Raw local files expose no portable
content-conditional replace, and Astra does not attempt ACID, rollback, or
automatic merge semantics.

## Implementation

- `FileObservationStore` — one scoped content-hash map and trusted changed-path
  set per agent/worker execution.
- `ReadFileTool` — hashes the complete exact UTF-8 file from the same stream used
  for a bounded visible range.
- `EditFileTool` / `WriteFileTool` — require a matching observation for an
  existing file, return a recoverable `Read again` result after a change, and
  advance only the executing agent's snapshot after success.
- `FileWriteCoordinator` — one process-wide canonical-path gate; same-file Astra
  writes serialize while different worker tasks remain independent.
- `Agent(access_mode="write")` — permission-classified opt-in write workers;
  `read_only` remains the default and is enforced before executor activation.
- `AgentLoopWorker` — reports canonical changed paths from scoped harness state.
- `samples/FileFreshnessDemo` — deterministic payoff without a model call.

The familiar external `Read` / `Edit` / `Write` schemas remain unchanged. Hashes
are harness-internal, and several edits from one response remain ordinary serial
calls rather than a synthetic MultiEdit transaction.

## Guarantee boundary

The hash detects content changes completed before the file tool's check. The
path gate orders only Astra file tools in this process. An IDE, linter, shell, or
arbitrary external process may still write during the final check/write window.
The existing sibling-temp rename prevents partial visibility for one admitted
write but is not CAS.

See [source-reconciliation.md](source-reconciliation.md) and
[teaching-notes.md](teaching-notes.md).

Assistant-side verification: formatter clean, 136/136 tests, zero-warning
Release build, Native AOT publish successful, and deterministic demo complete.
Regressions cover invalid UTF-8 with a BOM and releasing the write gate before
yielding a result. The demo checks its outcomes before printing `PASS`.
A real local `gpt-5.6-sol` run requested one `Agent(access_mode="write")`, showed
the outer `[y/N]` permission, created `worker-real.txt` through the worker, and
returned its completion. The file was exactly 15 UTF-8 bytes:
`WORKER_WRITE_OK`.

## Payoff — learner run required

From the parent repository root:

```powershell
dotnet run --project agent\refs\Astra\samples\FileFreshnessDemo -c Release
```

Read four outcomes in the output:

1. A and B both observe `H0`.
2. A writes `H1`; B receives `Read again` and cannot overwrite A.
3. B reads `H1` once and then succeeds.
4. One agent performs two ordered same-file edits without a redundant read.

D9 is complete only after the learner runs this artifact and confirms that this
is the intended recovery behavior.
