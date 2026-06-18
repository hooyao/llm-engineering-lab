# D4 — Control layer: real cancellation (process-tree kill on stop)

Track D Day 4. The curriculum's nominal D4 ("streaming + stream reassembly +
tool_use detection") was **pulled forward into D2** (see d02 notes / progress.md),
so the real work left for D4 is the **control layer** — the three interruption
scenarios d02 parked under "OPEN for D4". This day does the foundational piece of
that: making "stop" actually stop the work, not just stop watching it.

## What this day is (and is not)

d02 split the control-layer work along two orthogonal axes:

| Axis | What | This day |
|---|---|---|
| **Mechanism** (really kill) | cancel kills the child process **tree**, then reaps | **DONE** |
| **Control plane** (signal in) | back-channel to inject input / a policy that decides to stop | **deferred** |

The three d02 scenarios map onto these:

- **Scenario 2** (user says "stop" mid-tool) = cancel + mechanism. The cancel path
  already existed; D4 adds the missing **real kill**. **Closed this day.**
- **Scenario 3** (agent autonomously kills on error) = a policy watching the
  Progress stream + cancel + mechanism. The *mechanism* half is shared with
  scenario 2 and is now done; the *policy* half (who decides) is deferred.
- **Scenario 1** (inject user input mid-turn) = pure control-plane (a back-channel
  + soft/hard restart). Deferred.

Scope call (consistent with Astra's own CLAUDE.md "no abstraction before concrete
code forces it"): build the mechanism — a real correctness hole with a concrete
fix and a by-construction test — and defer the control-plane abstractions until a
consumer needs them. The deferred pieces are listed under "Open / deferred".

## The hole that was closed

Pre-D4, `BashTool` wired `ct` only as far as *reading* the child's output. On
cancel, `OperationCanceledException` unwound and `using var process` ran
`Dispose()` — which releases the handle and **sends no signal**. The spawned
`sh -c "…"` (and its `npm`/`node`/… descendants) were **reparented to init and
kept running**. "Stop" meant "I stop watching", not "it stopped". Full derivation
in `teaching-notes.md`.

## What was built

`BashTool.ExecuteAsync` (Astra submodule) — the drain + `WaitForExit` is now
wrapped in `try { … } finally { await KillTreeAsync(process); }`, plus a new
private `KillTreeAsync`:

- **`finally`, not `ct.Register`.** One linear control flow that fires on every
  exit path — normal completion (no-ops, process already exited), cancellation
  (OCE), and consumer break (`await foreach` calls the iterator's `DisposeAsync`,
  which runs the `finally`). `ct.Register` would risk `Kill` on a disposed Process
  and need its own registration lifetime. `try/finally` with **no catch** is legal
  around `yield` (CS1626 bans only catch).
- **`Kill(entireProcessTree: true)`, not `Kill()`.** The spawned process is the
  *shell*; the work is in its descendants. A bare `Kill()` of the shell orphans
  `npm`. Tree kill is the only correct option for a shell-wrapping tool.
- **Two races swallowed.** (1) TOCTOU between `HasExited` and `Kill` →
  `InvalidOperationException` caught and ignored (process already gone = success).
  (2) Reap with `WaitForExitAsync(CancellationToken.None)` — the caller's token is
  typically already cancelled; passing it would return before the tree finished
  dying. `None` forces the wait so "stopped" is true on return.

## Tests (+2)

`tests/Astra.Core.Tests/BashToolCancellationTests.cs`:

- **`ExecuteAsync_Cancelled_KillsWholeTree_GrandchildStopsTicking`** — the
  load-bearing, by-construction proof. A shell backgrounds a **grandchild**
  subshell that appends a tick to a marker file every 100 ms; on cancel the marker
  must stop growing. A bare-`Kill()` implementation (no tree) leaves the
  reparented grandchild ticking → counts differ → fail. A timing test could not
  catch the original bug (it proves the *read* stopped, not the *process*); this
  one cannot pass without a genuine tree kill. **POSIX-only** (sh subshell
  semantics); on Windows BashTool uses cmd.exe so it early-returns (xunit 2.9 has
  no runtime `Assert.Skip`).
- **`ExecuteAsync_Cancelled_ThrowsPromptly_NotAfterFullRuntime`** — cross-platform
  guard: a 30 s command, cancelled, throws OCE in well under 10 s (not after the
  full runtime), proving the cancel path is wired and the `finally`'s reap does
  not deadlock.

Local result: **38/38 pass on Windows** (36 prior + 2). On Windows the tree-kill
test early-returns, so its core assertion is **not** exercised here — see the gate
below.

## >>> VERIFICATION GATE (must run before D4 is "verified")

The tree-kill assertion only executes on Linux/macOS. This session's machine had
no reachable Linux target (GX10 `192.168.1.200` SSH timed out; no WSL distro; no
Docker daemon), so the core guarantee is implemented and reviewed but **not yet
executed green on a POSIX box**. Before claiming D4 verified:

```bash
# On the GX10 (aarch64 Linux) once reachable, in the Astra checkout:
dotnet test Astra.slnx --filter "FullyQualifiedName~BashToolCancellation"
# Expect: the GrandchildStopsTicking test runs (not early-return) and passes.
```

Then record the green result in progress.md and drop this gate. Until then, D4 is
"implemented, Windows-green, Linux-pending".

## Open / deferred (the rest of the control layer)

- **Scenario 1 — inject user input mid-turn.** Needs the one-directional loop to
  gain a back-channel (a `Channel`/queue alongside `ct`), and a choice of *soft*
  (queue input, apply as next turn's user message) vs *hard* (cancel current
  stream/tool on Enter, re-compose `_messages` with the partial + new input,
  restart — the Esc-then-type UX). Also requires splitting `AgentApp`'s serial
  `await foreach`→`ReadLine` into a concurrent input-listener. Deferred until the
  CLI needs interactive interruption.
- **Scenario 3 policy half.** A middleware that subscribes to the Progress stream
  live, matches failure markers (error/FAILED/panic), and triggers cancel+kill.
  Its home is the loop's per-tool section where Progress events are in hand. The
  *mechanism* it would call (tree kill) is now done; only the *decider* is
  missing. Deferred until there is a concrete policy to enforce.
- **`contextModifier` (cd-changes-cwd).** Still unbuilt (noted since D3); applies
  only on the serial path when a stateful tool appears.

## Git state

To be committed: branch `track-d/d4-control-layer` off Astra main (`9ac91aa`),
modifying `src/Astra.Core/BashTool.cs`, adding
`tests/Astra.Core.Tests/BashToolCancellationTests.cs`; then PR, squash-merge, bump
the parent-repo submodule pointer — same flow as D1/D2/D3. CRLF note: Windows-
edited tracked files need `git add --renormalize` before commit (see progress.md
env notes); the merge needs `gh pr merge N --squash --admin`.
