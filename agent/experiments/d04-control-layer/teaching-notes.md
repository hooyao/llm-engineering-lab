# D4 teaching note — "stop" must kill the tree, not stop watching

> Written because the D4 hole turned on two facts the curriculum assumed but did
> not state: (1) `Process.Dispose()` sends no signal, so cancelling a tool
> *orphans* its child instead of stopping it; and (2) the child is never the real
> work — it is a shell wrapper, so only an *entire-tree* kill stops anything. The
> code this backs is `BashTool.ExecuteAsync`'s `try/finally` + `KillTreeAsync` in
> the Astra submodule.

## The hole, precisely

D2/D3 left `BashTool` spawning a shell and streaming its output, with
cancellation wired only as far as *reading*:

```csharp
using var process = new Process { ... };
process.Start();
await foreach (var line in channel.Reader.ReadAllAsync(ct))   // ct cancels HERE
    yield return new ToolOutput.Progress(line);
await process.WaitForExitAsync(ct);                            // or HERE
```

When `ct` fires, `OperationCanceledException` unwinds out of the method and the
`using` runs `process.Dispose()`. The intuition "Dispose cleans up the process"
is wrong, and that wrongness is the whole bug.

## Fact 1 — `Process.Dispose()` sends no signal

`Process` is a *managed handle* to an OS process, not the process itself.
`Dispose()` releases the handle (and the internal wait registration); it does
**not** call `kill`, send `SIGTERM`, or close the child's stdin. The OS process
keeps running with no one holding its handle.

What happens to a running child whose parent stops tracking it:

- it is **not** killed;
- when the parent process exits (or just stops waiting), the child is
  **reparented** — on Linux it is re-attached to `init`/`systemd` (PID 1), or a
  subreaper; on Windows the parent/child link is already loose. The child is now
  an **orphan**, still consuming CPU/IO.

So the pre-D4 behavior of "stop" was literally: *Astra stops watching; the
command keeps running.* `ct` cancelled "do I keep reading its output", never the
output's producer. The fix has to send a real kill on the cancellation path.

## Fact 2 — the child is a shell; the work is in its descendants

`BashTool` never execs the user's command directly. It runs:

```
/bin/sh -c "npm install"        (POSIX)
cmd.exe  /c "npm install"       (Windows)
```

The process Astra spawns is the **shell**. The thing that does work — `npm`,
which spawns `node`, which spawns more — is the shell's **child**, i.e. Astra's
**grandchild and beyond**. The process tree at runtime:

```
dotnet (Astra)
└── sh -c "npm install"          ← the only process a bare Kill() targets
    └── npm
        └── node
            └── (build subprocesses…)   ← where the CPU/time actually goes
```

`process.Kill()` signals **only the direct child** — the shell. Kill just the
shell and `npm` is orphaned and **keeps installing**. This is why, for a
shell-wrapping tool, `Kill(entireProcessTree: true)` is not "the more thorough
option" — it is the **only correct one**. The thing you care about is never at
the root of the tree you hold.

`Process.Kill(bool entireProcessTree)` walks the child/descendant set and
signals all of them (`SIGKILL` on POSIX, `TerminateProcess` on Windows). That is
the `kill -9 -<pgid>` you would otherwise write by hand.

## Where the kill goes — `finally`, not `ct.Register`

Two candidate sites; only one has a clean lifetime.

**`ct.Register(() => process.Kill(true))`** — register a cancellation callback.
The trap is *lifetime*: the callback is owned by the `CancellationTokenSource`,
which can outlive the `Process`. If the `using` disposes `process` first and the
token is cancelled afterward, the callback runs `Kill` on a **disposed** Process
and throws `ObjectDisposedException` on a thread-pool thread with no one to catch
it. Making it safe means also managing the registration's lifetime
(`using var reg = ct.Register(...)`) — more moving parts, racing the cancel
thread against disposal.

**`try { drain + WaitForExit } finally { await KillTreeAsync(process); }`** — one
linear control flow. The `finally` runs exactly once on every exit path:

- **normal completion** — the loop drained, the process already exited;
  `KillTreeAsync` sees `HasExited == true` and no-ops (no false kill);
- **cancellation** — `ReadAllAsync(ct)` / `WaitForExitAsync(ct)` throw OCE; the
  `finally` runs before the exception leaves the method, killing the live tree;
- **consumer break** — and this is the subtle one: `await foreach` over an async
  iterator calls the iterator's **`DisposeAsync()`** when the consumer breaks or
  faults, and a `finally` inside the iterator body **runs during that
  `DisposeAsync`**. So even if `AgentLoop` simply stops pulling events, the kill
  still fires. Same single thread as the iterator, no race with the canceller.

### The CS1626 footnote

D2/D3 hit CS1626 ("cannot yield in the body of a try block with a catch"). It
bans only `try`/**catch**. `try`/`finally` with **no** catch is legal around
`yield return`, which is exactly what this needs — the kill is cleanup, not error
handling, so it belongs in `finally` and the constraint never bites.

## The two races `KillTreeAsync` must swallow

```csharp
private static async Task KillTreeAsync(Process process)
{
    if (process.HasExited) return;                 // normal path: nothing to kill
    try { process.Kill(entireProcessTree: true); }
    catch (InvalidOperationException) { return; }  // race 1
    await process.WaitForExitAsync(CancellationToken.None);  // race 2 note
}
```

1. **TOCTOU between `HasExited` and `Kill`.** The process can exit in the window
   after the check passes and before `Kill` runs; `Kill` then throws
   `InvalidOperationException` ("process has exited"). That is success, not
   failure — the goal (process gone) already holds — so it is caught and ignored.
2. **Reap with `CancellationToken.None`, deliberately.** We are usually here
   *because* the caller's `ct` was cancelled. If we passed that cancelled token to
   the reaping `WaitForExitAsync`, it would throw immediately and we would return
   **before the tree finished tearing down** — handing control back while children
   are still dying. `None` forces us to wait for the kill to actually complete.
   `Kill` only *requests* termination; without the wait, "stopped" is still a lie,
   just a shorter one.

## Proving it — by construction, not by clock

A timing test ("cancel, assert it returned fast") proves the *read* stopped, not
that the *process* died — the exact bug we are fixing would pass it. The D4 test
instead makes a **grandchild** observable and checks it actually stopped:

```
( while true; do echo tick >> MARKER; sleep 0.1; done ) &   # grandchild subshell
echo started; wait                                          # shell blocks alive
```

The shell backgrounds a subshell that appends to a marker file every 100 ms. On
cancel, `KillTreeAsync` must take the **whole tree**. Then: sample the marker line
count right after the kill, wait ~800 ms, sample again. If the tree died the count
is unchanged; if only the shell died, the reparented grandchild keeps ticking and
the counts differ → test fails. This cannot pass without a genuine tree kill.

Platform note: the construction is POSIX `sh` semantics. On Windows `BashTool`
uses `cmd.exe` and the test early-returns; the guarantee is verified on
Linux/macOS. (xunit 2.9 has no runtime `Assert.Skip`, hence the early return.)

## One-line takeaway

Cancelling a shell tool must **kill the process tree and reap it**, because
`Dispose()` only drops the handle (orphaning the child) and the child is a shell
whose descendants hold the real work — so `Kill(entireProcessTree: true)` in a
`finally`, guarded against the exit-race and reaped with an uncancelled token, is
the smallest correct "stop".
