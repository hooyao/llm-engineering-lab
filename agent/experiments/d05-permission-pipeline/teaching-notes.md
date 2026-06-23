# D5 teaching note — why a permission decision is three-state, not a bool

> Written because the instinct when adding a permission check is `bool CanRun(call)`
> — and that instinct is wrong in a way that matters. The decision has THREE
> outcomes, and the third one (`ask`) is not a missing bool, it is a different kind
> of result: one that has to suspend and call a human. This note is why, and where
> the two distinct fail-closed points live. The code it backs is
> `src/Astra.Core/Permissions/` + the gate in `AgentLoop.RunOneToolAsync`.

## The bool trap

The obvious shape for "should this tool run?" is:

```csharp
bool CheckPermission(call);   // true = run, false = block
```

It collapses on the first real requirement: **some calls can't be decided by the
harness at all — a human has to look.** "Delete this file?" is neither a
statically-known yes nor a statically-known no; it is "go ask." A bool has no room
for that. You would have to bolt on a side channel ("return false AND set a
`needsPrompt` flag AND stash a message somewhere"), which is just a three-state
enum wearing a bool costume — and a leaky one, because callers forget the flag.

So the decision is a **three-state union**, and `ask` is a first-class member:

```csharp
PermissionDecision = Allow(updatedArgs?) | Deny(reason) | Ask(message)
```

- `Allow` — run it. Optionally with rewritten arguments (a policy that normalizes
  a path returns the normalized one here).
- `Deny` — don't run it; `reason` goes back to the LLM as the tool result so the
  model adapts instead of the turn silently vanishing.
- `Ask` — the harness can't decide; a human must. `message` is what they see.

## `Ask` is not a value — it is a suspension

The deeper reason a bool fails: `Allow` and `Deny` are *values the engine
computes and returns*. `Ask` is **not a value** — it is "I need to stop and wait
for a human, then continue." Resolving it is asynchronous and may block for
minutes. That is why:

1. The decision type carries `Ask` as a distinct case (the engine can't pre-resolve
   it — it doesn't own the UI), and
2. resolving `Ask` is a separate injected interface, `IUserConfirmation`, whose
   method is `Task<bool>` — the only genuinely async, suspending step in the
   pipeline. The engine turns an approved Ask into `Allow`, a declined one into
   `Deny`, before returning. So the engine's *public* result is always terminal
   (`Allow`/`Deny`); `Ask` lives only inside.

Crucially, this suspension must **gate the guarded tool** — approval has to happen
*before* the side effect, never after. You cannot ask "run `rm -rf`?" once it has
already run. (This is the opposite of a mid-turn user interrupt, which must NOT
block — a different mechanism entirely, deferred to the control-plane work.)

## Two fail-closed points, guarding two different gaps

"Fail-closed" got stated loosely as "unknown → deny." Building D5 made it precise:
there are **two** independent fail-closed defaults, guarding **two different
unknowns**, and they fail to *different* safe values.

| Point | The gap it guards | Fail-closed to |
|---|---|---|
| **Classify** (D2) | "I don't know what this *tool* does" | `ToolAction.Other` — the strictest *class* |
| **Rule match** (D5) | "I have no *policy* for this specific call" | `Ask` (interactive) / `Deny` (headless) |

They are not the same mechanism applied twice. The D2 one runs when a tool fails
to classify its own behavior — it defaults the *category* to the most restrictive
(`Other`), so an unclassified tool is treated as dangerous, not safe. The D5 one
runs when no rule and no class-default produces a decision — it defaults the
*outcome* to "ask a human," not "silently allow."

And the D5 default is **`Ask`, not `Deny`** — with one exception. The safe move
when policy is silent is to ask the human, not to refuse outright (refusing
everything trains users to disable permissions). The exception is **headless**:
when there is no `IUserConfirmation` wired (a service, CI, a cron job — nobody to
ask), `Ask` has nowhere to go, so it collapses to `Deny`. So the full rule is:

> no decision → ask the human; **no human** → deny.

That is more precise than "unknown → deny": *unknown-and-interactive* → ask;
*unknown-and-headless* → deny. Both are safe; neither is a silent allow.

## How the three layers compose (what actually runs per call)

```
RunOneToolAsync(call):
  Layer 1  validate    -> unknown tool? Deny.            (guards malformed/unregistered)
  Layer 2  policy       -> Allow | Deny | Ask | NoOpinion
                            NoOpinion => fall back to Ask  (fail-closed #2)
  Layer 5  if Ask:      -> IUserConfirmation? approve->Allow / decline->Deny
                            no confirmer => Deny           (headless fail-closed)
  -> terminal Allow/Deny
     Deny short-circuits: reason -> LLM tool_result + ToolDenied event; tool NEVER runs
     Allow: ExecuteAsync (with rewritten args if the policy supplied them)
```

The load-bearing property — *a denied call's `ExecuteAsync` is never entered* — is
proven by construction in the tests: a `SpyTool` counts its executions and the
deny test asserts the count is zero. A test that only checked the returned
decision could pass while the tool still ran; counting executions cannot.

## Why the policy is a separate injectable thing (the SDK angle)

The decision logic (`IPermissionPolicy`) is pulled out of the engine on purpose:
an SDK consumer swaps *what the decision is* (class-default "Y", always-ask "X", or
their own remote policy) without touching *how an ask is resolved*
(`IUserConfirmation`) or the orchestration (`IPermissionEngine`). Three small
single-responsibility seams instead of one `bool CheckPermission`. The policy
interface returns `ValueTask` — async so a custom policy can do I/O without a
sync-over-async deadlock, `ValueTask` because the default policy decides from
in-memory rules and completes synchronously on the per-call hot path.

## One-line takeaway

A permission decision is three-state because `ask` is not a missing bool but a
suspension that must call a human *before* the side effect; and "fail-closed" is
two defaults for two unknowns — an unclassified tool defaults to the strictest
*class*, an unpoliced call defaults to *ask a human* (or *deny* when there is no
human).
