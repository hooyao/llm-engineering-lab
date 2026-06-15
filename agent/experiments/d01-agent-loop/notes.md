# D1 — The agent loop (`AgentLoop.cs`)

Track D Day 1. Goal: internalize *the dumb loop* — intelligence lives in the
model; the harness is just a `while(true)`. The deliverable is a test that
asserts the loop terminates on `end_turn`.

Status: Astra already had `AgentLoop.SubmitAsync()` from a prior session (commit
`8b080d8 Rename to Astra and implement core agent loop`). So D1 is *reading the
existing core to ground truth*, fixing one real gap, and adding the missing test
— not writing the loop from scratch.

## Mental model

`AgentLoop.SubmitAsync()` is the whole agent. It does not "think". One pass:

```
send transcript + tool ads → get model response
  has tool_use blocks?  yes → execute tools, append results, loop
                        no  → done, yield break
```

`while(true)`: **each iteration = one round-trip to the model.** One user input
may take several iterations (model calls a tool → sees result → calls another →
finally answers) or finish in one (model answers directly).

## Three pieces of per-session state (ctor)

| Field | Role | Keyword |
|---|---|---|
| `_toolMap` (`name → ITool`) | **execution**: dispatch a call by name | dispatch |
| `_aiTools` (`List<AITool>`) | **advertisement**: serialized into the request so the model knows what exists | advertisement |
| `_messages` (`List<ChatMessage>`) | the entire transcript — **the agent's only memory** | state |

`_messages` is an instance field, not a local. Second `SubmitAsync` call appends
to it → multi-turn memory is just this list growing. There is no other state.

## The single most important insight: advertise ≠ invoke

`AIFunction` (M.E.AI) does two jobs and they must not be conflated:

| Job | Which part of `AIFunction` | Needed? |
|---|---|---|
| **advertise** tool (Name/Description/JsonSchema → request) | the three properties | **yes** — legitimate provider-abstraction use |
| **invoke** tool (SDK auto-execution) | `InvokeCoreAsync` body | **no, and must not** — this is the auto-invoke we deliberately avoid |

Why MEAI at all: provider plumbing (Anthropic ↔ DeepSeek ↔ Azure wire formats)
is commodity. `IChatClient` abstracts it; `AITool` is the tool's wire
representation. That is the *only* reason `AIFunction` is used — to advertise.

Grep proof (whole `src/`): no `UseFunctionInvocation`, no `ChatClientBuilder`, no
invocation middleware. `AsIChatClient(...)` only adapts formats; it does not add
auto-invoke. Therefore `InvokeCoreAsync`'s body was **dead code** — it only
exists because `AIFunction.InvokeCoreAsync` is `protected abstract` and the class
won't compile without an override. The compiler forces a body; we don't want the
logic.

### The fix (fail-closed)

The "fine-grained control / permission" does **not** come from `InvokeCoreAsync`.
It comes from the *manual* dispatch at `AgentLoop.cs` (`tool.ExecuteAsync(...)`
inside the loop) — precisely from *not* taking the auto-invoke path. So
`InvokeCoreAsync` should throw, not fall back to executing the tool:

```csharp
protected override ValueTask<object?> InvokeCoreAsync(
    AIFunctionArguments arguments, CancellationToken cancellationToken) =>
    throw new NotSupportedException(
        "Tools are dispatched manually by AgentLoop; auto-invocation is intentionally disabled.");
```

- **fallback-to-execute = failed open**: if a middleware ever wakes this path,
  tools run silently, bypassing D3 read/write partitioning, D5 permission, D7
  compaction. Silent bypass is the worst failure mode for a permission system.
- **throw = failed closed**: same situation crashes loudly with a message that
  says exactly what's wrong. Astra's CLAUDE.md mandates fail-closed → throw wins.
- Dropped `async` (body no longer awaits) → expression-bodied `=> throw`, no
  async state machine generated. `arguments`/`ct` stay unused (override forces
  the signature). Build: 0 warnings, 0 errors.

This is *why* Claude Code and every serious framework refuse SDK auto-invoke and
hand-write the loop: the permission / partition / compaction seams only exist on
the manual path. Auto-invoke gives you none of them.

## Walk-through landmarks (file is the source of truth; line numbers drift)

- Stream loop: each chunk is **both** accumulated (`updates.Add`, to reassemble a
  complete response for tool detection) **and** yielded immediately
  (`yield TextDelta`, why the CLI prints token-by-token). = *stream reassembly*.
- `_messages.AddMessages(response)` — easy to miss, must not omit. Appends the
  model's own reply (incl. tool_use) back to the transcript, or the next
  iteration goes blind to what it just said/called.
- `CallId` is the pairing key: a tool result must carry the **same** id the model
  issued, so it can align results to requests when several run in one turn.
- Tool errors are **fed back to the model as a string result**, not thrown out of
  the loop (except `OperationCanceledException`, which rethrows). The model sees
  "it failed" next iteration and can recover. Unknown tool name → same path.
- Tools run **serially** here (`foreach` + `await`). D3 adds read-parallel /
  write-serial partitioning. Not done yet.

## The real gap D1 exposes: stop condition vs `stop_reason`

> CORRECTION (after reading the Claude Code source — see "Source check" below).
> An earlier draft of this note said "the loop should end on `stop_reason`
> instead of on the presence of tool calls." **That is wrong.** The source
> proves the opposite: do NOT trust `stop_reason` as the loop-exit signal.

The loop ends on **"this turn produced no tool_use block"**. That is the
**correct** exit signal, and it is exactly what Claude Code does too. What is
*not* handled is a different `stop_reason` case: `max_tokens` truncation.

Two separate things, do not conflate them:

1. **Loop continue/exit** — decided by *"did a tool_use block appear this
   turn?"*, NOT by `stop_reason`. Astra's `toolCalls.Count == 0 → yield break`
   is right. `stop_reason == 'tool_use'` is unreliable (Claude Code's own
   comment, `query.ts:554`) — never use it as the exit signal.
2. **`max_tokens` truncation** — `FinishReason == Length` (hit `MaxOutputTokens`,
   configured 10_000): no tool call **and** not a clean `end_turn`, the model was
   cut off mid-sentence. Current Astra treats this as "done" and silently
   `yield break`s, returning a half-answer as if complete. Claude Code instead
   *recovers*: appends a "resume directly, break work into smaller pieces" user
   message and retries, up to 3× (`MAX_OUTPUT_TOKENS_RECOVERY_LIMIT`,
   `query.ts:164` and `:1188-1256`). **This recovery is the real missing piece**,
   not "switch the exit signal to stop_reason."

D1 term to internalize: `stop_reason` (`end_turn` vs `tool_use` vs `max_tokens`)
— the first two are the same exit decision (presence of tool_use), the third is
a recovery case Astra has not implemented yet.

## Source check — does Claude Code's loop match this mental model? (yes)

Read `refs/claude-code-sourcemap/restored-src/src/query.ts` (the `while(true)` at
`:307`). The skeleton is identical to Astra's `AgentLoop.cs`:

| Step | query.ts | AgentLoop.cs |
|---|---|---|
| `while(true)` | `:307` | `:29` |
| stream model | `:659` `deps.callModel(...)` | `:35` `GetStreamingResponseAsync` |
| detect tool_use | `:829-835` → `needsFollowUp = true` | `:51` `OfType<FunctionCallContent>()` |
| run tools locally | `:1382` `runTools(...)` | `:71` `tool.ExecuteAsync(...)` |
| feed results back | `:1395` `toolResults.push(...)` | `:93` `_messages.Add(Tool, ...)` |
| text only → done | `:1062` `if (!needsFollowUp)` → `:1357` `return {reason:'completed'}` | `:55` `if (toolCalls.Count==0) yield break` |

Key source facts:

- **`:553-557`** — explicit comment: `stop_reason === 'tool_use'` is unreliable,
  "not always set correctly." The sole loop-exit signal is whether a tool_use
  block arrived during streaming. Confirms point 1 above.
- The skeleton (model → tool? → run → feed back → text=done) is ~50 lines; the
  file is 1730. The other ~97% is production survival the skeleton does NOT
  change, layered *around* it:
  1. compaction every iteration BEFORE the request (`:365-543`) — Track D D7.
  2. token-budget gate → `return {reason:'blocking_limit'}` (`:628-648`).
  3. model fallback on overload (`:894-951`).
  4. **multiple exit reasons** — `completed` / `blocking_limit` / `model_error` /
     `aborted_streaming` / `prompt_too_long` / `stop_hook_prevented` /
     `image_error`. Astra collapses all of these into one implicit `yield break`.
  5. stop hooks (`:1267`) can VETO the end and pull the loop back (`:1282`
     blockingErrors → `continue`) — "looks done, but a hook says keep going."
- **Dangling tool_result on error** (`:984` `yieldMissingToolResultBlocks`): if
  the loop throws after a tool_use was emitted but before its tool_result, it
  back-fills an `is_error: true` result for each orphan — else the next API call
  400s on an unpaired tool_use. Astra's `:74-78` converts tool *exceptions* to
  string results (good) but does NOT handle the throw-mid-loop orphan case.

Takeaway: the mental model is 100% right — it IS a dumb loop, and Claude Code
uses the same tool_use-presence exit signal, not stop_reason. Every difference is
*outside* the skeleton and is what D5/D6/D7 add back.

## Token-cost note (the quantify directive)

Every iteration re-sends the **entire** `_messages`. An N-iteration session sends
the transcript N times, each longer than the last → ~O(n²) token cost in turns.
This is why D6 (prompt cache, stop re-billing the static prefix) and D7
(compaction) are survival mechanisms, not optimizations. This version has
neither → long sessions get linearly more expensive until `prompt_too_long`.

## What's left to finish D1

1. **The test** (the actual deliverable): a test project with a fake
   `IChatClient` (no live endpoint in this environment) asserting:
   - model returns text only → loop terminates after 1 iteration;
   - model returns tool_use then end_turn → loop runs 2 iterations then stops.
2. Decide whether to make `stop_reason` explicit now or note it as a D-later TODO
   (it's arguably a D4 "API layer" concern; D1 only needs to *see* the gap).

## Done this session

- `InvokeCoreAsync` → fail-closed `throw` (advertise-only adapter). Build clean.
- This note.
