# D2 — Tool contract + behavioral permission flags

> Post-D8 update (2026-08-31): the behavioral-classification decision below is
> retained, but metadata/classification now lives in immutable
> `ToolDefinition`; execution lives in `IToolExecutor`, activated as a keyed
> transient only after permission. The remainder records the original D2 design
> dialogue and should be read as historical derivation.

Track D Day 2. Goal: replace the loop's "any tool just runs" with a **permission
contract** on the tool itself, so D5 (the permission engine) has typed seams to
attach to. D1 gave us the manual-dispatch path (`tool.ExecuteAsync` in the loop,
not SDK auto-invoke) precisely so this contract has somewhere to live.

Status: design exploration. No code committed yet — the user wants to pick the
shape first (tutor mode). This note captures (a) how four real agents model tool
permissions, (b) the design axes that fall out, (c) the user's stated direction,
(d) the three decisions still open.

## Current Astra `ITool` (the D1 version — what we're extending)

```csharp
public interface ITool
{
    string Name { get; }
    string Description { get; }
    JsonElement InputSchema { get; }
    ValueTask<string> ExecuteAsync(IDictionary<string, object?>? arguments, CancellationToken ct);
}
```

No permission information at all. Every advertised tool is equally runnable. The
loop dispatches by name and executes — nothing classifies *what* the call does.

## The crude starting point (Claude Code's three bools)

Claude Code's `Tool.ts` (source read, **verified**, `:402-406`):

```ts
isConcurrencySafe(input): boolean   // can run in parallel with others?
isReadOnly(input): boolean          // no side effects on the world?
isDestructive?(input): boolean      // irreversible? (optional)
```

- All three are **input-dependent** — `bash` is read-only for `ls`, not for
  `rm -rf`. This is the key idea worth keeping: *the same tool has different
  permission depending on its arguments.* Behavioral flags over class hierarchy.
- `TOOL_DEFAULTS` (`:757-769`) are **fail-closed**: a tool that doesn't override
  is treated as not-readonly, not-concurrency-safe. `buildTool` factory
  (`:783-792`) spreads `{...TOOL_DEFAULTS, ...def}` so a tool only states its
  deviations. (The factory exists only because TS has no default interface
  methods — C# does; see Q1.)

User's critique (correct): three orthogonal bools is **crude**. It can't say
"this `bash` invocation is an *execute*, that one is a *write*" as a single
classification; it forces every caller to AND/OR three predicates. The user
wants a **category** per call (read / write / execute / other), each mapping to a
permission level.

## Four real permission models (the comparison the user asked for)

> Sourcing: **CC = verified** (read `Tool.ts` source). **OpenCode = captured**
> last session from opencode.ai/docs/permissions. **Codex / LangGraph =
> unverified this session** (training knowledge; web search MCPs not mounted,
> WebFetch unreliable for these two). Re-verify before quoting as fact.

### 1. Claude Code — input-dependent boolean flags + `canUseTool` callback

- Decision data lives **on the tool** (`isReadOnly(input)` etc.).
- The actual allow/deny/ask decision is a **callback** (`canUseTool`) the harness
  calls before executing; it returns `{behavior: 'allow' | 'deny' | 'ask', ...}`.
- So: tool *describes* itself (flags), host *decides* (callback). Two layers.

### 2. Codex CLI — two orthogonal axes: approval policy × OS sandbox  [unverified]

- `approval_policy`: `untrusted` / `on-failure` / `on-request` / `never` — *when
  to ask the human*.
- `sandbox_mode`: `read-only` / `workspace-write` / `danger-full-access` — *what
  the process is mechanically allowed to touch*, enforced by the **OS**
  (Landlock+seccomp on Linux, Seatbelt on macOS). `workspace-write` = write cwd +
  $TMPDIR, network off by default.
- UI presets ("Read Only" / "Auto" / "Full Access") are just **named points** in
  the (approval × sandbox) grid.
- **Key idea CC/OpenCode lack: defense in depth.** Even if the policy says
  "don't ask," the sandbox still mechanically blocks out-of-workspace writes. The
  permission flag is advisory; the sandbox is enforcement. Two *independent*
  mechanisms, not one.

### 3. OpenCode — per-tool keys, values allow/ask/deny, bash by command pattern  [captured]

- Values: `allow` / `ask` / `deny`.
- Keyed per tool: `read`, `edit`, `bash`, `glob`, `grep`, `task`, `skill`,
  `webfetch`, ...
- `bash` is special: **pattern-keyed rules**, e.g. `{"git *": "allow", "rm *":
  "deny"}`, **last matching rule wins**.
- Defaults: mostly `allow`; `doom_loop` / `external_directory` = `ask`; `.env` =
  `deny`. Fail-*open* by default except for a deny-list of dangerous cases — the
  opposite of CC's fail-closed. (Trade-off: usability vs safety. Astra's CLAUDE.md
  mandates fail-closed, so we follow CC here, not OpenCode.)

### 4. LangGraph — runtime interrupt, not a static per-tool policy  [unverified]

- No permission flags on tools at all. Instead `interrupt(value)` pauses graph
  execution at *any* node (incl. before a tool call), the checkpointer persists
  state, and a human resumes with `Command(resume=...)`.
- Four HITL shapes: approve/reject, edit-state, review-tool-call, multi-turn.
- **Key idea: the decision point is dynamic + resumable**, decoupled from the
  tool definition. Powerful for human-in-the-loop, but it's an *orchestration*
  feature (pause/resume the whole graph), not a *tool contract*. Different layer
  than what D2 is building.

## Design axes that fall out of the comparison

| Axis | CC | Codex | OpenCode | LangGraph |
|---|---|---|---|---|
| Where the info lives | on tool (flags) | config (2 axes) | config (per-tool) | in graph (interrupt) |
| Input-dependent? | **yes** | sandbox no / approval no | **yes** (bash pattern) | yes (human sees it) |
| Decision granularity | predicate per concern | (approval × sandbox) grid | allow/ask/deny per tool | per-node pause |
| Default posture | **fail-closed** | read-only sandbox | fail-open + deny-list | explicit interrupt |
| Enforcement layer | app (`canUseTool`) | **OS sandbox** + app | app (pattern match) | orchestration |
| Decision timing | before exec | before exec | before exec | any node, resumable |

Three takeaways for Astra's contract:

1. **Keep input-dependence** (CC, OpenCode-bash). The same tool, different
   permission by argument, is the whole point — and what the user wants as a
   *category* rather than three bools.
2. **Separate "what the call is" from "what we do about it"** (CC's
   flags-vs-callback split; Codex's sandbox-vs-approval split). D2 = the tool
   *classifies* its call. D5 = the engine *decides* allow/ask/deny. Don't fuse
   them — keep the seam.
3. **Fail-closed by default** (CC, not OpenCode). Unknown / unclassified ⇒ the
   most restrictive bucket. Matches Astra's CLAUDE.md.

(Codex's OS-sandbox enforcement and LangGraph's resumable interrupt are real and
better-in-depth, but they're *later* layers — D5/D6 territory, possibly a
"sandbox the subprocess" task of its own. D2 is just the contract.)

## User's direction (decided)

> "same bash tool should have read / write / execute / other, each mapping to a
> different input, mapping to a different permission. isReadOnly /
> isConcurrencySafe / isDestructive is crude. Use C# default interface methods
> with the safest defaults."

So: a **category** the tool computes from its input, fail-closed default, via
default interface methods (not a base class — D1 already established we dislike
inheritance for tools; flags-on-behavior beat type hierarchy).

Likely shape (NOT final — pending the Q1-Q3 answers below):

```csharp
public enum ToolAction { Read, Write, Execute, Other }   // Other = safest bucket

public interface ITool
{
    // ... existing Name / Description / InputSchema / ExecuteAsync ...

    // Default interface method, fail-closed: a tool that doesn't classify its
    // input is treated as the most dangerous category by the D5 engine.
    ToolAction Classify(IDictionary<string, object?>? arguments) => ToolAction.Other;
}
```

`Other` (or a dedicated `Unknown`) as the default = fail-closed: an unclassified
call lands in the bucket the permission engine treats most strictly. Maps cleanly
onto Codex's tiers (read-only / workspace-write / full) and subsumes CC's three
bools (Read ≈ isReadOnly; Write/Execute ≈ not-readonly + maybe destructive).

## Open decisions — user to pick (tutor mode: user defines, I implement)

**Q1 — default interface method vs abstract base class?**
C# 8+ allows a body directly in the interface (`ToolAction Classify(...) =>
ToolAction.Other;`). Alternative: a `ToolBase` with a `virtual` default. D1 we
rejected inheritance for tools. But: a default interface method's default can be
*silently forgotten* (a tool author never overrides → everything is `Other` →
over-restrictive but safe; the inverse, a too-permissive default, would be
fail-open and dangerous). Pick one and say why.

**Q2 — does `Classify` take the input?**
Curriculum wants the demo `Classify("ls") == Read` vs `Classify("rm -rf") ==
Execute/Write`. That *requires* taking the input. But Astra's input is the
weakly-typed `IDictionary<string, object?>?` bag — pulling `arguments["command"]`
out of it is ugly. Tension: input-dependence (the whole point) vs the ugliness of
the weak bag. Keep the bag, or introduce a typed input per tool?

**Q3 — which two demo tools?**
Curriculum: one read, one write. Two clean options:
- a `ReadFileTool` (always Read) + a `WriteFileTool` (always Write) — fixed
  category per tool, simplest, but doesn't *demonstrate* input-dependence.
- a single `BashTool` whose `Classify` returns Read for `ls`, Execute for
  `rm -rf` — one tool, category varies by input. **This is the one that proves
  "behavioral flags over inheritance."** Pick based on what better demonstrates
  the D2 thesis.

## What's left to finish D2

1. User answers Q1-Q3 (direction).
2. Implement the chosen contract on `ITool` + the demo tool(s).
3. Tests: `Classify` returns the right `ToolAction` per input (incl. the
   input-dependent `bash` ls-vs-rm case if Q3 picks bash); default-method
   fail-closed path returns the safest bucket when not overridden.
4. (Defer to D5) the engine that maps `ToolAction` → allow/ask/deny. D2 stops at
   the contract; do not build the decision engine yet — keep the seam.

## VERIFIED: why Claude Code's bash permission UX fails (read the source)

Read this session, all **verified** (CC source on disk):
- `tools/BashTool/bashPermissions.ts` (2622 lines)
- `utils/shell/prefix.ts`
- `utils/shell/readOnlyCommandValidation.ts` (1894 lines)

User's lived complaint: "bash args change a little → re-authorize → authorization
becomes useless → I just turn on bypass permissions." This is a **real design
failure**, and the source shows exactly why.

Decision flow (`bashToolCheckPermission`, :1050-1178): exact rules (deny/ask/allow)
→ prefix rules (deny/ask/allow) → path constraints → `BashTool.isReadOnly(input)`
auto-allow → else `passthrough` → **prompt the user**.

On approval at the prompt, CC saves a rule. *What* rule is decided by
`getSimpleCommandPrefix` (:161-188): takes only the **first two tokens**, second
must look like a subcommand (`^[a-z][a-z0-9]*(-[a-z0-9]+)*$`).
- `git commit -m "x"` → prefix `git commit` → reusable rule `Bash(git commit:*)`. Good.
- `ls -la` → 2nd token is a flag → **null** → falls back to **exact-match** rule.
- `cat file.txt` → 2nd token is a filename → **null** → **exact-match**.
- `chmod 755 f` → 2nd token is a number → **null** → **exact-match**.

An exact-match rule matches only that exact string. `cat a.txt` approved ⇒
`cat b.txt` does NOT match ⇒ re-prompt. That is "args change a little →
re-authorize", verbatim from the source.

NOT a bug — the security/usability tension made concrete:
- prefix too wide (`rm:*`, `git`, `bash:*`) → one approval covers dangerous
  variants. CC forbids these: `BARE_SHELL_PREFIXES` (bashPermissions :196-226),
  `DANGEROUS_SHELL_PREFIXES` (prefix.ts :28-44).
- prefix too narrow (exact) → re-prompt on every arg change.
- CC chose narrow (safety-first). The UX cost is what drove the user to bypass.

## The key insight: category-based keys sidestep CC's failure

CC's root cause = **the authorization key is the command STRING (exact or prefix),
not the command's BEHAVIOR.** Args are part of the string, so any arg change can
miss the key.

Astra's `Classify(args) → ToolAction` makes the key the **behavior class**:
- `Classify("ls -la") == Read`, `Classify("ls -lah /x") == Read` → same class.
- User approves the **Read class** once ("always allow reads") → every Read
  command passes regardless of args. No re-prompt on arg drift.

This is *why* the user's instinct is right: lifting the grain from command-string
to behavior-class fixes the exact UX failure CC has.

**Cost (fail-open risk), which D5 must handle:** a class is coarse.
`Classify("rm -rf /") == Execute`; if the user sets Execute = always-allow, then
`rm -rf /` is auto-allowed too. Category-based trades away per-command control.

**Conclusion — two layers, and this IS the D2/D5 split:**
- **Layer 1 = D2 `Classify`**: semantic class (Read/Write/Execute/Other). Drives
  *bulk* decisions ("allow all reads").
- **Layer 2 = D5 rule engine**: on top of the class, a deny-list snipes specific
  dangerous commands (`rm -rf` stays ask/deny even when Execute is allowed).

CC has both layers, but its **default decision lands at exact-command grain**, so
usability collapses. Astra's lesson, write into D5: **default decision lands at
class grain; exact-command is reserved for deny-list exceptions.**

## bash `Classify` is an allowlist engine, not a few ifs (scope for D2 v1)

`readOnlyCommandValidation.ts` (1894 lines) shows the production cost of "is this
bash command read-only" — NOT a bool, a per-command allowlist:
- `GIT_READ_ONLY_COMMANDS`: every git subcommand enumerates its safe flags + each
  flag's arg type (none/number/string).
- `additionalCommandIsDangerousCallback`: positional-arg guard. `git branch`
  (list) is read-only; `git branch foo` (create) writes `.git/refs/heads/foo` →
  dangerous. Same for `git tag foo`, `git reflog expire`.
- Plus `GH_READ_ONLY_COMMANDS` (anti DNS-exfil), `DOCKER_*`, `RIPGREP_*`.

The git read-only subset alone is hundreds of lines.

**D2 v1 must NOT build that.** D2 = the contract + mechanism, proven by the
input-dependence demo. D2 v1 `BashTool.Classify`:
- Read: `ls`, `cat`, `pwd`, `echo`, `grep`, `head`, `tail`, `find` w/o `-exec`.
- Execute (most dangerous): `rm`, `mv`, `dd`, `mkfs`, fork bombs.
- Write: `>` redirect, `tee`, `touch`, `mkdir`.
- else → Other (fail-closed).
- leave `// TODO D5: full allowlist engine (cf. CC readOnlyCommandValidation.ts)`.

Demos `Classify("ls")==Read` vs `Classify("rm -rf")==Execute` (the D2 thesis)
without drowning in production scope.

## Decisions locked (Q1-Q3)

- **Q1 = default interface method (DIM).** Not a base class. Fail-closed default
  `ToolAction.Other` in the interface body. Two .NET caveats: (1) a DIM default is
  callable only through an `ITool` reference — tests that `new` a tool relying on
  the default must cast `((ITool)tool).Classify(...)`; a tool that `public`-
  implements `Classify` (e.g. `BashTool`) is callable both ways. (2) fail-closed
  `Other` makes "forgot to override" land in the strictest bucket — safe — which
  is the whole reason DIM beats a base class here (no single-inheritance slot
  burned, no type hierarchy D1 rejected).
- **Q2 = `Classify` takes the input bag.** Bound to Q3 (bash needs it). Isolate
  the stringly-typed `arguments?["command"] as string` in one private helper
  `GetCommand(arguments)` that both `Classify` and `ExecuteAsync` call.
- **Q3 = single `BashTool`, small allowlist v1** (scope above). One tool whose
  class varies by input — proves "behavioral flags over inheritance".

## Streaming refactor (B contract) — done in D2, anticipates D4

User pushed back on the first `ExecuteAsync` signature: `ValueTask<string>` blocks
until the process exits and returns output in one blob — unacceptable for a long
`npm install`, a build, or `tail -f` (user stares at a blank screen until it ends).
Changed the contract mid-D2 rather than ship a knowingly-wrong one.

**New contract (option B — structured, chosen over plain `IAsyncEnumerable<string>`):**

```csharp
public abstract record ToolOutput {
    public sealed record Progress(string Text) : ToolOutput;  // live, for the human; never sent to LLM
    public sealed record Result(string Text)   : ToolOutput;  // the single complete tool_result, for the LLM
}
IAsyncEnumerable<ToolOutput> ExecuteAsync(IDictionary<string,object?>? args, CancellationToken ct);
```

Why B (Progress/Result split) over A (raw string chunks): the final Result is NOT
required to equal the concatenation of Progress chunks. A tool may stream
"downloading… 50%…" to the human but hand the LLM a terse "installed 5 packages".
Two consumers, opposite needs — the human wants it live, the LLM wants one
complete block. The loop forwards every Progress to the consumer and feeds only
the last Result back into `_messages`.

**Implementation landmarks:**
- `BashTool.ExecuteAsync` bridges the process's event-based output
  (`OutputDataReceived`) into a pull-based async stream via
  `System.Threading.Channels` (unbounded, single-reader, multi-writer for
  stdout+stderr). Each line is yielded as `Progress` and accumulated for the
  final `Result`.
- `AgentLoop` dispatch: `yield return` cannot live inside a `try/catch` (CS1626),
  and Progress must NOT be buffered until completion (that would defeat
  streaming). So the loop drives the tool's enumerator **by hand** —
  `MoveNextAsync`/`Current` inside the try, the `yield` outside it. This is the
  standard .NET pattern for "need try/catch around MoveNext in an iterator".
- New event `AgentEvent.ToolProgress(ToolName, CallId, Text)` alongside
  `ToolResult`. `AgentApp` renders Progress live (`  | <line>`).
- D1's event-order assertion is unchanged: `FakeTimeTool` yields only one Result
  (no Progress), so `ToolUse → ToolResult → TextDelta` still holds. Streaming is
  covered by a new dedicated test, not by perturbing the D1 assertion.

Tests: 27/27 (D1 ×2, D2 classify ×24, new streaming test ×1).

## OPEN for D4 — three interruption scenarios the user raised (DO NOT FORGET)

User asked, while reviewing the streaming change, how the loop handles three
*different* interruptions. The streaming refactor is the FIRST HALF of the answer
(it makes mid-execution state observable); the SECOND HALF (inject signals back,
real process kill) belongs to **D4 (streaming / control layer)**. Recorded here
so D4 picks them up.

The common shape: today the loop is **one-directional** — `while(true)` emits
events, the consumer only reads. All three want the consumer (user OR a policy)
to **inject a signal back**. The two carriers are already threaded through the
loop: `CancellationToken` (the "stop" channel) and a future back-channel
(`Channel`/queue) for "inject content".

1. **Mid-turn the user wants to add something.** Trigger: loop is spinning in
   `while(true)`, hasn't returned control. Today `AgentApp` is
   `await foreach` (output) then `ReadLine` (input) — strictly serial, so the
   user can't type until the turn ends. Two fixes: *soft* (queue the input, apply
   it as the next turn's user message after the LLM produces text — simple, safe,
   user waits) vs *hard* (cancel the current LLM stream/tool on Enter, re-compose
   `_messages` with the partial + new input, restart — the Esc-then-type UX).
   Architectural prerequisite: a concurrent input-listener task talking to the
   loop via a shared CTS/channel; the serial foreach→ReadLine must split.

2. **User sees a tool half-run and says "stop, that's wrong."** Trigger: a tool's
   `ExecuteAsync` is mid-stream (the code we just wrote). The streaming change
   makes this *reachable*: the `ct` on `channel.Reader.ReadAllAsync(ct)` is the
   hook — cancel it → `OperationCanceledException` → exit the stream; the loop's
   manual enumerator deliberately rethrows OCE rather than swallowing it.
   **KNOWN HOLE (intentionally left for D4):** cancelling the read does NOT kill
   the child process — `ct` cancels "do I keep reading its output", and
   `using var process`'s Dispose only frees the handle. So "stop" currently means
   "I stop watching", not "it stopped".

3. **Agent autonomously watches tool output; on error, kill -9.** Trigger: not a
   user interrupt — an agent/policy decision to stop AND really kill. Splits into
   two concerns:
   - *Mechanism (how to really kill):* `BashTool` lacks this.
     `process.Kill(entireProcessTree: true)` is the `kill -9` equivalent (must be
     the whole tree, else killing `sh -c "npm install"` leaves npm running). Wire
     it into the cancellation/`finally` path: token cancels → `Kill(true)` →
     `WaitForExitAsync`. **TODO for D4.**
   - *Policy (who decides to stop):* something must consume the Progress stream
     live and judge it. This is exactly what the Progress/Result split buys — a
     middleware can subscribe to Progress, match "error"/"FAILED"/panic, and
     trigger cancel+kill. Its home is the loop's manual-enumerator section, where
     `enumerator.Current` is already in hand.

D4 takeaway to implement: turn the one-directional loop into a loop that can be
intervened in both directions (inject input, cancel, kill-tree), carried by
`CancellationToken` + a back-channel.

## Sourcing / re-verify TODO

- Codex axes + values: re-verify against openai/codex `docs/config.md` when a
  working web path exists. [unverified this session]
- LangGraph interrupt/resume API: re-verify against current LangGraph docs (docs
  moved to docs.langchain.com). [unverified this session]
- OpenCode: captured last session; spot-check the bash-pattern "last match wins"
  rule. [captured, not re-read this session]
