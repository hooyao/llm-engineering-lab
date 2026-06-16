# D2 — Tool contract + behavioral permission flags

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

## Sourcing / re-verify TODO

- Codex axes + values: re-verify against openai/codex `docs/config.md` when a
  working web path exists. [unverified this session]
- LangGraph interrupt/resume API: re-verify against current LangGraph docs (docs
  moved to docs.langchain.com). [unverified this session]
- OpenCode: captured last session; spot-check the bash-pattern "last match wins"
  rule. [captured, not re-read this session]
