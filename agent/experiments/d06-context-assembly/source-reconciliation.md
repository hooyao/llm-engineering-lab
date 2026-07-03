# D6 — Context assembly: source-of-truth reconciliation (written BEFORE coding)

Per the process fix adopted after D4: read `refs/claude-code-sourcemap` +
`claude-reviews-claude` first, state what Claude Code actually does, then decide
where Astra agrees / simplifies. This note is that read; the design +
implementation notes come after.

Sources read (this session):
- `architecture/10-context-assembly.md` (the teaching chapter)
- `src/context.ts` (190 lines — the source of truth for layers b's memoization)
- A real captured request: `copilot-bridge/request-traces/20260616-025201-0001-inbound-req.json`
  (a live Claude Code code-reviewer subagent request; the ground truth for the wire shape)

## 0. The one fact that anchors the whole day: prompt-cache prefix contract

The API is stateless — every turn resends system + full history. A provider-side
**prompt cache** stores the KV state of a token *prefix* so a matching prefix skips
prefill. Three properties we build against:

1. **prefix, exact-match, from token 0.** Cache matches a contiguous prefix, byte-exact.
2. **`cache_control` is a "cache up to here" breakpoint**, not a "cacheable-from-here"
   marker. In the real trace, breakpoints sit at the END of `system[]` and at the
   LAST known message (`messages[19]`, the only one carrying `cache_control`).
   The provider then finds the longest byte-identical prefix ending at a breakpoint.
3. **Invalidation cascades forward.** If byte k differs from the cached prefix,
   everything from k to the end is recomputed; only 0..k-1 can still hit.

Consequence, the whole design goal of D6: **arrange context as `stable prefix +
per-turn tail`, and keep the stable prefix byte-stable across turns.** Mutable
things sink to the tail so they only pay their own prefill and never cascade-evict
the expensive stable prefix behind them.

## 1. The three layers (verified against the real trace)

| Layer | What | Lifetime | Wire location (in trace 0001) |
|---|---|---|---|
| **a** system prompt | identity + agent prompt | per-session, never changes | `body.system[1]` ("You are Claude Code", 57 ch) + `system[2]` front (reviewer prompt) |
| **b** session context | CLAUDE.md + env + git status + recent commits | computed once at session start, **frozen** | `body.system[2]` back (env + `gitStatus: ...(clean)` + recent commits) |
| **c** attachments | @-files, skill listing, reminders, diagnostics | recomputed per turn, used-then-discarded | inside `body.messages[]` — appended to the current user turn, or injected as periodic `role:system` messages |

Cross-checks that pin this down in the trace:
- `system[2]` carries `cache_control:{ephemeral}` and literally contains
  *"This is the git status at the start of the conversation. Note that this status
  is a snapshot in time, and will not update during the conversation."* — b is
  frozen, and CC tells the model so.
- Across three main-agent turns (seq 3/5/6) the per-block hash of `system[1]` and
  `system[2]` is byte-identical (`2719b7a469`, `c4981de86b`) while `messages` grew
  571→576. b froze; only the tail moved. This is the design working, measured.
- `messages[8] == messages[13] == messages[20]` — the same 421-char task-reminder
  text, re-injected every few turns as a `role:system` message. This is a c-layer
  attachment (periodic reminder), NOT part of `body.system`. Proves c has two
  distinct wire落点: (i) appended to the user turn (@-file content), (ii) injected
  as a periodic system message (reminders, skill listing).

## 2. Layer b's mechanism in the source: `memoize`

`context.ts` exposes `getSystemContext()` (:116) and `getUserContext()` (:155),
each wrapped in `lodash memoize`. `getGitStatus()` (:36) is also memoized and runs
`git status --short` + `git log --oneline -n 5` + `git config user.name` **once**,
then returns a fixed string with the "snapshot in time" preamble. The memo cache is
cleared only on explicit events (`setSystemPromptInjection`, worktree enter/exit,
`/memory`, compaction) — never per turn. So b = "compute once, reuse verbatim every
turn." That is the exact behavior Astra must reproduce: a provider that runs the
subprocess at session start and hands back the same string thereafter.

Why freeze a value (git status) that genuinely changes mid-session? Two reasons,
in priority order:
1. **Protect the a+b cache prefix.** git status / date change for reasons unrelated
   to the conversation (a mid-session commit, midnight date rollover). If b re-read
   live each turn, one such change would cascade-evict the whole a+b prefix + tail.
   Freezing makes a+b a byte-stable constant for the session.
2. (Secondary) avoid re-spawning the `git` subprocess every turn.
The accepted cost: the model sees a stale snapshot. CC accepts "stale but stable"
over "live but cache-busting", and tells the model it's stale. Live state, when
truly needed, is fetched via a tool (just-in-time — that's D9).

## 3. Layer c's mechanism: per-turn compute with a 1s timeout

`getAttachments()` (attachments.ts) runs every turn under an `AbortController` with
`setTimeout(ac => ac.abort(), 1000)`. Each attachment source is an external I/O
(disk read for @-files, IPC to the IDE, subprocess for linters, network to MCP).
The timeout exists because c is computed **on the critical path** (user has hit
enter, is waiting) and gathers content the model *did not ask for*. A single hung
source (e.g. an unresponsive MCP server) must not hold the turn hostage. On timeout,
that attachment is simply dropped — CC prefers "send the turn with less context" to
"block the user". The cost is bounded because c is per-turn: a dropped attachment is
recomputed next turn (delayed by one turn, not lost).

a and b need NO timeout: a is a static string, b is computed once at session start
(off the per-turn critical path). Only c is both per-turn AND external-I/O-bound.

## 4. What Astra CAN and CANNOT do at the M.E.AI layer (the real constraint)

Astra sends via `Microsoft.Extensions.AI` `IChatClient.GetStreamingResponseAsync`.
That abstraction is provider-neutral and does **not** expose Anthropic's
`cache_control` breakpoint field. So:

- **CAN (D6 scope):** structure context as three layers with correct lifetimes;
  make the a+b prefix byte-stable across turns; memoize b once; compute c per turn
  with a timeout; append c to the current user turn.
- **CANNOT here (defer, like D5's sandbox):** actually emit `cache_control` markers.
  That requires reaching past `IChatClient` to the raw Anthropic request. Note it as
  a provider-layer TODO. The *value* of D6 — a byte-stable prefix — does not depend
  on emitting the marker; the marker is a provider optimization on top of a prefix
  we made stable. (If the prefix isn't byte-stable, the marker buys nothing anyway.)

## 5. What NOT to copy (CC complexity to drop for D6)

- **30+ attachment types, the 3,998-line attachments.ts.** D6 builds the
  *mechanism* (an `IAttachmentProvider` list, per-turn, timeout-bounded), plus one
  or two concrete providers (a reminder, a skill/tool listing). The other 28 types
  are more providers on the same seam — cite, don't build.
- **`DYNAMIC_BOUNDARY` split inside the system prompt** (global-scope prefix vs
  session-scope suffix). That is a two-tier cache-scope optimization specific to
  Anthropic's `scope:'global'`. We keep a single stable a+b prefix; the finer split
  is a provider-layer refinement, deferred with the `cache_control` work.
- **The replacement chain** (override > coordinator > agent > custom > default).
  D6 keeps a single `SystemPrompt` string (already how AgentLoop takes it). Multiple
  agent definitions are D8 (multi-agent) territory.
- **@include recursion, glob-gated rules, CLAUDE.md discovery walk** (claudemd.ts,
  1,480 lines). Off-the-shelf memory-file discovery; not the agentic lesson. b's
  provider takes whatever string it's given (a real git snapshot in the demo).

## Design consequences carried into the implementation

1. `IContextLayer` split into three lifetimes:
   - `SystemPrompt` (a): a plain string, set once (AgentLoop already has this).
   - `ISessionContextProvider` (b): `ValueTask<string> GetAsync(ct)`, memoized —
     called once, result cached for the session. A `GitStatusContextProvider` runs
     the real subprocess once.
   - `IAttachmentProvider` (c): `ValueTask<string?> GetAsync(ct)`, called every turn,
     under a per-turn timeout; null / timeout => omitted.
2. Assembly in `AgentLoop`:
   - Build the system message ONCE as `a + "\n\n" + b` (b awaited once, memoized) —
     a byte-stable prefix for the whole session.
   - Each `SubmitAsync`: gather c from all attachment providers with a `TimeSpan`
     budget; append the surviving attachments as text to the current user message
     (c's落点 (i)). Keep periodic `role:system` reminders as a later provider (落点
     (ii)) — cite for now, build one simple reminder to prove the seam.
3. Byte-stability is the testable invariant: assert the system message is
   reference/byte-identical across turns; assert c can differ per turn; assert a
   hung attachment provider is dropped at the deadline and the turn still proceeds.
4. Fail-safe: an attachment provider that throws or times out is omitted (logged),
   never fatal — c is best-effort by construction (matches CC's `maybe()` wrapper).
