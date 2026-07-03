# D6 — learning notes (context assembly, learner-paced)

> Same per-learner format as A2/A4/A5. Honor all three: complete coverage; depth set
> by THIS learner's familiarity (compress what they know — ChatGPT-era API mental
> model, systems/precision instinct; expand what they don't); phrased the way that
> worked in dialogue. Pacing: small segment -> clarify -> fold Q&A -> next. Read top
> to bottom to replay the lesson.
>
> This day was unusual: most of the value was in **correcting two specific
> misconceptions**, not in new derivation. Those two corrections are Segments 3 and
> 5 and are the highest-value part of this note.

---

## What D6 teaches (one line)

**Arrange the per-turn context as a byte-stable prefix (a + b) plus a per-turn tail
(c), so a provider's prompt cache keeps hitting the expensive prefix and only the new
tail is recomputed.** a = system prompt (never changes), b = session context
(git/env/CLAUDE.md — computed once, frozen), c = attachments (recomputed every turn,
timeout-bounded).

---

## Segment 0 — the foundation the learner already had (compressed)

The learner brought the right base from the ChatGPT era: the API is **stateless**, you
resend the whole conversation each call, history is **append-only** (never edit past
messages, just add to the end). All correct — kept as the anchor. Two things sat on top
of it that the learner did NOT have yet, and those became the day (Segments 1, 3).

Also correct upfront: prompt caching = "the provider stores the KV state of a prefix;
on the next call a matching prefix skips prefill, only new tokens are computed." The
learner correctly deferred the GPU-internal 'what is in the KV cache / how attention
reuses it' to A10/B13 — D6 does NOT need that layer. D6 only needs the **external API
contract** of the cache, which is Segment 1.

## Segment 1 — the prompt-cache prefix contract (the day's geometry)

Three properties, stated as the rules everything else follows from:

1. **prefix, exact-match, from token 0.** The cache matches a contiguous prefix,
   byte-exact.
2. **invalidation cascades forward.** If byte k differs from the cached prefix,
   everything from k to the end is recomputed; only 0..k-1 still hits.
3. (refined later in Seg 4) **`cache_control` is an "up-to-here" breakpoint**, not a
   "cacheable-from-here" flag.

The consequence the learner derived himself: **things that change must sink to the
tail.** A mutable item early in the prefix cascade-evicts everything after it. Worked
number that made it concrete:
```
correct  [a 200tok][b 800tok][c 500tok]   c changes -> recompute tail 500, hit 1000
wrong    [a 200tok][c 500tok][b 800tok]   c changes -> from c on all dead, recompute 1300, hit only 200
```
So "mutable in the tail" is not style; it is money. This is the same instinct as the
model-side "activation is use-then-discard" — here it is "mutable context is
tail-only".

## Segment 2 — the three layers, by lifetime

The learner ordered a, b, c correctly and gave the right reason: **c is the only thing
that changes this turn; a and b already have cache, so c goes last.** Fixed the rule as
"sort the prefix by change-frequency, low to high":

| layer | what | lifetime | why here |
|---|---|---|---|
| **a** system prompt | identity, rules, tool guidance | never changes | most cacheable -> front |
| **b** session context | CLAUDE.md + env + git status + commits | computed once, frozen | stable for the session -> middle |
| **c** attachments | @files, reminders, IDE selection, diagnostics | recomputed per turn | mutable -> tail |

## Segment 3 — MISCONCEPTION #1 CORRECTED: `system` and `messages` are TWO fields

This was the first real gap. The learner's ChatGPT model was "there is one array, you
append to it." True — but that array is `messages`, and it is **only half** of what
goes to the API. The request has **two parallel top-level fields**:
```
POST /v1/messages
{
  "system":   [ ... ],   <- a and b live HERE. rendered BEFORE messages.
  "messages": [ ... ]    <- the append-only array the learner knew.
}
```
So "append-only, don't edit history" is **true for messages**. The a/b/c ordering
lesson is mostly about how the `system` field + the newest message are arranged, NOT
about editing history. Once this landed, the rest of the day was unblocked. The learner
asked for a real example next (Segment 4), which is exactly right — this is a
learner who reasons from concrete wire bytes, not prose.

**Where c actually goes (subtle, resolved here):** c is NOT a mysterious third region.
It is **appended to the current turn's user message**, which is itself appended to
`messages`. So c "every turn new, doesn't edit history" is fully consistent with the
ChatGPT model. The word "recompute" earlier was misleading — it means **prefill of the
new tokens on the GPU**, not "rewrite past messages". History is never rewritten; only
the newly-appended tokens are prefilled.

## Segment 4 — the real trace (this is what made it click)

The learner asked for an actual request dump instead of prose. We used a real captured
Claude Code request: `copilot-bridge/request-traces/20260616-025201-0001-inbound-req.json`
(a code-reviewer subagent turn). What it showed, byte for byte:

- `body` has `system: list[3]`, `messages: list[21]`, `tools: list[123]` — the two
  fields + tool definitions, exactly Segment 3's structure, on real bytes.
- `system[1]` = `"You are Claude Code, ..."` (57 chars) with
  `cache_control:{ephemeral}` = **layer a**.
- `system[2]` (4583 chars) = the reviewer's agent prompt (front) + the env/git block
  (back) with `cache_control:{ephemeral}`. The git block literally contains
  *"This is the git status at the start of the conversation ... will not update during
  the conversation"* and `Status: (clean)` = **layer b, frozen, on the wire**.
- Across three main-agent turns (seq 3/5/6) the per-block hash of `system[1]` and
  `system[2]` was byte-identical (`2719b7a469`, `c4981de86b`) while `messages` grew
  571->576. **b froze; only the tail moved.** The learner saw the design working as a
  measurement, not a claim.
- Inside `messages`, `[8] == [13] == [20]` — the same 421-char task-reminder
  (`role:system`) re-injected every few turns. That is a c-layer attachment. Proves c
  has two wire落点: (i) appended to the user turn (@-file content), (ii) injected as a
  periodic `role:system` message (reminders, skill listing).

**`cache_control` refined (property 3):** in the trace, breakpoints sit at the END of
`system` and at the LAST known message (`messages[19]` — the only one carrying it). So
`cache_control` marks "cache everything from token 0 up to HERE", not "the stuff after
me is cacheable". The provider then finds the longest byte-identical prefix ending at a
breakpoint. That is why a tiny mutable thing before a breakpoint is cheap but not free —
Claude Code keeps the mutable billing header (`system[0]`, ~30 tokens, no
`cache_control`) small precisely for this reason.

## Segment 5 — MISCONCEPTION #2 CORRECTED: an @-file attachment HITS cache next turn

The learner's second wrong turn: "the previous turn's c can't be cached for this turn"
— stated as if c is always lost. This is **backwards for one of the two kinds of c**,
and getting it right is the sharpest idea of the day. Split c by where it goes:

```
c has two kinds, different fates:
  (i)  @file content + the user's question   -> APPENDED to the user message
                                              -> stays in history -> from NEXT turn on, HITS cache
  (ii) reminder / IDE selection / diagnostics -> injected transiently, NOT kept
                                              -> recomputed every turn, old copy discarded
```

The correction, stated plainly: kind (i) is **miss on the turn it first appears** (new
tokens, must prefill), then it sinks into stable history and **hits cache every turn
after**. It is not "c can't be cached" — it is "c is a one-turn miss, then a permanent
hit." The learner had it inverted.

Kind (ii) really is "gone next turn" — but that is **a harness choice, not a cache
limitation**. It describes instantaneous state ("you selected lines 10-20 right now"),
so keeping a stale copy would be wrong; the harness recomputes it every turn on purpose.

Why kind (ii) is safe to recompute-and-discard: the cost of "missing one attachment this
turn" is bounded because it reappears next turn. This directly justifies Segment 6.

## Segment 6 — the 1-second timeout on c (and why a/b need none)

The learner nailed the setup: **a/b are static — no computation at request time, so no
"could time out". c is computed fresh each turn, so only c can time out.** Correct, and
the missing piece was *what* c computes that is slow: **c is external I/O.** Each
attachment source is a disk read (@file), an IPC to the IDE, a linter subprocess, or a
network call to an MCP server. The latency is not CPU (string building never times out);
it is an outside source that might hang.

So the timeout (Claude Code: `getAttachments()` under an `AbortController` with
`setTimeout(ac => ac.abort(), 1000)`) protects the turn: c is gathered **on the critical
path** (user hit enter, is waiting) and gathers content **the model did not ask for**.
If one source (a hung MCP server) blocks 30s, the whole turn is held hostage. The
harness prefers "send with one attachment missing" to "block the user". The cost is
bounded — kind (ii) reappears next turn (Segment 5). a and b need no timeout: a is a
static string; b runs once at session start, off the per-turn critical path.

**Contrast with tools (the learner's own good challenge):** the learner objected that
"reading a file is a tool's job." Correct — the distinction is not *what* is done but
*who initiated it*:
```
             tool (D2/D3)                 attachment / c (D6)
initiator    the MODEL emits tool_use     the HARNESS injects it
model asked? yes, explicitly              no, pushed passively
into context tool_result message, NEXT    on THIS turn's user message
in history?  yes (append-only)            kind (i) yes / kind (ii) no
model waits? yes (it called it)           no (doesn't know it exists)
```
`@file` is the boundary case: the *user* said "attach this now", the harness reads it
this turn (trigger = "user @mentions a file"). Model-initiated file read is a Read
tool_use next turn. Both read disk; initiator + lifetime differ.

---

## The Astra implementation (what we built, mapped to the layers)

The constraint that shaped the scope: Astra sends through
`Microsoft.Extensions.AI.IChatClient`, which is provider-neutral and does **not** expose
Anthropic's `cache_control` field. So D6 could NOT emit the breakpoint marker — that is
deferred to the provider layer (like D5's sandbox). But the *value* of D6 is a
byte-stable prefix, and that is entirely the harness's job; the marker is a provider
optimization on top of a prefix we made stable. If the prefix isn't stable, the marker
buys nothing anyway.

What shipped in `Astra.Core/Context/`:
- `ISessionContextProvider` (b): `ValueTask<string> GetAsync(ct)`.
- `MemoizedSessionContext` (b): wraps a provider so its work runs **once** for the
  session. Caches the `Task`, not the string, so a first-turn race awaits the same
  in-flight work (one git subprocess, not two). = lodash `memoize` in context.ts:116.
- `GitStatusContextProvider` (b): runs REAL `git status --short` + `git log --oneline
  -5` once, prepends the "snapshot in time" preamble. = getGitStatus() context.ts:36.
- `IAttachmentProvider` (c): `ValueTask<string?> GetAsync(ct)`, null = contribute
  nothing.
- `AttachmentGatherer` (c): runs all providers concurrently under ONE shared deadline
  (linked CTS + CancelAfter). A throw/timeout drops that one, the rest survive; result
  in provider order (deterministic). = getAttachments() 1s AbortController.
- `PeriodicReminderProvider` (c): emits a fixed reminder every N turns = the
  msg[8]==[13]==[20] task-reminder.
- `AgentLoop` wiring: builds the system message once as `a + "\n\n" + b` (b awaited
  once, on the first turn — it's async, can't run in a ctor), then never rebuilds it
  (that's what keeps the prefix byte-stable). Each turn gathers c and prepends the
  surviving `<attachment>` blocks to the user message. All three params optional ->
  no-providers reproduces pre-D6 behavior (54 old tests unchanged).

## Payoff the learner saw (samples/ContextAssemblyDemo, no LLM)

```
PART 1  a+b prefix, 3 turns, REAL git status of this repo:
  turn 1/2/3 sha256[:12] = b31edda7479c  (identical) ; b subprocess ran ONCE
PART 2  reminder every 2 turns rides the USER message; system prefix hash unchanged all 4 turns
PART 3  a 30s-hung provider dropped at a 200ms deadline -> turn sent after 205ms,
        fast provider survived, hung one absent
```
Tests: 58/58 (54 pre-D6 + 4 new: byte-stable prefix + memoized-once, per-turn c on user
message, hung provider dropped at deadline, no-providers backward-compat).

---

## Segment 7 — prefill vs decode, and why a long prompt is slow even with cache
> (Learner asked this at the end; it fills in the KV-cache-internal layer he had
> wisely deferred to A10/B13 at the start of the day. Kept here because it is the
> "why compaction exists" motivation that sets up D7.)

**prefill** = phase 1 of generation: feed the ENTIRE input prompt (system + history +
this turn's new tokens) through one forward pass, in PARALLEL, to compute each input
token's KV and fill the KV cache. Output: the cache is built + the first output token.

**decode** = phase 2: emit output tokens ONE AT A TIME, serially. Each step appends the
new token, computes its KV, attends over ALL existing KV, produces the next token. M
output tokens = M forward passes.
```
prefill:  N input tokens -> 1 parallel forward -> KV cache built + 1st output token
decode:   1 forward -> 1 token, repeated M times (serial)
```
A cross-request **prompt cache** hit saves the prefill of the matching prefix (those
KVs were computed last time). It does NOT touch decode.

**Why a longer prompt is slower, two independent causes:**
- **A (prefill / TTFT):** the new-or-uncached portion still must be prefilled. Longer
  uncached input -> longer wait for the first token.
- **B (decode, the main cause, cache does NOT help):** every decode step attends over
  the WHOLE sequence — cost per output token ∝ sequence length L. The KV cache saves
  *recomputing* the L old KVs, but you still must *read all L of them* and do attention
  each step. At L=50000 every output token is ~50× more attention work than at L=1000.
```
per-output-token attention cost ∝ L (must scan L stored KVs, every step)
KV cache = "don't RECOMPUTE the old KVs", NOT "don't READ them each step"
```
- **GX10 twist:** decode is **memory-bandwidth-bound**, not compute-bound. Each step
  streams the entire KV cache from memory; KV cache bytes ∝ L
  (`2 × layers × seq_len × d_model × dtype_bytes`). At L=50000 that's 50× the bytes over
  the shared 273 GB/s LPDDR5x every step — same bandwidth wall as A4's post-saturation
  micro-batch finding.

Bottom line: prompt cache zeros out (A) for the cached prefix but does nothing for (B).
A 50k-token context decodes each token ~tens-of-× slower than a 1k one even at 100%
cache hit. **This is a second reason compaction (D7) exists** — not only "don't overflow
the window" but "long context makes every output token slower and costlier."

---

## Learner diagnostic update (append to the running profile from A2)

- **Reasons from concrete wire bytes, strongly.** The day only clicked when we stopped
  describing and dumped a real request (`20260616-...0001`) and hashed `system[]` across
  turns. For this learner, prefer "show the actual bytes / the real trace" over analogy
  or prose, every time. He explicitly asked for it ("给我举个实际的例子，发给api的
  context window长什么样子").
- **Two misconceptions this day, both the SAME shape as his known weak spot #1
  (direction inversion):** (#1) thought there was one array; the fix was seeing `system`
  and `messages` as two fields. (#2) thought "c can't be cached next turn"; the truth is
  the reverse — an @file c is a one-turn miss then a permanent hit. Both were
  containment/direction inversions, the umbrella-vs-kind pattern in a new domain. When
  this learner states a "X can't Y" rule, check whether it's inverted before accepting.
- **Challenges the framing productively** ("reading a file is a tool's job", "is Track D
  actually following the source"). These objections are correct and sharpen the lesson —
  answer them head-on (the tool-vs-attachment table came from his objection). Do not
  brush past them.
- Unchanged: conversation 中文, ML/systems terms in English, never translated; terminal
  can't render LaTeX (math as code); systems/precision instinct is strong (he
  immediately read the timeout as "protect the tail latency", read memoization as "cache
  the subprocess").
