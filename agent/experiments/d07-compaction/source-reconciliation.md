# D7 — Compaction Source Reconciliation

Written before implementation. The goal is to preserve the load-bearing behavior
from Claude Code while removing provider-specific and product-specific machinery
that Astra cannot justify yet.

## Sources inspected

- Teaching layer: `refs/claude-reviews-claude/architecture/11-compact-system.md`
- Claude Code source:
  - `src/services/compact/microCompact.ts`
  - `src/services/compact/timeBasedMCConfig.ts`
  - `src/services/compact/apiMicrocompact.ts`
  - `src/services/compact/autoCompact.ts`
  - `src/services/compact/compact.ts`
  - `src/services/compact/prompt.ts`
  - `src/services/compact/grouping.ts`
  - `src/query.ts` around preflight and reactive recovery
- Research layer: `agent/research/2026-agent-patterns.md`, A2–A5

## What the current source actually does

1. `microcompact` runs before auto-compaction. Its local content-clear path keeps
   message structure and `tool_use_id`, replacing only selected old tool-result
   payloads with `[Old tool result content cleared]`.
2. The local path is used after an idle gap when the provider cache is presumed
   cold. The warm-cache path is first-party and feature-gated: it leaves local
   messages unchanged and sends `cache_reference` plus `cache_edits` at the API
   layer.
3. Auto-compaction reserves up to 20K output tokens, then subtracts a 13K safety
   buffer. It runs after cheaper context reduction has had a chance to bring the
   request below threshold.
4. Full compaction invokes a one-turn, no-tools summarizer and replaces active
   history only after a valid summary exists. The summary prompt prioritizes
   user intent, technical decisions, files/code, errors/fixes, pending work, and
   the exact continuation point.
5. The compactor fork attempts to reuse the main prompt-cache prefix. Full
   compaction does not explicitly delete the old server cache; it only resets
   client telemetry and replaces client-side active history.
6. A compact request that is itself too long is retried with fewer complete API
   rounds. The general fallback permits three retries and may discard old rounds
   as a lossy final escape path.
7. Reactive compaction is a recovery trigger, not a distinct summary format. It
   reacts to an actual `prompt_too_long`, summarizes a reduced prefix, preserves
   a recent tail verbatim, and retries the original model call once.

## Terminology correction: mechanisms versus triggers

The teaching chapter describes three compression mechanisms:

1. microcompact;
2. session-memory compact;
3. full compact.

The curriculum's fourth tier, reactive compact, is a trigger/recovery path that
uses full compaction after a real provider rejection. Astra keeps these axes
separate:

- mechanism: `Microcompact` or `FullCompact` in D7;
- trigger: `Automatic`, `Reactive`, or `Manual`.

Session memory remains deferred because D7 does not yet have the D9 durable
memory primitive.

## Astra D7 decisions

### 1. Explicit result union

`CompactionResult` has three terminal cases: `NotNeeded`, `Applied`, and
`Failed`. No `null` or boolean encodes control flow. `Applied` owns a detached
candidate message array; only that case may replace the live history.
Cancellation propagates as `OperationCanceledException`. Expected summarizer
failures use the `Failed` case.

### 2. Atomic history replacement

All filtering, summarization, validation, and token recounting happen against a
candidate. A failed or cancelled operation leaves the original history intact.
The loop swaps the list reference only after an `Applied` result.

### 3. Preflight location

Run compaction inside `AgentLoop`'s `while (true)`, immediately before every
`GetStreamingResponseAsync` call. A single user turn can add a large tool result
and perform another model round-trip, so checking only once at `SubmitAsync`
entry is insufficient.

### 4. Provider-neutral microcompact

`IChatClient` exposes no Anthropic `cache_edits`, so Astra implements detached
local content clearing. It preserves the most recent tool results and replaces
older eligible result payloads without removing their `FunctionResultContent`
or call IDs. It never mutates caller-owned messages.

To avoid destroying a useful warm prefix, local content clearing runs only
under token pressure or after the last timestamped assistant response is at
least 60 minutes old. Missing timestamps do not imply a cold cache.

The first version uses an explicit `CompactableToolNames` allowlist and keeps a
configurable recent count. A result without a matching allowlisted
`FunctionCallContent` is preserved. Unknown tools therefore fail closed toward
retention instead of losing potentially irreproducible output.

### 5. Full compact with a verbatim recent turn

Astra summarizes only completed older user turns and preserves the current user
turn verbatim. This applies the source's reactive-tail safety to the smaller
framework and avoids orphaning a current `tool_use`/`tool_result` trajectory.
The leading system message is preserved byte-for-byte.

The summarizer receives no tools. Its prompt asks for text only and follows the
research tuning order: maximize recall of future-critical facts first, then
remove irrelevant or re-fetchable detail for precision.

### 6. Token accounting

Exact tokenization is provider/model-specific. D7 introduces an injectable
`IChatTokenEstimator`; the default is explicitly named and reported as a rough
UTF-8 estimate. Tests use a deterministic estimator. The report carries tokens
before/after and every applied step so the payoff does not hide estimation.

### 7. Failure behavior

- Below threshold: `NotNeeded`.
- Successful microcompact and/or full compact: `Applied` with the final safe
  candidate and ordered step reports.
- Provider error, empty summary, no compactable prefix, or a result still above
  threshold: `Failed`; the model call must not proceed with an unsafe context.
- Cancellation: propagate; never convert it into a domain failure.

## Deliberately deferred

- Anthropic `cache_edits` and cache-reference placement: provider-layer work.
- Background session-memory summary: depends on D9 memory design.
- Full reactive retry wiring: retain the trigger in the result model, but D7's
  implementation and payoff exercise automatic preflight compaction.
- Exact provider token counting: use provider usage when available in a future
  estimator; D7 keeps the estimate explicit.

## Payoff contract

1. Deterministic demo: print the exact old tool payloads, cleared call IDs,
   message structure, and token counts before/after microcompact and full
   compact.
2. Real demo: use the local OpenAI Responses-compatible endpoint at
   `http://localhost:8765/codex` with `gpt-5.6-sol`, force a small demonstration
   threshold, show the generated summary, and prove the next main model call
   continues from the compacted history.
