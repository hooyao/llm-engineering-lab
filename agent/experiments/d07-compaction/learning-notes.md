# D7 — Learning Notes (Compaction, Learner-Paced)

> Build this note incrementally: explain one small segment, settle the learner's
> questions, record the explanation that worked, and only then continue. The
> learner reasons best from concrete wire content and measured token counts.

## Day target

Implement two complementary mechanisms in Astra:

1. deterministic `microcompact` for old, reproducible tool-result payloads;
2. LLM-backed `full compact` for semantic compression of a near-full conversation.

The payoff will include both a deterministic before/after inspection and a real
CLI session that crosses the configured threshold, compacts, and continues.

## Segment 1 — `microcompact` preserves the event; it discards the old payload

`microcompact` does not generate a summary. It targets old, high-volume,
reproducible `tool_result` content while retaining the protocol structure and a
recent working window.

Concrete example, using `keepRecent = 2`:

```text
Total context before: 172K tokens

Old tool results:
  call_1 Read = 12K
  call_2 Bash =  8K
  call_3 Grep =  7K
  call_4 Read =  6K

Recent tool results:
  call_5 Bash =  5K
  call_6 Read =  4K
```

After `microcompact`, `call_1` through `call_4` retain their `tool_result`
blocks and `tool_use_id` values, but each payload becomes:

```text
[Old tool result content cleared]
```

`call_5` and `call_6` retain their complete payloads. Approximately 33K tokens
are removed, so the context falls from 172K to about 139K tokens. The model can
still observe that `call_1` ran and returned a result, but it can no longer read
the result's verbatim content.

### Prompt-cache consequence: two source paths

The learner correctly inferred that changing an old `tool_result` changes the
prompt bytes. The precise consequence depends on the path:

- **Local content-clear path:** the cache can match only through the prefix before
  the first changed block. That block and everything after it must be rebuilt.
  Claude Code uses this path after a sufficiently long idle gap (default 60
  minutes), when it assumes the server cache has already expired; the rewrite
  therefore does not sacrifice a useful warm cache.
- **Server `cache_edits` path:** for a warm cache, Claude Code leaves the local
  messages unchanged and asks the Anthropic API to delete selected cached tool
  results. This path is designed to reclaim tokens without rebuilding the cached
  prefix, but it is provider-specific and feature-gated.

Therefore, "`microcompact` invalidates the cache" is true for direct local
content replacement, but it is not a universal property of `microcompact`.

## Segment 2 — Warm cache and the automatic fallback boundary

A warm prompt cache means that the provider still retains a reusable KV prefix
and the next request has the same bytes through that prefix. The client still
sends the complete message history over the wire. For example:

```text
Cached prefix after turn 9: 150K tokens
Turn 10 unchanged prefix:  150K tokens -> cache read
Turn 10 new tail:             5K tokens -> new prefill
```

A direct local replacement in an old tool result changes the prefix bytes. The
unchanged prefix before that block may still match, but the changed block and
the suffix after it must be rebuilt. Claude Code's time-based path accepts this
only after the cache is presumed cold already.

The provider-specific warm-cache path instead assigns a `cache_reference` to a
tool result and sends a `cache_edits` delete operation for that reference. Its
observable contract is that the provider edits the cached context while
preserving cache reuse; local history remains unchanged. The edit is pinned and
re-sent at the same message position on later requests. This path is
first-party and feature-gated in the inspected Claude Code source, so it is not
a portable Astra `IChatClient` mechanism.

### Why full compaction cannot be manual-only

The learner correctly identified all three costs of full compaction:

1. it invalidates almost all of the previous prompt-cache prefix;
2. the summarizer consumes a request containing nearly the full context;
3. the generated summary irreversibly loses detail.

Those costs justify delaying full compaction, but not making it exclusively a
user decision. A hard context limit makes an automatic fallback necessary.
Claude Code computes:

```text
effective_context_window
  = context_window - min(model_max_output_tokens, 20K)

auto_compact_threshold
  = effective_context_window - 13K
```

`auto_compact_threshold` therefore already includes both output-token headroom
and an additional 13K safety buffer. The control flow is:

```text
microcompact first
  -> recount input tokens
  -> if still above auto_compact_threshold: run full compact automatically
  -> otherwise: continue without full compact
```

The important condition is not merely "microcompact was inapplicable." It may
run successfully but reclaim too few tokens; full compact is still required if
the resulting context remains above the threshold.

## Segment 3 — Full compaction has two separate cache lifecycles

The learner correctly separated the two stages:

1. The compactor request reads the old near-full conversation. Claude Code's
   forked compactor reuses the main conversation's cached prefix by default, so
   most of a 170K-token input can be cache-read rather than newly prefilled.
2. After compaction, the client's active history is replaced by a compact
   boundary, the generated summary, and restored attachments. The old
   conversation history is no longer sent by the main loop.

Full compaction does **not** explicitly ask the provider to delete the old
prompt-cache entry. The inspected source shows:

- the compactor fork uses `skipCacheWrite: true`, so it can reuse the old prefix
  without creating a separate compactor cache entry;
- `notifyCompaction()` only resets the client's cache-break telemetry baseline
  (`prevCacheReadTokens = null`);
- there is no whole-history server-cache deletion call in the full-compaction
  path.

The old server-side entry may therefore coexist temporarily with the new
post-compaction prefix. It is no longer reachable from the active conversation,
receives no further cache hits, and eventually expires or is evicted under the
provider's cache policy. Claude Code can request a one-hour ephemeral TTL for
eligible calls; other calls may use a shorter provider default. This lifecycle
is provider-managed rather than triggered immediately by full compaction.

The API contract exposes a logical prompt cache. Calling its physical storage a
KV cache is a useful model, but the provider does not guarantee that the stored
representation is literally raw per-layer K/V tensors.

`cache_edits` is the separate, feature-gated exception used by cached
microcompaction to delete referenced tool-result content inside a cached
context. Full compaction does not use that mechanism to purge the old session
prefix.

## Segment 4 — Summary priority is future value, not source size

Given a choice between a 200-token unique error and 40K tokens of reproducible
file contents, the learner correctly kept the error. The file contents can be
loaded again just in time, while an error that was not persisted elsewhere may
be irrecoverable.

Irrecoverability alone is not sufficient, because a summary cannot retain every
unique sentence. Retention has two gates:

1. **Future relevance:** will later work depend on this information?
2. **Recoverability:** if needed later, can the agent retrieve it exactly from a
   file, command, trace, or another durable source?

This gives the practical rule:

```text
future-relevant + irrecoverable -> preserve, verbatim when exact bytes matter
future-relevant + recoverable   -> preserve an identifier/path, retrieve later
resolved + low future value     -> omit, or retain only root cause and fix
```

This is the meaning of "maximize recall first, then improve precision." Recall
protects all future-critical facts first. Precision then removes irrelevant or
re-fetchable material until the summary fits its budget.

## Segment 5 — Reactive compaction must shrink the compactor request itself

Reactive compaction is the fail-safe after the provider returns
`prompt_too_long`. It cannot submit the same oversized context unchanged to the
compactor. It splits on complete API-round boundaries, sends an older prefix to
the summarizer, and keeps a recent tail verbatim on the client.

Concrete example:

```text
Provider input limit:   180K
Rejected input:         183K

Static system/tools:     20K
Old history prefix:     133K
Recent history tail:     30K
Compact prompt:           2K
```

The compactor receives only:

```text
20K static + 133K old prefix + 2K compact prompt = 155K
```

The 30K recent tail remains in the client's `messagesToKeep`. If the compactor
returns an 8K summary, the next active context becomes approximately:

```text
20K static + 8K summary + 30K recent tail = 58K
```

The learner identified the key property: the tail does not need to enter the
summarizer request because the client appends it verbatim after the summary.
Splits must occur at API-round boundaries so a `tool_use` is not separated from
its `tool_result`, and related streamed/thinking blocks are not orphaned.

If a compactor request is still too long, the input must be reduced again and
retried. The inspected general full-compaction fallback allows at most three
prompt-too-long retries and drops old API-round groups as a lossy last resort.
The reactive path instead preserves a recent tail and is guarded against
repeating indefinitely; if recovery still cannot produce a valid compacted
context, the original error must surface.

## Segment 6 — Compaction commits atomically

Astra currently stores the active conversation in a mutable
`List<ChatMessage>`. The learner selected the required failure behavior:

```text
success   -> replace the active list with the complete compacted result
cancel    -> leave the original list unchanged
failure   -> leave the original list unchanged
```

The compactor must therefore construct and validate a candidate history without
mutating the live list. Only after summary generation and result assembly both
succeed may `AgentLoop` replace the active history in one commit step. In-place
clearing before an awaited summarizer call would make cancellation or provider
failure destroy the only usable conversation state.

## Segment 7 — Check before every model round-trip

One `SubmitAsync()` call is one user turn, but its internal `while (true)` may
perform several model round-trips:

```text
append user message
-> model call
-> append assistant tool_use
-> execute tool
-> append tool_result
-> model call again
```

A compaction check only at method entry misses both the newly appended user
message and any large tool results produced later in the same turn. The learner
therefore selected the correct integration point: inside the loop, immediately
before every `GetStreamingResponseAsync()` call.

```text
while true:
    observe current complete history
    apply microcompact if eligible
    recount tokens
    apply full compact if still above threshold
    call model
    append response/tool results
```

This placement makes the context safe for the first model call and every
tool-follow-up call, while preserving the atomic-commit rule from Segment 6.

## Segment 8 — The implemented Astra contract

The learner rejected `null` as the representation of "no compaction needed."
Astra therefore follows its existing `PermissionDecision` style with an
explicit sealed-record union:

```text
CompactionResult
  NotNeeded  -> safe to continue; no messages changed
  Applied    -> detached candidate + report; caller may commit
  Failed     -> no candidate; original history remains authoritative
```

`OperationCanceledException` remains cancellation and propagates rather than
being converted into `Failed`. Expected failures such as provider rejection,
empty summary, no compactable older turn, and a result still above threshold
use typed `CompactionFailureKind` values.

One transaction records an ordered list of steps rather than one tier value:

```text
Steps = [Microcompact]
     or [FullCompact]
     or [Microcompact, FullCompact]
```

This matters because clearing old tool results may reclaim some tokens while
still leaving the context above the full-compaction threshold.

The provider-neutral local microcompact is cache-aware:

```text
run local content-clear when:
  input >= pressure threshold
  OR last assistant response is at least 60 minutes old (cache presumed cold)
```

Below pressure with a warm cache, it does not rewrite history merely to save a
small number of tokens. Only tool names on an explicit allowlist are eligible;
an unknown tool result is retained. This preserves the recall-first rule for
potentially irreproducible output.

The full compactor preserves the leading system prefix and the current user turn,
summarizes completed older turns through an injected `IChatClient`, validates the
new token count, and returns a detached candidate. `AgentLoop` invokes the
pipeline immediately before every model round-trip and emits a
`CompactionCompleted` event after an atomic commit.

## Verification observed before learner payoff

- Baseline before D7: 58 tests.
- D7 automated suite: 70 tests passing.
- Solution build: zero warnings and zero errors.
- Deterministic microcompact: 12,090 -> 6,108 estimated tokens; `call-1` and
  `call-2` cleared, recent `call-3` and `call-4` retained, original history
  unchanged.
- Deterministic full compact: 6,169 -> 126 estimated tokens; reproducible bulk
  removed, exact retention code and current user turn retained.
- Repeated real `gpt-5.6-sol` runs through `http://localhost:8765/codex`:
  6,169 -> 326–362 estimated tokens; every generated summary and continuation
  retained `RETENTION-CODE-7429` exactly.

The learner directly inspected and confirmed both payoff paths on 2026-08-28.
D7 therefore satisfies the curriculum's visible-reward rule and is complete.
