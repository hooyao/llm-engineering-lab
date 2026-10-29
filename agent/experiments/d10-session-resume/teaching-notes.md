# D10 Teaching Notes — File Recovery and Durable Session State

## 1. Agreed file-content recovery rule

Consider a file-content operation that appends one line:

1. The agent reads the file and observes its complete content hash, H0.
2. The harness durably records that observation and the pending tool call.
3. Execution may modify the file, but the process crashes before the tool result
   is durably recorded.
4. On restart, the result is missing. Compare current content against H0.

If the hash matches, the current file content equals the observed content, so
the pending file-content operation may proceed through its normal validation,
permission, and same-path write gate. If the hash differs, do not retry the
write blindly or infer which actor changed the file. Return a recoverable
request to read again, then let the model decide using the new content and the
explicit fact that the previous call's outcome was not persisted.

The learner confirmed that current content is the relevant decision criterion.
The design does not need to establish whether the file changed temporarily and
returned to the same bytes. Do not turn this distinction into a requirement for
filesystem ACID, CAS, rollback, or automatic rebase.

A durably recorded terminal tool result is restored, not re-executed. A recorded
failure or denial remains a failure or denial; it is not converted to success.

This rule concerns file-content operations with a known target and observation.
It is not a generic recovery rule for arbitrary PowerShell commands, HTTP calls,
or other compound side effects. The append operation is a worked example, not
a claim that Astra currently exposes an Append tool.

## 2. Observations must survive the process

D9's `FileObservationStore` is agent-scoped and currently in memory. Restoring
conversation text alone does not restore its hashes. D10 must preserve the
observation relevant to a pending call before allowing that call to execute.
This includes observations advanced by successful writes, not only explicit
Read calls. Keep observation ownership isolated per agent/worker.

The exact record schema and persistence implementation are not decided yet.
The relationship among a tool result, its observation updates, and subsequent
calls must be explicit so recovery cannot associate a call with another agent's
snapshot or with the wrong version.

## 3. Proposed ownership: persistence in the session, freshness in file tools

The session mechanism should restore durable records and distinguish recorded
terminal outcomes from unresolved calls. It should not duplicate path handling,
hash computation, or the file tools' write checks.

Current `EditFileTool.EditAsync` already takes the same-path gate, compares the
agent observation with current complete content, and returns `Read again` on a
mismatch. Existing-file replacement in `WriteFileTool` has equivalent checks.
Restoring the correct observation allows those checks to be reused.

Do not conclude that every unresolved call can therefore be re-dispatched.
Recovery admission for other tools, create/delete cases, and any new generic
tool recovery contract still need design. Missing recovery support must not
silently authorize retries of arbitrary side effects. Resume must not bypass
the ordinary permission boundary.

## 4. AgentEvent is not the durable session record

Source inspected on 2026-09-17:

- `AgentEvent.cs` exposes live text deltas, tool progress/results, permission
  denials, errors, and compaction metrics. It does not carry accepted user
  messages, complete assistant messages, or replacement compaction content.
- `AgentLoop.SubmitCoreAsync` accumulates provider updates into a complete
  response and retains conversation messages only in `_messages`.
- `RunOneToolAsync` publishes per-call results to a channel, while
  `SubmitCoreAsync` appends the aggregate tool message only after all batches
  have finished. UI result visibility is not a durability acknowledgement.
- Applied compaction replaces `_messages`; the emitted `CompactionCompleted`
  contains a report, not the replacement messages needed for restoration.

Consequently, writing each AgentEvent from the CLI to JSON is insufficient.
The proposed integration records complete model calls before tool dispatch and
records each tool's terminal outcome independently before publishing its final
result. It must not wait for every tool in the model response to finish before
making an individual completion durable.

Live text/progress may remain transient; a separate UI transcript can preserve
them if the host needs it. That transcript is not authoritative recovery state.
Compaction needs its committed replacement content or a durable reference, not
only token-count metrics. A persistence failure must not be treated as an
ordinary tool error that allows dependent side effects to continue.

These are design notes, not implemented contracts. No runtime or tool API has
changed, and no Claude Code recovery behavior is inferred from Astra's design.
