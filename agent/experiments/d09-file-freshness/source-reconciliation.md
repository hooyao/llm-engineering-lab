# D9 Source Reconciliation — File Freshness and Same-Path Serialization

## Demonstrated failure

A file starts with these exact bytes at snapshot `H0`:

```text
hello world
```

Two isolated workers read `H0`. Worker A proposes `hello world -> hello astra`;
worker B proposes `hello world -> hello beta`. Serial execution alone does not
tell B that its observation became stale after A wrote `H1`. Before B edits,
the harness must compare the current file hash with B's last observed hash and
ask B to read again when they differ.

The required behavior is freshness detection, not conflict resolution and not
a filesystem transaction. The agent only needs a current observation before it
acts. It can make a new model decision after a changed-file result.

## Sources inspected before implementation

### Claude Code file observation

- `restored-src/src/utils/fileStateCache.ts` stores normalized-path entries with
  observed content, `timestamp`, read `offset`/`limit`, and a partial-view flag.
- `restored-src/src/tools/FileReadTool/FileReadTool.ts` records the observed
  content and `mtime` after a successful read.
- `restored-src/src/tools/FileEditTool/FileEditTool.ts` and
  `FileWriteTool/FileWriteTool.ts` require prior read state without the
  `isPartialView` flag, synchronously load the current file immediately before mutation, compare it with
  `readFileState`, and return `FILE_UNEXPECTEDLY_MODIFIED_ERROR` on mismatch.
- The observation is harness state. Claude Code does not add a checksum argument
  to the public `Edit` schema and does not require the model to copy a token.
- A successful Claude Code write updates `readFileState` to the post-write
  content and timestamp, so the same agent can continue editing without a
  redundant `Read` while another agent's older observation remains stale.

Claude Code primarily uses `mtime`; a complete observed-content comparison is a
Windows false-positive fallback. Astra will use a content hash because freshness
depends on the bytes the agent observed, not metadata-only changes.

An explicit bounded `Read` is not the same as `isPartialView`. The latter marks
transformed auto-injected content, such as a truncated instruction attachment.
Normal range reads store `offset` and `limit` and can authorize an edit while the
recorded `mtime` is current; the complete-content fallback requires a full state.

### Multiple edits

- The public Claude Code `Edit` schema contains one `old_string` /
  `new_string` pair.
- `restored-src/src/tools/FileEditTool/utils.ts:getPatchForEdits` can prepare a
  list in memory, but Claude Code's normal orchestrator still executes separate
  public non-read calls serially.
- Astra's existing `ToolBatching` already treats each write as a serial barrier
  inside one loop.

Astra will not add an ACID-like MultiEdit transaction. Adjacent same-file calls
remain ordinary ordered calls. After each successful edit, the executing
agent's observation advances to the exact bytes it wrote, so a later call can
edit that result. Zero or ambiguous exact matches remain recoverable tool
failures. This is compatible with models trained to issue several `Edit` calls
for one file.

### Atomic file replacement

The .NET runtime sources for `FileSystem.Windows.MoveFile` and
`FileSystem.Unix.MoveFile` show that a same-directory
`File.Move(temp, target, overwrite: true)` uses Windows `MoveFile` and Unix
`rename`. Astra retains its existing sibling-temp replacement so one admitted
write does not expose partially written bytes. This is an implementation detail
of one write, not a filesystem transaction or conditional commit.

## Guarantee boundary

A per-path process-local lock serializes Astra `Edit`/`Write` operations for the
same canonical file across agent scopes. Different paths remain independent.
The lock does not control IDEs, linters, shells, or other processes. A content
check immediately before an Astra write detects changes completed before that
check; it cannot eliminate the final external time-of-check/time-of-use window.
Windows `ReplaceFile` and POSIX `rename` do not accept an expected content hash.

D9 therefore makes no ACID, CAS, or arbitrary-external-writer claim. Scheduling
should avoid assigning the same file to multiple workers, while the hash guard
provides recovery when that still happens. Shell writes remain outside this
contract.

## Astra before D9

- `ReadFileTool` returns bounded UTF-8 content but records no observation.
- `EditFileTool` reads current content, applies one exact replacement, and uses
  `WorkspaceFileSystem.WriteTextAtomicallyAsync`; it has no read-before-edit or
  stale-observation check.
- `WriteFileTool` overwrites an existing path without requiring an observation.
- `AgentLoop` serializes writes inside one loop, but separate worker scopes have
  no shared same-path write gate.

The current exact-match rule catches only changes that remove `old_string`. It
silently accepts a stale edit when unrelated bytes changed but the old string
still exists.

## Selected Astra contract

1. A scoped observation store belongs to one agent/worker execution.
2. A successful bounded `Read` records a SHA-256 hash of the complete file bytes
   read from the same stream as the visible range. The hash remains
   harness-internal.
3. An existing-file `Edit` or complete `Write` requires a stored observation.
4. The writer acquires a shared process-local gate for the canonical path, reads
   the current bytes, and compares their hash with that agent's observation.
5. A mismatch returns a recoverable changed-file result and invalidates the old
   observation. The agent must `Read` once and decide again; Astra does not
   rebase or merge its proposal.
6. A matching observation permits the normal exact edit or complete write.
   Success advances only the executing agent's observation to the hash of the
   bytes it wrote.
7. Multiple writes from one response execute in model emission order. No
   transaction grouping or rollback is added. The next call compares against
   the snapshot advanced by the preceding successful call.
8. A new-file `Write` is allowed without a prior observation. Same-path Astra
   writers are serialized; if the path already exists when the call executes,
   it is treated as an existing file and requires `Read` first.
9. Write-capable workers can use the same scoped observations and shared path
   gates after stale-observation, ordering, permission, BOM/line-ending, and
   cancellation tests pass. Worker scheduling should still minimize overlapping
   file ownership.

## Intentional differences from Claude Code

| Boundary | Claude Code | Astra D9 |
|---|---|---|
| Observation | Content + `mtime` | SHA-256 of complete exact bytes |
| Model contract | Hidden state | Same hidden state; no checksum argument |
| Bounded reads | Range content + `mtime`; transformed partial views cannot authorize edits | Hash complete file while returning a bounded range |
| Same-process writers | Event-loop serialization | Shared same-path gate across agent scopes |
| Successful write | Advances read state | Advances the executing agent's observation |
| Multiple calls | Serial calls | Serial calls; no synthetic transaction |

## Payoff gate

The learner-visible artifact will print these outcomes:

1. Two sessions observe `H0`; A writes `H1`; B is told to read again and `H1`
   remains intact.
2. B reads `H1`, retries its model-selected edit, and advances to `H2`.
3. One agent emits several ordered edits to the same file; each succeeds without
   a redundant read because the harness advances that agent's snapshot.
Same-path exclusion and independent-path admission are additionally checked by
deterministic tests.
