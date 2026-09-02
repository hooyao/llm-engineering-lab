# D8 Coordination Correctness Boundaries

## Concrete conflict

Assume one file starts at version `V0`:

```text
hello world
```

Worker A and worker B both read `V0`. A proposes `hello world -> hello astra`;
B proposes `hello world -> hello beta`.

Serial execution only establishes order:

```text
V0 --A--> V1 (hello astra) --B--> conflict
```

It does not decide whether `astra` or `beta` is the intended final value. After
A commits `V1`, B's premise is stale. If the edits target different files,
there is no such conflict and they may run concurrently.

## Claude Code's layers

Claude Code handles this at several independent layers:

1. The coordinator schedules write-heavy workers one at a time for overlapping
   file sets. This prevents the common conflict before execution.
2. `readFileState` records the content and modification timestamp observed by a
   worker's prior full `Read`.
3. `Edit` reads the current file again immediately before writing. If the file
   changed since that worker's read snapshot, it returns
   `FILE_UNEXPECTEDLY_MODIFIED_ERROR` without writing.
4. Exact `old_string` matching is another precondition. If A already removed
   `hello world`, B cannot silently apply its proposed replacement.
5. After a successful edit, the executing session updates its read snapshot to
   the new content and timestamp.

The file-history subsystem may use content hashes for backup identity, but the
edit conflict check itself stores the observed content plus timestamp rather
than exposing a checksum to the model.

## Permission is separate

Permission answers whether a tool call is authorized. It does not establish
that the worker reasoned from the current file version. A worker permission
request can be approved and still fail the subsequent version or exact-match
precondition safely.

## Astra D8 implication

Multiple worker loops will need one shared, per-path write coordinator. A safe
write sequence is:

```text
acquire path lock
read current bytes
compare with the worker's observed version
apply the exact edit
atomically replace the file
record the new version
release path lock
```

A content hash can represent the observed version, but the compare and write
must occur under the same path lock; otherwise a time-of-check/time-of-use race
remains. The version should be harness-internal rather than another value the
LLM must copy through `Edit` arguments.

### Chosen v1 conflict policy

The learner chose strict conflict semantics. Any file-version change since the
worker's observation rejects the proposed write, even when its `old_string`
still exists uniquely in the current file. Astra will not automatically rebase
an edit because textual applicability does not prove that reasoning performed
against the old version remains valid. Recovery requires a new `Read`, a new
model decision, and a new write attempt.

### Chosen same-file edit policy

The coordinator should instruct the model to combine multiple changes to one
file into one edit operation whenever possible. Prompting is an optimization,
not a correctness boundary: the harness must still handle a response that
contains multiple same-file edit calls.

Claude Code's public `Edit` schema carries one `old_string` / `new_string` pair,
while its internal `getPatchForEdits` utility already operates on an edit list.
Astra can preserve the familiar external contract and normalize eligible calls
from one assistant response into one internal file-edit transaction. The
transaction uses one observed version, validates every replacement, requests
permission once, and performs one all-or-nothing atomic write. It still emits a
result for every original tool call ID.

After that transaction commits, a later edit operation for the same file must
perform a new `Read`. Astra does not silently treat the writer's post-write
state as a model observation: seeing a successful tool result is not equivalent
to inspecting the resulting file content.

Only adjacent same-file edits from the same assistant response are candidates
for normalization. Astra must not merge across a read/tool barrier, across
assistant responses, or across different paths because those boundaries may
carry ordering dependencies.

Required D8 conflict test: two workers start from the same observed version and
submit contradictory edits to one file. Exactly one edit may succeed; the other
must return a typed stale-version or exact-match conflict, and the winner's
content must remain intact.

## Worker result boundary

Claude Code does not copy a subagent's complete conversation or tool transcript
into the caller's context. `finalizeAgentTool` extracts only text blocks from the
last assistant message, falling back to the most recent assistant text when the
last message contains only tool calls. The caller receives that final text plus
small metadata such as agent ID, token count, tool-use count, and duration.
Background completion wraps the same final text in the `<result>` element of a
`<task-notification>` user-role message.

The raw worker history has a different lifetime. Claude Code can retain it in
worker/task state so `SendMessage` can continue that worker with its previous
context. An output-file path may also remain available for explicit inspection,
but the coordinator prompt discourages reading it because that would import tool
noise into coordinator context.

The learner chose the same D8 boundary for Astra:

- coordinator context receives only a condensed worker report and usage;
- tool calls, tool results, intermediate assistant turns, and reasoning are not
  copied into coordinator history;
- a live worker may retain its private history for explicit continuation;
- worker history is disposed when continuation is no longer possible or the
  coordinator session ends.

The 1,000–2,000-token report should be the worker's own final response, shaped by
its output contract, rather than a second summarization API call. This avoids an
extra model round trip while keeping the coordinator input bounded.

## Worker completion contract

The completion contract has two layers because model-authored text is not an
authoritative source for runtime facts.

### Trusted harness envelope

The harness, not the worker model, supplies:

- `task_id` and `worker_id`;
- terminal `status`: completed, failed, cancelled, or blocked;
- measured usage: input, output, cache-read, and cache-write tokens, model-call
  count, tool-call count, and wall-clock duration;
- failure metadata when applicable: error code, message, and retryability;
- durable artifacts such as worktree path or commit hash;
- an internal access log containing canonical read/write paths and observed file
  versions.

The complete access log and private transcript remain internal. Only small
fields needed for coordinator decisions, such as changed paths, enter the
notification.

### Untrusted bounded worker report

The worker's final response contains:

- `summary`: the outcome and central conclusion;
- `findings`: claims paired with concrete evidence such as `file:line`, symbol
  names, or a short output excerpt;
- `changes`: changed paths and the semantic change made, empty for research;
- `verification`: exact command/check, exit status, and meaningful result;
- `risks`: known uncertainty, assumptions, and unverified areas;
- `open_questions`: information still required from the coordinator or user.

`recommended_next_action` is deliberately not required. The coordinator owns
synthesis and must not outsource its decision to a worker recommendation.

The entire model-authored report is capped at 2,000 estimated tokens. Raw tool
output, copied files, intermediate reasoning, and the worker transcript are
excluded. When embedded in XML, every model-authored field must be escaped by a
real XML writer; string interpolation would allow worker output or repository
content to forge notification elements.

## Worked completion example

The coordinator assigns one read-only worker this task:

```text
Inspect why Astra runs compaction before every model round trip. Report the
implementation location, the test proving post-tool-result preflight, and any
remaining limitation. Do not modify files.
```

The worker privately calls `Grep` and `Read`, inspects `AgentLoop.cs` and
`AgentLoopCompactionTests.cs`, and runs one focused test command. None of those
intermediate messages or tool results enter coordinator context.

Its model-authored `WorkerReport` is conceptually:

```text
summary:
  AgentLoop executes compaction inside its while loop immediately before each
  IChatClient request, so a tool result appended during the same user turn is
  included in the next preflight.

findings:
  - AgentLoop.cs:124-143 performs CompactIfNeededAsync and atomically installs
    Applied.CandidateMessages before the model call.
  - AgentLoopCompactionTests.cs:77 verifies two model calls, two compactor
    calls, and a FunctionResultContent visible only to the second preflight.

changes: []

verification:
  - dotnet test ... --filter FullyQualifiedName~AgentLoopCompactionTests
    exit_code: 0
    result: passed

risks:
  - The test uses a fake compactor and fake chat client; provider-side
    prompt_too_long retry remains separate work.

open_questions: []
```

The harness then creates `WorkerCompletion` around that report:

```text
task_id: task-d8-demo-01
worker_id: worker-compaction-a
status: completed
report: <the WorkerReport above>
usage:
  input_tokens: 18420
  output_tokens: 812
  cache_read_tokens: 12300
  cache_write_tokens: 0
  model_calls: 3
  tool_calls: 5
  duration_ms: 4210
failure: null
changed_paths: []
artifacts: []
```

The usage numbers above are illustrative, not measurements. In a real run the
harness derives them from provider responses and tool execution. The
coordinator receives a bounded XML projection of this completion; the private
worker session remains addressable by `worker_id` only if continuation is still
enabled.

## Why workers cannot share one AgentLoop

The learner correctly identified that shared execution would mix tool calls,
make response-to-worker routing ambiguous, and leave cancellation without a
clear target. The current Astra implementation has additional concrete races:

- `_messages` is one unsynchronized mutable `List<ChatMessage>`. Concurrent
  `SubmitAsync` iterators can interleave user messages, assistant responses, and
  tool results into a transcript that belongs to neither worker.
- `EnsureSystemAssembledAsync` sets `_systemAssembled = true` before awaiting the
  session-context provider. A second iterator can observe `true`, skip assembly,
  and call the model before the first iterator inserts the system message.
- Each iterator has local tool batches and call IDs, but both append results to
  the same conversation. A correctly matched call ID can still land at the
  wrong position relative to another worker's messages.
- One worker may replace `_messages` with a compacted candidate while another
  worker is appending to the previous list, losing updates or compacting the
  wrong combined history.
- Cancelling one iterator can leave partial assistant/tool state visible to the
  other iterator, even though their `CancellationToken` values are distinct.

The isolation unit is therefore an `AgentLoop` instance, not merely a worker ID
attached to calls. Every worker owns its loop, transcript, context-assembly
state, compaction history, usage counters, observation versions, and
`CancellationTokenSource`. The coordinator routes only bounded completions via
worker ID.

## DI lifetime ownership

The concrete lifetime boundary is one worker execution, represented by an
async dependency-injection scope:

```text
application root
└── coordinator-session scope
    ├── coordinator AgentLoop
    ├── WorkerCoordinator
    └── AgentTool
        ├── worker-execution scope A
        │   ├── IWorker
        │   ├── AgentLoop
        │   ├── IChatClient
        │   └── UsageTrackingChatClient
        └── worker-execution scope B
            └── independent instances of the same scoped graph
```

`WorkerCoordinator` contains scheduling state but no executable worker
instance. It depends on `IWorkerSessionFactory`; after a worker has acquired its
concurrency and writer gates, the factory creates a scope and resolves exactly
one scoped `IWorker`. The scope is disposed before the terminal completion is
published. A queued or cancelled-before-admission worker therefore never
allocates a provider client or an `AgentLoop`.

The coordinator normally provides structured execution lifetime by awaiting
`IWorkerSession.RunAsync` inside `await using`. The session must still own an
independent disposal path: another owner may dispose it while execution is
active. Worker execution therefore returns `Task<WorkerCompletion>`, which is a
durable multi-await handle. The session stores and returns that same Task,
links the caller token with a session-lifetime cancellation source, and on
disposal performs cancel → await execution → dispose scope. This preserves
concurrent-disposal safety without a `ValueTask -> Task -> ValueTask` conversion.

Microsoft.Extensions.DependencyInjection does not implement inherited nested
scopes. The worker scopes are technically independent scopes created from the
root `IServiceScopeFactory`; the coordinator owns them logically through the
session objects and linked cancellation. Runtime values (`WorkerTaskId`,
`WorkerId`, and `WorkerRequest`) remain explicit session inputs rather than
mutable scoped accessors or `AsyncLocal` state.

The CLI registers the provider `IChatClient` as scoped. This avoids relying on
the provider-neutral `IChatClient` interface to guarantee concurrent use. A
provider adapter may later share an underlying HTTP transport or connection
pool while retaining one stateful wrapper graph per worker scope.

## Invocation-time tool activation

An advertised tool has two distinct lifetimes:

```text
agent-session lifetime
  ToolDefinition: name, description, static schema, classifier

single admitted invocation
  IToolExecutor: created after permission, executes once, then becomes unreachable
```

`AgentLoop` needs definitions before the model call because the provider request
must contain every advertised tool schema. It does not need executable objects
at that point. The execution path is therefore:

```text
model tool call
  -> definition lookup
  -> input-dependent classification
  -> permission decision
  -> keyed transient executor activation
  -> ExecuteAsync
```

Unknown calls stop at lookup. Denied calls stop at permission. Neither path
constructs an executor. All built-in schema documents are parsed once into
static readonly `JsonElement` values, and executor constructors only receive
dependencies; per-call files, processes, buffers, and streams are created and
released inside `ExecuteAsync`.

Microsoft.Extensions.DependencyInjection keyed transient registration provides
a new executor for every admitted call. Since the built-in executors are not
disposable and no collection retains them, each becomes eligible for collection
when its async execution completes. The agent/worker scope still owns shared
dependencies such as `WorkspaceFileSystem` and `WorkerCoordinator`.

## Options-backed composition root

Configuration data is runtime state, not composition logic. The CLI binds four
typed option objects:

```text
Llm                  -> IOptions<LlmConfig>
Compaction           -> IOptions<CompactionOptions>
Tools workspace      -> IOptions<WorkspaceOptions>
Tools PowerShell     -> IOptions<PowerShellOptions>
```

The corresponding runtime services receive these through constructors.
`CompactionOptionsPostConfigure` copies the provider output limit into the
compaction reserve before `CompactionOptionsValidator` runs. Disabling
compaction does not remove `IContextCompactor` from the graph; the same concrete
service returns `NotNeeded` before token estimation or an LLM call. This removes
the nullable/optional service branch from `AgentLoop` composition.

`AstraCliServiceCollectionExtensions.AddAstraCli` owns registration and binding.
The executable entry point only establishes configuration sources, applies the
registration module, creates one coordinator scope, and resolves `AgentApp`.
Configuration binding source generation is enabled, preserving Native AOT
without reflection-based fallback.

## Recoverable tool failure is not a terminal agent error

The learner-run payoff exposed a consumer-contract ambiguity. `Read` returned
`File not found`; `AgentLoop` correctly converted the exception into a
model-visible `ToolResult` so the next model round could choose a corrected
path, but it also emitted the generic `AgentEvent.Error`. `MultiAgentDemo`
treated every `Error` as terminal and aborted before recovery.

The event protocol now separates the two outcomes:

```text
recoverable tool exception
  -> ToolFailure (operator-visible)
  -> ToolResult("Error: ...") (model-visible)
  -> next model round

terminal agent failure
  -> Error
  -> current turn stops
```

The regression test requires this exact order:

```text
ToolUse -> ToolFailure -> ToolResult -> TextDelta
```

It also asserts two model calls. This makes recovery part of the typed event
contract rather than a convention inferred from whether an exception object is
present. The demo additionally prints bounded tool arguments and tells both the
baseline and workers that relative paths resolve from the Astra repository
root.
