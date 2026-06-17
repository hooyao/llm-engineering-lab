# D3 teaching note — tool orchestration is instruction scheduling

> Written because the user's instinct — "this is the same as instruction
> reordering in a compiler" — is exactly right, and the curriculum named
> "concurrent reads, serial writes" as the goal without deriving *why* that rule
> is correct. The why is a data-hazard argument lifted straight from compiler /
> CPU instruction scheduling. This note is the derivation; the code it backs is
> `Astra.Core/ToolBatching.cs` and the partition step in `AgentLoop.cs`.

## The problem

In one turn the model can emit several tool calls at once, e.g.

```
[ read A,  write B,  read C ]
```

Running them strictly in order is always correct but slow: independent reads
could overlap. We want to parallelize what is safe to parallelize — and *only*
that. The danger is parallelizing (or reordering) something that isn't safe and
silently changing what the agent observes.

## Map tool calls onto loads and stores

Treat each tool call as a memory operation on the "world" (the filesystem, the
environment, external state):

- a **read** (`ls`, `cat`, `grep`) is a **load** — it observes state, changes nothing.
- a **write/execute** (`touch`, `rm`, a spawned process) is a **store** — it mutates state.

Now "can I reorder / overlap these two tool calls?" becomes the identical
question a compiler or an out-of-order CPU asks about two instructions: **is
there a data hazard between them?**

## The three hazards (and the one non-hazard)

For two operations on the same address:

| Hazard | Pattern | Reorderable? | Why |
|---|---|---|---|
| RAW | write→read | no | true dependency: the read must see the write's value |
| WAR | read→write | no | anti-dependency: the write must not clobber before the read |
| WAW | write→write | no | output dependency: final value must be the later write's |
| **RAR** | **read→read** | **yes** | **no dependency** — neither changes state, order is irrelevant |

The whole justification for "reads can run in parallel" is the last row: **two
reads carry no hazard.** It is not a vibe ("reads feel safe"); it is that RAR is
provably not a dependency. Everything else is.

## The aliasing problem forces conservatism

A compiler can sometimes prove two memory operations touch *different* addresses
(no alias) and reorder them even across a store. We have **no alias analysis over
the world**: we cannot prove that `write B` and `read C` touch different files.
`B` could be `touch /x` and `C` could be `cat /x`. With no way to prove
independence, we must assume the worst — that any write may alias any
read/write around it.

Conclusion: **every non-read call is a barrier.** No read may be hoisted across
it; no write may be reordered past another write. This is exactly how a compiler
treats a `volatile` access, a memory fence, or a call to an opaque function whose
side effects it can't see: it stops reordering across that point.

## The model's emission order is program order — it is the contract

A compiler must preserve the *observable semantics* of program order. Same here:
the model emitted `read C` **after** `write B` for a reason — possibly because it
wants C to observe B's effect (a RAW it is relying on). If it wanted C to see the
*old* state, it would have emitted C before B. So the emission order encodes the
model's intended dependencies. The harness has no right to reorder it. We are
allowed exactly one transformation, the one that is always semantics-preserving:
**overlap adjacent operations that have no hazard between them** (consecutive
reads).

## Therefore: stable partition, never sort

The naive idea — "collect all reads to the front and run them together, push
writes to the back" — is a **sort**. It hoists reads across write barriers and
changes semantics. Wrong.

The correct transform is a **stable partition** (a coalesce): scan in original
order, merge *consecutive* reads into one concurrent batch, and let any non-read
close the current batch and stand alone as a serial batch. Order between batches
is never changed.

Worked examples:

```
[ A:read, B:read, C:write ]   ->   batch1 = parallel(A, B)        // RAR: overlap
                                    batch2 = serial(C)             // barrier

[ A:read, C:write, B:read ]   ->   batch1 = [A]                   // alone (barrier follows)
                                    batch2 = [C]                   // the barrier
                                    batch3 = [B]                   // separate — NOT joined to A
```

The second example is the trap: A and B are both reads, yet they must **not**
share a batch, because the write C sits between them. Reads only coalesce within
a maximal *contiguous* run of reads.

## Where "is this a read" comes from — D2's `Classify`

There is no separate "concurrency-safe" flag in Astra. Concurrency safety is
*derived* from the D2 behavioral classification:

```
isConcurrencySafe(call)  ==  ( Classify(call.Arguments) == ToolAction.Read )
```

`Read` is the only hazard-free class. `Write`, `Execute`, and `Other` are all
barriers. This also gives fail-closed behavior for free: an unknown tool
classifies as `Other` (D2's strictest bucket), so it becomes a barrier and runs
alone — the worst outcome is *less* parallelism, never an unsafe overlap.

## The fold (the one piece of real logic)

`ToolBatching.Partition` is the stable partition as a single pass:

```
openReadBatch = null
for call in calls:
    if Classify(call) == Read:
        if openReadBatch is null: open a new concurrent batch
        append call to openReadBatch
    else:                       # barrier
        openReadBatch = null    # close the run so later reads can't coalesce across
        emit a serial batch containing just this call
```

This is structurally identical to Claude Code's `partitionToolCalls`
(`services/tools/toolOrchestration.ts:91`), which folds with the rule "join the
previous batch iff this call *and* the previous batch are concurrency-safe." Same
invariant: reads coalesce only with an adjacent open read run; anything else
breaks the run.

## What concurrency buys, and its bound

Overlap only helps the wall-clock of a *read batch* — its latency drops from
sum-of-reads toward max-of-reads. Writes serialize regardless, so a turn that is
all writes gets no speedup (correctly). The concurrency is bounded
(`MaxConcurrentTools = 10`, matching Claude Code's
`CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY`) because each read is a real process /
file handle, and an unbounded fan-out of a 50-read turn would exhaust handles.

## The merge mechanism (so the loop can consume N streams as one)

A concurrent batch has N tools each streaming `ToolOutput` (Progress + one
Result). To present them to the loop as a single `IAsyncEnumerable`, they fan in
through one `System.Threading.Channels.Channel<AgentEvent>`: N producer tasks
write, the loop's iterator is the single reader. Two details make it correct:

- every event carries its `CallId`, so once the streams interleave we can still
  map each Result back to the call that produced it (and feed results back to the
  LLM in the model's *original* order, not completion order);
- a background producer owns all the `try/catch` and always `Complete()`s the
  channel in a `finally`, so the iterator's drain terminates even on fault or
  cancellation — and `yield` stays out of the try/catch (the CS1626 constraint
  from D2).

## One-line takeaway

Tool orchestration is instruction scheduling with no alias analysis: RAR is the
only safe reorder, every write is a fence, the model's emission order is program
order, so the only legal optimization is to overlap maximal runs of adjacent
reads — a stable partition, never a sort.
