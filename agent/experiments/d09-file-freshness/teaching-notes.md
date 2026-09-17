# D9 Teaching Note — Freshness Is the Agent Requirement

## Concrete execution

The file begins as:

```text
hello world
```

Worker A and worker B both call `Read`. Their private observation stores contain
the same content hash, named `H0` here. The hash is never copied into model tool
arguments.

```text
A observation: shared.txt -> H0
B observation: shared.txt -> H0
disk:          shared.txt -> H0
```

A obtains the canonical-path gate, confirms `disk == A observation`, replaces
`hello world` with `hello astra`, and records `H1` in A's store:

```text
A observation: shared.txt -> H1
B observation: shared.txt -> H0
disk:          shared.txt -> H1
```

B then obtains the same gate. `disk H1 != B observation H0`, so its proposed
`hello world -> hello beta` edit does not run. The tool invalidates B's old
entry and returns a model-visible instruction to `Read` again. B's next read
records `H1`; its next model decision can then edit the current content.

No component decides whether `astra` or `beta` is semantically correct. The
harness establishes only that B's decision was based on old bytes.

## Why successful writes advance the observation

The executing agent supplied the complete replacement and Astra knows the exact
bytes it committed, including UTF-8 BOM handling. Recording their hash avoids an
unnecessary model-visible `Read` call before the same agent's next ordered edit:

```text
Read:   hello world  -> H0
Edit 1: hello astra  -> H1
Edit 2: hello beta   -> H2
```

Another agent does not receive these updates. Observation lifetime follows the
agent scope, not the file or application singleton.
The next `Edit` still reads current disk bytes locally for hashing and exact
replacement; it avoids sending another file tool result to the model.

This supports models trained to emit multiple `Edit` calls for one file. Astra
does not merge them into a synthetic transaction. Each call retains normal
zero-match and ambiguous-match behavior, and each result returns to the model.

## Ordering ownership

There are two independent ordering levels:

1. One `AgentLoop` already preserves model emission order for every write-class
   call.
2. `FileWriteCoordinator` serializes the same canonical path across independent
   agent scopes.

`WorkerCoordinator` no longer serializes every write worker globally. Workers
that own different files can overlap. Coordinator prompts should still assign
non-overlapping file sets because prevention avoids a failed tool round-trip.

## What atomic rename does and does not provide

`WorkspaceFileSystem` writes a sibling temporary file and then replaces the
target with one OS rename operation. A reader sees the old file or the new file,
not a partially truncated destination.

That operation does not accept an expected hash. The following external race is
still possible:

```text
Astra checks H0
external process writes H1
Astra replaces the path with its candidate derived from H0
```

The process-local path gate cannot constrain the external process. Therefore the
correct product claim is a freshness guard among Astra file tools, with
best-effort detection of completed external changes. It is not filesystem ACID,
isolation, rollback, or CAS. PowerShell and Bash writes are also outside the
file-tool gate.

## Why content hash instead of modification time

Claude Code stores content plus `mtime` and uses full-content comparison as a
Windows fallback. Astra hashes complete bytes directly:

- metadata-only touches do not force a model round-trip;
- any byte change, including outside a bounded visible range, changes the
  observation;
- the store retains 64 hexadecimal characters per observed file instead of the
  complete content.

The cost is one complete sequential read to establish a new observation. The
visible tool result remains bounded; only the local hashing work covers the full
file.
