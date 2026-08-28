# D7 — Compaction

Status: complete. The learner directly inspected and confirmed both payoff paths
on 2026-08-28.

## Artifacts

- `learning-notes.md` — the learner-paced explanation and resolved questions.
- `source-reconciliation.md` — Claude Code source findings and the deliberate
  Astra simplifications recorded before implementation.
- `agent/refs/Astra/src/Astra.Core/Compaction/` — result union, token estimator,
  policy, microcompact, and LLM full compact.
- `agent/refs/Astra/samples/CompactionDemo/` — deterministic and real-provider
  payoff.

## Run the payoff

From `agent/refs/Astra`:

```powershell
# Deterministic before/after: no provider call
dotnet run --project samples/CompactionDemo

# Adds a real gpt-5.6 summary and continuation through the local bridge
dotnet run --project samples/CompactionDemo -- --real
```

Optional overrides for the real run:

```powershell
$env:ASTRA_LLM_ENDPOINT = 'http://localhost:8765/codex'
$env:ASTRA_LLM_MODEL = 'gpt-5.6-sol'
$env:ASTRA_LLM_API_KEY = ''
```

Expected observations:

1. `call-1` and `call-2` payloads become the cleared marker while their IDs
   remain; `call-3` and `call-4` stay verbatim.
2. The deterministic full compact removes the reproducible bulk, retains
   `RETENTION-CODE-7429`, and keeps the current user turn verbatim.
3. The real summarizer retains the same exact code, and the following model call
   answers from the compacted history.

## Verification

```powershell
dotnet test
dotnet build
dotnet format Astra.slnx --verify-no-changes --no-restore
dotnet publish src/Astra.Cli -c Release -r win-x64
```
