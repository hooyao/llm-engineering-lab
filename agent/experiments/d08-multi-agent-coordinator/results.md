# D8 Measured Payoff

## Command

```powershell
dotnet run --project samples/MultiAgentDemo -c Release --no-build -- --real --root .
```

Model: `gpt-5.6-sol` through the local OpenAI Responses-compatible endpoint.
Repository task: inspect compaction ordering and permission ordering with exact
file-and-test evidence, then explain how the guards compose.

## Assistant-side run — 2026-08-31

| Run | Total tokens | Cached input | Model calls | Tool calls | Wall time |
|---|---:|---:|---:|---:|---:|
| Single agent | 130,251 | 94,030 | 6 | 17 | 29,652 ms |
| Coordinator only | 6,292 | 0 | 3 | 2 | included below |
| Two workers | 120,124 | not printed | 9 | 33 | included below |
| Multi-agent total | 126,416 | — | 12 | 35 | 66,288 ms |

Measured multi-agent token multiple versus the single-agent baseline:

```text
126,416 / 130,251 = 0.97x
```

Other checks:

- the coordinator emitted two `Agent` calls in one response;
- two worker reports completed and were delivered in one synthesis batch;
- the coordinator-only marker did not appear in either worker report;
- both final answers cited the relevant implementation and tests correctly;
- neither read-only worker modified files.

## Assistant-side post-DI-refactor run — 2026-08-31

This run used the same task after moving every worker to an independent
dependency-injection scope with a scoped provider client, `AgentLoop`, and
telemetry wrapper.

| Run | Total tokens | Cached input | Model calls | Tool calls | Wall time |
|---|---:|---:|---:|---:|---:|
| Single agent | 156,501 | 120,639 | 7 | 19 | 31,049 ms |
| Coordinator only | 6,829 | 0 | 4 | 2 | included below |
| Two workers | 264,862 | 210,509 | 19 | 44 | included below |
| Multi-agent total | 271,691 | 210,509 | 23 | 46 | 60,140 ms |

```text
271,691 / 156,501 = 1.74x tokens
60,140 / 31,049 = 1.94x wall time
```

Both scoped workers completed concurrently, the coordinator received exactly
two bounded reports, and the isolation marker remained absent. The worker
duration maximum was 44,691 ms versus an 83,394 ms sum, confirming actual
overlap. This validates the new scope ownership on the real local endpoint.

The token difference between the first run (0.97x) and this run (1.74x) came
from stochastic tool behavior: the post-refactor workers made 19 model calls
and 44 tool calls versus 9 and 33 previously. Both measurements support the
same latency conclusion; one sample is not a stable token-cost estimate.

## Assistant-side post-tool-activation run — 2026-08-31

This run verified immutable tool advertisement plus keyed transient executor
activation against the real endpoint.

| Run | Total tokens | Cached input | Model calls | Tool calls | Wall time |
|---|---:|---:|---:|---:|---:|
| Single agent | 23,507 | 4,998 | 3 | 8 | 29,670 ms |
| Coordinator only | 6,637 | 0 | 3 | 2 | included below |
| Two workers | 173,611 | 122,132 | 12 | 27 | included below |
| Multi-agent total | 180,248 | 122,132 | 15 | 29 | 55,213 ms |

```text
180,248 / 23,507 = 7.67x tokens
55,213 / 29,670 = 1.86x wall time
```

The baseline happened to finish after only eight tool calls, so this sample's
token multiple is not comparable as a stable benchmark estimate. Its purpose
was integration verification: `Glob`, `Grep`, `Read`, and both `Agent` calls
activated successfully on demand; both worker scopes completed; reports were
batched; and the isolation marker remained absent.

## Interpretation

This task was narrow and required one final explanation combining two closely
related guards. Across the three assistant-side samples, multi-agent used
0.97x-7.67x the tokens and took 1.86x-2.24x the wall time. The result is
evidence against using multi-agent by default: orchestration and synthesis
latency outweighed parallelism here.

Anthropic's approximately 15x figure compares multi-agent systems with ordinary
chat interactions, not every multi-agent run with an already-agentic single
agent performing the same repository task. This local measurement therefore
does not contradict that report.

## Learner run

Pending. Record the learner's observed values and interpretation here before
marking D8 complete.
