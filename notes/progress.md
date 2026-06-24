# Progress — append-only log + current snapshot

> **New Claude session: read this file first.** The snapshot block below tells you
> what physically exists on the GX10 right now. The log below tells you how we got
> here. The "Open threads" section at the bottom tells you what to ask the user
> about / pick up next.

---

## STATE SNAPSHOT (last updated 2026-06-17)

### Hardware

- **Box**: ASUS Ascent GX10, single unit, in user's home in China.
- **Network**: UniFi UXG Fiber, GX10 at static IP `192.168.1.200`. SSH key auth from
  the user's Windows laptop. **You can `ssh hooyao@192.168.1.200` without a password.**
- **Sudo**: still requires password `123`. Pattern that works inside a remote command:
  ```bash
  ssh hooyao@192.168.1.200 'cat > /tmp/askpass.sh <<EOF
  #!/bin/sh
  echo 123
  EOF
  chmod +x /tmp/askpass.sh
  SUDO_ASKPASS=/tmp/askpass.sh sudo -A <command>'
  ```
  Do NOT pipe `echo 123 | sudo -S` into a command whose own stdin you also need
  (wget, tee from a pipeline, etc.) — that bit us during apt setup. Use the askpass
  helper above for anything non-trivial.

### Software state

- DGX OS fully updated. Driver `580.159.03`, CUDA `13.0`, kernel `6.17.0-1021-nvidia`.
- All firmware capsules current (`fwupdmgr get-updates` → "No updates available").
- Docker `29.2.1` working. User `hooyao` is in `docker` group.
- NVIDIA Container Toolkit + CDI spec at `/var/run/cdi/nvidia.yaml` present.
- `nvtop` and `btop` installed for monitoring.
- `gh` CLI 2.45.0 installed (user logged in).

### Containers pulled

| Tag | Size | Purpose |
|---|---|---|
| `nvcr.io/nvidia/pytorch:26.04-py3` | 23.5 GB | **default for new work** — CUDA 13.2, torch 2.12.0a0, torchao 0.17 |
| `nvcr.io/nvidia/pytorch:25.11-py3` | 19.5 GB | older, kept for fallback / reproducibility — needs pinned deps |
| `nvcr.io/nvidia/tensorrt:25.11-py3` | 10.6 GB | inference optimization (Track A10) |
| `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` | 6.59 GB | full CUDA dev image (nvcc) for custom kernel work |
| `nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04` | 415 MB | minimal GPU sanity test |
| `hello-world` | 5 KB | left over from docker smoke test, harmless |

### Models on disk (`/home/hooyao/models/`)

178 GB, 264 files, all SHA256-verified against `SHA256SUMS` in the same directory.
The catalog (which model is which size, where it goes in the curriculum) lives in
`notes/curriculum.md`. Top-level orgs present:

```
allenai/       argilla/       BelleGroup/    databricks/    google/
HuggingFaceH4/ HuggingFaceTB/ meta-llama/    nvidia/        Qwen/
yahma/
```

`/dev/nvme0n1p2` (1 TB ext4) currently ~28% used, ~660 GB free.

### Verified GPU performance on this unit

Recorded in `notes/hardware-gx10.md` § "Measured on this specific unit":

- **97.3 TFLOPS BF16 sustained** at N=16384 GEMM (78% of nameplate 125 TFLOPS)
- 89 W GPU die, 79 °C, P0, 96% util — no throttle observed
- ~1.09 TFLOPS/W efficiency, comparable to H100 SXM

LoRA 3B at batch 4 / seq 1024 only pulled ~44 W — small-GEMM workloads don't
saturate the GPU. To stress-test, use `experiments/bench/bf16_peak.py`.

### Repo state (`hooyao/dgx-spark-playground`, private)

```
.
├── CLAUDE.md                                 ← core directives + hardware spec
├── .gitignore / .gitattributes
├── .gitmodules
├── notes/
│   ├── progress.md                           ← YOU ARE HERE
│   ├── bootstrap-gx10.md                     ← first-boot recovery (read on failures)
│   ├── hardware-gx10.md                      ← cited GB10 specs + measured numbers
│   ├── curriculum.md                         ← static asset catalog + memory budgets
│   └── curriculum-v2-execution.md            ← day-by-day learning plan (3 tracks)
├── tools/
│   ├── download_models.py                    ← office-side HF downloader (already used)
│   ├── verify_models.py                      ← SHA256 integrity checker
│   └── launch_pytorch.sh                     ← standard `docker run` wrapper
├── experiments/
│   ├── smoke-test/                           ← LoRA-3B + nvidia-smi dmon harness
│   └── bench/
│       └── bf16_peak.py                      ← square BF16 GEMM throughput benchmark
└── dgx-spark-playbooks/                      ← submodule → NVIDIA/dgx-spark-playbooks
```

### Stuck pinned dependency versions (the one PyTorch container gotcha)

**Default container is now `nvcr.io/nvidia/pytorch:26.04-py3`.** It ships `torchao 0.17.0+git`,
so modern peft / transformers install cleanly with no pins. Last verified 2026-06-05:

```
transformers 5.10.1  +  peft 0.19.1  +  datasets 4.8.4  +  accelerate 1.13.0
```

`experiments/smoke-test/run-26.04.sh` and `experiments/bench/run-26.04.sh` use this
container with no pins.

**Legacy: `nvcr.io/nvidia/pytorch:25.11-py3`.** Ships `torchao 0.14.0+git`, which
collides with modern peft. Use these pins inside this container:

```
transformers >= 4.50, < 4.55
peft         >= 0.15, < 0.18
accelerate   >= 1.0,  < 1.5
datasets     >= 3.0,  < 5.0
```

Already pinned in `experiments/smoke-test/run.sh` and `experiments/bench/run.sh`.
Keep these as fallback / reproducibility, but prefer the 26.04 variants for new work.

---

## LEARNING STATE (where the user is across all tracks)

> A new session: this is the "what have I actually done" block. The hardware
> snapshot above is what *exists*; this is how far the *learning* has gone.
> Update the "next" column whenever a day is finished.

| Track | What | Done so far | Next concrete step |
|---|---|---|---|
| **A** Fine-tuning | `notes/curriculum-v2-execution.md` | **A1 + A2 + A3 + A4 done (incl. full per-learner teaching).** A1: `budget.py` + concept notes. A2: first full-param SFT (Llama-3.2-1B, 500 ex, peak 13.84 GB → corrected A1 to 12 B/param) **+ taught segment-by-segment in `experiments/a02-sft-1b/learning-notes.md` (Seg 1–6e, ~720 lines, with a learner diagnostic at the end — READ IT before teaching)**. A3: `experiments/a03-eval-1b/results.md` has the learner's OWN before/after observations + the SFT-definition / dataset trace. **A4: gradient accumulation — explicit hand-written loop (A2's `trainer.train()` debt PAID), 3-config sweep on 3B. Learner read `loop_explained.py`, predicted the table, and caught the non-monotonic step_time himself. Taught in `experiments/a04-grad-accum/learning-notes.md` (Seg 0–5).** | **A5 — activation checkpointing + seq_len** (`experiments/a05-ckpt-seqlen/`). 3B + LoRA r=16, sweep seq_len × checkpointing on/off, 2D table of peak_mem/step_time. Note: A4 used bf16 m/v (8 B/param, Seg-6d footgun) — fine for the demo; mention fp32 states for real runs. |
| **B** Pretrain+RLHF | same file | none | B1 — micrograd (`experiments/b01-micrograd/`) |
| **C** Math | same file | none (reading Parr & Howard in parallel) | C1 — derivatives review |
| **D** Agent eng | `agent/curriculum-agent.md` | **D1–D5 done, Linux-verified** — D1: agent loop. D2: `Classify -> ToolAction` + streaming. D3: tool orchestration (`ToolBatching.Partition` + `Channel` fan-in). D4: control layer (cancel kills the process **tree** then reaps). D5: permission pipeline — 3-state decision (Allow/Deny/Ask), pluggable `IPermissionPolicy` (default `ClassDefaultPolicy`: Read→Allow, else→Ask, +rule exceptions) + `IUserConfirmation` + `DefaultPermissionEngine`, gated in `RunOneToolAsync` before execute, fail-closed. Notes: `agent/experiments/d0{1..5}-*`. **Astra submodule at `e3b52a6`; 54 tests.** | D6 — context assembly (three-layer cache strategy) per `curriculum-agent.md`; OR the deferred D4 control-plane / D5 CLI confirmation UI + InputSchema validation. |
| **Career** | `notes/career-transition-research.md` | research complete (4 reports) | Phase 0 — build portfolio, contact CPH/Dublin HMs |

**For Track D specifically:** the next-step state above only tracks *which day*.
Once building starts, the agent-core implementation state lives in the **Astra
submodule** (`agent/refs/Astra/progress.md` + its `CLAUDE.md`). Read those before
writing Astra code so you don't re-do or contradict prior work there.

---

## Open threads (the "next session, pick up here" list)

When the user is ready to continue, the natural next steps are:

1. **Track A Day 1** of `curriculum-v2-execution.md`: write
   `experiments/a01-mem-budget/budget.py` (memory arithmetic calculator). The user
   has not started any curriculum days yet.
2. The `experiments/smoke-test/` script has been verified to run end-to-end (5 steps
   of LoRA-3B succeeded). The user has NOT yet executed the full 200-step run.
   If they ask "did we finish the smoke test?" the answer is "the pipeline is
   verified; the 200-step thermal/stability characterization wasn't run because
   we already got the same data from the BF16 GEMM benchmark."
3. An HF token was pasted into the conversation on the bootstrap day
   (string redacted; original was `hf_GnecFHEN...` -- shortened here so the
   GitHub push-protection secret scanner doesn't flag this file). **You
   told the user to revoke it.** Do not assume they did; if they reference
   HF auth, re-mention this.
4. User mentioned reading Parr & Howard's "Matrix Calculus You Need for Deep
   Learning" in parallel (Track C). They are NOT blocked on math to start tracks A or B.
5. **Track D (agent engineering) exists now** — `agent/curriculum-agent.md`,
   D1–D16, two phases. Day 1 is `agent/experiments/d01-agent-loop/`: implement
   the `while(true)` agent loop in the Astra submodule. Not started. The project
   is now **three legs** (model side / agentic side / career) — see CLAUDE.md
   "Repository Purpose". The agent-side work is mode-gated like Tracks A/B/C
   (tutor mode for new Track D subsystem work).

---

## LOG (append new entries at the top)

### 2026-06-24 — Track A Day 4 done (gradient accumulation; A2's hidden loop unrolled)

A4 taught in tutor mode, learner-paced (the A2 method), then run on the GX10. The
learner explicitly wanted to READ the code before any run, and A4 is where A2's
`trainer.train()` four-beat loop finally gets unrolled into an explicit, line-by-line
loop. Both honored.

**Teaching arc (`experiments/a04-grad-accum/learning-notes.md`, Seg 0–5):** learner
derived gradient accumulation himself ("keep one accumulator row, add onto it, divide at
the end" = the mechanism, complete). Cleared a real confusion (gradient is per-sample AND
per-parameter — I'd said "16 numbers" sloppily). Asked the precise math question (why does
scaling the scalar `loss` divide every gradient — answer: gradient is linear in loss,
const-multiple rule) and hit a product-rule slip (derivative of a constant is 0, not 1)
that traced to his known weak spot #2 (value vs rate-of-change fused). Connected `loss /
ACCUM` to A2 Seg-6d loss scaling (same linearity, opposite direction). PyTorch Q&A folded
in: why `model(**batch)` returns a loss (the collator injects `labels` -> HF CausalLM
self-scores), eager vs TF1 graph + async CUDA dispatch + `torch.compile`, and that reading
loss in a debugger implicitly syncs (so it IS visible, but `.item()`/`print` in a hot loop
is a stall).

**Deliverables (`experiments/a04-grad-accum/`):** `loop_explained.py` (annotated skeleton,
the file the learner read), `train.py` (runnable hand-written loop — no Trainer — with
DataLoader, peak-mem/step-time measurement, mem-check diagnostic), `run.sh` (3-config
sweep), `notes.md` (payoff + findings), `learning-notes.md`.

**Payoff — the 3-config sweep (Llama-3.2-3B full SFT, BF16, seq 1024, 30 opt-steps),
same effective batch 16:**
```
            peak_mem    step_time   final_loss
micro=1     30.16 GB    4363 ms     1.3526
micro=4     35.82 GB    3378 ms     1.2469   <- sweet spot
micro=8     46.96 GB    3834 ms     1.2489   <- slower AND more memory
loss spread (max-min): 0.1057
```
- Core claim verified: same effective batch -> same training (loss band 0.11 << step
  spread 985 ms), independent of the micro/accum split. The Seg-2/3 math identity, on metal.
- **The learner's best catch (from raw feel, "doesn't seem much faster"):** step_time is
  NON-MONOTONIC. micro 1->4 is -23%, but micro 4->8 is **+13% (slower)** while costing
  +11 GB. Past GPU saturation (micro=4 already fills the GEMM), micro=8 gains no compute
  and goes memory-bandwidth-bound on the shared 273 GB/s LPDDR5x. Corrected rule:
  micro-batch has an OPTIMUM ("just saturate the GPU"), not "bigger = faster."
- Ground-truth findings: m/v dtype printed `torch.bfloat16` -> 8 B/param fixed (not A2's
  12 — raw `torch.optim.AdamW` on bf16 params keeps bf16 states; the Seg-6d footgun, fine
  for 30 steps). mem-check showed ~6 B/param resident between steps because
  `zero_grad(set_to_none=True)` frees grad entirely (grad's 2 B is transient, only in peak)
  — literal proof of "gradient is use-then-discard."

**Data-integrity catch:** the first live (Monitor) read of micro=8 was POLLUTED
(41.55GB/3194ms) by a brief GPU overlap during the sweep tail; the authoritative
results.jsonl and an isolated idle-GPU re-run both gave 46.96GB/3834ms bit-for-bit.
Trusting the polluted read would have reported "micro=8 fastest" — the opposite of truth.
Lesson recorded: peak_mem/step_time only valid on an otherwise-idle GPU; verify surprises
with an isolated re-run.

**Infra correction:** the box's `git pull` WORKS now (the old "gh token invalid, scp only"
note from the 2026-06-22 A2 entry is stale). The user pulled successfully this session; the
only friction was untracked a02/a03/a04 dirs on a 28-commit-stale box HEAD colliding with
what origin already had. Resolved by `git reset --hard origin/main` + `git clean` (after
backing up the box-only `.log` run logs), restoring box to a clean mirror at `aab1eec`.
A4 was still scp'd this session because it wasn't committed yet — once committed, the box
can just `git pull` it.

**Next:** A5 — activation checkpointing + seq_len sweep. Payoff co-designed on the day.

### 2026-06-23 — Track D Day 5 done (permission pipeline, Astra PR #6 merged) + resolved a parent-repo merge conflict

**D5 — permission pipeline.** A tool that can run `rm -rf` needs a gate before
execution. Built layers 1/2/5 of CC's 7-layer model, wired into `RunOneToolAsync`
BEFORE `ExecuteAsync` so a denied side effect never happens. Tutor mode — the user
set two design constraints (class-based default "Y" not always-ask; and "make it
highly modular for SDK release"), which shaped the interfaces; I wrote the code.

Per the post-D4 process fix, D5 **started with a source read**
(`d05-permission-pipeline/source-reconciliation.md`, written before coding) of CC's
`utils/permissions/*`. That read gave the design directly: 3-state decision,
deny>ask>allow, default=ask, the `passthrough`/`NoOpinion` boundary, and CC
complexities to skip.

What shipped (Astra `e3b52a6`, PR #6):
- `src/Astra.Core/Permissions/`: `PermissionDecision` (Allow/Deny/Ask — 3-state,
  never a bool; Ask is a suspension that calls a human BEFORE the side effect),
  `PolicyVerdict` (+NoOpinion at the policy boundary only), `PermissionRule`
  (toolName + optional command prefix, deny>ask>allow), three interfaces, and the
  defaults `ClassDefaultPolicy`/`AlwaysAskPolicy` + `DefaultPermissionEngine`.
- Modularity (the SDK requirement): `IPermissionPolicy` (WHAT — primary swap point,
  `ValueTask` so a host can do I/O without sync-over-async deadlock),
  `IUserConfirmation` (HOW an ask resolves — the only async step), `IPermissionEngine`
  (orchestration, escape hatch). Independent, composable.
- `AgentLoop` gains optional `IPermissionEngine?` (null = unguarded, backward-compat
  — the pre-D5 38 tests unchanged). Deny short-circuits: reason → LLM tool_result +
  `AgentEvent.ToolDenied`. Allow may rewrite args.
- Two fail-closed points clarified: Classify (D2) → unknown tool = `Other` (strictest
  class); rule-match (D5) → unpoliced call = `Ask` interactive / `Deny` headless.
- Tests +16 (**54/54**): policy decision table, engine 3-state + headless deny, and
  the load-bearing `DeniedCall_ToolNeverRuns` (a SpyTool asserts Executions==0 — the
  side effect provably did not run). Notes + teaching-notes (why 3-state, the two
  fail-closed points) in `agent/experiments/d05-permission-pipeline/`.

Deferred (cited): full InputSchema validation (Layer 1 is tool-existence only),
wildcard/exact rule matching + source tiers, per-tool checkPermissions veto, a CLI
confirmation UI (`IUserConfirmation` has no terminal impl yet — CLI still builds the
loop with no engine), and permission modes (bypass/acceptEdits/plan/dontAsk).

**Merge conflict resolved.** Pulling main brought 7 commits of parallel Track A
work (A2/A3 + an A4 cross-machine handoff) done on another machine. Only
`notes/progress.md` conflicted — append-only LOG, both machines added top entries.
Resolved by keeping BOTH (Track A and Track D entries), deleting the markers; the
LEARNING STATE table auto-merged cleanly (A row + D row changed independently). No
content lost, no duplicate entries. The D5 work was uncommitted at the time and
stayed out of the merge commit (`d574188`), then was committed separately as above.

**Next:** D6 — context assembly (three-layer cache strategy). OR finish a deferred
thread: D5's CLI confirmation UI + InputSchema validation, or D4's control-plane
(mid-turn interrupt as an abort-reason branch, per d04 source reconciliation).

### 2026-06-22 — D4 Linux-verified on WSL; fixed a Linux-only D2 streaming race; D4 source-of-truth reconciliation

Three connected things this session, prompted by the user's (correct) challenge:
"is Track D actually following the Claude Code source, or are you winging it?"

**1. D4 verified on real Linux.** The D4 tree-kill test is POSIX-only and had
never executed its core assertion (Windows early-returns; GX10 was unreachable —
`192.168.1.200` SSH timed out). This machine turned out to have a WSL Ubuntu 24.04
(x86_64) distro. A background agent installed .NET SDK 10.0.301 (official
dot.net install script, user-local `~/.dotnet`), copied Astra to the native FS
(`~/astra-verify`, NOT the 9p `/mnt` mount), and ran the suite. Result: **38/38 on
Linux**; `KillsWholeTree_GrandchildStopsTicking` ran its real body (~1 s) and
passed — the reparented grandchild stopped ticking after the kill, proving the
whole process tree dies. D4 gate in d04 notes is now CLEARED. (WSL is the standing
Linux target for this env since the GX10 is unreachable here — `wsl.exe -e bash
-lc '...'` from the Bash tool; install dotnet per-distro as above.)

**2. Fixed a Linux-only D2 streaming regression (Astra PR #5, `11b6aed`).** The
same Linux run surfaced a real bug: `ExecuteAsync_StreamsProgressThenSingleResult`
failed 3/3 on Linux (passed on Windows). Root cause: `BashTool` completed the
output `Channel` in the `Process.Exited` handler — a race against the threadpool
pumping `OutputDataReceived`. For a fast-exiting command (`printf`), `Exited` won,
`TryComplete()` closed the channel, and the data lines' `TryWrite` calls dropped
silently → zero Progress. Windows timing hid it. Fix: complete the channel when
BOTH stdout+stderr hit EOF (each fires once with `e.Data == null`), tracked by an
`Interlocked` countdown — decoupled from exit, no data dropped. Re-verified 38/38
on Linux (streaming 3/3, cancellation unregressed). Lesson: "streaming done" (D2)
was only ever Windows-tested; the contract had never actually held on Linux.

**3. Source-of-truth reconciliation — a real process miss on D4.** The user was
right. Honest grading of how closely each day tracked `refs/claude-code-sourcemap`:
D3 genuinely traced the source (`partitionToolCalls` fold, confirmed); D2 read it
and deliberately diverged (behavior-class vs command-string, with rationale); **D4
I built from .NET first principles and did NOT read the source first**, contrary
to the repo's source-tiering rule. Reconciled after the fact (in d04 notes):
CC kills the tree too (`tree-kill` + `detached:true` process-group in
`ShellCommand.ts`/`bashProvider.ts`) so the *conclusion* matched — but by luck,
not method, and I missed a real design point: **CC's `#abortHandler` branches on
the abort *reason*** — a real cancel kills the tree, but an `interrupt` (user typed
mid-turn) does NOT kill, it backgrounds the process so the model keeps the partial
output. That is d02's scenario 1 + scenario 2 as **two branches of one signal**,
not two separate mechanisms. Carry-forward: model cancellation as a signal with a
reason (`Cancel` vs `Interrupt`) when the deferred control-plane is built.
**Process fix going forward: every Track D day writes a "Source-of-truth
reconciliation" section BEFORE coding** (grep the sourcemap, state what CC does and
where Astra agrees/diverges and why), not after.

**Next:** D5 — permission pipeline (Layer 1 schema validation + Layer 2 rule
matching policy>user>project>session, fail-closed + Layer 5 one confirmation hook).
Builds on D2's `Classify`. Per the process fix above, D5 starts with the source
read of `architecture/07-permission-pipeline.md` + CC `hooks/` + the permission flow.

### 2026-06-18 — Track D Day 4 done (control layer: process-tree kill on cancel, Astra PR #4 merged)

curriculum's nominal D4 (streaming + tool_use) was already pulled forward into
D2, so the real D4 work is the **control layer** the d02 notes parked under "OPEN
for D4". This day did its foundational piece: making "stop" actually stop the
work. Tutor mode — the user reasoned out both facts (Dispose sends no signal;
the child is a shell so the work is in its descendants) and chose the finally-
over-ct.Register design; I wrote the code.

**The hole closed:** pre-D4, `ct` was wired only to *reading* the child output.
On cancel, OCE unwound and `using var process` ran `Dispose()` — which frees the
handle and **sends no signal**, so the spawned `sh -c "..."` + npm/node/...
descendants were reparented to init and kept running. "Stop" = "I stop watching".

**What shipped (Astra `0117e60`, squash-merged PR #4):**
- `src/Astra.Core/BashTool.cs` — drain + WaitForExit now wrapped in
  `try { } finally { await KillTreeAsync(process); }`. The finally fires on every
  exit path: normal completion (no-op), cancellation (OCE), AND consumer break
  (`await foreach` calls the iterator's DisposeAsync, which runs the finally).
  try/finally with NO catch is legal around `yield` (CS1626 bans only catch).
  Chosen over `ct.Register` (which would risk Kill on a disposed Process + need
  its own registration lifetime). `KillTreeAsync` = `Kill(entireProcessTree:true)`
  (the child is the SHELL; a bare Kill orphans npm) + swallow TOCTOU
  InvalidOperationException + reap with `WaitForExitAsync(CancellationToken.None)`
  (caller's token is already cancelled; passing it would return before the tree
  finished dying).
- `tests/Astra.Core.Tests/BashToolCancellationTests.cs` (+2): GrandchildStopsTicking
  (by-construction — a backgrounded grandchild subshell stops appending to a
  marker file after cancel; a bare-Kill impl leaves it ticking; a timing test
  could not catch the original bug. POSIX-only, early-returns on Windows) +
  ThrowsPromptly (cross-platform cancel-path guard). **38/38 pass on Windows.**
- `agent/experiments/d04-control-layer/{teaching-notes,notes}.md` — the
  Dispose-sends-no-signal / shell-wraps-the-work / finally-not-Register / two-races
  derivation, + scope and the gate below.

**⚠️ VERIFICATION GATE (D4 is "implemented, Windows-green, Linux-pending"):**
the tree-kill assertion only executes on Linux/macOS (POSIX sh subshell
semantics; Windows uses cmd.exe and early-returns). This session had NO reachable
POSIX box — GX10 `192.168.1.200` SSH **timed out**, no WSL distro, no Docker
daemon. Before claiming D4 verified, on the GX10 once reachable:
```
cd <Astra checkout> && dotnet test Astra.slnx --filter "FullyQualifiedName~BashToolCancellation"
```
Expect GrandchildStopsTicking to actually run (not early-return) and pass. Then
record the green result here and drop the gate.

**Deferred (rest of the control layer):** scenario 1 (inject user input mid-turn
— back-channel + soft/hard restart + split AgentApp's serial foreach→ReadLine),
scenario 3 policy half (a middleware watching Progress that decides cancel+kill —
the mechanism it calls is now done), and `contextModifier` (cd-changes-cwd, serial
path only). All in d04 notes "Open / deferred".

**Next:** D5 — permission pipeline (layered, fail-closed) per curriculum-agent.md;
OR circle back to finish the deferred D4 control-plane. User's call.

### 2026-06-22 (later) — A3 finished to spec; HANDOFF for "continue A4" on a new machine

The learner is switching to a different computer and will open a FRESH Claude Code
session, say "continue learning A4," and expects it to pick up seamlessly with full
memory of their learning history. This entry is the handoff. **New session: read
this whole block before teaching A4.**

**A3 is now genuinely done** (not just "eval ran"): the learner read all 10
before/after pairs themselves and wrote their own observations — sharper than the
tutor's earlier take. Their conclusion: *"SFT made answers more concise, but it
loses content, and instruction-following got worse."* They independently (a) caught
that #8/#10 show degraded instruction-following, (b) asked whether the cut-offs were
script or model (answer: the `max_new_tokens=120` cap — a real experiment-improvement
catch), and (c) learned SFT = Supervised Fine-Tuning on paired instruction→answer
data, traced the behavior back to the training set (`yahma/alpaca-cleaned`, 500 of
51,760 examples, terse list-style outputs → model over-fit "be concise"). All in
`experiments/a03-eval-1b/results.md`.

**How to resume A4 (do this in order):**
1. Read `experiments/a02-sft-1b/learning-notes.md` IN FULL — especially the
   **learner diagnostic** at the very end (strengths, recurring weak spots, the
   proven teaching method). This is the single most important file for teaching this
   learner well. Don't skip it.
2. A4 day spec: `notes/curriculum-v2-execution.md` § A4 (gradient accumulation;
   3B model, seq=1024, three configs micro/accum = 1/16, 4/4, 8/2, all effective
   batch 16; deliverable = table of peak_mem/step_time/final_loss; the payoff is
   "same effective batch → near-identical loss, different memory/speed").
3. **Teach in TUTOR mode, learner-paced:** one small segment, let them clarify in
   place, fold their Q&A into a NEW `experiments/a04-grad-accum/learning-notes.md`
   (same per-learner format as A2's). **Do NOT rush to run code** — the learner
   explicitly wants to READ and understand the code before any run.
4. **A4 is where A2's hidden loop gets unrolled.** The learner couldn't fully read
   A2's `train.py` because `trainer.train()` hides the four beats (forward/loss/
   backward/optimizer). They CHOSE to defer that to A4, where gradient accumulation
   forces an explicit training loop. So when teaching A4, show the explicit loop and
   map each line to the four beats they already learned — then A2's train.py makes
   sense in hindsight. (This is an owed debt: "A2 train.py walkthrough, deferred to A4.")

**Learner profile (full version in learning-notes.md diagnostic):** strong systems/
precision instinct (independently re-derived 8-bit Adam and loss scaling); asks "why
this design not that"; learns from concrete numbers + linear-regression anchors, not
abstractions. Recurring weak spots: inverts containment direction (umbrella-vs-kind,
hit 3×) and fuses nested scales (neuron-vs-layer) — state direction and scale
explicitly. Background concepts present but fuzzy (2015 Andrew Ng). Terminal does NOT
render LaTeX sub/superscripts — write math as code (`y[i]`, for-loops), never as
rendered subscripts. Conversation in 中文, all ML terms in English (bias/weight/
gradient/...), never translated.

**Env reminders for the new machine:** GX10 SSH needs the NVIDIA Sync key — set up a
`Host GX10` block in `~/.ssh/config` pointing at the `nvsync.key` (see the 2026-06-17
A2 log entry for the exact path/pattern) and use `ssh GX10`. The box's `gh` token is
invalid, so `git pull` fails ON THE BOX — scp scripts to it, or re-auth gh there. The
A2 checkpoint lives on the box at `~/runs/a02-sft-1b/`.

### 2026-06-22 — A2 taught properly (per-learner learning-notes, Seg 1-6e) + diagnostic

The user pushed for A2 to be re-taught their way: one small segment at a time, they
clarify in place, each explanation + Q&A folded back into a per-learner note. This
produced `experiments/a02-sft-1b/learning-notes.md` (~720 lines) — the new
"learning note" type (CLAUDE.md): complete, dialogue-shaped, depth set by THIS
learner's familiarity, math written terminal-safe (code, not LaTeX subscripts).

Segments: 1 (what A2 does / fine-tuning = adjust existing model), 2 (parameter is
the umbrella; weight/bias are kinds), 3 (12 B/param is storage cost, orthogonal to
kind), 4 (text->token->ID->logits->softmax->probs), 4b (activation functions +
the "why not x^2" question -> universal approximation), 5 (cross-entropy collapses
to -log(p)), 6a/6b (backward: per-param gradient, chain rule, why it flows
loss->input), 6c (AdamW m/v = A1's 8 bytes, full loop closes), 6d (fp16/fp32/bf16
deep dive: why m/v need fp32), 6e (one neuron = n weights + 1 bias, each an
independent parameter with its own gradient/m/v).

Highlights: the learner independently re-derived **8-bit Adam** and **loss scaling**
from systems instinct, and articulated the general rule "low precision is fine for
used-then-discarded quantities (gradient/activation), not accumulated ones (m/v)."
Their extension questions exceeded the core lesson in value.

Added a **learner diagnostic** at the end of learning-notes: strengths (systems/
precision instinct, asks "why this design not that", self-corrects from anchors),
recurring weak spots (umbrella-vs-kind direction inversion — hit it 3x; fused
nested scales like neuron-vs-layer; 2015-Ng concepts fuzzy/half-swapped; can read
Σ but terminal won't render it), and the proven teaching method. Future sessions
should read this before teaching this learner.

No new GX10 runs this session — pure teaching off the existing A2 checkpoint and
A3 results. A2 (run + payoff + lesson) is now fully complete. Next curriculum day:
A4 (gradient accumulation), payoff co-designed on the day.

### 2026-06-17 — A3 done as the A2 payoff; "visible reward per day" rule added

A2 was technically complete (a fine-tuned checkpoint) but the user pushed back hard
and correctly: they learned the loop yet never *saw* the model behave differently,
so the day failed as teaching — "no reward, I can't keep learning." Fixed two ways:

1. **Ran A3 immediately as the A2 payoff** — `experiments/a03-eval-1b/compare.py`:
   base Llama-3.2-1B-Instruct vs the A2 checkpoint, same 10 prompts, greedy. The
   diff is visible and instructive: SFT drops the "Here are…" preamble, is more
   on-task/concise, and **knowledge is unchanged** (#5 capital-of-France byte-
   identical) — the textbook "SFT changes format/style not facts." `results.md`
   (10 pairs) committed; saved on box at `~/runs/a03-eval/results.md`.
   (One gotcha: a too-narrow monitor grep made the live output look empty; the
   generations were there all along — check `~/runs/...` host path, not `/runs/...`.)

2. **CLAUDE.md: new hard rule "Every day must pay off in something the learner can
   SEE."** Plus a teaching gap surfaced about curriculum design: the *structure*
   (every day has a payoff section) is fixed, but the *specific payoff is decided
   live with the learner*, not pre-written — the A2 payoff was right because it
   was generated in response to the user, which a static syllabus can't do. Rule
   text says exactly this.

Next: A4 — gradient accumulation (`experiments/a04-grad-accum/`). Its payoff (to be
co-designed on the day) is likely "three configs, same effective batch → same loss,
different memory/speed" shown as a table the user reads themselves.

### 2026-06-17 — Track A Day 2 done (first full-parameter SFT) + A1 memory correction

First real fine-tune on the GX10. Full-parameter SFT (LoRA removed) of
Llama-3.2-1B-Instruct, `experiments/a02-sft-1b/` (train.py + run.sh).

Result: 500 alpaca-cleaned examples, 1 epoch, batch=4 seq=1024 bf16 lr=2e-5 cosine.
125 steps in 82.9 s (1.51 steps/s), loss 1.72 → final train_loss 1.478 (noisy, small
data). 1.236B/1.236B trainable confirmed. Model saved to `~/runs/a02-sft-1b/`
(model.safetensors 2.47 GB = 1.236B × 2 bytes bf16, exact). Loss started ~1.7 not
~2.5 because the **-Instruct** base is already tuned — not a cold start.

**Memory finding (A1 → A2 closed loop, corrects A1).** A1 predicted 16 B/param
(mixed-precision Adam + fp32 master) ≈ 18.4 GB. **Measured peak 13.84 GB**, which
matches the **12 B/param** recipe (pure bf16: w2 + g2 + fp32 m,v 8 = 12 → 13.81 GB)
to 0.2%. Conclusion: **HF `Trainer(bf16=True)` keeps NO fp32 master weight** — it's
12 B/param, not 16. The 16 figure is DeepSpeed/FSDP mixed-precision. Corrected
`a01-mem-budget/notes.md` + `teaching-notes.md`; `budget.py` already supported it via
`master_dtype=None`. So for HF-Trainer full SFT, use **12 B/param**; treat 16 as the
DeepSpeed/FSDP upper bound.

**Infra notes for next session:**
- **SSH changed.** Password-less `ssh hooyao@192.168.1.200` no longer works
  (Permission denied, publickey/password). The working path is the **NVIDIA Sync**
  key: host alias `GX10` in `~/.ssh/config` includes
  `C:\Users\yahu2\AppData\Local\NVIDIA Corporation\Sync\config\ssh_config`
  (Hostname 192.168.1.200, User hooyao, IdentityFile `…\Sync\config\nvsync.key`).
  Git-bash ssh doesn't parse that Windows-path `Include`, so a `Host GX10` block was
  added directly to `~/.ssh/config` pointing at `nvsync.key`. **Use `ssh GX10`** (or
  `-i …/nvsync.key`). Driver now reports **595.58.03** via CUDA forward-compat
  (kernel driver still 580.159.03).
- **GX10 git is broken for pull.** Remote is HTTPS and the box's `gh` token is
  **invalid** (`gh auth status` → "token in default is invalid"), so
  `git pull` fails with `could not read Username for https://github.com`. Worked
  around by `scp`-ing the A2 scripts directly. To fix properly: on the box run
  `gh auth login -h github.com` (interactive — user must do it), or switch the
  remote to SSH. Until then, scp new scripts to the box.
- Container files in `~/runs/a02-sft-1b/` are `root:root` (docker runs as root).

Next: A3 — generation quality before vs after SFT (`experiments/a03-eval-1b/`),
peer mode. Uses the saved `~/runs/a02-sft-1b/` checkpoint vs the base 1B.

### 2026-06-18 — sourced the 2150 MHz clamp + measured what capping costs

Two-part follow-up to the 2026-06-05 emergency playbook, which had recorded
`nvidia-smi -lgc 200,2150` as a community mitigation but flagged the 2150 value
as "NO SOURCE — no authoritative number."

**Part 3 (added later same day) — sustained sweep + community cross-check.**
Re-capped to 2150 and ran a 60s/size sustained sweep (`--sizes 4096,8192,12288,
16384 --duration 60`). Clock pinned at 2132 MHz all 4 minutes, zero jitter, no
shutdown. Sweet spot is **N=12288 → 95.7 TFLOPS** (4096=83.5, 8192=93.0,
16384=94.7 — 16384 drops back as it goes LPDDR5x-bandwidth-bound, p99 jumps
39→106 ms). Long-run 8192 = 93.0 vs short-run 92.9 → **the 2150 clamp has zero
sustained-duration decay**, which is its real benefit (kills the SW Power Cap
jitter). Net: capped best (95.7 @ 12288) is **98.9% of uncapped best** (96.8 @
8192) — the clamp is near-free.
Community cross-check (verified, see hardware-gx10.md): DGX Spark BF16 numbers
span 3 non-comparable tiers — mmapeak ~213 (raw MMA, not real GEMM), real
cuBLAS GEMM ~45–96 (our tier; Carmack ~60 is early-firmware+power-capped),
broken-stack ~11. **Our 95–96 is the top of the real-GEMM tier and correct**
(26.04 container = right Blackwell kernels + mature firmware + best size).
Recorded both as this unit's BF16 characteristic + the clamp best-practice in
`notes/hardware-gx10.md`. Clamp best-practice framing: 2150 is **near-free
stability insurance (~1% throughput), NOT the cause of the 95 number** — don't
conflate them. Machine reset to default (3003 max) after.

**Part 1 — found the source.** The 2150 MHz number traces to two real,
verified community artifacts (not official NVIDIA docs):
- `github.com/eugr/spark-vllm-docker` README → Known Issues (2026-03-17 entry):
  firmware "may cause sudden shutdown event on one or both Sparks during heavy
  inference"; workaround `sudo nvidia-smi -lgc 200,2150`; notes default 2411,
  boost 3000; **the lock only survives until reboot**.
- NVIDIA forum thread "DGX Spark (GB10) reproducibly hard powers-off under GPU
  load — fully updated, zero crash capture" (t/373251). Real symptom (hard
  power-off ~60s into GPU load, no pstore/vmcore/Xid). BUT: the elaborate
  "PMIC hard cutoff / SoC spikes 100°C in microseconds / engineering consensus
  fixed 2150 as the physical limit" narrative the user was shown is **embellished**
  — no NVIDIA staff in the thread (all users), "PMIC" never appears, SoC>100°C
  is one user's speculation, and the actually-confirmed fixes were repaste
  (dry/brittle TIM) + case-off + 120mm fan. Clamping is one workaround, not a
  proven physical limit. Note also: that thread says default boost 2411; **this
  ASUS unit's max is 3003** (confirmed again today), so even the "2411 default"
  in the source doesn't match this SKU.

**Part 2 — measured the cost of the clamp.** Ran `bf16_peak.py --sizes 8192
--iters 200` (single size, safe profile — sustained `--duration` is the exact
shutdown trigger, deliberately avoided) capped vs uncapped, container 26.04:

| Round | clock policy | loaded gclk | sustained TFLOPS | temp | GPU pwr |
|---|---|---|---|---|---|
| A | `-lgc 200,2150` | **2138 MHz** (locked, verified) | **92.9** | 51 °C | 71 W |
| B | `-rgc` default | **2366 MHz** (never hit 3003 cap) | **96.8** | 56 °C | 94 W |

- **Cost of the 2150 clamp ≈ 4% BF16 throughput** (96.8 → 92.9) for ~24% less
  GPU power (94 → 71 W) and 5 °C cooler. Cheap insurance if shutdowns appear.
- Key observation: **even uncapped, the GPU self-limited to 2366 MHz** under
  BF16 GEMM — nowhere near the 3003 hw cap. Consistent with the SW Power Cap /
  shared-140W-SoC-envelope story from 2026-06-05. The clamp removes the top
  ~230 MHz of a range the chip wasn't using at full load anyway, which is why
  the throughput hit is small.
- **Caveat carried forward:** this measures the *throughput cost* of the clamp,
  NOT whether the clamp prevents shutdown. The forum thread's stronger root-cause
  candidate is SoC/CPU-side heat (GPU 79°C while CPU ~96°C); lowering GPU clock
  only reduces total SoC power indirectly. Unverified on this unit — no shutdown
  has ever been observed here (now 2 weeks uptime across sessions).
- Machine left in clean state: `-rgc` applied, max back to 3003, idle 208 MHz.

Connection gotcha for next session: `ssh hooyao@192.168.1.200` from this Windows
laptop fails (`Permission denied`) — the default `~/.ssh/` has **no private key**.
SSH is managed by NVIDIA Sync: key is `C:\Users\yahu2\AppData\Local\NVIDIA
Corporation\Sync\config\nvsync.key`, Host alias `GX10`. Git Bash doesn't parse the
spaced-path `Include` in `~/.ssh/config`, so the alias doesn't resolve — connect
explicitly with `ssh -i "<nvsync.key path>" hooyao@192.168.1.200`.

### 2026-06-17 (later) — Track D Day 3 done (tool orchestration, Astra PR #3 merged)

D3 implemented in the Astra submodule. Tutor mode: the user supplied the framing
("this is compiler instruction reordering") and reviewed; I wrote the code.

**What shipped (Astra working tree, detached HEAD at `0488676` = D2):**
- `src/Astra.Core/ToolBatching.cs` (NEW) — pure `Partition(calls, classify) ->
  List<ToolBatch>`. Stable partition (coalesce adjacent reads), NOT a sort.
  Concurrency-safety derived from D2: `safe == (Classify == ToolAction.Read)`.
  `Write/Execute/Other` are barriers (each runs alone). Fail-closed: unknown tool
  -> `Other` -> barrier.
- `src/Astra.Core/AgentLoop.cs` (REWRITE of the tool-dispatch section) — D2's
  serial `foreach (call in toolCalls)` replaced by `foreach (batch in batches)`.
  Concurrent batch = N producer tasks fan in through one `Channel<AgentEvent>`,
  single-reader iterator drains+yields (keeps `yield` out of try/catch, CS1626).
  Bounded by `SemaphoreSlim(MaxConcurrentTools=10)`. Serial batch = same path,
  width 1. Results keyed by `CallId` into a `ConcurrentDictionary`, fed back to
  `_messages` in the model's ORIGINAL call order (not completion order).
- Tests: `ToolBatchingTests.cs` (7) + `AgentLoopOrchestrationTests.cs` (3).
  **36/36 pass** (D1 x2, D2 x25, D3 x9). Load-bearing ones:
  `Partition_ReadWriteRead_DoesNotHoistAcrossBarrier` (the trap: write splits two
  reads into 3 batches) and `TwoReads_RunConcurrently` (proves real overlap via a
  rendezvous latch — serial execution would time out, not silently pass).
- `agent/experiments/d03-tool-orchestration/teaching-notes.md` — the hazard /
  instruction-scheduling derivation (RAR is the only safe reorder; no alias
  analysis over the filesystem -> every write is a fence; emission order is
  program order -> stable partition, never sort). This is the D3 conceptual core.
- `agent/experiments/d03-tool-orchestration/notes.md` — design/impl record.
- Astra `CLAUDE.md` — orchestration section updated from "will partition (not yet
  implemented)" to the implemented contract; assembly-pipeline marked D15.

**Scope decisions (user-confirmed):**
- Assembly-sort (built-in prefix / MCP suffix for prompt-cache) DEFERRED to D15 —
  no MCP tools exist yet, nothing to sort. Pointer left in d03 notes + Astra
  CLAUDE.md. D3 = runtime tool-call partition only.
- `ToolOutput` Progress/Result typing unchanged; D3 only fans in N such streams.

**Committed:** Astra PR #3 squash-merged to main (`9ac91aa`), branch deleted;
parent repo submodule pointer bumped `0488676` -> `9ac91aa` together with the
`agent/experiments/d03-*` notes and this progress.md update. The pre-existing
1-line `BashTool.cs` D4-kill TODO comment went in with the D3 commit (harmless,
marks the D4 spot). Flow was identical to D1 #1 / D2 #2; merge needed `--admin`
(base-branch policy gate, no CI checks configured) — same as the prior days.

**Then D4** — streaming/control layer. The queue is already written in
`agent/experiments/d02-tool-contract/notes.md` under "OPEN for D4": (a) process-
TREE kill on cancel (`process.Kill(true)`; the `BashTool.cs` TODO comment marks
the exact spot — today's ct-threading makes the cancel path reachable but it still
only stops *watching*, not the process), (b) bidirectional interruption — 3
scenarios (mid-turn user input injection, "stop" mid-tool, agent-autonomous
kill-on-error), all needing the one-directional loop to gain a back-channel
(CancellationToken + a Channel/queue). Also: `contextModifier` (cd-changes-cwd)
NOT built — when a stateful tool appears, its modifier applies ONLY on the serial
path, never concurrent (see d03 notes "Open / deferred").

**Env notes (D3 commit session):**
- Edit/Write worked fine this session; the D2-session multi-line parse failures
  did not recur. (If they do: fall back to `python - <<EOF` via Bash.)
- **CRLF gotcha when committing in Astra:** `core.autocrlf=true` + `safecrlf=true`
  + `.gitattributes eol=lf` means a file edited on Windows (CRLF in the working
  tree) aborts `git add` with "CRLF would be replaced by LF". Fix: `git add
  --renormalize <file>` to apply the repo's LF normalization, then commit. Hit
  this on `CLAUDE.md`; expect it on any Windows-edited tracked text file in Astra.
- Astra merge needs `gh pr merge N --squash --admin` — base-branch policy gate
  blocks plain merge and there are no CI checks to satisfy.

### 2026-06-17 — Track A Day 1 done (memory budget calculator) + two teaching notes

First model-side curriculum day executed. This machine (Windows laptop) is now the
**model-side** workstation; Track D (agent/Astra) is being done on a *separate*
machine — the two legs share this repo but don't touch each other's files (today's
A1 work and the same-day D2 push from the other box rebased cleanly).

Mode: **hybrid** (user's call) — tutor on conceptual days (A1/A6/A9), peer on the
mechanical run-and-read days. A1 is the one theory day, so tutor. It surfaced two
prerequisite gaps the day-plan assumed but didn't teach; both were explained from
scratch and persisted (this is the "teaching notes" practice now in CLAUDE.md):

- `experiments/a01-mem-budget/teaching-notes.md` — what a parameter physically is
  (a number in a bag of numbers), a 9-param toy model with a traced forward pass,
  matrix form = the GEMM the GB10 runs, scaling to Llama-3B, then the bytes/param
  bridge. Corrected in-place: full-SFT Adam is **16 B/param** (adds fp32 master),
  not the 12 used to build intuition.
- `experiments/a01-mem-budget/backprop-primer.md` (+ `.zh.md`) — the user had
  forgotten backprop entirely; this is a self-contained primer (chain rule → fully
  worked 2-layer backward with numbers → two-kinds-of-gradient → per-layer
  activation rule → tie-back to grad/activation memory → 30-line numpy) to read
  alongside a YouTube lecture as offline study. The `.zh` is a one-off Chinese
  exception the user explicitly requested; ML/systems terms stay English.

Deliverable: `budget.py` — `param_bytes` / `optimizer_bytes` (+ `training_state_bytes`,
`activation_bytes`) + a 1B/3B/8B × full-SFT/LoRA × checkpointing table with
comfortable/marginal/OOM verdicts, and a `--test` mode asserting against
`notes/curriculum.md` worked examples. Pure stdlib, no GPU. **GX10 was offline
(ssh timeout) when written, so `--test` was hand-verified** (3B full SFT 51.2 vs 51,
8B 128 vs 128, 14B LoRA 29.6 vs 30, 32B FP8 LoRA 32.8 vs 33 — all <2% off). Re-run
`python budget.py --test` on the box to confirm. Verdicts in `notes.md`: 1B/3B full
SFT + all LoRA comfortable; 8B full SFT marginal (needs 8-bit Adam + ckpt); ≥14B
full-SFT will OOM (LoRA/QLoRA/FP8 only).

Also this session: pinned the **no-translate rule** into CLAUDE.md directive 2
(never render ML/systems terms — bias/weight/gradient/... — in Chinese, the user
finds the renderings harder to read), and added a **"Teaching notes" subsection**
under tutor mode making these gap-notes a first-class, committed deliverable, with
a README pointer. Commits: `3bf63f1` (teaching-notes + practice), `654886b`
(backprop primer + no-translate), plus this session's budget.py/notes commit.

Next: A2 — first full-parameter SFT on Llama-3.2-1B (`experiments/a02-sft-1b/`),
peer mode. Needs the GX10 back online.

### 2026-06-17 — Track D Day 2 done (tool contract + streaming, Astra PR #2 merged)

Implemented the D2 contract decided in the prior design exploration, then —
prompted by user review — also brought streaming forward from D4.

What shipped (Astra `0488676`, squash-merged PR #2):
- **`ITool.Classify(arguments) → ToolAction {Read,Write,Execute,Other}`.**
  Input-dependent (`BashTool`: "ls" → Read, "rm -rf" → Execute), fail-closed
  default `Other` via a **C# default interface method** (not a base class — D1
  rejected tool inheritance). Replaces the crude 3-bool design. Key lesson
  (verified against CC source `bashPermissions.ts`/`readOnlyCommandValidation.ts`):
  CC keys bash approval on the command **string** (exact/prefix), so a small arg
  change misses the saved rule and re-prompts — the failure that drives users to
  bypass permissions. Classifying by behavior **class** lets a host approve "all
  reads" once. Two-layer split recorded for the permission day: class decides
  bulk, a deny-list snipes per-command exceptions.
- **Streaming `ExecuteAsync` → `IAsyncEnumerable<ToolOutput>`** with
  `ToolOutput.Progress` (live, for the human) vs `ToolOutput.Result` (the one
  complete tool_result, for the LLM; need not equal the Progress concatenation).
  Was `ValueTask<string>` (blocks until exit). `BashTool` bridges `Process`
  output events into the stream via `System.Threading.Channels`. `AgentLoop`
  drives the tool enumerator by hand (yield can't sit in try/catch, CS1626) and
  emits `AgentEvent.ToolProgress`. 27 tests pass.
- Astra `CLAUDE.md` Tool System section synced from the stale generic-3-bool
  draft to the implemented contract.

**OPEN for D4 (do not forget) — 3 interruption scenarios the user raised**, full
detail in `agent/experiments/d02-tool-contract/notes.md` ("OPEN for D4"):
1. mid-turn user wants to add input (loop is one-directional; needs concurrent
   input-listener + back-channel);
2. user says "stop" mid-tool (streaming made the `ct` hook reachable, BUT
   cancelling the read does NOT kill the child process — known hole);
3. agent autonomously watches Progress and kills on error (`process.Kill(true)`
   for the tree + a policy middleware on the Progress stream).
The common thread: turn the one-directional loop into one intervenable in both
directions, carried by `CancellationToken` + a back-channel. This is D4.

Git: Astra PRs #1 (D1) and #2 (D2) both squash-merged to Astra main by the user;
local PR branches deleted, main fast-forwarded to `0488676`. Main-repo submodule
pointer bumped to `0488676`.

### 2026-06-15 — Track D Day 2 design exploration (tool permission contract)

Compared four real agent permission models to design Astra's tool contract,
before writing code (tutor mode — user picks the shape).

What happened:

- **User rejected CC's 3-bool flags as crude.** `isReadOnly` / `isConcurrencySafe`
  / `isDestructive` (Claude Code `Tool.ts:402-406`) can't say "this bash call is
  an *execute*, that one a *write*" as one classification. User wants a
  **category** per call — `Read / Write / Execute / Other` — each mapping to a
  permission level, computed from the call's input.
- **Comparison captured** in `agent/experiments/d02-tool-contract/notes.md`:
  - CC = flags-on-tool + `canUseTool` callback (tool describes, host decides);
    fail-closed defaults via `buildTool` factory. **verified** (read source).
  - Codex = two orthogonal axes, `approval_policy` × `sandbox_mode`, the latter
    OS-enforced (Landlock/seccomp/Seatbelt) — *defense in depth*. **unverified**.
  - OpenCode = per-tool `allow/ask/deny`, bash by command pattern, last-match-wins,
    fail-*open* + deny-list. **captured prior session**.
  - LangGraph = no tool flags; runtime `interrupt()` + `Command(resume=)`,
    resumable HITL — orchestration layer, not a tool contract. **unverified**.
- **Three takeaways for Astra:** keep input-dependence; separate "what the call
  is" (D2, tool classifies) from "what we do about it" (D5, engine decides);
  fail-closed default (follow CC, not OpenCode). Astra's CLAUDE.md mandates
  fail-closed.
- **Direction chosen:** `enum ToolAction {Read,Write,Execute,Other}` on `ITool`
  via **C# default interface method** with `Other` (safest bucket) as the
  fail-closed default — not a base class (D1 already rejected inheritance for
  tools). `Classify(arguments)` computes the category from the call.

Three open decisions blocking implementation (user to answer):
1. default interface method vs abstract base class (and why);
2. does `Classify` take the weakly-typed input bag (needed to demo
   `Classify("ls")==Read` vs `Classify("rm -rf")==Execute`), or stay paramless;
3. demo tools — read+write pair (fixed category) vs single bash (category varies
   by input, better proves "behavioral flags over inheritance").

Git state: Astra `main` = `c9c6760` (D1 code), **ahead of origin/main by 1,
not pushed**. Main repo `019998e` (D1 notes+pointer), clean. D2 notes added this
session (not yet committed). Web-search MCPs (brave / Search-MCP) not mounted this
session and built-in WebSearch is broken → Codex/LangGraph marked unverified;
re-verify when a working web path exists.

Next: D2 impl once Q1-Q3 are answered. Also pending: decide whether to push Astra
to origin/main (currently the local D1 commit only exists on this machine).

### 2026-06-15 — Track D Day 1 done (agent loop in Astra)

First Track D day actually executed (prior session only designed the path).
Environment had no GX10 dependency — Track D is pure C# in the Astra submodule.

What happened:

- **Submodules checked out.** `git submodule update --init --recursive` for
  `agent/refs/{Astra, claude-code-sourcemap, claude-reviews-claude}` (Astra nests
  the two CC refs again; pulled too). `dgx-spark-playbooks` left uninit — not
  needed for Track D.
- **Read the existing loop.** Astra already had `AgentLoop.SubmitAsync()` from
  commit `8b080d8` (so Astra's own `progress.md` is stale — pre-rename, still says
  `MyClaude.*`; real tree is `Astra.Core` / `Astra.Cli` / `Astra.Providers`, no
  tests dir). D1 became: understand the core, fix one gap, add the missing test.
- **`InvokeCoreAsync` → fail-closed.** `ToolAIFunction` adapter was using
  `AIFunction` for *both* advertise (Name/Description/JsonSchema → request, the
  legitimate provider-abstraction use) *and* a dead-code auto-invoke body. Grep
  confirmed no `UseFunctionInvocation`/middleware anywhere, so the body never ran
  — it only exists because `AIFunction.InvokeCoreAsync` is `protected abstract`.
  Changed it to `=> throw new NotSupportedException(...)`: if a middleware ever
  wakes that path it crashes loudly instead of silently bypassing the manual
  dispatch (where D3 partition / D5 permission / D7 compaction will attach).
  This is *why* Claude Code hand-writes the loop instead of using SDK auto-invoke.
- **Tests.** New `tests/Astra.Core.Tests/` (xunit), added to `Astra.slnx`, AOT/trim
  disabled for the test project only. Fakes: `ScriptedChatClient` (stateless,
  decides output from the last message's role — mirrors a real stateless LLM, not
  a turn counter), `TextOnlyChatClient`, `FakeTimeTool`. Two `[Fact]`s, **2/2
  pass**: (1) text-only → terminates in 1 round-trip; (2) tool_use → 2 round-trips,
  event order `ToolUse→ToolResult→TextDelta`, **last event is TextDelta** (the
  load-bearing assertion — catches "stopped too early after the tool result").
- **Source check + notes correction.** Read `query.ts` (`while(true)` at `:307`).
  The skeleton matches Astra's loop exactly. Corrected an earlier wrong note: the
  loop-exit signal should NOT be `stop_reason` — Claude Code's own comment
  (`query.ts:554`) says `stop_reason==='tool_use'` is unreliable; both it and
  Astra exit on *presence of a tool_use block*. The real missing piece is
  `max_tokens` truncation recovery (`query.ts:1188-1256`), which Astra lacks.

Astra working tree is dirty (loop edit + new test project + packages). **Not
committed** — user hasn't asked. When committing: also fix Astra's stale
`progress.md`, then bump the submodule pointer from this repo.

Deliverables: `agent/experiments/d01-agent-loop/notes.md` (full walk-through +
source check), `agent/refs/Astra/tests/Astra.Core.Tests/`.

Next: D2 — tool contract (`ITool<TIn,TOut>` + behavioral flags). Astra's current
`ITool` is the non-generic single-method version; D2 adds `IsReadOnly(input)` etc.

### 2026-06-13 — Track B extended with minimind modern-stack sequel (B13–B16)

Evaluated `jingyaogong/minimind` (verified against the live repo, not just the
README: `model/model_minimind.py` single-file model + `trainer/train_*.py` per
stage). It was **not** previously in any curriculum. Decision: add it as a
4-day sequel to Track B, **after B12**, leaving B1–B11 hand-written days
untouched.

Why it earns a place (vs the existing nanoGPT/TinyStories Track B):

- **Architecture bridge.** B1–B12 build a GPT-2-era model (LayerNorm, learned
  abs pos, MHA, GELU). The model the user actually fine-tunes in Track A is
  **Qwen3-8B** (RMSNorm/RoPE/SwiGLU/GQA). minimind is a from-scratch
  **Qwen3-aligned** impl — closes that mismatch.
- **Framework-free RL.** Track B's PPO/DPO (B10/B11) use TRL wrappers. minimind
  hand-writes PPO, DPO, **GRPO** in native PyTorch — read after the user knows
  the math (C7–C9 + B9–B11).
- **Net-new, resume-relevant:** MoE-from-scratch (lands the long-pending
  "MoE extension" from `curriculum.md`), GRPO/CISPO, distillation, agentic RL.

New days written into `curriculum-v2-execution.md`:

- **B13** — architecture bridge: rewrite the B4 block into RMSNorm/RoPE/SwiGLU/GQA,
  verify against minimind's block.
- **B14** — MoE from scratch: `use_moe=True`, `num_experts=4`,
  `num_experts_per_tok=1`; active-vs-total param accounting + router-balance check.
- **B15** — GRPO hand-written, contrasted with B10 PPO / B11 DPO on one table.
- **B16** — pick one: knowledge distillation (black/white-box) OR agentic RL
  (`train_agent.py` + `rollout_engine.py`, explicitly wired to Track D).

Verified config field names/defaults from `MiniMindConfig`: `use_moe=False`,
`num_experts=4`, `num_experts_per_tok=1`, `hidden_size=768`,
`num_hidden_layers=8`, `num_attention_heads=8`, `num_key_value_heads=4`,
`vocab_size=6400`, `max_position_embeddings=32768`, `rope_theta=1e6`. MoE/size
are config-file edits, **not** CLI flags.

GX10 footprint: minimind-3 (64M) ~0.77 GB full state; MoE (~198M total) ~2.4 GB —
both trivial in the 128 GB pool. Train full data (`pretrain_t2t` 10 GB +
`sft_t2t` 14 GB + RL files, ~25 GB) from HF `jingyaogong/minimind_dataset`, not
the `_mini` subsets. README's "3 RMB / 2 h" is 1-epoch SFT on a 3090; GX10 faster.

Synced downstream refs: `why.md` Track B table (added B13–B16 row; checkpoint
B12→B16), `curriculum.md` (MoE extension now points at B13–B16). Track B title
12→16 evenings. **No curriculum days executed** — this is planning only;
Track B learning state is still "B1 not started."

### 2026-06-06 — project reframed to three legs; Track D (agent engineering) added

**Theme:** the repo is no longer a pure fine-tuning project — it is the workspace
for the user's full AI career transition. Three legs now: (1) model side
(Tracks A/B/C), (2) agentic side (new Track D), (3) career
(`notes/career-transition-research.md`).

What was done this session:

- **CLAUDE.md**: added directive 5 (English is default for everything except the
  user-facing conversation, incl. internal reasoning; flag any 中文 persistence
  for review before writing). Added a "Web Search Tooling" section: the built-in
  `WebSearch` tool is broken on this backend (API Error 400) — use the
  `brave-search` and `Search-MCP` MCP servers instead; `WebFetch` still works.
  Rewrote "Repository Purpose" to the three-leg framing. Extended the Tutor-mode
  trigger to cover `agent/experiments/d\d+-*` and Astra subsystem work. Updated
  the repo-layout block to include `agent/`.
- **New `agent/` folder (Track D — Agent Engineering):**
  - `README.md` — two-halves framing (CC source teaches the agent core's *how*;
    blogs/papers teach the frontier half CC lacks: RAG/eval/memory/interop) +
    the source-tiering rule (judge by whether a source discusses
    cost/latency/context-rot/eval/failure modes, NOT by domain name) + how the
    three submodules combine.
  - `why-agent.md` — motivation, wired to the career research: "AI/LLM Agent
    Engineer" is the most reachable model-facing role and the one direction that
    does NOT make the user give up their systems/infra moat. Maps to specific
    target JDs (MS Applied AI Engineer II, Dublin Agent Cloud, Copilot Tuning).
  - `curriculum-agent.md` — D1–D16, two phases. Phase D-I (D1–D8): re-implement
    the CC agent core in Astra (loop, tools, orchestration, streaming,
    permissions, context, compaction, multi-agent). Phase D-II (D9–D16): the
    frontier half (memory tool, agentic RAG, CRAG, RAGAS eval, LLM-as-judge,
    OTel tracing, MCP/A2A interop, capstone research agent).
  - `research/2026-agent-patterns.md` — the cited research report (see below),
    source of truth for the frontier half.
- **Three submodules under `agent/refs/`:**
  - `Astra` (https://github.com/hooyao/Astra) — the user's C# agent framework;
    **read-write** submodule (develop in its tree, bump the pinned commit like
    `dgx-spark-playbooks`). This is the implementation layer.
  - `claude-code-sourcemap` (https://github.com/hooyao/claude-code-sourcemap) —
    restored TS source of Claude Code v2.1.88 (`restored-src/src/`). Source of
    truth / "how does it actually do X". (Note: an earlier attempt used
    `claude-code-compilable` by mistake; the user corrected it — that repo is to
    be ignored. The swap is done.)
  - `claude-reviews-claude` (https://github.com/hooyao/claude-reviews-claude) —
    17-chapter CC architecture analysis. Teaching layer (read the chapter first).
- **Research basis:** ran the `deep-research` workflow (106 sub-agents, ~8M
  tokens, 5 angles, 25/25 claims verified at 3 votes, 0 refuted) + a targeted
  brave-search/Search-MCP supplemental pass to fill the two gaps the funnel
  missed (Agentic RAG internals, MCP/A2A interop). The verified corpus is
  heavily Anthropic-primary (appropriate for "re-implement a Claude Code-style
  agent"); the multi-agent 90.2% result is internal/self-reported with a token-
  spend confound — recorded as a caveat in the report.

**Not committed.** All of the above is staged in the working tree but not yet
committed (the user has not asked to commit). When committing: the three new
submodule pointers + `.gitmodules` go in the same commit as the `agent/` docs.

**Next:** Track D Day 1 — `agent/experiments/d01-agent-loop/` — implement the
`while(true)` agent loop in Astra (`AgentLoop.SubmitAsync()` →
`IAsyncEnumerable<AgentEvent>`), no tools yet.

### 2026-06-05 (evening) — cold reboot fully recovers BF16 throughput

User ran a cold reboot after the diagnostic chain earlier today. Re-ran
the N=16384 BF16 GEMM benchmark immediately after boot:

| Metric | Yesterday (initial) | This morning (post-uptime) | Tonight (cold reboot) |
|---|---:|---:|---:|
| sustained | 97.3 | 67-70 | **93.4** ✅ |
| peak(best) | 99.9 | 84.9-97.0 | **101.3** |
| GPU power | 89 W | 75-83 W | 89.6 W |
| GPU temp | 79 °C | 63-67 °C | 63 °C |
| idle clock (pre-bench) | unknown | 208 MHz (P8 deep sleep) | 2119 MHz |

**Cold reboot is the actual fix.** `systemctl stop gdm` only recovered
+13 % (67 → 76); the remaining 22 % gap required a full reboot. The
93.4 TFLOPS figure is now reproducible, so the original 97.3 was not a
clean-room oddity but the genuine ceiling for this unit when the EC and
driver power-management state are fresh.

Updated `notes/hardware-gx10.md`:
- Added 4th column to the comparison table (cold-reboot run)
- Rewrote "Revised operational numbers" — `93-97 TFLOPS post-reboot` is now
  the official ceiling, not "the 97 was a clean-room oddity"
- Reordered "Mitigations" — full reboot is now #1, gdm-stop is #2,
  acknowledging measured effectiveness
- Added the `nvidia-smi -q -d PERFORMANCE | grep "SW Power Capping"` rate
  check as a diagnostic to know in advance whether you're in the
  contended regime

**Operational takeaway for the curriculum:** before any benchmark you
actually care about reporting, reboot. For training runs (LoRA / SFT /
QLoRA), this whole story is irrelevant — those don't hit the regime where
the SW Power Cap binds.

### 2026-06-05 (later) — emergency playbook added (clock clamping, EC reset, ASUS-vs-FE firmware)

User received a Gemini-generated "operations instruction set" suggesting:
clock-clamp to 2150 MHz via `nvidia-smi -lgc 1665,2150`, persist via
systemd, and apply `fwupdmgr update` to align to specific firmware
versions (EC 3.3.2, USB PD 0.5.22, SoC 2.152.15).

Verified each claim against NVIDIA docs and community sources:

| Gemini claim | Verdict |
|---|---|
| `nvidia-smi -lgc` works on GB10 | TRUE (tested; needs sudo, not unsupported) |
| EC firmware version 3.3.2, USB PD 0.5.22, SoC 2.152.15 are real | TRUE (NVIDIA DGX Spark release notes June 2026) |
| The 80 W EC death lock | PARTIALLY MISLEADING — community reports 14 W lock, not 80 W |
| 2150 MHz is the correct clamp value | NO SOURCE — community reports clamping helps but no authoritative number |
| 2411 MHz default boost | OFF — this unit's Applications Clock is 2418 MHz, Max is 3003 MHz |
| `fwupdmgr update` is the right path for this unit | NO — this unit is ASUS Ascent GX10 (partner system), NVIDIA explicitly says FE firmware versions only apply to FE |

Decision: do NOT deploy any of these as defaults on this unit. The 14 W
death lock and the sustained-shutdown crashes are real community-reported
issues, but neither has been observed on this unit during 33h of testing.
Recorded as an **emergency playbook** in `notes/hardware-gx10.md` so the
mitigations are available if symptoms appear, without becoming a "just
in case" deployment.

Specifically the playbook covers:
- Symptom triage table (4 distinct failure modes, distinguishing them)
- 30-second power-brick unplug procedure for the 14 W lock
- Clock clamping with `nvidia-smi -lgc`, with the strong caveat that the
  community has no authoritative max value and to descend progressively
- Explicit warning that ASUS GX10 ≠ DGX Spark FE for firmware purposes
- Three-pass verification protocol for any firmware operation (Gemini's
  one good idea was the three-pass gate, kept that)
- Links to NVIDIA forum, spark-doctor CLI, Dre Dyson writeups

### 2026-06-05 (morning, continued) — confirmed industry-wide GB10 SW Power Cap issue

After yesterday's re-measurement showed the 97→67 TFLOPS regression at N=16384,
searched community + NVIDIA forums. Confirmed this is not a per-unit problem:

- NVIDIA forum thread "DGX Spark Performance Degradation - GPU Power Draw
  Issue" has 65+ replies; multiple users report the same `SW Power Capping`
  behavior; persistent open topic.
- Hard version of the issue: GPU locks to 14 W after a crash, even with 96 %
  utilization. Workaround = unplug power brick 30 s to reset the EC controller.
  Multiple threads (March, June 2026).
- A CTO published a detailed write-up after 14 of his company's DGX Spark
  units all showed the same pattern. Recommends treating advertised
  performance as 50% achievable in fleet planning.
- A community CLI tool `spark-doctor` was specifically built to detect this
  ("power.low_draw_under_load" rule is its first detect rule).
- NVIDIA's own 2025-10-31 forum clarification: 140 W TDP is shared by **CPU +
  GPU + memory controller**; `nvidia-smi` shows GPU-only power, which hides
  the SoC-level contention story.

Did a partial mitigation experiment:
- Stopped gdm with `sudo systemctl stop gdm`, closed VS Code Remote-SSH
- Re-ran benchmark at N=16384: **67 → 76 TFLOPS (+13%)**, GPU power 75 → 83 W
- Still 21% below the 97 TFLOPS recorded on 2026-06-04
- Concluded: 97 was clean-room with a fresh uptime; 76 is the realistic
  ceiling with normal dev-machine sessions + dockerd running; the remaining
  gap probably requires a full reboot + minimal-session run, which has zero
  practical value for training work

Updated `notes/hardware-gx10.md`:
- "Mitigations" section now includes measured impact (+13%) of stopping gdm
- New section "This is a known systemic issue, not a per-unit defect" with
  links to the relevant NVIDIA forum threads, CTO write-up, and spark-doctor
- New section "Practical implications for this curriculum" pointing out that
  LoRA / SFT (the actual planned work) doesn't trigger the issue
- Sources expanded with 4 new community references

**No action needed for planned Track A / B work**. The 5 use cases for
fine-tuning in 2026 (see `notes/why.md`) all involve LoRA-scale training that
stays in the 40-70 W per-step regime, where SoC envelope contention is not the
binding constraint. The SW Power Cap story is a benchmark-credibility issue,
not a training-throughput issue.

### 2026-06-05 (morning) — BF16 benchmark re-measurement: SW Power Cap discovered

User re-ran the BF16 GEMM benchmark on both the 25.11 and the newly-pulled
26.04 containers. Surface result: 97.3 TFLOPS at N=16384 (the 2026-06-04
headline number) had dropped to **67-70 TFLOPS** -- 30% regression.

Diagnostic chain:

- Same benchmark on cold-start, single-size `--sizes 16384`: 25.11=67.4,
  26.04=67.3. **Containers within noise**, so not a software stack issue.
- Temperature 63-67 °C, well below throttle, `HW Thermal Slowdown=0us`.
- `nvidia-smi -q -d PERFORMANCE` shows `SW Power Capping` counter at
  ~33,900 seconds over 33 hours uptime -- the **NVIDIA driver itself** is
  actively capping clocks at the SoC-envelope level, not the silicon limit.
- During a benchmark, the counter advances ~3 sec per 30 sec sampling window
  (10% intermittent capping).
- Causal context: machine has been running 33 h with GNOME desktop, 6 sshd
  sessions, dashboard service, node, VS Code Remote-SSH all alive. The 140 W
  GB10 SoC TDP is shared CPU↔GPU↔LPDDR5x; any CPU spikes steal envelope.
- N≤8192 GEMMs still hit ~93-95 TFLOPS (fit in cache, less memory bandwidth
  pressure). The regression is specifically large-N (≥12288) where the GEMM
  becomes LPDDR5x bandwidth bound and any contending memory traffic visibly
  hurts.

**Conclusion:** 97 TFLOPS was a *clean-room ceiling*. The realistic daily
sustained number for this unit under normal working conditions is ~70 TFLOPS
at N=16384 (or ~90-95 at N=8192). Recorded both in `notes/hardware-gx10.md`
with a new "Re-measurement on 2026-06-05" section, plus mitigations if the
headroom is ever needed back (`systemctl stop gdm`, close extra sessions,
benchmark immediately after reboot).

**No action required for planned training workloads.** LoRA / SFT GEMMs are
N≤4096; they don't hit this regression. Budget sustained 90-95 TFLOPS at
LLM-typical sizes, but treat anything claiming >95 as a clean-room headline,
not your daily baseline.

### 2026-06-05 — container refresh: 26.04 default, plus TensorRT + CUDA devel

**Theme:** kill the dependency pin pain by upgrading the PyTorch container.

Pulled three new images:

- `nvcr.io/nvidia/pytorch:26.04-py3` (23.5 GB) — primary upgrade
- `nvcr.io/nvidia/tensorrt:25.11-py3` (10.6 GB) — for Track A10 (vLLM / inference)
- `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` (6.59 GB) — has nvcc for custom kernels

Disk: 250G → 268G after pytorch:26.04, then to 272G after tensorrt, then 275G
after cuda-devel. Total new ~25 GB on disk. Free space 594 GB.

**26.04 vs 25.11 — the dependency story:**

| | 25.11 (old) | 26.04 (new) |
|---|---|---|
| torch | 2.10.0a0 | 2.12.0a0 |
| torchao | 0.14.0+git | **0.17.0+git** |
| CUDA | 13.0 | 13.2 |
| Modern peft (≥ 0.18) works? | NO (requires torchao ≥ 0.16) | YES |
| Pins needed | yes, narrow ranges | none — install latest |

The torchao 0.17 jump is what unblocks everything. peft 0.18+ asserts
`torchao >= 0.16` on import; 25.11's torchao 0.14 fails the assertion.
26.04's torchao 0.17 satisfies it, so installing latest transformers/peft/
datasets/accelerate works clean.

**Verification (5-step LoRA training inside 26.04):**

- `transformers 5.10.1` + `peft 0.19.1` + `datasets 4.8.4` + `accelerate 1.13.0`
- Loads Llama-3.2-3B-Instruct + LoRA r=16 in 45.3s, runs 5 steps in 5.5s
  (0.91 steps/s, 3.64 samples/s)
- Final loss 2.0767 (≈ identical to 25.11's 2.0566)
- Peak GPU mem 18.13 GB (≈ identical to 25.11's 18.08 GB)
- **Conclusion: 26.04 is a pure dep-resolution upgrade. Same hardware
  throughput, same memory footprint, but no more pin maintenance.**

**New scripts:**

- `experiments/smoke-test/run-26.04.sh` — same as `run.sh` but 26.04 image + no pins
- `experiments/bench/run-26.04.sh` — same as `run.sh` but 26.04 image

**Original `run.sh` files unchanged** — 25.11 path stays bit-reproducible.

**Recommendation for the curriculum going forward:** use the `-26.04.sh` variants
unless you specifically need to reproduce older results.

---

### 2026-06-04 — bootstrap day

**Theme:** zero to "can train 32B QLoRA" in one evening.

What happened (chronological):

1. **First boot recovery.** User had forgotten the password set at OOBE. Turned out
   to be a case/typo issue, not truly lost. Once in, used `sudo passwd hooyao` to
   set `123` (root path bypasses pam_pwquality enforcement). Documented this in
   `bootstrap-gx10.md` so future-them doesn't get scared the next time.
2. **Network.** Pinned GX10 to `192.168.1.200` via UniFi Fixed IP. Set up SSH key
   auth from the Windows laptop. mDNS skipped — UniFi UXG Fiber + IoT VLAN
   isolation makes a static IP simpler.
3. **System update — Phase 2 horror show.** DGX Dashboard updater stuck → 500
   error → manual `apt dist-upgrade` path. Hit:
   - `aptd` holding apt lock for 21 minutes (Dashboard left it). Stopped
     PackageKit, killed aptd, recovered.
   - **`thunderbird` snap pre-install pulls 220 MB from Canonical's London CDN
     at 100-600 KB/s.** This is THE slowest part of any DGX OS update from China.
     Pitfalls A/B in `bootstrap-gx10.md`.
   - User initially removed `thunderbird` solo, leaving 18 `thunderbird-locale-*`
     packages and `nvidia-system-station-apps` in `iU` state. Whole `apt` chain
     stopped. Resolved by reinstalling `thunderbird` (slow snap download, twice)
     — user opted to restore rather than purge the dependency tree.
4. **Firmware** already current (Dashboard had applied it on first boot).
5. **Docker.** `docker.service` was failing on startup with
   `error initializing buildkit: invalid database`. The systemd start-limit had
   tripped. Fix:
   ```bash
   sudo systemctl stop docker.socket
   sudo rm -rf /var/lib/docker/buildkit       # daemon rebuilds it
   sudo systemctl reset-failed docker.service
   sudo systemctl start docker.service
   ```
   Documented as Pitfall D in `bootstrap-gx10.md`. Added user to `docker` group.
6. **PyTorch container.** Pulled `nvcr.io/nvidia/pytorch:25.11-py3` (19.5 GB).
   First attempt from `nvcr.io` was 5-30 MB/s; user's proxy helped. Docker Hub
   itself was failing TLS handshake from China — `nvcr.io` is much more reliable.
   Added a CLAUDE.md section about `nvcr.m.daocloud.io` as a Shanghai mirror
   fallback.
7. **First GPU-in-container test passed:** sm_121 detected, BF16 GEMM works.
8. **External drive — 178 GB models.** User downloaded the Track A model set at
   office (fast network), brought home on a 250 GB external. The drive has
   24 medium-error sectors (`dmesg`), but `--ignore-errors` rsync got 100% of
   the files. SHA256 verification: 264/264 OK. **Recommend user replace this
   drive — SMART says OK but the dmesg errors are real.**
9. **GitHub repo created** as `hooyao/dgx-spark-playground` (private). Pushed.
10. **GitHub CLI installed** (`gh 2.45.0`). User logged in.
11. **Smoke test gymnastics.** First two attempts at the LoRA-3B training script
    hit dependency version conflicts inside the container:
    - First try: `transformers` requires `tokenizers<0.21` but container has 0.22.1
    - Second try (`pip install ... transformers peft`): pulled transformers 5.10
      and peft 0.19, which requires `torchao >=0.16`, container has 0.14
    - Third try (upgrading torchao): torchao 0.17 requires torch >=2.11, container
      has torch 2.10 — that path is dead
    - **Fourth try works:** pin `transformers<4.55, peft<0.18`. Documented in
      `experiments/smoke-test/run.sh` AND in this snapshot's "Stuck pinned
      dependency versions" block.
12. **Monitoring.** Installed `nvtop` and `btop`.
13. **Performance characterization.** Ran `experiments/bench/bf16_peak.py`:
    97.3 TFLOPS sustained at N=16384, 89 W, 79 °C. Recorded in
    `notes/hardware-gx10.md`. Higher than Carmack's 60 TFLOPS — newer driver +
    ASUS chassis thermal advantage + pure-GEMM workload.
14. **Curriculum.** Wrote `notes/curriculum-v2-execution.md` — three parallel
    tracks (fine-tune, pretrain+RLHF, math), ~1.5h/evening. Pointed
    `curriculum.md` to be the static-reference companion.
15. **Submodule update.** Bumped `dgx-spark-playbooks` to upstream main.

What did NOT happen:

- No curriculum days executed yet.
- No full-epoch training run.
- No real model output produced beyond the 5-step smoke pipeline (loss went
  2.06 → 1.x, but that's pipeline check, not a trained model).

User's stated next steps when resuming:

- "Tomorrow start curriculum Day A1." (Or whichever track they pick first.)
- Math refresh in parallel with whatever ML track they're on; not blocking.
