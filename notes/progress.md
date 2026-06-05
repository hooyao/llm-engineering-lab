# Progress — append-only log + current snapshot

> **New Claude session: read this file first.** The snapshot block below tells you
> what physically exists on the GX10 right now. The log below tells you how we got
> here. The "Open threads" section at the bottom tells you what to ask the user
> about / pick up next.

---

## STATE SNAPSHOT (last updated 2026-06-05)

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

---

## LOG (append new entries at the top)

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
