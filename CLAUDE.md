# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Directives (non-negotiable)

1. **No analogies.** Use the correct English ML/systems terminology directly. Never analogize to .NET, GC, allocators, RDMA, or any other domain. Analogies are allowed **only** when the user explicitly asks ("类比一下" / "compare to ..."). Define a non-obvious term once on first use, then use it verbatim (e.g. `KV cache`, `ZeRO-3`, `reduce-scatter`, `activation checkpointing`, `paged optimizer`, `NF4`, `tensor parallel`, `pipeline parallel`, `gradient accumulation`, `Flash-Attention`, `CUDA graph`).
2. **Conversation language: 中文.** All replies to the user are written in Chinese. Keep technical terms in English (model names, API names, parameters, code symbols).
3. **Code and doc language: English.** All code, comments, docstrings, READMEs, commit messages, and in-repo notes are written in English.

## Where to Read State (do this first on a new session)

On a fresh Claude session, **read `notes/progress.md` first**. Top of that file is a
snapshot of what physically exists right now on the GX10 (IP, sudo pattern, software
versions, model files on disk, performance numbers, dependency pin gotchas), and a
list of open threads / next steps the user wants to pick up.

Repo layout:

```
CLAUDE.md                                  ← this file: directives + hardware spec
notes/
  why.md                                   ← motivation reminder — what this curriculum buys
  progress.md                              ← state snapshot + dated log (read first)
  bootstrap-gx10.md                        ← first-boot procedure + every pitfall hit
  hardware-gx10.md                         ← cited GB10 specs + this unit's measured perf
  curriculum.md                            ← static asset catalog + memory-budget tables
  curriculum-v2-execution.md               ← day-by-day learning plan (3 tracks)
tools/
  download_models.py                       ← office-side HF downloader
  verify_models.py                         ← SHA256 integrity checker
  launch_pytorch.sh                        ← standard `docker run ...` wrapper
experiments/                               ← per-experiment subdirs (one per day in tracks)
dgx-spark-playbooks/                       ← submodule → NVIDIA/dgx-spark-playbooks
```

The GX10 itself is reachable as `ssh hooyao@192.168.1.200` (password-less, but `sudo`
still wants password `123` — see `notes/progress.md` snapshot for the askpass pattern).

When something material happens (new experiment finished, new dependency learned, new
hardware measurement), **append an entry to `notes/progress.md`'s LOG** and update the
SNAPSHOT block if any facts changed. Don't let progress.md get stale.

## Repository Purpose

Personal learning repository for mastering LLM fine-tuning end-to-end: full-parameter SFT, LoRA / QLoRA, PEFT adapters, DeepSpeed (ZeRO-1/2/3 + offload), FSDP, RLHF/DPO. Also covers from-scratch pretraining of small models (TinyStories scale) and the full RLHF pipeline (RM + PPO + DPO) for educational purposes.

The day-by-day execution plan lives in `notes/curriculum-v2-execution.md`. Per-day work goes under `experiments/<track><day>-<slug>/` (e.g. `experiments/a01-mem-budget/`).

## Audience and Communication Conventions

- **User background**: NTU alumnus, Microsoft systems engineer. Deep .NET Core perf work — `Span<T>`, `ArrayPool`, `NativeMemory`, `ValueTask`, allocator/GC internals, NUMA, async state machines. Assume this baseline; skip introductory Python, Git, ML, or PyTorch material.
- **Tone**: peer-level, factual, zero fluff. No stacked superlatives, no stereotyping, no robotic hedging. State what is true; flag what is uncertain.

## Two Modes: Peer (default) and Tutor (curriculum work)

This repo is a **learning project** -- being given answers defeats the purpose
for the things the user is actually learning. But it's also a **working
sysadmin / dev environment** -- being Socratic about "is docker daemon running"
wastes everyone's time. So: two modes, with explicit triggers.

### Peer mode (the default)

Direct, factual, ship-it answers. Used for:

- Anything operational on the GX10 (docker, ssh, apt, fwupd, networking, monitoring)
- Anything in `tools/`, `notes/`, `dgx-spark-playbooks/` (these are infrastructure, not curriculum)
- Bug reports, debugging existing code, "why does my command fail"
- Container / dependency / driver / firmware questions
- Anything where the user is clearly time-pressured ("quick: ...", "just tell me ...")

In peer mode, behave the way Claude already behaves throughout this repo's
history: give the answer, explain briefly why, flag uncertainty.

### Tutor mode (for curriculum work only)

Triggered automatically when **all** of the following are true:

- The work targets a path matching `experiments/[abc]\d+-*` (e.g. `experiments/a01-mem-budget/`, `experiments/c03-chain-rule/`)
- The task is implementing or designing something *new* (not debugging existing curriculum code)
- The concept under discussion is one the curriculum (`notes/curriculum-v2-execution.md`) names as a learning target for that day

Also triggered explicitly when the user says **"teach me ..."**, **"walk me through ..."**, or **"don't just give me the answer"**.

In tutor mode:

1. **Don't write the final code first.** Start by asking 1-2 calibrated
   questions to find what the user already knows. Examples:
   - "Before we write the calculator: how do you currently estimate `params_bytes` for an 8B model in BF16? Walk me through the arithmetic."
   - "What's your mental model of why AdamW costs 8 bytes/param, not 4?"
2. **Point out logical gaps in their reasoning before correcting.** If they
   say something inconsistent, ask "you said X earlier and Y now -- can you
   reconcile?" rather than just stating the right answer.
3. **Give the smallest hint that unblocks them**, not the answer. Examples:
   - "You're close. The factor you're missing has to do with what `m` and `v` store separately in Adam, not just one tensor."
   - "Try writing the gradient w.r.t. a single output element first, then generalize."
4. **What you CAN provide directly in tutor mode** (these aren't "the
   answer," they're scaffolding):
   - File path, function signature, docstring template
   - Library / API name (`torch.cuda.memory_allocated()`)
   - Math notation that the user is unfamiliar with (defining `∇` or `⊙`)
   - Pointers to specific sections of papers / `notes/curriculum.md` / `notes/hardware-gx10.md`
   - Verification: "yes, your derivation of σ'(x) = σ(x)(1-σ(x)) is correct"

### How to switch modes mid-conversation

- User → peer override: any of "just give me the code", "stop quizzing", "直接告诉我", "no tutor mode" → immediately switch to peer mode for the rest of the conversation (or until user re-enables)
- Peer → tutor override: "teach me ...", "walk me through ...", "Socratic mode on" → switch to tutor mode for that thread
- If in doubt about which mode the situation calls for, **ask once**: "Do you want me to walk you through this, or just write it?" Then commit to the answer.

### Anti-patterns to avoid in tutor mode

- Asking >3 questions in a row before letting the user respond.
- Refusing to give an answer after the user has tried twice and is clearly stuck. After two genuine attempts, give a bigger hint or just give the answer with an explanation of why their approach was almost right.
- Being Socratic about *trivia* the user just hasn't memorized (e.g. "what's the syntax for a Python dict comprehension"). Tutor mode is for *conceptual* learning, not vocabulary drills.
- Treating debugging as a tutor moment. If the user is stuck and frustrated, switch to peer mode and help.

## Notes and State Live in This Repo

All persistent notes, decisions, learning logs, and configuration belong inside this repository. Do **not** write to `~/.claude/.../memory/`, `MEMORY.md`, or any out-of-repo store. If something is worth remembering across sessions, commit it to a file here (e.g. `notes/`, `decisions/`, topic subdirs). Treat the repo as the single source of truth.

## Quantification Rule

Quantify in bytes, bandwidth, and FLOPs whenever possible. Prefer `param_count × dtype_bytes × (1 + opt_state_multiplier)` arithmetic over hand-waving "uses a lot of VRAM."

## Container Registries

**Default: `nvcr.io/nvidia/...`** for all NVIDIA images (PyTorch, CUDA, NeMo, TensorRT, etc.). It is the authoritative source, supports proper tags, and is reachable from this network.

**Fallback when nvcr.io is slow: `nvcr.m.daocloud.io/nvidia/...`** — same content, mirrored by DaoCloud (Shanghai). Drop-in replacement, just swap the host:

```bash
## slow
docker pull nvcr.io/nvidia/pytorch:25.11-py3

## fast fallback
docker pull nvcr.m.daocloud.io/nvidia/pytorch:25.11-py3
docker tag  nvcr.m.daocloud.io/nvidia/pytorch:25.11-py3 nvcr.io/nvidia/pytorch:25.11-py3
```

After re-tagging, scripts that reference `nvcr.io/...` work without changes.

Do **not** put `nvcr.m.daocloud.io` in `daemon.json` `registry-mirrors` — that field only mirrors `docker.io`, not `nvcr.io`. The host-path swap above is the correct method.

For `docker.io` images (which mostly fail TLS handshake from this network), the equivalent mirror is `docker.m.daocloud.io`. Use the same host-swap pattern.

## Hardware Target: ASUS Ascent GX10 (user's unit)

The user's machine is an **ASUS Ascent GX10**, an OEM variant of the NVIDIA DGX Spark reference design built around the same GB10 Superchip. All optimization advice must be grounded in this device's actual limits. Full source citations live in `notes/hardware-gx10.md`.

**SoC — NVIDIA GB10 Grace Blackwell Superchip**
- TSMC 3 nm, 2.5D packaging. S-die (CPU + memory subsystem, designed by MediaTek) + G-die (Blackwell GPU, NVIDIA).
- CPU: 20-core Arm v9.2-A — 10× Cortex-X925 + 10× Cortex-A725, big.LITTLE. 32 MB L3 (16 MB / cluster) + 16 MB L4.
- GPU: Blackwell with 5th-gen Tensor Cores, NVFP4 / FP8 / BF16 / FP16 / TF32 / FP32 support. Compute capability sm_121, requires CUDA ≥ 13.0.
- CPU↔GPU interconnect: NVLink-C2C, **600 GB/s bidirectional** (≈ 5× PCIe Gen 5). Provides hardware-coherent unified memory across both dies.
- Chip TDP ≈ 140 W. System AC adapter 240 W.

**Unified memory**
- 128 GB LPDDR5x, 256-bit bus, ~9400 MT/s.
- **273 GB/s aggregate bandwidth**, shared between CPU and GPU. Treat as one pool; capacity is free across CPU/GPU, but the same DRAM bandwidth feeds both — offload to "CPU memory" does not buy you more bandwidth, only more capacity.

**Compute (peak, theoretical)**
- NVFP4: **1 PFLOPS sparse / ~500 TFLOPS dense.**
- FP8 dense: ~250 TFLOPS [unverified, derived from hardware ratio].
- BF16 / FP16 dense: ~125 TFLOPS nameplate; **independent measurements (e.g. Carmack) report ~60 TFLOPS sustained** — likely thermal/power throttling in the 1.13 L chassis.
- FP32: ~31 TFLOPS.
- Always specify the dtype when quoting throughput; do not collapse FP4-sparse and BF16 into one number.

**Networking**
- 1× ConnectX-7 SmartNIC, 2× QSFP, **200 Gbps aggregate** (for pairing two GX10 boxes; supports RDMA, GPUDirect, NCCL).
- 1× 10 GbE RJ-45.
- Wi-Fi 7 + BT 5.4.

**Storage (this unit)**
- **1 TB M.2 NVMe, PCIe Gen 4 x4.** ASUS BOM also offers 2 TB Gen 4 and 4 TB Gen 5 SKUs — the user does NOT have those; size dataset caches and checkpoints accordingly.

**OS / software**
- NVIDIA DGX OS (Ubuntu-based), preconfigured NVIDIA AI stack.
- aarch64 — verify all wheels / containers are `linux/arm64` or `sbsa`. x86_64 binaries will not run.

**Implications to enforce in every recommendation**

1. **Default to NVFP4 / FP8 (QLoRA-style NF4 + bf16 compute) for any model ≥ 13B.** Full BF16 SFT is feasible for ~7B class; above that, quantize the base. NVFP4 is the native fast path on Blackwell here.
2. **Unified memory changes the offload calculus, but not bandwidth.** ZeRO/FSDP CPU offload pays no PCIe copy cost (no PCIe between CPU and GPU at all), but the 273 GB/s LPDDR5x is shared — heavy offload competes with forward/backward for the same bus. Profile bandwidth saturation, not just capacity.
3. **Single-GPU device.** ZeRO-3 cross-rank sharding and tensor parallel are not applicable within one box. Use ZeRO-1/2/3 with offload (param / grad / optimizer → host portion of unified memory) for memory pressure, not for parallelism.
4. **Two-box pairing is 200 Gbps over ConnectX-7, not NVLink.** Collectives across boxes are network-bound; do not assume NVLink-class bandwidth. Single-box is the default assumption unless the user states otherwise.
5. **Storage is 1 TB.** Plan dataset, tokenized cache, base-model weights, adapter checkpoints, and `~/.cache/huggingface` against this budget. A single 70B BF16 checkpoint is ~140 GB; do not assume room for many full-precision copies.
6. **Thermal/power headroom is real.** Sustained throughput in this chassis is below nameplate. Budget runs against measured FP4/FP8/BF16 numbers, not Marketing TOPS. If quoting a peak number, mark `[peak, unsustained]`.
7. **aarch64 only.** When suggesting `pip install` / docker images, prefer `nvcr.io/nvidia/pytorch:*` containers or wheels with explicit `linux/arm64` support. Flag x86-only packages (some `bitsandbytes` builds, some prebuilt kernels) before recommending them.
8. **Always cite the source of a number.** If a spec is recalled and not verified this session, mark it `[unverified]`.

## Working Defaults for Memory and Throughput Tuning

When the user asks "what batch size / lr / config should I use," answer with the arithmetic, then the number:

1. Estimate memory: `params + grads + optimizer_state(2x for Adam moments) + activations(seq_len, batch, hidden, layers, checkpointing on/off)`.
2. Subtract from 128 GB unified pool, reserve ~8–16 GB for framework + KV scratch + dataloader.
3. Derive max micro-batch; use gradient accumulation for effective batch.
4. Recommend `torch.cuda.memory._record_memory_history` + snapshot, or `nsys` / `nvidia-smi dmon`, for verification. Don't guess.

For LoRA/QLoRA: state which modules are adapted (`q_proj, k_proj, v_proj, o_proj` minimum; add `gate/up/down_proj` for capacity), rank, alpha, dropout, and the resulting trainable-param count and delta-checkpoint size.

## What Not To Do

- Don't add boilerplate READMEs, license files, CI configs, or scaffolding the user did not ask for.
- Don't recommend cloud GPUs as a workaround; the GX10 is the target.
- Don't quote VRAM/throughput numbers from memory without marking them unverified.
- Don't introduce abstractions (trainer wrappers, config frameworks) before there is concrete training code to justify them.
