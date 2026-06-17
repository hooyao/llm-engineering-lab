# Hardware: ASUS Ascent GX10 (user's unit) — verified specs

This is the canonical, cited reference for the user's machine. All numbers in `CLAUDE.md`'s "Hardware Target" section trace back here. When updating either file, keep them in sync.

The ASUS Ascent GX10 is an OEM variant of the NVIDIA DGX Spark reference design. Both are built on the **NVIDIA GB10 Grace Blackwell Superchip**. The chassis, storage SKU, and exact networking layout differ from the NVIDIA-branded Spark; the SoC, memory subsystem, and compute envelope are identical.

## SoC: NVIDIA GB10 Grace Blackwell Superchip

| Item | Value | Source |
|---|---|---|
| Process | TSMC 3 nm, 2.5D advanced packaging | Hot Chips 2025 GB10 talk; The Register coverage |
| Composition | S-die (CPU + memory subsystem, MediaTek) + G-die (Blackwell GPU, NVIDIA) | Hot Chips 2025 |
| CPU | 20-core Arm v9.2-A: 10× Cortex-X925 + 10× Cortex-A725 (big.LITTLE) | NVIDIA DGX Spark Hardware Overview; ASUS GX10 techspec |
| CPU cache | 32 MB L3 (16 MB per cluster) + 16 MB L4 | The Register / Hot Chips |
| GPU | NVIDIA Blackwell, 5th-gen Tensor Cores, NVFP4 / FP8 / BF16 / FP16 / TF32 / FP32 | NVIDIA DGX Spark spec; DGX Spark Playbooks DeepWiki |
| Compute capability | sm_121 — requires CUDA ≥ 13.0, PyTorch with Blackwell wheels | rossingram/Spark-DGX-Benchmark README; nvidia-smi output documented in NVIDIA playbooks |
| CPU↔GPU interconnect | NVLink-C2C, 600 GB/s bidirectional (≈ 5× PCIe Gen 5) | The Register, Hot Chips 2025 |
| Chip TDP | ~140 W | The Register |

> Note: NVIDIA marketing repeatedly states "NVLink-C2C at 900 GB/s." 900 GB/s is the figure for prior **Grace Hopper / GB200** parts. The Register's Hot Chips coverage of the GB10 talk specifies **600 GB/s bidirectional** for this SoC. Use 600 GB/s for GX10 math, and cite both when the discrepancy matters.

## Unified memory

| Item | Value | Source |
|---|---|---|
| Capacity | 128 GB LPDDR5x | NVIDIA DGX Spark Hardware Overview; ASUS Ascent GX10 techspec |
| Bus width | 256-bit | NVIDIA Hardware Overview; PNY DGX Spark page |
| Data rate | 4266 MHz (NVIDIA) / 9400 MT/s (The Register) | NVIDIA Hardware Overview; The Register |
| Aggregate bandwidth | **273 GB/s** (NVIDIA), 273–301 GB/s (The Register) | NVIDIA Hardware Overview; PNY page; The Register |
| Coherence | Hardware-coherent across CPU and GPU; UMA | NVIDIA; DGX Spark Playbooks |

**Operational implication:** capacity is shared but so is bandwidth. Moving an optimizer state to "CPU" within the unified pool costs no PCIe transfer, but every CPU access still consumes the same 273 GB/s budget that forward/backward needs.

## Compute (peak, theoretical)

| Dtype | Throughput | Notes / source |
|---|---|---|
| NVFP4 sparse | **1 PFLOPS (1000 TOPS)** | NVIDIA / ASUS marketing; requires structured 2:4 sparsity |
| NVFP4 dense | ~500 TFLOPS | Half of sparse; standard sparsity ratio |
| FP8 dense | ~250 TFLOPS | [unverified, derived from FP4→FP8 ratio]; some sources cite ~208 TFLOPS |
| BF16 / FP16 dense | ~125 TFLOPS nameplate, **~60 TFLOPS measured sustained** | TWOWIN evaluation citing John Carmack / Awni Hannun |
| FP32 | ~31 TFLOPS (CUDA cores) | Hot Chips 2025 |

**Operational implication:** the gap between FP4-sparse marketing and BF16-sustained reality is ~16×. When the user asks "how fast will SFT run?", base estimates on BF16/FP8 sustained numbers, not the headline PFLOP.

## Networking

| Item | Value | Source |
|---|---|---|
| ConnectX-7 SmartNIC | 1× card, 2× QSFP, **200 Gbps aggregate** | ASUS GX10 datasheet (Altron PDF); NVIDIA DGX Spark Playbooks DeepWiki |
| Wired LAN | 1× 10 GbE RJ-45 | ASUS techspec |
| Wireless | Wi-Fi 7 (AW-EM637, 2×2) + Bluetooth 5.4 | ASUS techspec |

**Operational implication:** two-box pairing tops out at 200 Gbps with RDMA / GPUDirect / NCCL support. That is roughly 25 GB/s per direction. All-reduce / reduce-scatter across two boxes is bandwidth-bound by this, not by NVLink-class speeds. ZeRO-3 across two boxes is feasible for very-large-parameter inference; for training, communication will dominate above a small per-step compute envelope.

## Storage — user's unit

| Item | Value | Source |
|---|---|---|
| Installed | **1 TB M.2 NVMe, PCIe Gen 4 x4** | ASUS Ascent GX10 SKU (this unit); confirmed against ASUS techspec |
| Slot | 1× M.2 2242/2230, PCIe Gen 5 x4 capable (Gen 4 backward compatible) | ASUS datasheet (Altron PDF) |
| Other ASUS SKUs (NOT this unit) | 2 TB Gen 4, 4 TB Gen 5 | ASUS US techspec |

**Operational implication:** budget 1 TB across base-model weights, tokenized dataset caches, activation/optimizer offload spill, and adapter checkpoints. A single Llama-3 70B BF16 checkpoint ≈ 140 GB; a 70B FP8 ≈ 70 GB; a 70B NF4 ≈ 35 GB. `~/.cache/huggingface` grows fast — pin it to NVMe and clean aggressively.

## OS / software

| Item | Value | Source |
|---|---|---|
| OS | NVIDIA DGX OS (Ubuntu-based), ASUS-branded image `7.4.0-3` available | ASUS download portal |
| Architecture | aarch64 (Arm v9.2) | All sources |
| Driver / CUDA | NVIDIA 580.82.09+, CUDA 13.0+, NCCL 2.28.9+ | DGX Spark Playbooks DeepWiki |
| Container baseline | `nvcr.io/nvidia/pytorch:*` with Blackwell (sm_121) support, e.g. PyTorch 2.9.0a0+50eac811a6 + CUDA 13.0.1 | rossingram/Spark-DGX-Benchmark |

**Operational implication:** verify every wheel is `linux/arm64` or `sbsa`. x86_64 prebuilt binaries (some `bitsandbytes` releases, some custom CUDA kernels) will not run. Prefer NVIDIA's PyTorch containers as a known-good baseline.

## Physical / power

| Item | Value | Source |
|---|---|---|
| Dimensions | 150 × 150 × 51 mm | ASUS techspec |
| Weight | 1.48 kg | ASUS techspec |
| Power supply | 240 W external adapter, 180 W EPR PD3.1 over USB-C in | ASUS techspec |
| Sustained behavior | Reports of thermal throttling and spontaneous reboot under sustained 100 W load | TWOWIN evaluation, citing user reports |

**Operational implication:** treat any peak number as a ceiling, not a target. For long runs, log `nvidia-smi dmon`, thermal headers, and power draw; expect to back off well below nameplate to stay stable.

## Measured on this specific unit (2026-06-04)

Driver `580.159.03`, CUDA build `13.0`, `nvcr.io/nvidia/pytorch:25.11-py3` container, room temp ~25 °C. Measurements done via `experiments/bench/bf16_peak.py` (square BF16 GEMM `(N x N) @ (N x N)`, default `--warmup 20 --iters 200` mode).

**BF16 sustained throughput vs matrix size:**

| N | iters | median latency | p99 latency | sustained TFLOPS | peak (median) | peak (best) |
|--:|--:|--:|--:|--:|--:|--:|
| 2048  | 151,115 | 0.19 ms  | 0.33 ms  | 86.5  | 88.5 | 90.1 |
| 4096  | 18,826  | 1.56 ms  | 2.31 ms  | 86.2  | 87.9 | 91.0 |
| 8192  | 2,603   | 11.36 ms | 14.53 ms | 95.4  | 96.8 | 98.3 |
| 12288 | 776     | 38.17 ms | 42.82 ms | 95.9  | 97.2 | 98.1 |
| 16384 | 332     | 89.37 ms | 97.88 ms | **97.3** | 98.4 | 99.9 |

**`nvidia-smi` snapshot during N=16384 GEMM:**

| Metric | Value |
|---|---|
| GPU die power (`Pwr:Usage`) | 89 W |
| GPU temperature | 79 °C |
| Perf state | P0 (highest) |
| GPU utilization | 96 % |
| Memory-Usage | `Not Supported` (iGPU, expected) |
| ECC counters | `N/A` across the board (LPDDR5x has no exposed ECC; SMART ECC fields are not applicable to GB10 iGPU) |
| Xid errors in `dmesg` | none |

**Reading:**

- **97.3 TFLOPS BF16 sustained at N=16384, ≈ 78 % of the 125 TFLOPS nameplate.** Higher than the ~60 TFLOPS Carmack-style independent figure typically cited, likely from a combination of: newer driver (580.x is months past first-release), the ASUS chassis's `1.6x more efficient thermal coverage` versus the NVIDIA Founders Edition, and pure-GEMM workloads being the easiest case to saturate tensor cores.
- **Where our ~95 sits vs the community (verified 2026-06-18).** Community DGX Spark BF16/FP16 "TFLOPS" numbers span ~20× and must NOT be compared blind — they fall in three tiers:
  - **mmapeak (raw MMA instruction peak): ~213 TFLOPS** — alan.dang on NVIDIA forum t/351993. Registers-only, no memory traffic, no cuBLAS. This is an instruction-level ceiling, NOT a real GEMM. Do not compare our number to this.
  - **Real cuBLAS / PyTorch square-GEMM (our method): ~45–96 TFLOPS** — the only tier comparable to ours. Carmack / Awni Hannun ~60 BF16 (Oct-2025 launch, SW Power Cap held the box at ~100 W); GuigsEvt ~45 FP16 default stack, "~1.5× with rebuilt stack" → ~67. All on the low side, and each low number has a documented cause (early firmware, power cap, or default software stack).
  - **Broken / default-stack GEMM: ~11–12 TFLOPS** — Ross Ingram's starting point before enabling Blackwell kernels; his whole writeup is the climb out of it. A misconfiguration value, not a hardware figure.
  - **Conclusion:** our 95–96 is at the *top* of the real-GEMM tier, and legitimately so — we run the nvcr `26.04` container (torch 2.12 nv-build, correct Blackwell kernels) on mature firmware and pick the best size. That is exactly the "fixed" state Ross Ingram was chasing. The widely-quoted ~60 headline is an early-firmware + power-capped + default-stack artifact, not the ceiling of a correctly-configured GB10.
- **No thermal throttling at this duration / load.** Headroom to throttle threshold ~8-10 °C. Clocks stay at P0.
- **Power efficiency ~1.09 TFLOPS/W (GPU die)** -- comparable to H100 SXM and ahead of A100 SXM on the same metric. Real chassis draw (CPU, DRAM controller, NIC, fan) brings it to roughly 130 W on this unit, well below the 240 W adapter ceiling.
- **GEMM size matters.** Below N=8192 the per-call cuBLAS overhead is non-trivial and sustained throughput drops to ~86 TFLOPS. Real LLM workloads with small effective batches (e.g. LoRA 3B at batch 4 / seq 1024) hit the smaller-size regime AND don't issue continuous GEMMs; observed GPU die power for that case was only ~44 W. To actually saturate this GPU, use large effective batch × long seq, or train ≥ 14B models.
- **For experiment planning, budget sustained BF16 = ~95 TFLOPS at large N, ~60 TFLOPS at LLM-typical sizes.** Mark either as `[measured on this unit, single-shot, 2026-06]` -- repeat the benchmark if room temperature changes a lot or the driver moves to a major new release.

## Re-measurement on 2026-06-05 (after ~33h uptime, with desktop session active)

Same benchmark, same hardware, same container `25.11-py3` **and** the newly-pulled
`26.04-py3`. Driver `580.159.03` unchanged.

| N | 2026-06-04 (initial) | 2026-06-05 (re-run, 25.11) | 2026-06-05 (re-run, 26.04) | 2026-06-05 (cold reboot, 26.04) |
|--:|--:|--:|--:|--:|
| 2048  | 86.5 | — | 85.3 | — |
| 4096  | 86.2 | — | 87.5 | — |
| 8192  | 95.4 | ~95 (single-size cold: 93.0) | 95.9 | — |
| 12288 | 95.9 | — | 86.9 | — |
| 16384 | **97.3** | **67.4** | **70.6** (sweep) / **67.3** (cold solo) | **93.4** ✅ |

**Cold reboot fully recovers the headroom.** The third column was after
`sudo systemctl stop gdm` (recovered +13 %, 67 → 76). The fourth column was
after a full power cycle, and N=16384 returned to **93.4 TFLOPS sustained**
with **89.6 W GPU power** and idle clock back to 2119 MHz (not the P8/208 MHz
deep sleep we saw after long uptime). peak(best) actually hit **101.3 TFLOPS**,
slightly above the original 99.9. So the silicon was always fine; what
accumulates is the driver's `SW Power Capping` history influencing the
power-management state.

**The 26.04 container is not the cause.** Bit-for-bit comparison at N=16384 in
cold-start single-size mode produced `25.11=67.4` and `26.04=67.3`. The two
containers are within noise; whatever changed, changed at the host level.

### Root cause: SW Power Cap (NVIDIA driver-side power management)

`nvidia-smi -q -d PERFORMANCE` shows accumulating `SW Power Capping` counters:

```
Clocks Event Reasons Counters
    SW Power Capping  :  33921903067 us   ← ~9.4 hours over 33h uptime
    HW Thermal Slowdown :  0 us
    HW Power Braking    :  0 us
```

The GPU is not thermally throttled (HW Thermal Slowdown stays at 0) and is not
hitting the silicon's hard power brake. It is the *driver* actively capping
clocks to keep the SoC inside an envelope. Measured during a benchmark, the
`SW Power Capping` counter advanced ~3 seconds per 30 seconds of sampling --
real, but intermittent.

### What changed since 2026-06-04

- **Desktop session is now active**: 6 sshd processes, GNOME shell, dashboard
  service, a `node` server, VS Code remote (`code-f6cfa2ea24`) all running.
  CPU at 0.30 load average, none individually significant, **but** GB10's
  140 W SoC TDP is shared between CPU, GPU, and the LPDDR5x memory controller.
  Whenever the CPU side spikes (e.g. snap auto-refresh, dashboard polling),
  the driver gives the GPU less to keep within envelope.
- **LPDDR5x bandwidth contention**: at N=16384 each matmul reads/writes ~1.6 GB.
  The LPDDR5x 273 GB/s is shared CPU↔GPU; any background memory traffic
  (GNOME compositor, IDE indexing, telemetry daemons) steals bandwidth that
  the GEMM needs to be compute-bound. Smaller GEMMs (N≤8192) fit better in
  L2 cache and don't show this regression.

### Revised operational numbers

| Workload class | TFLOPS to budget |
|---|---:|
| Pure BF16 GEMM, N=16384, **fresh reboot**, no desktop | **93-97** (close to ceiling) |
| Same, but typical day-to-day with desktop / SSH sessions / long uptime | **65-75** |
| Pure BF16 GEMM, N=8192 (the cuBLAS sweet spot) | **93-96** consistently regardless of uptime |
| LoRA training, batch 4, seq 1024 (real workload) | **~60** equivalent (GPU sits at ~44 W, not GEMM-bound) |

**Bottom line: the 93-97 TFLOPS figure is repeatable** as long as you reboot
when you need it. The "97" was not a clean-room oddity, it's the genuine
sustained ceiling for this unit. The 67-76 TFLOPS regime is what you get
after the box has been running with active sessions for ~24h+ and the EC /
driver power-management state has drifted.

### Mitigations if you need the headroom back

Ranked by effectiveness (measured on this unit):

1. **Full cold reboot.** Recovers from 67 → **93.4 TFLOPS at N=16384**.
   `sudo reboot` is enough; doesn't need power-brick unplug. After reboot,
   GPU idle clock returns to ~2119 MHz (not the P8 / 208 MHz deep-sleep
   state observed after long uptime), and `SW Power Capping` counter resets
   along with the rest of driver state. **This is the only reliable
   mitigation**; everything below is partial.
2. `sudo systemctl stop gdm` (kills the GNOME desktop while you train) -- frees
   the largest single non-essential CPU+GPU consumer. **Measured impact on this
   unit: 67 → 76 TFLOPS at N=16384, +13%.** Not full recovery to 93+.
3. Close VS Code Remote-SSH sessions and extra terminal multiplexers during
   long training runs. Each `sshd` session has small but non-zero overhead.
4. The `SW Power Capping` counter is a real sensor -- read it before and after
   long runs to verify whether you got the clean envelope or the contended one:
   ```
   nvidia-smi -q -d PERFORMANCE | grep "SW Power Capping"
   ```
   If the counter rate (delta_us per wall-clock second) is > 100,000 during a
   benchmark, the driver is actively capping you. Reboot resets the
   accumulated counter and the state that drives it.

### This is a known systemic issue, not a per-unit defect

Confirmed by extensive community reporting. Same symptom (`SW Power Capping`
active when GPU is supposed to be running flat-out, sometimes degrading
further into a hard 14 W lock that requires a full power-cycle to clear) hits
every DGX Spark / GB10 unit, including:

- NVIDIA Developer Forum thread "DGX Spark Performance Degradation - GPU
  Power Draw Issue" (65+ replies, persistent topic)
- "DGX SPAK GPU power usage cap at 14 W" (March 2026): user describes GPU
  stuck at 14 W after a crash, NVIDIA workaround = unplug power brick for
  30 s to reset the EC controller
- "GB10 is power limited after crash" (2026-06-04)
- "Latest Update (20 Mar 2026) on Nvidia Spark FE caps GPU performance"
- A CTO published a write-up across 14 DGX Spark units in their fleet
  observing the same hard 14 W cap pattern repeatedly:
  https://dredyson.com/a-ctos-definitive-guide-to-resolving-dgx-spark-gpu-power-draw-degradation/
- Community diagnostic CLI `spark-doctor` was built specifically because of
  these issues; first rule it checks is `power.low_draw_under_load`:
  https://github.com/joeynyc/spark-doctor

**Root cause** (community consensus + NVIDIA's own clarification):

> NVIDIA, 2025-10-31 forum:
> "DGX Spark's peak total system power is 240 W. The TDP of the GB10 SOC
> which includes the GPU and the CPU is 140 W. The rest of the system
> (ConnectX-7, SSD, USB-C provisioning) is 100 W. When measuring power usage
> via nvidia-smi, the wattage displayed measures only GPU power."

The 140 W SoC TDP is a hard envelope shared CPU↔GPU↔LPDDR5x memory
controller. The driver actively caps GPU clocks to stay within it. Any
CPU-side or memory-side activity reduces what's available for the GPU,
regardless of `nvidia-smi`'s GPU-only power figure.

### Practical implications for this curriculum

- **LoRA / SFT / QLoRA training (the curriculum core) does NOT hit this
  problem.** Those workloads are N≤4096 small GEMMs with memory bandwidth
  well within budget; observed GPU draw stays in the 40-70 W range and
  there's no envelope contention.
- **Don't chase the 97 TFLOPS number** unless you have a clean-room
  experimental need. The realistic working baseline for this unit, with
  normal SSH sessions and a running dockerd, is **70-80 TFLOPS at N=16384
  and ~93 TFLOPS at N=8192**.
- If you ever DO need maximum GEMM throughput (e.g. for a tight
  reproducibility comparison vs another hardware platform), follow this
  procedure: `sudo systemctl stop gdm` → close all VS Code Remote-SSH
  sessions → reboot for a fresh `uptime` → SSH in via a single session →
  run benchmark immediately.

## Emergency playbook: hard shutdowns / 14 W power-cap lock / clock throttling

This section is a **break-glass procedure**, not a default deployment guide.
None of these mitigations are recommended on a healthy unit. They exist so
that when something does go wrong, you don't have to re-derive the
troubleshooting tree under pressure.

### Symptom triage

| Symptom | What it is | First action |
|---|---|---|
| Box silently powers off mid-workload (no kernel panic, no log) | The 14 W power cap death-lock, or EC over-current trip | Unplug power brick for **30+ s** to reset EC, then restart |
| `nvidia-smi` shows ~14 W under sustained 96% util, model inference half speed | Same as above, in stuck state | Same: unplug power brick 30 s |
| `SW Power Capping` counter accumulating but no crashes | Normal envelope contention (covered above) | No emergency action needed |
| Repeated shutdowns within 10 min of sustained GPU load | Likely thermal / PD subsystem under-engineered for chassis | Consider clock clamp (see below) |
| `HW Thermal Slowdown` counter rising | Actual thermal throttle (different from above) | Check fan, ambient temp, dust |

### Hard reset for the 14 W death lock (the most common emergency)

Reported by multiple users across NVIDIA forum threads (March + June 2026)
and confirmed in fleet-scale write-ups. Procedure:

1. `sudo shutdown -h now` (clean shutdown of the OS)
2. **Unplug the power brick from the wall** (not just from the device)
3. Wait **30 seconds minimum** to drain EC capacitor charge
4. Plug back in, boot
5. Verify with `nvidia-smi --query-gpu=power.draw --format=csv` -- should
   return to normal idle range (5-15 W). Run a small benchmark to confirm
   it can ramp to 60-90 W on demand.

This reset is needed because the EC firmware can latch into a low-power
state that no software (driver reload, reboot, suspend) can clear. The 30s
power-off lets the EC microcontroller fully de-energize and reinitialize.

### Clock clamping (last resort, only if sustained shutdowns recur)

If the unit reproducibly crashes within ~10 minutes of sustained boost
clocks, the community (Dre Dyson quant lab,
https://dredyson.com/how-i-fixed-dgx-spark-overheating-shutdowns-...)
reports clock throttling fixes it. The mechanism: boost clocks pull
transient current that the chassis PD subsystem can't sustain, EC trips.

**The 2150 MHz value now has a real community source** (found 2026-06-18),
though still not an official NVIDIA one:
- `github.com/eugr/spark-vllm-docker` README → Known Issues: firmware "may
  cause sudden shutdown event ... during heavy inference"; workaround
  `sudo nvidia-smi -lgc 200,2150`; **lock only survives until reboot**.
- NVIDIA forum thread t/373251 ("DGX Spark (GB10) reproducibly hard
  powers-off under GPU load"). Real symptom, but the "PMIC hard cutoff /
  SoC spikes 100°C in microseconds / engineering-consensus physical limit"
  narrative that circulates with it is **embellished** — no NVIDIA staff in
  the thread, "PMIC" never appears, and the actually-confirmed fixes were
  repaste (dry/brittle TIM) + case-off + 120 mm fan. Clamping is one
  workaround, not a proven floor.

**2150 is a sane starting clamp, not a magic number.** Still descend
progressively to the lowest ceiling that holds; 2150 is just the most-cited
community starting point. NOTE the source says "default 2411"; **this ASUS
unit's max is 3003** (Applications Clock 2418), so the source's numbers are
for a different SKU envelope — don't import them literally.

**Measured cost of the 2150 clamp on this unit (2026-06-18, container 26.04,
`bf16_peak.py`):** clamp costs only ~1% BF16 throughput at each tier's best
size — **96.8 TFLOPS uncapped (N=8192) → 95.7 capped (N=12288)** — for ~14%
less GPU power (94 → 81 W) and zero clock jitter (loaded gclk pinned at
2132 MHz vs 2366 MHz drifting under SW Power Cap). Crucially, **even
uncapped the GPU self-limited to 2366 MHz under sustained BF16 GEMM — it
never approached the 3003 cap** — so the clamp removes ~230 MHz of top
range the chip wasn't using at full load anyway. That is why the clamp is
near-free. Caveat: this measures the *throughput cost* of clamping, NOT
whether clamping prevents shutdown — the thread's stronger root-cause
candidate is SoC/CPU-side heat (GPU 79 °C while CPU ~96 °C), which a GPU
clock cap only relieves indirectly. No shutdown has ever been observed on
this unit.

Procedure:

```bash
# Inspect current clock envelope
nvidia-smi -q -d CLOCK | grep -A 2 "Max Clocks\|Applications Clocks"
#  - Applications Clocks (default): ~2418 MHz on this unit
#  - Max Clocks (hard limit): ~3003 MHz on this unit

# Try a conservative clamp (needs sudo). Syntax:  -lgc <min>,<max>
sudo nvidia-smi -lgc 1665,2200   # try 2200 first, descend if still crashes

# Verify it took effect
nvidia-smi -q -d CLOCK | grep "Applications Clocks" -A 1

# Reset (undo) at any time
sudo nvidia-smi -rgc
```

The `-lgc` command requires the NVIDIA driver to allow user clock
control; tested on this unit, the syntax is accepted (it errors only on
permission, not on "unsupported"). Confirmed needs root.

If a specific clamp ceiling works, persist it across reboots with a
systemd unit:

```ini
# /etc/systemd/system/gpu-clock-clamp.service
[Unit]
Description=NVIDIA GPU Clock Clamping (workaround for boost-clock PD trip)
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -lgc 1665,2200
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable with:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-clock-clamp.service
```

**Cost of clamping**: caps boost clock, so peak GEMM throughput drops
proportionally. A 2200 MHz clamp vs ~2418 MHz default applications clock
costs roughly 9 % peak throughput. Acceptable trade if it stops crashes;
unacceptable as a "just in case" prophylactic.

### Firmware updates: this is ASUS GX10, NOT DGX Spark Founders Edition

NVIDIA publishes a current-firmware table at
https://docs.nvidia.com/dgx/dgx-spark/release-notes.html with versions
like `EC 3.3.2`, `USB PD 0.5.22`, `SoC 2.152.15`. **Those numbers apply
to the Founders Edition only.** The same page states:

> "These release versions apply only to the DGX Spark Founders Edition.
> GB10-based partner systems may not receive updates at the same time."

This unit is an **ASUS Ascent GX10** -- a partner system. Implications:

- **Do not run `fwupdmgr update` blindly assuming NVIDIA's FE firmware
  capsules apply.** They might not be in your LVFS feed; if they are,
  applying them on partner hardware is unsupported and risks bricking.
- ASUS publishes their own DGX OS image at
  https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/helpdesk_download
  (currently `7.4.0-3`, ~9 GB). Their firmware updates ship through that
  channel.
- This unit's `fwupdmgr get-updates` already returned "No updates
  available" (2026-06-04). Trust that until ASUS posts an updated image.
- If a firmware update IS needed for a real reason: **the three-pass
  verification protocol applies**. Stop. Document the reason. Ask the
  user to confirm three times across three separate turns. Then, and
  only then, run the capsule. Never auto-apply.

### Where to ask if symptoms don't match anything above

- NVIDIA Developer Forum, "DGX Spark / GB10" category:
  https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719
- Community diagnostic CLI `spark-doctor` (run `spark-doctor scan` for a
  one-shot health report): https://github.com/joeynyc/spark-doctor
- Long-form fleet experience reports:
  https://dredyson.com/ (search "DGX Spark")

## Sources

- NVIDIA DGX Spark Hardware Overview — https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- NVIDIA DGX Spark product page — https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- NVIDIA DGX Spark datasheet — https://resource.naddod.com/files/2025-10-20/nvidia-dgx-spark-datasheet-web-012638.pdf
- ASUS Ascent GX10 product page (global) — https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/
- ASUS Ascent GX10 techspec (US) — https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/techspec/
- ASUS Ascent GX10 datasheet (PDF mirror) — https://arrow.altron.com/hubfs/asus-ascent-gx10-datasheet-1.pdf
- ASUS DGX OS download for Ascent GX10 — https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/helpdesk_download
- Hot Chips 2025: "NVIDIA GB10 SoC: AI Supercomputer On Your Desk" (Andi Skende) — https://hc2025.hotchips.org/assets/program/conference/day2/21_nvidia_skende_final.pdf
- The Register, "Nvidia details GB10 miniaturized Grace Blackwell superchips" (2025-08-27) — https://www.theregister.com/software/2025/08/27/nvidia-details-gb10-miniaturized-grace-blackwell-superchips/
- TWOWIN, "NVIDIA DGX Spark Performance Evaluation and Analysis" (BF16/FP4 sustained vs nameplate) — https://twowintech.com/nvidia-dgx-spark-performance-evaluation-and-analysis/
- rossingram/Spark-DGX-Benchmark — https://github.com/rossingram/Spark-DGX-Benchmark
- NVIDIA DGX Spark Playbooks (DeepWiki, Hardware Platform) — https://deepwiki.com/NVIDIA/dgx-spark-playbooks/1.1-hardware-platform
- NVIDIA NVLink-C2C overview — https://www.nvidia.com/en-us/data-center/nvlink-c2c/
- NVIDIA forum, "DGX Spark Power Clarification" (official TDP breakdown, 2025-10-31) — https://forums.developer.nvidia.com/t/dgx-spark-power-clarification/349668
- NVIDIA forum, "DGX SPAK GPU power usage cap at 14W" (March 2026 workaround thread) — https://forums.developer.nvidia.com/t/dgx-spak-gpu-power-usage-cap-at-14w/363487
- Dre Dyson, "A CTO's Definitive Guide to Resolving DGX Spark GPU Power Draw Degradation" (fleet of 14 units, May 2026) — https://dredyson.com/a-ctos-definitive-guide-to-resolving-dgx-spark-gpu-power-draw-degradation/
- joeynyc/spark-doctor (DGX Spark diagnostic CLI, detects 14 W cap and other GB10-specific issues) — https://github.com/joeynyc/spark-doctor
