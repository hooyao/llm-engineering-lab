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


## PyTorch device semantics on GB10 unified memory (2026-06-29 discussion)

This section records the current working model for PyTorch on DGX Spark / ASUS GX10. The copy
benchmark that was planned at the end of this section **has now been run on the box** — see
"Unified-memory behavior measured on this unit (2026-07-06)" below for the numbers, which
replace the earlier estimates.

### Unified memory does not remove PyTorch device semantics

PyTorch still exposes distinct `cpu` and `cuda` devices:

```python
x_cpu = torch.empty(..., device="cpu")
x_gpu = x_cpu.to("cuda")
model = model.to("cuda")
```

The hardware has one 128 GB LPDDR5x pool, but PyTorch's allocator, kernel dispatch, and
autograd logic still distinguish CPU tensors from CUDA tensors. A CPU tensor is not
automatically a CUDA tensor just because the platform is UMA. CUDA kernels still expect
CUDA tensors.

Relevant CUDA concepts, kept distinct:

```text
UVA (Unified Virtual Addressing):
  CPU and GPU pointers live in one virtual address space; the runtime can classify them.

cudaMallocManaged / CUDA Unified Memory:
  managed allocation where the driver handles CPU/GPU access and migration/mapping.

Pinned host memory:
  page-locked CPU memory used for faster or asynchronous transfers.

Hardware-coherent UMA:
  CPU and GPU share one physical memory pool with coherence in hardware.
```

GB10 provides hardware-coherent UMA. PyTorch ordinary CPU tensors are still CPU tensors, and
ordinary CUDA tensors are still CUDA tensors.

### Does `model.to("cuda")` create two copies?

For a plain tensor conversion:

```python
x_cpu = torch.empty(...)
x_cuda = x_cpu.to("cuda")
```

there are two live storages while both tensors are referenced:

```text
CPU tensor storage
CUDA tensor storage
```

On GX10 both consume the same 128 GB LPDDR5x capacity pool, but they are separate
allocations with separate virtual addresses.

For `nn.Module.to("cuda")`, the module conversion is in-place at the module-object level:
parameters and buffers are moved/replaced and the method returns the same module object.
Stable-state memory therefore usually has only the CUDA parameter storage, provided no other
references keep the old CPU tensors alive.

The dangerous part is the load/move peak:

```python
model = AutoModelForCausalLM.from_pretrained(path)  # may instantiate full CPU weights
model.to("cuda")                                    # then allocate CUDA weights
```

During that transition, large models can temporarily require both CPU and CUDA storage. On
GX10 those are not separate host RAM vs VRAM budgets; both count against the same 128 GB
unified pool. A 70B BF16 checkpoint is about 140 GB for one copy, so a naive CPU-load-then-
move path is not viable.

Prefer loading paths that place or stream weights directly to the target device, for example
HF/Accelerate low-CPU-memory and `device_map`-style paths, rather than constructing a full
CPU model and then calling `.to("cuda")`.

### Copy bandwidth: what is known vs unknown

> **Superseded by measurement (2026-07-06).** The "no local measurement yet" list below has
> been filled in — see "Unified-memory behavior measured on this unit (2026-07-06)". Kept for
> the reasoning; the actual numbers (H2D/D2H ~59 GB/s copy-engine, D2D ~114, in-place kernel
> ~198–242, the pinned-is-useless and preallocate-your-host-buffer findings) live in that
> section. The "~136 GB/s payload sanity bound" guess turned out to bound the *kernel* path,
> not the *copy-engine* path, which is lower at ~59 GB/s.

Public hardware numbers for this platform are:

```text
LPDDR5x aggregate bandwidth:          273 GB/s
CPU<->GPU coherent interconnect:      600 GB/s bidirectional (GB10 figure used in this repo)
CUDA copy engines listed by NVIDIA:   2
```

There is no local measurement yet for:

```text
PyTorch CPU tensor -> CUDA tensor `.to("cuda")` bandwidth
CUDA `cudaMemcpy` H2D/D2H bandwidth on GB10
pinned vs pageable transfer difference on GB10
managed/shared direct-access bandwidth on GB10
```

Because source and destination reside in the same LPDDR5x pool, a semantic CPU-to-CUDA copy
is not a PCIe host-DRAM-to-HBM transfer. For large explicit copies, the practical upper
bound is likely DRAM-bandwidth-bound rather than NVLink-C2C-bound, because 273 GB/s memory
bandwidth is below the 600 GB/s coherent interconnect figure. A copy also reads N bytes and
writes N bytes, so a rough payload upper bound from aggregate memory bandwidth is:

```text
273 GB/s / 2 ~= 136 GB/s payload
```

Actual results can be lower due to copy-engine behavior, cache effects, page state,
alignment, concurrent CPU/GPU traffic, and power/thermal limits. Treat the number above as a
sanity bound, not a measured result.

### Benchmark plan once GX10 is reachable — DONE (2026-07-06)

The four measurements planned here were run on the box on 2026-07-06. Results are in the next
section. Kept here as a pointer so the plan and its execution stay linked.

## Unified-memory behavior measured on this unit (2026-07-06)

Driver `580.159.03`, CUDA forward-compat `13.2` (driver 595.58.03), containers
`nvcr.io/nvidia/pytorch:26.04-py3` (PyTorch `2.12.0a0`) for the torch tests and
`nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` (`nvcc 13.2`, `-arch=sm_121`) for the CUDA
C++ probe. Scripts live on the box under `~/uvm-probe/` (`uvm_probe.cu`, `torch_bw.py`,
`torch_d2h.py`, `big_tensor.py`). This section resolves the earlier "known vs unknown" and
"benchmark plan" placeholders with actual data, and corrects two guesses that were wrong.

### Capability flags (ground truth from `cudaGetDeviceProperties`)

```text
name = NVIDIA GB10   compute capability = 12.1 (sm_121)   SMs = 48
totalGlobalMem (CUDA)   = 121.6 GB
total_memory (torch)    = 124546 MB   (~121.6 GB visible to the CUDA allocator)
mem_get_info total      = 130.6 GB    (torch.cuda.mem_get_info, whole pool view)

unifiedAddressing                      = 1
managedMemory                          = 1
concurrentManagedAccess                = 1   # CPU and GPU may access managed mem concurrently
pageableMemoryAccess                   = 1   # a plain malloc() pointer is legal inside a kernel
pageableMemoryAccessUsesHostPageTables = 1   # ATS on: GPU walks the CPU page tables
directManagedMemAccessFromHost         = 0   # <-- see "the one documented-but-false flag" below
```

`nvidia-smi --query-gpu=memory.total` returns `[N/A]` on this box — NVML does not expose a
framebuffer size for the GB10 iGPU. Use `torch.cuda.mem_get_info()` (returns
`free,total`) or `cudaMemGetInfo`, not NVML, for capacity queries.

### Four functional experiments (CUDA C++, `uvm_probe.cu`)

| # | Test | Result | Meaning |
|---|---|---|---|
| A | `cudaMallocManaged`; CPU writes 1.0, GPU kernel `+1`, CPU reads back | CPU reads `2.0`; `hostPtr == devicePtr` (same VA), `pointerAttr.type = managed` | Managed memory is genuinely shared in place — one allocation, one address, both processors see each other's writes after sync |
| B | Plain `malloc()` pointer dereferenced **directly inside a kernel** (no `cudaMemcpy`, no managed alloc) | `err = no error`; CPU immediately sees the kernel's writes | ATS works: system-allocated (`malloc`) memory is directly addressable by the GPU with zero copy |
| C | `cudaMemcpy` H2D, then mutate the source, then read device | device retains the pre-mutation value → **independent copy** | `cudaMemcpy` always produces a real, separate copy — it is **never** silently aliased to zero-copy even though src and dst are the same physical DRAM |
| — | PyTorch: allocate one `bfloat16` CUDA tensor of 85.9 GB | succeeds; `memory_allocated = 85.9 GB`, `mem_get_info` free drops 81.5 → 1.5 GB | A single ordinary `device="cuda"` tensor can occupy most of the 128 GB pool with no special allocator — capacity "just works" |

### Bandwidth (512 MB buffer, aggregate DRAM peak 273 GB/s)

CUDA C++ (`uvm_probe.cu`):

| Path | GB/s | Note |
|---|---:|---|
| `cudaMemcpy` H2D, pageable `malloc` source | 59.3 | one-directional payload |
| `cudaMemcpy` H2D, **pinned** source | 58.9 | **pinning gives no speedup** (no PCIe to hide) |
| `cudaMemcpy` D2H, dev → pinned | 58.9 | symmetric with H2D |
| `cudaMemcpy` D2D, dev → dev | 113.2 | copy-engine, one-directional payload |
| kernel `streamCopy` dev ← dev | 242.5 | R+W summed; ~89% of 273 GB/s peak |
| kernel `streamCopy` dev ← **`malloc` host** (ATS, no memcpy) | 197.8 | R+W summed; GPU reads host memory in place over C2C |

PyTorch `.to()` (`torch_bw.py`, `torch_d2h.py`), same 512 MB:

| Path | GB/s | Note |
|---|---:|---|
| pageable CPU → cuda, `.to("cuda")` | 55.9 | matches the C++ H2D number |
| pinned CPU → cuda, `.to("cuda", non_blocking=True)` | 59.0 | pinning ≈ no gain, as in C++ |
| cuda → cuda, `.clone()` | 114.2 | matches C++ D2D |
| cuda → CPU, `.to("cpu")` (result discarded each call) | 59.5 | true D2H bandwidth; allocator recycles the dst |
| cuda → CPU, `.to("cpu")` (**result held alive** = offload) | **0.1** | **allocation-bound, not bandwidth-bound — see below** |
| cuda → CPU, `.copy_()` into **preallocated reused** host dst | 59.2 | the correct offload pattern; pinning irrelevant |

### Three conclusions that change how we write PyTorch on this box

1. **`cudaMemcpy` / `.to()` is never secretly zero-copy (Exp C).** Semantically it always
   copies. If you want zero-copy CPU/GPU sharing you must use a *different* API (managed
   allocation, or hand a system pointer to a kernel via ATS), not `.to()`. The earlier
   section's device-semantics model is correct: UMA does not collapse `cpu` and `cuda`
   tensors into one storage.

2. **The `.to("cpu")` 0.1 GB/s cliff is an allocation trap, not a hardware limit.** Copying
   device→host measured 59.5 GB/s when the result tensor is *discarded* each call (PyTorch's
   host caching allocator recycles the buffer), but collapsed to **0.1 GB/s when the result is
   held alive** — which is exactly what an offload loop does, since you offload precisely to
   keep the tensor on the host. Holding it blocks allocator recycling, so every call really
   allocates fresh pageable host pages (page-fault + zero-fill). That ~600× gap is destination
   allocation, not transfer. Switching to `.copy_()` into a single preallocated, reused host
   buffer restored 59.2 GB/s. Practical rule for offload loops (ZeRO/FSDP-style optimizer or
   activation spill to the host portion of the pool): **preallocate one host landing buffer and
   `.copy_()` into it every step**; never allocate a new CPU tensor you keep. This dwarfs any
   pinned-vs-pageable consideration. (The originally-reported 0.6 GB/s came from a variant that
   held exactly one prior result; holding several makes the trap sharper at ~0.1.)

3. **Pinned memory buys nothing for transfers here (both C++ and torch).** On a discrete GPU,
   `pin_memory()` lets DMA bypass a staging copy across PCIe. GB10 has no PCIe between CPU and
   GPU, so pinned and pageable H2D/D2H are within noise (58.9 vs 59.3; 59.0 vs 55.9). Skip
   `pin_memory=True` as a transfer optimization on this box — it only adds pinning cost. (It
   may still matter for overlapping copy with compute via streams; that was not tested.)

### Why explicit copies top out at ~59 GB/s while kernels hit ~198–242

The `cudaMemcpy` / `.to()` path uses the 2 copy engines and measured ~59 GB/s one-directional.
A compute kernel that reads host memory in place over ATS reached 197.8 GB/s (R+W summed), and
a pure device-resident kernel reached 242.5 GB/s (~89% of the 273 GB/s aggregate). So on GB10
the fast way to consume host-resident data from the GPU is **not** to `cudaMemcpy` it over and
then compute — it is to let the kernel read it directly (ATS / system-allocated pointer, or a
managed allocation). The explicit-copy path is the slow path here, the inverse of the
discrete-GPU intuition where you copy to VRAM precisely to get bandwidth. This matches the
earlier "~136 GB/s payload sanity bound" reasoning only loosely: the *copy-engine* path is
well below that bound (59, not 136), while the *in-place kernel* path exceeds it because it is
not a copy at all.

### The one documented-but-false flag: `directManagedMemAccessFromHost = 0`

NVIDIA's CUDA Programming Guide (Unified and System Memory) states that on NVLink-C2C + ATS
systems (Grace Hopper / Grace Blackwell), `cudaDevAttrDirectManagedMemAccessFromHost` is 1 —
i.e. GPU-resident *managed* memory can be read by the CPU without migration. On this GB10 the
attribute reads **0**, while `pageableMemoryAccessUsesHostPageTables = 1` confirms ATS is
genuinely on and Exp A/B confirm in-place sharing works functionally.

Working interpretation (flagged as such, not asserted as NVIDIA-confirmed): GB10 is a desktop
GB10 SKU, and its driver appears to keep the `cudaMallocManaged` path on the **traditional UVM
migration model** (`directManaged=0`), while the **system-allocated / ATS path** (plain
`malloc`, ordinary torch CPU tensors handed to a kernel) is the one that is truly in-place and
zero-copy — which Exp B and the 197.8 GB/s in-place kernel read both support. So on this box,
prefer the ATS/system-pointer route over `cudaMallocManaged` when you specifically want
in-place host access from the GPU. Do not rely on the documented Grace-Hopper managed-memory
direct-access behavior here; it did not reproduce.

### PyTorch managed/UVM allocator availability (as of this container)

`nvcr.io/nvidia/pytorch:26.04-py3` ships PyTorch `2.12.0a0`. Its `torch.cuda.memory` exposes
`MemPool` and `use_mem_pool` but **no** managed/UVM context manager. PyTorch main has since
merged one — but as an **internal** helper `torch.cuda._use_uvm()` (underscore-prefixed, not
in `__all__`), backed by a new `_make_uvm_allocator()` that builds a `cudaMallocManaged` +
`cudaMemAdvise(SetPreferredLocation/SetAccessedBy)` allocator via `cuda-python`'s
`cuda.bindings.runtime`, wraps it in `torch._C._cuda_customAllocator`, and drives it through a
`MemPool`. There is **no** public `use_uvm` / `managed_memory` API even on main — checked the
`__all__` list directly (2026-07-06).

Availability checked on this box (2026-07-06):

| Build | `hasattr(torch.cuda, "_use_uvm")` |
|---|---|
| container `2.12.0a0+…nv26.04` | **False** |
| stock nightly `2.14.0.dev20260706+cu130` (aarch64) | **True**, allocates managed mem on GB10 |

Two facts worth keeping:

1. **Stock aarch64 wheels run on GB10.** From PyTorch 2.11, `pip install torch` on aarch64
   defaults to CUDA 13.0 wheels; the nightly `2.14.0.dev+cu130` wheel ran real sm_121 matmuls
   and `_use_uvm()` on this unit with no NVIDIA container. So a plain `pip install --pre torch`
   is a viable path to the newest APIs when the container lags — though the user's default is
   to stay inside the NGC container.
2. **You don't need the nightly to use UVM in the 26.04 container.** Every dependency of
   `_use_uvm` already exists there (`MemPool`, `use_mem_pool`, `torch._C._cuda_customAllocator`,
   and the preinstalled `cuda-python`). `experiments/bench/uvm_pool.py` backports the upstream
   `_make_uvm_allocator` + `_use_uvm` verbatim so `with use_uvm(): ...` works in the current
   container. Self-check on this box (`run-uvm-pool-26.04.sh`, 2026-07-06): a tensor allocated
   inside `use_uvm()` reports `cudaPointerGetAttributes.type = 3 (MANAGED)`, a GPU kernel on it
   is numerically correct, and an ordinary `device="cuda"` tensor stays `type = 2 (device)` for
   contrast. Drop the backport and call `torch.cuda._use_uvm()` directly once a container ships
   a torch build that has it.

In practice this is rarely needed on GB10: ordinary `device="cuda"` tensors already reach
nearly the full 128 GB pool (the 85.9 GB single-tensor test above), so managed memory is a
*shared-pointer / zero-copy / oversubscription* tool here, not a *capacity* tool — and the
upstream docstring itself warns UVM is slower than explicit placement for workloads that fit,
due to page faults.

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
- NVIDIA CUDA Programming Guide, "Unified and System Memory" (ATS, directManagedMemAccessFromHost, Grace Hopper/Blackwell behavior) — https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html
- NVIDIA CUDA Runtime API, Memory Management (`cudaMallocManaged`, managed-memory access rules) — https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
- PyTorch main, `torch/cuda/memory.py` (merged `cudaMallocManaged`/UVM context manager, not yet in the 26.04 container) — https://github.com/pytorch/pytorch/blob/main/torch/cuda/memory.py
- PyTorch issue #124296 / #98481 (requests for native managed-memory allocator support; CUDAPluggableAllocator status) — https://github.com/pytorch/pytorch/issues/124296
