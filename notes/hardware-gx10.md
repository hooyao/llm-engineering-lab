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
- **No thermal throttling at this duration / load.** Headroom to throttle threshold ~8-10 °C. Clocks stay at P0.
- **Power efficiency ~1.09 TFLOPS/W (GPU die)** -- comparable to H100 SXM and ahead of A100 SXM on the same metric. Real chassis draw (CPU, DRAM controller, NIC, fan) brings it to roughly 130 W on this unit, well below the 240 W adapter ceiling.
- **GEMM size matters.** Below N=8192 the per-call cuBLAS overhead is non-trivial and sustained throughput drops to ~86 TFLOPS. Real LLM workloads with small effective batches (e.g. LoRA 3B at batch 4 / seq 1024) hit the smaller-size regime AND don't issue continuous GEMMs; observed GPU die power for that case was only ~44 W. To actually saturate this GPU, use large effective batch × long seq, or train ≥ 14B models.
- **For experiment planning, budget sustained BF16 = ~95 TFLOPS at large N, ~60 TFLOPS at LLM-typical sizes.** Mark either as `[measured on this unit, single-shot, 2026-06]` -- repeat the benchmark if room temperature changes a lot or the driver moves to a major new release.

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
