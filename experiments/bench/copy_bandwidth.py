#!/usr/bin/env python3
"""
GB10 unified-memory copy-bandwidth benchmark (PyTorch level).

Goal: measure what CPU<->CUDA data movement actually costs in PyTorch on the
GB10 unified-memory platform (DGX Spark / ASUS GX10), where "CPU RAM" and "GPU
VRAM" are the same 128 GB LPDDR5x pool. The discrete-GPU intuitions (copy to
VRAM for bandwidth; pin host memory to speed DMA) do not transfer, and this
benchmark shows why with numbers.

What it measures (payload GB/s = bytes moved / wall time; a copy that reads N
and writes N reports N as payload, not 2N):

    1. pageable CPU tensor -> CUDA, x.to("cuda")
    2. pinned   CPU tensor -> CUDA, x.to("cuda", non_blocking=True)
    3. CUDA -> CPU, x.to("cpu")   with the result DISCARDED each call
    4. CUDA -> CPU, x.to("cpu")   with the result HELD (kept alive, as an
                                  offload loop does) -> allocation trap
    5. CUDA -> CPU, dst.copy_(x) into PREALLOCATED pinned host buffer
    6. CUDA -> CUDA, x.clone()                      [D2D reference]
    7. capacity check: allocate one large bf16 CUDA tensor and read mem_get_info

Three findings this reproduces (see notes/hardware-gx10.md,
"Unified-memory behavior measured on this unit (2026-07-06)"):

    - H2D / D2H over the copy engines tops out ~59 GB/s; D2D ~114 GB/s. These
      are far below the 273 GB/s aggregate because cudaMemcpy uses the 2 copy
      engines, not the full memory system. (An in-place ATS kernel read hits
      ~198 GB/s — that path is measured by uvm_probe.cu, not here.)
    - pin_memory() gives ~no speedup: there is no PCIe between CPU and GPU to
      hide, so pinned == pageable within noise. Skip it as a transfer opt here.
    - The offload allocation trap: x.to("cpu") measured ~59 GB/s when the result
      is discarded (PyTorch's host caching allocator recycles the buffer, no
      page fault) but collapses to ~0.1 GB/s when the result is HELD alive —
      which is exactly what an offload loop does (you offload precisely to keep
      the tensor on the host). The cost is destination allocation (page-fault +
      zero-fill of fresh pageable host pages), NOT transfer bandwidth: copy_()
      into a preallocated, reused host buffer stays at ~59 GB/s. Rule for
      offload loops: preallocate ONE host buffer and copy_ into it every step;
      never allocate a new CPU tensor you keep.

Run inside the NVIDIA PyTorch 26.04 container:
    bash experiments/bench/run-copy-bw-26.04.sh
    bash experiments/bench/run-copy-bw-26.04.sh --mb 1024 --iters 50

For the CUDA-C++ functional probe (managed sharing, ATS malloc-in-kernel,
memcpy-is-not-zero-copy, in-place kernel bandwidth), see uvm_probe.cu /
run-uvm-probe.sh in this directory.
"""

import argparse
import statistics
import time

import torch


def _time_op(fn, iters: int) -> float:
    """Median seconds over `iters` calls of fn(), with warmup + sync bracketing."""
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench_copies(nbytes: int, iters: int) -> list[dict]:
    """Run the copy-path matrix on a buffer of `nbytes` bytes (float32)."""
    n = nbytes // 4  # float32 elements
    dev = torch.device("cuda:0")

    src_page = torch.empty(n, dtype=torch.float32)                 # pageable host
    src_pin = torch.empty(n, dtype=torch.float32).pin_memory()     # pinned host
    src_gpu = torch.empty(n, dtype=torch.float32, device=dev)      # device
    dst_pin = torch.empty(n, dtype=torch.float32).pin_memory()     # prealloc pinned

    def gbs(med_s: float) -> float:
        return nbytes / med_s / 1e9  # payload GB/s

    # The offload allocation trap: holding each .to("cpu") result alive prevents
    # PyTorch's host caching allocator from recycling the buffer, so every call
    # really allocates fresh pageable host pages (page-fault + zero-fill). This
    # is what an offload loop does — it keeps the offloaded tensor. Discarding
    # the result instead lets the allocator recycle and hides the cost.
    held: list[torch.Tensor] = []

    def held_to_cpu():
        held.append(src_gpu.to("cpu"))
        if len(held) > 3:
            held.pop(0)  # keep a few alive so recycling stays blocked

    rows = [
        ("pageable CPU -> cuda   .to()",
         gbs(_time_op(lambda: src_page.to(dev), iters))),
        ("pinned   CPU -> cuda   .to(non_blocking)",
         gbs(_time_op(lambda: src_pin.to(dev, non_blocking=True), iters))),
        ("cuda -> CPU  .to('cpu')  [result DISCARDED]",
         gbs(_time_op(lambda: src_gpu.to("cpu"), iters))),
        ("cuda -> CPU  .to('cpu')  [result HELD = offload]",
         gbs(_time_op(held_to_cpu, iters))),
        ("cuda -> CPU  .copy_(nb)  [prealloc reused pinned]",
         gbs(_time_op(lambda: dst_pin.copy_(src_gpu, non_blocking=True), iters))),
        ("cuda -> cuda .clone()    [D2D]",
         gbs(_time_op(lambda: src_gpu.clone(), iters))),
    ]

    held.clear()
    del src_page, src_pin, src_gpu, dst_pin
    torch.cuda.empty_cache()
    return [{"path": p, "gbs": g} for p, g in rows]


def capacity_check(gb: float) -> dict:
    """Allocate one large bf16 CUDA tensor; report mem_get_info before/after."""
    free0, total0 = torch.cuda.mem_get_info()
    n = int(gb * (1024 ** 3)) // 2  # bf16 = 2 bytes/elem
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda:0")
    x.fill_(1.0)
    torch.cuda.synchronize()
    alloc_gb = x.numel() * 2 / 1e9
    free1, total1 = torch.cuda.mem_get_info()
    del x
    torch.cuda.empty_cache()
    return {
        "requested_gb": gb,
        "allocated_gb": alloc_gb,
        "free_before_gb": free0 / 1e9,
        "free_after_gb": free1 / 1e9,
        "total_gb": total0 / 1e9,
    }


def gpu_snapshot(label: str) -> None:
    """Print GPU state from nvidia-smi (NVML memory.total is N/A on GB10, so skip it)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,temperature.gpu,power.draw,clocks.current.graphics,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        cols = [c.strip() for c in out.split(",")]
        print(f"  [{label}]  name={cols[0]}  temp={cols[1]}C  pwr={cols[2]}W  "
              f"gclk={cols[3]}MHz  util={cols[4]}%")
    except Exception as e:
        print(f"  [{label}]  (nvidia-smi snapshot failed: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=512,
                    help="Copy buffer size in MiB (float32).")
    ap.add_argument("--iters", type=int, default=30,
                    help="Timed iterations per copy path (median reported).")
    ap.add_argument("--cap-gb", type=float, default=80.0,
                    help="Size of the single-tensor capacity check, in GB. "
                         "Set 0 to skip (e.g. when the pool is already full).")
    args = ap.parse_args()

    print("=" * 78)
    print("GB10 unified-memory copy-bandwidth benchmark")
    print("=" * 78)
    print(f"torch       : {torch.__version__}")
    print(f"cuda build  : {torch.version.cuda}")
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"compute cap : {torch.cuda.get_device_capability(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"mem_get_info: free={free/1e9:.1f} GB  total={total/1e9:.1f} GB")
    print(f"buffer      : {args.mb} MiB float32, {args.iters} iters/path")
    print()

    gpu_snapshot("idle")
    print()

    print("-" * 78)
    print("COPY PATHS (payload GB/s; buffer reads N and writes N, reported as N)")
    print("-" * 78)
    rows = bench_copies(args.mb * 1024 * 1024, args.iters)
    for r in rows:
        print(f"  {r['path']:<52} = {r['gbs']:6.1f} GB/s")

    if args.cap_gb > 0:
        print()
        print("-" * 78)
        print(f"CAPACITY CHECK (one {args.cap_gb:.0f} GB bf16 CUDA tensor, no special allocator)")
        print("-" * 78)
        cap = capacity_check(args.cap_gb)
        print(f"  allocated   : {cap['allocated_gb']:.1f} GB  (requested {cap['requested_gb']:.0f})")
        print(f"  free before : {cap['free_before_gb']:.1f} GB")
        print(f"  free after  : {cap['free_after_gb']:.1f} GB")
        print(f"  pool total  : {cap['total_gb']:.1f} GB")

    print()
    print("=" * 78)
    print("READING THE RESULT")
    print("=" * 78)
    print("  - H2D/D2H ~59, D2D ~114: copy engines, not the 273 GB/s memory system.")
    print("  - pinned ~= pageable: no PCIe to hide, so pin_memory() buys nothing here.")
    print("  - .to('cpu') DISCARDED ~59 but HELD ~0.1: the gap is host allocation, not")
    print("    bandwidth. An offload loop HOLDS the result, so it hits the slow path.")
    print("    Fix: copy_() into ONE preallocated host buffer, reused every step (~59).")
    print("  - Capacity 'just works': one ordinary cuda tensor spans most of the pool.")
    print("  - For in-place host reads from the GPU (~198 GB/s over ATS), see uvm_probe.cu;")
    print("    the fast path on GB10 is to let a kernel read host memory, not to copy it.")


if __name__ == "__main__":
    main()
