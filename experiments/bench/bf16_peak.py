#!/usr/bin/env python3
"""
GB10 BF16 GEMM peak / sustained benchmark.

Goal: find out what the GB10 actually delivers in sustained BF16 matmul
on this physical unit (chassis, ambient temp, current driver), independent
of any training framework overhead.

Method: a single large square GEMM (N x N) @ (N x N) in BF16, in a tight
Python loop with cuda.synchronize() bracketing the wall clock. Total FLOPs
per matmul = 2 * N^3. We warm up first (GPU clock ramp + cuBLAS algo
selection) and then time a long enough window to see sustained behavior.

Outputs:
    - per-matrix-size: avg / median / p99 wall time, TFLOPS, GB/s mem BW
    - the size that maximized sustained TFLOPS

Compare to:
    - GB10 nameplate BF16 dense: ~125 TFLOPS
    - Independent reports (Carmack et al.): ~60 TFLOPS sustained, likely
      thermal/power capped in the 1.13 L SFF chassis

Run inside the NVIDIA PyTorch container started by tools/launch_pytorch.sh,
or via experiments/smoke-test/run.sh pattern.

Usage:
    python experiments/bench/bf16_peak.py
    python experiments/bench/bf16_peak.py --sizes 2048,4096,8192,16384
    python experiments/bench/bf16_peak.py --warmup 50 --iters 500
    python experiments/bench/bf16_peak.py --duration 60   # run each size for N seconds
"""

import argparse
import statistics
import time

import torch


def bench_size(n: int, warmup: int, iters: int | None, duration_s: float | None) -> dict:
    """
    Run (N x N) BF16 @ BF16 GEMM in a loop.

    If `duration_s` is set, ignore `iters` and run for that wall-clock window
    (better for sustained / thermal characterization).
    """
    dev = torch.device("cuda:0")
    a = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    b = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    c = torch.empty(n, n, dtype=torch.bfloat16, device=dev)

    flops_per_iter = 2.0 * n * n * n        # one matmul = 2*N^3 FLOPs
    bytes_per_iter = 3.0 * n * n * 2        # 3 matrices BF16 (2 read + 1 write)

    # ---- warmup: lets clocks ramp, cuBLAS pick its algo, allocator settle ----
    for _ in range(warmup):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()

    # ---- timed loop ----
    times: list[float] = []
    t_start = time.perf_counter()

    if duration_s is not None:
        deadline = t_start + duration_s
        i = 0
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            torch.matmul(a, b, out=c)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            i += 1
    else:
        assert iters is not None
        for _ in range(iters):
            t0 = time.perf_counter()
            torch.matmul(a, b, out=c)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    total_wall = time.perf_counter() - t_start

    times_sorted = sorted(times)
    avg = statistics.mean(times)
    med = statistics.median(times)
    p99 = times_sorted[int(0.99 * (len(times) - 1))]
    best = times_sorted[0]

    sustained_tflops = (flops_per_iter * len(times)) / total_wall / 1e12
    peak_tflops_best = flops_per_iter / best / 1e12
    peak_tflops_med  = flops_per_iter / med  / 1e12
    mem_bw_gbs       = bytes_per_iter / med / 1e9       # rough; matmul reuses operands heavily

    del a, b, c
    torch.cuda.empty_cache()

    return {
        "n": n,
        "iters": len(times),
        "total_s": total_wall,
        "avg_ms": avg * 1000,
        "med_ms": med * 1000,
        "p99_ms": p99 * 1000,
        "best_ms": best * 1000,
        "sustained_tflops": sustained_tflops,
        "peak_tflops_best": peak_tflops_best,
        "peak_tflops_med":  peak_tflops_med,
        "mem_bw_gbs": mem_bw_gbs,
    }


def fmt_row(r: dict) -> str:
    return (
        f"  N={r['n']:>5}  "
        f"iters={r['iters']:>5}  "
        f"med={r['med_ms']:7.2f} ms  "
        f"p99={r['p99_ms']:7.2f} ms  "
        f"sustained={r['sustained_tflops']:6.1f} TFLOPS  "
        f"peak(med)={r['peak_tflops_med']:6.1f}  "
        f"peak(best)={r['peak_tflops_best']:6.1f}"
    )


def gpu_snapshot(label: str) -> None:
    """Print GPU state from nvidia-smi (works regardless of pynvml availability)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        cols = [c.strip() for c in out.split(",")]
        print(f"  [{label}]  name={cols[0]}  temp={cols[1]}C  pwr={cols[2]}W  "
              f"gclk={cols[3]}MHz  mclk={cols[4]}MHz  util={cols[5]}%")
    except Exception as e:
        print(f"  [{label}]  (nvidia-smi snapshot failed: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2048,4096,8192,12288,16384",
                    help="Comma-separated matmul N values.")
    ap.add_argument("--warmup", type=int, default=20,
                    help="Warmup iterations per size (clock ramp + algo selection).")
    ap.add_argument("--iters", type=int, default=200,
                    help="Timed iterations per size (ignored if --duration set).")
    ap.add_argument("--duration", type=float, default=None,
                    help="If set, run each size for this many seconds (overrides --iters). "
                         "Use 30+ for sustained / thermal measurements.")
    args = ap.parse_args()

    print("=" * 78)
    print("GB10 BF16 GEMM benchmark")
    print("=" * 78)
    print(f"torch       : {torch.__version__}")
    print(f"cuda build  : {torch.version.cuda}")
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"compute cap : {torch.cuda.get_device_capability(0)}")
    print(f"bf16 ok     : {torch.cuda.is_bf16_supported()}")
    print(f"sizes       : {args.sizes}")
    if args.duration:
        print(f"mode        : sustained, {args.duration}s per size")
    else:
        print(f"mode        : fixed, warmup={args.warmup} iters={args.iters} per size")
    print()

    gpu_snapshot("idle")

    # Allow the implicit-TF32 matmul; for BF16 inputs this is a no-op but harmless.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sizes = [int(s) for s in args.sizes.split(",")]
    results: list[dict] = []
    for n in sizes:
        print(f"\n[bench] N={n}...")
        gpu_snapshot("pre ")
        r = bench_size(n, args.warmup, None if args.duration else args.iters, args.duration)
        gpu_snapshot("post")
        print(fmt_row(r))
        results.append(r)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        print(fmt_row(r))
    best = max(results, key=lambda r: r["sustained_tflops"])
    print()
    print(f"Best sustained:  N={best['n']}  {best['sustained_tflops']:.1f} TFLOPS BF16")
    print()
    print("Reference:")
    print("  - GB10 nameplate BF16 dense (NVIDIA spec)  : ~125 TFLOPS")
    print("  - Independent measurement (Carmack)         : ~60 TFLOPS sustained")
    print("  - If your result is well under 60, suspect : thermal throttle,")
    print("    other GPU users, framework overhead. Re-run with --duration 60.")


if __name__ == "__main__":
    main()
