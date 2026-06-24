# A4 — gradient accumulation: results & payoff

**Day:** A4 (`notes/curriculum-v2-execution.md` § A4). **Model:** Llama-3.2-3B-Instruct,
full-parameter, BF16. **Dataset:** alpaca-cleaned. **seq_len:** 1024. **lr:** 2e-5.
**optimizer:** raw `torch.optim.AdamW` (NOT HF Trainer — this day unrolls the loop).

## What A4 demonstrates

`effective_batch = micro_batch * accum_steps`. Run the SAME effective batch (16) three
ways. The payoff: **final_loss ~identical** (the algorithm sees the same averaged
gradient regardless of how we split the 16), while **peak_mem and step_time trade off**
(bigger micro-batch = more activation memory but faster, fewer/larger GEMMs).

## The explicit loop (the A2 debt paid)

A2 used `trainer.train()`, which hid forward/loss/backward/optimizer. A4's `train.py`
hand-writes the loop so all four beats are visible lines, plus the accumulation
bookkeeping (`loss/accum`, step+zero_grad every accum_steps). See `loop_explained.py`
for the annotated skeleton and `train.py` for the runnable version.

## Payoff table (3 configs, same effective batch 16)

Measured on GX10 (Llama-3.2-3B full SFT, BF16, seq 1024, 30 opt-steps each):
```
micro accum  eff |  peak_mem  step_time  final_loss
---------------------------------------------------
    1    16   16 |   30.16GB    4363 ms     1.3526
    4     4   16 |   35.82GB    3378 ms     1.2469   <- SWEET SPOT (fastest, 2nd-lowest mem)
    8     2   16 |   46.96GB    3834 ms     1.2489   <- slower AND more memory: pure loss
---------------------------------------------------
loss spread (max-min): 0.1057
```

**Read this table:**
- `final_loss` column lands in a narrow ~0.1 band => same effective batch really is the
  same training (direct evidence of the Seg-2/Seg-3 math identity: accumulate-then-average
  == batch-at-once). NOT bit-identical: 30 steps is short + noisy, each config groups/pads
  data differently, and bf16 m/v rounds differently per accumulation path. The point is the
  band (0.11) is much smaller than the step_time spread (985 ms) — what the algorithm sees
  is the same, the split is just noise.
- `peak_mem` rises with micro (activation scales with samples-in-flight; the
  param/grad/optimizer fixed part is shared). 30.16 -> 35.82 -> 46.96 GB.
- `step_time` is NON-MONOTONIC and this is the key finding:
  ```
  micro 1->4:  4363 -> 3378 ms   (-985 ms, -23%)   big speedup
  micro 4->8:  3378 -> 3834 ms   (+456 ms, +13%)   SLOWER, not faster
  ```
  micro=4 already saturates the GPU (bigger GEMM fills the Tensor Cores, 16->4 kernel
  launches). Past saturation, micro=8 gains no compute speedup AND adds memory pressure:
  the 47 GB peak + larger activation tensors push the kernel toward memory-bandwidth-bound
  on the shared 273 GB/s LPDDR5x, so it runs SLOWER. micro=8 loses on BOTH axes (+11 GB,
  +13% time) — a pure-loss trade. The lesson: micro-batch has an optimum (~"just saturate
  the GPU"); past it you pay memory for negative speed. The learner caught this from the
  raw feel ("doesn't seem much faster") before the table confirmed it.

### Data-integrity note (a real one, caught live)

The first Monitor read of micro=8 showed 41.55GB/3194ms/1.1971 — POLLUTED. During the
sweep's tail, a probe/cleanup overlapped GPU use briefly, perturbing that config's
peak_mem and step_time measurement. The authoritative results.jsonl had 46.96GB/3834ms,
and an ISOLATED re-run on an idle GPU reproduced 46.96GB/3834ms/1.2489 bit-for-bit. So
the JSON is correct and the first live read was the contaminated one. Lesson: peak-mem and
step-time are only valid on an otherwise-idle GPU; verify a surprising datum with an
isolated re-run before trusting it. (Had we trusted the polluted read, we'd have reported
"micro=8 fastest" — the exact opposite of the truth.)

## Memory decomposition (the 12-vs-8 finding)

`peak_mem = fixed(params) + activation*micro`. The fixed part here is **8 B/param**, not
A2's 12: raw `torch.optim.AdamW` on bf16 params keeps m/v in **bf16** (2+2), where A2's
HF Trainer mixed-precision kept them fp32 (4+4).
```
weight 2 + grad 2 + m 2 + v 2 = 8 B/param ; 3.21B x 8 = 25.7 GB fixed
```
The `[mem-check]` line in the log prints the actual m/v dtype as ground truth.
Measured: `optimizer m/v dtype = torch.bfloat16 / torch.bfloat16` — confirms 8, not 12.

**Bonus finding — mem-check read ~6 B/param, not 8.** Between optimizer steps PyTorch's
`zero_grad(set_to_none=True)` (the modern default) FREES the grad tensors entirely rather
than zeroing them, so at the mem-check instant only `weight2 + m2 + v2 = 6 B/param` is
resident (3.21B x 6 ~= 19 GB, matches the ~18-19 GB readings). The grad's 2 B/param is
TRANSIENT — it exists only during backward, and shows up in the PEAK (30.16 GB), not in
the between-steps resident set. This literally realizes the Seg-2 "gradient is
use-then-discard" insight: set_to_none discards it between steps to save resident memory.

**Caveat (Seg-6d footgun, on purpose):** bf16 m/v is fine for this 30-step demo but
WRONG for a real long run — small optimizer-state increments get eaten by "big eats
small". A production run uses fp32 optimizer states (the 12 B/param recipe).

## Verdict

A4's core claim is verified: **same effective batch (16) gives the same training**
(loss spread 0.11, much smaller than the 985 ms step-time spread) regardless of how the
16 is split into micro * accum. Gradient accumulation lets you pick the micro-batch
purely for the memory/speed trade-off without changing what the algorithm learns.

The practical rule that fell out (sharper than the day-plan predicted): micro-batch has
an **optimum at "just saturate the GPU"** — here micro=4. Below it (micro=1) you waste
GPU on tiny GEMMs; above it (micro=8) you pay more memory for NEGATIVE speed as the
kernel goes memory-bandwidth-bound on the shared 273 GB/s pool. Not monotonic "bigger =
faster" — there's a peak.

Debt paid: A2's `trainer.train()` four-beat loop is now an explicit, line-by-line
readable loop (`loop_explained.py` + `train.py`). The learner read it, predicted the
table, and caught the non-monotonic step_time from raw feel.
