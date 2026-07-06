# GX10 task list

> Central queue for work that needs the ASUS Ascent GX10 / DGX Spark-class box.
> Keep this file short and executable. Detailed learning history stays in
> `notes/progress.md`; detailed hardware facts stay in `notes/hardware-gx10.md`.
>
> Current blocker: this Windows machine cannot reach GX10. Resume this list from a
> GX10-reachable machine.

## Connection / environment assumptions

- Host: `ssh hooyao@192.168.1.200` or the `GX10` SSH alias, depending on the machine.
- Repo on GX10 should be updated with `git pull` before running new tasks.
- Default container for new model work: `nvcr.io/nvidia/pytorch:26.04-py3`.
- Existing notes say modern `transformers`, `datasets`, `accelerate`, and `peft` install
  cleanly in the 26.04 container without legacy pins.
- Sudo still needs password `123`; use the askpass pattern in `notes/progress.md` for
  non-trivial remote sudo commands.

## P0 — Resume Track A blocked metal runs

### 1. A6 LoRA 4-config sweep

**Why:** Finish A6's prediction-vs-measured payoff. Theory is complete; only the GX10 sweep
is missing.

**Inputs / references:**

- `experiments/a06-lora-sweep/predictions.md`
- `experiments/a06-lora-sweep/learning-notes.md`
- `experiments/a06-lora-sweep/teaching-notes.md`
- `notes/curriculum-v2-execution.md` § A6

**Run shape:**

- Base: Llama-3.1-8B-Instruct or Qwen3-8B, whichever is the cleaner local path.
- Dataset: `allenai/tulu-3-sft-mixture`, about 5000 samples. Confirm availability on box;
  download if absent.
- Four configs, all with `alpha/r = 2`:

```text
r=8,  alpha=16,  attn-only   (q,k,v,o)
r=16, alpha=32,  attn-only
r=16, alpha=32,  attn+mlp    (q,k,v,o + gate,up,down)
r=64, alpha=128, attn+mlp
```

**Record:**

- Printed trainable params; compare to predicted `6.82M / 13.63M / 41.94M / 167.77M`.
- `adapter_model.safetensors` byte size; divide by trainable params to infer fp32 vs bf16
  serialization.
- Peak memory.
- Final loss.
- Qualitative generation on about 5 prompts.
- Which config would be shipped and why.

**Write results to:**

- Fill the MEASURED columns in `experiments/a06-lora-sweep/predictions.md`.
- Add/extend a results note under `experiments/a06-lora-sweep/`.
- Append `notes/progress.md` log entry.

### 2. A7 QLoRA 8B run

**Why:** Validate the QLoRA theory against the A6 BF16-LoRA baseline.

**Inputs / references:**

- `experiments/a07-qlora-8b/theory-notes.md`
- `notes/curriculum-v2-execution.md` § A7
- Best A6 config once A6 sweep is complete.

**Implementation:**

- Create `experiments/a07-qlora-8b/train.py`.
- Start from the best A6 LoRA recipe.
- Load the frozen base with:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

**Fallback:**

If `bitsandbytes` fails on aarch64 / sm_121, use the Track A fallback path from the
curriculum: FP8-native model path instead of NF4. Log the exact failure and the fallback
choice.

**Record:**

- Peak memory vs A6 BF16-LoRA.
- Final loss vs A6.
- Whether NF4 imported cleanly on this GB10 stack.
- Any throughput hit from dequantization.

**Write results to:**

- `experiments/a07-qlora-8b/results.md`
- `notes/progress.md`

## P1 — Hardware characterization needed by later decisions

### 3. PyTorch CPU/CUDA copy bandwidth on GB10 unified memory — DONE 2026-07-06

**Status:** Completed 2026-07-06. Results in `notes/hardware-gx10.md` § "Unified-memory
behavior measured on this unit (2026-07-06)" and logged in `notes/progress.md`. Probes ran
ad-hoc on the box under `~/uvm-probe/` (CUDA C++ `uvm_probe.cu` + torch `torch_bw.py` /
`torch_d2h.py` / `big_tensor.py`) rather than as a committed `experiments/bench/` script;
port them into `experiments/bench/copy_bandwidth.py` only if a repeatable regression check is
wanted. Headline numbers: H2D/D2H ≈ 59 GB/s (copy engine), D2D ≈ 114, in-place ATS kernel read
≈ 198, pure device kernel ≈ 242; pinning gives no gain; `.to("cpu")` fresh-dst is an
allocation trap (0.6 → 59.2 GB/s with preallocated `.copy_()`); `cudaMemcpy`/`.to()` is never
zero-copy; one bf16 CUDA tensor allocated 85.9 GB.

**Why:** Answer the open PyTorch/UMA question with measured data instead of inference.

**Reference:** `notes/hardware-gx10.md` § "PyTorch device semantics on GB10 unified memory".

**Create:** `experiments/bench/copy_bandwidth.py`

**Measure:**

```text
pageable CPU tensor -> CUDA tensor .to("cuda")
pinned CPU tensor   -> CUDA tensor .to("cuda", non_blocking=True)
CUDA tensor         -> CPU tensor
optional: managed/shared direct-access benchmark if a small CUDA extension is warranted
```

**Record:**

- Payload GB/s.
- Tensor sizes and dtype.
- Container tag, driver, CUDA, PyTorch version.
- Whether desktop/session/background workloads were active.
- Compare against sanity bounds: 273 GB/s LPDDR5x aggregate, rough copy payload upper bound
  around 136 GB/s if read+write both hit DRAM.

**Write results to:**

- `notes/hardware-gx10.md`, next to the existing GEMM measurements.
- `notes/progress.md`.

## P2 — Continue Track A scale and serving

### 4. A8 QLoRA scale: 14B then 32B

**Why:** Prove the large-model path, not just the 8B path.

**Reference:** `notes/curriculum-v2-execution.md` § A8.

**Run shape:**

```text
Qwen3-8B          BF16 LoRA baseline
Qwen3-14B         QLoRA NF4 if bitsandbytes works, otherwise fallback path
Qwen3-32B-FP8     FP8 + LoRA native path
```

**Record:**

- Peak memory.
- Tokens/s.
- Final loss.
- Where training starts to feel slow on GX10.

**Write results to:** `experiments/a08-qlora-scale/` and `notes/progress.md`.

### 5. A10 vLLM serving benchmark

**Why:** Track A is not complete until a trained adapter can be served and benchmarked.

**Reference:** `notes/curriculum-v2-execution.md` § A10.

**Record:**

- Tokens/s at batch=1.
- Tokens/s at batch=32.
- p50/p99 first-token latency.
- With and without LoRA adapter.
- Note how inference memory differs from training memory: no gradients or optimizer state;
  KV cache dominates.

**Write results to:** `experiments/a10-serve/` and `notes/progress.md`.

## P3 — Track B runs that benefit from GX10

### 6. B5 nanoGPT Shakespeare

**Why:** First end-to-end pretraining run. Small enough to finish quickly on GX10.

**Reference:** `notes/curriculum-v2-execution.md` § B5.

**Record:** annotated `train.py` / `model.py` copy and sampled text under
`experiments/b05-nanogpt/`.

### 7. B6 TinyStories pretraining

**Why:** Main Track B pretraining payoff: a small but real language model trained from
scratch.

**Reference:** `notes/curriculum-v2-execution.md` § B6.

**Run shape:**

- Model: about 33M params, e.g. `n_layer=6`, `n_head=6`, `n_embd=384`.
- Data: TinyStories from `roneneldan/TinyStories`.
- Expect about an overnight run on GX10; verify actual throughput once running.

**Record:** checkpoint, loss curve, and 5 generated stories. Follow with B7 analysis.

## Maintenance rule

When any task above is completed:

1. Update this file: mark the task done or remove it from the active queue.
2. Append a dated entry to `notes/progress.md`.
3. If a hardware measurement changed or was added, update `notes/hardware-gx10.md`.
4. Commit and push before switching machines.
