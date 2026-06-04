# Curriculum reference — assets, budgets, defaults

> **For the day-by-day execution plan see `curriculum-v2-execution.md`.**
> This file is the static reference: what models/datasets are on disk, the memory
> arithmetic, default LoRA config. Look here when a script needs a number.

The asset list itself is defined in `tools/download_models.py` (single source of
truth for what got downloaded).

## Memory budget — 128 GB unified LPDDR5x (shared CPU+GPU)

Reserve ~12 GB for framework + KV scratch + dataloader → ~116 GB usable. The same
273 GB/s bus feeds CPU and GPU, so "offload to host" buys capacity, not bandwidth.

| Method | bytes/param (weights+grads+opt state) |
|---|---|
| Full SFT, mixed-precision Adam | 16  (fp32 master 4 + m 4 + v 4 + bf16 weight 2 + bf16 grad 2) |
| Full SFT, 8-bit Adam | ~10 |
| LoRA (base frozen, bf16) | 2 (frozen base) + ~0 (adapter ≪ 1%) |
| QLoRA (base NF4 4-bit) | ~0.5 + ~0 |
| FP8 base + LoRA | ~1 + ~0 |

Add activations on top: `≈ seq_len × micro_batch × hidden × layers × bytes`, cut hard
by gradient/activation checkpointing. Worked examples (weights+opt only):

- Full SFT 3B, Adam: `3.2e9 × 16 ≈ 51 GB` → comfortable.
- Full SFT 8B, Adam: `8.0e9 × 16 ≈ 128 GB` → over budget; use 8-bit Adam (~80 GB) + checkpointing.
- LoRA 14B (bf16 base): `14.8e9 × 2 ≈ 30 GB` → large headroom for batch/seq.
- LoRA 32B (FP8 base): `32.8e9 × 1 ≈ 33 GB` → roomy.

Verify, don't guess: `torch.cuda.memory._record_memory_history()` + snapshot, or
`nvidia-smi dmon` / `nsys`.

## Models

### Tier 1 — full SFT + first LoRA (≤8B, three architectures)

| Model (`/models/...`) | Params | Teaches | Method |
|---|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | 1.2B | smoke test; full SFT in minutes | full SFT |
| `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | comfortable full SFT; memory profiling | full SFT |
| `meta-llama/Llama-3.1-8B-Instruct` | 8.0B | Llama-arch SFT at the budget edge | full SFT (8-bit Adam) / LoRA |
| `Qwen/Qwen3-1.7B` | 1.7B | fast LoRA iteration loop | LoRA |
| `Qwen/Qwen3-4B-Instruct-2507` | 4.0B | #2 fine-tune base (2026 benchmarks) | full SFT / LoRA |
| `Qwen/Qwen3-8B` | 8.2B | **primary base** (#1 fine-tune base) | LoRA / 8-bit full SFT |
| `google/gemma-3-4b-it` | 4.3B | third architecture; cross-arch LoRA A/B | LoRA |

### Tier 2 — BF16 LoRA at scale (mid dense)

| Model (`/models/...`) | Params | Teaches | Method |
|---|---|---|---|
| `Qwen/Qwen3-14B` | 14.8B | BF16 LoRA sweet spot; NF4-QLoRA target | LoRA / QLoRA |
| `google/gemma-3-12b-it` | 12.2B | non-Qwen 12B LoRA; arch diversity at scale | LoRA |

### Tier 3 — Blackwell native FP8 path (large dense + MoE)

| Model (`/models/...`) | Params | Teaches | Method |
|---|---|---|---|
| `Qwen/Qwen3-32B-FP8` | 32.8B (FP8) | large dense on the native path, no bitsandbytes | FP8 + LoRA |
| `Qwen/Qwen3-30B-A3B-FP8` | 30.5B total / 3.3B active (FP8) | MoE fine-tuning: full experts resident, sparse compute | FP8 + LoRA |

> On sm_121 the native low-precision path is **NVFP4 / FP8** (Transformer Engine), not
> bitsandbytes NF4. NF4 QLoRA needs a bitsandbytes aarch64 + sm_121 wheel — unverified
> on this unit. Tier 3 ships FP8 so the large-model lessons work regardless.

## Datasets

### SFT (`/models/...`)
| Dataset | Size | Use |
|---|---|---|
| `yahma/alpaca-cleaned` | 52k | smoke-test baseline |
| `databricks/databricks-dolly-15k` | 15k | human-written instructions |
| `HuggingFaceTB/smoltalk` | ~1M | current-standard SFT mix (main) |
| `allenai/tulu-3-sft-mixture` | 939k | current-standard, 7 domains (main) |
| `BelleGroup/train_0.5M_CN` | 500k | Chinese instruction tuning |
| `HuggingFaceH4/ultrachat_200k` | 200k | legacy; A/B vs smoltalk/tulu-3 |

### Preference / DPO (`/models/...`)
| Dataset | Size | Use |
|---|---|---|
| `HuggingFaceH4/ultrafeedback_binarized` | 63k | DPO reference standard |
| `argilla/dpo-mix-7k` | 7k | tiny DPO pipeline starter |
| `nvidia/HelpSteer2` | 21k | attribute-labeled preference data |

## Suggested sequence

See `curriculum-v2-execution.md`. The catalog above is consumed by Track A
(fine-tuning) and the MoE extension; Track B (pretrain from scratch) downloads
its own data (TinyStories) at the time.

## LoRA target modules (reference)

Minimum: `q_proj, k_proj, v_proj, o_proj`. Add `gate_proj, up_proj, down_proj` for capacity.

```python
from peft import LoraConfig
cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
```

Training scripts live under topic dirs (`lora/`, `qlora/`, `dpo/`, ...), added as each
lesson is built — not pre-scaffolded here.
