# Curriculum — what's on the drive and what each piece teaches

The asset list itself is defined in `tools/download_models.py` (single source of truth).
This file maps each downloaded model/dataset to the lesson it serves, the fine-tuning
method, and a rough memory budget on the GX10.

## Setup (once per session)

Models synced from the portable drive land at `~/models/<org>/<model>` (see
`bootstrap-gx10.md` Phase 3 for the rsync, Phase 7 for the container). Launch the
training container, which bind-mounts them read-only at `/models`:

```bash
cd ~/fine-tuning
bash tools/launch_pytorch.sh                 # mounts ~/models -> /models:ro
## inside container — core stack:
pip install -U transformers peft datasets trl accelerate
```

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

1. **Full SFT mechanics** — `Llama-3.2-1B` on `alpaca-cleaned` (smoke), then `Llama-3.2-3B`
   on `smoltalk`. Profile memory; turn gradient checkpointing on/off and watch the delta.
2. **LoRA** — `Qwen3-8B` on `tulu-3-sft-mixture`. Sweep rank/alpha/target modules; record
   trainable-param % and adapter-checkpoint size. Repeat on `gemma-3-4b-it` for a cross-arch read.
3. **QLoRA vs FP8** — `Qwen3-14B` NF4 (if bitsandbytes works on sm_121) against
   `Qwen3-32B-FP8` on the native path. Compare peak memory and tokens/s.
4. **MoE** — `Qwen3-30B-A3B-FP8`. Adapt attention only vs. including router/experts; note
   3.3B active params (cheap compute) but the full 30B must stay resident.
5. **DPO** — pipeline on `dpo-mix-7k`, then real run on `ultrafeedback_binarized` over a
   `Qwen3-8B` SFT checkpoint.

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
