# A7 — QLoRA theory notes (NF4, double quantization, paged optimizer)

> Status: theory started on 2026-06-29 while GX10 was unreachable. No metal run yet.
> Next implementation step is still `experiments/a07-qlora-8b/train.py`: reuse the best A6
> LoRA config, load the frozen base in 4-bit, and compare peak memory + final loss against A6.

## One-line bridge from A6

A6 LoRA freezes the base model and trains small adapter matrices. A7 QLoRA keeps that LoRA
training setup, but stores the frozen base weights in 4-bit form.

```text
LoRA:   W frozen in bf16, train small A/B
QLoRA:  W frozen in NF4 4-bit storage, train small A/B
```

QLoRA is not 4-bit training of all parameters. The 4-bit part is the frozen base weight
storage.

## What gets quantized

The learner correctly identified the target: QLoRA quantizes the frozen base `W`, not the
LoRA adapter as the primary idea.

```text
base W:
  stored as NF4 4-bit codes + quantization metadata
  no gradient
  no Adam m/v

LoRA A/B:
  trainable
  normally trained in a bf16/fp32 optimizer path
  has gradients and optimizer state
```

For one adapted linear layer, the forward path remains the LoRA path:

```text
output = W*x + (alpha/r) * B*A*x
```

The difference is how `W` is stored and materialized for compute:

```text
1. W is stored as NF4 codes plus metadata     # about 0.5 byte/param for the codes
2. W blocks are dequantized to bf16 temporarily
3. matmul uses bf16 compute
4. temporary dequantized blocks are discarded
```

So the important distinction is:

```text
NF4  = storage format
bf16 = compute dtype
```

## NF4

NF4 means NormalFloat4. It is a 4-bit format with 16 non-uniform code points chosen for
normally distributed values. Pretrained weights are roughly concentrated around zero, so
uniformly spaced 4-bit bins waste precision in the tails and underserve the dense region
near zero.

```text
uniform int4:
  code points are evenly spaced across a range

NF4:
  code points are denser near zero and sparser in the tails
```

This is why NF4 is a better fit for storing frozen pretrained base weights than a plain
uniform int4 format, at the same 4-bit code budget.

## The BitsAndBytesConfig lines

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

Interpretation:

```text
load_in_4bit=True:
  load the frozen base weights in 4-bit form

bnb_4bit_quant_type="nf4":
  use NF4 as the 4-bit storage codebook for those base weights

bnb_4bit_compute_dtype=torch.bfloat16:
  dequantize blocks to bf16 and run the matmul in bf16

bnb_4bit_use_double_quant=True:
  quantize the quantization constants / scales to reduce metadata overhead
```

Do not read the first line as "LoRA adapter is 4-bit". The main 4-bit storage target is
the frozen base model.

## Double quantization

Blockwise quantization stores more than the 4-bit codes. A typical block has:

```text
W_block ~= scale * code
```

The codes are the 4-bit NF4 values; `scale` maps the normalized codebook to the actual
magnitude range of that block. Since there are many blocks, scales become metadata overhead.

Double quantization applies another quantization step to those scales:

```text
first quantization:   W values -> NF4 codes + scales
second quantization:  scales   -> quantized scales
```

It does not turn the main weight codes from 4-bit into 2-bit. It reduces quantization
metadata, not the primary 0.5 byte/param code payload.

## Paged optimizer

Paged optimizer addresses training-time memory spikes. It is not the mechanism that makes
base weights 4-bit.

In QLoRA, optimizer state exists for the LoRA trainable parameters, not for the frozen base:

```text
base W:
  no grad, no Adam m/v

LoRA A/B:
  grad + Adam m/v + optimizer-step temporaries
```

Paged optimizer can move optimizer state through CPU/unified memory when the GPU allocator
is under pressure. On GX10 / DGX Spark, CPU and GPU share the same 128 GB LPDDR5x pool, so
this helps capacity pressure more than it helps bandwidth. The same 273 GB/s aggregate DRAM
bandwidth feeds CPU and GPU traffic.

## Memory arithmetic

For full-parameter mixed-precision AdamW SFT, a common floor is:

```text
weight 2 + grad 2 + Adam m 4 + Adam v 4 = 12 bytes/param
```

QLoRA avoids applying that floor to the base model because the base is frozen:

```text
QLoRA training floor ~= base_params * 0.5
                      + LoRA_trainable_params * 12
                      + activation
                      + framework / temporary overhead
```

For a 70B base:

```text
base NF4 codes ~= 70B * 0.5 byte = 35 GB
```

If the LoRA adapter is about 0.5-1.0% of the base:

```text
0.5% trainable: 350M params * 12 bytes ~= 4.2 GB
1.0% trainable: 700M params * 12 bytes ~= 8.4 GB
```

This is why 70B QLoRA can be plausible in a 128 GB unified-memory box, while 70B full SFT is
not close:

```text
70B full SFT floor ~= 70B * 12 bytes = 840 GB
70B BF16 weights only ~= 70B * 2 bytes = 140 GB
70B NF4 base floor ~= 35 GB, plus adapter training state and activation
```

## Activation boundary

QLoRA solves base-weight storage. It does not make the backward activation cheap.

The activation term still scales with:

```text
batch * seq_len * hidden * layers
```

The activation tensors are still full-size bf16-ish tensors because forward/backward compute
runs through the dequantized path and gradients must propagate through frozen layers to reach
earlier LoRA adapters. Therefore large-model QLoRA still needs:

```text
gradient checkpointing
small micro-batch
gradient accumulation
```

## Minimal mental model

```text
LoRA:
  freeze W in bf16
  train A/B

QLoRA:
  freeze W in NF4 4-bit storage
  dequantize W blocks to bf16 for compute
  train A/B the same way

NF4:
  non-uniform 4-bit codebook matched to normally distributed weights

double quantization:
  quantize the quantization constants / scales

paged optimizer:
  manage optimizer-state memory spikes; on GX10 this buys capacity, not new bandwidth

activation:
  still full-size; QLoRA does not remove checkpointing pressure
```
