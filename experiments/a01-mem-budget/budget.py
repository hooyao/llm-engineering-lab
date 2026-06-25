"""A1 — memory budget calculator for fine-tuning on the GX10 (128 GB unified).

The only theory day in Track A. Every later decision ("does Llama-8B full SFT
fit?", "how long a sequence can I LoRA a 14B at?") is this arithmetic. Compute the
answer instead of trial-and-error launching.

Three terms, summed:

    total = param_bytes + optimizer/grad/master state + activation_bytes

The per-param byte counts are aligned with `notes/curriculum.md` § Memory budget,
which is the source of truth this file is checked against (see the table printed by
`main()` and the asserts in `_selftest()`).

Run:  python budget.py            # prints the comparison table
      python budget.py --test     # checks against curriculum.md worked examples
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

GB = 1024**3  # bytes per GiB (binary); curriculum.md figures are ~this scale


# --------------------------------------------------------------------------- #
# dtype sizes — bytes to store ONE number at a given precision.               #
# This is the per-number cost factor; everything downstream multiplies by it. #
# --------------------------------------------------------------------------- #
DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "tf32": 4.0,   # stored as 4 bytes; the 19-bit mantissa is a compute detail
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "nf4": 0.5,    # 4-bit; QLoRA base weights
}


def param_bytes(num_params: int, dtype: str) -> float:
    """Bytes to store `num_params` weights at `dtype`.

    The single multiply the whole calculator is built on.
        param_bytes(3_200_000_000, "bf16") == 6.4e9  (~6 GB)
    """
    return num_params * DTYPE_BYTES[dtype]


# --------------------------------------------------------------------------- #
# Optimizer + training state, in bytes PER TRAINABLE PARAM.                    #
#                                                                             #
# "Trainable" matters: for LoRA only the adapter params carry optimizer       #
# state, which is why LoRA's optimizer cost is ~0 against the frozen base.    #
#                                                                             #
# The headline number is full-SFT mixed-precision Adam = 16 B/param:          #
#   bf16 weight 2 + bf16 grad 2 + fp32 master 4 + Adam m 4 + Adam v 4 = 16    #
# The fp32 master weight is the piece people forget: bf16 has ~7 bits of      #
# precision, so lr*grad updates are often too small to survive being added to #
# a bf16 weight. Keep an fp32 master copy, accumulate the update there, cast  #
# back to bf16 for the next forward. That extra 4 bytes is the 12 -> 16 jump. #
# --------------------------------------------------------------------------- #
# bytes/trainable-param for the optimizer MOMENTS ONLY (not weight/grad/master)
_OPT_MOMENT_BYTES: dict[str, float] = {
    "adamw": 8.0,        # m (fp32 4) + v (fp32 4)
    "adamw_8bit": 2.0,   # m,v quantized to 8-bit -> 1 byte each
    "lion": 4.0,         # single momentum buffer, fp32
    "sgd": 0.0,          # vanilla SGD: no moment state
    "sgd_momentum": 4.0, # one fp32 velocity buffer
}


def optimizer_bytes(num_trainable_params: int, optimizer: str) -> float:
    """Bytes of *optimizer moment* state for `num_trainable_params`.

    This is ONLY the m/v (or momentum) buffers — NOT the weight, gradient, or
    fp32 master copy. Those are added by `training_state_bytes` so each piece is
    visible separately. Kept as its own function because A1's spec asks for it
    and because LoRA's whole point is shrinking *this* term to ~0.
        optimizer_bytes(n, "adamw")      == 8 * n
        optimizer_bytes(n, "adamw_8bit") == 2 * n
    """
    return num_trainable_params * _OPT_MOMENT_BYTES[optimizer]


def training_state_bytes(
    num_trainable_params: int,
    *,
    optimizer: str = "adamw",
    grad_dtype: str = "bf16",
    master_dtype: str | None = "fp32",
) -> float:
    """Per-trainable-param training overhead = grad + (optional fp32 master) + moments.

    Does NOT include the weight itself — that is `param_bytes` on the full param
    count (which for LoRA/QLoRA is the *frozen base*, stored once at its own
    dtype). Splitting it this way lets the table show "frozen base" and "trainable
    overhead" as separate columns.

    Default (adamw + bf16 grad + fp32 master) gives the curriculum's 14 B/param of
    *overhead* on top of the 2 B bf16 weight = 16 B/param total for full SFT.
        master_dtype=None drops the fp32 master (pure-bf16 recipe -> 12 B total).
    """
    grad = num_trainable_params * DTYPE_BYTES[grad_dtype]
    master = 0.0 if master_dtype is None else num_trainable_params * DTYPE_BYTES[master_dtype]
    moments = optimizer_bytes(num_trainable_params, optimizer)
    return grad + master + moments


# --------------------------------------------------------------------------- #
# Activations — the intermediate tensors the forward pass saves so the         #
# backward pass can compute weight gradients (see backprop-primer.md). Scales  #
# with how much data is in flight (seq_len x batch) and how deep/wide the net. #
#                                                                             #
# The multiplier captures "how many live intermediate tensors per block":      #
#   checkpointing ON  -> ~1x  (store only each block's input; recompute rest)  #
#   checkpointing OFF  -> ~6x (attn scores, qkv, mlp hidden, norms, ... alive) #
# This is an ESTIMATE (the real factor depends on arch/impl); A1 only needs it #
# to land within ~20% of measured, and to show the checkpointing trade.        #
# --------------------------------------------------------------------------- #
_ACT_MULT_NO_CKPT = 6.0
_ACT_MULT_CKPT = 1.0


def activation_bytes(
    seq_len: int,
    batch: int,
    hidden: int,
    layers: int,
    dtype: str = "bf16",
    checkpointing: bool = False,
) -> float:
    """Estimated activation memory for a transformer forward pass held for backward.

    activation = seq_len * batch * hidden * layers * dtype_bytes * mult
    where mult ~6 without checkpointing, ~1 with it.
        Llama-3B @ seq=2048 batch=4, no ckpt -> see table in main().
    """
    mult = _ACT_MULT_CKPT if checkpointing else _ACT_MULT_NO_CKPT
    return seq_len * batch * hidden * layers * DTYPE_BYTES[dtype] * mult


# --------------------------------------------------------------------------- #
# Model + run specs, and the full-budget assembler.                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Model:
    name: str
    params: int
    hidden: int
    layers: int


# A few from notes/curriculum.md § Models (Tier 1/2). hidden/layers from configs.
MODELS = [
    Model("Llama-3.2-1B", 1_240_000_000, hidden=2048, layers=16),
    Model("Llama-3.2-3B", 3_210_000_000, hidden=3072, layers=28),
    Model("Llama-3.1-8B", 8_030_000_000, hidden=4096, layers=32),
]


def full_sft_bytes(
    m: Model,
    seq_len: int,
    batch: int,
    *,
    optimizer: str = "adamw",
    weight_dtype: str = "bf16",
    master_dtype: str | None = "fp32",
    checkpointing: bool = False,
) -> dict[str, float]:
    """Full-parameter SFT: every param is trainable. Returns a breakdown dict."""
    weights = param_bytes(m.params, weight_dtype)
    overhead = training_state_bytes(
        m.params, optimizer=optimizer, grad_dtype="bf16", master_dtype=master_dtype
    )
    acts = activation_bytes(seq_len, batch, m.hidden, m.layers, checkpointing=checkpointing)
    return {"weights": weights, "train_overhead": overhead, "activations": acts,
            "total": weights + overhead + acts}


def lora_bytes(
    m: Model,
    seq_len: int,
    batch: int,
    *,
    rank: int = 16,
    base_dtype: str = "bf16",
    optimizer: str = "adamw",
    target_frac: float = 0.01,  # adapter params as a fraction of base (~1% typical)
    checkpointing: bool = False,
) -> dict[str, float]:
    """LoRA/QLoRA: base frozen (stored once at base_dtype), only adapter trains.

    `target_frac` approximates "adapter params / base params" so optimizer state
    is computed on the small slice. For a real run, compute the exact adapter
    count from rank x (in+out dims) x num_target_modules; ~1% is the right ballpark
    for r=16 on attn+mlp and is plenty for the comfortable/marginal/OOM verdict.
    """
    base = param_bytes(m.params, base_dtype)          # frozen, no grad/opt state
    adapter_params = int(m.params * target_frac)
    adapter_weights = param_bytes(adapter_params, "bf16")
    adapter_overhead = training_state_bytes(
        adapter_params, optimizer=optimizer, grad_dtype="bf16", master_dtype="fp32"
    )
    acts = activation_bytes(seq_len, batch, m.hidden, m.layers, checkpointing=checkpointing)
    return {"frozen_base": base, "adapter": adapter_weights + adapter_overhead,
            "activations": acts, "total": base + adapter_weights + adapter_overhead + acts}


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
USABLE_GB = 116.0  # 128 GB pool minus ~12 GB framework/KV/dataloader reserve


def _verdict(total_bytes: float) -> str:
    gb = total_bytes / GB
    if gb < 0.7 * USABLE_GB:
        return "comfortable"
    if gb <= USABLE_GB:
        return "marginal"
    return "WILL OOM"


def main() -> None:
    seq_len, batch = 2048, 4
    print(f"GX10 budget: {USABLE_GB:.0f} GB usable (128 - ~12 reserve)")
    print(f"Config: seq_len={seq_len}, batch={batch}, AdamW, bf16 weights+grad, fp32 master\n")

    header = f"{'model':<14}{'method':<22}{'ckpt':<6}{'weights':>9}{'train':>9}{'acts':>8}{'TOTAL':>9}  verdict"
    print(header)
    print("-" * len(header))

    def row(name, method, ckpt, b):
        w = b.get("weights", b.get("frozen_base", 0)) / GB
        t = b.get("train_overhead", b.get("adapter", 0)) / GB
        a = b["activations"] / GB
        tot = b["total"] / GB
        print(f"{name:<14}{method:<22}{'on' if ckpt else 'off':<6}"
              f"{w:>8.1f}G{t:>8.1f}G{a:>7.1f}G{tot:>8.1f}G  {_verdict(b['total'])}")

    for m in MODELS:
        for ckpt in (False, True):
            row(m.name, "full SFT (Adam,16B)", ckpt, full_sft_bytes(m, seq_len, batch, checkpointing=ckpt))
        for ckpt in (False, True):
            row(m.name, "LoRA r=16 (bf16 base)", ckpt, lora_bytes(m, seq_len, batch, checkpointing=ckpt))
        print()


def _selftest() -> None:
    """Check against notes/curriculum.md § Memory budget worked examples (weights+opt
    only, no activations). curriculum uses 16 B/param for full-SFT Adam and decimal
    GB (1e9); match within 20%."""
    GB10 = 1e9  # curriculum.md uses decimal GB for its headline figures

    def w_plus_opt(params, dtype, optimizer, master):
        return param_bytes(params, dtype) + training_state_bytes(
            params, optimizer=optimizer, grad_dtype="bf16", master_dtype=master)

    checks = [
        # (label, computed_bytes, expected_GB_from_curriculum)
        ("Full SFT 3B Adam (16B/param)", w_plus_opt(3_200_000_000, "bf16", "adamw", "fp32"), 51),
        ("Full SFT 8B Adam (16B/param)", w_plus_opt(8_000_000_000, "bf16", "adamw", "fp32"), 128),
        ("LoRA 14B bf16 base (~2B/param)", param_bytes(14_800_000_000, "bf16"), 30),
        ("LoRA 32B FP8 base (~1B/param)", param_bytes(32_800_000_000, "fp8"), 33),
    ]
    print("Self-test vs notes/curriculum.md (target within 20%):\n")
    all_ok = True
    for label, got, exp in checks:
        got_gb = got / GB10
        err = abs(got_gb - exp) / exp
        ok = err <= 0.20
        all_ok &= ok
        print(f"  [{'OK ' if ok else 'XX '}] {label:<34} computed {got_gb:6.1f} GB"
              f"  vs {exp:3d} GB  ({err*100:4.1f}% off)")
    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="check against curriculum.md")
    args = ap.parse_args()
    if args.test:
        _selftest()
    else:
        main()
