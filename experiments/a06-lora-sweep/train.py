#!/usr/bin/env python3
"""
A6 — LoRA hyperparameter sweep on Llama-3.1-8B-Instruct.

The three LoRA hyperparameters this day taught, and which one each config varies:
  1. rank r        — adapter capacity (the bottleneck dim of B·A). LINEAR in params.
  2. alpha         — LoRA branch scale; the forward pass uses (alpha/r)·B·A·x.
                     HELD at alpha/r = 2 across ALL configs, so the sweep isolates
                     capacity (r) and coverage (target modules), NOT branch strength.
  3. target modules— which W's get an adapter. attn-only = q,k,v,o (4/layer);
                     attn+mlp adds gate,up,down (7/layer). The wide mlp matrices
                     (one dim = d_ff = 14336) are only affordable under LoRA because
                     LoRA pays the PERIMETER r·(d_in+d_out), not the AREA d_out·d_in.

The 4 configs (adjacent pairs isolate one variable):
  (1) r=8  alpha=16  attn      ->  6,815,744 params  (baseline)
  (2) r=16 alpha=32  attn      -> 13,631,488 params  ((1) with r x2)
  (3) r=16 alpha=32  attn+mlp  -> 41,943,040 params  ((2) with +mlp)
  (4) r=64 alpha=128 attn+mlp  -> 167,772,160 params ((3) with r x4)

This script runs ONE config (clean per-config peak memory + adapter file). run.sh
loops the 4. What each run measures, mapped to predictions.md:
  - trainable params  -> CERTAIN column (must match the formula above, pure arithmetic)
  - adapter bytes/param-> resolves the fp32(~4)-vs-bf16(~2) serialize-dtype bet
  - peak memory       -> secondary (LoRA fixed part is tiny; base 8B BF16 ~16 GB dominates)
  - final loss        -> does MORE adapter capacity lower loss on this data?

All configs share seed + data order + seq_len + batch + lr, so the ONLY differences
are r and target modules. That is what makes the loss column comparable.

W shapes for Llama-3.1-8B-Instruct (W = [d_out, d_in], real numbers):
  d_model = 4096, d_ff = 14336, num_layers = 32
  attn: q_proj [4096,4096]  k_proj [1024,4096]  v_proj [1024,4096]  o_proj [4096,4096]
        (k/v are [1024,4096], NOT [4096,4096] — GQA narrows them; easy to autopilot wrong)
  mlp : gate_proj [14336,4096]  up_proj [14336,4096]  down_proj [4096,14336]
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# Llama-3.1-8B-Instruct architecture constants (see docstring).
D_MODEL = 4096
D_FF = 14336
N_LAYERS = 32
# Per-layer LoRA param multipliers, learner-derived: params = r * (d_in + d_out) per W.
#   attn-only per layer = (4096+4096)+(1024+4096)+(1024+4096)+(4096+4096) = 26,624
#   mlp      per layer  = (14336+4096)*3                                   = 55,296
PER_LAYER = {"attn": 26_624, "attn+mlp": 26_624 + 55_296}  # 26624, 81920

TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn+mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

# Fixed prompts for the qualitative payoff — SAME across all 4 configs so gens are comparable.
GEN_PROMPTS = [
    "Explain what LoRA is to a systems engineer in two sentences.",
    "Write a Python function that returns the nth Fibonacci number.",
    "What is the capital of France?",
]


def expected_params(r: int, target: str) -> int:
    """Pure arithmetic: per-layer multiplier * num_layers * r. This is the CERTAIN column."""
    return PER_LAYER[target] * N_LAYERS * r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dataset", default="/models/allenai/tulu-3-sft-mixture")
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--alpha", type=int, required=True)
    ap.add_argument("--target", choices=["attn", "attn+mlp"], required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA tolerates a higher lr than full SFT.")
    ap.add_argument("--opt-steps", type=int, default=120)
    ap.add_argument("--out-dir", required=True, help="Where to save the adapter (per config).")
    ap.add_argument("--result-json", default=None)
    args = ap.parse_args()

    tag = f"r{args.r}/{args.target}"
    print("=" * 74)
    print(f"A6 — {tag}  (alpha={args.alpha}, alpha/r={args.alpha/args.r:.0f}, "
          f"base=Llama-3.1-8B-Instruct, batch={args.batch_size}, seq={args.seq_len})")
    print("=" * 74)
    assert torch.cuda.is_available(), "CUDA not available inside container"
    print(f"device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    torch.manual_seed(0)

    # ---------------- Tokenizer + base model (BF16, frozen) ----------------
    print("[load] tokenizer + base 8B model (BF16)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")

    # ---------------- LoRA wrap (the sweep variables live here) ----------------
    lora = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
        target_modules=TARGETS[args.target], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.train()

    # ---- CERTAIN column check: measured trainable params vs the pure-arithmetic formula ----
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    exp = expected_params(args.r, args.target)
    ok = "PASS" if trainable == exp else f"MISMATCH (expected {exp:,})"
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  trainable params : {trainable:,}  ({trainable/1e6:.2f}M)   [{ok}]")
    print(f"  predicted        : {exp:,}  ({exp/1e6:.2f}M)")
    print(f"  all params       : {total:,}   trainable% = {100*trainable/total:.3f}")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    # ---------------- Data: tulu-3 chat -> apply_chat_template -> tokenize ----------------
    # tulu-3-sft-mixture rows are {id, messages:[{role,content},...]}. The Instruct
    # tokenizer's chat template renders the whole conversation to the exact string the
    # model was post-trained on. Dynamic padding (collator pads to longest in batch,
    # capped at seq_len). Collator sets pad positions in labels to -100 (no loss on pad).
    print(f"[data] {args.dataset} (chat_template, truncate {args.seq_len})")
    ds = load_dataset("parquet",
                      data_files=os.path.join(args.dataset, "data", "*.parquet"),
                      split="train")

    def tokenize(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(text, truncation=True, max_length=args.seq_len)

    need = args.batch_size * args.opt_steps + args.batch_size
    ds = ds.shuffle(seed=42).select(range(min(len(ds), need)))
    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=4)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator, num_workers=2, drop_last=True)
    data_iter = iter(loader)

    # ---------------- Training loop (explicit, same idiom as A4/A5) ----------------
    print(f"[train] {args.opt_steps} steps")
    print("-" * 74)
    torch.cuda.reset_peak_memory_stats()
    step_times, losses = [], []
    for step in range(args.opt_steps):
        t_step = time.time()
        micro = next(data_iter)
        micro = {k: v.to("cuda:0") for k, v in micro.items()}
        outputs = model(**micro)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.time() - t_step)
        losses.append(loss.item())
        if step % 20 == 0 or step == args.opt_steps - 1:
            print(f"  step {step+1:>3}/{args.opt_steps}  loss {loss.item():.4f}  "
                  f"step_time {step_times[-1]*1000:7.1f} ms")

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    final_loss = sum(losses[-10:]) / len(losses[-10:])  # steady mean of last 10 steps
    mean_step = sum(step_times[len(step_times)//2:]) / len(step_times[len(step_times)//2:])

    # ---------------- Save adapter + measure serialize dtype ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir)
    adapter_file = os.path.join(args.out_dir, "adapter_model.safetensors")
    adapter_bytes = os.path.getsize(adapter_file) if os.path.exists(adapter_file) else -1
    bytes_per_param = adapter_bytes / trainable if trainable else float("nan")
    dtype_guess = ("fp32" if abs(bytes_per_param - 4) < abs(bytes_per_param - 2) else "bf16")

    print("=" * 74)
    print(f"[RESULT] {tag}")
    print(f"  trainable params : {trainable:,}  ({trainable/1e6:.2f}M)  vs predicted {exp/1e6:.2f}M [{ok}]")
    print(f"  peak_mem         : {peak_gb:.2f} GB")
    print(f"  final_loss       : {final_loss:.4f}  (mean last 10 steps)")
    print(f"  step_time        : {mean_step*1000:.0f} ms (steady mean)")
    print(f"  adapter file     : {adapter_bytes:,} bytes  ({adapter_bytes/1024**2:.1f} MiB)")
    print(f"  bytes / param    : {bytes_per_param:.3f}  ->  serialized {dtype_guess}")
    print("=" * 74)

    # Write the measured training data NOW — before the (secondary) gen step — so a
    # generation failure can never lose the numbers that ARE the payoff.
    if args.result_json:
        with open(args.result_json, "a") as f:
            f.write(json.dumps({
                "config": tag, "r": args.r, "alpha": args.alpha, "target": args.target,
                "trainable_params": trainable, "predicted_params": exp,
                "params_match": trainable == exp,
                "peak_gb": round(peak_gb, 2),
                "final_loss": round(final_loss, 4),
                "step_ms": round(mean_step * 1000, 0),
                "adapter_bytes": adapter_bytes,
                "bytes_per_param": round(bytes_per_param, 3),
                "serialize_dtype": dtype_guess,
            }) + "\n")

    # ---------------- Qualitative gen (shared prompts, for the payoff) ----------------
    # apply_chat_template with tokenize=True returns a BatchEncoding (dict-like) in
    # transformers 5.x, so pass return_dict=True and unpack with **inp; a bare tensor
    # index (inp.shape) raises KeyError:'shape'. Wrapped so a gen error is non-fatal.
    print("[gen] qualitative generation on shared prompts:")
    model.eval()
    try:
        with torch.no_grad():
            for p in GEN_PROMPTS:
                msgs = [{"role": "user", "content": p}]
                inp = tok.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True).to("cuda:0")
                prompt_len = inp["input_ids"].shape[1]
                out = model.generate(**inp, max_new_tokens=64, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                gen = tok.decode(out[0][prompt_len:], skip_special_tokens=True)
                print(f"  Q: {p}")
                print(f"  A: {gen.strip()[:300]}")
                print()
    except Exception as e:
        print(f"  [gen skipped: {type(e).__name__}: {e}]")


if __name__ == "__main__":
    main()
