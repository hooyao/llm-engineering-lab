#!/usr/bin/env python3
"""
A5 — activation checkpointing vs seq_len. The trade the learner reasoned out:
checkpointing trades RECOMPUTE (time) for ACTIVATION MEMORY (space).

Why activation memory exists (learner's Q1): backward needs each layer's FORWARD
input to compute that layer's gradient — `dloss/dw = x * dloss/dz`, where x is the
layer's forward-input activation. So forward must keep every layer's x alive until
backward consumes it (late, after the whole forward). That pile of x's = activation
memory, and it scales with batch * seq_len * hidden * layers.

Checkpointing (learner's Q2/Q3): in forward, save activations only at a few
checkpoint boundaries (per transformer block) and DROP the rest; in backward,
re-run forward from the nearest checkpoint to recompute the dropped x's on demand.
Less memory, one extra partial forward = more time. Can't drop everything (Q3) —
block-boundary checkpoints stay, so the save fraction tops out at k (~30-50%).

This run is LoRA (r=16) on 3B so the FIXED part (weight+grad+m+v) is small and the
ACTIVATION part dominates the contrast — exactly what makes the checkpointing effect
legible. We sweep seq_len x {ckpt off,on} and measure peak_mem + step_time.

ONE config per process (clean peak-memory). run.sh loops the 8. Inside 26.04.
"""

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dataset", default="/models/yahma/alpaca-cleaned")
    ap.add_argument("--batch-size", type=int, default=2, help="A5 spec: fixed batch 2.")
    ap.add_argument("--seq-len", type=int, required=True, help="Sweep axis: 512/1024/2048/4096.")
    ap.add_argument("--checkpointing", choices=["off", "on"], required=True,
                    help="Sweep axis: activation/gradient checkpointing.")
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA tolerates a higher lr than full SFT.")
    ap.add_argument("--lora-rank", type=int, default=16, help="A5 spec: r=16.")
    ap.add_argument("--opt-steps", type=int, default=20,
                    help="Optimizer steps (no accumulation here — A5 is about activation, not batch).")
    ap.add_argument("--result-json", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print(f"A5 — ckpt={args.checkpointing}  seq_len={args.seq_len}  "
          f"(3B + LoRA r={args.lora_rank}, batch={args.batch_size})")
    print("=" * 70)
    assert torch.cuda.is_available(), "CUDA not available inside container"
    print(f"device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    # ---------------- Tokenizer + base model (BF16) ----------------
    print("[load] tokenizer + base model (BF16)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")

    # ---------------- Activation checkpointing toggle (the A5 control) ----------------
    # gradient_checkpointing_enable() = save activations only at block boundaries,
    # recompute the rest during backward. use_reentrant=False is the modern impl.
    if args.checkpointing == "on":
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()   # needed so ckpt works through a frozen (LoRA) base
        print("  gradient checkpointing: ON  (recompute activations in backward)")
    else:
        model.gradient_checkpointing_disable()
        print("  gradient checkpointing: OFF (store all activations)")

    # ---------------- LoRA wrapper (r=16, attn+mlp) ----------------
    # LoRA keeps the FIXED part tiny: base frozen (no grad/optimizer), only the small
    # A/B adapters train. So the activation part dominates peak_mem — the point of A5.
    lora = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  loaded in {time.time()-t0:.1f}s | LoRA trainable {trainable/1e6:.1f}M "
          f"/ {total/1e9:.3f}B ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    # ---------------- Data: pad EVERY sample to exactly seq_len ----------------
    # A5 needs activation memory to depend ONLY on seq_len, so we pad/truncate every
    # sample to a fixed seq_len (not dynamic padding) — otherwise the 2-D token block's
    # width would vary and the memory reading would be noisy.
    print(f"[data] {args.dataset} (every sample fixed to seq_len={args.seq_len})")
    ds = load_dataset(args.dataset, split="train")

    def tokenize(ex):
        prompt = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
        text = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{ex['output']}<|eot_id|>")
        return tok(text, truncation=True, max_length=args.seq_len,
                   padding="max_length")          # <- fixed width = seq_len, every row

    need = args.batch_size * args.opt_steps + args.batch_size
    ds = ds.shuffle(seed=42).select(range(min(len(ds), need)))
    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=4)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator, num_workers=2, drop_last=True)
    data_iter = iter(loader)

    # ---------------- Training loop (explicit, like A4; no accumulation here) ----------------
    print(f"[train] {args.opt_steps} steps")
    print("-" * 70)
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    oom = False
    try:
        for step in range(args.opt_steps):
            t_step = time.time()
            micro = next(data_iter)
            micro = {k: v.to("cuda:0") for k, v in micro.items()}
            outputs = model(**micro)              # forward (ckpt on => stores only boundaries)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()                       # backward (ckpt on => recompute dropped activations)
            optimizer.step()
            torch.cuda.synchronize()
            dt = time.time() - t_step
            step_times.append(dt)
            if step % 5 == 0 or step == args.opt_steps - 1:
                print(f"  step {step+1:>3}/{args.opt_steps}  loss {loss.item():.4f}  "
                      f"step_time {dt*1000:7.1f} ms")
    except torch.cuda.OutOfMemoryError:
        oom = True
        print(f"  !! OOM at seq_len={args.seq_len} ckpt={args.checkpointing}")

    # ---------------- Results ----------------
    if oom:
        peak_gb, mean_step = float("nan"), float("nan")
        print("=" * 70)
        print(f"[RESULT] ckpt={args.checkpointing} seq_len={args.seq_len}  -> OOM")
    else:
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        steady = step_times[len(step_times)//2:]
        mean_step = sum(steady) / len(steady)
        print("=" * 70)
        print(f"[RESULT] ckpt={args.checkpointing} seq_len={args.seq_len}")
        print(f"  peak_mem  : {peak_gb:.2f} GB")
        print(f"  step_time : {mean_step*1000:.1f} ms (steady-state mean)")
    print("=" * 70)

    if args.result_json:
        with open(args.result_json, "a") as f:
            f.write(json.dumps({
                "ckpt": args.checkpointing, "seq_len": args.seq_len,
                "peak_gb": None if oom else round(peak_gb, 2),
                "step_ms": None if oom else round(mean_step * 1000, 1),
                "oom": oom,
            }) + "\n")


if __name__ == "__main__":
    main()
