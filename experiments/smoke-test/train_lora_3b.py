#!/usr/bin/env python3
"""
Stability / thermal / power smoke test.

Goal: keep the GB10 GPU busy for ~5-10 minutes with a realistic LoRA training
workload (BF16 forward + backward + optimizer step on Llama 3.2 3B). While it
runs, a separate `nvidia-smi dmon` writes power / temp / utilization to a log.

We do NOT care about model quality here. Loss should decrease (sanity), but
the real outputs are:
    - sustained power draw under load (vs 5W idle)
    - GPU temperature ceiling and whether it throttles
    - any Xid errors in dmesg
    - fan noise (you sit next to the box)

Reads model from /models bind mount (rsync'd from external drive).
Reads dataset from /models too.

Run inside container started by tools/launch_pytorch.sh.
"""

import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/meta-llama/Llama-3.2-3B-Instruct",
                    help="Path to base model (local directory).")
    ap.add_argument("--dataset", default="/models/yahma/alpaca-cleaned",
                    help="Path to dataset (local directory).")
    ap.add_argument("--steps", type=int, default=200,
                    help="Number of training steps (default 200 ≈ 5-10 min on GB10).")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Per-device training batch size.")
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="Sequence length (truncation).")
    ap.add_argument("--lora-rank", type=int, default=16,
                    help="LoRA rank.")
    ap.add_argument("--output-dir", default="/workspace/experiments/smoke-test/out",
                    help="Where to dump (small) adapter checkpoint + logs.")
    args = ap.parse_args()

    # ---------------- Banner ----------------
    print("=" * 70)
    print("GB10 SMOKE TEST — LoRA SFT for stability / thermal / power profile")
    print("=" * 70)
    print(f"model       : {args.model}")
    print(f"dataset     : {args.dataset}")
    print(f"steps       : {args.steps}")
    print(f"batch_size  : {args.batch_size}")
    print(f"seq_len     : {args.seq_len}")
    print(f"lora_rank   : {args.lora_rank}")
    print(f"output_dir  : {args.output_dir}")
    print()

    assert torch.cuda.is_available(), "CUDA not available inside container"
    print(f"torch       : {torch.__version__}")
    print(f"cuda        : {torch.version.cuda}")
    print(f"device      : {torch.cuda.get_device_name(0)}")
    print(f"sm          : {torch.cuda.get_device_capability(0)}")
    print(f"bf16 ok     : {torch.cuda.is_bf16_supported()}")
    print()

    # ---------------- Tokenizer + model ----------------
    print("[load] tokenizer + model (BF16)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print(f"  loaded in {time.time() - t0:.1f}s, "
          f"params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # ---------------- LoRA ----------------
    print(f"[lora] rank={args.lora_rank} on q/k/v/o + gate/up/down")
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable / 1e6:.2f}M / {total / 1e9:.2f}B "
          f"({100 * trainable / total:.3f}%)")

    # ---------------- Dataset ----------------
    print(f"[data] {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    print(f"  raw examples: {len(ds)}")

    def format_example(ex):
        # Alpaca columns: instruction / input / output
        prompt = ex["instruction"]
        if ex.get("input"):
            prompt += "\n\n" + ex["input"]
        text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{ex['output']}<|eot_id|>"
        )
        return tok(text, truncation=True, max_length=args.seq_len, padding=False)

    ds = ds.map(format_example, remove_columns=ds.column_names, num_proc=4)
    ds = ds.shuffle(seed=42).select(range(min(len(ds), args.steps * args.batch_size * 4)))
    print(f"  tokenized examples: {len(ds)}")

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # ---------------- Trainer ----------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="no",      # don't waste disk; we only care about runtime behavior
        report_to=[],            # no wandb/tensorboard
        gradient_checkpointing=False,
        dataloader_num_workers=2,
        warmup_steps=10,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    # ---------------- Train ----------------
    print()
    print(f"[train] starting {args.steps} steps...")
    print("=" * 70)
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    print("=" * 70)
    print(f"[done] {args.steps} steps in {elapsed:.1f}s "
          f"({args.steps / elapsed:.2f} steps/s, "
          f"{args.steps * args.batch_size / elapsed:.2f} samples/s)")
    print(f"       final loss: {result.training_loss:.4f}")

    # ---------------- Post-mortem GPU state ----------------
    print()
    print("[gpu] post-run state:")
    torch.cuda.synchronize()
    print(f"  memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  memory reserved : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  peak allocated  : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
