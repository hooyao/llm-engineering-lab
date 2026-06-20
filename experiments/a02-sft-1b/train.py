#!/usr/bin/env python3
"""
A2 — first full-parameter SFT (the smallest model that exists).

Goal of this day (notes/curriculum-v2-execution.md § A2): get the end-to-end
*supervised fine-tuning* training loop into your head BEFORE introducing any
memory tricks (LoRA/QLoRA come in A6/A7). So this is the smoke-test script with
the LoRA wrapper REMOVED — every parameter of Llama-3.2-1B is trained.

What "full-parameter SFT" costs here (the A1 calculator, mixed-precision Adam):
    1.24e9 params x 16 bytes/param = ~18.6 GB of weights+grad+master+opt state,
    + activations. Comfortable in the 128 GB pool. (Run A1's budget.py to see it.)

------------------------------------------------------------------------------
TERMS (defined once, used verbatim after — see CLAUDE.md directive 1):

- *causal language modeling* (CLM): the training objective. The model predicts
  the next token given all previous tokens; loss is how wrong those next-token
  predictions are, averaged over the sequence. SFT = CLM on instruction/response
  text so the model learns to follow the instruction format.
- *loss*: a single number measuring prediction error this step. Cross-entropy
  here. Lower = better next-token predictions. We expect ~2.5 -> ~1.5 over the run.
- *batch* (here: per-device train batch size): how many sequences are pushed
  through the forward pass together before computing one loss/gradient.
- *micro-batch*: the batch that actually fits on the device in one forward pass.
  With gradient_accumulation_steps=1 (this script), micro-batch == batch. A4 will
  split effective batch from micro-batch via gradient accumulation.
- *step* (optimizer step): one weight update. With accumulation=1, one step
  consumes one micro-batch: forward -> loss -> backward -> optimizer.step().
- *epoch*: one full pass over the training dataset. We do 1 epoch over 500
  examples; at batch=4 that's ~125 steps -- enough to watch loss fall.
------------------------------------------------------------------------------

Reads model + dataset from /models bind mount. Run inside the 26.04 container
started by run.sh.
"""

import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/meta-llama/Llama-3.2-1B-Instruct",
                    help="Path to base model (local directory).")
    ap.add_argument("--dataset", default="/models/yahma/alpaca-cleaned",
                    help="Path to dataset (local directory).")
    ap.add_argument("--num-examples", type=int, default=500,
                    help="How many examples to train on (A2 spec: 500).")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="Number of epochs (A2 spec: 1).")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Per-device micro-batch size (A2 spec: 4).")
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="Sequence length / truncation (A2 spec: 1024). "
                         "Try 8192 to deliberately OOM and read the trace.")
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="Learning rate. Full-param SFT uses ~1e-5..2e-5; LoRA's "
                         "2e-4 would diverge when every weight is trainable.")
    ap.add_argument("--output-dir", default="/runs/a02-sft-1b",
                    help="Where to save the trained model (host ~/runs via mount).")
    args = ap.parse_args()

    # ---------------- Banner ----------------
    print("=" * 70)
    print("A2 — FULL-PARAMETER SFT on Llama-3.2-1B (no LoRA)")
    print("=" * 70)
    print(f"model        : {args.model}")
    print(f"dataset      : {args.dataset}")
    print(f"num_examples : {args.num_examples}")
    print(f"epochs       : {args.epochs}")
    print(f"batch_size   : {args.batch_size}")
    print(f"seq_len      : {args.seq_len}")
    print(f"lr           : {args.lr}")
    print(f"output_dir   : {args.output_dir}")
    print()

    assert torch.cuda.is_available(), "CUDA not available inside container"
    print(f"torch   : {torch.__version__}")
    print(f"cuda    : {torch.version.cuda}")
    print(f"device  : {torch.cuda.get_device_name(0)}")
    print(f"sm      : {torch.cuda.get_device_capability(0)}")
    print(f"bf16 ok : {torch.cuda.is_bf16_supported()}")
    print()

    # ---------------- Tokenizer + model ----------------
    print("[load] tokenizer + model (BF16)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # NOTE: no LoRA. The whole model is loaded in BF16 and every parameter has
    # requires_grad=True, so the optimizer carries m/v state for all 1.24B params.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  trainable: {trainable / 1e9:.3f}B / {total / 1e9:.3f}B "
          f"({100 * trainable / total:.1f}%)  <-- full-parameter SFT")

    # ---------------- Dataset ----------------
    print(f"[data] {args.dataset}")
    ds = load_dataset(args.dataset, split="train")
    print(f"  raw examples: {len(ds)}")

    def format_example(ex):
        # Alpaca columns: instruction / input / output. Build a Llama-3 chat
        # turn. We train CLM over the whole text (prompt + response). Masking the
        # prompt tokens out of the loss is a refinement deferred past A2 -- here
        # the point is the end-to-end loop, not loss-masking subtlety.
        prompt = ex["instruction"]
        if ex.get("input"):
            prompt += "\n\n" + ex["input"]
        text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{ex['output']}<|eot_id|>"
        )
        return tok(text, truncation=True, max_length=args.seq_len, padding=False)

    ds = ds.shuffle(seed=42).select(range(min(len(ds), args.num_examples)))
    ds = ds.map(format_example, remove_columns=ds.column_names, num_proc=4)
    print(f"  training examples: {len(ds)} (1 epoch ~= {len(ds) // args.batch_size} steps "
          f"at batch={args.batch_size})")

    # mlm=False -> causal LM collator: labels = input_ids shifted by the model.
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # ---------------- Trainer ----------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,      # micro-batch == batch this day (see A4)
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",              # save the trained model (A2 deliverable)
        report_to=[],
        gradient_checkpointing=False,
        dataloader_num_workers=2,
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
    print(f"[train] {args.epochs} epoch(s) over {len(ds)} examples...")
    print("=" * 70)
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    nsteps = trainer.state.global_step

    print("=" * 70)
    print(f"[done] {nsteps} steps in {elapsed:.1f}s "
          f"({nsteps / elapsed:.2f} steps/s, "
          f"{nsteps * args.batch_size / elapsed:.2f} samples/s)")
    print(f"       final train loss: {result.training_loss:.4f}")

    # Print the per-step loss history so the curve is visible in the log.
    print()
    print("[loss curve] (step: loss)")
    for rec in trainer.state.log_history:
        if "loss" in rec:
            print(f"  {rec['step']:>4}: {rec['loss']:.4f}")

    # ---------------- Save ----------------
    print()
    print(f"[save] writing model + tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    # ---------------- Post-mortem GPU state ----------------
    print()
    print("[gpu] post-run state:")
    torch.cuda.synchronize()
    print(f"  peak allocated  : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"  (A1 predicted ~18.6 GB weights+opt for 1.24B full SFT, + activations)")


if __name__ == "__main__":
    main()
