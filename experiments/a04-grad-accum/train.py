#!/usr/bin/env python3
"""
A4 — gradient accumulation, with the training loop A2's Trainer hid made EXPLICIT.

This is loop_explained.py turned into runnable code. Unlike A2 (which used
transformers.Trainer and hid forward/loss/backward/optimizer behind trainer.train()),
here the loop is HAND-WRITTEN so every one of A2 Seg-6's four beats is a line you read:
    [FWD]  forward     model(**micro_batch) -> logits
    [LOSS] loss        outputs.loss  (cross-entropy, already averaged over micro-batch)
    [BWD]  backward     loss.backward()  -> grad per param (ACCUMULATES, grad +=)
    [OPT]  optimizer    optimizer.step() -> AdamW updates each param
...plus the one new A4 thing:
    [ACC]  accumulation  loss/ACCUM_STEPS ; step+zero_grad every ACCUM_STEPS only

The lesson: effective_batch = micro_batch * accum_steps. We run the SAME effective
batch (16) three ways and watch: final loss ~identical (the algorithm sees the same
thing), but peak_mem and step_time differ (memory traded for speed).

Run ONE config per process (so peak-memory measurement is clean); run.sh loops the 3.
Reads model + dataset from /models bind mount. Inside the 26.04 container.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--dataset", default="/models/yahma/alpaca-cleaned")
    ap.add_argument("--seq-len", type=int, default=1024, help="A4 spec: fixed 1024.")
    ap.add_argument("--lr", type=float, default=2e-5)
    # --- the two hyperparameters that ARE the lesson ---
    ap.add_argument("--micro-batch", type=int, required=True,
                    help="Samples the GPU holds at once (sets activation memory).")
    ap.add_argument("--accum-steps", type=int, required=True,
                    help="Micro-batches summed before one optimizer step.")
    # effective_batch = micro_batch * accum_steps  (16 for all three day-plan configs)
    ap.add_argument("--opt-steps", type=int, default=30,
                    help="How many OPTIMIZER steps (effective batches) to run. Each "
                         "consumes micro_batch*accum_steps samples. 30 is enough to "
                         "see loss fall and to time steadily.")
    ap.add_argument("--result-json", default=None,
                    help="If set, append one JSON line of results here (run.sh reads it).")
    args = ap.parse_args()

    eff_batch = args.micro_batch * args.accum_steps

    print("=" * 70)
    print(f"A4 — grad accumulation | micro={args.micro_batch} accum={args.accum_steps} "
          f"-> EFFECTIVE BATCH {eff_batch}")
    print("=" * 70)
    assert torch.cuda.is_available(), "CUDA not available inside container"
    print(f"device : {torch.cuda.get_device_name(0)}  torch {torch.__version__}")

    # ---------------- Tokenizer + model (BF16, full-parameter, like A2) ----------
    print("[load] tokenizer + model (BF16)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.train()                                    # train mode (dropout on, etc.)
    model.gradient_checkpointing_disable()           # A5's trick; off here for a clean A4
    total = sum(p.numel() for p in model.parameters())
    print(f"  loaded {total/1e9:.3f}B params in {time.time()-t0:.1f}s (all trainable)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ====================================================================
    # THE DATALOADER — this is the piece the learner asked to see.
    # ====================================================================
    # Three stages turn raw JSON rows into batched GPU-ready tensors:
    #
    #   (1) load_dataset      -> a table of {instruction, input, output} text rows
    #   (2) .map(tokenize)    -> each row becomes {input_ids:[int], attention_mask:[int]}
    #                            i.e. text -> token IDs (A2 Seg 4's text->ID step), as a
    #                            VARIABLE-LENGTH python list per row. No tensors yet.
    #   (3) DataLoader+collator-> groups `micro_batch` rows into ONE padded 2-D tensor
    #                            batch, and — crucially — the collator also builds the
    #                            `labels` field. THAT is why model(**batch) returns a loss
    #                            (see the big note at beat 2 below).
    print(f"[data] {args.dataset}")
    ds = load_dataset(args.dataset, split="train")

    def tokenize(ex):
        prompt = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
        text = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{ex['output']}<|eot_id|>")
        # truncation/padding=False: keep variable length here; the collator pads later.
        return tok(text, truncation=True, max_length=args.seq_len, padding=False)

    # We need micro*accum*opt_steps samples total. Take a bit more, shuffled.
    need = eff_batch * args.opt_steps
    ds = ds.shuffle(seed=42).select(range(min(len(ds), need + eff_batch)))
    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=4)

    # collator: mlm=False -> CAUSAL LM. On each batch it (a) pads input_ids to the
    # longest in the batch, (b) sets labels = input_ids (the model shifts internally),
    # (c) sets label = -100 on pad positions so padding contributes ZERO to the loss.
    # This injected `labels` is exactly what makes the loss computable inside model().
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # DataLoader yields ONE micro-batch (micro_batch rows) per iteration, already
    # collated into a dict of 2-D tensors: {input_ids, attention_mask, labels}.
    loader = DataLoader(ds, batch_size=args.micro_batch, shuffle=False,
                        collate_fn=collator, num_workers=2, drop_last=True)
    data_iter = iter(loader)

    # ---------------- the explicit training loop (A2's hidden loop, unrolled) ------
    print(f"[train] {args.opt_steps} optimizer steps "
          f"(each = {args.accum_steps} micro-batches of {args.micro_batch})")
    print("=" * 70)
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()                            # [ACC] accumulator starts empty

    step_times, losses = [], []
    for opt_step in range(args.opt_steps):
        t_step = time.time()
        accum_loss = 0.0                             # for logging only (the reported loss)

        for _ in range(args.accum_steps):            # accumulate accum_steps micro-batches
            micro = next(data_iter)
            micro = {k: v.to("cuda:0") for k, v in micro.items()}   # CPU -> GPU (the only PCIe copy)

            outputs = model(**micro)                 # [FWD] token IDs -> logits
            # ============================================================
            # WHY model(**micro) RETURNS A LOSS (the learner's question).
            # ============================================================
            # `micro` contains a `labels` key (the collator put it there). A HF
            # CausalLM forward, when it receives `labels`, does the loss itself:
            #     logits = the model's next-token scores  (A2 Seg 4)
            #     shift logits/labels by one, cross_entropy(logits, labels)  (A2 Seg 5)
            # and returns it as outputs.loss. No labels -> outputs.loss is None and you'd
            # compute CE yourself. So "model directly computes loss" = "we handed it the
            # answer key (labels), so it scores itself." The averaging is over all
            # non-(-100) token positions in this micro-batch.
            loss = outputs.loss                      # [LOSS] one number, GPU scalar

            loss = loss / args.accum_steps           # [ACC] sum->average correction (Seg 3)
            loss.backward()                          # [BWD] grad += (accumulates onto .grad)
            accum_loss += loss.item() * args.accum_steps  # undo /accum for honest logging

        # one effective batch is now summed across accum_steps micro-batches:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # same as A2's max_grad_norm
        optimizer.step()                             # [OPT] AdamW: grad + m + v -> update params
        optimizer.zero_grad()                        # [ACC] wipe accumulator for next eff batch

        torch.cuda.synchronize()                     # finish GPU work before timing

        # one-time diagnostic: ground-truth the bytes/param of optimizer state.
        # A2 (HF Trainer, mixed precision) kept m/v in fp32 -> 12 B/param. This raw
        # torch.optim.AdamW on bf16 params keeps m/v in whatever zeros_like(p) gives
        # -> bf16 -> 8 B/param. We print the dtype so the peak-mem number isn't a
        # mystery. NOTE: bf16 m/v is the Seg-6d "big eats small" footgun; fine for a
        # 30-step demo, NOT for a real long run (use fp32 states there).
        if opt_step == 0:
            p0 = next(p for p in model.parameters() if p.requires_grad)
            st = optimizer.state[p0]
            nparam = sum(p.numel() for p in model.parameters())
            alloc = torch.cuda.memory_allocated() / 1024**3
            print(f"  [mem-check] optimizer m/v dtype = {st['exp_avg'].dtype} / "
                  f"{st['exp_avg_sq'].dtype}")
            print(f"  [mem-check] allocated now {alloc:.2f} GB for {nparam/1e9:.2f}B params "
                  f"=> {alloc*1024**3/nparam:.1f} bytes/param (state+activation+ctx)")

        dt = time.time() - t_step
        step_times.append(dt)
        losses.append(accum_loss / args.accum_steps)
        print(f"  opt_step {opt_step+1:>3}/{args.opt_steps}  "
              f"loss {losses[-1]:.4f}  step_time {dt*1000:7.1f} ms")

    # ---------------- results (the payoff numbers) ----------------
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    # use the second half of steps for a steady step-time (skip warmup/compile/alloc)
    steady = step_times[len(step_times)//2:]
    mean_step = sum(steady) / len(steady)
    final_loss = sum(losses[-5:]) / len(losses[-5:])   # avg of last 5, less noisy

    print("=" * 70)
    print(f"[RESULT] micro={args.micro_batch} accum={args.accum_steps} eff={eff_batch}")
    print(f"  peak_mem   : {peak_gb:.2f} GB")
    print(f"  step_time  : {mean_step*1000:.1f} ms  (steady-state mean)")
    print(f"  final_loss : {final_loss:.4f}  (mean of last 5 steps)")
    print("=" * 70)

    if args.result_json:
        with open(args.result_json, "a") as f:
            f.write(json.dumps({
                "micro": args.micro_batch, "accum": args.accum_steps, "eff": eff_batch,
                "peak_gb": round(peak_gb, 2), "step_ms": round(mean_step*1000, 1),
                "final_loss": round(final_loss, 4),
            }) + "\n")


if __name__ == "__main__":
    main()
