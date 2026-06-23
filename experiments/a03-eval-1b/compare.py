#!/usr/bin/env python3
"""
A3 (also the A2 payoff) — generation quality, BEFORE vs AFTER fine-tuning.

This is the reward step: A2 produced a fine-tuned checkpoint but you never SAW it
behave differently. This script puts the original Llama-3.2-1B-Instruct and your
A2 checkpoint side by side on the same prompts, so the effect of full-parameter
SFT is something you read with your own eyes, not a loss number.

What to look for (SFT changes FORMAT and STYLE more than knowledge):
  - the fine-tuned model tends to answer more directly / on-task
  - it follows the alpaca instruction style it was trained on
  - it may be more concise, or stop more cleanly (eot handling)
  - knowledge is roughly the same — SFT is not teaching new facts

Greedy decoding (temperature 0 / do_sample=False) so the comparison is
deterministic and reproducible — rerun gives the same text.

Run inside the 26.04 container via run.sh.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompts chosen to expose instruction-following / format / style, not trivia.
PROMPTS = [
    "Give me three tips for staying focused while working from home.",
    "Explain what a hash map is to a 10-year-old.",
    "Write a haiku about autumn.",
    "List the steps to make a cup of tea.",
    "What is the capital of France?",
    "Rewrite this sentence to be more polite: 'Send me the report now.'",
    "Translate 'good morning' into French, Spanish, and Japanese.",
    "Name three programming languages and one thing each is good at.",
    "Summarize the plot of Cinderella in two sentences.",
    "Give me a vegetarian dinner idea.",
]


def build_chat(tok, user_msg: str) -> str:
    """Render one user turn with the model's chat template (the format A2 trained on)."""
    msgs = [{"role": "user", "content": user_msg}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(model, tok, user_msg: str, max_new_tokens: int) -> str:
    text = build_chat(tok, user_msg)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # greedy: deterministic, clean comparison
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    # Only decode the newly generated part (strip the prompt).
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/models/meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--tuned", default="/runs/a02-sft-1b")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", default="/runs/a03-eval/results.md",
                    help="Also write the side-by-side to this markdown file.")
    args = ap.parse_args()

    print("=" * 70)
    print("A2 PAYOFF — generation BEFORE vs AFTER full-parameter SFT")
    print("=" * 70)
    print(f"BEFORE (base) : {args.base}")
    print(f"AFTER (tuned) : {args.tuned}")
    print(f"decoding      : greedy (deterministic), max_new_tokens={args.max_new_tokens}")
    print()

    print("[load] base...")
    base, base_tok = load(args.base)
    print("[load] tuned...")
    tuned, tuned_tok = load(args.tuned)
    print()

    lines = ["# A2 payoff — generation before vs after full-parameter SFT\n",
             f"Base: `{args.base}`  |  Tuned: `{args.tuned}`  |  greedy decoding\n"]

    for i, p in enumerate(PROMPTS, 1):
        b = generate(base, base_tok, p, args.max_new_tokens)
        t = generate(tuned, tuned_tok, p, args.max_new_tokens)
        block = (
            f"\n{'='*70}\n[{i}] PROMPT: {p}\n{'-'*70}\n"
            f">>> BEFORE (base 1B-Instruct):\n{b}\n"
            f"{'-'*70}\n"
            f">>> AFTER  (your A2 SFT):\n{t}\n"
        )
        print(block)
        lines.append(f"\n## {i}. {p}\n\n**BEFORE (base):**\n\n> {b}\n\n"
                     f"**AFTER (A2 SFT):**\n\n> {t}\n")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[saved] side-by-side written to {args.out}")
    print("\nRead each pair. Where the AFTER differs — length, directness, format,")
    print("how cleanly it stops — that difference IS what your fine-tune did.")


if __name__ == "__main__":
    main()
