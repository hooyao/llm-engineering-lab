# A3 (also the A2 payoff) — generation quality, before vs after SFT

**Deliverable for A3** + the visible reward A2 owed. Code: `compare.py`, launcher:
`run.sh`, raw side-by-side: `results.md` (all 10 prompt pairs, full text).

The point: A2 produced a fine-tuned checkpoint but the learner never *saw* it
behave differently — only a loss number. This day puts the original
Llama-3.2-1B-Instruct and the A2 checkpoint on the same 10 prompts, greedy
decoding (deterministic), so the effect of full-parameter SFT is readable directly.

## How it was run

```bash
bash experiments/a03-eval-1b/run.sh   # base vs ~/runs/a02-sft-1b, writes results.md
```

Greedy decoding (`do_sample=False`) so the comparison is reproducible. 10 prompts
chosen to expose instruction-following / format / style, not trivia.

## What the SFT actually changed (read from `results.md`)

The diff is visible. Dominant patterns:

1. **Dropped the preamble.** The biggest change. Base prefixes answers with
   "Here are three tips for…" / "Here's a rewritten version…"; the tuned model
   goes straight to the answer.
   - #1 (focus tips): base opens "Here are three tips…"; tuned starts at "1. Create…".
   - #6 (be more polite): base gives a preamble + three variants + a closing
     paragraph; tuned gives the single rewritten sentence and stops.

2. **More concise / on-task.** #9 (Cinderella in two sentences): base sprawls and
   invents a wedding-sabotage subplot; tuned stays closer to a 2-sentence summary.

3. **Knowledge is unchanged — the key lesson.** SFT changes format/style, not
   facts. #5 (capital of France): base and tuned are *identical*
   ("The capital of France is Paris."). #3 (haiku): identical but one word.

4. **One accuracy shift (don't over-read it).** #7 (translate "good morning"):
   base gives wrong Spanish ("Hola" = hello) and wrong Japanese ("Konnichiwa" =
   good afternoon); tuned gives correct "buenos días" / "ohayou gozaimasu". Likely
   a side effect of the style shift, NOT a reliable capability gain — 500 examples,
   1 epoch. The A3 discipline: don't draw capability conclusions from single
   samples.

## Why this is the payoff

If the fine-tune had done nothing, all 10 pairs would be identical. They are not —
the tuned model is consistently more direct, drops the "Here is…" scaffolding, and
keeps its knowledge. That observable difference is both the reward and the skill:
reading model outputs to judge *what your training changed*.

## Caveat / honesty

- 1B-Instruct is already instruction-tuned, so "before" is already decent; the SFT
  nudges style toward the alpaca format rather than teaching from scratch. On a
  *base* (non-instruct) model the before/after gap would be far larger.
- A `clean_up_tokenization_spaces` BPE warning appears in the log; cosmetic, does
  not affect the generations.
- Results saved on host at `~/runs/a03-eval/results.md` (root-owned, container).
