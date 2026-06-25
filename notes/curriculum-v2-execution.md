# Curriculum — three parallel tracks, ~1.5h per evening

> **Why are you doing this?** See `notes/why.md`. Re-read when motivation flags.
> Every day below maps to one of the 5 surviving fine-tuning use cases in 2026.

This file is the **execution plan**. Three independent tracks, each ~1.5h/day, ~10-12
sessions per track. Pick any track on any night; do not feel obliged to do all three
the same day. You can run all three in ~6 weeks (5 sessions/week) or in ~4 weeks
(7-evening pace).

What this assumes about you:

- 10+ years software architecture experience (so `git`, `python`, `bash`, `docker`,
  refactoring, debugging, profiling: zero introduction).
- ML/DL: **none assumed**. Every ML term defined the first time it's used in the
  experiment script comments, and the line of code that embodies it is highlighted.
- Math: you're refreshing in Track 3 in parallel; nothing in Track 1 or 2 blocks
  on it. When a derivative shows up in code (e.g. AdamW update), the comment shows
  the equation and points to the day in Track 3 that derives it.

Conventions used in every track:

- **Each day produces a runnable artifact** (script, notebook, or one paragraph
  of notes). No "read a chapter and think about it" days — you'll forget.
- **One concept per day.** If a day seems to bundle two, split it into two evenings.
- **Container is the unit of compute.** Everything runs via `tools/launch_pytorch.sh`
  unless explicitly stated.
- **Catalog is `notes/curriculum.md`** (this file's sibling). Numbers, paths, datasets:
  go look there.

Track layout in this file:

- **Track A** — Fine-tuning (this is the "use a pretrained model" path)
- **Track B** — Pretraining from scratch + RLHF (this is the "write a small LLM" path)
- **Track C** — Math refresh (focused on what shows up in Track A/B)

> **PyTorch:** as a software engineer you don't need a PyTorch course. The
> official 60-minute Blitz is enough as a one-time prereq before Track A starts;
> Track B1 (Karpathy micrograd) and Track C6 (hand-derive backprop) make the
> autograd internals concrete. See **Track A § A0** for the exact reading list.

Each track:
- Day-by-day plan with concrete deliverable
- Single recommended resource per day (no rabbit holes)
- An end-of-track checkpoint

---

# Track A — Fine-tuning (10 evenings)

**Goal at the end:** You can take any open-source 1B-32B model, pick an appropriate
fine-tuning method (full SFT / LoRA / QLoRA / DPO) for your memory budget, run it
on GX10, judge the result, and serve it for inference. You know **why** each choice
was made, not just **how**.

## Setup once (Day 0, no time budget, do it on the weekend)

```bash
ssh hooyao@192.168.1.200
cd ~/dgx-spark-playground
bash tools/launch_pytorch.sh        # confirm container starts and sees /models
## inside container, smoke test as in experiments/smoke-test/ was already verified
exit
```

If you haven't already, glance once at `notes/curriculum.md` § Memory budget — you'll
keep referring to it.

## A0 — PyTorch crash course (one-time, ~1 hour, before A1)

**Why this is short:** as a 10+ year software engineer, you don't need a multi-week
PyTorch course. You need just enough to read training scripts without stumbling on
API. The deep "how does it work" is solved on **Track C6** (hand-derive backprop)
and **Track B1** (Karpathy micrograd) — neither is a PyTorch tutorial.

**Do — pick ONE (don't do both):**

- **Recommended: Sebastian Raschka, *PyTorch in One Hour: From Tensors to Training Neural
  Networks on Multiple GPUs*** (2025-07, free, ~1h reading time):
  https://sebastianraschka.com/teaching/pytorch-1h/

  Best fit because: (1) same author as the *Build a LLM From Scratch* book you're
  reading, so the API conventions and notation will match seamlessly when you
  start the book's chapters; (2) targeted at LLM engineers — skips CV-heavy fluff;
  (3) covers PyTorch 2.x current best practices (`torch.compile`, modern device
  placement, autograd anatomy). Single-page HTML, scannable.

- **Alternative if you prefer video**: Daniel Bourke, *PyTorch 101 Crash Course For
  Beginners in 2026* on YouTube (1h03m, free): https://www.youtube.com/watch?v=LyJtbe__2i0.
  Slower than reading, but if you absorb better watching someone type code, this is the
  most up-to-date video tutorial in 2026.

- **Skip** the PyTorch official "60-minute Blitz" — partially stale (still references
  old API patterns from the 1.x era), and structured around image classification examples
  that aren't useful for LLM work.

- **Skip** LinkedIn Learning / Pluralsight / Coursera PyTorch courses (8-15h, typically
  12-18 months stale, oversized for what you need).

**Deliverable:** none. Just be able to look at `experiments/smoke-test/train_lora_3b.py`
and recognize every PyTorch primitive in it. If anything still looks alien after
the Raschka tutorial, jot it in `notes/a00-pytorch-questions.md` and we'll cover it on the
relevant day rather than upfront.

## A1 — Memory budget calculator (the only theory day in this track)

**Why:** Every ML decision in the next 9 days is constrained by "does it fit in 128 GB
shared between CPU and GPU." Knowing how to *compute* the answer beats trial-and-error.

**Do:**

- Create `experiments/a01-mem-budget/budget.py` with three functions:
  - `param_bytes(num_params: int, dtype: str) -> int`
  - `optimizer_bytes(num_trainable_params: int, optimizer: str) -> int`
    (cover: `adamw` = 8 bytes/param, `adamw_8bit` = 2, `lion` = 4, `sgd` = 0)
  - `activation_bytes(seq_len, batch, hidden, layers, dtype, checkpointing: bool) -> int`
    (formula: `seq * batch * hidden * layers * dtype_bytes * (1 if checkpointing else ~6)`)
- Print a table: for Llama-3.2-1B / 3B / 8B, show **full SFT** and **LoRA r=16** memory
  for `seq=2048 batch=4` with checkpointing on and off.
- Compare to nameplate from `notes/curriculum.md` § Memory budget. Match within 20%.

**Deliverable:** the script + a 10-line `notes.md` listing which model+method combos
"will obviously OOM," "marginal," "comfortable" on this unit.

**Resource:** the explanation in `notes/curriculum.md` § Memory budget (already present).
Don't read papers tonight.

## A2 — First full SFT (smallest model that exists)

**Why:** Get the end-to-end training loop in your head before introducing any tricks.

**Do:**

- Create `experiments/a02-sft-1b/train.py` based on `experiments/smoke-test/train_lora_3b.py`
  but **remove the LoRA wrapper**. Target `Llama-3.2-1B-Instruct` (3.21B → 1.24B params, fits trivially).
- Use `alpaca-cleaned`, 500 samples, 1 epoch, batch=4, seq=1024, bf16.
- Confirm it OOMs gracefully if you set `seq=8192` — practice reading the OOM trace.

**Deliverable:** loss curve printed to terminal showing loss decreasing from ~2.5 to ~1.5
in 100-ish steps. Save the resulting model to `~/runs/a02-sft-1b/`.

**Term defined in the script comments:** *epoch*, *batch*, *step*, *micro-batch*,
*loss*, *causal language modeling*.

## A3 — Generation quality, before vs after SFT

**Why:** SFT is fundamentally about **format following and style**, not knowledge.
Best way to feel this: prompt both the original and your trained model with the same
10 inputs, eyeball the diff.

**Do:**

- `experiments/a03-eval-1b/compare.py`:
  10 fixed prompts (mix instructions, questions, role-play); generate from
  `Llama-3.2-1B-Instruct` and from your A2 checkpoint; print side by side.
- Pay attention: SFT model probably gives shorter, more on-task replies. May lose
  some knowledge breadth.

**Deliverable:** `experiments/a03-eval-1b/results.md` with the 10 prompt pairs and
your own 3-line observation per prompt.

**Term:** *generation* vs *loss*, *chat template* — define
the first time each `tokenizer.apply_chat_template(...)` is called. (*temperature*
and *top_p* are touched here only as `generate` arguments; they get their own
payoff and full treatment in **A9.5**.)

## A4 — Gradient accumulation and effective batch size

**Why:** GPU memory caps `micro_batch`. Algorithm cares about `effective_batch`. The
bridge is gradient accumulation.

**Do:**

- `experiments/a04-grad-accum/train.py`: 3B model, fix `seq=1024`. Run three configs:
  - `micro=1 accum=16` (effective batch 16, slow, low mem)
  - `micro=4 accum=4`  (effective batch 16, faster, more mem)
  - `micro=8 accum=2`  (effective batch 16, even faster, near OOM)
- Plot or table: peak memory, step time, **loss after 100 effective steps should be
  near-identical for all three** (sanity).

**Deliverable:** table of (peak_mem, step_time, final_loss) for the 3 configs.

**Term:** *gradient accumulation*, *effective batch size*, *optimizer step* (the
actual `param -= lr * m_hat / (sqrt(v_hat)+eps)` happens **after** accumulation, not
per micro-batch).

## A5 — Activation checkpointing and seq_len

**Why:** Activations dominate memory at long seq_len. Checkpointing trades recompute
for memory. Knowing the trade quantitatively means you stop guessing.

**Do:**

- `experiments/a05-ckpt-seqlen/sweep.py`: fix 3B + LoRA r=16 + batch=2. Sweep
  `seq_len ∈ {512, 1024, 2048, 4096}` × `checkpointing ∈ {off, on}` = 8 configs.
- Record peak memory and step time. Make a 2D table.
- Observation you should see: checkpointing **saves 30-50% memory** but **adds 25-40%
  step time**. The save scales with seq_len.

**Deliverable:** `sweep_results.csv` and 5-line interpretation in `notes.md`.

**Term:** *forward pass*, *backward pass*, *activation*, *recompute*, *transformer
block*.

## A6 — LoRA properly: rank, alpha, target modules

**Why:** Until now you've used my default `r=16 on q/k/v/o + gate/up/down`. Time to
understand what each hyperparameter does.

**Do:**

- `experiments/a06-lora-sweep/`: 8B base, `tulu-3-sft-mixture` 5000 samples. Run 4 configs:
  - `r=8,  α=16, attn-only` (smallest adapter)
  - `r=16, α=32, attn-only`
  - `r=16, α=32, attn+mlp`  (this was D5's default)
  - `r=64, α=128, attn+mlp` (largest reasonable)
- For each: trainable params, adapter checkpoint size on disk, final loss,
  qualitative gen quality on 5 prompts.

**Deliverable:** table comparing the 4 configs. Note which one you'd pick for
"production" and why.

**Term:** *low-rank decomposition* (W + ΔW where ΔW = BA, rank(BA) = r);
*α scaling* (`adapter contribution × α/r` — alpha controls how much of the adapter
"leaks into" the base output); *adapter*.

**Resource (15 min read):** LoRA paper §3 only — https://arxiv.org/abs/2106.09685.
Skip the eval sections; the §3 architecture is the whole point.

## A7 — QLoRA: NF4 quantization + paged optimizer

**Why:** Same A6 setup but loads the base in 4-bit. Drops memory ~4× with ~0 quality
loss. This is how 70B models become trainable on a single 128 GB machine.

**Do:**

- `experiments/a07-qlora-8b/train.py`: same as A6 best config, but load base with
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`.
- Compare peak memory to A6: should drop from ~28 GB → ~10 GB.
- **If bitsandbytes refuses to import on aarch64 sm_121:** fall back to FP8 (Tier 3
  in `notes/curriculum.md`), use `Qwen3-32B-FP8` instead — note in your log which
  path you took.

**Deliverable:** peak memory + final loss compared to A6. One paragraph on **what
NF4 actually is** (a 4-bit datatype with non-uniform bins matched to a normal
distribution — base weights are roughly normal, so bins fit them well).

**Term:** *quantization*, *NF4*, *double quantization*, *paged optimizer* (offloads
optimizer state to CPU memory when GPU is full, swaps as needed — on GX10 this is
free since CPU memory == GPU memory).

**Resource (10 min):** QLoRA paper §3 — https://arxiv.org/abs/2305.14314.

## A8 — Scale up: 14B then 32B with QLoRA

**Why:** Prove you can handle models that don't fit BF16. This is the GX10's main
selling point.

**Do:**

- `experiments/a08-qlora-scale/`: same recipe, three model sizes:
  - `Qwen3-8B` (BF16 LoRA, baseline) — should still work
  - `Qwen3-14B` (QLoRA NF4 or FP8 if BnB broken)
  - `Qwen3-32B-FP8` (already FP8 on disk, just LoRA)
- 200 steps each, record peak memory, tokens/s, final loss.

**Deliverable:** scaling table. Comment on where you start to feel slow.

## A9 — DPO: alignment without RL

**Why:** Direct Preference Optimization is the modern alternative to RLHF/PPO.
Cheaper, simpler, and what almost everyone uses in 2026. **You will derive the loss
in Track C2 around the same time — bring questions back here.**

**Do:**

- `experiments/a09-dpo-8b/train.py`: take your A7 LoRA adapter as the SFT base,
  run DPO with `argilla/dpo-mix-7k` (small) first to verify the pipeline, then
  `ultrafeedback_binarized` for the real run.
- Use TRL's `DPOTrainer`. Beta = 0.1.
- Watch `rewards/chosen`, `rewards/rejected`, and most importantly
  `rewards/margins` (chosen - rejected) trending up.

**Deliverable:** loss + reward margin plot or table after 500 steps. Generate from
your DPO'd model vs the SFT-only model on 5 prompts that have a "preferred" style
(e.g., "explain X to a 5-year-old" — DPO version should follow tone constraint better).

**Term:** *reference model*, *implicit reward*, *KL constraint* (β controls how
much DPO is allowed to move away from the reference).

**Resource (20 min):** DPO paper §3 + §4 — https://arxiv.org/abs/2305.18290.
Read the loss derivation in §4 carefully; this is the heart of it.

## A9.5 — Decoding strategies (greedy / temperature / top-k / top-p / beam)

**Why:** Up to here every model output was produced by *some* decoding strategy
that was never examined — A2/A3 just called `model.generate(...)` and the A3 slide
leaned on "deterministic decoding → identical output" as *evidence* without ever
opening the box. Decoding is the step that turns the final-layer logits
(`[vocab_size]`, e.g. 128256 for Llama-3.2-1B) into the next token. It is a small,
self-contained concept that deserves its own payoff instead of a footnote on a
fine-tuning slide. It is the inference-side twin of A10's KV cache: same per-step
decode loop — A9.5 is "how you pick the next token," A10 is "what gets recomputed
to pick it."

**Do:**

- `experiments/a09b-decoding/sweep.py`: one fixed prompt, one model
  (`Llama-3.2-1B-Instruct`), two experiments in one script:
  1. **Temperature sweep** — same prompt, `temperature ∈ {0.0, 0.3, 0.7, 1.0, 1.2}`
     (0.0 = greedy), `do_sample=True` except at 0.0. Print each output. Watch it go
     deterministic → fluent-but-varied → word-salad.
  2. **Strategy grid** — same prompt, four decoders side by side: greedy,
     top-k (k=50), top-p (p=0.9), beam (num_beams=4). Print all four.
- Note in the script *why* greedy is deterministic (argmax over logits; softmax is
  monotonic so `argmax(logits) == argmax(softmax(logits))` — no need to normalize).
- Pin the shape once: `generate` consumes a logits vector of shape `[vocab_size]`
  per step (`vocab_size = 128256` for this model); the strategy decides which index
  becomes the emitted token, then it is appended and the loop runs again
  (autoregressive decode).

**Deliverable:** `experiments/a09b-decoding/results.md` — the temperature-sweep
outputs and the 4-strategy grid, with a 2-3 line observation on what changed and
which setting you'd serve for a factual assistant vs a brainstorming tool.

**Term:** *logits*, *decoding* vs *sampling*, *greedy / argmax*, *temperature*,
*top-k*, *top-p (nucleus)*, *beam search*, *autoregressive decode*.

**Payoff (decided live, but the form is fixed here):** the learner runs `sweep.py`
themselves and *sees* the same prompt produce a deterministic line at temp 0 and
drift into incoherence at temp 1.2 — one parameter (temperature), a visible
gradient of randomness —
plus the four-strategy grid making greedy-vs-sampling concrete. This is the "single
number that means something" reward: temperature, watched live, not read back from
a log. Ties straight back to the A3 slide — *now* the learner knows exactly why
"deterministic decoding → byte-identical output" was a valid proof there.

## A10 — Inference: vLLM serve and benchmark

**Why:** "Training the model" ≠ "having a useful service." Inference economics
(throughput, latency, memory) are a different optimization problem.

**Do:**

- `experiments/a10-serve/`: run `vllm serve /path/to/your/checkpoint --lora-modules
  myadapter=/path/to/adapter`.
- Benchmark with `vllm benchmark` or a hand-rolled loop: measure
  - tokens/s at batch=1 (latency-bound)
  - tokens/s at batch=32 (throughput-bound)
  - p50 and p99 first-token latency.
- Try with and without the LoRA adapter — see latency penalty.

**Deliverable:** benchmark table. One paragraph on **what was different about
inference vs training** (no grad, no optimizer state, KV cache dominates memory,
batch can be much higher).

**Term:** *KV cache*, *prefill* vs *decode*, *paged attention* (vLLM's main idea),
*continuous batching*.

### Track A checkpoint

You should now be able to look at any model card on HuggingFace, eyeball the
parameter count, decide "BF16 LoRA, QLoRA, or out of reach," sketch the memory
budget, and start a run within 5 minutes of looking at the page. Write a 1-page
recap in `notes/track-a-recap.md`.

---

# Track B — Pretraining + RLHF from scratch (16 evenings)

**Goal at the end:** You've written every line of a small LLM, from tokenizer to
attention to optimizer to RLHF, in clean PyTorch. You can read any production LLM
repo and follow the data flow.

**Important framing:** You are NOT going to train a competitive 7B model. You're
training a ~33M model on TinyStories that produces grammatical English stories.
This is enough to internalize **every step**.

## B1 — micrograd: backprop on scalars

**Why:** Backpropagation is the one ML algorithm you must internalize. Karpathy's
micrograd does it on **scalar values** in ~100 lines — no tensors to obscure things.

**Do:**

- Watch Karpathy "The spelled-out intro to neural networks and backpropagation"
  (Zero to Hero #1, 2h25m). https://youtu.be/VMj-3S1tku0
- Type along: don't paste. Your `experiments/b01-micrograd/` should have your own
  `Value` class with `__add__`, `__mul__`, `backward()`.
- Verify your gradients match `torch.autograd` for a small 3-layer MLP.

**Deliverable:** your micrograd implementation + a notebook showing a single
training step matches PyTorch numerically.

**This is the most important day in Track B.** If something doesn't make sense,
re-watch the section. Don't move on.

## B2 — makemore: bigram and MLP language models

**Why:** Move from scalars to tensors. Build a character-level language model
that learns to generate plausible names.

**Do:**

- Watch Karpathy makemore #1 (bigram) and #2 (MLP), ~3h total.
- `experiments/b02-makemore/bigram.py` and `mlp.py`.
- Generate 20 made-up names. Some should sound English-like.

**Term:** *embedding*, *softmax*, *cross-entropy loss*, *negative log-likelihood*.

## B3 — Tokenization: BPE from scratch

**Why:** Every LLM starts with a tokenizer. You should understand byte-pair encoding
deeply enough to read `tiktoken` or HF tokenizers source.

**Do:**

- Watch Karpathy "Let's build the GPT Tokenizer" (2h13m). https://youtu.be/zduSFxRajkE
- `experiments/b03-bpe/` implement your own BPE in pure Python (~200 lines).
- Train on Shakespeare; show your merges look reasonable.
- Compare your tokenizer's vocab and a few sample encodings to `tiktoken`'s `gpt2`
  encoding on the same text.

**Term:** *BPE*, *merge*, *vocab*, *special token*, *byte-level*.

## B4 — Attention and the Transformer block

**Why:** This is the architectural primitive that everything else is glued around.

**Do:**

- Watch Karpathy "Let's build GPT: from scratch" (1h57m). https://youtu.be/kCc8FmEb1nY
- `experiments/b04-attention/attn.py`: implement single-head, then multi-head
  self-attention from `torch.matmul` + `torch.softmax`. **No `nn.MultiheadAttention`.**
- Implement the full transformer block: attention + residual + LN + MLP + residual.
- Verify forward pass numerically equivalent to PyTorch's `nn.TransformerEncoderLayer`
  (within float tolerance) on a fixed input.

**Term:** *Q/K/V*, *causal mask*, *layer normalization*, *residual connection*,
*positional encoding* (absolute, RoPE, ALiBi — define all three but only implement
absolute today).

## B5 — nanoGPT on Shakespeare

**Why:** First end-to-end pretraining. Tiny enough to finish on GX10 in <30 min.

**Do:**

- `git clone https://github.com/karpathy/nanoGPT` to `~/external/`
- Run `python prepare.py` then `python train.py config/train_shakespeare_char.py
  --device=cuda --compile=False`
- Read EVERY line of `train.py` and `model.py`. Annotate things you don't recognize.
- Sample from the checkpoint after training: `python sample.py`. Expect mediocre
  Shakespeare.

**Deliverable:** your annotated copy of `train.py` (in `experiments/b05-nanogpt/`),
plus a sampled poem.

## B6 — TinyStories pretraining (the big one)

**Why:** A real pretraining run that you can actually complete on GX10.

**Do:**

- Read TinyStories paper (skim, 30 min) — https://arxiv.org/abs/2305.07759
- Adapt nanoGPT config or use the official `tinystories` repo
  (https://github.com/karpathy/llama2.c → `tinystories.py`):
  - Model: ~33M params (n_layer=6, n_head=6, n_embd=384)
  - Data: TinyStories from `roneneldan/TinyStories` on HF (~2 GB after tokenization)
  - Use **your own BPE** from B3 to tokenize (or use GPT-2's, both fine).
- Set up the run, start it. Expect **~12 hours** on GX10. Run overnight.

**Deliverable:** trained checkpoint + 5 generated stories that have coherent
plot/characters. **You wrote everything except the optimizer.**

## B7 — Read the training run

**Why:** Pretraining outputs a learning curve, gradient norms, perplexity. Reading
them tells you what went well.

**Do:**

- Load the loss log from B6.
- Plot: loss vs step, loss vs FLOPs spent (Chinchilla-style compute axis).
- Compute final perplexity on a held-out chunk.
- Compare to TinyStories paper's reported numbers (Figure 4 in paper).

**Deliverable:** `experiments/b07-analysis/charts.ipynb` with at least 3 plots.

## B8 — SFT on your pretrained model

**Why:** Connect Track B (you have a base model) back to Track A (fine-tune it).

**Do:**

- Use your TinyStories model + an instruction-formatted subset of TinyStories
  (or a small custom dataset: "given a topic, write a 3-paragraph story").
- Run SFT for a few hundred steps (very fast on a 33M model).
- Compare gen before/after SFT.

**Deliverable:** before/after gen comparison on 5 prompts.

## B9 — Reward model

**Why:** Classic RLHF needs a reward model. PPO needs it; DPO is what you do **instead**
of all of B9-B11, but understanding the classic pipeline first makes DPO concrete.

**Do:**

- Take your B8 SFT model. Generate pairs of completions for the same prompt.
- Hand-label preferred completion for 50 pairs (this is a real ML data-labeling
  experience; budget the full session for this).
- Add a linear scalar head on top of your SFT model's last hidden state.
- Train it with the Bradley-Terry pairwise preference loss
  `loss = -log(sigmoid(reward_chosen - reward_rejected))`.

**Deliverable:** reward model + accuracy on a held-out 10 pairs.

**Resource:** Nathan Lambert RLHF Book §3 — https://rlhfbook.com

## B10 — PPO RLHF

**Why:** PPO was the dominant RLHF algorithm until DPO appeared. You should
implement one PPO update **once** in your life.

**Do:**

- Use TRL's `PPOTrainer` (no need to reimplement PPO from scratch on a 33M model —
  the value is seeing the loop structure).
- Train your B8 SFT model using the B9 reward model.
- Just 100 steps. Watch the KL divergence to the reference model.

**Deliverable:** training log showing reward going up, KL controlled.

**Resource:** Lambert RLHF Book §6 — PPO chapter.

## B11 — DPO comparison

**Why:** Now you have the comparison. Same SFT base, same preference data, two
algorithms.

**Do:**

- Take your 50 hand-labeled pairs from B9.
- Run DPO directly (skip the reward model step entirely).
- Compare gen quality vs PPO from B10.

**Deliverable:** A/B table. Note that DPO needed ~half the steps and zero reward-
model training. This is why everyone moved to it.

**Resource:** Lambert RLHF Book §4.

## B12 — Read OLMo

**Why:** You've trained a 33M model end to end. Now read what a production-grade
7B pretrain looks like.

**Do:**

- Skim AllenAI OLMo paper (1h): https://arxiv.org/abs/2402.00838
- Specifically look for things you now know: their data mix, optimizer, LR schedule,
  z-loss, gradient clipping, tokenizer, eval suite.
- Don't try to run it. Just identify every component and where it lives in
  https://github.com/allenai/OLMo

**Deliverable:** `notes/olmo-map.md` with one line per OLMo component you now
recognize, and which day in Track B introduced it.

---

## B13–B16 — Modern-stack sequel (minimind)

**Framing.** Everything up to B12 built a **GPT-2-era** model: LayerNorm, learned
absolute positions, multi-head attention, GELU, a GPT-2/BPE vocab. But the model
you actually fine-tune in Track A is **Qwen3-8B**, which uses RMSNorm, RoPE,
SwiGLU, and grouped-query attention (GQA) — none of which you've written. These
four days close that gap using **minimind** (`jingyaogong/minimind`,
https://github.com/jingyaogong/minimind), a from-scratch, native-PyTorch
implementation of a **Qwen3-aligned** tiny LLM. The whole model is one ~19 KB
file (`model/model_minimind.py`); the training stages are one script each under
`trainer/`.

**Why minimind and not "keep extending nanoGPT":** minimind is deliberately
modern-aligned (RMSNorm + RoPE + SwiGLU + GQA + optional MoE) and it **hand-writes
the things Track B's RLHF days delegated to TRL** (PPO, DPO, GRPO are full native
PyTorch loops, not `trl` wrappers). So it serves two distinct purposes: (1) an
architecture bridge to the Qwen3 stack, and (2) a "now that you know the math,
read a framework-free implementation" reference for the RL algorithms. It is a
**read / annotate / run-and-modify** reference (like B5's nanoGPT), **not** a
replacement for the B1–B4 hand-written days.

**Order matters:** these come **after B11**, because reading minimind's hand-written
PPO/DPO/GRPO only pays off once you've (a) done the TRL versions in B10/B11 and
(b) derived the losses in Track C7–C9. Don't pull these earlier.

**GX10 footprint (so you size runs correctly).** minimind targets a single 3090
(24 GB). On the GX10 the 128 GB unified pool makes memory a non-issue — train the
full-size data, not the `_mini` subsets:

- minimind-3 dense, `hidden_size=768`, `num_hidden_layers=8`, ~64M params.
  BF16 weights + AdamW (m,v = 8 B/param) + grads (2 B/param):
  `64M × (2 + 8 + 2) ≈ 0.77 GB`. Activations at seq_len 512, small batch add a
  few hundred MB. Trivial here.
- minimind-3-moe, 4 experts top-1, ~198M total / ~64M active. State is sized by
  **total** params (all experts' weights live in memory):
  `198M × 12 ≈ 2.4 GB`. Still trivial.
- Dataset: full `pretrain_t2t.jsonl` (10 GB) + `sft_t2t.jsonl` (14 GB) + the
  preference/RL files (~150 MB) ≈ **~25 GB on disk** — fits the 1 TB NVMe with
  room. Pull from HF `jingyaogong/minimind_dataset` into `./dataset/`.
- The README's "~3 RMB / ~2 h" headline is **1 epoch of SFT on one 3090**. On the
  GX10 expect faster; budget against measured BF16 throughput, not that figure.

**Setup (once, before B13):**

```
git clone --depth 1 https://github.com/jingyaogong/minimind ~/external/minimind
# download dataset files from HF jingyaogong/minimind_dataset into ~/external/minimind/dataset/
```

All training scripts run from `cd ~/external/minimind/trainer` and support
`--from_resume 1` and `--use_wandb`. Single-GPU is `python train_xxx.py`
(or `torchrun --nproc_per_node 1 train_xxx.py`). **Model size and the MoE toggle
are NOT CLI flags** — they live in the `MiniMindConfig` dataclass at the top of
`model/model_minimind.py` (see B14).

---

## B13 — Architecture bridge: GPT-2 block → Qwen3 stack

**Why:** Make the four modern primitives concrete by diffing your own B4 block
against minimind's. After today, every architectural choice in the Qwen3 model
you fine-tune in Track A has a line of code you've read and understood.

**Do:**

- Read `model/model_minimind.py` top to bottom (~19 KB, one sitting). Map each
  piece back to your B4 implementation:
  - **RMSNorm** vs your LayerNorm — why drop the mean-subtraction and the bias.
  - **RoPE** vs your absolute positional embedding — find where `rope_theta=1e6`
    enters and how the rotation is applied to Q and K (not V).
  - **SwiGLU** vs your GELU MLP — note it's a *gated* MLP with **three** weight
    matrices (gate, up, down), not two.
  - **GQA** vs your MHA — `num_attention_heads=8`, `num_key_value_heads=4`: K/V
    are shared across head groups. Compute the KV-cache saving vs full MHA.
- In `experiments/b13-modern-block/`, take your B4 `attn.py` transformer block and
  **rewrite it** into the modern stack: replace LayerNorm→RMSNorm, absolute
  pos→RoPE, GELU-MLP→SwiGLU, MHA→GQA. Keep it a single forward-pass module.
- Verify your rewritten block's output matches minimind's `MiniMindBlock` (or the
  equivalent) within float tolerance on a fixed random input.

**Term:** *RMSNorm*, *RoPE* (rotary position embedding), *SwiGLU*, *GQA*
(grouped-query attention), *KV head*.

**Deliverable:** `experiments/b13-modern-block/block.py` (your modern block) +
a short `diff.md` listing the four swaps and, for GQA, the KV-cache-size formula
`2 × num_kv_heads × head_dim × seq_len × layers × dtype_bytes` and the ratio vs MHA.

## B14 — MoE from scratch

**Why:** Mixture-of-Experts is the architecture behind most frontier 2025 models
(routed FFN, only top-k experts active per token). minimind's MoE is small enough
to read end to end. This day finally lands the long-pending "MoE extension."

**Do:**

- In `model/model_minimind.py`'s `MiniMindConfig`, the MoE hyperparameters are:
  `use_moe=False` (toggle), `num_experts=4` (routed experts),
  `num_experts_per_tok=1` (top-k routing). Read the MoE FFN class: how the router
  scores experts, how top-1 is selected, how outputs are combined, and where the
  **load-balancing / aux loss** is computed.
- Set `use_moe=True` and pretrain the MoE variant (`python train_pretrain.py`
  from `trainer/`, after editing the config). Compare to the dense run:
  - **Active** vs **total** params (only `num_experts_per_tok` of `num_experts`
    fire per token).
  - Memory: state is sized by **total** params (all experts resident), throughput
    by **active** params. Write both numbers.
- Watch the router: log which experts get picked. Confirm the aux loss is keeping
  utilization from collapsing onto one expert.

**Term:** *MoE*, *router / gating*, *top-k routing*, *routed vs shared experts*,
*expert load balancing*, *auxiliary (load-balance) loss*, *active vs total params*.

**Deliverable:** `experiments/b14-moe/notes.md` — dense vs MoE param counts
(active + total), a memory/throughput table, and a plot or table of per-expert
token counts showing the router isn't collapsed.

## B15 — GRPO from scratch (and the PPO/DPO contrast)

**Why:** GRPO is the RL algorithm behind DeepSeek-R1-style reasoning training —
it drops PPO's value network and estimates advantage from a **group** of sampled
completions. You did PPO (B10, via TRL) and DPO (B11, via TRL); now read a
hand-written GRPO and place all three on one map.

**Do:**

- Read `trainer/train_grpo.py` (~20 KB, native PyTorch — no `trl`). Identify:
  group sampling, the group-relative advantage (reward normalized within the
  group, no critic), the PPO-style clipped ratio, and the KL-to-reference term.
- Contrast against `trainer/train_ppo.py` (has a value head + GAE) and
  `trainer/train_dpo.py` (no sampling at all — closed-form on preference pairs).
  One table: **what each needs** (reward model? value net? online sampling?
  preference pairs?) and **what each optimizes**.
- Run a short GRPO job on your B8/SFT-equivalent minimind checkpoint
  (`python train_grpo.py`). Just enough steps to watch reward rise and KL stay
  bounded. Note: CISPO is the same script with `loss_type=cispo`.

**Term:** *GRPO*, *group-relative advantage*, *critic-free RL*, *reward
normalization*, *CISPO*, *clipped surrogate objective*.

**Deliverable:** `experiments/b15-grpo/compare.md` — the PPO vs DPO vs GRPO table
(inputs + objective + what's removed), plus a GRPO training log (reward up, KL
controlled).

## B16 — Distillation OR agentic RL (pick one)

**Why:** Two frontier techniques minimind implements that Track B otherwise never
touches. Pick the one closer to your job target. Both are resume-level.

**Option A — Knowledge distillation** (`trainer/train_distillation.py`):

- Read both modes: **black-box** (student trains on teacher *outputs* / generated
  text) and **white-box** (student matches teacher *logits* via KL on the soft
  distribution, temperature-scaled).
- Distill a larger minimind (or a Track-A Qwen3) teacher into a smaller student.
  Compare student-alone SFT vs distilled student on held-out prompts.
- **Term:** *knowledge distillation*, *soft targets / logit matching*,
  *temperature*, *black-box vs white-box distillation*.

**Option B — Agentic RL / tool use** (`trainer/train_agent.py` +
`trainer/rollout_engine.py`) — **wires Track B into Track D**:

- Read how multi-turn tool-calling rollouts are generated and scored. Note the
  rollout engine can drive generation through SGLang
  (`--rollout_engine sglang --sglang_base_url ... --data_path ../dataset/agent_rl_math.jsonl`).
- Run a short agentic-RL job on `agent_rl_math.jsonl`. Connect what you see here
  to the agent loop you build by hand in **Track D** (`agent/curriculum-agent.md`):
  same tool-call protocol, but here it's the *training* signal, not inference.
- **Term:** *agentic RL*, *multi-turn rollout*, *tool-call reward*, *rollout
  engine*.

**Deliverable:** `experiments/b16-<distill|agent>/notes.md` — for A, a
student-vs-distilled comparison; for B, a rollout trace + one paragraph linking
it to the Track D agent loop.

### Track B checkpoint

You have a trained-from-scratch LLM, taken end to end **twice**: a GPT-2-era 33M
storyteller (B1–B11: your own tokenizer, attention, training loop, hand-derived
reward model, then PPO and DPO via TRL), and a **modern Qwen3-aligned** model
(B13–B16: RMSNorm/RoPE/SwiGLU/GQA, optional MoE, and hand-written GRPO +
distillation/agentic RL with zero framework wrappers). You can open any
pretraining or RLHF repo — old or 2025-frontier — and name every piece.
Write 1 page in `notes/track-b-recap.md`.

---

# Track C — Math refresh (10 evenings)

**Goal at the end:** You can read any ML paper without skipping the equations.
You derive gradients on paper for AdamW, LoRA, cross-entropy, and DPO. You stop
treating `loss.backward()` as magic.

**Anchor resource:** Parr & Howard, "The Matrix Calculus You Need For Deep Learning"
(you're already reading it). https://explained.ai/matrix-calculus/

Skip every day where the topic is already comfortable. Track C is the only place
in this curriculum where "skim and move on" is OK if you already know it.

## C1 — Scalars: derivatives review

**Why:** Foundation. Many ML papers' equations are scalar in disguise.

**Do:**

- Read Parr & Howard §1-2.
- Solve 10 exercises by hand: d/dx of polynomials, exp, log, sigmoid, ReLU at
  x=0 (subgradient!), softmax for a 3-vector.
- Verify each with `sympy` or by numeric finite differences.

**Deliverable:** scratch notebook with 10 derivatives, each with hand derivation
and a numerical check.

## C2 — Partial derivatives and the gradient vector

**Why:** Every loss in deep learning is a function of millions of parameters. The
gradient vector is the basic object.

**Do:**

- Read Parr & Howard §3.
- For `L = (x*y + z*sin(w))^2`, compute ∂L/∂x, ∂L/∂y, ∂L/∂z, ∂L/∂w by hand.
- Implement it in PyTorch with `requires_grad=True` on all 4. Confirm match.

**Deliverable:** the comparison + a 3-sentence answer to "why is gradient
`pointing toward steepest ascent`?"

## C3 — The chain rule

**Why:** Backprop = repeated application of the chain rule. Do this until trivial.

**Do:**

- Read Parr & Howard §4.
- Derive d/dx of `sigmoid(W*x + b)` for `x ∈ R^n`, `W ∈ R^{m×n}`, `b ∈ R^m`.
- Match to autograd.

**Deliverable:** derivation on paper, photographed and committed to
`experiments/c03-chain-rule/`.

## C4 — Jacobian: when output is a vector

**Why:** Cross-entropy loss has scalar output but most layers have vector output;
you need the Jacobian to chain through them.

**Do:**

- Read Parr & Howard §5.
- Compute the Jacobian of softmax: ∂softmax(z)_i / ∂z_j. The closed form is
  `softmax(z)_i * (δ_ij - softmax(z)_j)`.
- Derive cross-entropy loss gradient w.r.t. logits: should simplify to
  `softmax(logits) - y_onehot`. Magical simplification.

**Deliverable:** the derivation. Note: this simplification is **why** classification
models use logits+CE jointly — the gradient is one subtraction.

## C5 — Matrix calculus: gradients w.r.t. matrices

**Why:** `nn.Linear`'s weight is a matrix. Its gradient is a matrix of the same
shape. You need to write these gradients without confusion.

**Do:**

- Read Parr & Howard §6.
- For `y = Wx + b`, derive ∂L/∂W in terms of ∂L/∂y. Result: `∂L/∂W = ∂L/∂y * x^T`.
- For `y = x^T W x`, derive ∂y/∂W and ∂y/∂x.

**Deliverable:** worked derivations + PyTorch check.

## C6 — Backprop end to end, derived

**Why:** All ML autodiff is "use the rules from C1-C5 in topological order."
Derive the gradient through a 2-layer MLP by hand once.

**Do:**

- 2-layer MLP: `h = ReLU(W1 x + b1)`, `y = W2 h + b2`, `L = CE(y, target)`.
- Derive ∂L/∂W2, ∂L/∂b2, ∂L/∂W1, ∂L/∂b1, ∂L/∂x.
- Implement the forward and your manual backward in numpy (no autograd).
  Compare to PyTorch.

**Deliverable:** numpy + PyTorch match within float tolerance. This is the day
backprop stops being magic.

## C7 — AdamW: the gradient ↦ step computation

**Why:** Knowing the AdamW update equations means you can debug "loss not going
down" without flailing.

**Do:**

- Read the AdamW paper, equations only (4 lines):
  ```
  m_t = β1 * m_{t-1} + (1 - β1) * g_t
  v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
  m_hat = m_t / (1 - β1^t)
  v_hat = v_t / (1 - β2^t)
  param_t = param_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param_{t-1})
  ```
- Implement AdamW in 30 lines of Python on top of a list of (param, grad) tuples.
- Verify it matches `torch.optim.AdamW` on a 10-step toy run.

**Deliverable:** your AdamW + comparison.

**Bonus understanding:** the `m` and `v` are 32-bit floats per parameter. That's
where "AdamW costs 8 bytes/param" comes from (you already used this in Track A1).

## C8 — KL divergence, entropy, cross-entropy

**Why:** Show up in every loss function in modern LLM training. PPO and DPO both
have KL terms.

**Do:**

- Read Wikipedia "Kullback-Leibler divergence" and "Cross-entropy."
- Prove: `CE(P, Q) = H(P) + KL(P || Q)`.
- For your B7 trained model, compute the KL between two checkpoints' output
  distributions on a small held-out chunk. Reflect on what units (nats vs bits) it's in.

**Deliverable:** the proof + the KL number.

## C9 — DPO loss, derived

**Why:** This is the day Track A9 connects to Track C. DPO's loss comes from a
clever algebraic rearrangement of the RLHF objective.

**Do:**

- Read DPO paper §4 carefully (you read it in A9; now do it with C1-C8 in your head).
- Reproduce the derivation from "RLHF objective" to "DPO loss" on paper.
- The key step: under their constrained optimum, the implicit reward
  `r(x,y) = β log(π(y|x) / π_ref(y|x)) + log Z(x)`, and `log Z(x)` cancels in pairwise
  comparisons.

**Deliverable:** your derivation + a 1-paragraph note "why this means we don't
need a reward model."

## C10 — Read one ML paper end to end, no equation skipped

**Why:** This is the only test that matters: can you read an ML paper now without
skipping the math?

**Do:**

- Pick one of:
  - LoRA (https://arxiv.org/abs/2106.09685) — easy, matches Track A6
  - QLoRA (https://arxiv.org/abs/2305.14314) — medium, has algorithm pseudocode
  - DPO (https://arxiv.org/abs/2305.18290) — challenging, dense algebra
  - GRPO from DeepSeek-R1 — hard, current-cutting-edge
- Read every section. Every equation. Write down what each variable is.
- Allow yourself one paper reread.

**Deliverable:** `notes/c10-paper-readthrough.md` — section-by-section summary
**in your own words**, with each equation typeset (or photographed) and explained.

### Track C checkpoint

You can read ML papers without skipping the math. You stop being intimidated by
∂, ∇, and KL. You can derive AdamW, LoRA, and DPO on a whiteboard.

---

# Suggested combined cadence

There's no single right order. Three working patterns:

## Pattern 1 — fully parallel (6 weeks, 5 evenings/week, fastest)

| Mon | Tue | Wed | Thu | Fri |
|---|---|---|---|---|
| Track A | Track B | Track C | Track A | Track B |

Skip days when you're tired. Don't double up.

## Pattern 2 — serial (10 weeks, 5 evenings/week, deepest)

Week 1-2: Track A → Week 3-5: Track B → Week 6-8: Track C.

Disadvantage: you'll forget Track A's mechanics by the time you do C9 (DPO derivation).

## Pattern 3 — Track A first, then B+C parallel (8 weeks, recommended)

Week 1-2: Track A only (build the SFT muscle memory).
Week 3-8: Track B and Track C in parallel.

Why this works: Track A is the most "habit-forming" — knowing the docker / dataloader /
HF stack frees your brain for Track B's pretraining details and Track C's math.

---

# When you get stuck

- **OOM during a Track A run:** consult `notes/curriculum.md` § Memory budget +
  your Track A1 calculator. The answer is in the arithmetic.
- **Gradient is `None` or `NaN`:** Track C6's hand-derived backprop is the antidote.
  Print intermediate tensor norms.
- **Karpathy video has a confusing bit:** rewind 30s, run the code yourself, then move on.
  Don't watch passively.
- **Math notation in a paper looks alien:** that's a "go to Track C and find the
  section that introduces this notation" signal, not a "I don't get ML" signal.

---

# What this curriculum does NOT cover

Deliberately. Each is a 3-month project of its own; do them after the 8 weeks
above if you're hooked.

- **Distributed training:** FSDP, DeepSpeed ZeRO-3, tensor parallel. Single-box GX10
  doesn't need any of it for the model sizes you'll touch.
- **MoE training:** Qwen3-30B-A3B-FP8 is on your disk; you can do it as a "Track A
  extension" once you've finished A8.
- **Speculative decoding / draft models:** an inference optimization, separate
  rabbit hole.
- **RLHF on big models:** the 33M model in B10/B11 is enough to learn the algorithm.
  Scaling it is a separate engineering project.
- **Multi-modal:** vision-language, audio. Out of scope, different stack.
