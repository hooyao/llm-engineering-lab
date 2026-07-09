# B4 — Attention and the Transformer block — learning notes

Per-learner, dialogue-shaped notes (A2/A5/D6 method). Tutor mode, learner-paced.
Conversation is 中文; this file is English. Math written as code (the terminal can't
render LaTeX sub/superscripts). Shapes spelled out per CLAUDE.md "Tensor Shapes —
Always Spell Them Out": every dim named, a real Llama-3.1-8B number attached, the
input->output transform shown, the scale level pinned.

Real Llama-3.1-8B numbers used throughout (the model in play):
d_model (hidden) = 4096, n_layers = 32, n_heads = 32, head_dim = 128,
GQA n_kv_heads = 8, d_ff = 14336, vocab = 128256.

## Why B4 is the day pulled forward (not B1–B3)

- Track B order is B1 micrograd (backward pass on scalars), B2 makemore
  (embedding / softmax / cross-entropy), B3 BPE tokenization, B4 attention + block.
- B1/B2 core content was already absorbed through Track A: A2 taught backward pass,
  gradient, chain rule, AdamW, and the text->token->ID->logits->softmax->cross-entropy
  chain; A4 unrolled the training loop by hand.
- B3 (tokenization) is genuinely un-taught but is orthogonal to the learner's actual
  pain (attention + block structure + shape tracking).
- The recurring stumbles (shape tracking = standing #1 difficulty; "a transformer
  isn't one matrix" at A6; seq_len / minimal-transformer at A5; KV cache +
  prefill/decode at D6) all trace to never studying transformer architecture
  end-to-end. That IS B4. So we go straight to it. Pure theory — no GX10 needed.
  (Decision recorded in progress.md LOG 2026-07-02.)

## Segment 0 — the frame: what attention is FOR

Pin the input shape first (learner saw this once at A5):

    X = [seq_len, d_model]      # one row = one token
    d_model = 4096 for Llama-3.1-8B
    A 10-token sentence -> X = [10, 4096]: 10 rows, 4096 numbers each.

The load-bearing sentence everything else hangs on:

> Inside a transformer layer, EVERYTHING except attention operates on each row
> (each token) INDEPENDENTLY. Rows do not talk to each other.

Concretely, the MLP (the gate/up/down W's the learner counted at A6) is applied
PER ROW: take row 3 of X, shape [d_model] = [4096], compute `y = W·x + b`, out comes
row 3 again — and row 3's output depends ONLY on row 3's input, nothing to do with
row 1. That per-row sub-network is exactly the learner's linear-regression-era
`y = Wx + b`, now run once per token in the sentence.

Consequence: a stack of such MLPs alone can NEVER let token i use token j's info.
But language needs exactly that:

    "The animal didn't cross the street because it was too tired."
    To represent "it", the model must look back at "animal".

A per-row MLP structurally cannot do this — when it computes the "it" row it has no
input port for the "animal" row.

Attention is the ONE part of the whole architecture that mixes ACROSS rows — it lets
row i read from row j and pull other tokens' information in. That is its entire job.
Pin the two scales (targets the learner's scale-fusion weak spot):

    per-token (per row, independent)  = MLP, norm     -> the old network, already known
    cross-token (across rows, mixing) = attention      -> the one new thing in B4

So: everything except attention is per-position; attention is the exception.

### Open question posed to learner (segment 0 close)
"When X = [seq_len, d_model] passes through one MLP layer, can row 3 (token 3) ever
see row 1's values? Instinct — yes or no, and why?"
(Steering toward: no — per-position independent. Their answer + Q&A folds in here.)
