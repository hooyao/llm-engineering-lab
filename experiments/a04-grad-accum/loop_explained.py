# A4 — the explicit training loop, gradient accumulation made visible
#
# READ THIS, don't run it (yet). This is the loop that A2's `trainer.train()` hid.
# Every line is tagged with which of A2 Seg-6's FOUR BEATS it is:
#   [FWD]  forward     token -> layers -> logits -> softmax -> probabilities
#   [LOSS] loss        probabilities vs one-hot true answer -> cross-entropy -> -log(p)
#   [BWD]  backward    flow back from loss, chain rule -> a gradient per parameter
#   [OPT]  optimizer   AdamW uses gradient + m + v -> update each parameter
# ...plus the ONE new thing A4 adds on top:
#   [ACC]  accumulation bookkeeping  (the divide, the step-timing, the zero_grad)
#
# The whole point: micro-batch (what the GPU holds) is decoupled from effective
# batch (what the algorithm sees). effective = micro * accum_steps.

import torch

# ---- the two hyperparameters that are the whole lesson ---------------------
MICRO_BATCH = 1      # samples the GPU holds at once  -> sets activation memory (A1)
ACCUM_STEPS = 16     # how many micro-batches we sum before stepping the optimizer
# effective batch = MICRO_BATCH * ACCUM_STEPS = 16, SAME for all three day-plan configs.
# The day-plan sweeps (micro=1,accum=16), (micro=4,accum=4), (micro=8,accum=2):
# all effective batch 16, so final loss ~identical; only peak_mem / step_time differ.

model = ...          # the 3B model, all params trainable (full SFT, like A2)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
dataloader = ...     # yields ONE micro-batch (MICRO_BATCH samples) at a time

# ---- the loop A2 hid -------------------------------------------------------
optimizer.zero_grad()                                    # [ACC] accumulator starts at [0,0,...]

for step, micro_batch in enumerate(dataloader):

    # --- beat 1: forward -----------------------------------------------------
    outputs = model(**micro_batch)                       # [FWD] token -> ... -> logits
    #   activation for THIS micro-batch lives in memory now. Only MICRO_BATCH
    #   samples' worth — that's the memory wall A4 walks around.

    # --- beat 2: loss --------------------------------------------------------
    loss = outputs.loss                                  # [LOSS] cross-entropy -> one number
    #   HF gives loss already averaged OVER THE MICRO-BATCH's samples.

    # --- the ONE correction gradient accumulation needs ----------------------
    loss = loss / ACCUM_STEPS                             # [ACC] see "why divide here" below

    # --- beat 3: backward ----------------------------------------------------
    loss.backward()                                      # [BWD] chain rule -> grad per param
    #   KEY: backward does `grad += ...` (ACCUMULATES onto existing grad), it does
    #   NOT overwrite. THIS is the accumulator-row from the notes, for free. After
    #   ACCUM_STEPS backward() calls, each param's .grad holds the SUM of this
    #   effective batch's per-sample gradients. The activation from beat 1 is freed
    #   here once it's been used — so only ONE micro-batch's activation is ever live.

    # --- only every ACCUM_STEPS do the real optimizer step -------------------
    if (step + 1) % ACCUM_STEPS == 0:                    # [ACC] one effective batch complete
        optimizer.step()                                 # [OPT] AdamW: grad + m + v -> update
        optimizer.zero_grad()                            # [ACC] wipe accumulator -> [0,0,...]
        #   zero_grad MUST be here (after the step), NOT after every backward.
        #   Skip it and the next effective batch's sum leaks last batch's gradients.

# ============================================================================
# WHY `loss = loss / ACCUM_STEPS`  (the one subtle line — read carefully)
# ============================================================================
# We want the optimizer to see the AVERAGE gradient over the effective batch (16
# samples), exactly like a real batch-of-16 would produce. But:
#
#   - backward() SUMS gradients across the 16 micro-batches (the `grad +=` above).
#   - a true batch-of-16 would AVERAGE them (sum / 16).
#
# sum and average differ by a factor of 16. If we stepped on the raw sum, every
# gradient would be 16x too big -> effectively lr * 16 -> divergence. Dividing each
# micro-batch's loss by ACCUM_STEPS before backward scales its gradient by 1/16, so
# the 16 summed-up gradients come out to the correct average. (Gradient is linear in
# loss, so dividing the loss divides the gradient — that's why scaling the loss is
# enough; we don't touch the gradient tensors directly.)
#
# (Edge case the day-plan ignores: if MICRO_BATCH doesn't divide the dataset evenly,
#  the last accumulation group has fewer samples and /ACCUM_STEPS slightly mis-weights
#  it. Real libraries correct this; we don't need to for the A4 sweep.)
#
# ============================================================================
# MAP BACK TO A2's `trainer.train()`
# ============================================================================
# A2 was MICRO_BATCH=4, ACCUM_STEPS=1 (no accumulation). Set ACCUM_STEPS=1 above and:
#   - `loss = loss / 1`            -> no-op
#   - `if (step+1) % 1 == 0`       -> TRUE every step
# ...so it collapses to: forward, loss, backward, step, zero_grad — every step. That
# is precisely the four-beat loop from A2 Seg 6c, the one trainer.train() ran for you
# 125 times. A4 = the same loop, but step/zero_grad fire every ACCUM_STEPS instead of
# every step, and the loss is pre-divided. Debt from A2 ("walk through train.py") paid.
