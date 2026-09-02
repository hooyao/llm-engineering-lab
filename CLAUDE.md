# CLAUDE.md

This file provides canonical guidance to Claude Code (claude.ai/code) and Codex when working with code in this repository.

## Core Directives (non-negotiable)

1. **No analogies.** Use the correct English ML/systems terminology directly. Never analogize to .NET, GC, allocators, RDMA, or any other domain. Analogies are allowed **only** when the user explicitly asks ("类比一下" / "compare to ..."). Define a non-obvious term once on first use, then use it verbatim (e.g. `KV cache`, `ZeRO-3`, `reduce-scatter`, `activation checkpointing`, `paged optimizer`, `NF4`, `tensor parallel`, `pipeline parallel`, `gradient accumulation`, `Flash-Attention`, `CUDA graph`).
2. **Conversation language: 中文.** All replies to the user are written in Chinese. Keep technical terms in English (model names, API names, parameters, code symbols). **Never translate ML/systems terms into Chinese — not even "common" ones.** Write `bias`, `weight`, `gradient`, `forward pass`, `backward pass`, `layer`, `embedding`, `tensor`, `loss`, `activation`, `parameter`, `batch`, `optimizer` verbatim in English inside the Chinese prose. Do NOT write 偏置 / 权重 / 梯度 / 前向传播 / 反向传播 / 层 / 嵌入 / 张量 / 损失 / 激活值 / 参数 / 批 / 优化器. The user finds the Chinese renderings harder to read than the English originals. This applies to teaching explanations too, where the temptation to "translate for clarity" is strongest — resist it; the English term IS the clear version. **Critically: the user is new to LLMs but knows the same terms from other contexts (e.g. `weight` from linear regression). Translating `weight`→权重 BREAKS that bridge — they can no longer see that the `weight` in linear regression and the `weight` in an LLM are the same word/concept. Mixing 中文 renderings with the English terms in code/papers actively creates confusion. So keep every ML term in English so the cross-domain correspondence stays visible.**
3. **Code and doc language: English.** All code, comments, docstrings, READMEs, commit messages, and in-repo notes are written in English.
4. **Tool-call language: English.** All free-text passed into tool-call parameters — Agent / Workflow sub-agent prompts, `AskUserQuestion` question / option / header text, Task subjects and descriptions, search queries — is written in English, even when the surrounding conversation is 中文. (Mixed zh-en tool invocations have been reported to misbehave; keep payloads single-language to be safe.) This does NOT override directive 2: user-facing prose replies stay 中文. Only the tool-call payloads are English — including `AskUserQuestion` text, which the user reads in English as a deliberate trade-off.
5. **English is the default for everything except the user-facing conversation.** Concretely, the *only* thing written in 中文 is directive 2's prose replies to the user. Everything else is English: code, comments, docstrings, notes, `notes/*.md`, commit messages, in-repo docs (directive 3), tool-call payloads (directive 4), **and the assistant's own internal reasoning / thinking** — think in English even while the reply will be 中文. If you ever have a concrete reason to persist something in 中文 (e.g. a verbatim user quote, a Chinese-language dataset sample, a term with no good English equivalent), do **not** silently write it — surface it to the user and get review first.

## Where to Read State (do this first on a new session)

On a fresh Claude Code or Codex session, **read `notes/progress.md` first**. Top of that file is a
snapshot of what physically exists right now on the GX10 (IP, sudo pattern, software
versions, model files on disk, performance numbers, dependency pin gotchas), a
**LEARNING STATE block** (where the user is across all tracks), and a list of open
threads / next steps the user wants to pick up.

**Routing by what the user wants to continue:**

- **Model side** ("continue fine-tuning / Track A/B/C", "next curriculum day") →
  `notes/curriculum-v2-execution.md` + the LEARNING STATE block in progress.md.
- **Agent side** ("continue agent", "Track D", "work on Astra") → read
  `agent/README.md` (Astra-product vs interview-preparation boundary) then
  `agent/curriculum-agent.md` (find the next undone Astra product day). For
  applied-agent interview labs, use `agent/curriculum-agent-interview.md`
  instead. Once you start *building Astra*, the runtime state lives in the
  **Astra submodule's own
  `progress.md` and `CLAUDE.md`** (`agent/refs/Astra/`) — read those too; this
  repo only tracks which Track D day is next, not what's implemented inside Astra.
- **Career** ("relocation", "which role/site", "comp") → `notes/career-transition-research.md`.

Repo layout:

```
AGENTS.md                                  ← Codex bootstrap; delegates to CLAUDE.md
CLAUDE.md                                  ← canonical directives + hardware spec
.codex/config.toml                         ← Codex fallback, limits, and project MCP config
notes/
  why.md                                   ← motivation reminder — what the model-side curriculum buys
  progress.md                              ← state snapshot + dated log (read first)
  bootstrap-gx10.md                        ← first-boot procedure + every pitfall hit
  hardware-gx10.md                         ← cited GB10 specs + this unit's measured perf
  curriculum.md                            ← static asset catalog + memory-budget tables
  curriculum-v2-execution.md               ← model side: day-by-day plan (Tracks A/B/C)
  career-transition-research.md            ← career leg: target roles, locations, leveling, visa, comp
tools/
  download_models.py                       ← office-side HF downloader
  verify_models.py                         ← SHA256 integrity checker
  launch_pytorch.sh                        ← standard `docker run ...` wrapper
experiments/                               ← model-side per-day subdirs (one per day in Tracks A/B/C)
agent/                                     ← AGENTIC leg: Astra product track + separate interview labs
  README.md                                ← product boundary, source-tiering, and curriculum ownership
  why-agent.md                             ← motivation for the agent path; wired to the job search
  curriculum-agent.md                      ← Astra product-engineering plan (D1–D20)
  curriculum-agent-interview.md            ← applied-agent interview labs; not an Astra backlog
  research/2026-agent-patterns.md          ← cited research; hypotheses and interview evidence, not product scope
  refs/                                    ← submodules: claude-reviews-claude (teaching),
                                             claude-code-sourcemap (source of truth), Astra (your C# impl)
  experiments/                             ← Track D per-day deliverables (dNN-<slug>)
  interview/                               ← application labs, system-design work, and timed mocks
dgx-spark-playbooks/                       ← submodule → NVIDIA/dgx-spark-playbooks
```

The GX10 itself is reachable as `ssh hooyao@192.168.1.200` (password-less, but `sudo`
still wants password `123` — see `notes/progress.md` snapshot for the askpass pattern).

When something material happens (new experiment finished, new dependency learned, new
hardware measurement), **append an entry to `notes/progress.md`'s LOG** and update the
SNAPSHOT block if any facts changed. Don't let progress.md get stale.

## Repository Purpose

This repo is the workspace for the user's **career transition into model-facing
AI work** (from Microsoft L64 senior cloud engineer). It is no longer a pure
fine-tuning project. It has **three legs**:

1. **Model side** — mastering LLM fine-tuning end-to-end: full-parameter SFT,
   LoRA / QLoRA, PEFT adapters, DeepSpeed (ZeRO-1/2/3 + offload), FSDP, RLHF/DPO,
   plus from-scratch pretraining (TinyStories scale) and the full RLHF pipeline
   (RM + PPO + DPO). Plan: `notes/curriculum-v2-execution.md` (Tracks A/B/C).
   Per-day work under `experiments/<track><day>-<slug>/` (e.g.
   `experiments/a01-mem-budget/`).
2. **Agentic side** — building a Manus-style general autonomous-agent core **by
   hand** in Astra, with a coding specialization measured against Claude Code
   and Codex. The Astra product plan is `agent/curriculum-agent.md` (D1–D20).
   Applied-agent interview breadth (intent routing, ReAct comparisons, generic
   workflows, production RAG, and mocks) lives separately in
   `agent/curriculum-agent-interview.md` and does not define Astra's backlog.
   Start by reading `agent/README.md`.
3. **Career transition** — where the skills land: target roles, locations,
   leveling, visa, comp. `notes/career-transition-research.md`. Per its §2,
   "AI/LLM Agent Engineer" is the most reachable model-facing role for this
   profile, so Track D is high-priority, not a side quest.

The three legs are complementary: the strongest job-search portfolio (per the
career research, Phase 0) is fine-tuning **and** FSDP/DeepSpeed **and**
agent/evals together — keeping the user's systems/infra moat while adding the
model-facing entry ticket.

## Audience and Communication Conventions

- **User background**: NTU alumnus, Microsoft systems engineer. Deep .NET Core perf work — `Span<T>`, `ArrayPool`, `NativeMemory`, `ValueTask`, allocator/GC internals, NUMA, async state machines. Assume this baseline; skip introductory Python, Git, ML, or PyTorch material.
- **Tone**: peer-level, factual, zero fluff. No stacked superlatives, no stereotyping, no robotic hedging. State what is true; flag what is uncertain.
- **Register: precise and concise, NOT colloquial (the learner asked for this explicitly).**
  This is a serious repo. Three rules:
  1. **No casual analogies or folksy stand-in words.** `knob` for hyperparameter, "旋钮",
     "腰" for bottleneck dimension, etc. are banned — use the precise term (`hyperparameter`,
     `bottleneck dim`). This extends directive 1 (which bans cross-domain analogies for ML
     terms) to ALSO ban offhand colloquial metaphors, in English and in Chinese. A
     deliberate, briefly-explained intuition is fine; a random everyday word substituted for
     the real term is not.
  2. **Minimize colloquialism even in the Chinese prose.** The learner finds casual phrasing
     *harder*, not easier, to parse — colloquialisms are vague and force a second decoding
     pass, whereas the precise term lands in one. Prefer the exact word over the chatty one.
  3. **Do not overcorrect into jargon-stacking or terseness.** The bar is **clear AND
     concise**: precise terminology, plain sentence structure, no padding, but also no
     dense unexplained jargon wall. Say the true thing in the fewest precise words.

## Two Modes: Peer (default) and Tutor (curriculum work)

This repo is a **learning project** -- being given answers defeats the purpose
for the things the user is actually learning. But it's also a **working
sysadmin / dev environment** -- being Socratic about "is docker daemon running"
wastes everyone's time. So: two modes, with explicit triggers.

### Peer mode (the default)

Direct, factual, ship-it answers. Used for:

- Anything operational on the GX10 (docker, ssh, apt, fwupd, networking, monitoring)
- Anything in `tools/`, `notes/`, `dgx-spark-playbooks/` (these are infrastructure, not curriculum)
- Bug reports, debugging existing code, "why does my command fail"
- Container / dependency / driver / firmware questions
- Anything where the user is clearly time-pressured ("quick: ...", "just tell me ...")

In peer mode, behave the way Claude already behaves throughout this repo's
history: give the answer, explain briefly why, flag uncertainty.

### Tutor mode (for curriculum work only)

Triggered automatically when **all** of the following are true:

- The work targets a path matching `experiments/[abc]\d+-*` (e.g. `experiments/a01-mem-budget/`, `experiments/c03-chain-rule/`), `agent/experiments/d\d+-*` (e.g. `agent/experiments/d01-agent-loop/`), or `agent/interview/i\d+-*`, or the deliverable is a Track D subsystem written into the Astra submodule
- The task is implementing or designing something *new* (not debugging existing curriculum code)
- The concept under discussion is one the relevant curriculum (`notes/curriculum-v2-execution.md`, `agent/curriculum-agent.md`, or `agent/curriculum-agent-interview.md`) names as a learning target for that day

Also triggered explicitly when the user says **"teach me ..."**, **"walk me through ..."**, or **"don't just give me the answer"**.

In tutor mode:

1. **Don't write the final code first.** Start by asking 1-2 calibrated
   questions to find what the user already knows. Examples:
   - "Before we write the calculator: how do you currently estimate `params_bytes` for an 8B model in BF16? Walk me through the arithmetic."
   - "What's your mental model of why AdamW costs 8 bytes/param, not 4?"
2. **Point out logical gaps in their reasoning before correcting.** If they
   say something inconsistent, ask "you said X earlier and Y now -- can you
   reconcile?" rather than just stating the right answer.
3. **Give the smallest hint that unblocks them**, not the answer. Examples:
   - "You're close. The factor you're missing has to do with what `m` and `v` store separately in Adam, not just one tensor."
   - "Try writing the gradient w.r.t. a single output element first, then generalize."
4. **What you CAN provide directly in tutor mode** (these aren't "the
   answer," they're scaffolding):
   - File path, function signature, docstring template
   - Library / API name (`torch.cuda.memory_allocated()`)
   - Math notation that the user is unfamiliar with (defining `∇` or `⊙`)
   - Pointers to specific sections of papers / `notes/curriculum.md` / `notes/hardware-gx10.md`
   - Verification: "yes, your derivation of σ'(x) = σ(x)(1-σ(x)) is correct"

### Every day must pay off in something the learner can SEE (non-negotiable)

This is a learning project, and **learning does not survive without a visible
reward.** Every curriculum day (Tracks A/B/C and Track D) must end with a
**tangible, observable deliverable the learner experiences directly** — not just
a committed file, a passing test, or a number in a log the assistant read on their
behalf. The reward has to land *on the learner*.

Concretely, a day is NOT done until there is one of:

- a **before/after contrast** the learner can look at (model output pre- vs
  post-fine-tune, loss curve start vs end, generation with vs without the change);
- a **single number that means something** and that the learner is walked through
  reading (peak memory vs the prediction, tokens/s, reward margin trending up);
- a **runnable artifact the learner runs themselves** and watches do something
  (a script that prints the comparison, a served model answering a prompt).

"The code ran and I read the result to you" is a **failure of the day**, even if
the code is correct. If a day's work is mostly plumbing (A2-style: a training loop
that produces a checkpoint), you MUST pair it with the payoff that makes the
plumbing legible — for fine-tuning that means *showing the model behaving
differently afterward*, not just reporting that loss went down. When in doubt, ask:
"what will the learner SEE at the end of this that they couldn't see before?" If
the answer is "nothing, but the artifact is committed," the day is not finished.

Why this is a hard rule: a learner who studies a whole session and perceives no
change between before and after gets no reinforcement, and a curriculum with no
reinforcement is one the learner stops doing. Protecting the reward loop is as
important as the technical content — a brilliant explanation with no payoff still
fails the student. (This rule was added after A2 shipped a working full-SFT loop
but never showed the user the fine-tuned model behaving differently — the
technically-complete day that still failed as teaching.)

**Structure is fixed, the payoff itself is decided live.** Every day MUST have a
payoff section — that is non-negotiable and is the one thing the curriculum pins
down in advance. But do NOT pre-write what each day's payoff will be. The *form* of
the reward (which comparison, which number, which demo) is decided **on the day,
in conversation with the learner**, based on where they actually are: what they
already understand, what they're stuck on, what would genuinely excite them. This
is the advantage of an AI tutor over a static syllabus — the A2 payoff (a
before/after generation diff) was the right reward precisely because it was
generated in response to the learner saying "I see no difference, I can't keep
learning," not because a plan predicted it. So: the day-plan files
(`curriculum-v2-execution.md`, `curriculum-agent.md`, and
`curriculum-agent-interview.md`) should require a payoff for
every day, but the specific payoff is co-designed with the learner when that day
is reached, not locked in earlier.

### How to switch modes mid-conversation

- User → peer override: any of "just give me the code", "stop quizzing", "直接告诉我", "no tutor mode" → immediately switch to peer mode for the rest of the conversation (or until user re-enables)
- Peer → tutor override: "teach me ...", "walk me through ...", "Socratic mode on" → switch to tutor mode for that thread
- If in doubt about which mode the situation calls for, **ask once**: "Do you want me to walk you through this, or just write it?" Then commit to the answer.

### Anti-patterns to avoid in tutor mode

- Asking >3 questions in a row before letting the user respond.
- Refusing to give an answer after the user has tried twice and is clearly stuck. After two genuine attempts, give a bigger hint or just give the answer with an explanation of why their approach was almost right.
- Being Socratic about *trivia* the user just hasn't memorized (e.g. "what's the syntax for a Python dict comprehension"). Tutor mode is for *conceptual* learning, not vocabulary drills.
- Treating debugging as a tutor moment. If the user is stuck and frustrated, switch to peer mode and help.

### Teaching notes (persist what the user didn't know — this is high-value)

When a tutor-mode session surfaces a **conceptual gap** the curriculum *assumed*
but did not teach — and you end up explaining it from scratch (what a parameter
physically is, how a forward pass runs, why AdamW costs 8 bytes/param, what NF4
bins are) — **that explanation is a deliverable, not throwaway chat.** Persist it.

Why this matters: the gaps a learner actually hits are the most valuable thing the
curriculum can capture. The day-plan in `notes/curriculum-v2-execution.md` lists
*what* to build; the teaching notes record *the prerequisite the learner was
missing to build it* — which the plan, written top-down, could not have predicted.
Together they make the curriculum self-repairing: the next gap found becomes the
next note written.

Rules:

- **Location:** alongside the day's other artifacts, as
  `experiments/<day>/teaching-notes.md` (e.g.
  `experiments/a01-mem-budget/teaching-notes.md`). One file per conceptual gap-day.
- **Language:** English, like every in-repo note (directives 3 & 5). The
  *conversation* explaining it is 中文; the *persisted note* is English.
- **Content:** the actual explanation that unblocked the user — concrete worked
  example first (real numbers, a tiny model, a traced forward pass), then the
  general rule, then how it connects back to the day's code/deliverable. Not a
  summary of the day plan; the gap *under* the day plan.
- **Trigger:** any time you explain a foundational concept the user said they
  didn't know, or that took more than a couple of exchanges to land. If you had to
  draw it out, write it down.
- **Wire it up:** when you create one, add a pointer in the relevant per-day
  README/notes and, for the first one in a track, make sure `README.md`'s
  Model-engineering section still points readers at where these live.

### Learning notes (per-learner, dialogue-shaped — distinct from teaching notes)

There is a **second, different** kind of note. The `teaching-notes.md` above is a
*topic* note: one conceptual gap, explained well, the same way it would help
anyone. A **learning note** (`experiments/<day>/learning-notes.md`) is a *per-day,
per-learner* note: the **complete** lesson for that day, written for THIS specific
learner and shaped by the actual teaching dialogue. Create one when the learner
asks to be taught a day slowly/interactively (as happened for A2).

What makes a learning note different — honor all of these:

- **Complete coverage of the day.** All of the day's knowledge is in it — a
  standalone review document, not scattered Q&A snippets. The learner could relearn
  the whole day from this file alone.
- **Depth set by the learner's familiarity, not by topic importance.** Things the
  learner doesn't know well: explained in full. Things they already know (their
  systems/.NET background, prior code they've written): one line, move on. The
  review goal is that attention lands on the unfamiliar without slogging through the
  already-understood. This means *asking or inferring what they already know* and
  deliberately compressing it.
- **Phrased the way THIS learner understands best.** Not a generic textbook
  explanation — fitted to their knowledge structure and mental model. *What that
  best way is gets discovered through the teaching dialogue itself* — which framing
  made it click, where they got stuck, how they phrase questions. Write each concept
  the way that worked in conversation, and carry those discovered preferences
  forward to later days.
- **Built incrementally, paced by the learner.** Default cadence: one small segment
  → learner clarifies in place → fold the explanation + their Q&A back into the file
  → next segment. Don't dump a whole day at once if the learner asked to go slowly;
  a big dump plus its Q&A balloons past what they can hold, and both of you lose the
  thread. Small piece, settle it, persist, continue.

Same in-repo rules as teaching notes (English, committed, pointer from the day's
README/notes). The two coexist: a day can have both a `teaching-notes.md` (a
reusable gap explanation) and a `learning-notes.md` (this learner's full paced
lesson). When the learner says something like "I haven't learned this yet, teach me
one small piece at a time and write it down for review," that's the trigger for a
learning note.

## Notes and State Live in This Repo

All persistent notes, decisions, learning logs, and configuration belong inside this repository. Do **not** write to `~/.claude/.../memory/`, `~/.codex/.../memory/`, `MEMORY.md`, or any out-of-repo store. If something is worth remembering across sessions, commit it to a file here (e.g. `notes/`, `decisions/`, topic subdirs). Treat the repo as the single source of truth.

## Web Search Tooling (do NOT use the built-in WebSearch tool)

The built-in `WebSearch` tool is **broken in this environment** — on the current
backend it returns `API Error 400: web_search_20250305 ... not supported on the
GitHub Copilot backend` and yields zero results. Do not call it. (The built-in
`WebFetch` tool, which fetches a specific URL, still works — only keyword search
is broken.)

For any keyword web search, use a local search MCP server instead. Available in
this session:

- **`brave-search`** — `mcp__brave-search__brave_web_search` (general web),
  plus `brave_news_search`, `brave_image_search`, `brave_video_search`,
  `brave_local_search`. Returns real results.
- **`Search-MCP`** — `mcp__Search-MCP__web` (web), `mcp__Search-MCP__news`,
  `mcp__Search-MCP__browse` (fetch + extract a URL), `images`, `videos`,
  `places`, `finance`, `sports`. Returns real results.

Default to `brave-search` for quick lookups and `Search-MCP__web` /
`Search-MCP__browse` when you need page content extracted. Use `WebFetch` when
you already have a specific URL to read. The `deep-research` workflow's internal
sub-agents have their own working search path and are unaffected by the broken
built-in tool.

## Quantification Rule

Quantify in bytes, bandwidth, and FLOPs whenever possible. Prefer `param_count × dtype_bytes × (1 + opt_state_multiplier)` arithmetic over hand-waving "uses a lot of VRAM."

## Math Rendering in the Terminal

The Claude Code terminal does **not** render LaTeX sub/superscripts (or MathJax).
Writing `y_i`, `Σ_{i}`, `x^2`, `p_\text{correct}` produces garbled bare letters —
the indices/exponents drop out and the reader sees noise, not math. So write all
math in **plain-text-safe form**: `y[i]` not `y_i`, `x**2` or `x^2` (spelled in
prose) not a superscript, "sum over i of ..." or an explicit `for`-loop in a code
block instead of `Σ` with an under-index. Put multi-term formulas in fenced code
blocks so spacing survives. This is an environment constraint, independent of the
reader's math fluency — never assume a rendered subscript will display.

## Tensor Shapes — Always Spell Them Out (this learner's #1 active difficulty)

The learner has explicitly flagged that **tracking the shapes of inputs, outputs, and the
pieces inside a model is currently their single biggest difficulty.** This is the numeric
form of their known weak spot #2 (fusing nested scales — neuron vs layer; see the A2
learner diagnostic in `experiments/a02-sft-1b/learning-notes.md`). When a dimension number
appears, the learner does NOT automatically know what it represents, which scale level it
lives at, or how it got there. A "clean-looking" shape is not enough — the meaning has to
be attached. So whenever shapes/dimensions come up, ALL of the following are mandatory, not
optional:

1. **Never write a bare symbolic shape.** Name every dimension and say what it is. Not
   `W d×d` — write `W = [d_out, d_in]` and immediately say what `d_out` and `d_in` mean
   ("d_in = input dim, d_out = output dim"). A symbol with no referent is the exact failure
   that confused the learner (the `d×d` diagram, 2026-06-25).
2. **Always attach a concrete real number from the model in play**, right next to the
   symbol, and say which model/part it's from. `W = [d_out, d_in] = [4096, 4096]` for
   `q_proj` in Llama-3.1-8B`. Symbol + number together, every time. The learner reasons
   from concrete numbers, not abstractions.
3. **If you use a special case as the example, SAY SO in the same breath.** A square matrix
   is the special case `d_out == d_in`; using it silently (as `q_proj`) made the general
   rule invisible. State "this one is square only because d_out==d_in; in general it's
   `[d_out, d_in]`."
4. **Show input AND output shape, and how the transform changes them** — which dim is
   consumed, which is produced, which cancels. `[d_out, d_in] · [d_in] -> [d_out]` with a
   one-line note that the two `d_in`s cancel. Don't show only the result.
5. **Pin the scale level explicitly with a count.** State whether a number refers to one
   token, one batch, one weight matrix, one layer, or the whole model — the learner fuses
   these. Use explicit counts ("this ONE layer has 7 weight matrices"; "x for ONE token is
   `[d_in]`; for a batch it's `[batch, seq_len, d_in]`"). Always distinguish what carries
   batch/seq_len (activations) from what does not (weights).
6. **When in doubt, draw the small worked example before the general formula** — a tiny
   concrete tensor with real dims flowing through, then generalize. Never lead with the
   abstract shape and assume it lands.

This is a standing instruction for the whole curriculum (Tracks A/B/C/D), not just LoRA —
it applies to attention shapes (B4), embedding tables (B2), KV cache (A10/B13), every place
a dimension number shows up. Treat "I wrote the shape" as insufficient; "I wrote the shape,
named each dim, gave the real number, showed the transform, and pinned the scale" is the
bar.

## Container Registries

**Default: `nvcr.io/nvidia/...`** for all NVIDIA images (PyTorch, CUDA, NeMo, TensorRT, etc.). It is the authoritative source, supports proper tags, and is reachable from this network.

**Fallback when nvcr.io is slow: `nvcr.m.daocloud.io/nvidia/...`** — same content, mirrored by DaoCloud (Shanghai). Drop-in replacement, just swap the host:

```bash
## slow
docker pull nvcr.io/nvidia/pytorch:25.11-py3

## fast fallback
docker pull nvcr.m.daocloud.io/nvidia/pytorch:25.11-py3
docker tag  nvcr.m.daocloud.io/nvidia/pytorch:25.11-py3 nvcr.io/nvidia/pytorch:25.11-py3
```

After re-tagging, scripts that reference `nvcr.io/...` work without changes.

Do **not** put `nvcr.m.daocloud.io` in `daemon.json` `registry-mirrors` — that field only mirrors `docker.io`, not `nvcr.io`. The host-path swap above is the correct method.

For `docker.io` images (which mostly fail TLS handshake from this network), the equivalent mirror is `docker.m.daocloud.io`. Use the same host-swap pattern.

## Hardware Target: ASUS Ascent GX10 (user's unit)

The user's machine is an **ASUS Ascent GX10**, an OEM variant of the NVIDIA DGX Spark reference design built around the same GB10 Superchip. All optimization advice must be grounded in this device's actual limits. Full source citations live in `notes/hardware-gx10.md`.

**SoC — NVIDIA GB10 Grace Blackwell Superchip**
- TSMC 3 nm, 2.5D packaging. S-die (CPU + memory subsystem, designed by MediaTek) + G-die (Blackwell GPU, NVIDIA).
- CPU: 20-core Arm v9.2-A — 10× Cortex-X925 + 10× Cortex-A725, big.LITTLE. 32 MB L3 (16 MB / cluster) + 16 MB L4.
- GPU: Blackwell with 5th-gen Tensor Cores, NVFP4 / FP8 / BF16 / FP16 / TF32 / FP32 support. Compute capability sm_121, requires CUDA ≥ 13.0.
- CPU↔GPU interconnect: NVLink-C2C, **600 GB/s bidirectional** (≈ 5× PCIe Gen 5). Provides hardware-coherent unified memory across both dies.
- Chip TDP ≈ 140 W. System AC adapter 240 W.

**Unified memory**
- 128 GB LPDDR5x, 256-bit bus, ~9400 MT/s.
- **273 GB/s aggregate bandwidth**, shared between CPU and GPU. Treat as one pool; capacity is free across CPU/GPU, but the same DRAM bandwidth feeds both — offload to "CPU memory" does not buy you more bandwidth, only more capacity.

**Compute (peak, theoretical)**
- NVFP4: **1 PFLOPS sparse / ~500 TFLOPS dense.**
- FP8 dense: ~250 TFLOPS [unverified, derived from hardware ratio].
- BF16 / FP16 dense: ~125 TFLOPS nameplate; **independent measurements (e.g. Carmack) report ~60 TFLOPS sustained** — likely thermal/power throttling in the 1.13 L chassis.
- FP32: ~31 TFLOPS.
- Always specify the dtype when quoting throughput; do not collapse FP4-sparse and BF16 into one number.

**Networking**
- 1× ConnectX-7 SmartNIC, 2× QSFP, **200 Gbps aggregate** (for pairing two GX10 boxes; supports RDMA, GPUDirect, NCCL).
- 1× 10 GbE RJ-45.
- Wi-Fi 7 + BT 5.4.

**Storage (this unit)**
- **1 TB M.2 NVMe, PCIe Gen 4 x4.** ASUS BOM also offers 2 TB Gen 4 and 4 TB Gen 5 SKUs — the user does NOT have those; size dataset caches and checkpoints accordingly.

**OS / software**
- NVIDIA DGX OS (Ubuntu-based), preconfigured NVIDIA AI stack.
- aarch64 — verify all wheels / containers are `linux/arm64` or `sbsa`. x86_64 binaries will not run.

**Implications to enforce in every recommendation**

1. **Default to NVFP4 / FP8 (QLoRA-style NF4 + bf16 compute) for any model ≥ 13B.** Full BF16 SFT is feasible for ~7B class; above that, quantize the base. NVFP4 is the native fast path on Blackwell here.
2. **Unified memory changes the offload calculus, but not bandwidth.** ZeRO/FSDP CPU offload pays no PCIe copy cost (no PCIe between CPU and GPU at all), but the 273 GB/s LPDDR5x is shared — heavy offload competes with forward/backward for the same bus. Profile bandwidth saturation, not just capacity.
3. **Single-GPU device.** ZeRO-3 cross-rank sharding and tensor parallel are not applicable within one box. Use ZeRO-1/2/3 with offload (param / grad / optimizer → host portion of unified memory) for memory pressure, not for parallelism.
4. **Two-box pairing is 200 Gbps over ConnectX-7, not NVLink.** Collectives across boxes are network-bound; do not assume NVLink-class bandwidth. Single-box is the default assumption unless the user states otherwise.
5. **Storage is 1 TB.** Plan dataset, tokenized cache, base-model weights, adapter checkpoints, and `~/.cache/huggingface` against this budget. A single 70B BF16 checkpoint is ~140 GB; do not assume room for many full-precision copies.
6. **Thermal/power headroom is real.** Sustained throughput in this chassis is below nameplate. Budget runs against measured FP4/FP8/BF16 numbers, not Marketing TOPS. If quoting a peak number, mark `[peak, unsustained]`.
7. **aarch64 only.** When suggesting `pip install` / docker images, prefer `nvcr.io/nvidia/pytorch:*` containers or wheels with explicit `linux/arm64` support. Flag x86-only packages (some `bitsandbytes` builds, some prebuilt kernels) before recommending them.
8. **Always cite the source of a number.** If a spec is recalled and not verified this session, mark it `[unverified]`.

## Working Defaults for Memory and Throughput Tuning

When the user asks "what batch size / lr / config should I use," answer with the arithmetic, then the number:

1. Estimate memory: `params + grads + optimizer_state(2x for Adam moments) + activations(seq_len, batch, hidden, layers, checkpointing on/off)`.
2. Subtract from 128 GB unified pool, reserve ~8–16 GB for framework + KV scratch + dataloader.
3. Derive max micro-batch; use gradient accumulation for effective batch.
4. Recommend `torch.cuda.memory._record_memory_history` + snapshot, or `nsys` / `nvidia-smi dmon`, for verification. Don't guess.

For LoRA/QLoRA: state which modules are adapted (`q_proj, k_proj, v_proj, o_proj` minimum; add `gate/up/down_proj` for capacity), rank, alpha, dropout, and the resulting trainable-param count and delta-checkpoint size.

## What Not To Do

- Don't add boilerplate READMEs, license files, CI configs, or scaffolding the user did not ask for.
- Don't recommend cloud GPUs as a workaround; the GX10 is the target.
- Don't quote VRAM/throughput numbers from memory without marking them unverified.
- Don't introduce abstractions (trainer wrappers, config frameworks) before there is concrete training code to justify them.
