# llm-engineering-lab

Hands-on LLM engineering, built from scratch on a single NVIDIA GB10 box
(ASUS Ascent GX10, 128 GB unified memory). Two tracks, every artifact runnable.

## Model engineering

Fine-tuning and pretraining end to end, sized against real hardware limits:

- **Fine-tuning** — full SFT, LoRA, QLoRA (NF4 + bf16), DPO — with the memory
  arithmetic to size any model before launching, and measured throughput on the
  GB10 (BF16 GEMM characterized at ~93 TFLOPS sustained).
- **From scratch** — tokenizer (BPE), attention, a small LLM pretrained on
  TinyStories, then the full RLHF pipeline (reward model + PPO + DPO).
- **Serving** — vLLM throughput/latency tradeoffs.

→ `notes/curriculum-v2-execution.md`

**Teaching notes.** Where a day assumed a concept it didn't teach, the gap is
explained from scratch and committed as `experiments/<day>/teaching-notes.md` —
concrete worked example first, then the rule, then how it feeds the day's code.
These record the prerequisites a top-down curriculum can't predict; the first is
[`experiments/a01-mem-budget/teaching-notes.md`](experiments/a01-mem-budget/teaching-notes.md)
(what a parameter physically is, how a forward pass runs, and where the training
bytes go). They are first-class deliverables, not scratch chat — see CLAUDE.md
§ "Teaching notes."

## Agent engineering

Building a Claude Code-style agent core **by hand** in C# ([Astra](https://github.com/hooyao/Astra)),
then the half Claude Code deliberately omits:

- **The core** — the agent loop, tool orchestration (concurrent reads / serial
  writes), streaming, a layered permission pipeline, context assembly,
  four-tier compaction, multi-agent coordination.
- **The frontier** — agentic RAG with corrective grading (CRAG), RAGAS + LLM-as-judge
  evaluation, long-term memory, OpenTelemetry tracing, MCP/A2A interop.

Grounded in primary engineering sources (Anthropic, Cognition) and the restored
Claude Code source, not framework tutorials.

→ `agent/README.md`

## Principle

Quantify everything in bytes, bandwidth, FLOPs, and tokens. Re-implement the
patterns rather than import a library — the goal is to understand every line,
not to wire one together.
