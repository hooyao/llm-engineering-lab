# D5 — Permission pipeline: design + implementation record

> Post-D8 update (2026-08-31): permission now consumes immutable
> `ToolDefinition` values. Classification and denial occur before keyed
> transient `IToolExecutor` activation, so a denied call constructs no executor.
> The permission ordering and decisions documented below are unchanged.

Track D Day 5. Goal: a tool that can run `rm -rf` needs a gate before execution.
Build the load-bearing layers (1/2/5) of Claude Code's 7-layer permission model,
wired into the loop BEFORE `ExecuteAsync` so a denied side effect never happens.
Builds directly on D2's `Classify -> ToolAction`.

Deliverables (Astra submodule, PR #6, `e3b52a6`):
- `src/Astra.Core/Permissions/` — `PermissionDecision`, `PolicyVerdict`,
  `PermissionRule`, the three interfaces (`IPermissionPolicy`, `IUserConfirmation`,
  `IPermissionEngine`), and the defaults (`ClassDefaultPolicy` + `AlwaysAskPolicy`,
  `DefaultPermissionEngine`).
- `AgentLoop` — optional `IPermissionEngine?`; the gate in `RunOneToolAsync`.
- `AgentEvent.ToolDenied` — the human-facing refusal event.
- Tests: `ClassDefaultPolicyTests` (6), `DefaultPermissionEngineTests` (6),
  `AgentLoopPermissionTests` (4). **54/54 pass.**
- This dir also holds `source-reconciliation.md` (the source read, written BEFORE
  coding per the post-D4 process fix) and `teaching-notes.md` (why 3-state, the two
  fail-closed points). `interface-skeleton.cs` was the pre-impl review draft; the
  real code lives in Astra now, so it is removed.

## Process: source read came first this time

Per the D4 retro, D5 started with `source-reconciliation.md` — a read of CC's
`utils/permissions/{permissions,PermissionRule,shellRuleMatching,PermissionMode}.ts`
and `types/permissions.ts` — BEFORE any code. That read produced the design
directly: 3-state decision, deny>ask>allow precedence, default = ask, the
`passthrough`/`NoOpinion` boundary, and a list of CC complexities to NOT copy.

## Design (user-directed, tutor mode)

The user set two design constraints that shaped the interfaces:

1. **Default = class-based ("Y"), not always-ask ("X").** No rule matched falls
   back to the behavior class: Read -> Allow, else -> Ask. Rules are the *exception*
   layer (deny a dangerous command, pre-allow a safe one). This is the two-layer
   model CLAUDE.md specified at D2; it avoids the "every `ls` prompts" UX failure
   that drives users to disable permissions.
2. **Highly modular for SDK release.** Not one engine but three seams:
   `IPermissionPolicy` (WHAT the decision is — the primary swap point),
   `IUserConfirmation` (HOW an ask resolves), `IPermissionEngine` (orchestration,
   the escape hatch). A host customizes policy and confirmation independently;
   replacing the whole engine is the last resort.

Two scope calls:
- **Policy-only, no per-tool veto.** A tool exposes only its `ToolAction` (D2
  `Classify`); it cannot return its own allow/deny/ask. (CC has a
  `tool.checkPermissions()` step; we omit it — `ITool` stays stateless per
  CLAUDE.md. Added later if a tool ever needs a self-veto beyond its class.)
- **`IPermissionPolicy.EvaluateAsync` returns `ValueTask`, not sync.** The user's
  call: a public SDK extension point must not be locked to sync, or a host wanting
  a remote policy is forced into `.Result` (sync-over-async deadlock). `ValueTask`
  (not `Task`) because the default policy completes synchronously on the per-call
  hot path — no Task allocation. Awaited exactly once in the engine; never stored.

## The decision model (see teaching-notes.md for the why)

- `PermissionDecision = Allow(updatedArgs?) | Deny(reason) | Ask(message)` — 3-state,
  never a bool. `Allow` can rewrite arguments; `Deny`'s reason goes back to the LLM.
- `PolicyVerdict` = the same three + `NoOpinion` (CC's `passthrough`), confined to
  the policy boundary — the engine maps `NoOpinion` to its fallback and never
  propagates it.
- Fail-closed, two points: Classify (D2) defaults an unknown *tool* to `Other` (the
  strictest class); rule-match (D5) defaults an unpoliced *call* to `Ask`
  interactively, or `Deny` headless (no confirmer = nobody to ask).

## Layer mapping

| Layer | This day |
|---|---|
| 1 input validation | minimal — tool existence. Full InputSchema validation is a marked TODO (seam in place). |
| 2 rule matching | `ClassDefaultPolicy`: class default + prefix rule exceptions, deny>ask>allow. |
| 5 user confirm | `IUserConfirmation` — one injected delegate; CLI/headless/test. |
| 3,4,6,7 | cited in source-reconciliation.md, deferred. |

## Wiring + the load-bearing test

`AgentLoop` gains `IPermissionEngine? permissionEngine = null` (null = unguarded,
backward-compatible — the pre-D5 38 tests are unchanged). `RunOneToolAsync` calls
`CheckAsync(call, ClassifyCall(call), ct)` before `ExecuteAsync`; a `Deny`
short-circuits — the reason becomes the tool result fed back to the LLM, plus a
`ToolDenied` event — and the tool never runs. An `Allow` may carry rewritten args.

`AgentLoopPermissionTests.DeniedCall_ToolNeverRuns_ReasonFedToLlm` is the load-
bearing test: a `SpyTool` counts executions and the test asserts `Executions == 0`.
A test that only inspected the decision could pass while the tool still ran;
counting executions proves the side effect did not happen.

## Open / deferred

- **Full Layer 1** — validate arguments against `ITool.InputSchema` (JSON Schema).
  TODO marked in `DefaultPermissionEngine`; the seam is there.
- **Richer rule matching** — CC supports wildcard (`git c*`) and exact in addition
  to prefix, plus source tiers (policy>user>project>session) with an ordering. D5
  does prefix + a flat list. Expand when real config loading exists.
- **Per-tool `checkPermissions`** — omitted by the policy-only scope; revisit if a
  tool needs a self-veto beyond its `ToolAction`.
- **CLI confirmation UI** — `IUserConfirmation` has no terminal implementation yet
  (tests use a fake; the CLI currently constructs `AgentLoop` with no engine).
  Wiring a `[y/N]` prompt into `AgentApp` is the natural next step to make D5
  user-visible.
- **Modes** — only the implicit `default` mode exists. `bypassPermissions`,
  `acceptEdits`, `plan`, `dontAsk` (and CC's bypass-immune safety/content asks) are
  noted in source-reconciliation.md, deferred.

## Git state

Astra PR #6 squash-merged to main as `e3b52a6`; parent-repo submodule pointer
bumped `11b6aed -> e3b52a6` together with this dir's notes and the progress.md
update. Same flow as D1–D4; merge needed `--admin`. (This session also resolved a
parent-repo merge conflict in progress.md from parallel Track A work on another
machine — append-only LOG, both entries kept; see that merge commit.)
