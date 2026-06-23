# D5 — Permission pipeline: source-of-truth reconciliation (written BEFORE coding)

Per the process fix adopted after D4: read `refs/claude-code-sourcemap` first,
state what Claude Code actually does, then decide where Astra agrees / simplifies.
This note is that read; the design + implementation notes come after.

Sources read (this session, via an Explore pass over the decision core, not the
React UI):
- `src/types/permissions.ts` — the decision + rule types
- `src/utils/permissions/permissions.ts` — `hasPermissionsToUseToolInner` (the
  evaluation order)
- `src/utils/permissions/shellRuleMatching.ts` — bash command matching
- `src/utils/permissions/PermissionMode.ts` — the modes
- `src/hooks/toolPermission/PermissionContext.ts` — how an "ask" is resolved

## 1. The decision is a 3-state union, not a bool

```ts
type PermissionDecision =
  | PermissionAllowDecision   // { behavior:'allow', updatedInput? }
  | PermissionAskDecision     // { behavior:'ask',  message, suggestions?, ... }
  | PermissionDenyDecision    // { behavior:'deny', decisionReason }
```

The load-bearing fact for our design: **`ask` is a first-class result**, not an
exception or a bool=false. `allow` may carry an `updatedInput` (the tool can
rewrite its own args before running — e.g. normalize a path); `deny` carries a
reason; `ask` carries a human-readable `message` (+ suggestions). So our C# type
must be a discriminated union of three cases, and `Allow` must be able to carry a
possibly-rewritten argument bag.

**How `ask` resolves:** NOT a return value the engine computes. CC enqueues a
pending request (`permissionQueue` in `PermissionContext`) and a resolver
(`createResolveOnce`) that fires when the user clicks Allow/Deny in the UI. So the
permission check is *async and may suspend* waiting for a human. For Astra this
means `CheckPermissionAsync` returns a `Task<PermissionDecision>` and the `ask`
path must call out to an injected "ask the user" delegate — the engine does not
own the UI.

## 2. Rules match by tool name, then by command content (prefix/wildcard/exact)

```ts
type PermissionRule = {
  source: PermissionRuleSource           // policy/user/project/session tier
  ruleBehavior: 'allow' | 'deny' | 'ask'
  ruleValue: { toolName: string; ruleContent?: string }  // e.g. Bash("npm install")
}
```

- A rule with **no `ruleContent`** matches the whole tool (e.g. "all of Bash").
- A rule with `ruleContent` matches a specific command; `shellRuleMatching.ts`
  supports **prefix** (`npm:*`), **wildcard** (`git c*` → regex), and **exact**.

This is the command-string keying that D2 deliberately moved away from for the
*bulk* decision (we classify by behavior class instead). Reconciliation: the two
are **complementary, not competing** — D2's `ToolAction` is the coarse "allow all
reads" bulk decision; CC's per-command rules are the *exception* layer (a deny
for `Bash(npm publish:*)` even though bash-as-a-class might be allowed). D5
should keep behavior-class as the default decision and add a **thin rule layer
for per-command allow/deny/ask exceptions** — exactly the "two-layer permission
model" Astra's own CLAUDE.md promised at D2.

## 3. Precedence + fail-closed: DENY > ASK > ALLOW, default = ASK

Evaluation order in `hasPermissionsToUseToolInner` (simplified to what we keep):

1. deny rules on the whole tool → **deny** (wins over everything)
2. ask rules on the whole tool → **ask**
3. tool's own `checkPermissions()` → may deny / ask / passthrough
4. mode short-circuits (`bypassPermissions` → allow) — but safety/content asks are
   **bypass-immune** (a `Bash(npm publish:*)` deny and `.git/` safety check still
   prompt even under bypass)
5. allow rules on the whole tool → **allow**
6. nothing matched → `passthrough` → **converted to `ask`**

**Fail-closed default is `ask`, not `allow` and not silent `deny`.** When no rule
decides, CC asks the human. Within rules, **deny is checked before allow** so a
deny always wins a conflict. Two distinct fail-closed points, which answers the
D5 design question directly:
- **Classify fail-closed (D2):** unknown tool → `ToolAction.Other` (strictest
  *class*). Guards "we don't know what this tool does."
- **Rule-match fail-closed (D5):** no rule matches → `ask` (don't auto-run).
  Guards "we have no policy for this specific call." Different gap, different
  default; both fail safe.

## 4. Modes (we implement a subset)

`default` (prompt on ask), `acceptEdits` (auto-allow edits in cwd),
`bypassPermissions` (skip rule checks → allow, except bypass-immune safety/content
asks), `plan` (paused), `dontAsk` (ask→deny). For D5 we implement **`default`**
and leave a seam for `bypassPermissions`; the rest are later.

## 5. Layer mapping (curriculum says: build 1, 2, 5)

| Layer | CC | D5 scope |
|---|---|---|
| 1 input validation | schema + abort check | **build** — validate args against `InputSchema` before anything |
| 2 rule matching | deny>ask>allow, tiered sources | **build** — the thin exception layer over D2 Classify |
| 3 domain security | bash AST, dangerous patterns | cite (D2's Classify is the v1 stand-in) |
| 4 classifier | YOLO/transcript AI side-query | cite, defer |
| 5 user confirm | `ask` → UI prompt | **build** — one injected confirm delegate |
| 6 sandbox | OS isolation | cite, defer |
| 7 workspace trust | trust dialog | cite, defer |

## 6. What NOT to copy (CC complexity to drop)

- **`passthrough` intermediate state** — exists only to defer to step 3 then become
  `ask`. We let a tool's permission check return `ask`/`deny`/`null(=no opinion)`
  directly; no third internal behavior.
- **Competing resolvers / `ResolveOnce` queue** — CC races classifier auto-approve
  vs. user click. We have no classifier in D5, so `ask` is a single `await` on the
  injected confirm delegate. No queue, no race.
- **Three parallel rule arrays** (allow/deny/ask per source) — use one ordered
  rule list; the behavior is a field, precedence is deny>ask>allow applied at match
  time.
- **Feature-gated classifier branches** — out of scope; leave an interface seam,
  not conditionals.

## Design consequences carried into the implementation note

1. `PermissionDecision` = C# discriminated union (record hierarchy):
   `Allow(IDictionary? updatedArgs)`, `Deny(string reason)`, `Ask(string message)`.
2. `CheckPermissionAsync(call, ct) -> Task<PermissionDecision>` lives at the seam
   in `AgentLoop.RunOneToolAsync`, BEFORE `tool.ExecuteAsync`.
3. The engine evaluates: input-validate → rule list (deny>ask>allow) → fall back
   to a per-`ToolAction` default (Read→allow, Write/Execute/Other→ask). `ask` calls
   an injected `IUserConfirmation` delegate; the CLI implements it, tests fake it.
4. Fail-closed everywhere: unknown tool / no rule / Other class → `ask` (or deny in
   a non-interactive host), never silent allow.
