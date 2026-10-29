# D10 — Durable Session Resume

Status: design discussion started; runtime implementation has not started.
D9's learner-run payoff remains pending independently of this discussion.

The first agreed recovery rule reuses D9 file freshness: persist the executing
agent's content observation and the pending call, retry a file-content operation
when current content matches that observation, and otherwise return a recoverable
request to read again so the model can decide from current evidence.

See [teaching-notes.md](teaching-notes.md) for the worked example, scope boundary,
and proposed separation between session persistence and tool recovery.

The learner-visible D10 payoff has not yet been selected in conversation.
The curriculum requires a real interrupted task to resume without blindly
duplicating completed side effects; it is not satisfied by a passing unit test.
