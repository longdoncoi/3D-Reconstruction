# ADR 0011 — Command sandbox rejects inline interpreter code and shell substitution

- Status: Accepted
- Date: 2026-10-10
- Supersedes: none

## Context

`run_command` is a code-execution tool gated by approval. The sandbox filter
blocked shell metacharacters but still allowed `python -c "...code..."`,
`python -m module`, or a stdin script (`python -`): each of those is arbitrary
code execution disguised as a "command", and `$(...)` / backquote substitution
lets a caller smuggle a second command past the token filter (Sprint 1.2
review).

## Decision

`tools/sandbox.py` rejects, purely from argv inspection:

- inline interpreter code: `-c`, `-m`, and the stdin script marker `-`
  (message: `Inline interpreter code (-c, -m, -) is rejected by sandbox
  policy.`);
- interactive `-i` and final-result `-O` flags on a script invocation;
- shell substitution: backticks and `$(` anywhere in the command,
  independently of the existing metacharacter filter.

Running an interpreter *on a script file* with only the allow-listed
informational flags (`--version`, `--help`, `-V`, `-h`, `-q`, `-B`, `-s`,
`-E`) stays allowed, so legitimate project tooling keeps working.

## Consequences

- Arbitrary code execution outside an approved script file is closed.
- The check is static (no shell involved), so it cannot be bypassed by quoting
  tricks and is trivially unit-testable.
- Command tools that legitimately need `python -c` must be replaced by an
  explicit approved-script workflow instead.