# ADR 0001 — One-way dependency graph, enforced; one composition root

- Status: Accepted
- Date: 2026-10-12
- Supersedes: none

## Context

`cvtrain` had grown to 14 modules, and nothing stopped one module from importing
a neighbour it had no business knowing. A policy module importing the
orchestrator, a primitive importing a policy, or `TrainModel.py` reaching past
`cli` would all "work" in CI until the design rotted past the point any single
review could see it. The Qt app ships regardless of the Python side, so drift
here is silent and expensive.

## Decision

Dependencies point exactly one way, from orchestration down to primitives, and
the map is **a table in the test suite, not a comment**: `tests/test_architecture.py`
parses every `cvtrain/*.py` with `ast` and fails the build when any import lands
outside the declared `ALLOWED_IMPORTS` map.

- `cli` is the package's only **composition root**: it is the single module
  allowed to name the concrete `UltralyticsBackend`, and it passes the instance
  to `trainer.run(cfg, backend=…)` where the parameter is required.
- `trainer` orchestrates and imports no framework.
- `gate` and `preflight` hold policy; `manifest`/`dataset`/`backend` hold
  documents and seams; `fsio`, `checks`, `errors`, `progress`, `config`,
  `logging_setup` are primitives.
- The heavy stack (`ultralytics`, `yaml`) is importable from exactly one module
  (`backend`), and only lazily, so `--help`/`--check` and the offline suite run
  with no ML packages installed.
- A new edge is a change to the table — reviewed in the same commit, by the
  same build — not an invisible infra change.

## Consequences

- Layering cannot rot silently while the suite stays green.
- Adding a module, or a dependency, is a small but explicit act with a review
  surface.
- `fsio` imports `errors` (it raises a typed `ConcurrentRunError` from the
  advisory lock); every other primitive is a true leaf.