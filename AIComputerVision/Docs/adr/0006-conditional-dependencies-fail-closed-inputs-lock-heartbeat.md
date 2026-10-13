# ADR 0006 — Conditional dependencies, fail-closed inputs, lock heartbeat

- Status: Accepted
- Date: 2026-10-13
- Supersedes: partially 0005 (its "heartbeated locks" non-goal and its
  write-side `OSError` carve-out)

## Context

An independent audit of the 0005 state found no defect in the default path,
but named gaps that all share one shape: text asserting more than the code
does, plus two places where a failure still escaped the typed taxonomy.

- `--tensorboard` was not part of preflight's dependency list, yet `prepare`
  logged "TensorBoard logging enabled" unconditionally. Ultralytics drops
  TensorBoard *silently* when the package is missing, so an operator could get
  a green preflight, a success line, and no logs at all.
- `required_modules` was a constant, so it could not express a dependency that
  exists only under a flag.
- `atomic_write_text`, `atomic_copy` and `directory_lock` let `OSError` escape
  as a traceback with exit `1`, while the reads beside them were already
  `StorageError` (exit `2`): same cause, two codes.
- `--verify-manifest` read `Models/` without the advisory lock, so a run
  publishing at that instant reported a mismatch that existed only as a race.
- The 24 h stale-lock rule had no renewal. 0005 declined a renewal *thread*;
  the failure it was meant to prevent was nevertheless real, because
  `MAX_EPOCHS` is deliberately generous.
- `gate.enforce_quality_gate` trusted its input's shape: a hand-edited manifest
  carrying a string where a metric map belongs raised `AttributeError`, and a
  recorded `NaN` satisfies `actual < minimum` in IEEE-754 arithmetic —
  publishing a model that met no bar at all.
- Doc drift: ADR 0001 named `backend` as `yaml`'s owner (the enforced table
  says `dataset`), `manifest`'s docstring claimed the Qt app reads the
  manifest (nothing under `src/` does), and `missing_dataset_paths` documented
  remote URLs as accepted while rejecting them.
- The C++ contract tests skipped themselves when a source file was missing —
  correct for a Python-only checkout, but a *rename* produced the same
  silence, leaving CI green with no comparison running.

## Decision

- `TrainingBackend.required_modules` becomes a method taking the config.
  `UltralyticsBackend` returns its base modules plus `tensorboard` when
  `--tensorboard` is set, so a feature the pipeline cannot turn on fails in
  preflight (exit `2`) instead of an hour into training. `prepare` probes both
  the package and the framework's settings object and logs what actually
  happened — never a success it did not have.
- The framework keyword mapping moves from `config` to
  `backend.train_kwargs`, so `TrainConfig` holds no framework syntax — what
  ADR 0002 already described.
- Write-side failures (`mkdir`, `mkstemp`, `fsync`, `replace`) — and the lock's
  own `O_EXCL` create and token write — become `StorageError`. One carve-out
  remains: the *source* of `atomic_copy`, an export that produced no file is a
  bug and keeps its traceback. Cleanup (`_discard_temp`, `_unlink_quietly`) is
  best effort: it warns, so a leftover temp file can never replace the verdict
  that was already on its way out, and a lock that cannot be deleted never
  turns a held lock into a traceback.
- `--verify-manifest` takes the same advisory lock as the writers.
- The lock is refreshed once per epoch (`fsio.touch_lock`), so "stale" keeps
  meaning "crashed". Refreshing is best effort — it warns, never fails a run
  that already owns the lock — and no renewal thread exists.
- `gate` accepts `object` and keeps only real, finite numbers
  (`_usable_metrics`): an unreadable or `NaN` metric fails the threshold
  instead of raising or passing. `manifest.load_manifest` quarantines an entry
  that is not a map, for the same fail-closed reason.
- The contract tests gain a non-skipping presence guard: when `src/` is
  checked out, the files they read must exist.

## Consequences

- A feature the pipeline cannot enable is now an environment failure *before*
  the expensive work starts, and a log line never claims more than happened.
- One code per cause: storage that cannot be written reports `2`, exactly like
  storage that cannot be read; only "the input should exist and does not"
  remains a bug.
- Verification can no longer race a publish, and a long run can no longer be
  mistaken for a dead one.
- The other rejected items from 0005 stand unchanged: relative provenance
  paths (a rewrite would change skip semantics) and git-commit provenance
  (untestable branches in a possibly-unzipped tree).
