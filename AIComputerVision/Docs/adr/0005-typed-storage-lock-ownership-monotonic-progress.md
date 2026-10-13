# ADR 0005 — Typed storage errors, lock ownership, monotonic progress

- Status: Accepted
- Date: 2026-10-13
- Supersedes: none (extends 0004)

## Context

Three reviews of 0004's mechanisms showed gaps that are small individually
but each break a contract the suite claims to hold:

- Hashing or measuring a file (`sha256_file`, `stat`) raised bare `OSError`
  (exit `1` + traceback) while the identical condition in `--verify-manifest`
  is exit `2`. Same cause, two codes.
- The lock release deleted unconditionally: a steal that landed mid-run
  (past `stale_after`) would be unlocked from under its new owner on exit.
- The progress bar could move backwards on an out-of-order call, a state the
  Qt dock has no rendering for.
- `MAX_OPSET` and `ONNXRUNTIME_VERSION` (CMakeLists) were linked only by a code
  comment: a runtime bump could silently invalidate the export ceiling.
- `mypy` ran without `disallow_untyped_calls` / `disallow_subclassing_any` /
  `warn_unreachable`, and `ignore_missing_imports` was global rather than
  scoped to the lazily-imported stack.

## Decision

- New `StorageError(CvTrainError)`, exit `2`: an unreadable/unmeasurable file
  during provenance, verification or publishing-input is "cannot vouch", not
  a bug. The atomic publish path itself keeps `OSError`: a vanished framework
  output that was supposed to exist *is* a bug.
- The lock file carries `token=<pid>-<ns>`; release deletes only while the
  token matches, warns and leaves a foreign lock otherwise. Run names carry
  the PID too, so two processes can no longer collide under `exist_ok=False`.
- `_emit` holds out-of-order percentages with a warning instead of publishing
  a regression; the diagnostic line carries no `pct=` token, or the dock's
  own regex would rewind the bar. A 1-based epoch counter degrades to a full
  bar (`5/5`), never `6/5`.
- `tests/test_contract.py` pins `ONNXRUNTIME_VERSION -> opset` as an explicit
  tripwire map: a runtime bump fails until a reviewer re-confirms the ceiling.
- `mypy` enables the three strict flags; `ignore_missing_imports` is scoped
  to `ultralytics.*` / `yaml`, the two lazily-imported modules — the
  import-light design (`--help` without the ML stack) is what the override
  protects, documented at the override site.
- `last_run` records `python` and `min_metrics` (informational; the skip rule
  does not compare them) and `updated_at` refreshes on every provenance
  write; directory entries get a best-effort `fsync` after each rename.

## Consequences

- One taxonomy for storage: expected (exit `2`, one line) vs bug (exit `1`,
  traceback), decided at the `fsio` boundary rather than per call site.
- The lock can no longer be released by the wrong run; the bar can no longer
  move backwards; the opset ceiling cannot rot silently past a runtime bump.
- Deliberately *not* done: heartbeated locks (a 24 h stale window already
  exceeds any real run; a renewal thread is complexity without a failure to
  fix), relative provenance paths (fail-closed absolute paths are safe; a
  rewrite would change skip semantics), git-commit provenance (best-effort
  subprocess in a possibly-unzipped tree, untestable branches).
