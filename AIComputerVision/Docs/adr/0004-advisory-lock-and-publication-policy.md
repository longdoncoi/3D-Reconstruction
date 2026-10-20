# ADR 0004 — Advisory lock, atomic publish, quality gate

- Status: Accepted
- Date: 2026-10-12
- Supersedes: none

## Context

Two runs sharing `Models/` would each read `manifest.json`, train, and then
write — the last writer silently discards the other's provenance. The pipeline
"assumed" a single process, but an assumption a crash or a double-click can
violate is not a guarantee. Publishing also has to be safe to interrupt: a torn
ONNX on disk is worse than no ONNX.

## Decision

- A training run — and `--write-manifest` — takes an **advisory lock**
  (`Models/.cvtrain.lock`, created with `O_CREAT | O_EXCL`, the one atomic
  create-if-absent primitive every supported platform shares) for the whole
  read-train-write window. A second run fails fast with `ConcurrentRunError`
  (exit `2`) instead of racing. The lock is released with the process; a lock
  older than `LOCK_STALE_SECONDS` (24 h) is treated as abandoned and replaced
  with a warning, so a crash cannot wedge the next run forever. Stealing is
  serialised by a `.cvtrain.lock.steal` sidecar claim (`O_EXCL`) and rechecks
  the lock age under that claim (amended 2026-10-20): the previous
  check-then-delete path let two processes both observe the same stale lock
  and the slower one delete the lock the faster one had just created. A claim
  abandoned by a crashed stealer ages out after `STEAL_STALE_SECONDS` (60 s).
- Publishing replaces the previous ONNX in place and keeps one generation at
  `<name>.onnx.bak`. The backup is a recovery artifact only: it has no manifest
  entry, `--verify-manifest` ignores it, and `--skip-unchanged` never trusts it.
- Every write goes through `fsio`'s atomic primitives (write-copy + fsync +
  replace), so even an outside writer cannot leave a torn file.
- `--min-metric` is a publication boundary, not a logging preference: a model
  that fails the bar never reaches `Models/`, the previously published artifact
  stays intact, and the run exits `3` with both numbers in the message.

## Consequences

- The single-writer assumption is enforced rather than stated; the failure mode
  is a clear refusal, not silent provenance loss.
- A crash after export leaves at most one stale generation next to the good
  file, never a torn artifact.
- Quality is *published or not* — the gate runs between training and publishing,
  so "clean run" always implies "every artifact is vouchable and good enough".