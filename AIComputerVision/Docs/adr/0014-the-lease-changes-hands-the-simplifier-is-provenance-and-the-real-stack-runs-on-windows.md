# ADR 0014 — A resumed run claims the lease, the simplifier is provenance, and the real stack runs on Windows

- Status: Accepted
- Date: 2026-10-24
- Supersedes: none
- Amends: ADR 0008 (preflight proving "importable"), ADR 0010 (retention's
  "unknown activity" direction), ADR 0012 (the real-dependency job runs on the
  shipping platform too), ADR 0013 (the lease names the process writing *now*)

## Context

A second architecture review against the package's own claims found five
places where the code still did something other than what the documentation,
the scorecard and a green suite said. Each was reproduced before it was fixed;
none is hypothetical.

**A resumed run kept the dead process's lease.** `touch_run_marker()` refreshes
the mtime of `.cvtrain-alive` but deliberately leaves an existing owner line
alone — "refreshing a lease is not taking it". That is the right rule for an
epoch callback, but a `--resume` reuses the interrupted run's directory: its
marker still names the process that was killed. Until the first epoch ends
(hours, for a real model), `_run_owner_alive()` reports "dead"; a concurrent
`--prune-runs` sees a checkpoint older than the retention window and can delete
the *live* resumed run. The crash-recovery story ADR 0013 introduced was one
callback short of true.

**The exporter identity omitted the simplifier.** `exporter_identity()` recorded
`cvtrain`, `ultralytics`, `torch` and `onnx`, but not `onnxslim`. The pinned
Ultralytics (8.4.41) exports ONNX with `simplify=True` by default and runs
`onnxslim` over the graph; `requirements.txt` pins `onnxslim==0.1.91`. A missing
or upgraded simplifier emits different bytes under the same recorded identity,
so `--skip-unchanged` could skip an artifact that no longer reproduces — the
exact failure the identity exists to prevent.

**Preflight proved one package importable and assumed the rest.** `find_spec`
sees a distribution whose own dependency (or native DLL) is missing;
`check_importable()` imported `ultralytics` only and reported "framework
importable". An `onnxruntime`/`onnx`/`onnxslim` install broken at the DLL level
passed preflight, trained for hours and failed at export. On Windows — the
platform the product ships — that is the common failure shape.

**"Unknown activity" read as "oldest".** `_run_activity_mtime()` returned `0.0`
when it could not `stat()` a run directory, which sorts the run last and hands
it to the deletion rule. Retention is the one destructive operation in the
package; an age that cannot be proven is not evidence of abandonment, and the
safe direction is to keep.

**The real dependency stack was never exercised on Windows.** The gates already
run on both platforms (ADR 0012), but the nightly real-dependency job
(`ubuntu-latest`) was the only place the actual `torch`/`ultralytics`/`onnx`/
`onnxslim` wheels were installed, loaded, trained and exported. The product
ships Windows; a Linux-only verdict leaves the native install path unchecked.
The CI comments had also drifted (443 tests / 80 mutants / 11 fsio mutants
against 499 / 101 / 19), which is a C5 failure in the scorecard's own terms.

## Decision

- **`touch_run_marker` refreshes; `claim_run_marker` takes.** `claim_run_marker`
  writes the current PID/host unconditionally and is called at the top of
  `UltralyticsBackend.train()` when the plan resumes, before the framework is
  imported and the first epoch starts. The directory lock already refuses a
  second training writer, so an owner line naming another process is a crash
  leftover, not an owner to protect. The epoch callback keeps using
  `touch_run_marker`, so two processes still cannot silently swap a live lease.
- **The exporter identity includes `onnxslim`.** It is recorded as a fifth field;
  a move (or a missing package, recorded as `unknown`) invalidates
  `--skip-unchanged`. `onnxruntime` stays out on purpose: it validates the
  exported graph but does not produce it. `required_modules()` now also
  demands `onnxslim`, because without it the export is a different graph under
  the same version.
- **Preflight imports every module the backend declared.** `check_importable()`
  calls `import_module()` for each entry of `required_modules(cfg)` and turns
  any exception — `ImportError`, `OSError` from a DLL that will not load, or an
  error raised by the package's own import-time code — into a failing
  preflight verdict naming the package and the install command. No traceback,
  no hours of training before the discovery.
- **Unknown run activity is kept.** `_run_activity_mtime()` returns "now" when
  the directory cannot be inspected: the run sorts as active and survives the
  sweep. Retention remains a policy of the filesystem module (it shares the
  typed-error boundary and the lease reader); it is not split into a second
  module, because that would move 10+ mutation anchors and the Windows subset
  without changing a single decision.
- **The real stack runs on Windows.** `cvtrain-nightly.yml` gains a
  `windows-train-smoke` job: the same pinned CPU stack, a toy dataset, one
  detection epoch, `--verify-manifest`, and an assertion that
  `yolo11n.onnx` exists. The Linux job keeps the full det + seg + resume
  contract; the Windows job proves the native install → train → export path.
- **CI comments stop carrying measured counts.** The budget comments point at
  `Docs/SCORECARD.md` §1; the scorecard is the one place with dated numbers, so
  a new test or mutant cannot leave a stale number in a workflow comment.

## Consequences

- Coverage stays at 100 % of statements and branches, now **1 919 / 490**; the
  suite grows from 499 to **508 tests**. The mutation set grows from 101 to
  **107** (two fsio retention/lease mutants, four backend mutants) and
  `MIN_MUTANTS` is raised in the same change, so deleting one is a red gate.
- The three artifacts currently in `Models/` carry the four-field identity
  recorded before this decision. The next `--skip-unchanged` run will retrain
  them: the identity moved, and the fail-closed rule treats an unprovable
  artifact as stale. This is intended; retraining is the correct price of a
  provenance field that was missing.
- The Windows job adds roughly the install + one-epoch cost to the nightly and
  to pull requests touching the package. The mutation sweep gains six mutants
  (a few minutes on Linux; the Windows subset grows by two).
- `fsio.py` remains one module. Its docstring now states the real scope
  (primitives, locks, leases, retention) instead of describing only hashing and
  atomic writes — a cohesion smell is recorded honestly rather than paid for
  with a boundary that would invalidate the mutation anchors and the Windows
  subset.
