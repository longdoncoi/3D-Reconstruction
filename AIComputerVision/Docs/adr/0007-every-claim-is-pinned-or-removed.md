# ADR 0007 — Every claim is pinned or removed; the mutation set is the contract

- Status: Accepted
- Date: 2026-10-20
- Supersedes: none
- Applies to: `AIComputerVision/cvtrain`, `AIComputerVision/tests`,
  `AIComputerVision/scripts/mutation_check.py`

## Context

`AIAssistant` audits itself with a rule from its own ADR 0015: *a claim must
be enforced by a test or removed*. `AIComputerVision` had no equivalent audit,
and an independent review of the package found three claims that were not
enforced:

- **`exporter_identity()` recorded the wrong thing.** The function exists so
  that an export-relevant dependency moving invalidates a `--skip-unchanged`
  verdict (`cvtrain/backend.py:77`, ADR 0003). Its test
  (`tests/test_backend.py`) asserted only that the keys `torch` and `onnx`
  *exist* — not what they hold. Replacing `f"onnx={_package_version('onnx')},"`
  with a hardcoded `f"onnx=unknown",` kept the whole suite green. The rule the
  function exists for could therefore be switched off without a red test.
- **The mutation gate pinned 8 decisions out of ~57.** The script's own
  docstring calls it a check of "the pipeline's safety rules", but the set was
  concentrated in `gate.py`, `config.py`, `preflight.py`, `dataset.py`,
  `fsio.py` and `manifest.py`. The `onnx_contract` module — the guard that
  decides whether the Qt runtime can consume an export at all — had **no**
  mutant, although a probe of eight candidate mutants showed the suite kills
  all eight. The coverage was real; only the *record* of it was missing.
- **The shipped provenance contradicted ADR 0003.** ADR 0003 states that every
  entry records "the composite exporter identity
  (`cvtrain`/`ultralytics`/`torch`/`onnx`)". The checked-in
  `Models/manifest.json` records `backend_version: "8.4.41"` — the legacy bare
  version — for `yolo11n.onnx` and `yolo11n-seg.onnx`, and no provenance at all
  for `yolo11x-tracking.onnx`. The claim was true of the code and false of the
  artifact.

A fourth, smaller drift was in the same family: `AGENTS.md` and this package's
README pointed the contract at `src/core/AppConstants.h`, while the constants
live in `src/core/AIConstants.h` (`AppConstants.h` only includes it).

The first attempt at the expanded mutation set then exposed a defect in the
*gate itself*: the script mutated the working tree in place and ran the suite
there. The CLI tests call `main()`, and the CLI's mode dispatch is one of the
decisions being inverted — so the mutant that disabled `--verify-manifest` let
that command fall through to the training path, which started a **real YOLO
training** against the machine's dataset and checkpoints and wrote a run
directory into `runs/`. The suite did not fail; it hung until the timeout.
A gate that can train on the developer's machine is not a gate.

## Decision

- **The mutation set is the contract.** `scripts/mutation_check.py` names
  every load-bearing decision it pins, grouped by the module that owns it, and
  a decision that is *not* listed there is a decision nobody has pinned. The
  set is reviewed like the import table in `tests/test_architecture.py`: adding
  a guard means adding its mutant in the same commit.
- **A value is pinned by an assertion on that value.** Presence is not
  identity. `exporter_identity` is now pinned twice: the exact string for a
  known dependency set, and the fact that moving one dependency changes the
  identity.
- **The gate runs in a sandbox, never in the working tree.** The script copies
  the package, its tests, the README and the C++ sources into a temporary
  directory and mutates *that*. The sandbox has no `Dataset/`, no checkpoints
  and no `Models/`, so a mutant that reaches the training path fails in
  preflight instead of starting a real run. The working tree is never written
  to at all, and the byte-for-byte restore is kept as well so every mutant
  starts from the pristine source.
- **A mutant that does not fail fast is a failure of its own.** The suite has
  a per-mutant timeout; exceeding it is reported as `HUNG` — distinct from
  `KILLED` — and fails the gate. One hanging mutant must not abort the check
  of the remaining 56.
- **The tests must be hermetic too.** The CLI's manifest commands now assert
  that they never call the orchestrator, so a regression that lets a mode fall
  through is caught by the test itself instead of by the mutation script.
- **A claim about an artifact is a claim about that artifact.** ADR 0003's
  sentence is restated to describe what the shipped manifest actually holds,
  and the difference is recorded as measured debt in
  `Docs/SCORECARD.md` instead of being papered over.
- **Fail-closed is the safe direction for the debt itself.** A manifest entry
  whose `backend_version` is not the composite identity is *unprovable*, so
  `--skip-unchanged` retrains the model. The debt therefore costs training
  time, never correctness — and the honest fix is to retrain the models (or to
  record explicitly that the exporter identity of an artifact is unknown),
  never to write an identity that was not recorded at training time.
- **Measured numbers live in one place.** `Docs/SCORECARD.md` holds the gate
  table, the measurement date, and the debt table. The README links to it
  instead of restating numbers that can rot.

## Consequences

- The two claims that were not enforced are now enforced: a hardcoded
  `onnx=unknown` is a red test, and the export contract has eight mutants of
  its own.
- The mutation gate grew from 8 to 57 decisions, so the CI job that runs it
  needs a larger timeout — the cost is paid in CI minutes, not in ambiguity.
- The gate can no longer touch the developer's data, checkpoints or published
  models, and it reports a hanging mutant instead of dying with a traceback.
- The package now has the same shape as the rest of the repository: a
  measurable scorecard, ADRs for the decisions, and an audit rule that says a
  claim must be enforced or removed.
- The provenance debt of the shipped models is visible, dated and bounded
  instead of being implied away by a document that says something the files do
  not say.
