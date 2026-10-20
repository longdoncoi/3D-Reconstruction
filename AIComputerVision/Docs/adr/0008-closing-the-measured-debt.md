# ADR 0008 — Closing the measured debt: digest-based provenance, resumable runs, observable fingerprinting, proven importability

- Status: Accepted
- Date: 2026-10-21
- Supersedes: partially 0003 (its path-as-provenance comparison) and 0005 (its "relative provenance paths" non-goal, which was about *storage*, not about *comparison*)

## Context

`Docs/SCORECARD.md` records five debts. They are not cosmetic: each one is a
place where the pipeline's answer is "we cannot know", and each one costs
something real — hours of retraining, a frozen progress bar, a report that
says "ready" about an environment that is not.

- **N1 — the shipped models' provenance is in the old format.**
  `Models/manifest.json` carries `backend_version: "8.4.41"` — the bare
  Ultralytics version — while the current code writes the composite identity
  `cvtrain=…;ultralytics=…;torch=…;onnx=…`. An entry in the old shape is
  *unprovable* against the current stack, so `--skip-unchanged` retrains the
  shipped models: the idempotence win does not work out of the box.
- **N2 — provenance is bound to one machine.** The manifest records absolute
  paths (`dataset`, `run_dir`), and the skip rule compared the *path string*.
  The same dataset copied to another directory was a different experiment,
  which is not what "same dataset" means.
- **N3 — an interrupted training run is lost.** Every run creates a fresh
  timestamped directory; a crash at epoch 4 of 5 means four epochs of GPU
  time are gone, even though the framework wrote `weights/last.pt` after
  every epoch.
- **N4 — the fingerprint is invisible.** `--skip-unchanged` proves the data by
  hashing every byte of it. On a large dataset that is minutes during which
  the Qt dock's bar stands still at 2 % — indistinguishable from a hang.
- **N5 — "installed" is not "usable".** The preflight check used
  `find_spec`, which proves a distribution is on disk but not that it can be
  imported: a wheel whose own dependency is missing passes the check and
  fails later. The report said "environment ready" about a stack that could
  not be loaded.

## Decision

- **N1 — the shipped models are retrained, and the rule about provenance
  stands.** The models are retrained by this build so the manifest carries
  the composite identity that the code actually writes. The rule from ADR
  0007 is unchanged and is the reason this is a retrain rather than an edit:
  an identity is written by the run that produced the artifact, never
  reconstructed afterwards. A model that was *adopted* from disk keeps
  saying so.
- **N2 — the dataset is compared by its digests, not by its path.** The path
  stays in the manifest as provenance (it is a fact about the run), but the
  skip rule no longer *compares* it: `dataset_sha256` and
  `dataset_content_sha256` are what prove "the same dataset". A copy of the
  same dataset at another absolute path is the same experiment. This is
  narrower than ADR 0005's rejected proposal: that one was about *storing*
  relative paths (which changes what the document means); this is about
  *comparing* by content, which changes nothing about the document.
- **N3 — `--resume` continues an interrupted run.** The adapter picks the
  newest `weights/last.pt` of the model under `runs/<model>/` and trains
  with `resume=True`, so the framework restores the interrupted run's own
  arguments and epoch counter. The flag is opt-in: a fresh run is the
  reproducible one. With nothing to resume from, the pipeline says so and
  starts a fresh run rather than failing.
- **N4 — the fingerprint reports progress, and the protocol has a band for
  it.** `dataset_content_fingerprint` takes an `on_file` callback, the
  trainer hands it the reporter, and the progress protocol grows a `dataset`
  phase (2–8 %) between `preflight` and the models. The Qt dock reads
  `pct=` only, so the wire protocol is unchanged; the `stage=` field is
  informational.
- **N5 — the backend proves the stack can be loaded, inside the preflight
  report.** `TrainingBackend.check_importable` returns a verdict, and the
  trainer adds it to the same report as the dependency checks: a broken
  install is a *fatal preflight failure* (exit 2), not a surprise later.
  The import happens in the adapter — the module that owns the heavy imports
  (ADR 0001) — so the package stays import-light and the architecture map is
  unchanged.

## Consequences

- `--skip-unchanged` works for the shipped models again, and works across
  machines: the verdict depends on what the data *is*, not on where it lives.
- An interrupted training run costs one command, not one run.
- The bar moves while the dataset is hashed, and says what it is doing.
- The preflight report cannot claim an environment is ready when the
  framework cannot be imported — and the claim is still made in the one
  module allowed to make it.
- Two of the changes touch things the Qt side reads: the progress bands
  moved (the dock parses `pct=`, so it is unaffected) and a new CLI flag
  appeared (the dock does not pass it, so it is unaffected too). Both are
  pinned by the cross-language contract tests.
- The fingerprint costs what it always cost — every byte is read — and now
  says so while it does.
