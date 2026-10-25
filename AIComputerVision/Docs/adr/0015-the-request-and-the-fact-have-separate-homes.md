# ADR 0015 — The request and the fact have separate homes

- Status: Accepted
- Date: 2026-10-24
- Supersedes: none
- Amends: ADR 0003 (what `--skip-unchanged` compares), ADR 0009 (the device
  request/fact split, extended to batch and image size), ADR 0013 (which
  fields are effective facts)
- Amended by: ADR 0016 (the device fact is canonical and the fact source is the
  framework's argument set, not `trainer.device`)

## Context

A second review of the package, one day after ADR 0014, tested the skip rule at
the real seam instead of through the fake backend. It found that the rule had
two different jobs and was using one set of fields for both:

* **audit** — what did this run actually do? (`trainer.args` after training,
  the checkpoint's saved arguments, the framework's own counters);
* **skip identity** — what would the *next* invocation ask for, before any work
  starts?

The manifest recorded the first and compared it as if it were the second. The
defect was proven against `ultralytics 8.4.41` + `torch 2.7.0+cu128` on an RTX
3050, on the real `Trainer`:

| `--device` | framework's `args.device` | recorded `device` | compared with | skip? |
|---|---|---|---|---|
| omitted (`auto`) | `"0"` | `"0"` | `"auto"` | **never** |
| `"0"` | `"0"` | `"0"` | `"0"` | yes |
| `"cuda:0"` | `"0"` | `"0"` | `"cuda:0"` | **never** |
| `"cpu"` | `"cpu"` | `"cpu"` | `"cpu"` | yes |

Only the two spellings that happen to survive Ultralytics' rewrite worked.
Every job in CI passed `--device cpu`, and no job ever ran a second identical
invocation, so the real-dependency check could not see it either.

The same shape appeared twice more:

* **`--batch -1`** (the documented automatic batch) was recorded as the
  resolved integer (`8`, `16`, …) and compared with the request `-1`; an
  automatic batch could never skip.
* **`--imgsz`** was recorded as the framework's rounded value. A related bug
  was worse than a stale skip: preflight accepted `641`, the framework trained
  at `672`, and the export contract compared the exported static shape with
  `641` — so the run failed at export, after the full training.

The export was not part of the skip identity at all. `export()` passed only
`format` and `opset`, relied on `model.args["imgsz"]` for the input size, and
the manifest recorded neither that image size nor the simplifier flags. Two
different graphs could therefore share every recorded training field.

Finally, `onnx_contract` imported `MAX_OPSET` from `preflight` — the policy
module of an earlier stage — only to read a runtime constant. That is an edge
in the wrong direction in the one-way graph of ADR 0001, and it existed because
the constant had no leaf home.

A mutation experiment made the gap measurable: changing `_record_entry` to
`device=effective("device", …)` in a copy of the tree left **all 508 tests
green**. The old decision was not pinned at all; the fake backend handed the
request back as the "fact", so no test could tell the two apart.

## Decision

- **An unsuffixed manifest field is a request; a `*_resolved` field is a
  fact.** `device`, `batch`, `imgsz`, `seed`, `opset` and `workers` are what
  the next invocation asks for. `device_resolved`, `batch_resolved` and
  `imgsz_resolved` are what the framework did with that request. The rule is
  uniform, so a new field has one obvious home.
- **`gate.already_current` compares the request under the request's name.**
  Replaying the same request reproduces the same resolution — it is a pure
  function of the request and the frozen backend identity — while recording the
  resolution in a compared field makes a run that the machine adjusted
  (`--batch 16` clamped to 8, `--imgsz 640` rounded by a future stride)
  impossible to reproduce. Two facts remain compared because a run cannot
  re-derive them: `device_resolved` (the hardware is not in `auto`, which is the
  entire point of ADR 0009) and the effective `epochs` budget (a resumed run
  takes it from the checkpoint). `imgsz_resolved`/`batch_resolved` are audit
  only. ADR 0016 later fixed the device comparison itself: both sides are
  read through `canonical_device()`, and the fact comes from the framework's
  argument set (which preserves a multi-GPU list) before `trainer.device`.
- **`device` is read from `TrainConfig.device_class`, and the backend keeps no
  device argument at all.** An earlier draft renamed the misleading
  `RunFacts.device` to `device_arg`; it is now removed. The framework's own
  argument (often `"0"`) is neither the request nor a fact about the artifact —
  the resolved device already records what ran — so keeping it anywhere only
  invited the same mistake again. `RunFacts.seed` is gone for the same reason:
  the framework does not resolve a seed, so the request is the fact.
- **`--batch -1` is never skippable.** The framework sizes the batch from the
  machine's memory *during* training; two runs that both recorded `-1` are not
  evidence that they used the same batch. "Cannot know what this run would
  build" is retrained, exactly like an unresolved device (ADR 0011).
- **The export request is part of the artifact identity.** `export_args`
  records `format`, `imgsz`, `opset` and the pinned simplification flags
  (`simplify`, `dynamic`, `half`, `int8`, `nms`). `UltralyticsBackend.export`
  is driven by exactly `export_args(cfg)`, the recorder stores that mapping,
  and the gate compares it. An entry without it is unprovable and retrained.
  The flags are passed explicitly rather than inherited from a package default
  that can move between releases.
- **The export image size is the request, not the run's rounded argument.**
  `export_args` passes `cfg.imgsz`; preflight rejects a size that is not a
  multiple of `IMGSZ_STRIDE` (32) with a message that names `--imgsz`, so the
  failure happens before training instead of at the contract check.
- **Shared constants live in the leaf.** `IMGSZ_STRIDE`, `MAX_OPSET` and
  `AUTO_BATCH` move to `cvtrain/config.py`. `preflight` and `onnx_contract`
  both read them from there, so the contract module no longer imports a policy
  module, and the gate/preflight/backend cannot spell the automatic-batch
  sentinel differently.
- **The real stack proves the skip verdict.** The nightly Linux job trains
  detection *without* `--device` (the runner has no GPU, so `auto` resolves to
  `cpu`), then runs the identical invocation with `--skip-unchanged`, greps
  `already matches this invocation; skipping`, and compares the artifact's
  SHA-256 before and after. A CI-contract tripwire rejects a skip step that
  names a device, so the check cannot drift back to the one spelling that
  happened to work.

## Consequences

- A default-device re-run now skips instead of retraining forever. The same
  holds for an explicit batch count and for every valid image size.
- The manifest answers two different questions without conflating them: "what
  did this run do?" (`*_resolved`, `epochs_completed`) and "what would a re-run
  ask for?" (the unsuffixed fields). The two values a re-run cannot derive —
  the resolved device and the effective budget — are compared as facts, not
  recorded under the request's name.
- The shipped `Models/manifest.json` predates this split. It has a four-field
  `backend_version`, a `device: "auto"`/`device_resolved: "cuda:0"` pair from
  ADR 0009, and no `export_args`; it also predates `epochs_completed`,
  `batch_resolved` and `imgsz_resolved`, which are audit-only. The next
  `--skip-unchanged` run retrains all three models because the exporter identity
  no longer matches and the export request is absent — the intended fail-closed
  cost, recorded as N17 in `Docs/SCORECARD.md` because the artifact can only be
  regenerated by a run with a real dataset.
- The suite grows from 508 to **516** tests and coverage stays at 100 % of
  statements and branches (1 934 / 494). The new tests are the composition
  tests the fake could not have written before: an identical default-device
  run skips, an automatic batch is retrained, the recorded request differs from
  the reported fact, a changed/absent `export_args` is retrained, and an
  off-stride image size fails in preflight.
- The mutation set grows from 107 to **116**, and `MIN_MUTANTS` is raised in
  the same change. The new mutants pin the three request fields, the
  automatic-batch refusal, the export-request comparison, the recorded export
  mapping, the exporter call, the off-stride rejection and the identity between
  the mapping compared and the mapping recorded. The decision that survived the
  review — recording a resolved value as the request — is now pinned: the
  equivalent mutant (`device=cfg.device_resolved`) turns the suite red.
- The nightly job gains a real-stack check of `--skip-unchanged`. Until now the
  only proof of the skip rule was the fake backend, which is precisely how this
  defect survived.
