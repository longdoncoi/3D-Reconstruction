# ADR 0003 — Provenance manifest and the fail-closed `--skip-unchanged` rule

- Status: Accepted
- Date: 2026-10-12
- Supersedes: none
- Amended by: ADR 0015 (compared values are the **requests**; the framework's
  resolutions are recorded separately and never under the request's name — the
  resolved device is still compared as its own fact)

## Context

Retraining is expensive (hours on a GPU), so `--skip-unchanged` is the product's
key idempotence win — but trust is only as good as the recorded provenance. A
file name alone cannot prove that `yolo11n.onnx` was made by *this* dataset, *this*
checkpoint and *these* hyper-parameters. The desktop app also needs to know
whether the checked-in models are fresh, without the user re-training to find
out.

## Decision

Every export is recorded in `Models/manifest.json` with:

- SHA-256 of the artifact and of the base checkpoint actually trained from;
- the digest of `data.yaml` *and* a content fingerprint of every file the
  referenced `train`/`val` splits reach — directories, globs, single files,
  `.txt` listings and list-valued splits — captured **before** `backend.train`
  starts (amended 2026-10-20; the original revision deliberately hashed only
  the YAML and was blind to same-path content/glob changes);
- epochs, imgsz, seed, opset, batch, workers, the device class, the exporter
  options, and the composite exporter identity
  (`cvtrain`/`ultralytics`/`torch`/`onnx`/`onnxslim`), because any of those
  dependencies can change the emitted graph.

> **Amended 2026-10-24 (ADR 0015): those fields are the *request*, and a
> resolution is never recorded or compared under the request's name.** The
> framework may rewrite the device argument (`"auto"` → `"0"`), resolve
> `--batch -1` to a machine-sized integer, or round `--imgsz`; those results are
> recorded as `device_resolved`, `batch_resolved` and `imgsz_resolved` for audit,
> never under the compared name. `device_resolved` is itself compared as a fact
> because hardware is not derivable from `auto`, and the effective `epochs`
> budget is compared because a resumed run takes it from the checkpoint. The
> export request (`format`, `imgsz`, `opset`, `simplify`, `dynamic`, `half`,
> `int8`, `nms`) is recorded in `export_args` and is part of the artifact
> identity, and an automatic batch is never skippable because its answer is not
> knowable before training. Recording a resolved value as the request made an
> identical re-run retrain forever (N15 in `Docs/SCORECARD.md`).

> **Amended 2026-10-20 (ADR 0007): this describes what a run *writes*, not
> what every checked-in entry holds.** The models shipped in `Models/` were
> trained before the composite identity existed: their entries carry the legacy
> bare `backend_version` (`"8.4.41"`), and `yolo11x-tracking.onnx` carries no
> provenance at all. Such an entry is **unprovable**, so `--skip-unchanged`
> retrains those models — the fail-closed direction (see ADR 0003's
> consequence below and the debt table in `Docs/SCORECARD.md`). The honest fix
> is to retrain them, or to record explicitly that the exporter identity of an
> artifact is unknown — never to write an identity that was not recorded at
> training time.

A model is skipped only when *everything* matches; any missing or unknown
provenance field means "not vouchable" and the model is retrained (fail closed).
In particular an entry without `dataset_content_sha256` is retrained: the
cheaper `data.yaml` digest alone does not prove which images produced the
artifact.
`--skip-unchanged` never bypasses the quality gate — a kept model is judged
against the metrics recorded when it was trained.

New optional entry fields do not bump the schema (`total=False`); a document
written by a newer build is read as-is and never down-stamped, so an older
fixture cannot erase version information a future reader needs. A corrupt or
structurally invalid manifest is **quarantined** (`manifest.json.corrupt-<ts>`)
so the next write cannot silently destroy the only record of what was there —
that recovery path belongs to the *recovering* reader a training run uses;
`--verify-manifest` reads strictly instead and exits `2` without renaming
anything (ADR 0006).

## Consequences

- `--skip-unchanged` is safe to default to: a false "unchanged" verdict is the
  expensive direction and is the fallback for every unknown.
- `--verify-manifest` makes truncation or accidental replacement detectable by
  recomputing digests; a training run re-verifies exactly the models it just
  published, so a pre-existing stray file cannot fail an otherwise clean run.