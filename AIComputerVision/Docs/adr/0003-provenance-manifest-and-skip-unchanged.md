# ADR 0003 — Provenance manifest and the fail-closed `--skip-unchanged` rule

- Status: Accepted
- Date: 2026-10-12
- Supersedes: none

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
- the digest of `data.yaml` — renamed classes / moved splits behind an
  unchanged path are a different experiment, while the images themselves are
  deliberately not hashed (O(dataset) on every re-run);
- epochs, imgsz, seed, opset, batch, workers, the device class, and the backend
  version.

A model is skipped only when *everything* matches; any missing or unknown
provenance field means "not vouchable" and the model is retrained (fail closed).
`--skip-unchanged` never bypasses the quality gate — a kept model is judged
against the metrics recorded when it was trained.

New optional entry fields do not bump the schema (`total=False`); a document
written by a newer build is read as-is and never down-stamped, so an older
fixture cannot erase version information a future reader needs. A corrupt or
structurally invalid manifest is **quarantined** (`manifest.json.corrupt-<ts>`)
so the next write cannot silently destroy the only record of what was there.

## Consequences

- `--skip-unchanged` is safe to default to: a false "unchanged" verdict is the
  expensive direction and is the fallback for every unknown.
- `--verify-manifest` makes truncation or accidental replacement detectable by
  recomputing digests; a training run re-verifies exactly the models it just
  published, so a pre-existing stray file cannot fail an otherwise clean run.