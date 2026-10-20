# ADR 0009 — The device is resolved before training: `auto` is a request, the resolution is the fact

- Status: Accepted
- Date: 2026-10-21
- Supersedes: none

## Context

The manifest recorded the *requested* device. With `--device auto` — the
default, and what the Qt dialog uses — that recorded the string `auto`, which
is a request, not a fact: "let the framework choose" answers differently on a
machine with a GPU and on one without, and differently again if the GPU is
busy. Two artifacts trained on different hardware therefore carried the same
provenance, and the skip rule could not tell them apart.

The obvious fix — record the device the framework actually used — does not
work: the framework only reports it *after* training, and by then the decision
to skip has long been made. The skip verdict is made before any weight is
touched, so the fact it needs has to exist before that.

The way out is to stop delegating the choice. The pipeline already owns the
run; it can resolve the device itself, before training, and hand the answer to
the framework.

## Decision

- **`TrainingBackend.resolve_device(cfg)` answers with a concrete device.**
  The policy is explicit and simple: an explicit request is its own answer
  (`0,1` stays `0,1`), and `auto` becomes the first CUDA device when the
  runtime has one, `cpu` otherwise. The policy lives in the adapter, the one
  module that owns the heavy imports (ADR 0001), so `torch` never reaches the
  rest of the package.
- **No silent fallback.** "The GPU looked busy, so CPU it is" would make the
  provenance a lie: the run would record a device it did not use. There is no
  such fallback — a failed GPU run fails, and the operator picks another
  device explicitly.
- **The orchestrator resolves once, before anything is trained**, and the
  framework is *told* which device to use instead of choosing for itself.
  That removes the framework's own variation from the picture: the recorded
  device is the device that ran.
- **The provenance records both.** `device` keeps its meaning — the request —
  and the new optional field `device_resolved` carries the fact. Adding an
  optional entry field does not bump the schema (ADR 0003), and a document
  written by an older build is still read without being down-stamped.
- **The skip rule compares the resolution as well.** The request alone cannot
  distinguish a CPU run from a GPU run; the resolution can. An entry that
  carries no resolution is *unprovable* about its device and is retrained
  rather than trusted — the same fail-closed rule the rest of the provenance
  follows.

## Consequences

- The provenance is now reproducible across machines: a run on a CPU-only
  machine resolves to `cpu`, which does not match an artifact trained on
  `cuda:0`, and the model is retrained instead of silently reused.
- The recorded device is the device that ran, so an auditor reading
  `device_resolved` is reading a fact.
- The framework no longer chooses: `auto` is resolved by the pipeline and
  forwarded, which makes a run reproducible in a second sense — the same
  command on the same machine always trains on the same device.
- The shipped artifacts had to be regenerated once: an entry written before
  this decision carries no resolution, so it is unprovable and is retrained.
  That is the fail-closed direction, and it is visible in the manifest rather
  than hidden.
- `device` alone is no longer enough to compare two runs; the pair
  (`device`, `device_resolved`) is the unit of provenance. The gate compares
  both, and the manifest records both.
