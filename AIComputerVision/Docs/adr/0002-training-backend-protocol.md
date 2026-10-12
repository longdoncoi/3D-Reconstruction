# ADR 0002 — TrainingBackend protocol: the framework is a plugged-in adapter

- Status: Accepted
- Date: 2026-10-12
- Supersedes: none

## Context

The pipeline's only job is to turn a dataset into validated ONNX, but doing it
"directly" ties every module to Ultralytics — its import cost, its API churn and
its hardware requirements. The Qt app drives the pipeline from a GUI one screen
away from users, so a broken or missing ML stack must not break argument
parsing, environment checks or the help text.

## Decision

All framework access goes through the `TrainingBackend` protocol
(`prepare` / `train` / `export`), produced by `cli` and handed to `trainer.run`
as a required parameter.

- `UltralyticsBackend` is a thin adapter: it imports the ML stack only when
  training actually starts, and maps `TrainConfig` to the framework's keyword
  arguments in one controlled place.
- The trained model is `TrainedModel[H]`, generic over the opaque framework
  handle `H`. The pipeline never inspects it; it only hands it back to the same
  backend's `export`.
- `preflight` receives its dependency list from the backend
  (`required_modules`) instead of hard-coding a framework.
- Tests drive the whole pipeline with a fake backend, so orchestration logic is
  verified offline with 100% coverage; API drift inside `ultralytics` is caught
  by `cvtrain-nightly.yml`, which runs the real stack on a tiny dataset.

## Consequences

- `--help`, `--check`, `--dry-run`, linting, mypy and the unit suite run with
  no ML dependencies installed.
- A framework upgrade changes the adapter and the nightly contract test, not
  the orchestrator.
- A second framework can be added without touching `trainer` — only the
  composition root changes.