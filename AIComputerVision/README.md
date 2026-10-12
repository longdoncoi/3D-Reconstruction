# AIComputerVision

Training and export pipeline for the YOLO models consumed by the Qt
application's AI Processor plugin. It turns a labelled dataset into the ONNX
files in [`Models/`](Models/) and records their provenance in
[`Models/manifest.json`](Models/manifest.json).

## Layout

```
AIComputerVision/
├─ TrainModel.py          # entry point used by the desktop app
├─ cvtrain/               # the actual implementation
│  ├─ __init__.py         # package marker + __version__ (import-light)
│  ├─ cli.py              # composition root: argparse, dispatch, exit codes
│  ├─ config.py           # paths, defaults, model contract
│  ├─ checks.py           # Check / PreflightReport validation vocabulary
│  ├─ dataset.py          # data.yaml + the train/val paths it references
│  ├─ errors.py           # ExitCode enum + expected-failure hierarchy
│  ├─ fsio.py             # sha256 + crash-safe write/copy primitives
│  ├─ preflight.py        # fail-fast environment/hyper-parameter checks
│  ├─ gate.py             # publication policy: --skip-unchanged + quality gate
│  ├─ backend.py          # TrainingBackend protocol + ultralytics adapter
│  ├─ trainer.py          # orchestration: train -> gate -> export -> publish
│  ├─ manifest.py         # manifest schema (TypedDict), read/write/verify
│  ├─ progress.py         # [PROGRESS] protocol parsed by the Qt dock
│  └─ logging_setup.py    # stdout logging format
├─ Models/                # exported *.onnx (Git LFS) + manifest.json
├─ runs/                  # Ultralytics training logs (git-ignored)
├─ requirements.txt       # pinned runtime deps
└─ tests/                 # offline unit tests (no model weights required)
```

## Architecture

Dependencies point one way, from orchestration down to primitives:

```
TrainModel.py → cli ──► trainer ──┬─► preflight ─┬─► dataset
   (composition root,              ├─► manifest   └─► checks
    injects the backend)           ├─► gate ──────► fsio / config
                                   └─► backend ────► fsio / config
```

This diagram is a contract, not a suggestion: `tests/test_architecture.py`
parses `cvtrain/*.py` with `ast` and fails the build when an import lands
outside the map it encodes — a module importing a neighbour it should not
know about, a new file that has not declared its own dependencies in that map,
the heavy stack (`ultralytics`, `yaml`) reached from anywhere but its single
owner or imported at module level instead of lazily, or the concrete
`UltralyticsBackend` named outside `cli.py`. The layering cannot rot silently
while the suite stays green.

`cli.main` is the package's **composition root**: it is the only module that
names a concrete `TrainingBackend`, and it hands that instance to
`trainer.run(cfg, backend=…)` — where the parameter is *required*. The
orchestrator therefore has no default to invent and no adapter to import.

`trainer` owns *what happens* and never imports `ultralytics`. All framework
access goes through the `TrainingBackend` protocol (`prepare` / `train` /
`export`), which production binds to `UltralyticsBackend` — a thin adapter that
imports the ML stack only when training actually starts. The trained model is
`TrainedModel[H]`, generic over the opaque framework handle `H`: the pipeline
only ever hands it back to the same backend's `export`, never inspects it.
Three consequences:

* `--help`, `--check`, `--dry-run`, linting, type checking and the unit suite
  run with no ML dependencies installed;
* the whole pipeline is driven by a fake backend in `tests/test_trainer.py`,
  so preflight/dry-run/prompt/quality-gate/manifest-durability/progress are all
  verified offline;
* `preflight` receives its dependency list from the backend
  (`required_modules`) instead of hard-coding a framework.

Validation is split three ways so no module has two reasons to change:
`checks` owns the `Check`/`PreflightReport` vocabulary every contributor
shares, `dataset` owns the `data.yaml` document, and `preflight` owns the
environment and hyper-parameter rules and aggregates the rest. `progress` is
separate from `logging_setup` for the same reason: it is a wire contract with
another program, not a logging preference.

The two *verdicts* a run obeys are policy, not flow, so they live in
`gate.py` rather than in the orchestrator: `already_current` decides whether
the published ONNX already is what this invocation would build (the
`--skip-unchanged` rule, below), and `enforce_quality_gate` decides whether an
artifact is good enough to keep (the [Quality gate](#quality-gate)). `trainer`
asks for a verdict and obeys it; `manifest` records the provenance those
verdicts compare against.

## Setup

```powershell
pip install -r AIComputerVision/requirements.txt
```

The desktop app launches `python` from `PATH`. To use a specific
interpreter/virtual environment, set the `RECONSTRUCTION_PYTHON` environment
variable to its full path before starting the app.

Datasets live in `Dataset/data.yaml` at the repository root (git-ignored). The
file must define `train`, `val` and `names`; `--check` verifies this before any
training starts — including that the directories the `train`/`val` keys point
at exist on disk (relative paths resolve against the folder holding the YAML,
or against an explicit top-level `path:` key, mirroring Ultralytics).

The base checkpoints (`yolo11n.pt`, `yolo11n-seg.pt`, `yolo11x.pt`) live in
`AIComputerVision/` by default and are downloaded by Ultralytics on first use.
`--weights-dir` moves them, the same way `--models-dir` and `--runs-dir` move
the outputs — preflight, the trainer and the provenance digest all resolve a
checkpoint through `TrainConfig.weights_path`, so those three can never
disagree. Their SHA-256 is recorded with each manifest entry and re-checked
before `--skip-unchanged` trusts an artifact, because a replaced checkpoint is
a different model regardless of its file name.

## Usage

```powershell
# Train the default models (detection + segmentation)
python AIComputerVision/TrainModel.py --yes

# Train only the tracking model
python AIComputerVision/TrainModel.py --track --yes

# Validate the environment and print the plan without training
python AIComputerVision/TrainModel.py --check
python AIComputerVision/TrainModel.py --dry-run

# Idempotent re-run: train only the models whose ONNX no longer matches
python AIComputerVision/TrainModel.py --skip-unchanged --yes

# Override hyper-parameters
python AIComputerVision/TrainModel.py --det --epochs 50 --batch 8 --imgsz 640 --seed 42

# Publish only models that clear a quality bar (repeatable; see "Quality gate")
python AIComputerVision/TrainModel.py --yes --min-metric 'metrics/mAP50(B)=0.5' --min-metric 'metrics/mAP50-95(B)=0.3'

# Provenance tooling
python AIComputerVision/TrainModel.py --write-manifest
python AIComputerVision/TrainModel.py --verify-manifest
```

`--skip-unchanged` is about *whether to train*, not about permission to
overwrite: an interactive run still asks before touching a model that exists but
no longer matches (the desktop app passes `--yes` and never prompts). See
[Manifest](#manifest) for the exact rule a model has to satisfy to be skipped.

`--check`, `--dry-run`, `--write-manifest` and `--verify-manifest` are the four
*modes* and are mutually exclusive (a usage error, exit `2`, if two are given):
combining them used to give one of them silent precedence — notably
`--write-manifest --verify-manifest`, which would rewrite the manifest and then
"verify" the file it had just written. With none of them, the pipeline trains.

Every hyper-parameter is validated by preflight — `--epochs`, `--imgsz`,
`--batch`, `--workers`, `--opset`, `--device` and every `--min-metric`
threshold — so a value that cannot produce a usable model is rejected by name
(`[FAIL] param.opset: 99 (must be 11..21; pass --opset N)`) instead of failing
inside the framework much later. The opset ceiling is not arbitrary: ONNX
Runtime 1.20.1, the build `CMakeLists.txt` pins for the Qt app, implements
opset 21, and an artifact stamped higher would fail only when someone tried to
load it.

Exit codes are declared once in `cvtrain/errors.py` (`ExitCode`) and never
written as bare integers elsewhere: `0` success, `1` unexpected failure, `2`
preflight/manifest failure, `3` quality-gate failure, `130` interrupted. Expected
failures (`CvTrainError`) are logged as a single actionable line; anything else
keeps the full traceback. The Qt dock reports the number verbatim and only tests
it against zero, so adding a code is a compatible change.

## Quality gate

`--min-metric NAME=VALUE` (repeatable) refuses to publish a model whose reported
metric is below `VALUE`. The check runs *between* training and publishing: a
model that fails the bar never reaches `Models/`, the previously published
artifact stays where it was (still intact, still verified), and the run exits
`3` with both numbers in the message. Metric names are the ones the framework
reports; asking for a name nobody reports is a failure, and the error lists what
*was* reported so an easy typo is visible immediately.

The gate applies to a model `--skip-unchanged` decided to keep as well, against
the metrics recorded when that model was trained — skipping is never a way
around the bar. Thresholds are for higher-is-better metrics in `0..1`
(precision, recall, mAP, fitness), not for losses.

## Model contract (C++ ↔ Python)

The exported file names are a contract with `src/app/AppConstants.h`. A change
on either side must be mirrored; `tests/test_contract.py` fails CI if they
drift.

| Selection | Base weights   | Exported file           | Role         |
|-----------|----------------|-------------------------|--------------|
| `--det`   | `yolo11n.pt`   | `yolo11n.onnx`          | detection    |
| `--seg`   | `yolo11n-seg.pt` | `yolo11n-seg.onnx`    | segmentation |
| `--track` | `yolo11x.pt`   | `yolo11x-tracking.onnx` | tracking     |

When no selection flag is passed, `--det` and `--seg` are trained (legacy
behaviour). Missing base weights are downloaded by Ultralytics on first use.

## Progress protocol

`TrainModel.py` prints one line per milestone so the Qt dock can drive its
progress bar:

```
[PROGRESS] pct=42 stage=train key=seg epoch=3/5
[PROGRESS] pct=96 stage=export key=seg
[PROGRESS] pct=96 stage=skip key=track
[PROGRESS] pct=100 stage=manifest
[PROGRESS] pct=100 stage=done
```

Every run ends in a terminal marker — `stage=done` on success, or
`stage=error` (holding the last published percentage) on failure — so the bar
never stalls waiting for a line that will not arrive. A model skipped by
`--skip-unchanged` publishes `stage=skip` at the percentage a completed export
would occupy, so a run where every model is skipped still climbs monotonically
to the manifest band. The Qt side
(`AITrainDockWidget::onTrainProcessOutput`) only parses `pct=`; the other
fields are informational.

## Manifest

Each exported model is recorded in `Models/manifest.json` with its SHA-256,
size, dataset path *and the digest of `data.yaml`*, every hyper-parameter, the
compute device, the digest of the base checkpoint it was actually trained from,
the version of the backend that produced it, and (best-effort) metrics.
`--verify-manifest` recomputes the digests, which makes truncation or
accidental replacement detectable. Verification is bidirectional: an entry
whose file went missing is reported, and so is an ONNX file sitting in
`Models/` without an entry.

A training run re-verifies the manifest it just wrote and fails with exit code
`2` when that check fails — the same code `--verify-manifest` returns, raised
through the same `ManifestIntegrityError` — so an artifact the pipeline cannot
vouch for is never reported as a clean run.

The document carries a `schema_version`. On load, an older document (schema
`0`: written before the field existed) is migrated forward through
`cvtrain/manifest.py::_MIGRATIONS`; a document written by a **newer** `cvtrain`
is read as-is and explicitly *not* down-stamped, so an older build cannot erase
the version a future reader needs for its own migration. New *optional* entry
fields do not bump the schema: the shape is already `total=False`, and a
consumer that needs the field (the skip rule) treats its absence as "provenance
we do not have" and fails closed by retraining.

Publishing an export replaces the previous ONNX in place and keeps one
generation of it at `<name>.onnx.bak`. That file is a recovery artifact only:
it has no manifest entry, `--verify-manifest` ignores it, and the Qt app never
loads it. `--skip-unchanged` also refuses to trust it — a model is skipped only
when *all* of the following hold for the published file, so an artifact adopted
by `--write-manifest` (which records no provenance) is retrained:

* its digest matches what the manifest recorded;
* `epochs`, `imgsz`, `seed`, `opset`, `batch`, `workers` and the dataset path
  match this invocation exactly;
* the recorded backend version is the one that would produce it now (a framework
  upgrade changes the export);
* the device class matches (CPU and CUDA runs of the same parameters are
  different artifacts behind the same file name; "unspecified" and `auto` are
  the same device);
* `data.yaml` still hashes the same — the path alone is not provenance, since
  renamed classes or moved splits behind an unchanged name are a different
  experiment. The digest covers the *definition*, not the images: hashing a
  dataset is O(dataset) and would make every re-run pay for gigabytes the
  operator did not touch. Re-shuffled images under an unchanged `data.yaml` are
  the one blind spot `--skip-unchanged` has, and it is deliberate;
* the base checkpoint on disk still has the digest recorded when the artifact
  was trained — a replaced or re-downloaded `yolo11n.pt` is a different starting
  point, and a missing one cannot be verified at all.

The digests are compared last: they are the expensive part, so a mismatch in the
cheap hyper-parameters never pays for hashing a checkpoint.

The pipeline assumes a single training process at a time (the desktop app runs
one). Every write is atomic, so a concurrent run cannot leave a torn file, but
two runs sharing `Models/` would race on `manifest.json` and the last writer
wins.

## Tests

```powershell
cd AIComputerVision
ruff check .
python -m mypy
coverage run -m unittest discover -s tests -v
coverage report
```

`cvtrain` is gated at **100% statement and branch coverage** — every line is
exercised, so new code lands together with its test.

* `tests/test_trainer.py` drives the whole pipeline against a fake backend: no
  GPU, no dataset and no ML packages are needed — including the quality gate
  and every factor `--skip-unchanged` compares.
* `tests/test_architecture.py` enforces the Architecture section above: the
  import map, lazy heavy dependencies, the single owner of the adapter, module
  docstrings, and the README's own layout/flag inventory. A new module or a
  new import is a change to a table, reviewed in the same commit.
* `tests/test_backend.py` exercises the real `UltralyticsBackend` adapter
  against a stubbed `ultralytics` package (checkpoint resolution, callback
  wiring, export arguments, missing-dependency path).
* `tests/test_dataset.py` covers `data.yaml` validation: required keys,
  `path:` relocation, absolute/relative/glob references and the non-string
  forms the framework resolves itself.
* `tests/test_contract.py` fails if the exported file names drift from
  `src/app/AppConstants.h`.
* `tests/test_qt_contract.py` fails if the argv, checkbox labels or
  `pct=` regex used by `AITrainDockWidget.cpp` drift from what the CLI accepts
  and emits.

The offline suite fakes the ML framework, so API drift in the pinned
`ultralytics` can only be caught by actually training. `.github/workflows/
cvtrain-nightly.yml` runs the real stack (CPU torch) on a tiny synthetic dataset
— one epoch, train → ONNX → manifest → verify — nightly and on demand.
