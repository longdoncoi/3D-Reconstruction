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
│  ├─ cli.py              # argparse, command dispatch, exit-code mapping
│  ├─ config.py           # paths, defaults and the C++/Python model contract
│  ├─ errors.py           # ExitCode enum + expected-failure hierarchy
│  ├─ fsio.py             # sha256 + crash-safe write/copy primitives
│  ├─ preflight.py        # fail-fast environment/dataset validation
│  ├─ backend.py          # TrainingBackend protocol + ultralytics adapter
│  ├─ trainer.py          # orchestration: train -> export -> publish -> record
│  ├─ manifest.py         # manifest schema (TypedDict), read/write/verify
│  └─ logging_setup.py    # logging + [PROGRESS] protocol
├─ Models/                # exported *.onnx (Git LFS) + manifest.json
├─ runs/                  # Ultralytics training logs (git-ignored)
├─ requirements.txt       # pinned runtime deps
└─ tests/                 # offline unit tests (no model weights required)
```

## Architecture

Dependencies point one way, from orchestration down to primitives:

```
TrainModel.py → cli → trainer ─┬→ preflight ─┐
                               ├→ manifest ──┼→ fsio / config
                               └→ backend ───┘
```

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
  so preflight/dry-run/prompt/manifest-durability/progress are all verified
  offline;
* `preflight` receives its dependency list from the backend
  (`required_modules`) instead of hard-coding a framework.


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

## Usage

```powershell
# Train the default models (detection + segmentation)
python AIComputerVision/TrainModel.py --yes

# Train only the tracking model
python AIComputerVision/TrainModel.py --track --yes

# Validate the environment and print the plan without training
python AIComputerVision/TrainModel.py --check
python AIComputerVision/TrainModel.py --dry-run

# Override hyper-parameters
python AIComputerVision/TrainModel.py --det --epochs 50 --batch 8 --imgsz 640 --seed 42

# Provenance tooling
python AIComputerVision/TrainModel.py --write-manifest
python AIComputerVision/TrainModel.py --verify-manifest
```

Exit codes are declared once in `cvtrain/errors.py` (`ExitCode`) and never
written as bare integers elsewhere: `0` success, `1` unexpected failure, `2`
preflight/manifest failure, `130` interrupted. Expected failures
(`CvTrainError`) are logged as a single actionable line; anything else keeps
the full traceback.

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
[PROGRESS] pct=100 stage=manifest
[PROGRESS] pct=100 stage=done
```

Every run ends in a terminal marker — `stage=done` on success, or
`stage=error` (holding the last published percentage) on failure — so the bar
never stalls waiting for a line that will not arrive. The Qt side
(`AITrainDockWidget::onTrainProcessOutput`) only parses `pct=`; the other
fields are informational.

## Manifest

Each exported model is recorded in `Models/manifest.json` with its SHA-256,
size, dataset, hyper-parameters and (best-effort) metrics. `--verify-manifest`
recomputes the digests, which makes truncation or accidental replacement
detectable. Verification is bidirectional: an entry whose file went missing is
reported, and so is an ONNX file sitting in `Models/` without an entry.

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
  GPU, no dataset and no ML packages are needed.
* `tests/test_backend.py` exercises the real `UltralyticsBackend` adapter
  against a stubbed `ultralytics` package (checkpoint resolution, callback
  wiring, export arguments, missing-dependency path).
* `tests/test_contract.py` fails if the exported file names drift from
  `src/app/AppConstants.h`.
* `tests/test_qt_contract.py` fails if the argv, checkbox labels or
  `pct=` regex used by `AITrainDockWidget.cpp` drift from what the CLI accepts
  and emits.

The offline suite fakes the ML framework, so API drift in the pinned
`ultralytics` can only be caught by actually training. `.github/workflows/
cvtrain-nightly.yml` runs the real stack (CPU torch) on a tiny synthetic dataset
— one epoch, train → ONNX → manifest → verify — nightly and on demand.
