# Repository guide for AI agents

## Scope first

- C++/Qt application code is in `src/`; CMake tests are in `tests/`.
- The Python service is isolated in `AIAssistant/`.
- The YOLO training/export pipeline lives in `AIComputerVision/` (`cvtrain/`
  package). Exported `Models/*.onnx` files are Git LFS artifacts and
  `Models/manifest.json` records their provenance; keep the file names in sync
  with `src/app/AppConstants.h`.
- Treat `build*/`, `.vs/`, `AIAssistant/Cache/`, `AIAssistant/logs/`, and
  `AIComputerVision/runs/` as generated output. Never inspect or edit them
  unless a task explicitly concerns a generated artifact.
- Read the smallest relevant file set. Use `rg` before opening broad folders.

## Required checks

- C++ changes: configure/build the affected target; run the relevant CTest
  tests when dependencies are available.
- Python changes: from `AIAssistant/`, run `ruff check .` and
  `python -m compileall -q StartChatbotServer.py evals src`.
- AIComputerVision changes: from `AIComputerVision/`, run `ruff check .`,
  `python -m mypy`, then `coverage run -m unittest discover -s tests` and
  `coverage report` (the coverage gate is 100% statements + branches).
- Workflow changes: keep `workflow_dispatch`, path filters, and a finite job
  timeout. Do not cancel long dependency-bootstrap jobs.

## Change discipline

- Keep C++ and Python changes independent unless the interface requires both.
- Do not commit model weights outside Git LFS, caches, logs, build trees, IDE
  state, or secrets.
- Prefer one focused change plus a verification result over broad rewrites.
