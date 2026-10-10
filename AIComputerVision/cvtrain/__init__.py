"""cvtrain — modular YOLO training/export pipeline for AIComputerVision.

The package is intentionally import-light: ``ultralytics`` is only imported
when training actually runs, so linting, ``--help`` and the unit test suite
work without the heavy ML dependencies installed.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
