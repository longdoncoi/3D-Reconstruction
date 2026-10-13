"""cvtrain — modular YOLO training/export pipeline for AIComputerVision.

The package is intentionally import-light: ``ultralytics`` is only imported
when training actually runs, so linting, ``--help`` and the unit test suite
work without the heavy ML dependencies installed.
"""

from __future__ import annotations

__all__ = ["__version__", "generator_name"]

__version__ = "1.0.0"


def generator_name() -> str:
    """Provenance tag stamped into the manifest (``"cvtrain X.Y.Z"``).

    One spelling so the four call sites can never drift apart.
    """

    return f"cvtrain {__version__}"
