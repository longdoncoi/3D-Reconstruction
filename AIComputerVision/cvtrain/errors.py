"""Typed failure taxonomy and the process exit-code contract.

Two things live here so that neither can drift:

* :class:`ExitCode` — the numeric codes the Qt launcher and the README
  document. They are never written as bare integers anywhere else.
* :class:`CvTrainError` — failures the pipeline raises *deliberately*
  (bad dataset, broken manifest, missing backend). The CLI renders these as a
  single actionable log line instead of a traceback; anything else is a bug and
  keeps the full stack trace.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "BackendUnavailableError",
    "CvTrainError",
    "ExitCode",
]


class ExitCode(IntEnum):
    """Process exit codes. See README.md ("Exit codes")."""

    OK = 0
    """Everything succeeded."""

    FAILURE = 1
    """Unexpected failure; details are in the log."""

    PREFLIGHT = 2
    """Preflight validation failed, or the manifest is inconsistent."""

    ABORTED = 130
    """Interrupted by the user (Ctrl+C / SIGINT)."""


class CvTrainError(Exception):
    """Base class for an expected, already-explained pipeline failure.

    The CLI renders these as a single actionable log line (no traceback) and
    returns :attr:`exit_code`. Anything that is *not* a ``CvTrainError`` is a
    bug and keeps the full stack trace.
    """

    exit_code: ExitCode = ExitCode.FAILURE


class BackendUnavailableError(CvTrainError):
    """The ML backend could not be imported (missing or broken install)."""
