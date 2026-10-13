"""Typed failure taxonomy and the process exit-code contract.

Two things live here so that neither can drift:

* :class:`ExitCode` — the numeric codes the Qt launcher and the README
  document. They are never written as bare integers anywhere else.
* :class:`CvTrainError` — failures the pipeline raises *deliberately*
  (bad dataset, broken manifest, missing backend, failed quality gate). The CLI
  renders these as a single actionable log line instead of a traceback;
  anything else is a bug and keeps the full stack trace.
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "BackendUnavailableError",
    "ConcurrentRunError",
    "CvTrainError",
    "ExitCode",
    "ManifestIntegrityError",
    "QualityGateError",
    "StorageError",
]


class ExitCode(IntEnum):
    """Process exit codes. See README.md ("Exit codes")."""

    OK = 0
    """Everything succeeded."""

    FAILURE = 1
    """Unexpected failure; details are in the log."""

    PREFLIGHT = 2
    """Preflight validation failed, or the manifest is inconsistent."""

    QUALITY = 3
    """A trained model did not meet the requested quality gate (``--min-metric``).

    Distinct from :attr:`PREFLIGHT` on purpose: ``2`` means "this environment or
    artifact is not trustworthy", ``3`` means "we trained it and it is not good
    enough" — the operator's fix is different (more epochs, a better dataset,
    a lower bar), so the codes must not be conflated. The Qt dock only tests
    ``exitCode != 0``, so a new code is a compatible addition."""

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
    """The ML backend could not be imported (missing or broken install).

    An environment problem is a preflight-class failure, so this is exit code
    ``2`` like every other check that could have caught it earlier: preflight
    normally *does* catch it, and a broken install that slipped past
    ``find_spec`` must not report a different code for the same underlying cause.
    """

    exit_code: ExitCode = ExitCode.PREFLIGHT


class ConcurrentRunError(CvTrainError):
    """Another run owns the models directory, so this one must not proceed.

    Two processes writing ``Models/manifest.json`` at once would each read the
    document, train, and then save — the last writer silently discards the
    other's provenance. The advisory lock in :func:`cvtrain.fsio.directory_lock`
    turns that race into a fast, actionable refusal. Exit code ``2``: the
    environment is not ready for this run, exactly like a failed preflight
    check, and the fix ("wait, or remove a stale lock") is an environment fix.
    """

    exit_code: ExitCode = ExitCode.PREFLIGHT


class ManifestIntegrityError(CvTrainError):
    """The manifest cannot vouch for what is on disk (or the reverse).

    Raised by :func:`cvtrain.manifest.verify_or_raise`, which both callers of
    verification use — the ``--verify-manifest`` command and a training run
    that just re-verified the manifest it wrote. Exit code ``2`` like every
    other "this artifact is not trustworthy" verdict: the fix is the same
    (re-export, re-adopt with ``--write-manifest``, or restore the missing
    file), and reporting a clean run for an inconsistent manifest is the one
    outcome this pipeline must never produce.
    """

    exit_code: ExitCode = ExitCode.PREFLIGHT


class QualityGateError(CvTrainError):
    """A model's metrics did not satisfy a ``--min-metric`` threshold.

    Raised *after* training (or against the metrics recorded for a model
    ``--skip-unchanged`` decided to keep), never by preflight: this is a verdict
    on the artifact, not on the environment. The message names the model, the
    metric and both numbers so the operator knows what to change.
    """

    exit_code: ExitCode = ExitCode.QUALITY


class StorageError(CvTrainError):
    """A file the pipeline had to read, hash or measure is unusable.

    Hashing a checkpoint, measuring a just-published ONNX or scanning
    ``Models/`` can fail on a missing file, a permission or a dying disk.
    Those are environment facts — "this artifact is not trustworthy" — so they
    share exit code ``2`` with preflight and manifest failures instead of
    surfacing as an ``OSError`` traceback with exit ``1``. A traceback would
    imply a bug; here there is none, only storage the run cannot vouch for.

    Deliberately *not* used for the atomic publish path itself: a vanished
    framework output that was supposed to exist is a bug, and keeps its
    traceback.
    """

    exit_code: ExitCode = ExitCode.PREFLIGHT
