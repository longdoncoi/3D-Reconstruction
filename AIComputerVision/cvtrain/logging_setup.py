"""Logging setup and the machine-readable progress protocol.

The Qt dock widget parses ``[PROGRESS] pct=<0-100>`` lines from stdout to drive
its progress bar, so :class:`ProgressReporter` keeps that contract in one place.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from typing import Literal

PROGRESS_PREFIX = "[PROGRESS]"

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

Phase = Literal["preflight", "train", "export", "manifest"]
"""The stages of a run, in pipeline order. Each owns a slice of 0-100%."""

PHASES: tuple[Phase, ...] = ("preflight", "train", "export", "manifest")

# Percentage budget: preflight 0-2, models 2-97, manifest 97-100.
_PREFLIGHT_END = 2.0
_MODELS_END = 97.0
_MODELS_SPAN = _MODELS_END - _PREFLIGHT_END
_TRAIN_SHARE = 0.8
_EXPORT_SHARE = 1.0 - _TRAIN_SHARE


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure the package logger once and return it."""

    logger = logging.getLogger("cvtrain")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def compute_percent(
    model_index: int,
    model_count: int,
    phase: Phase,
    fraction: float = 0.0,
) -> int:
    """Map a pipeline stage to an overall 0-100 percentage.

    ``phase`` is one of :data:`PHASES`; ``fraction`` is the progress *within*
    the stage, in ``[0, 1]`` (values outside are clamped).

    Raises ``ValueError`` for an unknown phase rather than guessing a band: a
    typo would otherwise silently publish a plausible-but-wrong percentage to
    the Qt progress bar.
    """

    count = max(1, int(model_count))
    index = min(max(int(model_index), 0), count - 1)
    fraction = min(max(float(fraction), 0.0), 1.0)

    if phase == "preflight":
        percent = _PREFLIGHT_END * fraction
    elif phase == "manifest":
        percent = _MODELS_END + (100.0 - _MODELS_END) * fraction
    elif phase in ("train", "export"):
        span = _MODELS_SPAN / count
        start = _PREFLIGHT_END + span * index
        if phase == "train":
            percent = start + span * _TRAIN_SHARE * fraction
        else:
            percent = start + span * (_TRAIN_SHARE + _EXPORT_SHARE * fraction)
    else:
        raise ValueError(f"unknown phase {phase!r}; expected one of {', '.join(PHASES)}")

    return round(min(max(percent, 0.0), 100.0))


class ProgressReporter:
    """Emits ``[PROGRESS]`` markers understood by the Qt dock widget.

    The Qt side only reads ``pct=<0-100>``; every other field (``stage``,
    ``key``, ``epoch``) is informational and parsed by the log view. The
    reporter remembers the last percentage it published so a terminal
    :meth:`error` marker can repeat it instead of snapping the bar backwards.
    """

    def __init__(self, models: Iterable[str], logger: logging.Logger | None = None) -> None:
        self.models = tuple(models)
        self._logger = logger or logging.getLogger("cvtrain.progress")
        self._last_pct = 0

    @property
    def last_percent(self) -> int:
        """Most recently published percentage (``0`` before the first emit)."""

        return self._last_pct

    def _emit(self, percent: int, **fields: object) -> None:
        self._last_pct = min(max(int(percent), 0), 100)
        extras = " ".join(f"{key}={value}" for key, value in fields.items())
        suffix = f" {extras}" if extras else ""
        self._logger.info("%s pct=%d%s", PROGRESS_PREFIX, self._last_pct, suffix)

    def preflight(self, fraction: float = 1.0) -> None:
        self._emit(compute_percent(0, 1, "preflight", fraction), stage="preflight")

    def train_epoch(self, index: int, epoch: int, epochs: int, key: str) -> None:
        total = max(1, epochs)
        fraction = (epoch + 1) / total
        self._emit(
            compute_percent(index, len(self.models), "train", fraction),
            stage="train",
            key=key,
            epoch=f"{epoch + 1}/{total}",
        )

    def exporting(self, index: int, key: str) -> None:
        self._emit(compute_percent(index, len(self.models), "export", 0.0), stage="export", key=key)

    def exported(self, index: int, key: str) -> None:
        self._emit(compute_percent(index, len(self.models), "export", 1.0), stage="export", key=key)

    def manifest(self, fraction: float = 1.0) -> None:
        self._emit(compute_percent(0, 1, "manifest", fraction), stage="manifest")

    def done(self) -> None:
        """Terminal marker: the run finished successfully."""

        self._emit(100, stage="done")

    def error(self) -> None:
        """Terminal marker: the run failed. Holds the bar at its last value.

        Emitted on every failure path so the protocol always ends in a defined
        state (``done`` or ``error``) instead of simply going silent.
        """

        self._emit(self._last_pct, stage="error")
