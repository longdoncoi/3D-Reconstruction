"""Machine-readable progress protocol shared with the Qt dock widget.

``AITrainDockWidget`` parses ``[PROGRESS] pct=<0-100>`` lines from stdout to
drive its progress bar, so this module owns *both* halves of that contract —
the percentage arithmetic and the emitter — in one place. It deliberately knows
nothing about logging configuration (:mod:`cvtrain.logging_setup`) or about the
pipeline that drives it (:mod:`cvtrain.trainer`): the protocol is a leaf.

The logger name (``cvtrain.progress``) matches the module path so a marker can
be filtered by module without a second spelling of the same identity.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal

__all__ = [
    "PHASES",
    "PROGRESS_PREFIX",
    "Phase",
    "ProgressReporter",
    "compute_percent",
]

PROGRESS_PREFIX = "[PROGRESS]"
"""Marker the Qt dock requires at the start of a line before it reads ``pct=``."""

Phase = Literal["preflight", "train", "export", "manifest"]
"""The stages of a run, in pipeline order. Each owns a slice of 0-100%."""

PHASES: tuple[Phase, ...] = ("preflight", "train", "export", "manifest")

# Percentage budget: preflight 0-2, models 2-97, manifest 97-100.
_PREFLIGHT_END = 2.0
_MODELS_END = 97.0
_MODELS_SPAN = _MODELS_END - _PREFLIGHT_END
_TRAIN_SHARE = 0.8
_EXPORT_SHARE = 1.0 - _TRAIN_SHARE


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

    def skipped(self, index: int, key: str) -> None:
        """A model that needed no work still advances the bar past its slot.

        Published at the same percentage a completed export would occupy, so a
        run in which every model is skipped still climbs monotonically to the
        manifest band instead of jumping. ``stage=skip`` is informational — the
        Qt dock reads ``pct=`` only.
        """

        self._emit(compute_percent(index, len(self.models), "export", 1.0), stage="skip", key=key)

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
