"""Training backend: the seam between the pipeline and the ML framework.

The orchestrator in :mod:`cvtrain.trainer` never imports ``ultralytics``
directly — it talks to the small :class:`TrainingBackend` protocol instead.
That buys two things:

* **Lazy weight.** ``--help``, ``--check``, ``--dry-run``, linting and the unit
  suite all run without the ML stack installed, because only
  :class:`UltralyticsBackend` knows how to import it.
* **Testability.** The full train -> export -> publish -> provenance flow can be
  exercised with a fake backend in milliseconds (see ``tests/test_trainer.py``),
  which is what makes the orchestration logic verifiable at all.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from .config import ModelSpec, TrainConfig

LOG = logging.getLogger("cvtrain.backend")

__all__ = ["EpochCallback", "TrainedModel", "TrainingBackend", "UltralyticsBackend"]

EpochCallback = Callable[[int, int], None]
"""``(epoch_index, epoch_total)`` — 0-based index, 1-based total."""

H = TypeVar("H")
"""Type of the opaque framework object a backend returns from ``train``.

The orchestrator never inspects the handle — it only passes it back to the same
backend's :meth:`TrainingBackend.export` — so ``H`` ties ``train`` and ``export``
together without leaking framework types into :mod:`cvtrain.trainer`.
"""


@dataclass
class TrainedModel(Generic[H]):
    """Handle to a model that finished training, ready to be exported."""

    handle: H
    """Opaque framework object needed to export; the pipeline never touches it."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Metrics reported by the framework (``{}`` when it reported none)."""

    run_dir: Path = field(default_factory=Path)
    """Directory holding this run's logs/weights — recorded as provenance."""


@runtime_checkable
class TrainingBackend(Protocol[H]):
    """What the orchestrator needs from an ML framework."""

    version: str
    """Framework version, recorded in the manifest as provenance."""

    required_modules: tuple[str, ...]
    """Importable modules this backend needs — handed straight to preflight."""

    def prepare(self, cfg: TrainConfig) -> None:
        """Apply run-wide settings (once, before any model is trained)."""

    def train(self, spec: ModelSpec, cfg: TrainConfig, on_epoch: EpochCallback) -> TrainedModel[H]:
        """Train ``spec`` from its base checkpoint, reporting per-epoch progress."""

    def export(self, model: TrainedModel[H], cfg: TrainConfig) -> Path:
        """Export ``model`` to ONNX and return the path of the generated file."""


def metrics_from(results: object) -> dict[str, float]:
    """Coerce a framework result object into ``{name: float}``.

    Anything non-numeric (numpy scalars already converted, ``None``, nested
    objects) is dropped rather than smuggled into the JSON manifest, where it
    would make the file unreadable.
    """

    data = getattr(results, "results_dict", None)
    if not isinstance(data, dict):
        return {}
    clean: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            clean[str(key)] = float(value)
    return clean


def _run_timestamp() -> str:
    # Microsecond precision: two invocations in the same second would otherwise
    # collide on `name` under ultralytics' ``exist_ok=False`` and fail the run.
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


class UltralyticsBackend:
    """Adapter over ``ultralytics``. Imports happen on first real use only."""

    #: Modules preflight must see before this backend can train or export.
    #: ``yaml`` parses ``data.yaml`` and ``onnx`` is required by the export step.
    required_modules: tuple[str, ...] = ("ultralytics", "yaml", "onnx")

    def __init__(self) -> None:
        self.version = ""
        self._yolo: Any = None
        self._settings: Any = None

    def _ensure_loaded(self) -> None:
        if self._yolo is not None:
            return
        try:
            import ultralytics
            from ultralytics import YOLO
            from ultralytics.utils import SETTINGS
        except ImportError as exc:
            from .errors import BackendUnavailableError

            raise BackendUnavailableError(
                f"ultralytics is not importable ({exc}); run: pip install -r requirements.txt"
            ) from exc
        self._yolo = YOLO
        self._settings = SETTINGS
        self.version = str(ultralytics.__version__)

    def prepare(self, cfg: TrainConfig) -> None:
        self._ensure_loaded()
        if cfg.tensorboard and self._settings is not None:
            self._settings.update({"tensorboard": True})
            LOG.info("TensorBoard logging enabled. Inspect with: tensorboard --logdir %s", cfg.runs_dir)

    def train(self, spec: ModelSpec, cfg: TrainConfig, on_epoch: EpochCallback) -> TrainedModel[Any]:
        self._ensure_loaded()
        model = self._yolo(str(spec.weights_path))

        def _on_epoch_end(trainer: Any) -> None:
            on_epoch(
                int(getattr(trainer, "epoch", 0)),
                int(getattr(trainer, "epochs", cfg.epochs)),
            )

        model.add_callback("on_train_epoch_end", _on_epoch_end)

        project = cfg.runs_dir / spec.runs_subdir
        name = f"train-{_run_timestamp()}"
        results = model.train(project=str(project), name=name, **cfg.train_kwargs())
        run_dir = Path(getattr(model.trainer, "save_dir", project / name))
        return TrainedModel(metrics=metrics_from(results), run_dir=run_dir, handle=model)

    def export(self, model: TrainedModel[Any], cfg: TrainConfig) -> Path:
        return Path(model.handle.export(format="onnx", opset=cfg.opset))
