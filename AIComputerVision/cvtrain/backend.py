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
import numbers
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from . import __version__
from .config import ModelSpec, TrainConfig
from .errors import BackendUnavailableError

LOG = logging.getLogger("cvtrain.backend")

__all__ = [
    "EpochCallback",
    "TrainedModel",
    "TrainingBackend",
    "UltralyticsBackend",
    "exporter_identity",
    "train_kwargs",
]

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


def _package_version(name: str) -> str:
    """Installed distribution version, or ``"unknown"`` when it is absent."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def exporter_identity(ultralytics_version: str) -> str:
    """Canonical identity of the stack that produced an export.

    ``ultralytics.__version__`` alone is not enough: the same Ultralytics build
    delegates ONNX export to ``torch`` and the ``onnx`` package, and either can
    change the graph it emits. Recording all four versions means a
    ``--skip-unchanged`` verdict is invalidated when any export-relevant
    dependency moves, which is the direction a false "unchanged" costs hours.

    The format is a stable, greppable ``name=version;...`` string. ``unknown``
    is deliberate rather than an omission: a missing piece of the stack is
    still a fact, and comparing two unknowns does not prove anything by itself
    — the rest of the entry has to match as well.
    """

    return ";".join(
        (
            f"cvtrain={__version__}",
            f"ultralytics={ultralytics_version}",
            f"torch={_package_version('torch')}",
            f"onnx={_package_version('onnx')}",
        )
    )


@runtime_checkable
class TrainingBackend(Protocol[H]):
    """What the orchestrator needs from an ML framework.

    ``@runtime_checkable`` so ``tests/test_backend.py`` can assert structural
    conformance of a duck-typed double with ``isinstance``. That check only
    sees the members, never the behaviour — the type checker and the fake
    backend in ``tests/test_trainer.py`` are what pin down the semantics.
    """

    version: str
    """Composite exporter identity, recorded in the manifest as provenance.

    It names every export-relevant dependency (see :func:`exporter_identity`),
    not just the high-level framework: a backend upgrade and a ``torch``/``onnx``
    upgrade can both change the graph a re-run would build.
    """

    def required_modules(self, cfg: TrainConfig) -> tuple[str, ...]:
        """Importable modules this backend needs *for this run* — handed to preflight.

        A method rather than a constant because some dependencies are
        conditional on the configuration: ``--tensorboard`` needs the
        ``tensorboard`` package, and a feature the pipeline cannot turn on must
        be an environment failure *before* training starts, not a silent no-op
        halfway through it. Declared by the adapter — the module that knows
        which flag needs which package — so ``preflight`` still hard-codes no
        framework.
        """

    def prepare(self, cfg: TrainConfig) -> None:
        """Apply run-wide settings (once, before any model is trained)."""

    def train(self, spec: ModelSpec, cfg: TrainConfig, on_epoch: EpochCallback) -> TrainedModel[H]:
        """Train ``spec`` from its base checkpoint, reporting per-epoch progress."""

    def check_device(self, cfg: TrainConfig) -> tuple[bool, str]:
        """Validate device availability against the ML runtime environment."""

    def export(self, model: TrainedModel[H], cfg: TrainConfig) -> Path:
        """Export ``model`` to ONNX and return the path of the generated file."""


def metrics_from(results: object) -> dict[str, float]:
    """Coerce a framework result object into ``{name: float}``.

    Anything that is not a real number (``None``, nested objects, strings) is
    dropped rather than smuggled into the JSON manifest, where it would make the
    file unreadable. ``numbers.Real`` — not ``float`` — is the test, so the
    numpy scalar types frameworks like to report are kept instead of silently
    vanishing: a metric that disappears would turn a ``--min-metric`` request
    into an unexplanable "not reported" failure.
    """

    data = getattr(results, "results_dict", None)
    if not isinstance(data, dict):
        return {}
    clean: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, numbers.Real):
            clean[str(key)] = float(value)
    return clean


def _run_timestamp() -> str:
    # Microsecond precision plus the PID: two invocations in the same second
    # would otherwise collide on `name` under ultralytics' ``exist_ok=False``
    # and fail the run — and two *processes* can share a microsecond, so the
    # timestamp alone is not unique.
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{os.getpid()}"


def train_kwargs(cfg: TrainConfig) -> dict[str, object]:
    """Keyword arguments forwarded to ``YOLO.train``.

    This mapping is *framework syntax*, so it belongs to the adapter, not to
    :mod:`cvtrain.config` — ``exist_ok``/``deterministic``/``data`` are
    Ultralytics' names, and a second backend would spell its own. ``cfg`` stays
    a plain description of the run.
    """

    kwargs: dict[str, object] = {
        "data": str(cfg.data_yaml),
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "workers": cfg.workers,
        "seed": cfg.seed,
        "deterministic": True,
        "exist_ok": False,
        "verbose": cfg.verbose,
    }
    # ``auto`` is provenance spelling ("let the framework choose"), not a
    # device the framework accepts: only a concrete device is forwarded.
    if cfg.device and cfg.device_class != "auto":
        kwargs["device"] = cfg.device_class
    return kwargs


class UltralyticsBackend:
    """Adapter over ``ultralytics``. Imports happen on first real use only."""

    #: Modules preflight must see before this backend can train or export.
    #: ``yaml`` parses ``data.yaml``, ``onnx`` validates the exported graph and
    #: ``onnxruntime`` runs the post-export smoke inference that mirrors the
    #: Qt loader. ``tensorboard`` joins them only under ``--tensorboard`` (see
    #: :meth:`required_modules`): an optional feature still has to fail fast.
    BASE_MODULES: tuple[str, ...] = ("ultralytics", "yaml", "onnx", "onnxruntime")

    def __init__(self) -> None:
        self.version = ""
        self._yolo: Any = None
        self._settings: Any = None

    def required_modules(self, cfg: TrainConfig) -> tuple[str, ...]:
        if cfg.tensorboard:
            return (*self.BASE_MODULES, "tensorboard")
        return self.BASE_MODULES

    def _ensure_loaded(self) -> None:
        if self._yolo is not None:
            return
        try:
            import ultralytics
            from ultralytics import YOLO
            from ultralytics.utils import SETTINGS
        except ImportError as exc:
            raise BackendUnavailableError(
                f"ultralytics is not importable ({exc}); run: pip install -r requirements.txt"
            ) from exc
        self._yolo = YOLO
        self._settings = SETTINGS
        self.version = exporter_identity(str(ultralytics.__version__))

    def prepare(self, cfg: TrainConfig) -> None:
        self._ensure_loaded()
        if not cfg.tensorboard:
            return
        # Preflight already required the package when deps were fatal, but a
        # dry-run-shaped caller can reach here without it: ultralytics drops
        # TensorBoard *silently* when SummaryWriter is missing, so the log line
        # must describe what actually happened rather than what was requested.
        try:
            import tensorboard  # noqa: F401  # existence probe, not a use
        except ImportError:
            LOG.warning("tensorboard is not importable; no TensorBoard logs will be written")
            return
        if self._settings is None:
            LOG.warning("the framework exposes no settings; TensorBoard logging stays off")
            return
        self._settings.update({"tensorboard": True})
        LOG.info("TensorBoard logging enabled. Inspect with: tensorboard --logdir %s", cfg.runs_dir)

    def train(self, spec: ModelSpec, cfg: TrainConfig, on_epoch: EpochCallback) -> TrainedModel[Any]:
        self._ensure_loaded()
        model = self._yolo(str(cfg.weights_path(spec)))

        def _on_epoch_end(trainer: Any) -> None:
            on_epoch(
                int(getattr(trainer, "epoch", 0)),
                int(getattr(trainer, "epochs", cfg.epochs)),
            )

        model.add_callback("on_train_epoch_end", _on_epoch_end)

        project = cfg.runs_dir / spec.runs_subdir
        name = f"train-{_run_timestamp()}"
        results = model.train(project=str(project), name=name, **train_kwargs(cfg))
        run_dir = Path(getattr(model.trainer, "save_dir", project / name))
        return TrainedModel(metrics=metrics_from(results), run_dir=run_dir, handle=model)

    def check_device(self, cfg: TrainConfig) -> tuple[bool, str]:
        device = (cfg.device or "").strip().lower()
        if not device or device == "auto" or device == "cpu":
            return True, "device ok"
        if "cuda" in device or device.isdigit() or any(token.isdigit() for token in device.split(",")):
            try:
                import torch

                if not torch.cuda.is_available():
                    return False, f"device {cfg.device!r} requested but CUDA is not available in torch"
            except ImportError:
                pass
        return True, "device ok"

    def export(self, model: TrainedModel[Any], cfg: TrainConfig) -> Path:
        return Path(model.handle.export(format="onnx", opset=cfg.opset))
