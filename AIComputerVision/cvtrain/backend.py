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
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module, metadata
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from . import __version__
from .config import RUN_DIR_PREFIX, ModelSpec, TrainConfig, canonical_device
from .dataset import dataset_content_fingerprint
from .errors import BackendUnavailableError, ResumeError
from .fsio import claim_run_marker, clear_run_marker, touch_run_marker
from .metrics import coerce_metric

LOG = logging.getLogger("cvtrain.backend")

__all__ = [
    "EXPORT_OPTIONS",
    "EpochCallback",
    "RunFacts",
    "TrainPlan",
    "TrainedModel",
    "TrainingBackend",
    "UltralyticsBackend",
    "exporter_identity",
    "metrics_from",
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


def _as_int(value: object) -> int | None:
    """``value`` when it is a real integer (``bool`` is not a number here)."""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _normalise_device(value: object) -> str | None:
    """The canonical spelling of a device, or ``None`` when it was unset.

    ``None`` means "this run never named a device" and is kept distinct from
    ``"auto"``, which is a request. Every named device goes through
    :func:`cvtrain.config.canonical_device`, so ``"0"``, ``"cuda"`` and
    ``"cuda:0"`` are one answer and the resume log compares like with like.
    """

    if value is None:
        return None
    return canonical_device(value)


def _device_name(value: object) -> str | None:
    """A concrete device as the framework reports it (``cpu``, ``cuda:0``…).

    The framework reports a ``torch.device``; its ``str`` is already the
    canonical spelling for the common cases. The canonicaliser is applied
    anyway because a checkpoint may carry the operator's spelling (``"0"``)
    and the manifest must not record two names for one device.
    """

    if value is None:
        return None
    text = str(value).strip()
    return canonical_device(text) if text else None


@dataclass(frozen=True)
class TrainPlan:
    """What a training invocation will actually start from, decided up front.

    Resolving this *before* ``train`` is what lets the provenance recorder hash
    the real starting file: a resumed run starts from the run's ``last.pt``,
    not from the base checkpoint named in the model spec, and hashing after
    training would describe a checkpoint the run itself has since rewritten.
    """

    source_weights: Path
    """The file training starts from — base weights, or ``last.pt`` on resume."""

    resume_from: Path | None = None
    """The interrupted run's checkpoint when this plan continues one."""

    dataset_content_sha256: str | None = None
    """Digest of the dataset this invocation is about to train on.

    The backend compares it with the dataset recorded in the checkpoint before
    a resume: the framework restores ``data`` from the checkpoint, so continuing
    a run against different bytes would be a different experiment behind the
    same run name.
    """

    @property
    def resumed(self) -> bool:
        """Whether this plan continues an interrupted run."""

        return self.resume_from is not None


@dataclass(frozen=True)
class RunFacts:
    """The experiment a backend *actually executed*, as opposed to its request.

    Every field is ``None`` when the framework cannot report it; the recorder
    then keeps the invocation's value rather than inventing a fact. A resumed
    run restores its argument set from the checkpoint and ignores the command
    line, so these are the values that belong in the manifest.
    """

    epochs: int | None = None
    """Effective training budget (the checkpoint's total on resume)."""

    epochs_completed: int | None = None
    """Epochs actually completed, cumulative for a resumed run.

    The budget and the completed count are different facts: early stopping can
    finish a 100-epoch budget after 40 epochs, and the framework's own counter
    is the only thing that can say so.
    """

    dataset: str | None = None
    imgsz: int | None = None
    batch: int | None = None
    device_resolved: str | None = None
    """Device the run actually used, in the framework's spelling.

    ``torch.device`` is reported as ``cuda:0``/``cpu``; a multi-GPU argument is
    kept as the list the framework was told (``cuda:0,cuda:1``). The canonical
    spelling matters because the next run's resolution is written the same way
    (ADR 0016).
    """

    source_weights: str | None = None
    """The file the run started from."""

    resumed_from: str | None = None
    """The checkpoint a resumed run continued, when it was one."""


@dataclass
class TrainedModel(Generic[H]):
    """Handle to a model that finished training, ready to be exported."""

    handle: H
    """Opaque framework object needed to export; the pipeline never touches it."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Metrics reported by the framework (``{}`` when it reported none)."""

    run_dir: Path = field(default_factory=Path)
    """Directory holding this run's logs/weights — recorded as provenance."""

    facts: RunFacts = field(default_factory=RunFacts)
    """The effective facts of the run, read from the framework's own state.

    Deliberately a single object rather than parallel fields: every consumer
    (provenance, gate, tests) then reads one answer to "what actually ran", and
    a new fact cannot be recorded in one place and forgotten in another.
    """

    @property
    def epochs(self) -> int | None:
        """The effective training budget, when the framework can report it."""

        return self.facts.epochs

    @property
    def epochs_completed(self) -> int | None:
        """The epochs actually completed, when the framework can report it."""

        return self.facts.epochs_completed


def _package_version(name: str) -> str:
    """Installed distribution version, or ``"unknown"`` when it is absent."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def exporter_identity(ultralytics_version: str) -> str:
    """Canonical identity of the stack that produced an export.

    ``ultralytics.__version__`` alone is not enough: the same Ultralytics build
    delegates ONNX export to ``torch`` and the ``onnx`` package, and simplifies
    the graph with ``onnxslim``. Any of them can change the bytes it emits, so
    all five versions are recorded and a ``--skip-unchanged`` verdict is
    invalidated when one moves — the direction a false "unchanged" costs hours.
    ``onnxruntime`` is deliberately *not* part of this identity: it validates
    the exported graph (the pinned contract engine) but does not produce it.

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
            f"onnxslim={_package_version('onnxslim')}",
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

    def plan(
        self,
        spec: ModelSpec,
        cfg: TrainConfig,
        *,
        dataset_content_sha256: str | None = None,
    ) -> TrainPlan:
        """Decide what ``spec`` will train from, before any expensive work.

        This is also where ``--resume`` fails closed: a request with no
        checkpoint, or with a checkpoint that cannot be proven to describe the
        same dataset, raises instead of silently starting over. The plan is
        returned (not recomputed by :meth:`train`) so the provenance recorder
        can hash the exact file training will start from.
        """

    def train(self, spec: ModelSpec, cfg: TrainConfig, on_epoch: EpochCallback, plan: TrainPlan) -> TrainedModel[H]:
        """Train ``spec`` according to ``plan``, reporting per-epoch progress."""

    def check_device(self, cfg: TrainConfig) -> tuple[bool, str]:
        """Validate device availability against the ML runtime environment."""

    def resolve_device(self, cfg: TrainConfig) -> str:
        """The concrete device this run will train on.

        ``auto`` is a request, not a fact: the answer depends on the machine
        and on the moment, so an artifact whose provenance says only "auto"
        says nothing a later run can compare against. The pipeline resolves it
        itself — before any weight is touched — and records the answer next to
        the request.

        The policy is deliberately simple and explicit: the first CUDA device
        when the runtime has one, CPU otherwise. There is no silent fallback
        ("the GPU looked busy, so CPU it is"), because a fallback the
        provenance does not mention is a lie.
        """

    def check_importable(self, cfg: TrainConfig) -> tuple[bool, str]:
        """Whether the framework and its dependencies can actually be imported.

        The preflight's dependency check proves the packages are *installed*;
        this one proves they can be *imported*. A distribution whose own
        dependency is missing is installed and broken, and the difference
        matters: the report must not say "environment ready" for a stack that
        cannot be loaded.

        The framework is imported here — inside the adapter, the one module
        that owns the heavy imports (ADR 0001) — so the rest of the package
        stays import-light. The import is also what fills in
        :attr:`version`, the exporter identity the manifest records.
        """

    def export(self, model: TrainedModel[H], cfg: TrainConfig) -> Path:
        """Export ``model`` to ONNX and return the path of the generated file.

        The call must be driven by exactly :meth:`export_args`: the recorder
        stores that mapping as the artifact's identity, and a second run skips
        only when it would ask for the same export.
        """

    def export_args(self, cfg: TrainConfig) -> dict[str, object]:
        """The complete option set :meth:`export` will be driven by.

        Values come from the invocation (``--opset``, ``--imgsz``) and from the
        exporter's pinned defaults: the same weights exported with another
        opset are another artifact, so "what this run would build" includes
        the export request, not only the training request.
        """


def metrics_from(results: object) -> dict[str, float]:
    """Coerce a framework result object into ``{name: float}``.

    Anything that is not a usable measurement (see :func:`coerce_metric`) is
    dropped rather than smuggled into the JSON manifest, where a ``NaN`` or a
    boolean would make the file unreadable or the quality gate meaningless.
    ``numbers.Real`` — not ``float`` — is the test, so the numpy scalar types
    frameworks like to report are kept instead of silently vanishing: a metric
    that disappears would turn a ``--min-metric`` request into an
    unexplainable "not reported" failure.
    """

    data = getattr(results, "results_dict", None)
    if not isinstance(data, dict):
        return {}
    clean: dict[str, float] = {}
    for key, value in data.items():
        number = coerce_metric(value)
        if number is not None:
            clean[str(key)] = number
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
    # The *resolved* device wins when the pipeline has already resolved it —
    # that is the value the framework is told to use. It is deliberately not
    # the value the manifest records as the request: the request is
    # ``cfg.device_class`` and the answer is ``cfg.device_resolved``.
    if cfg.device_resolved:
        kwargs["device"] = canonical_device(cfg.device_resolved)
    elif cfg.device and cfg.device_class != "auto":
        kwargs["device"] = canonical_device(cfg.device_class)
    return kwargs


EXPORT_OPTIONS: tuple[tuple[str, object], ...] = (
    ("format", "onnx"),
    ("simplify", True),
    ("dynamic", False),
    ("half", False),
    ("int8", False),
    ("nms", False),
)
"""The fixed half of the ONNX export request.

These are Ultralytics defaults, spelled out on purpose: a default lives in a
moving package, and a version that flips one of them changes the emitted graph
without changing a single recorded training parameter. The pipeline passes
them explicitly, records the whole mapping in the manifest and compares it on
the next ``--skip-unchanged`` verdict (see
:meth:`TrainingBackend.export_args`). ``imgsz``/``opset`` are per-run requests
and are added by that method.
"""


class UltralyticsBackend:
    """Adapter over ``ultralytics``. Imports happen on first real use only."""

    #: Modules preflight must see before this backend can train or export.
    #: ``yaml`` parses ``data.yaml``, ``onnx`` validates the exported graph,
    #: ``onnxslim`` produces the simplified graph Ultralytics exports by
    #: default (without it the export is a different graph under the same
    #: version), and ``onnxruntime`` runs the post-export smoke inference that
    #: mirrors the Qt loader. ``tensorboard`` joins them only under
    #: ``--tensorboard`` (see :meth:`required_modules`): an optional feature
    #: still has to fail fast.
    BASE_MODULES: tuple[str, ...] = ("ultralytics", "yaml", "onnx", "onnxslim", "onnxruntime")

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

    def _resume_checkpoint(self, spec: ModelSpec, cfg: TrainConfig) -> Path | None:
        """The newest ``last.pt`` of ``spec``, or ``None`` when there is none.

        Ultralytics resumes a run from the checkpoint the framework wrote, not
        from the base weights: ``weights/last.pt`` carries the optimizer state
        and the epoch counter, so a run interrupted at epoch 4 continues at
        epoch 4 instead of starting over. The newest checkpoint wins, because
        the newest run is the one that was interrupted.

        Whether that checkpoint is *resumable* (interrupted, not completed) and
        describes the same dataset is checked by :meth:`_validate_resume`,
        which has the loaded checkpoint in hand.
        """

        project = cfg.runs_dir / spec.runs_subdir
        if not project.is_dir():
            return None
        candidates = sorted(
            (path for path in project.glob("train-*/weights/last.pt") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def plan(
        self,
        spec: ModelSpec,
        cfg: TrainConfig,
        *,
        dataset_content_sha256: str | None = None,
    ) -> TrainPlan:
        """Resolve the file this run starts from; ``--resume`` fails closed.

        A missing checkpoint used to start a fresh run with a warning. That is
        no longer honest: the operator asked to *continue* work, and silently
        starting over can overwrite the artifact they were trying to preserve.
        The same applies to a checkpoint recorded against another dataset — the
        framework restores ``data`` from it, so continuing would train on data
        the invocation did not name.
        """

        if not cfg.resume:
            return TrainPlan(
                source_weights=cfg.weights_path(spec),
                dataset_content_sha256=dataset_content_sha256,
            )
        resume_from = self._resume_checkpoint(spec, cfg)
        if resume_from is None:
            project = cfg.runs_dir / spec.runs_subdir
            raise ResumeError(
                f"--resume requested for {spec.output}, but no run checkpoint exists under {project}; "
                "run without --resume to start a new one"
            )
        return TrainPlan(
            source_weights=resume_from,
            resume_from=resume_from,
            dataset_content_sha256=dataset_content_sha256,
        )

    @staticmethod
    def _checkpoint_train_args(model: Any, checkpoint: Path) -> dict[str, Any]:
        """The saved training arguments of a resumable checkpoint, or a refusal.

        A completed run's ``last.pt`` has ``epoch == -1`` and no work left: the
        framework asserts when asked to resume it, which would surface as an
        exit-1 traceback instead of the actionable verdict this pipeline owes
        the operator. Missing arguments are just as unusable — the framework
        would silently fill them from its defaults.
        """

        ckpt = getattr(model, "ckpt", None)
        if not isinstance(ckpt, dict):
            raise ResumeError(f"checkpoint {checkpoint} carries no training state; start a new run")
        epoch = ckpt.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ResumeError(
                f"checkpoint {checkpoint} has no interrupted epoch (epoch={epoch!r}); "
                "the run is already complete - start a new one without --resume"
            )
        train_args = ckpt.get("train_args")
        if not isinstance(train_args, dict) or not train_args:
            raise ResumeError(
                f"checkpoint {checkpoint} records no training arguments; refusing to resume against framework defaults"
            )
        return train_args

    def _validate_resume(self, model: Any, plan: TrainPlan, cfg: TrainConfig) -> dict[str, Any]:
        """Prove the checkpoint is resumable and describes this dataset.

        The framework restores the checkpoint's argument set wholesale; it only
        permits a handful of overrides, and ``data`` is not one of them. When
        the checkpoint's dataset path no longer exists, Ultralytics quietly
        substitutes the invocation's ``--data`` — the exact silent mix this
        check exists to prevent. The bytes are compared, not the path: the same
        dataset at another checkout path is the same experiment.
        """

        checkpoint = cast(Path, plan.resume_from)
        train_args = self._checkpoint_train_args(model, checkpoint)
        recorded = train_args.get("data")
        if not isinstance(recorded, str) or not recorded.strip():
            raise ResumeError(
                f"checkpoint {plan.resume_from} records no dataset; refusing to resume against an unknown one"
            )
        checkpoint_data = Path(recorded)
        if not checkpoint_data.is_file():
            raise ResumeError(
                f"checkpoint {plan.resume_from} was trained on {checkpoint_data}, which is not on disk; "
                "restore that dataset or start a new run"
            )
        if plan.dataset_content_sha256 is None:
            raise ResumeError(
                f"cannot fingerprint the requested dataset {cfg.data_yaml}; refusing to resume without proof of the data"
            )
        if checkpoint_data.resolve() != cfg.data_yaml.resolve():
            recorded_digest = dataset_content_fingerprint(checkpoint_data)
            if recorded_digest is None or recorded_digest != plan.dataset_content_sha256:
                raise ResumeError(
                    f"checkpoint {plan.resume_from} was trained on {checkpoint_data}, "
                    f"which does not match {cfg.data_yaml}; start a new run for the new dataset"
                )
        self._log_resume_mismatches(train_args, cfg)
        return train_args

    @staticmethod
    def _log_resume_mismatches(train_args: dict[str, Any], cfg: TrainConfig) -> None:
        """Say which invocation values the checkpoint overrides.

        Only the dataset is a hard refusal (see :meth:`_validate_resume`);
        imgsz/batch/seed/device/epochs are the checkpoint's own experiment and
        may legitimately differ. The run continues with the checkpoint's
        values, and the log makes that visible instead of letting the operator
        believe the command line was applied.
        """

        requested: dict[str, object] = {
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "seed": cfg.seed,
            "data": str(cfg.data_yaml),
            "device": cfg.device_resolved or cfg.device_class,
        }
        for key, wanted in requested.items():
            actual = train_args.get(key)
            if actual is None:
                continue
            if key == "device":
                actual_text = _normalise_device(actual)
                wanted_text = _normalise_device(wanted)
            else:
                actual_text = str(actual)
                wanted_text = str(wanted)
            if actual_text != wanted_text:
                LOG.warning(
                    "Resume uses the checkpoint's %s=%s (the invocation asked for %s); the checkpoint's experiment wins",
                    key,
                    actual,
                    wanted,
                )

    @staticmethod
    def _effective_epochs(model: Any) -> int | None:
        """The effective training budget the framework used, if it can say.

        ``Trainer.epochs`` is the planned total after ``check_resume`` restored
        the checkpoint's arguments — which is exactly the budget the run
        obeyed. It is **not** the completed count: early stopping can end a
        100-epoch budget at epoch 40. See :meth:`_effective_facts`.
        """

        trainer = getattr(model, "trainer", None)
        epochs = getattr(trainer, "epochs", None)
        if isinstance(epochs, bool) or not isinstance(epochs, int):
            return None
        return epochs

    @classmethod
    def _effective_facts(cls, model: Any, plan: TrainPlan, cfg: TrainConfig) -> RunFacts:
        """Read the run's effective facts from the framework's own state.

        Priority is ``trainer.args`` (post-training, with auto-batch and image
        size resolved), then the checkpoint's ``train_args``, then the
        invocation. Nothing is synthesised: a value none of those can provide
        stays ``None`` and the recorder keeps the request.
        """

        trainer = getattr(model, "trainer", None)
        args = getattr(trainer, "args", None)
        ckpt = getattr(model, "ckpt", None)
        train_args = ckpt.get("train_args") if isinstance(ckpt, dict) else None

        def effective(name: str) -> object:
            value = getattr(args, name, None)
            if value is None and isinstance(train_args, dict):
                value = train_args.get(name)
            return value

        epochs = cls._effective_epochs(model)
        if epochs is None:
            epochs = _as_int(effective("epochs"))
        completed = _as_int(getattr(trainer, "epoch", None))
        if completed is not None and completed >= 0:
            completed += 1
        else:
            completed = None
        dataset = effective("data")
        dataset_text = str(dataset) if dataset is not None and str(dataset).strip() else None
        # The framework's effective argument set is the first fact: it preserves
        # a multi-GPU list (``"0,1"``), which ``trainer.device`` collapses to the
        # primary device. It is written in the canonical spelling so the next
        # run's resolution can be compared with it (ADR 0016). ``trainer.device``
        # is the fallback for a framework that does not echo the argument.
        reported_device = _device_name(effective("device"))
        if reported_device in (None, "auto"):
            reported_device = _device_name(getattr(trainer, "device", None))
        return RunFacts(
            epochs=epochs,
            epochs_completed=completed,
            dataset=dataset_text,
            imgsz=_as_int(effective("imgsz")),
            batch=_as_int(effective("batch")),
            device_resolved=reported_device or cfg.device_resolved,
            source_weights=str(plan.source_weights),
            resumed_from=str(plan.resume_from) if plan.resume_from is not None else None,
        )

    def train(self, spec: ModelSpec, cfg: TrainConfig, on_epoch: EpochCallback, plan: TrainPlan) -> TrainedModel[Any]:
        # A resumed run reuses the interrupted run's directory, whose lease
        # still names the killed process. Claim it *before* the framework is
        # imported and training begins: a concurrent ``--prune-runs`` reads
        # that marker as evidence of life, and the first epoch can be hours.
        if plan.resume_from is not None:
            claim_run_marker(plan.resume_from.parent.parent)
        self._ensure_loaded()
        project = cfg.runs_dir / spec.runs_subdir
        if plan.resumed:
            LOG.info("Resuming %s from %s", spec.output, plan.resume_from)
        model = self._yolo(str(plan.source_weights))
        if plan.resumed:
            self._validate_resume(model, plan, cfg)

        def _on_epoch_end(trainer: Any) -> None:
            # The run's own lease proves to ``--prune-runs`` that this process
            # is alive even between checkpoint writes: a long epoch must not
            # make an active run look abandoned. Removed when training returns.
            save_dir = getattr(trainer, "save_dir", None)
            if save_dir:
                touch_run_marker(Path(save_dir))
            on_epoch(
                int(getattr(trainer, "epoch", 0)),
                int(getattr(trainer, "epochs", cfg.epochs)),
            )

        model.add_callback("on_train_epoch_end", _on_epoch_end)

        name = f"{RUN_DIR_PREFIX}{_run_timestamp()}"
        try:
            if plan.resumed:
                # ``resume=True`` restores the run's own arguments from the
                # checkpoint, so the hyper-parameters of the interrupted run
                # are the ones that continue — not the ones this invocation
                # happens to pass. ``_validate_resume`` has already refused an
                # incomplete checkpoint, a different dataset or a completed run.
                results = model.train(resume=True)
            else:
                results = model.train(project=str(project), name=name, **train_kwargs(cfg))
        finally:
            # A completed (or failed) run is no longer live, even though this
            # process still is: leaving the marker would make the pruner keep
            # every finished run of this session forever. A killed process
            # cannot run this cleanup, and its marker correctly ages out.
            run_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", project / name))
            clear_run_marker(run_dir)
        run_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", project / name))
        return TrainedModel(
            metrics=metrics_from(results),
            run_dir=run_dir,
            handle=model,
            facts=self._effective_facts(model, plan, cfg),
        )

    def check_device(self, cfg: TrainConfig) -> tuple[bool, str]:
        device = (cfg.device or "").strip().lower()
        if not device or device == "auto" or device == "cpu":
            return True, "device ok"
        components = [component.strip() for component in device.split(",") if component.strip()]
        cuda_components = [
            component
            for component in components
            if component == "cuda" or component.startswith("cuda:") or component.isdigit()
        ]
        if not cuda_components:
            # ``mps``/``metal`` are platform devices; their availability is the
            # framework's answer, not something this check can prove.
            return True, "device ok"
        if len(cuda_components) != len(components):
            return False, (
                f"device {cfg.device!r} mixes CUDA and non-CUDA components; "
                "request either a CUDA device list or cpu, not both"
            )
        try:
            import torch
        except ImportError:
            # The dependency preflight owns the missing-package verdict; this
            # check must not turn "torch is not installed" into a device error.
            return True, "device ok"
        except Exception as exc:  # a broken DLL is not the same as a missing package
            return False, f"torch could not be loaded ({exc}); fix the installation or pass --device cpu"
        try:
            available = bool(torch.cuda.is_available())
        except Exception as exc:
            return False, f"torch cannot report CUDA availability ({exc}); fix the installation or pass --device cpu"
        if not available:
            return False, f"device {cfg.device!r} requested but CUDA is not available in torch"
        try:
            count = int(torch.cuda.device_count())
        except Exception as exc:
            return False, f"torch cannot report the CUDA device count ({exc}); pass --device cpu or fix the installation"
        for component in cuda_components:
            if component == "cuda":
                index = 0
            elif component.startswith("cuda:"):
                suffix = component[5:]
                if not suffix.isdigit():
                    return False, f"device {cfg.device!r} has an invalid CUDA index {component!r}; use cuda:N"
                index = int(suffix)
            else:
                index = int(component)
            if index >= count:
                return False, (
                    f"device {cfg.device!r} asks for CUDA index {index}, but torch sees {count} device(s); "
                    f"pass --device 0..{count - 1} or cpu"
                )
        return True, "device ok"

    def resolve_device(self, cfg: TrainConfig) -> str:
        requested = cfg.device_class
        if requested != "auto":
            # An explicit request is already the answer — but written in the
            # framework's spelling. ``--device 0`` must resolve to ``cuda:0``,
            # not to the string ``"0"``: the framework reports a ``torch.device``
            # (``cuda:0``), and comparing that fact with the raw request spelling
            # made an identical explicit-device re-run retrain forever (ADR 0016).
            return canonical_device(requested)
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def check_importable(self, cfg: TrainConfig) -> tuple[bool, str]:
        # ``_ensure_loaded`` proves the framework itself; every other module
        # preflight declared is imported here too. A distribution whose own
        # dependency (or native DLL) is missing is *installed and broken*:
        # ``find_spec`` sees it, training starts, and the failure surfaces at
        # export after the expensive part. This is the verdict that closes
        # that window, so it must cover the declared set rather than one
        # representative package.
        try:
            self._ensure_loaded()
        except BackendUnavailableError as exc:
            return False, str(exc)
        for name in self.required_modules(cfg):
            # A broken install can raise ImportError (missing dependency),
            # OSError (a DLL that will not load) or an error raised by the
            # package's own import-time code. All of them mean "not
            # importable", and the report is where the operator reads it.
            try:
                import_module(name)
            except Exception as exc:
                return False, f"{name} is not importable ({exc}); run: pip install -r requirements.txt"
        return True, "framework and declared dependencies importable"

    def export_args(self, cfg: TrainConfig) -> dict[str, object]:
        return {**dict(EXPORT_OPTIONS), "imgsz": cfg.imgsz, "opset": cfg.opset}

    def export(self, model: TrainedModel[Any], cfg: TrainConfig) -> Path:
        return Path(model.handle.export(**self.export_args(cfg)))
