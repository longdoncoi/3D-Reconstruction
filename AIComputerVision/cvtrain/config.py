"""Central configuration and the C++/Python model contract.

This module is the single source of truth for:

* where datasets, base checkpoints, training runs and exported models live,
* which base weights map to which exported ONNX file name,
* every default the command line starts from.

It deliberately holds no command-line *syntax*: turning a string such as
``metrics/mAP50(B)=0.5`` into a value is :func:`cvtrain.cli.parse_metric_threshold`,
and deciding whether that value is meaningful is :mod:`cvtrain.preflight`.

It holds no *framework* syntax either: mapping this config onto
``YOLO.train(...)`` keyword arguments is :func:`cvtrain.backend.train_kwargs`,
because ``exist_ok``/``deterministic`` are Ultralytics' names — a second
backend would spell its own, and a leaf must not know them.

The exported file names are a contract with the Qt application. Keep them in
sync with ``src/core/AIConstants.h`` (the constants live there — ``AppConstants.h``
only includes it); ``tests/test_contract.py`` enforces the match so the two
sides cannot silently drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = BASE_DIR.parent

DEFAULT_MODELS_DIR = BASE_DIR / "Models"
DEFAULT_RUNS_DIR = BASE_DIR / "runs"
DEFAULT_WEIGHTS_DIR = BASE_DIR
DEFAULT_DATA_YAML = PROJECT_ROOT / "Dataset" / "data.yaml"
MANIFEST_NAME = "manifest.json"

DEFAULT_EPOCHS = 5
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 0
DEFAULT_SEED = 0
DEFAULT_OPSET = 12

MAX_EPOCHS = 10000
"""Fail-fast ceiling: larger values are a typo, not a training plan."""

MAX_IMGSZ = 4096
"""Fail-fast ceiling: larger images OOM long before they train."""

MAX_WORKERS = 64
"""Fail-fast ceiling: beyond this the loader only adds contention."""

SUPPORTED_ROLES = frozenset({"detection", "segmentation", "tracking"})
"""Semantic roles the ONNX contract (and therefore the Qt loader) understands.

A ``ModelSpec`` may carry any role string, but only these three have a tensor
contract in :mod:`cvtrain.onnx_contract` and a parser in
``src/services/ai``. Preflight rejects anything else before training starts, so
an unsupported family fails in seconds instead of after an expensive run —
``tests/test_contract.py`` cross-checks this set against the contract factory so
the two cannot drift apart.
"""


@dataclass(frozen=True)
class ModelSpec:
    """Describes one trainable model and its exported artifact."""

    key: str
    """CLI/selection key, e.g. ``det``."""

    ckpt: str
    """Base weights used as the training starting point."""

    output: str
    """Exported ONNX file name (contract with the Qt app)."""

    runs_subdir: str
    """Sub-directory under ``runs/`` that holds this model's training logs."""

    role: str
    """Semantic role used by the manifest and the C++ loader."""

    label: str
    """Human readable label, mirrors the Qt training dialog."""


# The checkpoint a spec trains from is resolved through ``TrainConfig`` (which
# owns ``weights_dir``) — see :meth:`TrainConfig.weights_path`. Keeping it off
# the spec means the input directory is overridable exactly like the output
# directories, and a test can point the whole pipeline at a sandbox.


class ModelRegistry:
    """Registry for model specifications, allowing dynamic registration of model tasks."""

    def __init__(self, specs: Iterable[ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        if specs is not None:
            for spec in specs:
                self.register(spec)

    def register(self, spec: ModelSpec) -> None:
        """Register a new model specification. Fails if key or output is already taken."""
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"expected ModelSpec instance, got {type(spec).__name__}")
        if spec.key in self._specs:
            raise ValueError(f"model key {spec.key!r} already registered")
        for existing in self._specs.values():
            if existing.output == spec.output:
                raise ValueError(f"output {spec.output!r} already registered by model {existing.key!r}")
        self._specs[spec.key] = spec

    def get(self, key: str) -> ModelSpec:
        """Retrieve a ModelSpec by key."""
        if key not in self._specs:
            raise KeyError(f"unknown model key {key!r}; available: {', '.join(self._specs)}")
        return self._specs[key]

    def keys(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def values(self) -> tuple[ModelSpec, ...]:
        return tuple(self._specs.values())

    def items(self) -> tuple[tuple[str, ModelSpec], ...]:
        return tuple(self._specs.items())

    def __getitem__(self, key: str) -> ModelSpec:
        return self.get(key)

    def __contains__(self, key: object) -> bool:
        return key in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def as_dict(self) -> dict[str, ModelSpec]:
        return dict(self._specs)


# Order is significant: it defines the training order and the UI listing order.
DEFAULT_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="det",
        ckpt="yolo11n.pt",
        output="yolo11n.onnx",
        runs_subdir="detect",
        role="detection",
        label="Detection (yolo11n)",
    ),
    ModelSpec(
        key="seg",
        ckpt="yolo11n-seg.pt",
        output="yolo11n-seg.onnx",
        runs_subdir="segment",
        role="segmentation",
        label="Segmentation (yolo11n-seg)",
    ),
    ModelSpec(
        key="track",
        ckpt="yolo11x.pt",
        output="yolo11x-tracking.onnx",
        runs_subdir="tracking",
        role="tracking",
        # The artifact is a *detector*: association (IoU matching, track IDs)
        # lives in the Qt runtime, not in the ONNX graph. The historical file
        # name stays for the AIConstants.h contract; the label must not imply
        # that the model itself tracks.
        label="Tracking detector (yolo11x)",
    ),
)

TRACKER_INFO: dict[str, object] = {
    "algorithm": "iou",
    "iou_threshold": 0.3,
    "scope": "qt-runtime",
    "implementation": "src/services/ai/AIProcessor.cpp",
}
"""Association metadata for the ``tracking`` role.

The exported ONNX contains detection weights only; ``AIProcessor::runTracking``
matches consecutive-frame boxes and assigns IDs. Recording this in the manifest
keeps provenance honest: an auditor reading "tracking" knows the tracker is app
code, and ``tests/test_contract.py`` checks the threshold against the C++
source so the two cannot drift.
"""

MODEL_REGISTRY = ModelRegistry(DEFAULT_SPECS)
"""The single source of truth for trainable models.

Every consumer — preflight, the CLI, the trainer, the manifest — resolves
models through this object at runtime, never through a module-level snapshot.
That is what makes :func:`register_model_spec` a real extension point rather
than an API whose additions the pipeline immediately forgets: a spec
registered before the pipeline runs is visible to every ``keys()``/``get()``
call, because they all read the same live registry.
"""

# Backwards compatible default: the legacy script always trained det + seg.
DEFAULT_MODELS: tuple[str, ...] = ("det", "seg")


def register_model_spec(spec: ModelSpec) -> None:
    """Register a new ModelSpec into the live registry.

    The registry is the only bookkeeping a registration needs: every reader
    (CLI listing, preflight, trainer, manifest adoption) goes through it, so a
    newly registered model is immediately selectable everywhere. Raises
    ``ValueError`` if the key or the output file name is already taken.
    """
    MODEL_REGISTRY.register(spec)


def resolve_models(
    selected: Iterable[object] | None,
    registry: ModelRegistry | None = None,
) -> tuple[str, ...]:
    """Normalise a user selection into a canonical, ordered tuple of keys.

    An empty selection falls back to :data:`DEFAULT_MODELS` (filtered to the
    keys this registry actually has) so ``python TrainModel.py --yes`` keeps
    behaving like the legacy script, while the Qt dialog (which always passes
    explicit ``--det/--seg/--track`` flags) gets exactly what the user ticked.

    Unknown keys are a *rejection*, not a fallback: silently training the
    default set because the caller typo'd a model key would hide the mistake
    behind a plausible-looking run. A registry with no default key left is the
    same class of error (there is nothing the caller asked for we can run), so
    it is named rather than replaced by a guess.
    """

    reg = registry or MODEL_REGISTRY
    requested = tuple(str(item) for item in selected or ())
    if requested:
        unknown = [key for key in requested if key not in reg]
        if unknown:
            available = ", ".join(reg.keys()) or "none"
            raise ValueError(
                f"unknown model key(s): {', '.join(unknown)} (choose from {available})"
            )
        chosen = set(requested)
        return tuple(key for key in reg.keys() if key in chosen)
    defaults = tuple(key for key in DEFAULT_MODELS if key in reg)
    if not defaults:
        available = ", ".join(reg.keys()) or "none"
        raise ValueError(
            f"no model selected and none of the legacy default models "
            f"({', '.join(DEFAULT_MODELS)}) exist in this registry (available: {available})"
        )
    return defaults


@dataclass
class TrainConfig:
    """Resolved runtime configuration for a single training invocation."""

    models: tuple[str, ...] = DEFAULT_MODELS
    data_yaml: Path = DEFAULT_DATA_YAML
    models_dir: Path = DEFAULT_MODELS_DIR
    runs_dir: Path = DEFAULT_RUNS_DIR
    weights_dir: Path = DEFAULT_WEIGHTS_DIR
    """Directory holding the base checkpoints (``yolo11n.pt``, ...).

    Overridable like ``models_dir``/``runs_dir`` so inputs and outputs move
    together: preflight, the backend and the provenance digest all resolve a
    checkpoint through :meth:`weights_path` and can therefore never disagree."""

    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_IMGSZ
    batch: int = DEFAULT_BATCH
    workers: int = DEFAULT_WORKERS
    seed: int = DEFAULT_SEED
    device: str | None = None
    device_resolved: str | None = None
    """The concrete device this run trains on, resolved *before* training.

    ``device`` is what the operator asked for; ``auto`` is a request, not a
    fact — "let the framework choose" answers differently on a machine with a
    GPU and on one without, and differently again if the GPU is busy. The
    pipeline therefore resolves it itself, before any weight is touched, and
    records the answer next to the request.

    Two consequences, both wanted: the provenance names a device that can be
    *compared* with the next run's (a CPU run can no longer pass for a GPU
    run), and the framework is told which device to use instead of choosing
    for itself. Left ``None`` when the caller did not resolve it (a
    programmatic call that never reached the orchestrator).
    """
    opset: int = DEFAULT_OPSET

    auto_confirm: bool = False
    tensorboard: bool = False
    dry_run: bool = False
    check_only: bool = False
    verbose: bool = False
    skip_unchanged: bool = False
    """Skip models whose published ONNX already satisfies this exact invocation.

    Makes a run idempotent: re-running the same command trains nothing. See
    :func:`cvtrain.gate.already_current` for what "satisfies" means."""

    resume: bool = False
    """Continue the most recent run of each model instead of starting a new one.

    A training run is hours of work; an interrupted one should not have to
    start over. The framework writes ``weights/last.pt`` after every epoch, so
    ``--resume`` picks the newest checkpoint of the model and continues from
    it — with the hyper-parameters that run was started with, not with the
    ones this invocation happens to pass. Off by default: a fresh run is the
    reproducible one."""

    min_metrics: tuple[tuple[str, float], ...] = ()
    """Quality gate: ``(metric_name, minimum)`` pairs that a model must meet.

    Empty by default, so the gate is opt-in. Applied to the metrics a training
    run just produced *and* to the metrics recorded for a model skipped by
    ``skip_unchanged`` — skipping must never be a way around the gate."""

    def output_path(self, spec: ModelSpec) -> Path:
        """Where ``spec``'s exported ONNX is published for this invocation."""

        return self.models_dir / spec.output

    def weights_path(self, spec: ModelSpec) -> Path:
        """Where ``spec``'s base checkpoint lives for this invocation.

        The single way to resolve a checkpoint: preflight, the backend and the
        provenance digest must all agree, so they all go through here.
        """

        return self.weights_dir / spec.ckpt

    @property
    def manifest_path(self) -> Path:
        """Absolute path of the model manifest for this invocation."""

        return self.models_dir / MANIFEST_NAME

    @property
    def device_class(self) -> str:
        """The canonical spelling of the *requested* device, used as provenance.

        ``None``/blank means "let the framework choose" and is normalised to
        ``auto``; anything else is lower-cased and whitespace-stripped around
        commas, so ``"0, 1"`` and ``"0,1"`` describe the same experiment. Both
        the recording half (``trainer``) and the comparing half
        (``gate.already_current``) read this one property, so a re-run can never
        disagree with itself about which device was asked for.

        The *answer* lives in :attr:`device_resolved`: ``auto`` says nothing
        about the machine, the resolution says exactly what ran.
        """

        raw = (self.device or "").strip()
        if not raw:
            return "auto"
        return ",".join(part.strip().lower() for part in raw.split(","))
