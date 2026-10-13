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
sync with ``src/app/AppConstants.h``; ``tests/test_contract.py`` enforces the
match so the two sides cannot silently drift apart.
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


# Order is significant: it defines the training order and the UI listing order.
MODEL_SPECS: dict[str, ModelSpec] = {
    "det": ModelSpec(
        key="det",
        ckpt="yolo11n.pt",
        output="yolo11n.onnx",
        runs_subdir="detect",
        role="detection",
        label="Detection (yolo11n)",
    ),
    "seg": ModelSpec(
        key="seg",
        ckpt="yolo11n-seg.pt",
        output="yolo11n-seg.onnx",
        runs_subdir="segment",
        role="segmentation",
        label="Segmentation (yolo11n-seg)",
    ),
    "track": ModelSpec(
        key="track",
        ckpt="yolo11x.pt",
        output="yolo11x-tracking.onnx",
        runs_subdir="tracking",
        role="tracking",
        label="Tracking (yolo11x-tracking)",
    ),
}

ALL_MODEL_KEYS: tuple[str, ...] = tuple(MODEL_SPECS)

# Backwards compatible default: the legacy script always trained det + seg.
DEFAULT_MODELS: tuple[str, ...] = ("det", "seg")


def resolve_models(selected: Iterable[object] | None) -> tuple[str, ...]:
    """Normalise a user selection into a canonical, ordered tuple of keys.

    An empty selection falls back to :data:`DEFAULT_MODELS` so ``python
    TrainModel.py --yes`` keeps behaving like the legacy script, while the Qt
    dialog (which always passes explicit ``--det/--seg/--track`` flags) gets
    exactly what the user ticked.
    """

    chosen = {str(item) for item in selected} if selected else set()
    ordered = tuple(key for key in ALL_MODEL_KEYS if key in chosen)
    return ordered or DEFAULT_MODELS


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
        """The canonical spelling of ``device`` used as provenance.

        ``None``/blank means "let the framework choose" and is normalised to
        ``auto``; anything else is lower-cased and whitespace-stripped around
        commas, so ``"0, 1"`` and ``"0,1"`` describe the same experiment. Both
        the recording half (``trainer``) and the comparing half
        (``gate.already_current``) read this one property, so a re-run can never
        disagree with itself about which device trained an artifact.
        """

        raw = (self.device or "").strip()
        if not raw:
            return "auto"
        return ",".join(part.strip().lower() for part in raw.split(","))
