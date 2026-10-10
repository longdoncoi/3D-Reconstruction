"""Central configuration and the C++/Python model contract.

This module is the single source of truth for:

* where datasets, training runs and exported models live, and
* which base weights map to which exported ONNX file name.

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
DEFAULT_DATA_YAML = PROJECT_ROOT / "Dataset" / "data.yaml"
MANIFEST_NAME = "manifest.json"

DEFAULT_EPOCHS = 5
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_WORKERS = 0
DEFAULT_SEED = 0


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

    @property
    def weights_path(self) -> Path:
        """Absolute path of the base checkpoint this model trains from.

        Single way to resolve a checkpoint: preflight, the backend and any test
        must agree on it, so it is derived from :data:`BASE_DIR` in one place.
        """

        return BASE_DIR / self.ckpt


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

    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_IMGSZ
    batch: int = DEFAULT_BATCH
    workers: int = DEFAULT_WORKERS
    seed: int = DEFAULT_SEED
    device: str | None = None
    opset: int = 12

    auto_confirm: bool = False
    tensorboard: bool = False
    dry_run: bool = False
    check_only: bool = False
    verbose: bool = False

    def output_path(self, spec: ModelSpec) -> Path:
        """Where ``spec``'s exported ONNX is published for this invocation."""

        return self.models_dir / spec.output

    @property
    def manifest_path(self) -> Path:
        """Absolute path of the model manifest for this invocation."""

        return self.models_dir / MANIFEST_NAME

    def train_kwargs(self) -> dict[str, object]:
        """Keyword arguments forwarded to ``YOLO.train``."""

        kwargs: dict[str, object] = {
            "data": str(self.data_yaml),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "workers": self.workers,
            "seed": self.seed,
            "deterministic": True,
            "exist_ok": False,
            "verbose": self.verbose,
        }
        if self.device:
            kwargs["device"] = self.device
        return kwargs
