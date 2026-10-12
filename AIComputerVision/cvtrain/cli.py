"""Command line interface for the AIComputerVision training pipeline.

Kept as a thin wrapper so the legacy entry point (``TrainModel.py``, referenced
by ``AppConstants::AIProcessor::trainScript()``) keeps working unchanged.

This module is also the package's **composition root**: it is the only place
that names a concrete :class:`~cvtrain.backend.TrainingBackend` and hands it to
:mod:`cvtrain.trainer`, so the orchestrator stays free of framework choices.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import __version__, trainer
from . import manifest as manifest_mod
from .backend import UltralyticsBackend
from .config import (
    ALL_MODEL_KEYS,
    DEFAULT_BATCH,
    DEFAULT_DATA_YAML,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_MODELS,
    DEFAULT_MODELS_DIR,
    DEFAULT_OPSET,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEED,
    DEFAULT_WEIGHTS_DIR,
    DEFAULT_WORKERS,
    MANIFEST_NAME,
    MODEL_SPECS,
    TrainConfig,
    resolve_models,
)
from .errors import CvTrainError, ExitCode, ManifestIntegrityError
from .fsio import directory_lock
from .logging_setup import setup_logging

LOG = logging.getLogger("cvtrain.cli")


def parse_metric_threshold(raw: str) -> tuple[str, float]:
    """Parse a ``--min-metric`` value (``NAME=VALUE``) into a gate pair.

    Only *syntax* is checked here — there must be a name and something that
    converts to a number. The *semantics* (a finite threshold inside ``0..1``)
    belong to :mod:`cvtrain.preflight`, so a programmatically built
    :class:`~cvtrain.config.TrainConfig` is validated by exactly the same rules
    as one that came from a command line, in one place.

    Raises ``ValueError`` with an actionable message; :func:`_min_metric` adapts
    that to an ``argparse`` usage error (exit code ``2``).
    """

    name, separator, value = raw.partition("=")
    name = name.strip()
    if not separator or not name:
        raise ValueError(f"expected NAME=VALUE (e.g. 'metrics/mAP50(B)=0.5'), got {raw!r}")
    try:
        threshold = float(value)
    except ValueError as exc:
        raise ValueError(f"{name}: {value!r} is not a number") from exc
    return name, threshold


def _min_metric(raw: str) -> tuple[str, float]:
    """argparse adapter for ``--min-metric NAME=VALUE``.

    Only turns a syntax problem into an ``argparse`` usage error (exit ``2``,
    message preserved). The semantic rules live in preflight, so a gate that
    makes no sense is reported the same way whether it came from this command
    line or from a programmatically built :class:`~cvtrain.config.TrainConfig`.
    """

    try:
        return parse_metric_threshold(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="TrainModel.py",
        description="Train and export YOLO detection/segmentation/tracking models to ONNX.",
    )
    parser.add_argument("--version", action="version", version=f"cvtrain {__version__}")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Non-interactive: never prompt (used by the desktop app).",
    )

    # Exactly one mode. Letting two of these combine would give one of them
    # silent precedence (`--check --dry-run` used to answer only the first,
    # `--write-manifest --verify-manifest` used to overwrite and then claim the
    # manifest it had just rewritten verified), which is the same class of
    # surprise the preflight layer exists to remove.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", dest="check_only", action="store_true", help="run preflight only and exit")
    mode.add_argument("--dry-run", action="store_true", help="validate and print the plan without training")
    mode.add_argument(
        "--write-manifest",
        action="store_true",
        help="scan the models directory and (re)write manifest.json without training",
    )
    mode.add_argument(
        "--verify-manifest",
        action="store_true",
        help="verify manifest checksums and exit",
    )

    selection = parser.add_argument_group(
        "model selection",
        f"select one or more; with no flag the default set is trained ({', '.join(DEFAULT_MODELS)})",
    )
    for key in ALL_MODEL_KEYS:
        spec = MODEL_SPECS[key]
        selection.add_argument(
            f"--{key}",
            dest="models",
            action="append_const",
            const=key,
            help=f"train {spec.label} -> {spec.output}",
        )

    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help="dataset YAML file")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="training epochs")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="training image size")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="batch size")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed for reproducibility")
    parser.add_argument("--device", default=None, help="compute device, e.g. cpu, 0 or 0,1")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset for export")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="export directory")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="training log directory")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=DEFAULT_WEIGHTS_DIR,
        help="directory holding the base .pt checkpoints",
    )
    parser.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    parser.add_argument(
        "--skip-unchanged",
        action="store_true",
        help="skip models whose ONNX already matches this invocation (idempotent re-run)",
    )
    quality = parser.add_argument_group(
        "quality gate",
        "publish only models whose metrics meet a minimum; repeatable, and applied to a "
        "model skipped by --skip-unchanged as well",
    )
    quality.add_argument(
        "--min-metric",
        action="append",
        type=_min_metric,
        default=None,
        metavar="NAME=VALUE",
        help="require metric NAME >= VALUE after training, e.g. --min-metric 'metrics/mAP50(B)=0.5'",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        models=resolve_models(args.models or ()),
        data_yaml=Path(args.data),
        models_dir=Path(args.models_dir),
        runs_dir=Path(args.runs_dir),
        weights_dir=Path(args.weights_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
        opset=args.opset,
        auto_confirm=args.yes,
        tensorboard=args.tensorboard,
        dry_run=args.dry_run,
        check_only=args.check_only,
        verbose=args.verbose,
        skip_unchanged=args.skip_unchanged,
        min_metrics=tuple(args.min_metric or ()),
    )


def _write_manifest(models_dir: Path) -> int:
    # The same advisory lock a training run takes: adopting models while another
    # run trains would rewrite ``manifest.json`` out from under it.
    with directory_lock(models_dir):
        manifest = manifest_mod.build_from_scan(models_dir)
        path = manifest_mod.save_manifest(models_dir, manifest)
    count = len(manifest.get("entries") or {})
    LOG.info("Manifest written: %s (%d entries)", path, count)
    return int(ExitCode.OK)


def _verify_manifest(models_dir: Path) -> int:
    manifest_path = models_dir / MANIFEST_NAME
    if not manifest_path.exists():
        # Not delegated to ``verify_or_raise``: an empty directory verifies
        # clean against a skeleton manifest, and "there is nothing here" is a
        # different (still fatal) verdict than "these artifacts are corrupt".
        raise ManifestIntegrityError(f"Manifest missing: {manifest_path}")
    manifest = manifest_mod.load_manifest(models_dir)
    # One spelling of the verdict, shared with the post-training check: any
    # problem becomes a ManifestIntegrityError carrying exit code 2, rendered
    # by main() like every other expected failure.
    manifest_mod.verify_or_raise(models_dir, manifest)
    LOG.info("Manifest OK: %s", manifest_path)
    return int(ExitCode.OK)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    try:
        # Every mode runs inside the same failure taxonomy. The manifest
        # commands used to return early, so an I/O error there escaped as a
        # traceback with an undeclared code while the identical problem during
        # training was one typed line — one boundary, one rule.
        if args.write_manifest:
            return _write_manifest(Path(args.models_dir))
        if args.verify_manifest:
            return _verify_manifest(Path(args.models_dir))

        cfg = config_from_args(args)
        # Composition root: the orchestrator is handed a backend, never left to
        # invent one. Constructing the adapter is free — it imports nothing
        # until it is actually asked to prepare/train/export.
        return int(trainer.run(cfg, backend=UltralyticsBackend()))
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user.")
        return int(ExitCode.ABORTED)
    except CvTrainError as exc:
        # Expected, already-explained failure: one actionable line, no traceback.
        LOG.error("%s", exc)
        return int(exc.exit_code)
    except Exception as exc:
        LOG.exception("Training failed: %s", exc)
        return int(ExitCode.FAILURE)
