"""Command line interface for the AIComputerVision training pipeline.

Kept as a thin wrapper so the legacy entry point (``TrainModel.py``, referenced
by ``AppConstants::AIProcessor::trainScript()``) keeps working unchanged.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import __version__, trainer
from . import manifest as manifest_mod
from .config import (
    ALL_MODEL_KEYS,
    DEFAULT_BATCH,
    DEFAULT_DATA_YAML,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_MODELS,
    DEFAULT_MODELS_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEED,
    DEFAULT_WORKERS,
    MODEL_SPECS,
    TrainConfig,
    resolve_models,
)
from .errors import CvTrainError, ExitCode
from .logging_setup import setup_logging

LOG = logging.getLogger("cvtrain.cli")


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
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset for export")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="export directory")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="training log directory")
    parser.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan without training")
    parser.add_argument("--check", dest="check_only", action="store_true", help="run preflight only and exit")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="scan the models directory and (re)write manifest.json without training",
    )
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="verify manifest checksums and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        models=resolve_models(args.models or ()),
        data_yaml=Path(args.data),
        models_dir=Path(args.models_dir),
        runs_dir=Path(args.runs_dir),
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
    )


def _write_manifest(models_dir: Path) -> int:
    manifest = manifest_mod.build_from_scan(models_dir)
    path = manifest_mod.save_manifest(models_dir, manifest)
    count = len(manifest.get("entries") or {})
    LOG.info("Manifest written: %s (%d entries)", path, count)
    return int(ExitCode.OK)


def _verify_manifest(models_dir: Path) -> int:
    manifest_path = models_dir / manifest_mod.MANIFEST_NAME
    if not manifest_path.exists():
        LOG.error("Manifest missing: %s", manifest_path)
        return int(ExitCode.PREFLIGHT)
    manifest = manifest_mod.load_manifest(models_dir)
    problems = manifest_mod.verify_manifest(models_dir, manifest)
    if not problems:
        LOG.info("Manifest OK: %s", manifest_path)
        return int(ExitCode.OK)
    for problem in problems:
        LOG.error("manifest: %s", problem)
    return int(ExitCode.PREFLIGHT)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.write_manifest:
        return _write_manifest(Path(args.models_dir))
    if args.verify_manifest:
        return _verify_manifest(Path(args.models_dir))

    cfg = config_from_args(args)
    try:
        return int(trainer.run(cfg))
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
