"""Pipeline orchestration: preflight -> plan -> train/export/publish -> provenance.

Responsibilities are deliberately narrow and each one is a named function so the
flow reads top-down in :func:`run`:

1. validate the environment (:mod:`cvtrain.preflight`), fail fast with an
   actionable message;
2. decide whether to proceed (``--check`` / ``--dry-run`` / retrain prompt);
3. for every selected model train -> export -> publish -> record provenance;
4. verify the manifest it just wrote.

The heavy ML framework is reached only through :class:`TrainingBackend
<cvtrain.backend.TrainingBackend>`, injected as ``backend``. Production passes
``None`` and gets :class:`UltralyticsBackend`; tests pass a fake and run the
whole pipeline offline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from . import __version__, preflight
from . import manifest as manifest_mod
from .backend import EpochCallback, TrainedModel, TrainingBackend, UltralyticsBackend
from .config import BASE_DIR, MODEL_SPECS, ModelSpec, TrainConfig
from .errors import ExitCode
from .fsio import atomic_copy, sha256_file
from .logging_setup import ProgressReporter

LOG = logging.getLogger("cvtrain.trainer")

__all__ = ["confirm_retrain", "print_plan", "run"]


def print_plan(cfg: TrainConfig) -> None:
    """Echo the resolved configuration so a dry run can be reviewed by eye."""

    LOG.info("Training plan:")
    LOG.info("  dataset : %s", cfg.data_yaml)
    LOG.info("  models  : %s", ", ".join(cfg.models))
    LOG.info("  epochs  : %d | imgsz: %d | batch: %d | seed: %d", cfg.epochs, cfg.imgsz, cfg.batch, cfg.seed)
    LOG.info("  device  : %s", cfg.device or "auto")
    LOG.info("  runs    : %s", cfg.runs_dir)
    LOG.info("  output  : %s", cfg.models_dir)


def confirm_retrain(cfg: TrainConfig) -> bool:
    """Ask before overwriting models when running interactively.

    The desktop app always passes ``--yes`` so this prompt is only ever shown
    for manual terminal runs. When stdin is not a TTY (CI, piped input) we
    proceed without prompting rather than hang.
    """

    if cfg.auto_confirm or cfg.check_only or cfg.dry_run:
        return True
    existing = [MODEL_SPECS[key].output for key in cfg.models if cfg.output_path(MODEL_SPECS[key]).exists()]
    if not existing:
        return True
    if not sys.stdin.isatty():
        LOG.info("Existing models detected (%s); proceeding without prompt (no TTY).", ", ".join(existing))
        return True
    answer = input(f"Models already exist ({', '.join(existing)}). Retrain? Y/N: ").strip().upper()
    return answer == "Y"


def _run_preflight(cfg: TrainConfig, reporter: ProgressReporter, backend: TrainingBackend) -> ExitCode | None:
    """Log every check; return an exit code when a fatal one failed."""

    report = preflight.run(
        cfg,
        deps_fatal=not cfg.dry_run,
        modules=backend.required_modules,
    )
    for line in report.lines():
        LOG.info("%s", line)

    if not report.ok:
        for check in report.failures:
            LOG.error("Preflight failure -> %s", check.line())
        return ExitCode.PREFLIGHT

    reporter.preflight(1.0)
    return None


def _start_manifest(cfg: TrainConfig, backend_version: str) -> manifest_mod.Manifest:
    """Open the manifest and stamp the header describing this invocation."""

    manifest = manifest_mod.load_manifest(cfg.models_dir)
    manifest["schema_version"] = manifest_mod.SCHEMA_VERSION
    manifest["generator"] = f"cvtrain {__version__}"
    # Key name is historical: it records whichever backend produced the run.
    manifest["ultralytics"] = backend_version
    manifest["last_run"] = manifest_mod.LastRun(
        started_at=manifest_mod.now_utc_iso(),
        dataset=str(cfg.data_yaml),
        models=list(cfg.models),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        seed=cfg.seed,
        opset=cfg.opset,
    )
    return manifest


def _epoch_reporter(reporter: ProgressReporter, index: int, key: str) -> EpochCallback:
    """Bind a backend epoch callback to this model's slot in the overall bar."""

    def _on_epoch(epoch: int, total: int) -> None:
        reporter.train_epoch(index, epoch, total, key)

    return _on_epoch


def _export_one(
    spec: ModelSpec,
    cfg: TrainConfig,
    reporter: ProgressReporter,
    index: int,
    model: TrainedModel[Any],
    backend: TrainingBackend,
) -> Path:
    """Export through the backend, then publish atomically at ``spec.output``."""

    reporter.exporting(index, spec.key)
    generated = backend.export(model, cfg)
    target = cfg.output_path(spec)
    atomic_copy(generated, target)
    reporter.exported(index, spec.key)
    LOG.info("Exported %s -> %s (%.1f MB)", spec.ckpt, target, target.stat().st_size / (1 << 20))
    return target


def _record_entry(
    manifest: manifest_mod.Manifest,
    spec: ModelSpec,
    cfg: TrainConfig,
    target: Path,
    model: TrainedModel[Any],
) -> None:
    """Append this model's provenance and flush the manifest immediately.

    Flushing per model (not at the end) means a crash during model 3 of 3 still
    leaves a verified manifest for models 1 and 2.
    """

    manifest_mod.upsert_entry(
        manifest,
        spec.output,
        manifest_mod.ManifestEntry(
            role=spec.role,
            source_weights=spec.ckpt,
            sha256=sha256_file(target),
            size_bytes=target.stat().st_size,
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            seed=cfg.seed,
            opset=cfg.opset,
            trained_at=manifest_mod.now_utc_iso(),
            dataset=str(cfg.data_yaml),
            run_dir=str(model.run_dir),
            metrics=model.metrics,
        ),
    )
    manifest_mod.save_manifest(cfg.models_dir, manifest)


def _train_all(
    cfg: TrainConfig,
    reporter: ProgressReporter,
    backend: TrainingBackend,
    manifest: manifest_mod.Manifest,
) -> list[str]:
    """Run train -> export -> publish -> record for every selected model."""

    total = len(cfg.models)
    for index, key in enumerate(cfg.models):
        spec = MODEL_SPECS[key]
        LOG.info("[%d/%d] Training %s from %s", index + 1, total, spec.label, spec.ckpt)
        model = backend.train(spec, cfg, _epoch_reporter(reporter, index, spec.key))
        target = _export_one(spec, cfg, reporter, index, model, backend)
        _record_entry(manifest, spec, cfg, target, model)
    return [MODEL_SPECS[key].output for key in cfg.models]


def _execute(cfg: TrainConfig, reporter: ProgressReporter, backend: TrainingBackend | None) -> ExitCode:
    # Resolved before preflight: even a ``--check`` run has to know which
    # framework it is checking for. The constructor never imports anything.
    active = backend if backend is not None else UltralyticsBackend()

    failed = _run_preflight(cfg, reporter, active)
    if failed is not None:
        return failed

    if cfg.check_only:
        LOG.info("Environment is ready for training.")
        return ExitCode.OK

    if cfg.dry_run:
        print_plan(cfg)
        LOG.info("Dry run complete (no training performed).")
        return ExitCode.OK

    if not confirm_retrain(cfg):
        LOG.info("Retraining cancelled by user.")
        return ExitCode.OK

    active.prepare(cfg)

    manifest = _start_manifest(cfg, active.version)
    exported = _train_all(cfg, reporter, active, manifest)
    reporter.manifest(1.0)

    problems = manifest_mod.verify_manifest(cfg.models_dir, manifest)
    for problem in problems:
        LOG.warning("manifest: %s", problem)
    if not problems:
        LOG.info("Manifest verified: %s", cfg.manifest_path)

    LOG.info("Training complete. Exported: %s", ", ".join(exported))
    LOG.info("Base directory: %s", BASE_DIR)
    return ExitCode.OK


def run(cfg: TrainConfig, *, backend: TrainingBackend | None = None) -> int:
    """Execute the pipeline. Returns a process exit code (see :class:`ExitCode`).

    ``backend`` exists for tests; production code leaves it ``None``.
    """

    reporter = ProgressReporter(cfg.models)
    try:
        code = _execute(cfg, reporter, backend)
    except BaseException:
        # Always close the progress protocol, otherwise the Qt dock is left
        # waiting on a marker that will never arrive.
        reporter.error()
        raise

    if code == ExitCode.OK:
        reporter.done()
    else:
        reporter.error()
    return int(code)
