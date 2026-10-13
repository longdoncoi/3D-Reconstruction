"""Pipeline orchestration: preflight -> plan -> train/export/publish -> provenance.

Responsibilities are deliberately narrow and each one is a named function so the
flow reads top-down in :func:`run`:

1. validate the environment and the hyper-parameters
   (:mod:`cvtrain.preflight`), fail fast with an actionable message;
2. decide whether to proceed (``--check`` / ``--dry-run`` / retrain prompt);
3. for every selected model train -> gate the metrics -> export -> publish ->
   record provenance;
4. verify the manifest it just wrote.

The heavy ML framework is reached only through :class:`TrainingBackend
<cvtrain.backend.TrainingBackend>`, which the *composition root* —
:func:`cvtrain.cli.main` — constructs and injects as ``backend``. This module
therefore never names a concrete backend; tests pass a fake and run the whole
pipeline offline.

The two *verdicts* this flow obeys — "is the published artifact already what we
would build" and "is it good enough to keep" — are not decided here: they come
from :mod:`cvtrain.gate`, so the orchestrator orchestrates and the policy stays
in one place.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from . import gate, generator_name, preflight
from . import manifest as manifest_mod
from .backend import EpochCallback, TrainedModel, TrainingBackend
from .config import BASE_DIR, MODEL_SPECS, ModelSpec, TrainConfig
from .errors import ExitCode
from .fsio import atomic_replace_with_backup, directory_lock, file_size, sha256_file
from .progress import ProgressReporter

LOG = logging.getLogger("cvtrain.trainer")

__all__ = ["confirm_retrain", "print_plan", "run"]


def print_plan(cfg: TrainConfig) -> None:
    """Echo the resolved configuration so a dry run can be reviewed by eye."""

    LOG.info("Training plan:")
    LOG.info("  dataset : %s", cfg.data_yaml)
    LOG.info("  models  : %s", ", ".join(cfg.models))
    LOG.info("  epochs  : %d | imgsz: %d | batch: %d | seed: %d", cfg.epochs, cfg.imgsz, cfg.batch, cfg.seed)
    LOG.info("  device  : %s", cfg.device or "auto")
    gate = ", ".join(f"{name}>={minimum:g}" for name, minimum in cfg.min_metrics) or "none"
    LOG.info("  gate    : %s", gate)
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
    try:
        answer = input(f"Models already exist ({', '.join(existing)}). Retrain? Y/N: ").strip().upper()
    except EOFError:
        LOG.info("Retraining cancelled by user (no input).")
        return False
    return answer == "Y"


def _run_preflight(cfg: TrainConfig, reporter: ProgressReporter, backend: TrainingBackend[Any]) -> ExitCode | None:
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
    # ``load_manifest`` hands back a document already migrated to the current
    # schema (and never down-stamps a newer one), so only the run header is
    # stamped here.
    started_at = manifest_mod.now_utc_iso()
    manifest["generator"] = generator_name()
    # Key name is historical: it records whichever backend produced the run.
    manifest["ultralytics"] = backend_version
    manifest["last_run"] = manifest_mod.LastRun(
        started_at=started_at,
        dataset=str(cfg.data_yaml),
        models=list(cfg.models),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        seed=cfg.seed,
        opset=cfg.opset,
        batch=cfg.batch,
        workers=cfg.workers,
        device=cfg.device_class,
        python=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        min_metrics=[[name, minimum] for name, minimum in cfg.min_metrics],
    )
    manifest["updated_at"] = started_at
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
    backend: TrainingBackend[Any],
) -> Path:
    """Export through the backend, then publish atomically at ``spec.output``."""

    reporter.exporting(index, spec.key)
    generated = backend.export(model, cfg)
    # Publish atomically, keeping the artifact we are about to replace at
    # ``<name>.onnx.bak``: a bad export must be recoverable without retraining.
    target = atomic_replace_with_backup(generated, cfg.output_path(spec))
    reporter.exported(index, spec.key)
    LOG.info("Exported %s -> %s (%.1f MB)", spec.ckpt, target, file_size(target) / (1 << 20))
    return target


def _record_entry(
    manifest: manifest_mod.Manifest,
    spec: ModelSpec,
    cfg: TrainConfig,
    target: Path,
    model: TrainedModel[Any],
    backend_version: str,
) -> None:
    """Append this model's provenance and flush the manifest immediately.

    Flushing per model (not at the end) means a crash during model 3 of 3 still
    leaves a verified manifest for models 1 and 2.
    """

    trained_at = manifest_mod.now_utc_iso()
    entry = manifest_mod.ManifestEntry(
        role=spec.role,
        source_weights=spec.ckpt,
        sha256=sha256_file(target),
        size_bytes=file_size(target),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        seed=cfg.seed,
        opset=cfg.opset,
        batch=cfg.batch,
        workers=cfg.workers,
        backend_version=backend_version,
        # Normalised exactly the way ``gate.already_current`` compares it: "no
        # ``--device``" and "``--device auto``" are the same experiment, and
        # ``"0, 1"`` and ``"0,1"`` are the same too (see ``device_class``).
        device=cfg.device_class,
        trained_at=trained_at,
        dataset=str(cfg.data_yaml),
        run_dir=str(model.run_dir),
        metrics=model.metrics,
    )
    # Both digests are taken through the gate module so that what is written
    # down here and compared there can never disagree. Absent when the file is
    # not on disk: leaving the field out is honest, and ``gate.already_current``
    # re-reads that as "unproven" and retrains.
    weights_sha = gate.weights_digest(spec, cfg)
    if weights_sha is not None:
        entry["source_weights_sha256"] = weights_sha
    dataset_sha = gate.dataset_digest(cfg)
    if dataset_sha is not None:
        entry["dataset_sha256"] = dataset_sha
    manifest_mod.upsert_entry(manifest, spec.output, entry)
    # ``updated_at`` tracks the last provenance write: without this refresh a
    # training run leaves the header older than the entries it just published.
    manifest["updated_at"] = trained_at
    manifest_mod.save_manifest(cfg.models_dir, manifest)


def _train_all(
    cfg: TrainConfig,
    reporter: ProgressReporter,
    backend: TrainingBackend[Any],
    manifest: manifest_mod.Manifest,
) -> list[str]:
    """Run train -> gate -> export -> publish -> record for every selected model.

    The quality gate sits *between* training and publishing: nothing is written
    to ``Models/`` until the metrics clear the requested bar, and a model kept
    by ``--skip-unchanged`` is gated against the metrics recorded when it was
    trained rather than waved through.
    """

    total = len(cfg.models)
    for index, key in enumerate(cfg.models):
        spec = MODEL_SPECS[key]
        if cfg.skip_unchanged and gate.already_current(spec, cfg, manifest, backend.version):
            LOG.info("[%d/%d] %s already matches this invocation; skipping", index + 1, total, spec.output)
            kept = (manifest.get("entries") or {}).get(spec.output) or {}
            gate.enforce_quality_gate(cfg, spec, kept.get("metrics") or {})
            reporter.skipped(index, spec.key)
            continue
        LOG.info("[%d/%d] Training %s from %s", index + 1, total, spec.label, spec.ckpt)
        model = backend.train(spec, cfg, _epoch_reporter(reporter, index, spec.key))
        gate.enforce_quality_gate(cfg, spec, model.metrics)
        target = _export_one(spec, cfg, reporter, index, model, backend)
        _record_entry(manifest, spec, cfg, target, model, backend.version)
    return [MODEL_SPECS[key].output for key in cfg.models]


def _execute(cfg: TrainConfig, reporter: ProgressReporter, backend: TrainingBackend[Any]) -> ExitCode:
    # Resolved before preflight: even a ``--check`` run has to know which
    # framework it is checking for. The adapter's constructor never imports
    # anything, so building it is free — but *choosing* it is not this module's
    # business; ``cli.main`` is the composition root.

    failed = _run_preflight(cfg, reporter, backend)
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

    # Everything that reads-then-writes ``Models/`` runs under the advisory
    # lock: a second run aimed at the same directory is refused here instead of
    # silently overwriting this run's provenance in ``manifest.json``.
    with directory_lock(cfg.models_dir):
        backend.prepare(cfg)

        manifest = _start_manifest(cfg, backend.version)
        exported = _train_all(cfg, reporter, backend, manifest)

        # The artifacts this run just published have to be vouchable. This is the
        # same condition ``--verify-manifest`` reports, and that command exits 2 —
        # so the verdict is raised, not returned: ``run`` closes the progress
        # protocol with the terminal error marker (``done`` is reserved for a run
        # that actually verified) and the composition root turns the failure into
        # the same exit code an operator would get from the standalone command.
        # Only the models this invocation is responsible for are in scope: a stray
        # file already sitting in ``Models/`` must not fail an otherwise clean run
        # (``--verify-manifest`` still accounts for it).
        manifest_mod.verify_or_raise(cfg.models_dir, manifest, only=exported)

        reporter.manifest(1.0)
        LOG.info("Manifest verified: %s", cfg.manifest_path)
        LOG.info("Training complete. Exported: %s", ", ".join(exported))

    LOG.info("Base directory: %s", BASE_DIR)
    return ExitCode.OK


def run(cfg: TrainConfig, *, backend: TrainingBackend[Any]) -> int:
    """Execute the pipeline. Returns a process exit code (see :class:`ExitCode`).

    ``backend`` is required and injected by the composition root
    (:func:`cvtrain.cli.main`) — the only place in the package allowed to name
    a concrete implementation. Taking it as a required argument keeps this
    module free of a default it would have to invent (and of the import that
    default would need).
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
