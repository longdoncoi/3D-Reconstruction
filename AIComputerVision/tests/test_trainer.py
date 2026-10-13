"""Unit tests for the pipeline orchestrator.

``trainer`` is the module that decides *what happens* — preflight, dry-run, the
retrain prompt, per-model train/export/publish, manifest durability and the
progress protocol. It is exercised end-to-end here against a fake backend, so
the whole flow runs offline in milliseconds and without the ML stack.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from cvtrain import manifest as manifest_mod
from cvtrain.backend import TrainedModel, UltralyticsBackend
from cvtrain.checks import module_available
from cvtrain.config import MANIFEST_NAME, MODEL_SPECS, TrainConfig
from cvtrain.errors import ConcurrentRunError, ExitCode, ManifestIntegrityError, QualityGateError
from cvtrain.fsio import BACKUP_SUFFIX, LOCK_NAME, sha256_file
from cvtrain.progress import PROGRESS_PREFIX
from cvtrain.trainer import confirm_retrain, run

VALID_DATASET = "train: images/train\nval: images/val\nnames:\n  0: part\n"


class FakeBackend:
    """Deterministic stand-in for :class:`cvtrain.backend.UltralyticsBackend`."""

    version = "9.9.9-fake"

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on
        self.prepared: TrainConfig | None = None

    def required_modules(self, cfg: TrainConfig) -> tuple[str, ...]:
        # The fake drives preflight through the *backend-owned* dependency list
        # without asking for any framework module: the offline suite must not
        # need the ML stack to exercise this path.
        return ()

    def prepare(self, cfg: TrainConfig) -> None:
        self.prepared = cfg
        self.calls.append("prepare")

    def train(self, spec: Any, cfg: TrainConfig, on_epoch: Any) -> TrainedModel[Any]:
        self.calls.append(f"train:{spec.key}")
        if self.fail_on == spec.key:
            raise RuntimeError(f"backend exploded on {spec.key}")
        for epoch in range(cfg.epochs):
            on_epoch(epoch, cfg.epochs)
        run_dir = cfg.runs_dir / spec.runs_subdir / "train-test"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "args.yaml").write_text(f"epochs: {cfg.epochs}\n", encoding="utf-8")
        return TrainedModel(metrics={"metrics/mAP50(B)": 0.42}, run_dir=run_dir, handle=spec.key)

    def export(self, model: TrainedModel[Any], cfg: TrainConfig) -> Path:
        self.calls.append(f"export:{model.handle}")
        generated = cfg.runs_dir / f"{model.handle}.onnx"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"ONNX:" + str(model.handle).encode())
        return generated


def _config(root: Path, **overrides: Any) -> TrainConfig:
    """Build a config whose dataset always passes preflight.

    Preflight now fails fast when a referenced ``train``/``val`` directory is
    missing, so the sandbox must contain those directories alongside the YAML.

    The base checkpoints are materialised in a sandbox ``weights_dir`` too:
    ``_already_current`` records and re-checks their digest, so proving what a
    run trained from has to work without ever touching the real
    ``AIComputerVision/*.pt``.
    """

    data_yaml = root / "data.yaml"
    data_yaml.write_text(VALID_DATASET, encoding="utf-8")
    for name in ("images/train", "images/val"):
        (root / name).mkdir(parents=True, exist_ok=True)
    weights_dir = root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    for spec in MODEL_SPECS.values():
        (weights_dir / spec.ckpt).write_bytes(f"base-weights:{spec.ckpt}".encode())
    kwargs: dict[str, Any] = {
        "models": ("det", "seg"),
        "data_yaml": data_yaml,
        "models_dir": root / "Models",
        "runs_dir": root / "runs",
        "weights_dir": weights_dir,
        "epochs": 3,
        "auto_confirm": True,
    }
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


class _SandboxedTestCase(unittest.TestCase):
    """Gives each test an isolated temp directory as ``self.root``."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class HappyPathTests(_SandboxedTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = _config(self.root)
        self.backend = FakeBackend()

    def test_succeeds_and_publishes_every_selected_model(self) -> None:
        code = run(self.cfg, backend=self.backend)
        self.assertEqual(code, ExitCode.OK)
        for key in self.cfg.models:
            target = self.cfg.output_path(MODEL_SPECS[key])
            self.assertTrue(target.exists(), f"{target} was not published")
            self.assertEqual(target.read_bytes(), b"ONNX:" + key.encode())

    def test_backend_is_prepared_exactly_once_before_any_training(self) -> None:
        run(self.cfg, backend=self.backend)
        self.assertIs(self.backend.prepared, self.cfg)
        self.assertEqual(
            self.backend.calls,
            ["prepare", "train:det", "export:det", "train:seg", "export:seg"],
        )

    def test_every_epoch_keeps_the_advisory_lock_alive(self) -> None:
        # The stale-lock rule ("abandoned after LOCK_STALE_SECONDS") is only
        # safe if a run that is still working refreshes its lock. Without this
        # heartbeat a single model training past the window would be mistaken
        # for a crash and a second writer would be let into Models/.
        with mock.patch("cvtrain.trainer.touch_lock") as touch:
            self.assertEqual(run(self.cfg, backend=self.backend), ExitCode.OK)
        self.assertEqual(touch.call_count, self.cfg.epochs * len(self.cfg.models))

    def test_manifest_records_verifiable_provenance(self) -> None:
        run(self.cfg, backend=self.backend)
        manifest = manifest_mod.load_manifest(self.cfg.models_dir)

        self.assertEqual(manifest["ultralytics"], "9.9.9-fake")
        self.assertEqual(manifest["last_run"]["models"], ["det", "seg"])
        self.assertEqual(manifest["last_run"]["epochs"], 3)
        self.assertEqual(manifest["last_run"]["batch"], 16)
        self.assertEqual(manifest["last_run"]["workers"], 0)
        self.assertEqual(manifest["last_run"]["device"], "auto")
        self.assertEqual(
            manifest["last_run"]["python"],
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
        self.assertEqual(manifest["last_run"]["min_metrics"], [])
        # The header tracks the last provenance write, never older than entries.
        self.assertIn("updated_at", manifest)
        for key in self.cfg.models:
            entry = manifest["entries"][MODEL_SPECS[key].output]
            self.assertLessEqual(entry["trained_at"] or "", manifest["updated_at"] or "")

        for key in self.cfg.models:
            spec = MODEL_SPECS[key]
            entry = manifest["entries"][spec.output]
            self.assertEqual(entry["role"], spec.role)
            self.assertEqual(entry["source_weights"], spec.ckpt)
            self.assertEqual(entry["metrics"], {"metrics/mAP50(B)": 0.42})
            self.assertEqual(entry["sha256"], sha256_file(self.cfg.output_path(spec)))
            # Everything `_already_current` compares has to be on disk, or a
            # re-run could not prove the published file is still what it would
            # build.
            self.assertEqual(entry["batch"], 16)
            self.assertEqual(entry["workers"], 0)
            self.assertEqual(entry["backend_version"], "9.9.9-fake")
            self.assertEqual(entry["source_weights_sha256"], sha256_file(self.cfg.weights_path(spec)))
            self.assertEqual(entry["dataset_sha256"], sha256_file(self.cfg.data_yaml))
            self.assertEqual(entry["device"], "auto")

        self.assertEqual(manifest_mod.verify_manifest(self.cfg.models_dir, manifest), [])

    def test_verify_manifest_detects_a_swapped_artifact(self) -> None:
        run(self.cfg, backend=self.backend)
        spec = MODEL_SPECS["det"]
        self.cfg.output_path(spec).write_bytes(b"tampered")
        manifest = manifest_mod.load_manifest(self.cfg.models_dir)
        problems = manifest_mod.verify_manifest(self.cfg.models_dir, manifest)
        self.assertTrue(any(spec.output in problem and "mismatch" in problem for problem in problems))

    def test_progress_protocol_reaches_100_and_closes(self) -> None:
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            run(self.cfg, backend=self.backend)
        lines = [line for line in captured.output if PROGRESS_PREFIX in line]
        self.assertTrue(lines, "no progress markers were emitted")
        self.assertIn("stage=preflight", lines[0])
        joined = "\n".join(lines)
        self.assertIn("stage=train", joined)
        self.assertIn("stage=export", joined)
        self.assertIn("stage=manifest", joined)
        self.assertEqual(lines[-1], f"INFO:cvtrain.progress:{PROGRESS_PREFIX} pct=100 stage=done")


class StageShortcutTests(_SandboxedTestCase):
    def test_check_only_never_touches_the_backend(self) -> None:
        cfg = _config(self.root, check_only=True)
        backend = FakeBackend()
        self.assertEqual(run(cfg, backend=backend), ExitCode.OK)
        self.assertEqual(backend.calls, [])

    def test_dry_run_prints_the_plan_and_never_trains(self) -> None:
        cfg = _config(self.root, dry_run=True)
        backend = FakeBackend()
        with self.assertLogs("cvtrain.trainer", level="INFO") as captured:
            self.assertEqual(run(cfg, backend=backend), ExitCode.OK)
        text = "\n".join(captured.output)
        self.assertIn("Training plan:", text)
        self.assertIn("Dry run complete", text)
        self.assertEqual(backend.calls, [])
        self.assertFalse(cfg.models_dir.exists())

    def test_declining_the_retrain_prompt_aborts_cleanly(self) -> None:
        cfg = _config(self.root, auto_confirm=False)
        existing = cfg.output_path(MODEL_SPECS["det"])
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_bytes(b"previous")

        interactive = mock.Mock()
        interactive.isatty.return_value = True
        backend = FakeBackend()
        with mock.patch.object(sys, "stdin", interactive), mock.patch("builtins.input", return_value="N"):
            self.assertEqual(run(cfg, backend=backend), ExitCode.OK)

        self.assertEqual(backend.calls, [])
        self.assertEqual(existing.read_bytes(), b"previous")


class FailurePathTests(_SandboxedTestCase):
    def test_preflight_failure_returns_2_and_closes_the_protocol(self) -> None:
        cfg = _config(self.root, data_yaml=self.root / "missing.yaml")
        backend = FakeBackend()
        with self.assertLogs("cvtrain.progress", level="INFO") as progress:
            with self.assertLogs("cvtrain.trainer", level="ERROR") as trainer_log:
                code = run(cfg, backend=backend)
        self.assertEqual(code, ExitCode.PREFLIGHT)
        self.assertEqual(backend.calls, [])
        self.assertIn("stage=error", progress.output[-1])
        # The operator gets the actionable reason, not just a code.
        self.assertIn("Preflight failure", trainer_log.output[0])
        self.assertIn("missing.yaml", trainer_log.output[0])

    def test_backend_crash_propagates_and_closes_the_protocol(self) -> None:
        cfg = _config(self.root)
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            with self.assertRaisesRegex(RuntimeError, "exploded on seg"):
                run(cfg, backend=FakeBackend(fail_on="seg"))
        self.assertIn("stage=error", captured.output[-1])

    def test_manifest_of_earlier_models_survives_a_later_crash(self) -> None:
        # Provenance is flushed per model, so a crash on model 2 still leaves a
        # verified manifest for model 1 instead of losing both.
        cfg = _config(self.root)
        with self.assertRaises(RuntimeError):
            run(cfg, backend=FakeBackend(fail_on="seg"))
        manifest = manifest_mod.load_manifest(cfg.models_dir)
        self.assertIn(MODEL_SPECS["det"].output, manifest["entries"])
        self.assertNotIn(MODEL_SPECS["seg"].output, manifest["entries"])
        self.assertEqual(manifest_mod.verify_manifest(cfg.models_dir, manifest), [])

    def test_tensorboard_without_its_package_fails_in_preflight(self) -> None:
        # The claim in README/ADR 0006 is "exit 2, before any training": here
        # it is proved through the *real* adapter, so the wiring
        # cfg -> required_modules(cfg) -> preflight is what is under test. Only
        # the tensorboard probe is forced to report missing, so the assertion
        # on the dependency line cannot be satisfied by some other check.
        cfg = _config(self.root, tensorboard=True)
        real = module_available
        with (
            mock.patch(
                "cvtrain.preflight.module_available",
                side_effect=lambda name: False if name == "tensorboard" else real(name),
            ),
            self.assertLogs("cvtrain.trainer", level="INFO") as captured,
        ):
            code = run(cfg, backend=UltralyticsBackend())

        self.assertEqual(code, ExitCode.PREFLIGHT)
        # The *failure* line, not any line: every check is also logged at INFO,
        # so a bare "dependency.tensorboard" match would be satisfied by
        # "[OK] dependency.tensorboard: installed" — i.e. by the wiring having
        # been removed entirely on an image that already ships the package.
        failures = [line for line in captured.output if "Preflight failure" in line and "dependency.tensorboard" in line]
        self.assertEqual(len(failures), 1)


class ManifestVerificationTests(_SandboxedTestCase):
    def test_verification_problems_fail_the_run(self) -> None:
        cfg = _config(self.root)
        with mock.patch(
            "cvtrain.trainer.manifest_mod.verify_manifest",
            return_value=["ghost.onnx: missing on disk"],
        ):
            # Raised, not returned: the orchestrator closes the protocol with
            # the terminal error marker, and the composition root turns this
            # into the same exit code (2) ``--verify-manifest`` reports for the
            # identical condition. An artifact we cannot vouch for must never
            # be presented to the Qt dock as a clean run.
            with self.assertRaises(ManifestIntegrityError) as ctx:
                run(cfg, backend=FakeBackend())

        self.assertIn("ghost.onnx: missing on disk", str(ctx.exception))
        self.assertEqual(ctx.exception.exit_code, ExitCode.PREFLIGHT)

    def test_a_stray_onnx_does_not_fail_a_clean_run(self) -> None:
        # End-of-run verification is scoped to the models just published, so a
        # file already sitting in Models/ is not this run's problem. The global
        # ``--verify-manifest`` command still reports it.
        cfg = _config(self.root)
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        (cfg.models_dir / "stray.onnx").write_bytes(b"not-ours")

        self.assertEqual(run(cfg, backend=FakeBackend()), ExitCode.OK)

        manifest = manifest_mod.load_manifest(cfg.models_dir)
        problems = manifest_mod.verify_manifest(cfg.models_dir, manifest)
        self.assertIn("stray.onnx: on disk but not recorded in the manifest", problems)


class ProvenanceRecordingTests(_SandboxedTestCase):
    """What a run writes down about the inputs it used."""

    def test_a_run_without_a_checkpoint_records_no_digest(self) -> None:
        # Provenance is never guessed: when the checkpoint is not on disk the
        # field is left out, which is what makes the next run retrain.
        cfg = _config(self.root, models=("det",))
        cfg.weights_path(MODEL_SPECS["det"]).unlink()

        run(cfg, backend=FakeBackend())

        entry = manifest_mod.load_manifest(cfg.models_dir)["entries"][MODEL_SPECS["det"].output]
        self.assertNotIn("source_weights_sha256", entry)
        self.assertEqual(entry["source_weights"], MODEL_SPECS["det"].ckpt)

    def test_the_recorded_digest_matches_the_checkpoint_actually_used(self) -> None:
        cfg = _config(self.root, models=("det",))
        cfg.weights_path(MODEL_SPECS["det"]).write_bytes(b"a-different-checkpoint")

        run(cfg, backend=FakeBackend())

        entry = manifest_mod.load_manifest(cfg.models_dir)["entries"][MODEL_SPECS["det"].output]
        self.assertEqual(entry["source_weights_sha256"], sha256_file(cfg.weights_path(MODEL_SPECS["det"])))

    def test_a_dataset_yanked_mid_run_records_no_digest(self) -> None:
        # Recording never guesses either: if the definition vanished between
        # preflight and publication the entry carries no digest, and the next
        # ``--skip-unchanged`` retracts its skip rather than assume the dataset
        # is the one that produced the artifact.
        cfg = _config(self.root, models=("det",))

        class _YanksTheDataset(FakeBackend):
            def train(self, spec: Any, config: TrainConfig, on_epoch: Any) -> TrainedModel[Any]:
                config.data_yaml.unlink()
                return super().train(spec, config, on_epoch)

        run(cfg, backend=_YanksTheDataset())

        entry = manifest_mod.load_manifest(cfg.models_dir)["entries"][MODEL_SPECS["det"].output]
        self.assertNotIn("dataset_sha256", entry)


class ConfirmRetrainTests(_SandboxedTestCase):
    def test_auto_confirm_short_circuits(self) -> None:
        cfg = _config(self.root, auto_confirm=True)
        with mock.patch("builtins.input", side_effect=AssertionError("prompted")):
            self.assertTrue(confirm_retrain(cfg))

    def test_no_existing_models_skips_the_prompt(self) -> None:
        cfg = _config(self.root, auto_confirm=False)
        with mock.patch("builtins.input", side_effect=AssertionError("prompted")):
            self.assertTrue(confirm_retrain(cfg))

    def test_non_tty_proceeds_without_blocking(self) -> None:
        cfg = _config(self.root, auto_confirm=False)
        target = cfg.output_path(MODEL_SPECS["det"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"old")

        piped = mock.Mock()
        piped.isatty.return_value = False
        with mock.patch.object(sys, "stdin", piped), mock.patch(
            "builtins.input", side_effect=AssertionError("prompted")
        ):
            self.assertTrue(confirm_retrain(cfg))

    def test_closed_stdin_cancels_instead_of_crashing(self) -> None:
        # A TTY whose input is closed raises EOFError: cancel cleanly (exit 0)
        # instead of a traceback with exit 1.
        cfg = _config(self.root, auto_confirm=False)
        target = cfg.output_path(MODEL_SPECS["det"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"old")

        interactive = mock.Mock()
        interactive.isatty.return_value = True
        with mock.patch.object(sys, "stdin", interactive), mock.patch(
            "builtins.input", side_effect=EOFError
        ):
            self.assertFalse(confirm_retrain(cfg))


class SkipUnchangedTests(_SandboxedTestCase):
    """``--skip-unchanged`` turns an identical re-run into a no-op."""

    def _run(self, **overrides: Any) -> tuple[int, FakeBackend]:
        """One pipeline run; returns its exit code and the backend that drove it."""

        cfg = _config(self.root, **overrides)
        backend = FakeBackend()
        return run(cfg, backend=backend), backend

    def test_retraining_stays_the_default(self) -> None:
        # Opt-in: without the flag a re-run must keep overwriting, which is what
        # the Qt dialog and every existing caller rely on.
        self._run(models=("det",))
        _, backend = self._run(models=("det",))
        self.assertIn("train:det", backend.calls)

    def test_an_identical_invocation_trains_nothing(self) -> None:
        self._run(models=("det",), skip_unchanged=True)
        _, backend = self._run(models=("det",), skip_unchanged=True)
        # ``prepare`` still applies run-wide settings; no model is trained or
        # exported, so the published artifact is never touched again.
        self.assertEqual(backend.calls, ["prepare"])

    def test_a_skipped_model_still_advances_the_progress_bar(self) -> None:
        self._run(models=("det",), skip_unchanged=True)
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            self._run(models=("det",), skip_unchanged=True)
        lines = [line for line in captured.output if PROGRESS_PREFIX in line]
        self.assertTrue(any("stage=skip" in line for line in lines), lines)
        self.assertTrue(lines[-1].endswith("pct=100 stage=done"), lines[-1])

    def test_a_missing_artifact_is_trained(self) -> None:
        _, backend = self._run(models=("det",), skip_unchanged=True)
        self.assertIn("train:det", backend.calls)

    def test_a_tampered_artifact_is_trained(self) -> None:
        cfg = _config(self.root, models=("det",), skip_unchanged=True)
        run(cfg, backend=FakeBackend())
        cfg.output_path(MODEL_SPECS["det"]).write_bytes(b"tampered")

        _, backend = self._run(models=("det",), skip_unchanged=True)
        self.assertIn("train:det", backend.calls)

    def test_an_artifact_without_a_manifest_entry_is_trained(self) -> None:
        # A file dropped into Models/ by hand carries provenance nobody
        # recorded; "it exists" is not evidence that it is the requested model.
        cfg = _config(self.root, models=("det",))
        target = cfg.output_path(MODEL_SPECS["det"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"hand-made")

        _, backend = self._run(models=("det",), skip_unchanged=True)
        self.assertIn("train:det", backend.calls)

    def test_different_hyper_parameters_are_trained(self) -> None:
        self._run(models=("det",), skip_unchanged=True)
        _, backend = self._run(models=("det",), skip_unchanged=True, epochs=4)
        self.assertIn("train:det", backend.calls)

    def test_an_entry_without_provenance_is_trained(self) -> None:
        # ``--write-manifest`` records identity + digest but no hyper-parameters,
        # so nothing proves the file is what this invocation is asking for.
        cfg = _config(self.root, models=("det",), skip_unchanged=True)
        run(cfg, backend=FakeBackend())
        manifest = manifest_mod.load_manifest(cfg.models_dir)
        stored = manifest["entries"][MODEL_SPECS["det"].output]
        manifest["entries"][MODEL_SPECS["det"].output] = manifest_mod.ManifestEntry(
            role=stored["role"],
            sha256=stored["sha256"],
            size_bytes=stored["size_bytes"],
        )
        manifest_mod.save_manifest(cfg.models_dir, manifest)

        _, backend = self._run(models=("det",), skip_unchanged=True)
        self.assertIn("train:det", backend.calls)

    def test_a_changed_batch_is_trained(self) -> None:
        # "The same invocation" includes the batch size: two runs that sliced
        # the data differently are not the same artifact, even though the file
        # name they would publish is.
        self._run(models=("det",), skip_unchanged=True)
        _, backend = self._run(models=("det",), skip_unchanged=True, batch=8)
        self.assertIn("train:det", backend.calls)

    def test_a_changed_worker_count_is_trained(self) -> None:
        self._run(models=("det",), skip_unchanged=True)
        _, backend = self._run(models=("det",), skip_unchanged=True, workers=4)
        self.assertIn("train:det", backend.calls)

    def test_a_changed_device_is_trained(self) -> None:
        # Same file name, same hyper-parameters — but a CPU run and a GPU run
        # are different artifacts, so "what this run would build" is no longer
        # what is published. ``None`` and ``auto`` are the same device.
        self._run(models=("det",), skip_unchanged=True)
        _, backend = self._run(models=("det",), skip_unchanged=True, device="cpu")
        self.assertIn("train:det", backend.calls)

    def test_an_unspecified_device_matches_an_explicit_auto(self) -> None:
        # "" / None / "auto" are the same experiment: "let the framework choose".
        self._run(models=("det",), skip_unchanged=True)
        _, backend = self._run(models=("det",), skip_unchanged=True, device="")
        self.assertNotIn("train:det", backend.calls)
        _, backend = self._run(models=("det",), skip_unchanged=True, device="auto")
        self.assertNotIn("train:det", backend.calls)

    def test_equivalent_device_spellings_are_not_retrained(self) -> None:
        # Provenance stores the canonical device, so "0, 1" and "0,1" describe
        # the same experiment and the second run is skipped.
        self._run(models=("det",), skip_unchanged=True, device="0,1")
        _, backend = self._run(models=("det",), skip_unchanged=True, device="0, 1")
        self.assertNotIn("train:det", backend.calls)

    def test_a_changed_dataset_definition_is_trained(self) -> None:
        # The path is unchanged but the content is not: an edited data.yaml is
        # a different experiment (renamed classes, moved splits) hiding behind
        # the same file name. Only the definition is digested — see
        # gate.dataset_digest for why the images themselves are not.
        # The config is built before the edit, because building it is what
        # (re)writes the sandbox YAML.
        self._run(models=("det",), skip_unchanged=True)
        cfg = _config(self.root, models=("det",), skip_unchanged=True)
        cfg.data_yaml.write_text(VALID_DATASET.replace("part", "widget"), encoding="utf-8")

        backend = FakeBackend()
        run(cfg, backend=backend)
        self.assertIn("train:det", backend.calls)

    def test_an_entry_without_a_recorded_dataset_digest_is_trained(self) -> None:
        # Written before the digest was recorded: nothing proves the dataset
        # is the one that produced the artifact, so the skip is retracted.
        cfg = _config(self.root, models=("det",), skip_unchanged=True)
        run(cfg, backend=FakeBackend())
        manifest = manifest_mod.load_manifest(cfg.models_dir)
        manifest["entries"][MODEL_SPECS["det"].output].pop("dataset_sha256", None)
        manifest_mod.save_manifest(cfg.models_dir, manifest)

        _, backend = self._run(models=("det",), skip_unchanged=True)
        self.assertIn("train:det", backend.calls)

    def test_a_backend_upgrade_is_trained(self) -> None:
        # A newer ultralytics exports differently, so the recorded artifact no
        # longer answers to "what this run would build".
        self._run(models=("det",), skip_unchanged=True)
        upgraded = FakeBackend()
        upgraded.version = "9.9.10-fake"
        run(_config(self.root, models=("det",), skip_unchanged=True), backend=upgraded)
        self.assertIn("train:det", upgraded.calls)

    def test_changed_base_weights_are_trained(self) -> None:
        # ``yolo11n.pt`` is a file name, not provenance: the checkpoint on disk
        # can be replaced by a fine-tuned one and produce a different model.
        # The config is built *before* the swap, because building it is what
        # materialises the sandbox checkpoints.
        self._run(models=("det",), skip_unchanged=True)
        cfg = _config(self.root, models=("det",), skip_unchanged=True)
        cfg.weights_path(MODEL_SPECS["det"]).write_bytes(b"someone-fine-tuned-this")

        backend = FakeBackend()
        run(cfg, backend=backend)
        self.assertIn("train:det", backend.calls)

    def test_missing_base_weights_are_trained(self) -> None:
        # No checkpoint on disk means nothing proves what the artifact started
        # from; "cannot verify" must never be read as "matches".
        self._run(models=("det",), skip_unchanged=True)
        cfg = _config(self.root, models=("det",), skip_unchanged=True)
        cfg.weights_path(MODEL_SPECS["det"]).unlink()

        backend = FakeBackend()
        run(cfg, backend=backend)
        self.assertIn("train:det", backend.calls)

    def test_an_entry_without_a_recorded_checkpoint_digest_is_trained(self) -> None:
        # `_already_current` has nothing to compare the checkpoint against, so
        # the absent digest is read as "unproven" and the model is retrained.
        cfg = _config(self.root, models=("det",), skip_unchanged=True)
        run(cfg, backend=FakeBackend())
        manifest = manifest_mod.load_manifest(cfg.models_dir)
        manifest["entries"][MODEL_SPECS["det"].output].pop("source_weights_sha256", None)
        manifest_mod.save_manifest(cfg.models_dir, manifest)

        _, backend = self._run(models=("det",), skip_unchanged=True)
        self.assertIn("train:det", backend.calls)


class QualityGateTests(_SandboxedTestCase):
    """``--min-metric`` keeps an unusable model out of ``Models/``."""

    def test_a_passing_gate_publishes_the_model(self) -> None:
        cfg = _config(self.root, models=("det",), min_metrics=(("metrics/mAP50(B)", 0.4),))
        with self.assertLogs("cvtrain.gate", level="INFO") as captured:
            code = run(cfg, backend=FakeBackend())

        self.assertEqual(code, ExitCode.OK)
        self.assertTrue(cfg.output_path(MODEL_SPECS["det"]).exists())
        self.assertTrue(any("Quality gate ok" in line for line in captured.output))
        last_run = manifest_mod.load_manifest(cfg.models_dir)["last_run"]
        self.assertEqual(last_run["min_metrics"], [["metrics/mAP50(B)", 0.4]])

    def test_a_model_below_the_threshold_is_never_published(self) -> None:
        # Checked *before* the export is published, so a model nobody wants
        # never lands where the Qt app loads it from. The run ends in the
        # terminal error marker, and the CLI renders the reason as one
        # actionable line (see test_cli's ExitCodeTests).
        cfg = _config(self.root, models=("det",), min_metrics=(("metrics/mAP50(B)", 0.5),))
        with self.assertLogs("cvtrain.progress", level="INFO") as progress:
            with self.assertRaises(QualityGateError) as ctx:
                run(cfg, backend=FakeBackend())

        self.assertIn("0.4200", str(ctx.exception))
        self.assertIn("0.5000", str(ctx.exception))
        self.assertFalse(cfg.output_path(MODEL_SPECS["det"]).exists())
        self.assertIn("stage=error", progress.output[-1])

    def test_the_gate_uses_its_own_exit_code(self) -> None:
        # 2 means "environment or artifact is untrustworthy"; 3 means "we
        # trained it and it is not good enough". The operator's next action
        # differs, so the codes must not be conflated.
        cfg = _config(self.root, models=("det",), min_metrics=(("metrics/mAP50(B)", 0.5),))
        with self.assertRaises(QualityGateError) as ctx:
            run(cfg, backend=FakeBackend())
        self.assertEqual(ctx.exception.exit_code, ExitCode.QUALITY)

    def test_a_metric_the_backend_never_reported_fails_the_run(self) -> None:
        # Silently not checking a metric the operator asked to gate on would
        # turn the gate into decoration; the message lists what was reported so
        # an easy typo is visible immediately.
        cfg = _config(self.root, models=("det",), min_metrics=(("metrics/mAP50-95(B)", 0.1),))
        with self.assertRaises(QualityGateError) as ctx:
            run(cfg, backend=FakeBackend())

        self.assertIn("was not reported", str(ctx.exception))
        self.assertIn("metrics/mAP50(B)", str(ctx.exception))

    def test_the_gate_also_applies_to_a_model_skip_unchanged_keeps(self) -> None:
        first = _config(self.root, models=("det",), skip_unchanged=True, min_metrics=(("metrics/mAP50(B)", 0.4),))
        self.assertEqual(run(first, backend=FakeBackend()), ExitCode.OK)

        stricter = _config(self.root, models=("det",), skip_unchanged=True, min_metrics=(("metrics/mAP50(B)", 0.9),))
        with self.assertRaises(QualityGateError) as ctx:
            run(stricter, backend=FakeBackend())
        self.assertIn("0.4200", str(ctx.exception))

    def test_a_skipped_model_without_recorded_metrics_cannot_pass_the_gate(self) -> None:
        cfg = _config(self.root, models=("det",), skip_unchanged=True)
        run(cfg, backend=FakeBackend())
        manifest = manifest_mod.load_manifest(cfg.models_dir)
        stored = manifest["entries"][MODEL_SPECS["det"].output]
        stored.pop("metrics", None)
        manifest_mod.save_manifest(cfg.models_dir, manifest)

        gated = _config(self.root, models=("det",), skip_unchanged=True, min_metrics=(("metrics/mAP50(B)", 0.1),))
        with self.assertRaises(QualityGateError) as ctx:
            run(gated, backend=FakeBackend())
        self.assertIn("reported: none", str(ctx.exception))

    def test_a_recorded_metric_that_is_not_a_usable_number_cannot_pass_the_gate(self) -> None:
        # Three shapes a hand-edited (or older) manifest can carry, each of
        # which must fail closed: a scalar where a map belongs used to raise
        # AttributeError (exit 1 + traceback), a string used to raise TypeError,
        # and NaN satisfies ``actual < minimum`` in no arithmetic at all — so
        # an unconverged model would have sailed straight through the bar.
        for bogus in (
            "oops",
            {"metrics/mAP50(B)": "0.9"},
            {"metrics/mAP50(B)": float("nan")},
            {"metrics/mAP50(B)": True},
        ):
            with self.subTest(metrics=bogus):
                cfg = _config(self.root, models=("det",), skip_unchanged=True)
                run(cfg, backend=FakeBackend())
                path = cfg.models_dir / MANIFEST_NAME
                document = json.loads(path.read_text(encoding="utf-8"))
                document["entries"][MODEL_SPECS["det"].output]["metrics"] = bogus
                path.write_text(json.dumps(document), encoding="utf-8")

                gated = _config(
                    self.root, models=("det",), skip_unchanged=True, min_metrics=(("metrics/mAP50(B)", 0.1),)
                )
                with self.assertRaises(QualityGateError) as ctx:
                    run(gated, backend=FakeBackend())
                self.assertIn("was not reported", str(ctx.exception))


class PublishBackupTests(_SandboxedTestCase):
    """Replacing an artifact keeps one generation of the previous one."""

    def test_the_first_publish_leaves_no_backup(self) -> None:
        cfg = _config(self.root, models=("det",))
        run(cfg, backend=FakeBackend())
        target = cfg.output_path(MODEL_SPECS["det"])
        self.assertTrue(target.exists())
        self.assertFalse(target.with_name(target.name + BACKUP_SUFFIX).exists())

    def test_republishing_keeps_exactly_one_previous_generation(self) -> None:
        cfg = _config(self.root, models=("det",))
        target = cfg.output_path(MODEL_SPECS["det"])
        backup = target.with_name(target.name + BACKUP_SUFFIX)

        run(cfg, backend=FakeBackend())
        target.write_bytes(b"generation-1")
        run(cfg, backend=FakeBackend())
        self.assertEqual(target.read_bytes(), b"ONNX:det")
        self.assertEqual(backup.read_bytes(), b"generation-1")

        target.write_bytes(b"generation-2")
        run(cfg, backend=FakeBackend())
        # One generation only: the older backup is replaced, not accumulated.
        self.assertEqual(target.read_bytes(), b"ONNX:det")
        self.assertEqual(backup.read_bytes(), b"generation-2")


class SchemaForwardCompatibilityTests(_SandboxedTestCase):
    def test_a_training_run_does_not_down_stamp_a_newer_manifest(self) -> None:
        cfg = _config(self.root, models=("det",))
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        (cfg.models_dir / MANIFEST_NAME).write_text('{"schema_version": 99, "entries": {}}', encoding="utf-8")

        with self.assertLogs("cvtrain.manifest", level="WARNING"):
            code = run(cfg, backend=FakeBackend())

        self.assertEqual(code, ExitCode.OK)
        on_disk = json.loads((cfg.models_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(on_disk["schema_version"], 99)


class ConcurrentRunTests(_SandboxedTestCase):
    """A second run aimed at a busy ``Models/`` directory is refused, not raced."""

    def test_a_held_lock_refuses_the_run(self) -> None:
        cfg = _config(self.root, models=("det",))
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        (cfg.models_dir / LOCK_NAME).write_text("held", encoding="utf-8")

        with self.assertRaises(ConcurrentRunError) as ctx:
            run(cfg, backend=FakeBackend())

        self.assertEqual(ctx.exception.exit_code, ExitCode.PREFLIGHT)


if __name__ == "__main__":
    unittest.main()
