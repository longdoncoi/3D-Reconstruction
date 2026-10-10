"""Unit tests for the pipeline orchestrator.

``trainer`` is the module that decides *what happens* — preflight, dry-run, the
retrain prompt, per-model train/export/publish, manifest durability and the
progress protocol. It is exercised end-to-end here against a fake backend, so
the whole flow runs offline in milliseconds and without the ML stack.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from cvtrain import manifest as manifest_mod
from cvtrain.backend import TrainedModel
from cvtrain.config import MODEL_SPECS, TrainConfig
from cvtrain.errors import ExitCode
from cvtrain.fsio import sha256_file
from cvtrain.logging_setup import PROGRESS_PREFIX
from cvtrain.trainer import confirm_retrain, run

VALID_DATASET = "train: images/train\nval: images/val\nnames:\n  0: part\n"


class FakeBackend:
    """Deterministic stand-in for :class:`cvtrain.backend.UltralyticsBackend`."""

    version = "9.9.9-fake"
    required_modules: tuple[str, ...] = ()

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on
        self.prepared: TrainConfig | None = None

    def prepare(self, cfg: TrainConfig) -> None:
        self.prepared = cfg
        self.calls.append("prepare")

    def train(self, spec: Any, cfg: TrainConfig, on_epoch: Any) -> TrainedModel:
        self.calls.append(f"train:{spec.key}")
        if self.fail_on == spec.key:
            raise RuntimeError(f"backend exploded on {spec.key}")
        for epoch in range(cfg.epochs):
            on_epoch(epoch, cfg.epochs)
        run_dir = cfg.runs_dir / spec.runs_subdir / "train-test"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "args.yaml").write_text(f"epochs: {cfg.epochs}\n", encoding="utf-8")
        return TrainedModel(metrics={"metrics/mAP50(B)": 0.42}, run_dir=run_dir, handle=spec.key)

    def export(self, model: TrainedModel, cfg: TrainConfig) -> Path:
        self.calls.append(f"export:{model.handle}")
        generated = cfg.runs_dir / f"{model.handle}.onnx"
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"ONNX:" + str(model.handle).encode())
        return generated


def _config(root: Path, **overrides: Any) -> TrainConfig:
    """Build a config whose dataset always passes preflight.

    Preflight now fails fast when a referenced ``train``/``val`` directory is
    missing, so the sandbox must contain those directories alongside the YAML.
    """

    data_yaml = root / "data.yaml"
    data_yaml.write_text(VALID_DATASET, encoding="utf-8")
    for name in ("images/train", "images/val"):
        (root / name).mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "models": ("det", "seg"),
        "data_yaml": data_yaml,
        "models_dir": root / "Models",
        "runs_dir": root / "runs",
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

    def test_manifest_records_verifiable_provenance(self) -> None:
        run(self.cfg, backend=self.backend)
        manifest = manifest_mod.load_manifest(self.cfg.models_dir)

        self.assertEqual(manifest["ultralytics"], "9.9.9-fake")
        self.assertEqual(manifest["last_run"]["models"], ["det", "seg"])
        self.assertEqual(manifest["last_run"]["epochs"], 3)

        for key in self.cfg.models:
            spec = MODEL_SPECS[key]
            entry = manifest["entries"][spec.output]
            self.assertEqual(entry["role"], spec.role)
            self.assertEqual(entry["source_weights"], spec.ckpt)
            self.assertEqual(entry["metrics"], {"metrics/mAP50(B)": 0.42})
            self.assertEqual(entry["sha256"], sha256_file(self.cfg.output_path(spec)))

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


class ManifestVerificationTests(_SandboxedTestCase):
    def test_verification_problems_are_surfaced_as_warnings(self) -> None:
        cfg = _config(self.root)
        with mock.patch(
            "cvtrain.trainer.manifest_mod.verify_manifest",
            return_value=["ghost.onnx: missing on disk"],
        ):
            with self.assertLogs("cvtrain.trainer", level="WARNING") as captured:
                code = run(cfg, backend=FakeBackend())

        # The run still succeeds, but the operator must see *why* it is suspect.
        self.assertEqual(code, ExitCode.OK)
        self.assertIn("manifest: ghost.onnx: missing on disk", captured.output[0])


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


if __name__ == "__main__":
    unittest.main()
