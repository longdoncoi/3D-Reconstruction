"""Unit tests for argparse wiring, argument -> config mapping and exit codes.

``main()`` is the only place that turns a failure into a process exit code, so
its mapping is asserted here rather than left to the Qt dialog's error text.
"""

from __future__ import annotations

import contextlib
import io
import logging
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvtrain import manifest as manifest_mod
from cvtrain.cli import build_parser, config_from_args, main
from cvtrain.config import DEFAULT_MODELS, MODEL_SPECS
from cvtrain.errors import BackendUnavailableError, ExitCode


class ParserTests(unittest.TestCase):
    def test_help_exits_cleanly(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_unknown_flag_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(["--nope"])
        self.assertEqual(ctx.exception.code, 2)

    def test_model_flags_accumulate(self) -> None:
        args = build_parser().parse_args(["--yes", "--det", "--track"])
        self.assertEqual(args.models, ["det", "track"])
        self.assertTrue(args.yes)

    def test_no_selection_defaults_to_det_and_seg(self) -> None:
        args = build_parser().parse_args(["--yes"])
        self.assertIsNone(args.models)
        cfg = config_from_args(args)
        self.assertEqual(cfg.models, DEFAULT_MODELS)

    def test_hyperparameters_are_mapped(self) -> None:
        args = build_parser().parse_args(
            ["--det", "--epochs", "25", "--imgsz", "320", "--batch", "4", "--seed", "123", "--device", "cpu"]
        )
        cfg = config_from_args(args)
        self.assertEqual(cfg.epochs, 25)
        self.assertEqual(cfg.imgsz, 320)
        self.assertEqual(cfg.batch, 4)
        self.assertEqual(cfg.seed, 123)
        self.assertEqual(cfg.device, "cpu")

    def test_modes_are_mapped(self) -> None:
        args = build_parser().parse_args(["--check", "--dry-run", "--tensorboard", "-v"])
        cfg = config_from_args(args)
        self.assertTrue(cfg.check_only)
        self.assertTrue(cfg.dry_run)
        self.assertTrue(cfg.tensorboard)
        self.assertTrue(cfg.verbose)

    def test_tracking_flag_matches_tracking_spec(self) -> None:
        cfg = config_from_args(build_parser().parse_args(["--track"]))
        self.assertEqual(cfg.models, ("track",))


class _CliTestCase(unittest.TestCase):
    """Isolates ``main()``, which configures the package logger as a side effect."""

    def setUp(self) -> None:
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def tearDown(self) -> None:
        logger = logging.getLogger("cvtrain")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def _models_dir(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)


class ExitCodeTests(_CliTestCase):
    def test_success_returns_zero(self) -> None:
        with mock.patch("cvtrain.cli.trainer.run", return_value=0):
            self.assertEqual(main(["--yes"]), ExitCode.OK)

    def test_cvtrain_error_returns_its_code_without_a_traceback(self) -> None:
        # A missing dependency is expected: one actionable line, no stack trace.
        with mock.patch("cvtrain.cli.trainer.run", side_effect=BackendUnavailableError("ultralytics missing")):
            with mock.patch("cvtrain.cli.LOG") as fake_log:
                code = main(["--yes"])

        self.assertEqual(code, ExitCode.FAILURE)
        fake_log.error.assert_called_once()
        self.assertEqual(str(fake_log.error.call_args.args[1]), "ultralytics missing")
        fake_log.exception.assert_not_called()

    def test_unexpected_failure_keeps_the_traceback(self) -> None:
        with mock.patch("cvtrain.cli.trainer.run", side_effect=RuntimeError("boom")):
            with mock.patch("cvtrain.cli.LOG") as fake_log:
                code = main(["--yes"])

        self.assertEqual(code, ExitCode.FAILURE)
        fake_log.exception.assert_called_once()

    def test_interrupt_maps_to_the_sigint_code(self) -> None:
        # The Qt dock reports the raw number to the user, and 130 is the
        # conventional "terminated by SIGINT" status.
        with mock.patch("cvtrain.cli.trainer.run", side_effect=KeyboardInterrupt):
            self.assertEqual(main(["--yes"]), ExitCode.ABORTED)


class ManifestCommandTests(_CliTestCase):
    def test_write_manifest_scans_and_returns_ok(self) -> None:
        models_dir = self._models_dir()
        (models_dir / MODEL_SPECS["det"].output).write_bytes(b"det-bytes")

        code = main(["--write-manifest", "--models-dir", str(models_dir)])

        self.assertEqual(code, ExitCode.OK)
        manifest = manifest_mod.load_manifest(models_dir)
        self.assertIn(MODEL_SPECS["det"].output, manifest["entries"])

    def test_verify_missing_manifest_returns_preflight_code(self) -> None:
        code = main(["--verify-manifest", "--models-dir", str(self._models_dir())])
        self.assertEqual(code, ExitCode.PREFLIGHT)

    def test_verify_detects_a_tampered_artifact(self) -> None:
        models_dir = self._models_dir()
        target = models_dir / MODEL_SPECS["det"].output
        target.write_bytes(b"original")
        main(["--write-manifest", "--models-dir", str(models_dir)])

        target.write_bytes(b"tampered")
        with mock.patch("cvtrain.cli.LOG") as fake_log:
            code = main(["--verify-manifest", "--models-dir", str(models_dir)])

        self.assertEqual(code, ExitCode.PREFLIGHT)
        reported = [call.args[1] for call in fake_log.error.call_args_list if len(call.args) > 1]
        self.assertTrue(any("mismatch" in problem for problem in reported), reported)

    def test_verify_of_a_consistent_manifest_returns_ok(self) -> None:
        models_dir = self._models_dir()
        (models_dir / MODEL_SPECS["det"].output).write_bytes(b"original")
        main(["--write-manifest", "--models-dir", str(models_dir)])

        with mock.patch("cvtrain.cli.LOG") as fake_log:
            code = main(["--verify-manifest", "--models-dir", str(models_dir)])

        self.assertEqual(code, ExitCode.OK)
        fake_log.error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
