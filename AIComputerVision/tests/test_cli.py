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
from cvtrain.backend import UltralyticsBackend
from cvtrain.cli import build_parser, config_from_args, main, parse_metric_threshold
from cvtrain.config import DEFAULT_MODELS, DEFAULT_OPSET, MODEL_SPECS, TrainConfig
from cvtrain.errors import BackendUnavailableError, ExitCode, QualityGateError
from cvtrain.fsio import LOCK_NAME


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
        args = build_parser().parse_args(["--check", "--tensorboard", "-v"])
        cfg = config_from_args(args)
        self.assertTrue(cfg.check_only)
        self.assertTrue(cfg.tensorboard)
        self.assertTrue(cfg.verbose)

        dry = config_from_args(build_parser().parse_args(["--dry-run"]))
        self.assertTrue(dry.dry_run)
        self.assertFalse(dry.check_only)

    def test_two_modes_at_once_are_a_usage_error(self) -> None:
        # Silent precedence is how `--write-manifest --verify-manifest` used to
        # overwrite a manifest and then "verify" the file it had just written.
        for pair in (("--check", "--dry-run"), ("--write-manifest", "--verify-manifest"), ("--check", "--verify-manifest")):
            with self.subTest(pair=pair), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    build_parser().parse_args(list(pair))
                self.assertEqual(ctx.exception.code, 2)

    def test_tracking_flag_matches_tracking_spec(self) -> None:
        cfg = config_from_args(build_parser().parse_args(["--track"]))
        self.assertEqual(cfg.models, ("track",))

    def test_skip_unchanged_is_opt_in(self) -> None:
        # Opt-in, because the default must keep overwriting: that is the
        # behaviour every existing caller (and the Qt dialog) relies on.
        self.assertFalse(config_from_args(build_parser().parse_args(["--yes"])).skip_unchanged)
        self.assertTrue(config_from_args(build_parser().parse_args(["--skip-unchanged"])).skip_unchanged)

    def test_quality_gate_thresholds_are_mapped_and_repeatable(self) -> None:
        args = build_parser().parse_args(["--min-metric", "metrics/mAP50(B)=0.5", "--min-metric", "mAP=0.4"])
        self.assertEqual(config_from_args(args).min_metrics, (("metrics/mAP50(B)", 0.5), ("mAP", 0.4)))

    def test_the_quality_gate_is_opt_in(self) -> None:
        self.assertEqual(config_from_args(build_parser().parse_args(["--yes"])).min_metrics, ())

    def test_a_threshold_without_an_equals_sign_is_a_usage_error(self) -> None:
        # argparse exits 2 on a bad option, which the desktop app surfaces as a
        # "Mã lỗi: 2" dialog — and it must say *why*, not just reject.
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(["--min-metric", "mAP50"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("NAME=VALUE", stderr.getvalue())

    def test_a_non_numeric_threshold_is_a_usage_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ctx:
            build_parser().parse_args(["--min-metric", "mAP=high"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("is not a number", stderr.getvalue())

    def test_weights_dir_is_mapped(self) -> None:
        # Inputs move like outputs: the checkpoints a run starts from are
        # overridable the same way the export directory is.
        args = build_parser().parse_args(["--weights-dir", "C:/ckpts"])
        self.assertEqual(config_from_args(args).weights_dir, Path("C:/ckpts"))

    def test_the_opset_default_comes_from_the_config_module(self) -> None:
        # A literal here would let the parser and TrainConfig disagree the
        # moment either one changes; both must read the same constant.
        self.assertEqual(build_parser().parse_args(["--yes"]).opset, DEFAULT_OPSET)


class MetricThresholdSyntaxTests(unittest.TestCase):
    """``--min-metric`` syntax is parsed here; semantics belong to preflight."""

    def test_a_well_formed_threshold_is_split_into_a_pair(self) -> None:
        self.assertEqual(parse_metric_threshold("metrics/mAP50(B)=0.5"), ("metrics/mAP50(B)", 0.5))

    def test_the_name_is_trimmed_and_the_value_may_be_integral(self) -> None:
        self.assertEqual(parse_metric_threshold(" mAP = 1 "), ("mAP", 1.0))

    def test_a_missing_equals_sign_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_metric_threshold("mAP50")
        self.assertIn("NAME=VALUE", str(ctx.exception))

    def test_an_empty_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_metric_threshold("=0.5")
        self.assertIn("NAME=VALUE", str(ctx.exception))

    def test_a_non_numeric_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_metric_threshold("mAP=high")
        self.assertIn("is not a number", str(ctx.exception))

    def test_out_of_range_and_non_finite_values_are_left_for_preflight(self) -> None:
        # Deliberately accepted here: a programmatically built config reaches
        # the same preflight rule as a command line, in one place only.
        self.assertEqual(parse_metric_threshold("mAP=1.5")[1], 1.5)
        name, value = parse_metric_threshold("mAP=nan")
        self.assertEqual(name, "mAP")
        self.assertNotEqual(value, value)  # NaN: only isfinite() can reject it


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
        # A backend that cannot be imported is an environment failure: one
        # actionable line, no stack trace, and the same exit code (2) preflight
        # would have reported had it caught the problem first.
        with mock.patch("cvtrain.cli.trainer.run", side_effect=BackendUnavailableError("ultralytics missing")):
            with mock.patch("cvtrain.cli.LOG") as fake_log:
                code = main(["--yes"])

        self.assertEqual(code, ExitCode.PREFLIGHT)
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

    def test_a_failed_quality_gate_maps_to_its_own_code(self) -> None:
        # 3, not 1 and not 2: the run did what it was asked, and the artifact is
        # merely not good enough. The operator's fix is "train better", which is
        # a different instruction than "fix your environment".
        with mock.patch("cvtrain.cli.trainer.run", side_effect=QualityGateError("mAP too low")):
            with mock.patch("cvtrain.cli.LOG") as fake_log:
                code = main(["--yes"])

        self.assertEqual(code, ExitCode.QUALITY)
        fake_log.error.assert_called_once()
        self.assertEqual(str(fake_log.error.call_args.args[1]), "mAP too low")
        fake_log.exception.assert_not_called()


class CompositionRootTests(_CliTestCase):
    """``main()`` picks the backend; the orchestrator only ever receives one."""

    def test_main_injects_a_concrete_backend_into_the_orchestrator(self) -> None:
        with mock.patch("cvtrain.cli.trainer.run", return_value=0) as fake_run:
            main(["--yes"])

        fake_run.assert_called_once()
        cfg = fake_run.call_args.args[0]
        backend = fake_run.call_args.kwargs.get("backend")
        self.assertIsInstance(cfg, TrainConfig)
        self.assertIsInstance(backend, UltralyticsBackend)


class ManifestCommandTests(_CliTestCase):
    def test_write_manifest_scans_and_returns_ok(self) -> None:
        models_dir = self._models_dir()
        (models_dir / MODEL_SPECS["det"].output).write_bytes(b"det-bytes")

        code = main(["--write-manifest", "--models-dir", str(models_dir)])

        self.assertEqual(code, ExitCode.OK)
        manifest = manifest_mod.load_manifest(models_dir)
        self.assertIn(MODEL_SPECS["det"].output, manifest["entries"])

    def test_verify_detects_a_tampered_artifact(self) -> None:
        models_dir = self._models_dir()
        target = models_dir / MODEL_SPECS["det"].output
        target.write_bytes(b"original")
        main(["--write-manifest", "--models-dir", str(models_dir)])

        target.write_bytes(b"tampered")
        with mock.patch("cvtrain.cli.LOG") as fake_log:
            code = main(["--verify-manifest", "--models-dir", str(models_dir)])

        self.assertEqual(code, ExitCode.PREFLIGHT)
        # One actionable line carrying every problem, exactly like a training
        # run whose freshly-written manifest does not check out.
        reported = [str(call.args[1]) for call in fake_log.error.call_args_list]
        self.assertTrue(any("mismatch" in problem for problem in reported), reported)

    def test_verify_reports_a_manifest_that_is_absent(self) -> None:
        with mock.patch("cvtrain.cli.LOG") as fake_log:
            code = main(["--verify-manifest", "--models-dir", str(self._models_dir())])

        self.assertEqual(code, ExitCode.PREFLIGHT)
        reported = [str(call.args[1]) for call in fake_log.error.call_args_list]
        self.assertTrue(any("Manifest missing" in problem for problem in reported), reported)

    def test_verify_of_a_consistent_manifest_returns_ok(self) -> None:
        models_dir = self._models_dir()
        (models_dir / MODEL_SPECS["det"].output).write_bytes(b"original")
        main(["--write-manifest", "--models-dir", str(models_dir)])

        with mock.patch("cvtrain.cli.LOG") as fake_log:
            code = main(["--verify-manifest", "--models-dir", str(models_dir)])

        self.assertEqual(code, ExitCode.OK)
        fake_log.error.assert_not_called()

    def test_write_manifest_refuses_a_held_lock(self) -> None:
        # Adopting models while another run trains would rewrite manifest.json
        # out from under it, so the same advisory lock a run takes is honoured.
        models_dir = self._models_dir()
        (models_dir / LOCK_NAME).write_text("held", encoding="utf-8")

        with mock.patch("cvtrain.cli.LOG") as fake_log:
            code = main(["--write-manifest", "--models-dir", str(models_dir)])

        self.assertEqual(code, ExitCode.PREFLIGHT)
        reported = [str(call.args[1]) for call in fake_log.error.call_args_list]
        self.assertTrue(any("held by another cvtrain run" in problem for problem in reported), reported)


if __name__ == "__main__":
    unittest.main()
