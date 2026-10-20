"""Unit tests for environment and hyper-parameter preflight validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from cvtrain.checks import Check, PreflightReport, module_available
from cvtrain.config import MODEL_REGISTRY, ModelSpec, TrainConfig, register_model_spec
from cvtrain.preflight import (
    MAX_EPOCHS,
    MAX_IMGSZ,
    MAX_OPSET,
    MAX_WORKERS,
    MIN_IMGSZ,
    MIN_OPSET,
    check_dependencies,
    check_device,
    check_hyperparameters,
    check_metric_thresholds,
    check_model_roles,
    check_python,
    check_weights,
    run,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _make_dataset(tmp: str) -> Path:
    """A dataset tree preflight accepts, so only the check under test can fail."""

    root = Path(tmp)
    for name in ("images/train", "images/val"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return _write(
        root / "data.yaml",
        "train: images/train\nval: images/val\nnames:\n  0: part\n",
    )


class HyperparameterTests(unittest.TestCase):
    """Bad numbers fail here, with the flag named — not deep inside the framework."""

    def _cfg(self, **overrides: Any) -> TrainConfig:
        return TrainConfig(models=("det",), **overrides)

    def _failures(self, **overrides: Any) -> list[Check]:
        return [check for check in check_hyperparameters(self._cfg(**overrides)) if not check.ok]

    def test_the_defaults_pass(self) -> None:
        checks = check_hyperparameters(self._cfg())
        self.assertTrue(all(check.ok for check in checks), [check.line() for check in checks])

    def test_every_rule_is_reported_by_its_own_name(self) -> None:
        names = {check.name for check in check_hyperparameters(self._cfg())}
        self.assertEqual(
            names,
            {
                "param.models",
                "param.role.det",
                "param.epochs",
                "param.imgsz",
                "param.batch",
                "param.workers",
                "param.opset",
                "param.device",
            },
        )

    def test_zero_or_negative_epochs_fail_and_name_the_flag(self) -> None:
        for epochs in (0, -3):
            with self.subTest(epochs=epochs):
                failed = self._failures(epochs=epochs)
                self.assertEqual([check.name for check in failed], ["param.epochs"])
                self.assertIn("--epochs", failed[0].detail)

    def test_an_image_size_below_the_coarsest_stride_fails(self) -> None:
        # YOLO's coarsest stride is 32; a smaller image has no valid grid.
        failed = self._failures(imgsz=MIN_IMGSZ - 1)
        self.assertEqual([check.name for check in failed], ["param.imgsz"])
        self.assertEqual(self._failures(imgsz=MIN_IMGSZ), [])

    def test_the_automatic_batch_sentinel_is_accepted_but_zero_is_not(self) -> None:
        self.assertEqual(self._failures(batch=-1), [])
        self.assertEqual([check.name for check in self._failures(batch=0)], ["param.batch"])

    def test_zero_dataloader_workers_are_fine_but_negative_ones_are_not(self) -> None:
        self.assertEqual(self._failures(workers=0), [])
        self.assertEqual([check.name for check in self._failures(workers=-2)], ["param.workers"])

    def test_an_opset_the_app_cannot_load_is_rejected(self) -> None:
        # ONNX Runtime 1.20 (pinned in CMakeLists.txt) implements opset
        # MAX_OPSET: an artifact stamped higher loads nowhere in this product.
        for opset in (MIN_OPSET - 1, MAX_OPSET + 1):
            with self.subTest(opset=opset):
                failed = self._failures(opset=opset)
                self.assertEqual([check.name for check in failed], ["param.opset"])

    def test_an_empty_model_selection_fails(self) -> None:
        failed = [check for check in check_hyperparameters(TrainConfig(models=())) if not check.ok]
        self.assertEqual([check.name for check in failed], ["param.models"])

    def test_an_unknown_model_key_fails_instead_of_raising_keyerror(self) -> None:
        # A programmatic TrainConfig can carry a typo resolve_models would have
        # filtered on the CLI path: report it, never raise KeyError downstream.
        cfg = TrainConfig(models=("bogus",))
        failed = [check for check in check_hyperparameters(cfg) if not check.ok]
        self.assertEqual([check.name for check in failed], ["param.models"])
        self.assertIn("bogus", failed[0].detail)
        # The weights check skips unknown keys so the report stays complete.
        self.assertEqual(check_weights(cfg), [])

    def test_an_unsupported_role_fails_in_preflight_not_after_training(self) -> None:
        # A custom registry entry is allowed to exist, but exporting it would
        # fail in ``contract_for`` only after the expensive run. Preflight is
        # where the mismatch must surface.
        spec = ModelSpec("pose_check", "pose.pt", "pose.onnx", "pose", "pose", "Pose")
        try:
            register_model_spec(spec)
            cfg = TrainConfig(models=("pose_check",))
            [role_check] = check_model_roles(cfg)
            self.assertFalse(role_check.ok)
            self.assertEqual(role_check.name, "param.role.pose_check")
            self.assertIn("not supported", role_check.detail)
            failed = [check.name for check in check_hyperparameters(cfg) if not check.ok]
            self.assertEqual(failed, ["param.role.pose_check"])
        finally:
            MODEL_REGISTRY._specs.pop("pose_check", None)

    def test_every_built_in_role_is_supported(self) -> None:
        checks = check_model_roles(TrainConfig(models=("det", "seg", "track")))
        self.assertTrue(all(check.ok for check in checks), [check.line() for check in checks])

    def test_absurd_hyper_parameters_fail_fast(self) -> None:
        cases = (
            {"epochs": MAX_EPOCHS + 1},
            {"imgsz": MAX_IMGSZ + 1},
            {"workers": MAX_WORKERS + 1},
        )
        for overrides in cases:
            with self.subTest(**overrides):
                failed = self._failures(**overrides)
                self.assertEqual(len(failed), 1)
                self.assertTrue(failed[0].name.startswith("param."))

    def test_the_gate_thresholds_are_validated_with_the_rest(self) -> None:
        cfg = TrainConfig(models=("det",), min_metrics=(("mAP", 2.0),))
        failed = [check.name for check in check_hyperparameters(cfg) if not check.ok]
        self.assertEqual(failed, ["param.min_metric.mAP"])

    def test_the_device_is_validated_with_the_rest(self) -> None:
        cfg = TrainConfig(models=("det",), device="gpu")
        failed = [check.name for check in check_hyperparameters(cfg) if not check.ok]
        self.assertEqual(failed, ["param.device"])


class DeviceTests(unittest.TestCase):
    """``--device`` is rejected here rather than by the framework's selector."""

    def _check(self, device: str | None) -> Check:
        return check_device(TrainConfig(models=("det",), device=device))

    def test_no_device_means_the_framework_chooses(self) -> None:
        for device in (None, "", "   ", "auto", "AUTO", "  Auto  "):
            with self.subTest(device=device):
                check = self._check(device)
                self.assertTrue(check.ok, check.detail)
                self.assertIn("auto", check.detail)

    def test_every_documented_form_is_accepted(self) -> None:
        for device in ("cpu", "cuda", "cuda:0", "mps", "metal", "0", "0,1", "0, 1", "cpu,cuda:0"):
            with self.subTest(device=device):
                check = self._check(device)
                self.assertTrue(check.ok, check.detail)
                self.assertEqual(check.detail, device)

    def test_device_tokens_match_case_insensitively(self) -> None:
        # A case-insensitive match costs nothing and keeps ``--device CPU`` from
        # being rejected for a spelling the framework itself accepts.
        for device in ("CPU", "Cuda:0", "MPS"):
            with self.subTest(device=device):
                check = self._check(device)
                self.assertTrue(check.ok, check.detail)
                self.assertEqual(check.detail, device)

    def test_a_typo_is_named_instead_of_reaching_the_framework(self) -> None:
        for device in ("gpu", "cuda:x", "-1", "cpu,", ",0", "cuda:0.5"):
            with self.subTest(device=device):
                check = self._check(device)
                self.assertFalse(check.ok, device)
                self.assertEqual(check.name, "param.device")
                self.assertIn("use cpu", check.detail)
                self.assertIn(repr(device), check.detail)


class MetricThresholdTests(unittest.TestCase):
    def test_a_sane_threshold_produces_one_passing_check(self) -> None:
        [check] = check_metric_thresholds((("metrics/mAP50(B)", 0.5),))
        self.assertTrue(check.ok, check.detail)
        self.assertEqual(check.name, "param.min_metric.metrics/mAP50(B)")

    def test_no_threshold_means_no_gate_checks(self) -> None:
        self.assertEqual(check_metric_thresholds(()), [])

    def test_unnameless_non_finite_and_out_of_range_thresholds_fail(self) -> None:
        for name, threshold in (("", 0.5), ("mAP", float("nan")), ("mAP", 1.5), ("mAP", -0.1)):
            with self.subTest(name=name, threshold=threshold):
                [check] = check_metric_thresholds(((name, threshold),))
                self.assertFalse(check.ok)
                self.assertIn("finite threshold within 0..1", check.detail)

    def test_an_unnamed_threshold_is_still_named_in_the_report(self) -> None:
        # The check name has to stay printable even when the metric name is
        # empty, otherwise the report itself becomes unreadable.
        [check] = check_metric_thresholds((("", 0.5),))
        self.assertEqual(check.name, "param.min_metric.<unnamed>")


class WeightsResolutionTests(unittest.TestCase):
    def test_weights_checks_resolve_through_the_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / MODEL_REGISTRY["det"].ckpt).write_bytes(b"checkpoint")
            cfg = TrainConfig(models=("det",), weights_dir=root)

            [check] = check_weights(cfg)

            self.assertTrue(check.ok, check.detail)
            self.assertEqual(check.name, "weights.det")


class ReportTests(unittest.TestCase):
    def test_python_check_passes_on_supported_interpreter(self) -> None:
        self.assertTrue(check_python().ok)

    def test_module_lookup_never_raises_on_a_broken_sys_modules_entry(self) -> None:
        # find_spec raises ValueError/ImportError for half-initialised entries;
        # preflight must report "missing", not crash before printing a report.
        for failure in (ValueError("bad spec"), ImportError("no spec")):
            with self.subTest(failure=failure), mock.patch(
                "cvtrain.checks.importlib.util.find_spec", side_effect=failure
            ):
                self.assertFalse(module_available("anything"))

    def test_report_ok_ignores_non_fatal_failures(self) -> None:
        report = PreflightReport()
        report.add(Check("weights.det", False, "will download", fatal=False))
        self.assertTrue(report.ok)
        self.assertEqual(report.failures, [])

    def test_report_fails_on_fatal_check(self) -> None:
        report = PreflightReport()
        report.add(Check("dataset", False, "missing", fatal=True))
        self.assertFalse(report.ok)
        self.assertEqual(len(report.failures), 1)

    def test_run_flags_dependencies_without_importing_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_dataset(tmp)
            cfg = TrainConfig(data_yaml=path, models=("det",))
            report = run(cfg, deps_fatal=False, modules=("yaml", "not_a_real_module"))
            names = {check.name for check in report.checks}
            self.assertIn("param.epochs", names)
            self.assertIn("param.opset", names)
            self.assertIn("param.device", names)
            self.assertIn("dataset", names)
            self.assertIn("weights.det", names)
            self.assertIn("dependency.yaml", names)
            self.assertIn("dependency.not_a_real_module", names)
            # deps_fatal=False downgrades a missing module to a warning.
            self.assertTrue(report.ok, [c.line() for c in report.checks])

    def test_missing_dependency_is_fatal_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_dataset(tmp)
            cfg = TrainConfig(data_yaml=path, models=("det",))
            report = run(cfg, deps_fatal=True, modules=("not_a_real_module",))
            self.assertFalse(report.ok)
            self.assertEqual([check.name for check in report.failures], ["dependency.not_a_real_module"])
            self.assertIn("pip install", report.failures[0].detail)

    def test_backend_may_declare_no_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_dataset(tmp)
            cfg = TrainConfig(data_yaml=path, models=("det",))
            report = run(cfg, deps_fatal=True, modules=())
            self.assertNotIn("dependency.ultralytics", {check.name for check in report.checks})
            self.assertTrue(report.ok, [c.line() for c in report.checks])

    def test_dependencies_are_reported_without_importing_them(self) -> None:
        # A probe must never import the module it is probing: doing so would
        # defeat the whole point of checking first.
        checks = check_dependencies(fatal=True, modules=("json",))
        self.assertEqual([check.name for check in checks], ["dependency.json"])
        self.assertTrue(checks[0].ok, checks[0].detail)


if __name__ == "__main__":
    unittest.main()
