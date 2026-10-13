"""Unit tests for the training backend seam.

``UltralyticsBackend`` is exercised against a stub ``ultralytics`` package
injected into ``sys.modules``. That covers the adapter's real logic — checkpoint
resolution, callback wiring, run-directory naming, metric extraction, export
arguments and the missing-dependency path — without needing torch, a GPU or a
dataset, and on a CI image where ultralytics is not installed at all.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import types
import unittest
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from cvtrain.backend import (
    TrainedModel,
    TrainingBackend,
    UltralyticsBackend,
    _run_timestamp,
    metrics_from,
    train_kwargs,
)
from cvtrain.config import MODEL_SPECS, TrainConfig
from cvtrain.errors import BackendUnavailableError


class _Results:
    """Stand-in for ``ultralytics`' detection metrics object."""

    def __init__(self, results: dict[str, object]) -> None:
        self.results_dict = results


class ProtocolTests(unittest.TestCase):
    def test_structural_conformance_accepts_any_duck_typed_backend(self) -> None:
        class Minimal:
            version = "1.2.3"

            def required_modules(self, cfg: TrainConfig) -> tuple[str, ...]:
                return ()

            def prepare(self, cfg: TrainConfig) -> None:  # pragma: no cover - never called
                raise NotImplementedError

            def train(self, spec: Any, cfg: TrainConfig, on_epoch: Any) -> TrainedModel[Any]:  # pragma: no cover
                raise NotImplementedError

            def export(self, model: TrainedModel[Any], cfg: TrainConfig) -> Path:  # pragma: no cover
                raise NotImplementedError

        self.assertIsInstance(Minimal(), TrainingBackend)

    def test_missing_method_breaks_conformance(self) -> None:
        class Incomplete:
            version = "1.2.3"

            def required_modules(self, cfg: TrainConfig) -> tuple[str, ...]:
                return ()

        self.assertNotIsInstance(Incomplete(), TrainingBackend)


class MetricsFromTests(unittest.TestCase):
    def test_keeps_only_numeric_values(self) -> None:
        metrics = metrics_from(_Results({"metrics/precision(B)": 0.9, "epoch": 5, "bad": "text", "none": None}))
        self.assertEqual(metrics, {"metrics/precision(B)": 0.9, "epoch": 5.0})

    def test_keeps_a_real_number_that_is_not_a_python_float(self) -> None:
        # numpy scalar types register with ``numbers`` but are not ``float``;
        # the ``numbers.Real`` test keeps them instead of dropping a real metric
        # (which would turn a ``--min-metric`` request into a false "not
        # reported").
        metrics = metrics_from(_Results({"mAP": Fraction(1, 2)}))
        self.assertEqual(metrics, {"mAP": 0.5})

    def test_returns_empty_dict_when_framework_reports_nothing(self) -> None:
        self.assertEqual(metrics_from(None), {})
        self.assertEqual(metrics_from(object()), {})
        self.assertEqual(metrics_from(_Results({"not": "a"})), {})


class TrainKwargsTests(unittest.TestCase):
    """``TrainConfig`` -> ``YOLO.train(...)`` mapping lives in the adapter (ADR 0002)."""

    def test_train_kwargs_are_reproducible(self) -> None:
        cfg = TrainConfig(models=("det",), epochs=3, seed=7)
        kwargs = train_kwargs(cfg)
        self.assertEqual(kwargs["epochs"], 3)
        self.assertEqual(kwargs["seed"], 7)
        self.assertEqual(kwargs["data"], str(cfg.data_yaml))
        self.assertIs(kwargs["deterministic"], True)
        self.assertIs(kwargs["exist_ok"], False)
        self.assertNotIn("device", kwargs)

    def test_device_is_forwarded_when_set(self) -> None:
        self.assertEqual(train_kwargs(TrainConfig(device="cpu"))["device"], "cpu")

    def test_auto_device_is_not_forwarded_to_the_framework(self) -> None:
        # ``auto`` is provenance spelling ("let the framework choose"), not a
        # device the framework accepts: it must behave like the default.
        for device in ("auto", "AUTO", "  auto  "):
            with self.subTest(device=device):
                cfg = TrainConfig(device=device)
                self.assertEqual(cfg.device_class, "auto")
                self.assertNotIn("device", train_kwargs(cfg))


class _FakeYOLO:
    """Records everything the adapter asks the real YOLO class to do."""

    last: _FakeYOLO | None = None
    save_dir: str | None = None
    export_path = ""

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.callbacks: dict[str, Any] = {}
        self.trainer = SimpleNamespace()
        self.train_kwargs: dict[str, Any] = {}
        self.export_kwargs: dict[str, Any] = {}
        _FakeYOLO.last = self

    def add_callback(self, event: str, callback: Any) -> None:
        self.callbacks[event] = callback

    def train(self, **kwargs: Any) -> _Results:
        self.train_kwargs = kwargs
        if _FakeYOLO.save_dir is not None:
            self.trainer.save_dir = _FakeYOLO.save_dir
        return _Results({"metrics/mAP50(B)": 0.77, "non_numeric": "dropped"})

    def export(self, **kwargs: Any) -> str:
        self.export_kwargs = kwargs
        return _FakeYOLO.export_path


@contextlib.contextmanager
def _stub_ultralytics(
    *,
    save_dir: str | None = "auto",
    export_path: str = "",
    broken: bool = False,
) -> Iterator[tuple[Any, Any]]:
    """Install a fake ``ultralytics`` package for the duration of the block."""

    if broken:
        with mock.patch.dict(sys.modules, {"ultralytics": None, "ultralytics.utils": None}):
            yield None, None
        return

    _FakeYOLO.last = None
    _FakeYOLO.save_dir = save_dir
    _FakeYOLO.export_path = export_path

    settings: dict[str, Any] = {}
    package = types.ModuleType("ultralytics")
    # ``from ultralytics.utils import ...`` needs the stub to look like a
    # package; ModuleType has no declared __version__/YOLO/__path__ slots.
    setattr(package, "__path__", [])
    setattr(package, "__version__", "8.4.41-stub")
    setattr(package, "YOLO", _FakeYOLO)

    utils = types.ModuleType("ultralytics.utils")
    setattr(utils, "SETTINGS", settings)

    with mock.patch.dict(sys.modules, {"ultralytics": package, "ultralytics.utils": utils}):
        yield package, settings


def _config(root: Path) -> TrainConfig:
    return TrainConfig(
        models=("det",),
        data_yaml=root / "data.yaml",
        models_dir=root / "Models",
        runs_dir=root / "runs",
        epochs=3,
    )


class _SandboxedTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = _config(self.root)
        self.exported_onnx = self.root / "generated.onnx"
        self.exported_onnx.write_bytes(b"onnx-bytes")


class LazyImportTests(_SandboxedTestCase):
    def test_declares_the_modules_preflight_must_verify(self) -> None:
        # A method, not a constant: an optional feature can add an optional
        # dependency, and the adapter is the one module that knows which.
        self.assertEqual(UltralyticsBackend().required_modules(self.cfg), ("ultralytics", "yaml", "onnx"))

    def test_tensorboard_is_demanded_only_when_it_was_asked_for(self) -> None:
        backend = UltralyticsBackend()
        self.assertNotIn("tensorboard", backend.required_modules(self.cfg))
        self.cfg.tensorboard = True
        self.assertEqual(backend.required_modules(self.cfg)[-1], "tensorboard")

    def test_construction_is_lazy(self) -> None:
        # Building the backend must not import the ML stack, otherwise
        # --help / --check / the unit suite would need torch installed.
        backend = UltralyticsBackend()
        self.assertEqual(backend.version, "")
        self.assertIsNone(backend._yolo)

    def test_version_is_learned_on_first_use(self) -> None:
        with _stub_ultralytics():
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            self.assertEqual(backend.version, "8.4.41-stub")

    def test_missing_dependency_raises_a_typed_error_not_a_traceback(self) -> None:
        backend = UltralyticsBackend()
        with _stub_ultralytics(broken=True):
            with self.assertRaises(BackendUnavailableError) as ctx:
                backend.prepare(self.cfg)
        self.assertIn("pip install", str(ctx.exception))
        self.assertIn("cvtrain", type(ctx.exception).__module__)


class PrepareTests(_SandboxedTestCase):
    def test_tensorboard_is_enabled_only_when_requested(self) -> None:
        self.cfg.tensorboard = True
        with _stub_ultralytics() as (_, settings):
            UltralyticsBackend().prepare(self.cfg)
        self.assertEqual(settings, {"tensorboard": True})

    def test_prepare_is_inert_without_the_flag(self) -> None:
        with _stub_ultralytics() as (_, settings):
            UltralyticsBackend().prepare(self.cfg)
        self.assertEqual(settings, {})

    def test_a_missing_tensorboard_is_reported_instead_of_claimed(self) -> None:
        # ultralytics drops TensorBoard *silently* when the package is absent,
        # so the "enabled" line must never appear for a run that will write no
        # logs: say what actually happened, and leave the setting off.
        self.cfg.tensorboard = True
        with (
            _stub_ultralytics() as (_, settings),
            mock.patch.dict(sys.modules, {"tensorboard": None}),
            self.assertLogs("cvtrain.backend", level="WARNING") as captured,
        ):
            UltralyticsBackend().prepare(self.cfg)
        self.assertEqual(settings, {})
        self.assertTrue(any("no TensorBoard logs" in line for line in captured.output))

    def test_a_backend_without_a_settings_object_does_not_claim_success(self) -> None:
        # Half of the same rule: even with the package importable, there must
        # be somewhere to switch the writer on. No settings -> no claim.
        self.cfg.tensorboard = True
        backend = UltralyticsBackend()
        backend._yolo = object()  # already "loaded", so no framework import
        backend._settings = None
        with self.assertLogs("cvtrain.backend", level="WARNING") as captured:
            backend.prepare(self.cfg)
        self.assertTrue(any("TensorBoard logging stays off" in line for line in captured.output))


class TrainTests(_SandboxedTestCase):
    def test_loads_the_checkpoint_declared_by_the_model_spec(self) -> None:
        spec = MODEL_SPECS["det"]
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            backend.train(spec, self.cfg, lambda epoch, total: None)
        assert _FakeYOLO.last is not None
        self.assertEqual(Path(_FakeYOLO.last.weights), self.cfg.weights_path(spec))

    def test_forwards_hyperparameters_and_uses_the_run_subdirectory(self) -> None:
        spec = MODEL_SPECS["det"]
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            backend.train(spec, self.cfg, lambda epoch, total: None)
        assert _FakeYOLO.last is not None
        kwargs = _FakeYOLO.last.train_kwargs
        self.assertEqual(kwargs["data"], str(self.cfg.data_yaml))
        self.assertEqual(kwargs["epochs"], 3)
        self.assertEqual(Path(kwargs["project"]), self.cfg.runs_dir / spec.runs_subdir)
        self.assertTrue(Path(kwargs["name"]).name.startswith("train-"))
        # exist_ok=False: a re-run must not silently overwrite the last run.
        self.assertIs(kwargs["exist_ok"], False)

    def test_registers_the_epoch_callback_and_reports_progress(self) -> None:
        spec = MODEL_SPECS["det"]
        seen: list[tuple[int, int]] = []
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            backend.train(spec, self.cfg, lambda epoch, total: seen.append((epoch, total)))
        assert _FakeYOLO.last is not None
        self.assertIn("on_train_epoch_end", _FakeYOLO.last.callbacks)

        _FakeYOLO.last.callbacks["on_train_epoch_end"](SimpleNamespace(epoch=1, epochs=3))
        self.assertEqual(seen, [(1, 3)])

    def test_missing_epoch_attributes_fall_back_safely(self) -> None:
        spec = MODEL_SPECS["det"]
        seen: list[tuple[int, int]] = []
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            backend.train(spec, self.cfg, lambda epoch, total: seen.append((epoch, total)))
        assert _FakeYOLO.last is not None
        _FakeYOLO.last.callbacks["on_train_epoch_end"](SimpleNamespace())
        self.assertEqual(seen, [(0, self.cfg.epochs)])

    def test_returns_metrics_and_the_framework_reported_run_dir(self) -> None:
        spec = MODEL_SPECS["det"]
        run_dir = self.root / "runs" / "detect" / "train-20260101-000000"
        with _stub_ultralytics(save_dir=str(run_dir), export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            model = backend.train(spec, self.cfg, lambda epoch, total: None)
        self.assertEqual(model.metrics, {"metrics/mAP50(B)": 0.77})
        self.assertEqual(model.run_dir, run_dir)

    def test_run_dir_falls_back_to_the_project_directory(self) -> None:
        # A trainer object without save_dir must not crash the pipeline.
        spec = MODEL_SPECS["det"]
        with _stub_ultralytics(save_dir=None, export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            model = backend.train(spec, self.cfg, lambda epoch, total: None)
        self.assertEqual(model.run_dir, self.cfg.runs_dir / spec.runs_subdir / model.run_dir.name)
        self.assertTrue(model.run_dir.name.startswith("train-"))

    def test_run_names_carry_the_pid_for_cross_process_uniqueness(self) -> None:
        # Two processes share a clock: microseconds alone do not separate them
        # under ``exist_ok=False``, so the name carries the PID as well.
        self.assertTrue(_run_timestamp().endswith(f"-{os.getpid()}"))


class ExportTests(_SandboxedTestCase):
    def test_requests_onnx_with_the_configured_opset(self) -> None:
        self.cfg.opset = 17
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            trained = backend.train(MODEL_SPECS["det"], self.cfg, lambda epoch, total: None)
            produced = backend.export(trained, self.cfg)

        assert _FakeYOLO.last is not None
        self.assertEqual(_FakeYOLO.last.export_kwargs, {"format": "onnx", "opset": 17})
        self.assertEqual(produced, self.exported_onnx)


if __name__ == "__main__":
    unittest.main()
