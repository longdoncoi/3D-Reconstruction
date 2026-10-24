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
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from cvtrain.backend import (
    TrainedModel,
    TrainingBackend,
    TrainPlan,
    UltralyticsBackend,
    _device_name,
    _normalise_device,
    _run_timestamp,
    exporter_identity,
    metrics_from,
    train_kwargs,
)
from cvtrain.config import MODEL_REGISTRY, TrainConfig
from cvtrain.dataset import dataset_content_fingerprint
from cvtrain.errors import BackendUnavailableError, ResumeError


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

            def plan(self, spec: Any, cfg: TrainConfig, *, dataset_content_sha256: Any = None) -> Any:  # pragma: no cover
                raise NotImplementedError

            def train(self, spec: Any, cfg: TrainConfig, on_epoch: Any, plan: Any) -> TrainedModel[Any]:  # pragma: no cover
                raise NotImplementedError

            def check_device(self, cfg: TrainConfig) -> tuple[bool, str]:  # pragma: no cover
                raise NotImplementedError

            def check_importable(self, cfg: TrainConfig) -> tuple[bool, str]:  # pragma: no cover
                raise NotImplementedError

            def resolve_device(self, cfg: TrainConfig) -> str:  # pragma: no cover
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

    def test_rejects_booleans(self) -> None:
        # ``True`` is not a measurement: JSON ``true`` must never clear a
        # quality-gate threshold.
        self.assertEqual(metrics_from(_Results({"flag": True, "ok": 0.5})), {"ok": 0.5})

    def test_rejects_non_finite_values(self) -> None:
        # ``NaN`` compares false against every threshold (so it would pass a
        # gate) and ``Infinity`` is not valid JSON.
        metrics = metrics_from(_Results({"nan": float("nan"), "inf": float("inf"), "ok": 0.5}))
        self.assertEqual(metrics, {"ok": 0.5})

    def test_rejects_a_number_that_overflows_a_float(self) -> None:
        # ``float(10**400)`` raises OverflowError; a metric the gate cannot
        # read is "not reported", never a crash.
        self.assertEqual(metrics_from(_Results({"huge": 10**400, "ok": 0.5})), {"ok": 0.5})


class DeviceSpellingTests(unittest.TestCase):
    """The one spelling of a device request, shared by provenance and the gate."""

    def test_normalises_auto_and_separators(self) -> None:
        self.assertIsNone(_normalise_device(None))
        self.assertEqual(_normalise_device("auto"), "auto")
        self.assertEqual(_normalise_device("  AUTO  "), "auto")
        self.assertEqual(_normalise_device("0, 1"), "0,1")
        self.assertEqual(_normalise_device("CPU"), "cpu")

    def test_reports_a_concrete_device_or_nothing(self) -> None:
        self.assertIsNone(_device_name(None))
        self.assertIsNone(_device_name("   "))
        self.assertEqual(_device_name(" cuda:0 "), "cuda:0")


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
    epochs: int | None = None
    completed_epoch: int | None = None
    trainer_args: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    export_path = ""

    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.callbacks: dict[str, Any] = {}
        self.trainer = SimpleNamespace()
        self.ckpt = _FakeYOLO.checkpoint
        self.train_kwargs: dict[str, Any] = {}
        self.export_kwargs: dict[str, Any] = {}
        _FakeYOLO.last = self

    def add_callback(self, event: str, callback: Any) -> None:
        self.callbacks[event] = callback

    def train(self, **kwargs: Any) -> _Results:
        self.train_kwargs = kwargs
        if _FakeYOLO.save_dir is not None:
            self.trainer.save_dir = _FakeYOLO.save_dir
        if _FakeYOLO.epochs is not None:
            self.trainer.epochs = _FakeYOLO.epochs
        if _FakeYOLO.completed_epoch is not None:
            self.trainer.epoch = _FakeYOLO.completed_epoch
        if _FakeYOLO.trainer_args is not None:
            self.trainer.args = SimpleNamespace(**_FakeYOLO.trainer_args)
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
    epochs: int | None = None,
    completed_epoch: int | None = None,
    trainer_args: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> Iterator[tuple[Any, Any]]:
    """Install a fake ``ultralytics`` package for the duration of the block."""

    if broken:
        with mock.patch.dict(sys.modules, {"ultralytics": None, "ultralytics.utils": None}):
            yield None, None
        return

    _FakeYOLO.last = None
    _FakeYOLO.save_dir = save_dir
    _FakeYOLO.export_path = export_path
    _FakeYOLO.epochs = epochs
    _FakeYOLO.completed_epoch = completed_epoch
    _FakeYOLO.trainer_args = trainer_args
    _FakeYOLO.checkpoint = checkpoint

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
        # The resume check fingerprints the dataset on both sides, so the
        # sandbox needs a real YAML file rather than a path that does not
        # exist (which would be "cannot prove the data" instead of a mismatch).
        self.cfg.data_yaml.write_text(
            "train: images/train\nval: images/val\nnames:\n  0: part\n", encoding="utf-8"
        )
        self.exported_onnx = self.root / "generated.onnx"
        self.exported_onnx.write_bytes(b"onnx-bytes")

    def _dataset_digest(self, cfg: TrainConfig | None = None) -> str | None:
        return dataset_content_fingerprint((cfg or self.cfg).data_yaml)

    def _train(
        self,
        backend: UltralyticsBackend,
        spec: Any,
        cfg: TrainConfig,
        on_epoch: Any,
    ) -> TrainedModel[Any]:
        """Plan then train, the way the orchestrator does."""

        plan: TrainPlan = backend.plan(spec, cfg, dataset_content_sha256=self._dataset_digest(cfg))
        return backend.train(spec, cfg, on_epoch, plan)

    @staticmethod
    def _checkpoint_args(cfg: TrainConfig, *, epochs: int = 3, epoch: int = 0, data: Path | None = None) -> dict[str, Any]:
        """The subset of a real Ultralytics checkpoint the adapter reads."""

        return {
            "epoch": epoch,
            "train_args": {
                "epochs": epochs,
                "data": str(data if data is not None else cfg.data_yaml),
                "imgsz": cfg.imgsz,
                "batch": cfg.batch,
                "seed": cfg.seed,
                "device": cfg.device,
            },
        }


class LazyImportTests(_SandboxedTestCase):
    def test_declares_the_modules_preflight_must_verify(self) -> None:
        # A method, not a constant: an optional feature can add an optional
        # dependency, and the adapter is the one module that knows which.
        self.assertEqual(
            UltralyticsBackend().required_modules(self.cfg),
            ("ultralytics", "yaml", "onnx", "onnxslim", "onnxruntime"),
        )

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

    def test_version_records_the_whole_export_stack_on_first_use(self) -> None:
        # Ultralytics delegates the export to torch/onnx and simplifies the
        # graph with onnxslim, so every one of those identities belongs in the
        # provenance.
        with _stub_ultralytics():
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
        fields = dict(part.split("=", 1) for part in backend.version.split(";"))
        self.assertEqual(fields["cvtrain"], "1.0.0")
        self.assertEqual(fields["ultralytics"], "8.4.41-stub")
        self.assertIn("torch", fields)
        self.assertIn("onnx", fields)
        self.assertIn("onnxslim", fields)

    def test_the_identity_carries_the_actual_version_of_each_dependency(self) -> None:
        # The *value* of every field, not only its presence. Asserting that the
        # key exists is not enough: an identity that hardcoded ``onnx=unknown``
        # would still carry the key, would still pass the test above, and would
        # silently disable the rule this function exists for — an ``onnx``
        # upgrade has to invalidate a ``--skip-unchanged`` verdict, because
        # ``onnx`` can change the exported graph.
        def _versions(name: str) -> str:
            return {"torch": "2.7.0", "onnx": "1.21.0", "onnxslim": "0.1.91"}.get(name, "0.0.0")

        with mock.patch("cvtrain.backend.metadata.version", side_effect=_versions):
            identity = exporter_identity("8.4.41")
        self.assertEqual(
            identity,
            "cvtrain=1.0.0;ultralytics=8.4.41;torch=2.7.0;onnx=1.21.0;onnxslim=0.1.91",
        )

    def test_a_moved_dependency_changes_the_exporter_identity(self) -> None:
        # The second half of the rule above: each field must be *read* from the
        # installed distribution, so a dependency that moves is visible in the
        # manifest. A hardcoded constant would make every identity identical.
        def _versions(name: str) -> str:
            return {"torch": "2.7.0", "onnx": "1.21.0", "onnxslim": "0.1.91"}.get(name, "0.0.0")

        with mock.patch("cvtrain.backend.metadata.version", side_effect=_versions):
            first = exporter_identity("8.4.41")
        with mock.patch(
            "cvtrain.backend.metadata.version",
            side_effect=lambda name: {"torch": "2.8.0", "onnx": "1.21.0", "onnxslim": "0.1.91"}.get(name, "0.0.0"),
        ):
            second = exporter_identity("8.4.41")
        self.assertNotEqual(first, second, "a torch upgrade must change the exporter identity")

    def test_a_moved_simplifier_changes_the_exporter_identity(self) -> None:
        # Ultralytics exports with ``simplify=True`` by default and runs
        # onnxslim over the graph. A different simplifier emits different
        # bytes under the same ultralytics/torch/onnx versions, so it has to
        # invalidate --skip-unchanged on its own.
        def _versions(name: str) -> str:
            return {"torch": "2.7.0", "onnx": "1.21.0", "onnxslim": "0.1.91"}.get(name, "0.0.0")

        with mock.patch("cvtrain.backend.metadata.version", side_effect=_versions):
            before = exporter_identity("8.4.41")
        with mock.patch(
            "cvtrain.backend.metadata.version",
            side_effect=lambda name: {"torch": "2.7.0", "onnx": "1.21.0", "onnxslim": "0.2.0"}.get(name, "0.0.0"),
        ):
            after = exporter_identity("8.4.41")
        self.assertNotEqual(before, after, "an onnxslim upgrade must change the exporter identity")

    def test_a_missing_export_dependency_is_named_not_omitted(self) -> None:
        # "unknown" is a fact about the environment; silently dropping the key
        # would let a partial stack look like a complete identity.
        def _missing(name: str) -> str:
            raise metadata.PackageNotFoundError(name)

        with mock.patch("cvtrain.backend.metadata.version", side_effect=_missing):
            identity = exporter_identity("9.9.9")
        self.assertEqual(
            identity,
            "cvtrain=1.0.0;ultralytics=9.9.9;torch=unknown;onnx=unknown;onnxslim=unknown",
        )

    def test_missing_dependency_raises_a_typed_error_not_a_traceback(self) -> None:
        backend = UltralyticsBackend()
        with _stub_ultralytics(broken=True):
            with self.assertRaises(BackendUnavailableError) as ctx:
                backend.prepare(self.cfg)
        self.assertIn("pip install", str(ctx.exception))
        self.assertIn("cvtrain", type(ctx.exception).__module__)


class PrepareTests(_SandboxedTestCase):
    def test_tensorboard_is_enabled_only_when_requested(self) -> None:
        # The probe must succeed whether or not the package happens to be
        # installed on the machine running the suite (CI installs none), so
        # the import is stubbed in — the test exercises the settings wiring,
        # not the environment.
        self.cfg.tensorboard = True
        with (
            _stub_ultralytics() as (_, settings),
            mock.patch.dict(sys.modules, {"tensorboard": types.ModuleType("tensorboard")}),
        ):
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
        # Stub the package in (as above): without it the probe would fail
        # first on CI and the no-settings branch would never be reached.
        with (
            mock.patch.dict(sys.modules, {"tensorboard": types.ModuleType("tensorboard")}),
            self.assertLogs("cvtrain.backend", level="WARNING") as captured,
        ):
            backend.prepare(self.cfg)
        self.assertTrue(any("TensorBoard logging stays off" in line for line in captured.output))


class DeviceResolutionTests(_SandboxedTestCase):
    """``auto`` is resolved by the pipeline, before any weight is touched."""

    def _torch(self, *, cuda: bool) -> types.ModuleType:
        module = types.ModuleType("torch")
        setattr(module, "cuda", SimpleNamespace(is_available=lambda: cuda))
        return module

    def test_auto_resolves_to_the_first_cuda_device(self) -> None:
        with mock.patch.dict(sys.modules, {"torch": self._torch(cuda=True)}):
            self.assertEqual(UltralyticsBackend().resolve_device(TrainConfig()), "cuda:0")

    def test_auto_resolves_to_cpu_without_a_gpu(self) -> None:
        with mock.patch.dict(sys.modules, {"torch": self._torch(cuda=False)}):
            self.assertEqual(UltralyticsBackend().resolve_device(TrainConfig()), "cpu")

    def test_auto_resolves_to_cpu_without_torch(self) -> None:
        # A machine without the ML stack cannot have a GPU either.
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(UltralyticsBackend().resolve_device(TrainConfig()), "cpu")

    def test_an_explicit_device_is_its_own_answer(self) -> None:
        # No probing, no policy: the operator named the device.
        backend = UltralyticsBackend()
        self.assertEqual(backend.resolve_device(TrainConfig(device="0, 1")), "0,1")
        self.assertEqual(backend.resolve_device(TrainConfig(device="cpu")), "cpu")
        self.assertEqual(backend.resolve_device(TrainConfig(device="cuda:0")), "cuda:0")

    def test_the_resolved_device_is_what_the_framework_is_told_to_use(self) -> None:
        cfg = TrainConfig(models=("det",), device_resolved="cuda:0")
        self.assertEqual(train_kwargs(cfg)["device"], "cuda:0")
        # Without a resolution, an explicit request is still forwarded.
        self.assertEqual(train_kwargs(TrainConfig(device="cpu"))["device"], "cpu")
        self.assertNotIn("device", train_kwargs(TrainConfig()))


class ImportabilityTests(_SandboxedTestCase):
    """Preflight must prove the stack can be *loaded*, not only installed."""

    def test_a_working_stack_is_reported_importable(self) -> None:
        # Every module the adapter declared is imported, not just the
        # framework: a package that is installed and broken would otherwise
        # pass preflight and fail at export, after the expensive training.
        with _stub_ultralytics():
            with mock.patch("cvtrain.backend.import_module") as probe:
                ok, detail = UltralyticsBackend().check_importable(self.cfg)
        self.assertTrue(ok)
        self.assertIn("importable", detail)
        self.assertEqual(
            [call.args[0] for call in probe.call_args_list],
            ["ultralytics", "yaml", "onnx", "onnxslim", "onnxruntime"],
        )

    def test_a_broken_install_is_reported_not_raised(self) -> None:
        # "Installed" is not "usable": the distribution is on disk, but its own
        # dependency is missing. The preflight report must say so — and it
        # must be a verdict, not an exception, because the report is where an
        # operator reads what is wrong with their environment.
        with _stub_ultralytics(broken=True):
            ok, detail = UltralyticsBackend().check_importable(self.cfg)
        self.assertFalse(ok)
        self.assertIn("not importable", detail)

    def test_a_declared_dependency_that_cannot_import_is_named(self) -> None:
        # ``find_spec`` sees onnxslim, but the import is what proves it works.
        # The verdict names the package and the fix, not a traceback.
        def _probe(name: str) -> object:
            if name == "onnxslim":
                raise ImportError("DLL load failed while importing onnxslim")
            return object()

        with _stub_ultralytics():
            with mock.patch("cvtrain.backend.import_module", side_effect=_probe):
                ok, detail = UltralyticsBackend().check_importable(self.cfg)
        self.assertFalse(ok)
        self.assertIn("onnxslim is not importable", detail)
        self.assertIn("pip install", detail)

    def test_a_native_load_error_is_a_verdict_not_a_traceback(self) -> None:
        # A Windows DLL failure raises OSError, not ImportError. Both mean the
        # stack cannot be loaded, and neither may escape the preflight report.
        with _stub_ultralytics():
            with mock.patch("cvtrain.backend.import_module", side_effect=OSError("DLL load failed")):
                ok, detail = UltralyticsBackend().check_importable(self.cfg)
        self.assertFalse(ok)
        self.assertIn("ultralytics is not importable", detail)


class TrainTests(_SandboxedTestCase):
    def _write_checkpoint(self, run_name: str, age: float = 0.0) -> Path:
        """A ``last.pt`` in the layout the framework writes."""

        weights = self.cfg.runs_dir / "detect" / run_name / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        checkpoint = weights / "last.pt"
        checkpoint.write_bytes(b"checkpoint")
        if age:
            stamp = checkpoint.stat().st_mtime - age
            os.utime(checkpoint, (stamp, stamp))
        return checkpoint

    def _resume_cfg(self) -> TrainConfig:
        return TrainConfig(
            models=("det",),
            data_yaml=self.cfg.data_yaml,
            models_dir=self.cfg.models_dir,
            runs_dir=self.cfg.runs_dir,
            epochs=3,
            resume=True,
        )

    def test_resume_continues_from_the_newest_checkpoint(self) -> None:
        # A training run is hours of work: an interrupted one continues from
        # the framework's own checkpoint, not from the base weights.
        older = self._write_checkpoint("train-old", age=600.0)
        newest = self._write_checkpoint("train-new", age=1.0)
        cfg = self._resume_cfg()
        with _stub_ultralytics(checkpoint=self._checkpoint_args(cfg)):
            backend = UltralyticsBackend()
            model = self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        assert _FakeYOLO.last is not None
        self.assertEqual(_FakeYOLO.last.weights, str(newest))
        self.assertNotEqual(_FakeYOLO.last.weights, str(older))
        # ``resume=True`` restores the interrupted run's own arguments from the
        # checkpoint, so the hyper-parameters of this invocation are not
        # silently applied to it.
        self.assertEqual(_FakeYOLO.last.train_kwargs, {"resume": True})
        self.assertEqual(model.facts.resumed_from, str(newest))
        self.assertEqual(model.facts.source_weights, str(newest))

    def test_resuming_claims_the_run_lease_before_training(self) -> None:
        # The interrupted run's marker still names the killed process. The
        # resumed run must re-claim it *before* the first epoch, or a
        # concurrent --prune-runs reads a live run as abandoned for hours.
        checkpoint = self._write_checkpoint("train-crashed")
        cfg = self._resume_cfg()
        with _stub_ultralytics(checkpoint=self._checkpoint_args(cfg)):
            backend = UltralyticsBackend()
            with mock.patch("cvtrain.backend.claim_run_marker") as claim:
                self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        claim.assert_called_once_with(checkpoint.parent.parent)

    def test_a_fresh_run_does_not_claim_a_run_lease(self) -> None:
        # A fresh run has no directory yet, so there is no lease to take; the
        # epoch callback creates the marker once the framework names its dir.
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            with mock.patch("cvtrain.backend.claim_run_marker") as claim:
                self._train(backend, MODEL_REGISTRY["det"], self.cfg, lambda epoch, total: None)
        claim.assert_not_called()

    def test_resume_without_a_previous_run_is_refused(self) -> None:
        # Starting over would overwrite the artifact the operator asked to
        # continue. "There is nothing to resume" is a verdict, not a warning.
        with _stub_ultralytics():
            backend = UltralyticsBackend()
            with self.assertRaises(ResumeError) as ctx:
                backend.plan(MODEL_REGISTRY["det"], self._resume_cfg(), dataset_content_sha256="abc")
        self.assertIn("no run checkpoint", str(ctx.exception))
        self.assertIsNone(_FakeYOLO.last)

    def test_resume_refuses_a_completed_checkpoint(self) -> None:
        # A finished run's ``last.pt`` has epoch == -1. Ultralytics asserts when
        # asked to resume it; the pipeline owes the operator a verdict instead.
        self._write_checkpoint("train-finished")
        cfg = self._resume_cfg()
        checkpoint = self._checkpoint_args(cfg, epoch=-1)
        with _stub_ultralytics(checkpoint=checkpoint):
            backend = UltralyticsBackend()
            with self.assertRaises(ResumeError) as ctx:
                self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        self.assertIn("already complete", str(ctx.exception))

    def test_resume_refuses_a_checkpoint_without_training_arguments(self) -> None:
        # The framework would fill the missing arguments from its own defaults;
        # that is not the experiment the operator named.
        self._write_checkpoint("train-bare")
        cfg = self._resume_cfg()
        with _stub_ultralytics(checkpoint={"epoch": 0}):
            backend = UltralyticsBackend()
            with self.assertRaises(ResumeError) as ctx:
                self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        self.assertIn("no training arguments", str(ctx.exception))

    def test_resume_refuses_a_dataset_that_is_not_on_disk(self) -> None:
        # Ultralytics quietly substitutes the invocation's ``--data`` when the
        # checkpoint's path is gone. That silent mix is exactly what this check
        # turns into a refusal.
        self._write_checkpoint("train-moved")
        cfg = self._resume_cfg()
        checkpoint = self._checkpoint_args(cfg, data=self.root / "gone.yaml")
        with _stub_ultralytics(checkpoint=checkpoint):
            backend = UltralyticsBackend()
            with self.assertRaises(ResumeError) as ctx:
                self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        self.assertIn("not on disk", str(ctx.exception))

    def test_resume_refuses_a_checkpoint_trained_on_other_data(self) -> None:
        other = self.root / "other.yaml"
        other.write_text("train: images/train\nval: images/val\nnames:\n  0: other\n", encoding="utf-8")
        self._write_checkpoint("train-other-data")
        cfg = self._resume_cfg()
        checkpoint = self._checkpoint_args(cfg, data=other)
        with _stub_ultralytics(checkpoint=checkpoint):
            backend = UltralyticsBackend()
            with self.assertRaises(ResumeError) as ctx:
                self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        self.assertIn("does not match", str(ctx.exception))

    def test_resume_logs_the_values_the_checkpoint_overrides(self) -> None:
        # Only the dataset is a hard refusal. imgsz/batch/seed/device/epochs are
        # the checkpoint's own experiment and win, but the operator must be able
        # to see that the command line was not applied.
        self._write_checkpoint("train-resume")
        cfg = self._resume_cfg()
        checkpoint = self._checkpoint_args(cfg, epochs=10)
        checkpoint["train_args"]["imgsz"] = 64
        checkpoint["train_args"]["device"] = "0"
        with _stub_ultralytics(checkpoint=checkpoint, epochs=10):
            backend = UltralyticsBackend()
            with self.assertLogs("cvtrain.backend", level="WARNING") as captured:
                self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        joined = "\n".join(captured.output)
        self.assertIn("epochs=10", joined)
        self.assertIn("imgsz=64", joined)
        # The device comparison is normalised, not textual: the log names the
        # checkpoint's value so the operator can see it won.
        self.assertIn("device=0", joined)

    def test_resume_refuses_a_checkpoint_without_training_state(self) -> None:
        # A ``last.pt`` the framework cannot describe is not a checkpoint this
        # pipeline can continue; the framework would fill the gap with defaults.
        self._write_checkpoint("train-stateless")
        cfg = self._resume_cfg()
        with _stub_ultralytics(checkpoint=None):
            backend = UltralyticsBackend()
            with self.assertRaises(ResumeError) as ctx:
                self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        self.assertIn("no training state", str(ctx.exception))

    def test_resume_refuses_arguments_without_a_dataset(self) -> None:
        self._write_checkpoint("train-no-data")
        cfg = self._resume_cfg()
        checkpoint = {"epoch": 0, "train_args": {"epochs": 5}}
        with _stub_ultralytics(checkpoint=checkpoint):
            backend = UltralyticsBackend()
            with self.assertRaises(ResumeError) as ctx:
                self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        self.assertIn("records no dataset", str(ctx.exception))

    def test_resume_refuses_when_the_requested_dataset_cannot_be_fingerprinted(self) -> None:
        # Without a digest of the invocation's data there is no proof the two
        # sides are the same experiment, and "no proof" is a refusal.
        checkpoint_path = self._write_checkpoint("train-unprovable")
        cfg = self._resume_cfg()
        plan = TrainPlan(
            source_weights=checkpoint_path,
            resume_from=checkpoint_path,
            dataset_content_sha256=None,
        )
        with _stub_ultralytics(checkpoint=self._checkpoint_args(cfg)):
            backend = UltralyticsBackend()
            with self.assertRaises(ResumeError) as ctx:
                backend.train(MODEL_REGISTRY["det"], cfg, lambda epoch, total: None, plan)
        self.assertIn("cannot fingerprint", str(ctx.exception))

    def test_resume_accepts_the_same_data_at_another_path(self) -> None:
        # The experiment is the bytes, not the checkout location: a dataset
        # moved to another directory is still the same experiment, and refusing
        # it would make --resume useless after moving the project.
        copy = self.root / "data-copy.yaml"
        copy.write_bytes(self.cfg.data_yaml.read_bytes())
        checkpoint_path = self._write_checkpoint("train-copy")
        cfg = self._resume_cfg()
        checkpoint = self._checkpoint_args(cfg, data=copy)
        with _stub_ultralytics(checkpoint=checkpoint):
            backend = UltralyticsBackend()
            model = self._train(backend, MODEL_REGISTRY["det"], cfg, lambda epoch, total: None)
        self.assertEqual(model.facts.resumed_from, str(checkpoint_path))

    def test_without_resume_the_base_checkpoint_is_used(self) -> None:
        self._write_checkpoint("train-old")
        with _stub_ultralytics():
            backend = UltralyticsBackend()
            plan = backend.plan(MODEL_REGISTRY["det"], self.cfg, dataset_content_sha256=self._dataset_digest())
            self.assertEqual(plan.source_weights, self.cfg.weights_path(MODEL_REGISTRY["det"]))
            self.assertIsNone(plan.resume_from)
            backend.train(MODEL_REGISTRY["det"], self.cfg, lambda epoch, total: None, plan)
        assert _FakeYOLO.last is not None
        self.assertEqual(_FakeYOLO.last.weights, str(self.cfg.weights_path(MODEL_REGISTRY["det"])))

    def test_loads_the_checkpoint_declared_by_the_model_spec(self) -> None:
        spec = MODEL_REGISTRY["det"]
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            self._train(backend, spec, self.cfg, lambda epoch, total: None)
        assert _FakeYOLO.last is not None
        self.assertEqual(Path(_FakeYOLO.last.weights), self.cfg.weights_path(spec))

    def test_forwards_hyperparameters_and_uses_the_run_subdirectory(self) -> None:
        spec = MODEL_REGISTRY["det"]
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            self._train(backend, spec, self.cfg, lambda epoch, total: None)
        assert _FakeYOLO.last is not None
        kwargs = _FakeYOLO.last.train_kwargs
        self.assertEqual(kwargs["data"], str(self.cfg.data_yaml))
        self.assertEqual(kwargs["epochs"], 3)
        self.assertEqual(Path(kwargs["project"]), self.cfg.runs_dir / spec.runs_subdir)
        self.assertTrue(Path(kwargs["name"]).name.startswith("train-"))
        # exist_ok=False: a re-run must not silently overwrite the last run.
        self.assertIs(kwargs["exist_ok"], False)

    def test_registers_the_epoch_callback_and_reports_progress(self) -> None:
        spec = MODEL_REGISTRY["det"]
        seen: list[tuple[int, int]] = []
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            self._train(backend, spec, self.cfg, lambda epoch, total: seen.append((epoch, total)))
        assert _FakeYOLO.last is not None
        self.assertIn("on_train_epoch_end", _FakeYOLO.last.callbacks)

        _FakeYOLO.last.callbacks["on_train_epoch_end"](SimpleNamespace(epoch=1, epochs=3))
        self.assertEqual(seen, [(1, 3)])

    def test_the_epoch_callback_refreshes_the_run_lease(self) -> None:
        # A long epoch must not make a live run look abandoned to --prune-runs.
        spec = MODEL_REGISTRY["det"]
        run_dir = self.root / "runs" / "detect" / "train-live"
        with _stub_ultralytics(save_dir=str(run_dir), export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            with mock.patch("cvtrain.backend.touch_run_marker") as touch:
                self._train(backend, spec, self.cfg, lambda epoch, total: None)
                assert _FakeYOLO.last is not None
                _FakeYOLO.last.callbacks["on_train_epoch_end"](
                    SimpleNamespace(epoch=0, epochs=3, save_dir=str(run_dir))
                )
        touch.assert_called_once_with(run_dir)

    def test_a_finished_run_is_not_left_marked_as_live(self) -> None:
        # The process is still alive after training returns; leaving the marker
        # would make the pruner keep every finished run of this session.
        spec = MODEL_REGISTRY["det"]
        run_dir = self.root / "runs" / "detect" / "train-done"
        with _stub_ultralytics(save_dir=str(run_dir), export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            with mock.patch("cvtrain.backend.clear_run_marker") as clear:
                self._train(backend, spec, self.cfg, lambda epoch, total: None)
        clear.assert_called_once_with(run_dir)

    def test_missing_epoch_attributes_fall_back_safely(self) -> None:
        spec = MODEL_REGISTRY["det"]
        seen: list[tuple[int, int]] = []
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            self._train(backend, spec, self.cfg, lambda epoch, total: seen.append((epoch, total)))
        assert _FakeYOLO.last is not None
        _FakeYOLO.last.callbacks["on_train_epoch_end"](SimpleNamespace())
        self.assertEqual(seen, [(0, self.cfg.epochs)])

    def test_returns_metrics_and_the_framework_reported_run_dir(self) -> None:
        spec = MODEL_REGISTRY["det"]
        run_dir = self.root / "runs" / "detect" / "train-20260101-000000"
        with _stub_ultralytics(save_dir=str(run_dir), export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            model = self._train(backend, spec, self.cfg, lambda epoch, total: None)
        self.assertEqual(model.metrics, {"metrics/mAP50(B)": 0.77})
        self.assertEqual(model.run_dir, run_dir)

    def test_run_dir_falls_back_to_the_project_directory(self) -> None:
        # A trainer object without save_dir must not crash the pipeline.
        spec = MODEL_REGISTRY["det"]
        with _stub_ultralytics(save_dir=None, export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            model = self._train(backend, spec, self.cfg, lambda epoch, total: None)
        self.assertEqual(model.run_dir, self.cfg.runs_dir / spec.runs_subdir / model.run_dir.name)
        self.assertTrue(model.run_dir.name.startswith("train-"))

    def test_run_names_carry_the_pid_for_cross_process_uniqueness(self) -> None:
        # Two processes share a clock: microseconds alone do not separate them
        # under ``exist_ok=False``, so the name carries the PID as well.
        self.assertTrue(_run_timestamp().endswith(f"-{os.getpid()}"))

    def test_reports_the_effective_budget_and_the_completed_count(self) -> None:
        # The budget and the completed count are different facts: early
        # stopping can finish a 10-epoch budget after 4 epochs. Recording only
        # one of them would describe a run that never happened.
        spec = MODEL_REGISTRY["det"]
        with _stub_ultralytics(
            export_path=str(self.exported_onnx), epochs=10, completed_epoch=3
        ):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            model = self._train(backend, spec, self.cfg, lambda epoch, total: None)
        self.assertEqual(model.epochs, 10)
        self.assertEqual(model.epochs_completed, 4)
        self.assertNotEqual(self.cfg.epochs, 10)

    def test_effective_facts_come_from_the_trainer_arguments(self) -> None:
        # The framework may resolve auto-batch and image size itself; the
        # provenance must record what it used, not what was requested.
        spec = MODEL_REGISTRY["det"]
        with _stub_ultralytics(
            export_path=str(self.exported_onnx),
            epochs=10,
            trainer_args={
                "epochs": 10,
                "data": str(self.cfg.data_yaml),
                "imgsz": 320,
                "batch": 8,
                "seed": 7,
                "device": "cpu",
            },
        ):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            model = self._train(backend, spec, self.cfg, lambda epoch, total: None)
        self.assertEqual(model.facts.dataset, str(self.cfg.data_yaml))
        self.assertEqual(model.facts.imgsz, 320)
        self.assertEqual(model.facts.batch, 8)
        self.assertEqual(model.facts.seed, 7)
        self.assertEqual(model.facts.device, "cpu")

    def test_reports_no_epoch_count_when_the_framework_cannot_say(self) -> None:
        # "Cannot prove" stays absent: the recorder then keeps the request
        # rather than inventing a number.
        spec = MODEL_REGISTRY["det"]
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            model = self._train(backend, spec, self.cfg, lambda epoch, total: None)
        self.assertIsNone(model.epochs)
        self.assertIsNone(model.epochs_completed)


class ExportTests(_SandboxedTestCase):
    def test_requests_onnx_with_the_configured_opset(self) -> None:
        self.cfg.opset = 17
        with _stub_ultralytics(export_path=str(self.exported_onnx)):
            backend = UltralyticsBackend()
            backend.prepare(self.cfg)
            trained = self._train(backend, MODEL_REGISTRY["det"], self.cfg, lambda epoch, total: None)
            produced = backend.export(trained, self.cfg)

        assert _FakeYOLO.last is not None
        self.assertEqual(_FakeYOLO.last.export_kwargs, {"format": "onnx", "opset": 17})
        self.assertEqual(produced, self.exported_onnx)


class CheckDeviceTests(unittest.TestCase):
    def test_default_or_cpu_always_succeeds(self) -> None:
        backend = UltralyticsBackend()
        self.assertEqual(backend.check_device(TrainConfig(device=None)), (True, "device ok"))
        self.assertEqual(backend.check_device(TrainConfig(device="auto")), (True, "device ok"))
        self.assertEqual(backend.check_device(TrainConfig(device="cpu")), (True, "device ok"))
        self.assertEqual(backend.check_device(TrainConfig(device="mps")), (True, "device ok"))

    def test_cuda_device_passes_when_torch_cuda_is_available(self) -> None:
        backend = UltralyticsBackend()
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            ok, msg = backend.check_device(TrainConfig(device="cuda"))
            self.assertTrue(ok)
            self.assertEqual(msg, "device ok")

    def test_cuda_device_fails_when_torch_cuda_is_unavailable(self) -> None:
        backend = UltralyticsBackend()
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            ok, msg = backend.check_device(TrainConfig(device="0"))
            self.assertFalse(ok)
            self.assertIn("CUDA is not available", msg)

    def test_cuda_device_passes_when_torch_is_not_installed(self) -> None:
        backend = UltralyticsBackend()
        with mock.patch.dict("sys.modules", {"torch": None}):
            ok, _msg = backend.check_device(TrainConfig(device="0,1"))
            self.assertTrue(ok)



if __name__ == "__main__":
    unittest.main()
