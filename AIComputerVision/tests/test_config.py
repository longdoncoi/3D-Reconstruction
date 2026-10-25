"""Unit tests for the model contract and configuration resolution."""

from __future__ import annotations

import unittest
from pathlib import Path

from cvtrain.config import (
    DEFAULT_MODELS,
    MANIFEST_NAME,
    MODEL_REGISTRY,
    PACKAGE_DIR,
    ModelRegistry,
    ModelSpec,
    TrainConfig,
    canonical_device,
    register_model_spec,
    resolve_models,
)


class ModelSpecTests(unittest.TestCase):
    def test_keys_match_spec_map(self) -> None:
        self.assertEqual(MODEL_REGISTRY.keys(), ("det", "seg", "track"))
        for key, spec in MODEL_REGISTRY.items():
            self.assertEqual(key, spec.key)

    def test_exported_names_are_unique(self) -> None:
        outputs = [spec.output for spec in MODEL_REGISTRY.values()]
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_expected_contract_names(self) -> None:
        # These names are asserted against AIConstants.h in test_contract.py.
        self.assertEqual(MODEL_REGISTRY["det"].output, "yolo11n.onnx")
        self.assertEqual(MODEL_REGISTRY["seg"].output, "yolo11n-seg.onnx")
        self.assertEqual(MODEL_REGISTRY["track"].output, "yolo11x-tracking.onnx")

    def test_the_tracking_artifact_is_named_as_a_detector(self) -> None:
        # The ONNX graph contains detection weights; association is Qt code.
        # The label must not promise tracking the model does not implement.
        self.assertEqual(MODEL_REGISTRY["track"].label, "Tracking detector (yolo11x)")


class ResolveModelsTests(unittest.TestCase):
    def test_empty_selection_defaults(self) -> None:
        self.assertEqual(resolve_models(()), DEFAULT_MODELS)
        self.assertEqual(resolve_models(None), DEFAULT_MODELS)

    def test_single_selection(self) -> None:
        self.assertEqual(resolve_models(["det"]), ("det",))
        self.assertEqual(resolve_models(["track"]), ("track",))

    def test_selection_is_canonically_ordered(self) -> None:
        self.assertEqual(resolve_models(["track", "det"]), ("det", "track"))

    def test_unknown_keys_are_rejected_instead_of_silently_defaulting(self) -> None:
        # A typo'd key used to fall back to det+seg: a plausible green run that
        # trained something the caller never asked for. Fail closed instead.
        with self.assertRaisesRegex(ValueError, "unknown model key"):
            resolve_models(["bogus"])
        with self.assertRaisesRegex(ValueError, "bogus"):
            resolve_models(["seg", "bogus"])

    def test_duplicate_selection_is_deduplicated_in_registry_order(self) -> None:
        self.assertEqual(resolve_models(["track", "det", "track"]), ("det", "track"))


class TrainConfigTests(unittest.TestCase):
    def test_device_class_normalises_spelling(self) -> None:
        # The canonical spelling used as provenance: blank means "auto", and
        # whitespace/case around a comma list is not a different experiment.
        self.assertEqual(TrainConfig().device_class, "auto")
        self.assertEqual(TrainConfig(device="   ").device_class, "auto")
        self.assertEqual(TrainConfig(device="0, 1").device_class, "0,1")
        self.assertEqual(TrainConfig(device="CPU").device_class, "cpu")
        self.assertEqual(TrainConfig(device="cuda:0").device_class, "cuda:0")

    def test_canonical_device_is_the_frameworks_spelling(self) -> None:
        # The request keeps its own spelling; the resolution (and the fact the
        # framework reports) must have one spelling, or an identical
        # ``--device 0`` re-run is never skipped (ADR 0016).
        self.assertEqual(canonical_device(None), "auto")
        self.assertEqual(canonical_device("   "), "auto")
        self.assertEqual(canonical_device("AUTO"), "auto")
        self.assertEqual(canonical_device("cpu"), "cpu")
        self.assertEqual(canonical_device("0"), "cuda:0")
        self.assertEqual(canonical_device(" 0, 1 "), "cuda:0,cuda:1")
        self.assertEqual(canonical_device("cuda"), "cuda:0")
        self.assertEqual(canonical_device("cuda:00"), "cuda:0")
        self.assertEqual(canonical_device("0,,1"), "cuda:0,cuda:1")
        self.assertEqual(canonical_device("0,cuda:1"), "cuda:0,cuda:1")
        self.assertEqual(canonical_device("mps"), "mps")

    def test_the_quality_gate_is_opt_in(self) -> None:
        # No threshold by default: behaviour for every existing caller (the Qt
        # dialog included) must stay exactly what it was.
        self.assertEqual(TrainConfig().min_metrics, ())


class PathResolutionTests(unittest.TestCase):
    def test_weights_path_resolves_from_the_package_root_by_default(self) -> None:
        # preflight, the backend and the provenance digest must agree on where
        # checkpoints live, so all three resolve the path through TrainConfig.
        cfg = TrainConfig()
        for key, spec in MODEL_REGISTRY.items():
            with self.subTest(key=key):
                self.assertEqual(cfg.weights_path(spec), PACKAGE_DIR.parent / spec.ckpt)
                self.assertTrue(str(cfg.weights_path(spec)).endswith(spec.ckpt))

    def test_weights_dir_override_moves_the_checkpoint_lookup(self) -> None:
        # Inputs move like outputs: a caller that redirects models_dir/runs_dir
        # can redirect the checkpoints the run starts from too.
        spec = MODEL_REGISTRY["det"]
        cfg = TrainConfig(weights_dir=Path("C:/elsewhere/weights"))
        self.assertEqual(cfg.weights_path(spec), Path("C:/elsewhere/weights") / spec.ckpt)

    def test_output_path_honours_a_models_dir_override(self) -> None:
        spec = MODEL_REGISTRY["det"]
        cfg = TrainConfig(models_dir=Path("C:/elsewhere/Models"))
        self.assertEqual(cfg.output_path(spec), Path("C:/elsewhere/Models") / spec.output)

    def test_manifest_path_sits_beside_the_models(self) -> None:
        cfg = TrainConfig(models_dir=Path("C:/elsewhere/Models"))
        self.assertEqual(cfg.manifest_path, Path("C:/elsewhere/Models") / MANIFEST_NAME)


class ModelRegistryTests(unittest.TestCase):
    def test_empty_registry(self) -> None:
        reg = ModelRegistry()
        self.assertEqual(len(reg), 0)
        self.assertEqual(reg.keys(), ())
        self.assertEqual(reg.values(), ())
        self.assertEqual(reg.items(), ())
        self.assertEqual(reg.as_dict(), {})
        self.assertNotIn("det", reg)

    def test_register_and_lookup(self) -> None:
        reg = ModelRegistry()
        spec = ModelSpec("custom", "c.pt", "c.onnx", "c", "custom_role", "Custom")
        reg.register(spec)
        self.assertEqual(len(reg), 1)
        self.assertIn("custom", reg)
        self.assertEqual(reg.get("custom"), spec)
        self.assertEqual(reg["custom"], spec)
        self.assertEqual(reg.keys(), ("custom",))
        self.assertEqual(reg.values(), (spec,))
        self.assertEqual(reg.items(), (("custom", spec),))
        self.assertEqual(reg.as_dict(), {"custom": spec})

    def test_register_type_error(self) -> None:
        reg = ModelRegistry()
        with self.assertRaises(TypeError):
            reg.register("not_a_spec")  # type: ignore[arg-type]

    def test_register_duplicate_key_rejected(self) -> None:
        reg = ModelRegistry()
        spec1 = ModelSpec("det", "d1.pt", "d1.onnx", "d1", "role", "label")
        spec2 = ModelSpec("det", "d2.pt", "d2.onnx", "d2", "role", "label")
        reg.register(spec1)
        with self.assertRaises(ValueError):
            reg.register(spec2)

    def test_register_duplicate_output_rejected(self) -> None:
        reg = ModelRegistry()
        spec1 = ModelSpec("det1", "d1.pt", "same.onnx", "d1", "role", "label")
        spec2 = ModelSpec("det2", "d2.pt", "same.onnx", "d2", "role", "label")
        reg.register(spec1)
        with self.assertRaises(ValueError):
            reg.register(spec2)

    def test_get_unknown_key_raises_key_error(self) -> None:
        reg = ModelRegistry()
        with self.assertRaises(KeyError):
            reg.get("nonexistent")

    def test_register_model_spec_updates_registry(self) -> None:
        spec = ModelSpec("pose_test", "pose.pt", "pose.onnx", "pose", "pose", "Pose")
        try:
            register_model_spec(spec)
            self.assertIn("pose_test", MODEL_REGISTRY)
            self.assertEqual(MODEL_REGISTRY.get("pose_test"), spec)
            self.assertIn("pose_test", MODEL_REGISTRY.keys())
        finally:
            MODEL_REGISTRY._specs.pop("pose_test", None)

    def test_registered_model_is_selectable_end_to_end(self) -> None:
        # The point of a *live* registry: a model registered before the pipeline
        # runs is accepted everywhere a built-in model is — here through
        # preflight, the check that used to reject it against a static snapshot.
        from cvtrain.config import TrainConfig
        from cvtrain.preflight import run as preflight_run

        spec = ModelSpec("demo", "demo.pt", "demo.onnx", "demo", "detection", "Demo")
        try:
            register_model_spec(spec)
            cfg = TrainConfig(models=("demo",), weights_dir=Path("."))
            report = preflight_run(cfg, deps_fatal=False, modules=())
            failures = [check.line() for check in report.failures if check.name == "param.models"]
            self.assertEqual(failures, [], f"preflight rejected the registered model: {failures}")
        finally:
            MODEL_REGISTRY._specs.pop("demo", None)

    def test_resolve_models_with_custom_registry(self) -> None:
        spec = ModelSpec("custom", "c.pt", "c.onnx", "c", "role", "label")
        reg = ModelRegistry([spec])
        self.assertEqual(resolve_models(["custom"], registry=reg), ("custom",))
        # No silent fallback to names the registry does not contain.
        with self.assertRaisesRegex(ValueError, "legacy default"):
            resolve_models([], registry=reg)


if __name__ == "__main__":
    unittest.main()
