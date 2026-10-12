"""Unit tests for the model contract and configuration resolution."""

from __future__ import annotations

import unittest
from pathlib import Path

from cvtrain.config import (
    ALL_MODEL_KEYS,
    DEFAULT_MODELS,
    MANIFEST_NAME,
    MODEL_SPECS,
    PACKAGE_DIR,
    TrainConfig,
    resolve_models,
)


class ModelSpecTests(unittest.TestCase):
    def test_keys_match_spec_map(self) -> None:
        self.assertEqual(ALL_MODEL_KEYS, ("det", "seg", "track"))
        for key, spec in MODEL_SPECS.items():
            self.assertEqual(key, spec.key)

    def test_exported_names_are_unique(self) -> None:
        outputs = [spec.output for spec in MODEL_SPECS.values()]
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_expected_contract_names(self) -> None:
        # These names are asserted against AppConstants.h in test_contract.py.
        self.assertEqual(MODEL_SPECS["det"].output, "yolo11n.onnx")
        self.assertEqual(MODEL_SPECS["seg"].output, "yolo11n-seg.onnx")
        self.assertEqual(MODEL_SPECS["track"].output, "yolo11x-tracking.onnx")


class ResolveModelsTests(unittest.TestCase):
    def test_empty_selection_defaults(self) -> None:
        self.assertEqual(resolve_models(()), DEFAULT_MODELS)
        self.assertEqual(resolve_models(None), DEFAULT_MODELS)

    def test_single_selection(self) -> None:
        self.assertEqual(resolve_models(["det"]), ("det",))
        self.assertEqual(resolve_models(["track"]), ("track",))

    def test_selection_is_canonically_ordered(self) -> None:
        self.assertEqual(resolve_models(["track", "det"]), ("det", "track"))

    def test_unknown_keys_are_ignored(self) -> None:
        self.assertEqual(resolve_models(["bogus"]), DEFAULT_MODELS)
        self.assertEqual(resolve_models(["seg", "bogus"]), ("seg",))


class TrainConfigTests(unittest.TestCase):
    def test_train_kwargs_are_reproducible(self) -> None:
        cfg = TrainConfig(models=("det",), epochs=3, seed=7)
        kwargs = cfg.train_kwargs()
        self.assertEqual(kwargs["epochs"], 3)
        self.assertEqual(kwargs["seed"], 7)
        self.assertIs(kwargs["deterministic"], True)
        self.assertIs(kwargs["exist_ok"], False)
        self.assertNotIn("device", kwargs)

    def test_device_is_forwarded_when_set(self) -> None:
        cfg = TrainConfig(device="cpu")
        self.assertEqual(cfg.train_kwargs()["device"], "cpu")

    def test_data_path_is_stringified(self) -> None:
        cfg = TrainConfig(data_yaml=Path("Dataset/data.yaml"))
        self.assertEqual(cfg.train_kwargs()["data"], str(Path("Dataset/data.yaml")))

    def test_the_quality_gate_is_opt_in(self) -> None:
        # No threshold by default: behaviour for every existing caller (the Qt
        # dialog included) must stay exactly what it was.
        self.assertEqual(TrainConfig().min_metrics, ())


class PathResolutionTests(unittest.TestCase):
    def test_weights_path_resolves_from_the_package_root_by_default(self) -> None:
        # preflight, the backend and the provenance digest must agree on where
        # checkpoints live, so all three resolve the path through TrainConfig.
        cfg = TrainConfig()
        for key, spec in MODEL_SPECS.items():
            with self.subTest(key=key):
                self.assertEqual(cfg.weights_path(spec), PACKAGE_DIR.parent / spec.ckpt)
                self.assertTrue(str(cfg.weights_path(spec)).endswith(spec.ckpt))

    def test_weights_dir_override_moves_the_checkpoint_lookup(self) -> None:
        # Inputs move like outputs: a caller that redirects models_dir/runs_dir
        # can redirect the checkpoints the run starts from too.
        spec = MODEL_SPECS["det"]
        cfg = TrainConfig(weights_dir=Path("C:/elsewhere/weights"))
        self.assertEqual(cfg.weights_path(spec), Path("C:/elsewhere/weights") / spec.ckpt)

    def test_output_path_honours_a_models_dir_override(self) -> None:
        spec = MODEL_SPECS["det"]
        cfg = TrainConfig(models_dir=Path("C:/elsewhere/Models"))
        self.assertEqual(cfg.output_path(spec), Path("C:/elsewhere/Models") / spec.output)

    def test_manifest_path_sits_beside_the_models(self) -> None:
        cfg = TrainConfig(models_dir=Path("C:/elsewhere/Models"))
        self.assertEqual(cfg.manifest_path, Path("C:/elsewhere/Models") / MANIFEST_NAME)


if __name__ == "__main__":
    unittest.main()
