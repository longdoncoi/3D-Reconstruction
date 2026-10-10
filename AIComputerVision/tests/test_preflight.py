"""Unit tests for dataset/environment preflight validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvtrain.config import TrainConfig
from cvtrain.preflight import (
    Check,
    PreflightReport,
    _dataset_base,
    _missing_dataset_paths,
    _module_available,
    _ref_exists,
    check_data_yaml,
    check_python,
    run,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _make_dataset(tmp: str) -> Path:
    """A realistic dataset tree: ``data.yaml`` plus the referenced image dirs.

    New behaviour: ``check_data_yaml`` fails fast when a referenced ``train``/
    ``val`` directory is missing, so every test that expects a *valid* dataset
    must create the directories the YAML points at.
    """

    root = Path(tmp)
    for name in ("images/train", "images/val"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return _write(
        root / "data.yaml",
        "train: images/train\nval: images/val\nnames:\n  0: part\n",
    )


class DataYamlTests(unittest.TestCase):
    def test_missing_file_is_fatal(self) -> None:
        check = check_data_yaml(Path("does/not/exist.yaml"))
        self.assertFalse(check.ok)
        self.assertTrue(check.fatal)

    def test_empty_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "   \n")
            self.assertFalse(check_data_yaml(path).ok)

    def test_invalid_yaml_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "train: [unclosed\n")
            self.assertFalse(check_data_yaml(path).ok)

    def test_missing_keys_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "path: .\n")
            check = check_data_yaml(path)
            self.assertFalse(check.ok)
            self.assertIn("names", check.detail)

    def test_valid_dataset_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_dataset(tmp)
            check = check_data_yaml(path)
            self.assertTrue(check.ok, check.detail)
            self.assertIn("1 classes", check.detail)

    def test_top_level_must_be_a_mapping(self) -> None:
        # A bare list is valid YAML but not a dataset definition; without this
        # branch ultralytics would fail much later with a far worse message.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "- images/train\n- images/val\n")
            check = check_data_yaml(path)
            self.assertFalse(check.ok)
            self.assertIn("must be a mapping", check.detail)

    def test_unreadable_dataset_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # A directory where the file should be makes read_text raise OSError.
            path = Path(tmp) / "data.yaml"
            path.mkdir()
            check = check_data_yaml(path)
            self.assertFalse(check.ok)
            self.assertIn("unreadable", check.detail)

    def test_dataset_content_is_skipped_when_pyyaml_is_missing(self) -> None:
        # Offline/minimal images may lack PyYAML: report the file's presence and
        # say explicitly that the content was not validated, rather than fail.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "train: images/train\n")
            with mock.patch("cvtrain.preflight._module_available", return_value=False):
                check = check_data_yaml(path)
            self.assertTrue(check.ok, check.detail)
            self.assertIn("PyYAML missing, content not validated", check.detail)


class DatasetPathTests(unittest.TestCase):
    """The referenced ``train``/``val`` directories exist on disk."""

    def test_missing_train_or_val_directory_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Only the file, none of the image directories it references.
            path = _write(
                Path(tmp) / "data.yaml",
                "train: images/train\nval: images/val\nnames:\n  0: part\n",
            )
            check = check_data_yaml(path)
            self.assertFalse(check.ok)
            self.assertIn("missing path(s)", check.detail)
            self.assertIn("train='images/train'", check.detail)
            self.assertIn("val='images/val'", check.detail)

    def test_path_key_relocates_relative_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "images" / "train").mkdir(parents=True)
            (root / "data" / "images" / "val").mkdir(parents=True)
            path = _write(
                root / "data.yaml",
                "path: data\ntrain: images/train\nval: images/val\nnames:\n  0: part\n",
            )
            self.assertTrue(check_data_yaml(path).ok, check_data_yaml(path).detail)

    def test_path_key_may_be_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "images" / "train").mkdir(parents=True)
            (root / "data" / "images" / "val").mkdir(parents=True)
            path = _write(
                root / "data.yaml",
                f"path: {root / 'data'}\ntrain: images/train\nval: images/val\nnames:\n  0: part\n",
            )
            self.assertTrue(check_data_yaml(path).ok)

    def test_empty_path_key_falls_back_to_the_yaml_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_dataset(tmp)
            text = _write(path, "path: \n" + path.read_text(encoding="utf-8"))
            self.assertTrue(check_data_yaml(text).ok)

    def test_absolute_train_and_val_paths_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train").mkdir(parents=True)
            (root / "images" / "val").mkdir(parents=True)
            path = _write(
                root / "data.yaml",
                f"train: {root / 'images' / 'train'}\nval: {root / 'images' / 'val'}\nnames:\n  0: part\n",
            )
            self.assertTrue(check_data_yaml(path).ok, check_data_yaml(path).detail)

    def test_missing_absolute_reference_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train").mkdir(parents=True)
            missing = root / "images" / "nowhere"
            path = _write(
                root / "data.yaml",
                f"train: {root / 'images' / 'train'}\nval: {missing}\nnames:\n  0: part\n",
            )
            check = check_data_yaml(path)
            self.assertFalse(check.ok)
            self.assertIn(f"val='{missing}'", check.detail)

    def test_glob_references_validate_their_static_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "train").mkdir(parents=True)
            (root / "images" / "val").mkdir(parents=True)
            path = _write(
                root / "data.yaml",
                'train: "images/train/*.jpg"\nval: images/val\nnames:\n  0: part\n',
            )
            self.assertTrue(check_data_yaml(path).ok, check_data_yaml(path).detail)

    def test_glob_with_missing_prefix_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(
                Path(tmp) / "data.yaml",
                'train: "images/missing/*.jpg"\nval: images/val\nnames:\n  0: part\n',
            )
            check = check_data_yaml(path)
            self.assertFalse(check.ok)
            self.assertIn("train='images/missing/*.jpg'", check.detail)

    def test_bare_glob_checks_the_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "val").mkdir(parents=True)
            path = _write(
                root / "data.yaml",
                'train: "*.jpg"\nval: images/val\nnames:\n  0: part\n',
            )
            # No static prefix: the dataset root itself must exist (it does).
            self.assertTrue(check_data_yaml(path).ok, check_data_yaml(path).detail)

    def test_bare_glob_against_a_missing_root_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "val").mkdir(parents=True)
            path = _write(
                root / "data.yaml",
                'path: no-such-dir\ntrain: "*.jpg"\nval: images/val\nnames:\n  0: part\n',
            )
            check = check_data_yaml(path)
            self.assertFalse(check.ok)
            self.assertIn("missing path(s)", check.detail)

    def test_non_string_references_are_left_to_the_framework(self) -> None:
        # ``train`` can be a text file listing or a remote URL; preflight must
        # not reject a dataset it cannot fully understand.
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_dataset(tmp)
            text = _write(path, "train: [seed.txt, more.txt]\nval: images/val\nnames:\n  0: part\n")
            self.assertTrue(check_data_yaml(text).ok, check_data_yaml(text).detail)

    def test_ref_exists_missing_plain_path_falls_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "val").mkdir(parents=True)
            self.assertFalse(_ref_exists("images/train", root))

    def test_dataset_base_defaults_to_the_yaml_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(_dataset_base({"names": {"0": "part"}}, root), root)

    def test_missing_paths_skips_absent_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(_missing_dataset_paths({"train": "x"}, root), ["train='x' (resolved under " + str(root) + ")"])


class ReportTests(unittest.TestCase):
    def test_python_check_passes_on_supported_interpreter(self) -> None:
        self.assertTrue(check_python().ok)

    def test_module_lookup_never_raises_on_a_broken_sys_modules_entry(self) -> None:
        # find_spec raises ValueError/ImportError for half-initialised entries;
        # preflight must report "missing", not crash before printing a report.
        for failure in (ValueError("bad spec"), ImportError("no spec")):
            with self.subTest(failure=failure), mock.patch(
                "cvtrain.preflight.importlib.util.find_spec", side_effect=failure
            ):
                self.assertFalse(_module_available("anything"))

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


if __name__ == "__main__":
    unittest.main()
