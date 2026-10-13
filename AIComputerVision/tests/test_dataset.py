"""Unit tests for ``data.yaml`` validation and path resolution (:mod:`cvtrain.dataset`)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvtrain.dataset import check_data_yaml, dataset_base, missing_dataset_paths, ref_exists


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _make_dataset(tmp: str) -> Path:
    """A realistic dataset tree: ``data.yaml`` plus the referenced image dirs.

    ``check_data_yaml`` fails fast when a referenced ``train``/``val``
    directory is missing, so every test that expects a *valid* dataset must
    create the directories the YAML points at.
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

    def test_val_is_required_when_train_is_present(self) -> None:
        # README promises both splits are checked. Without this, a dataset that
        # has no validation split passes preflight and then either fails deep
        # inside the framework or trains without ever validating.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "images" / "train").mkdir(parents=True)
            path = _write(Path(tmp) / "data.yaml", "train: images/train\nnames:\n  0: part\n")
            check = check_data_yaml(path)
            self.assertFalse(check.ok)
            self.assertIn("missing required key(s): val", check.detail)

    def test_train_is_required_when_val_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "images" / "val").mkdir(parents=True)
            path = _write(Path(tmp) / "data.yaml", "val: images/val\nnames:\n  0: part\n")
            check = check_data_yaml(path)
            self.assertFalse(check.ok)
            self.assertIn("missing required key(s): train", check.detail)

    def test_valid_dataset_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_dataset(tmp)
            check = check_data_yaml(path)
            self.assertTrue(check.ok, check.detail)
            self.assertIn("1 classes", check.detail)

    def test_empty_names_fail(self) -> None:
        # A dataset with no classes trains nothing: fail here, not in the framework.
        for payload in (
            "train: images/train\nval: images/val\nnames: []\n",
            "train: images/train\nval: images/val\nnames: {}\n",
        ):
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    for name in ("images/train", "images/val"):
                        (root / name).mkdir(parents=True, exist_ok=True)
                    path = _write(root / "data.yaml", payload)
                    check = check_data_yaml(path)
                    self.assertFalse(check.ok)
                    self.assertIn("at least one class", check.detail)

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

    def test_dataset_content_is_skipped_when_pyyyaml_is_missing(self) -> None:
        # Offline/minimal images may lack PyYAML: report the file's presence and
        # say explicitly that the content was not validated, rather than fail.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "train: images/train\n")
            with mock.patch("cvtrain.dataset.module_available", return_value=False):
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
            self.assertTrue(check_data_yaml(text).ok)

    def test_ref_exists_missing_plain_path_falls_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images" / "val").mkdir(parents=True)
            self.assertFalse(ref_exists("images/train", root))

    def test_dataset_base_defaults_to_the_yaml_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(dataset_base({"names": {"0": "part"}}, root), root)

    def test_missing_paths_skips_absent_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(
                missing_dataset_paths({"train": "x"}, root),
                ["train='x' (resolved under " + str(root) + ")"],
            )


if __name__ == "__main__":
    unittest.main()
