"""Unit tests for ``data.yaml`` validation and path resolution (:mod:`cvtrain.dataset`)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from cvtrain.dataset import (
    check_data_yaml,
    dataset_base,
    dataset_class_count,
    dataset_content_fingerprint,
    missing_dataset_paths,
    ref_exists,
)


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

    def test_a_scalar_names_value_fails_instead_of_raising_typeerror(self) -> None:
        # ``names: part`` and ``names: 42`` used to reach ``len()``: a check
        # whose job is to report malformed input crashed with a traceback.
        for value in ("part", "42"):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    for name in ("images/train", "images/val"):
                        (root / name).mkdir(parents=True, exist_ok=True)
                    path = _write(
                        root / "data.yaml",
                        f"train: images/train\nval: images/val\nnames: {value}\n",
                    )
                    check = check_data_yaml(path)
                    self.assertFalse(check.ok)
                    self.assertIn("mapping or list", check.detail)

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

    def test_a_remote_reference_is_left_to_the_framework_as_promised(self) -> None:
        # Ultralytics downloads these itself: there is no local path to check,
        # and reporting one as "missing" would reject a shape the framework
        # supports — and this module's docstring promises to leave alone.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _write(
                root / "data.yaml",
                "train: https://example.com/train.zip\nval: https://example.com/val.zip\nnames:\n  0: part\n",
            )
            self.assertTrue(check_data_yaml(path).ok)
            self.assertEqual(missing_dataset_paths({"train": "https://example.com/train.zip"}, root), [])

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


class DatasetClassCountTests(unittest.TestCase):
    """The ONNX contract reads the class count from the same YAML the trainer uses."""

    def test_missing_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(dataset_class_count(Path(tmp) / "absent.yaml"))

    def test_unreadable_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.yaml"
            path.mkdir()  # a directory where a file is expected
            self.assertIsNone(dataset_class_count(path))

    def test_pyyaml_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "names:\n  0: part\n")
            with mock.patch("cvtrain.dataset.module_available", return_value=False):
                self.assertIsNone(dataset_class_count(path))

    def test_invalid_yaml_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "names: [unclosed\n")
            self.assertIsNone(dataset_class_count(path))

    def test_a_non_mapping_document_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "- one\n- two\n")
            self.assertIsNone(dataset_class_count(path))

    def test_mapping_names_return_their_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "names:\n  0: part\n  1: widget\n")
            self.assertEqual(dataset_class_count(path), 2)

    def test_sequence_names_return_their_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "names: [part, widget, screw]\n")
            self.assertEqual(dataset_class_count(path), 3)

    def test_a_non_name_value_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "names: part\n")
            self.assertIsNone(dataset_class_count(path))


class DatasetContentFingerprintTests(unittest.TestCase):
    def test_missing_or_empty_path_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(dataset_content_fingerprint(root / "nonexistent.yaml"))
            empty = _write(root / "empty.yaml", "   \n")
            self.assertIsNone(dataset_content_fingerprint(empty))
            self.assertIsNone(dataset_content_fingerprint(root))

    def test_pyyaml_missing_falls_back_to_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(Path(tmp) / "data.yaml", "train: a\nval: b\nnames:\n  0: x\n")
            with mock.patch("cvtrain.dataset.module_available", return_value=False):
                fp = dataset_content_fingerprint(path)
                self.assertIsNotNone(fp)
                self.assertEqual(len(fp or ""), 64)

    def test_yaml_error_or_non_mapping_falls_back_to_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_syntax = _write(root / "bad.yaml", ": invalid: yaml: [")
            fp1 = dataset_content_fingerprint(bad_syntax)
            self.assertIsNotNone(fp1)

            non_mapping = _write(root / "scalar.yaml", "- item1\n- item2\n")
            fp2 = dataset_content_fingerprint(non_mapping)
            self.assertIsNotNone(fp2)

    def test_directory_content_changes_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            val_dir = root / "images" / "val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            _write(train_dir / "img1.jpg", "content1")
            _write(val_dir / "img2.jpg", "content2")

            yaml_path = _write(root / "data.yaml", "train: images/train\nval: images/val\nnames:\n  0: part\n")
            fp1 = dataset_content_fingerprint(yaml_path)
            self.assertIsNotNone(fp1)

            # Adding a file changes fingerprint
            _write(train_dir / "img3.jpg", "content3")
            fp2 = dataset_content_fingerprint(yaml_path)
            self.assertNotEqual(fp1, fp2)

            # Modifying file size changes fingerprint
            _write(train_dir / "img3.jpg", "content3-longer-data")
            fp3 = dataset_content_fingerprint(yaml_path)
            self.assertNotEqual(fp2, fp3)

    def test_single_file_splits_and_remote_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_file = _write(root / "train.txt", "line1\nline2")
            _write(root / "val.txt", "line3")

            yaml_path = _write(
                root / "data.yaml",
                f"train: {train_file.name}\nval: https://example.com/data.zip\nnames:\n  0: part\n",
            )
            fp = dataset_content_fingerprint(yaml_path)
            self.assertIsNotNone(fp)

    def test_glob_splits_fingerprint_every_matching_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            val_dir = root / "images" / "val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            _write(train_dir / "a.jpg", "pixels-a")
            _write(train_dir / "notes.txt", "not an image")

            yaml_path = _write(
                root / "data.yaml",
                'train: "images/train/*.jpg"\nval: images/val\nnames:\n  0: part\n',
            )
            first = dataset_content_fingerprint(yaml_path)

            _write(train_dir / "a.jpg", "different-pixels")
            second = dataset_content_fingerprint(yaml_path)
            self.assertNotEqual(first, second)

            # A non-matching file in the same directory is not part of the split.
            _write(train_dir / "notes.txt", "still not an image")
            third = dataset_content_fingerprint(yaml_path)
            self.assertEqual(second, third)

            _write(train_dir / "b.jpg", "new-image")
            fourth = dataset_content_fingerprint(yaml_path)
            self.assertNotEqual(third, fourth)

    def test_list_valued_splits_fingerprint_every_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            val_dir = root / "images" / "val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            _write(train_dir / "a.jpg", "pixels-a")
            _write(val_dir / "b.jpg", "pixels-b")

            yaml_path = _write(
                root / "data.yaml",
                "train: [images/train, images/val]\nval: images/val\nnames:\n  0: part\n",
            )
            first = dataset_content_fingerprint(yaml_path)
            _write(val_dir / "b.jpg", "different-pixels")
            second = dataset_content_fingerprint(yaml_path)
            self.assertNotEqual(first, second)

    def test_txt_listing_fingerprints_the_files_it_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            val_dir = root / "images" / "val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            image = _write(train_dir / "a.jpg", "pixels-a")
            listing = _write(root / "train.txt", "# one image\nimages/train/a.jpg\n")

            yaml_path = _write(
                root / "data.yaml",
                "train: train.txt\nval: images/val\nnames:\n  0: part\n",
            )
            first = dataset_content_fingerprint(yaml_path)

            # The image itself, not just the listing, is part of the proof.
            image.write_bytes(b"different-pixels")
            second = dataset_content_fingerprint(yaml_path)
            self.assertNotEqual(first, second)

            listing.write_text("# one image\nimages/train/a.jpg\nimages/train/b.jpg\n", encoding="utf-8")
            third = dataset_content_fingerprint(yaml_path)
            self.assertNotEqual(second, third)

    def test_a_listing_in_a_subdirectory_handles_absolute_and_mixed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            val_dir = root / "images" / "val"
            lists_dir = root / "lists"
            external_dir = root.parent / f"{root.name}-external"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            lists_dir.mkdir()
            external_dir.mkdir()
            self.addCleanup(shutil.rmtree, external_dir, ignore_errors=True)
            _write(train_dir / "a.jpg", "pixels-a")
            external = _write(external_dir / "external.jpg", "outside-pixels")
            listing = _write(
                lists_dir / "train.txt",
                "https://example.com/remote.jpg\n"
                "\n"
                "# comment\n"
                "images/train/a.jpg\n"
                f"{external.as_posix()}\n",
            )

            yaml_path = _write(
                root / "data.yaml",
                f"train: lists/{listing.name}\nval: images/val\nnames:\n  0: part\n",
            )
            first = dataset_content_fingerprint(yaml_path)
            _write(external_dir / "external.jpg", "changed-outside-pixels")
            second = dataset_content_fingerprint(yaml_path)
            self.assertNotEqual(first, second)

    def test_an_unreadable_listing_contributes_its_own_bytes_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            val_dir = root / "images" / "val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            listing = _write(root / "train.txt", "images/train/a.jpg\n")
            yaml_path = _write(
                root / "data.yaml",
                f"train: {listing.name}\nval: images/val\nnames:\n  0: part\n",
            )
            real_read_text = Path.read_text

            def _read_text(path: Path, *args: Any, **kwargs: Any) -> str:
                if path.name == listing.name:
                    raise OSError("sharing violation")
                return real_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", _read_text):
                fingerprint = dataset_content_fingerprint(yaml_path)
            self.assertIsNotNone(fingerprint)

    def test_list_splits_ignore_remote_and_non_string_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            val_dir = root / "images" / "val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            _write(train_dir / "a.jpg", "pixels")
            yaml_path = _write(
                root / "data.yaml",
                "train: [https://example.com/data.zip, 42, images/train]\n"
                "val: images/val\nnames:\n  0: part\n",
            )
            self.assertIsNotNone(dataset_content_fingerprint(yaml_path))

    def test_a_missing_split_path_is_part_of_the_fingerprint(self) -> None:
        # "The directory is not there yet" is not the same dataset as "the
        # directory is there and empty": creating a previously missing split
        # must move the digest, or a deleted dataset could look unchanged.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = _write(
                root / "data.yaml",
                "train: images/train\nval: images/val\nnames:\n  0: part\n",
            )
            missing = dataset_content_fingerprint(yaml_path)

            train_dir = root / "images" / "train"
            val_dir = root / "images" / "val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            _write(train_dir / "a.jpg", "pixels")

            present = dataset_content_fingerprint(yaml_path)
            self.assertIsNotNone(missing)
            self.assertIsNotNone(present)
            self.assertNotEqual(missing, present)

    def test_subdirectories_inside_split_are_ignored(self) -> None:
        """rglob('*') yields subdirectories too; only files should be hashed."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            nested = train_dir / "subdir"
            nested.mkdir(parents=True)
            _write(train_dir / "img.jpg", "data")
            _write(nested / "deep.jpg", "deep-data")

            yaml_path = _write(root / "data.yaml", "train: images/train\nval: images/train\nnames:\n  0: part\n")
            fp = dataset_content_fingerprint(yaml_path)
            self.assertIsNotNone(fp)

    def test_an_unreadable_file_contributes_a_stable_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            train_dir.mkdir(parents=True)
            _write(train_dir / "img.jpg", "data")

            yaml_path = _write(root / "data.yaml", "train: images/train\nval: images/train\nnames:\n  0: part\n")
            orig_open = Path.open

            def _flaky_open(self: Path, *a: Any, **kw: Any) -> Any:
                import sys
                if sys._getframe(1).f_code.co_name == "_digest_file":
                    raise OSError("disk error")
                return orig_open(self, *a, **kw)

            with mock.patch.object(Path, "open", _flaky_open):
                fp = dataset_content_fingerprint(yaml_path)
                self.assertIsNotNone(fp)
                self.assertEqual(len(fp or ""), 64)

    def test_an_unreadable_single_file_contributes_a_stable_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_file = _write(root / "train.txt", "line1\nline2")

            yaml_path = _write(
                root / "data.yaml",
                f"train: {train_file.name}\nval: {train_file.name}\nnames:\n  0: part\n",
            )
            orig_open = Path.open

            def _flaky_open(self: Path, *a: Any, **kw: Any) -> Any:
                import sys
                if sys._getframe(1).f_code.co_name == "_digest_file":
                    raise OSError("disk error")
                return orig_open(self, *a, **kw)

            with mock.patch.object(Path, "open", _flaky_open):
                fp = dataset_content_fingerprint(yaml_path)
                self.assertIsNotNone(fp)
                self.assertEqual(len(fp or ""), 64)

    def test_same_size_content_change_changes_the_fingerprint(self) -> None:
        # The inventory-only fingerprint this replaces compared (name, size),
        # so rewriting an image with different pixels of the same length was
        # invisible to --skip-unchanged. The content hash must catch it.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            train_dir.mkdir(parents=True)
            _write(train_dir / "img.jpg", "aaaa")
            yaml_path = _write(root / "data.yaml", "train: images/train\nval: images/train\nnames:\n  0: part\n")

            first = dataset_content_fingerprint(yaml_path)
            _write(train_dir / "img.jpg", "bbbb")  # same length, different bytes
            second = dataset_content_fingerprint(yaml_path)

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertNotEqual(first, second)

    def test_moving_a_file_between_splits_changes_the_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_dir = root / "images" / "train"
            val_dir = root / "images" / "val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)
            _write(train_dir / "img.jpg", "pixels")
            yaml_path = _write(root / "data.yaml", "train: images/train\nval: images/val\nnames:\n  0: part\n")

            first = dataset_content_fingerprint(yaml_path)
            (train_dir / "img.jpg").rename(val_dir / "img.jpg")
            second = dataset_content_fingerprint(yaml_path)

            self.assertNotEqual(first, second)

    def test_nonexistent_split_path_does_not_crash(self) -> None:
        """A split pointing to a path that is neither a dir nor a file."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = _write(root / "data.yaml", "train: nonexistent_dir\nval: also_missing\nnames:\n  0: part\n")
            fp = dataset_content_fingerprint(yaml_path)
            self.assertIsNotNone(fp)


if __name__ == "__main__":
    unittest.main()
