"""Unit tests for the shared filesystem primitives."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cvtrain.fsio import atomic_copy, atomic_write_text, sha256_file


def _leftovers(directory: Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]


class Sha256Tests(unittest.TestCase):
    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.bin"
            path.write_bytes(b"")
            self.assertEqual(sha256_file(path), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

    def test_content_change_changes_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            path.write_bytes(b"a")
            first = sha256_file(path)
            path.write_bytes(b"b")
            self.assertNotEqual(sha256_file(path), first)


class AtomicWriteTextTests(unittest.TestCase):
    def test_creates_missing_parents_and_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a" / "b" / "manifest.json"
            atomic_write_text(target, '{"ok": true}\n')
            self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": true}\n')
            self.assertEqual(_leftovers(target.parent), [])

    def test_overwrites_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            target.write_text("old", encoding="utf-8")
            atomic_write_text(target, "new")
            self.assertEqual(target.read_text(encoding="utf-8"), "new")

    def test_forces_lf_newlines(self) -> None:
        # The manifest is diffed by Git; CRLF would churn every Windows clone.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "f.txt"
            atomic_write_text(target, "one\ntwo\n")
            self.assertEqual(target.read_bytes(), b"one\ntwo\n")

    def test_failed_publish_cleans_up_and_keeps_the_old_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "f.txt"
            target.write_bytes(b"previous")

            with mock.patch("cvtrain.fsio.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_write_text(target, "new")

            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(_leftovers(root), [])


class AtomicCopyTests(unittest.TestCase):
    def test_copies_bytes_and_replaces_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "generated.onnx"
            dst = root / "Models" / "yolo11n.onnx"
            src.write_bytes(b"model-v1")
            atomic_copy(src, dst)
            self.assertEqual(dst.read_bytes(), b"model-v1")

            src.write_bytes(b"model-v2-longer")
            atomic_copy(src, dst)
            self.assertEqual(dst.read_bytes(), b"model-v2-longer")
            self.assertEqual(_leftovers(dst.parent), [])

    def test_failed_copy_leaves_previous_artifact_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dst = root / "out.onnx"
            dst.write_bytes(b"previous-good-artifact")

            missing = root / "does-not-exist.onnx"
            with self.assertRaises(OSError):
                atomic_copy(missing, dst)

            # The publish target must never be truncated by a failed attempt.
            self.assertEqual(dst.read_bytes(), b"previous-good-artifact")
            self.assertEqual(_leftovers(root), [])


if __name__ == "__main__":
    unittest.main()
