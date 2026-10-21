"""Guard-path coverage for platform path resolution.

``resolve_paths`` merges three operator signals — an explicit directory, the
``APP_DATA_DIR`` environment override and the on-disk default — plus a legacy
``AITraining`` migration for models and caches. Each source is pinned, as is
the ``PermissionError`` tolerance on the docs directory (a read-only project
root must not break startup) and the cross-drive fallback of
``safe_relpath``.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_assistant.config.paths import (
    attachment_roots,
    get_paths,
    resolve_paths,
    safe_relpath,
)


class AttachmentRootTests(unittest.TestCase):
    """Where the desktop client stores the attachments it seals (ADR 0035).

    The AI service has to open the file the client just stored, so it needs
    the same directory — and it may not widen that to the whole disk: the
    HTTP chat route opens a *client-named* path.
    """

    def test_the_source_checkout_layout_is_covered(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APP_DATA_DIR", None)
            roots = attachment_roots()
        self.assertIn(str(get_paths().project_root / "Upload"), roots)

    def test_the_installed_layout_is_covered(self):
        """The production install puts the data root under the per-user
        application data directory, under the product folder."""
        with tempfile.TemporaryDirectory() as env_root:
            with patch.dict(os.environ, {"APP_DATA_DIR": env_root}):
                roots = attachment_roots()
        self.assertIn(
            str(Path(env_root).resolve() / "3D-Reconstruction" / "Upload"), roots,
        )


class DataDirSourceTests(unittest.TestCase):
    def test_an_explicit_data_dir_wins_over_everything(self):
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as data:
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("APP_DATA_DIR", None)
                paths = resolve_paths(Path(base) / "AIAssistant", explicit_data_dir=data)
            self.assertEqual(paths.data_dir, Path(data).resolve())

    def test_the_environment_override_scopes_data_under_its_name(self):
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as env_root:
            with patch.dict(os.environ, {"APP_DATA_DIR": env_root}):
                paths = resolve_paths(Path(base) / "AIAssistant")
            self.assertEqual(paths.data_dir, Path(env_root).resolve() / "AIAssistant")

    def test_the_default_data_dir_is_the_project_root(self):
        with tempfile.TemporaryDirectory() as project:
            paths = resolve_paths(Path(project) / "AIAssistant")
            self.assertEqual(paths.data_dir, Path(project).resolve())
            self.assertEqual(paths.project_root, Path(project).resolve())

    def test_a_legacy_training_cache_is_reused_until_it_migrates(self):
        """Models/Cache fall back to ``AITraining`` when the new names are absent."""
        with tempfile.TemporaryDirectory() as project:
            legacy_models = Path(project) / "AITraining" / "Models"
            legacy_models.mkdir(parents=True)
            paths = resolve_paths(Path(project) / "AIAssistant")
            self.assertEqual(paths.models_dir, legacy_models.resolve())
            # ...but an existing new directory always wins.
            fresh_models = Path(project) / "Models"
            fresh_models.mkdir()
            self.assertEqual(resolve_paths(Path(project) / "AIAssistant").models_dir,
                             fresh_models.resolve())


class DocsToleranceTests(unittest.TestCase):
    def test_an_unwritable_docs_directory_does_not_break_resolution(self):
        real_mkdir = Path.mkdir

        def _mkdir(self, *args, **kwargs):
            if self.name == "Docs":
                msg = "read-only project root"
                raise PermissionError(msg)
            return real_mkdir(self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as project:
            with patch.object(Path, "mkdir", _mkdir):
                paths = resolve_paths(Path(project) / "AIAssistant")
            self.assertEqual(paths.docs_dirs[0].name, "Docs")


class SafeRelpathTests(unittest.TestCase):
    def test_a_normal_path_is_relativised(self):
        self.assertEqual(safe_relpath("/a/b/c", "/a"), os.path.join("b", "c"))

    def test_an_impossible_relpath_falls_back_to_absolute(self):
        """Windows cross-drive paths raise ``ValueError``; the caller still
        gets a usable path instead of an exception."""
        with patch("os.path.relpath", side_effect=ValueError("different drives")):
            self.assertEqual(safe_relpath("C:/a/b", "D:/c"), os.path.abspath("C:/a/b"))


class DefaultPathsTests(unittest.TestCase):
    def test_get_paths_resolves_from_the_package_location(self):
        paths = get_paths()
        self.assertTrue(paths.base_dir.is_dir())
        self.assertTrue(paths.logs_dir.is_absolute())


if __name__ == "__main__":
    unittest.main()
