"""The kernel that owns path confinement (ADR 0035).

Three rules used to live in three layers and each was subtly wrong: a
blocked directory was compared case-*sensitively* while the secret file
names next to it were compared case-insensitively; a path was compared by its
*spelling* while the filesystem decides; and each layer had its own copy, so
fixing one left the others behind.

The tests here are deliberately **property-shaped**: they walk the rule sets
themselves, so adding a name to a set automatically requires the
case-insensitive behaviour — a drift-proof guard rather than a list of
examples that can fall out of date.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from ai_assistant.path_policy import (
    BLOCKED_DIR_NAMES,
    SECRET_FILE_NAMES,
    canonical_path,
    is_blocked_dir,
    is_secret_name,
    resolve_within,
)


class BlockedDirectoryRuleTests(unittest.TestCase):
    def test_the_rule_is_case_insensitive(self) -> None:
        """A Windows filesystem is case-insensitive: ``.GIT`` **is** ``.git``.

        The case-sensitive form is why ``read_file(".GIT/config")`` returned
        the repository's git configuration — the remote URL and its
        credential helper included — while ``.git/config`` was refused.
        """
        for name in BLOCKED_DIR_NAMES:
            with self.subTest(name=name):
                self.assertTrue(is_blocked_dir(name))
                self.assertTrue(is_blocked_dir(name.upper()))
                self.assertTrue(is_blocked_dir(name.capitalize()))

    def test_an_ordinary_directory_is_not_blocked(self) -> None:
        for name in ("src", "docs", "tests", "models"):
            with self.subTest(name=name):
                self.assertFalse(is_blocked_dir(name))
                self.assertFalse(is_blocked_dir(name.upper()))


class SecretFileNameRuleTests(unittest.TestCase):
    def test_the_rule_is_case_insensitive(self) -> None:
        for name in SECRET_FILE_NAMES:
            with self.subTest(name=name):
                self.assertTrue(is_secret_name(name))
                self.assertTrue(is_secret_name(name.upper()))

    def test_the_env_family_is_covered(self) -> None:
        for name in (".env", ".env.local", ".env.production", ".env.development"):
            with self.subTest(name=name):
                self.assertTrue(is_secret_name(name))
                self.assertTrue(is_secret_name(name.upper()))

    def test_an_ordinary_file_is_not_a_secret(self) -> None:
        for name in ("main.py", "README.md", "environment.toml"):
            with self.subTest(name=name):
                self.assertFalse(is_secret_name(name))
                self.assertFalse(is_secret_name(name.upper()))


class ResolveWithinTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")

    def test_a_path_inside_the_root_is_kept(self) -> None:
        resolved = resolve_within(str(self.root), "src/main.py")
        self.assertIsNotNone(resolved)
        self.assertTrue(os.path.isabs(str(resolved)))
        self.assertTrue(str(resolved).endswith(os.path.join("src", "main.py")))

    def test_an_absolute_path_inside_the_root_is_kept(self) -> None:
        resolved = resolve_within(str(self.root), str(self.root / "src" / "main.py"))
        self.assertIsNotNone(resolved)

    def test_a_path_outside_the_root_is_refused(self) -> None:
        outside = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        self.assertIsNone(resolve_within(str(self.root), os.path.join(outside, "secret.py")))
        self.assertIsNone(resolve_within(str(self.root), str(Path(outside) / "secret.py")))

    def test_a_sibling_sharing_the_name_prefix_is_refused(self) -> None:
        """``<root>-evil`` shares the root as a string prefix: only a real
        containment check refuses it."""
        self.assertIsNone(resolve_within(str(self.root), f"{self.root}-evil/x.cpp"))

    def test_a_symlink_out_of_the_root_is_refused(self) -> None:
        """The *real* target decides, not the spelling.

        A junction inside the root that points elsewhere is an escape, and a
        lexical comparison cannot see it.
        """
        outside = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        (Path(outside) / "secret.py").write_text("SECRET = 1\n", encoding="utf-8")
        link = self.root / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available on this filesystem")
        self.assertIsNone(resolve_within(str(self.root), "escape/secret.py"))

    def test_an_empty_candidate_is_refused(self) -> None:
        self.assertIsNone(resolve_within(str(self.root), ""))

    def test_a_comparison_that_cannot_be_made_is_a_refusal(self) -> None:
        """Different Windows drives share no root: that is a refusal, never
        a crash."""
        from unittest import mock

        with mock.patch(
            "ai_assistant.path_policy.Path.relative_to",
            side_effect=ValueError("mix of drives"),
        ):
            self.assertIsNone(resolve_within(str(self.root), "src/main.py"))


class CanonicalPathTests(unittest.TestCase):
    def test_the_parent_is_canonicalised_and_the_leaf_re_appended(self) -> None:
        """One spelling for both sides of a comparison.

        Windows hands out two spellings for one directory (the canonical long
        form and the 8.3 short form), and a link whose target does not exist
        keeps the spelling it was created with. Comparing those two strings
        made a path that never left the root read as an escape (ADR 0033).
        """
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "does-not-exist-yet.txt"
            canonical = canonical_path(target)
            self.assertTrue(str(canonical).startswith(str(Path(directory).resolve())))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
