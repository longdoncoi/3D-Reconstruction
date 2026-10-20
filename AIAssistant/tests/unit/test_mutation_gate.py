"""Tests for the mutation gate itself (ADR 0032).

The gate is what proves that the load-bearing decisions of the package are
pinned by a test that *fails* when the decision is inverted. Two properties
make it trustworthy:

1. Every mutant is anchored to a source anchor that must still exist — a
   mutant whose anchor drifted is *stale*, not *passed*.
2. The number of mutants is itself a gate. Without that floor, deleting a
   mutant is indistinguishable from passing it: the suite stays green, the
   coverage floor stays at 100 %, and the guard silently stops being guarded.

The script lives outside ``src/``, so it is loaded by path.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mutation_check.py"
_ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location("ai_assistant_mutation_check", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # The dataclass decorator resolves the module's namespace through
    # ``sys.modules``, so the module has to be registered before it runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MutationGateFloorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_script()

    def test_the_declared_count_meets_the_floor(self) -> None:
        self.assertGreaterEqual(len(self.script.MUTANTS), self.script.MIN_MUTANTS)

    def test_the_floor_names_the_guard_that_disappeared(self) -> None:
        message = self.script.floor_violation(self.script.MIN_MUTANTS - 1)
        self.assertIsNotNone(message)
        self.assertIn(str(self.script.MIN_MUTANTS), str(message))
        self.assertIn(str(self.script.MIN_MUTANTS - 1), str(message))

    def test_a_count_at_or_above_the_floor_passes(self) -> None:
        self.assertIsNone(self.script.floor_violation(self.script.MIN_MUTANTS))
        self.assertIsNone(self.script.floor_violation(self.script.MIN_MUTANTS + 1))

    def test_every_mutant_anchors_to_a_line_that_still_exists(self) -> None:
        """A stale anchor means the guard moved and the mutant no longer
        proves anything — the gate reports it as stale rather than passed."""
        for mutant in self.script.MUTANTS:
            source = _ROOT / mutant.file
            self.assertTrue(source.is_file(), f"{mutant.file} must exist")
            text = source.read_text(encoding="utf-8")
            self.assertIn(
                mutant.original,
                text,
                f"{mutant.file}: the anchor of {mutant.label!r} drifted",
            )

    def test_every_mutant_declares_the_suite_that_must_fail(self) -> None:
        for mutant in self.script.MUTANTS:
            self.assertTrue(mutant.tests, f"{mutant.label!r} declares no test")
            for suite in mutant.tests:
                self.assertTrue(
                    suite.startswith("tests."), f"{suite!r} is not a test module path"
                )

    def test_the_labels_are_unique(self) -> None:
        labels = [mutant.label for mutant in self.script.MUTANTS]
        self.assertEqual(len(labels), len(set(labels)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
