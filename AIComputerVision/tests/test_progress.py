"""Unit tests for the progress protocol shared with the Qt dock widget."""

from __future__ import annotations

import unittest
from typing import cast

from cvtrain.logging_setup import PHASES, PROGRESS_PREFIX, Phase, ProgressReporter, compute_percent


class ComputePercentTests(unittest.TestCase):
    def test_bounds(self) -> None:
        for phase in ("preflight", "train", "export", "manifest"):
            for fraction in (0.0, 0.5, 1.0, 2.0, -1.0):
                value = compute_percent(0, 3, phase, fraction)
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 100)

    def test_preflight_and_manifest_bands(self) -> None:
        self.assertEqual(compute_percent(0, 1, "preflight", 0.0), 0)
        self.assertEqual(compute_percent(0, 1, "preflight", 1.0), 2)
        self.assertEqual(compute_percent(0, 1, "manifest", 0.0), 97)
        self.assertEqual(compute_percent(0, 1, "manifest", 1.0), 100)

    def test_stages_do_not_overlap(self) -> None:
        count = 3
        for index in range(count):
            train_end = compute_percent(index, count, "train", 1.0)
            export_end = compute_percent(index, count, "export", 1.0)
            self.assertLess(train_end, export_end)
            if index + 1 < count:
                self.assertLessEqual(export_end, compute_percent(index + 1, count, "train", 0.0))

    def test_progress_is_monotonic_within_a_stage(self) -> None:
        previous = -1
        for step in range(6):
            value = compute_percent(1, 2, "train", step / 5)
            self.assertGreaterEqual(value, previous)
            previous = value

    def test_single_model_never_reaches_manifest_band(self) -> None:
        self.assertLessEqual(compute_percent(0, 1, "export", 1.0), 97)

    def test_unknown_phase_is_rejected_instead_of_guessing(self) -> None:
        # A typo must fail loudly, otherwise a plausible-but-wrong percentage
        # would be published to the Qt progress bar. The cast bypasses the
        # Literal type because this test exercises the runtime guard.
        with self.assertRaises(ValueError) as ctx:
            compute_percent(0, 3, cast(Phase, "trian"))
        self.assertIn("unknown phase", str(ctx.exception))

    def test_all_declared_phases_are_accepted(self) -> None:
        for phase in PHASES:
            self.assertIsInstance(compute_percent(0, 2, phase), int)


class ProgressReporterTests(unittest.TestCase):
    def test_emits_parseable_markers(self) -> None:
        reporter = ProgressReporter(("det", "seg"))
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            reporter.preflight(1.0)
            reporter.train_epoch(0, 0, 5, "det")
            reporter.exporting(0, "det")
            reporter.exported(0, "det")
            reporter.manifest(1.0)
        text = "\n".join(captured.output)
        self.assertIn(PROGRESS_PREFIX, text)
        self.assertIn("pct=", text)
        self.assertIn("key=det", text)
        self.assertIn("epoch=1/5", text)
        self.assertIn("stage=manifest", text)

    def test_done_is_the_terminal_success_marker(self) -> None:
        reporter = ProgressReporter(("det",))
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            reporter.preflight(1.0)
            reporter.done()
        self.assertIn("pct=100 stage=done", captured.output[-1])
        self.assertEqual(reporter.last_percent, 100)

    def test_error_repeats_the_last_percentage_instead_of_regressing(self) -> None:
        reporter = ProgressReporter(("det", "seg"))
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            reporter.preflight(1.0)
            reporter.train_epoch(0, 0, 5, "det")
            held = reporter.last_percent
            reporter.error()
        self.assertGreater(held, 0)
        self.assertEqual(reporter.last_percent, held)
        self.assertIn(f"pct={held} stage=error", captured.output[-1])


if __name__ == "__main__":
    unittest.main()
