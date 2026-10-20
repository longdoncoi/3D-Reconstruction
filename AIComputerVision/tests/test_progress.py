"""Unit tests for the progress protocol shared with the Qt dock widget."""

from __future__ import annotations

import unittest
from typing import cast

from cvtrain.progress import PHASES, PROGRESS_PREFIX, Phase, ProgressReporter, compute_percent


class ComputePercentTests(unittest.TestCase):
    def test_bounds(self) -> None:
        for phase in ("preflight", "dataset", "train", "export", "manifest"):
            for fraction in (0.0, 0.5, 1.0, 2.0, -1.0):
                value = compute_percent(0, 3, phase, fraction)
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 100)

    def test_preflight_and_manifest_bands(self) -> None:
        self.assertEqual(compute_percent(0, 1, "preflight", 0.0), 0)
        self.assertEqual(compute_percent(0, 1, "preflight", 1.0), 2)
        self.assertEqual(compute_percent(0, 1, "manifest", 0.0), 97)
        self.assertEqual(compute_percent(0, 1, "manifest", 1.0), 100)

    def test_the_dataset_band_sits_between_preflight_and_the_models(self) -> None:
        # The fingerprint reads the whole dataset before training starts; its
        # band is small but it is its own, so the bar can move during a phase
        # that used to look like a hang.
        self.assertEqual(compute_percent(0, 1, "dataset", 0.0), 2)
        self.assertEqual(compute_percent(0, 1, "dataset", 1.0), 8)
        self.assertEqual(compute_percent(0, 1, "dataset", 0.5), 5)
        # The models start where the dataset band ends.
        self.assertEqual(compute_percent(0, 1, "train", 0.0), 8)

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

    def test_a_skipped_model_occupies_the_slot_a_finished_export_would(self) -> None:
        # The bar must stay monotonic when models are skipped, so the marker
        # lands exactly where ``exported`` would have landed.
        reporter = ProgressReporter(("det", "seg"))
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            reporter.preflight(1.0)
            reporter.skipped(0, "det")

        self.assertIn("stage=skip", captured.output[-1])
        self.assertIn("key=det", captured.output[-1])
        self.assertEqual(reporter.last_percent, compute_percent(0, 2, "export", 1.0))

    def test_an_out_of_order_marker_holds_instead_of_rewinding(self) -> None:
        # The dock's bar has no "rewind": a regression is held with a warning
        # rather than published. The diagnostic carries no ``pct=`` token, or
        # the very regex the dock parses would move the bar backwards.
        reporter = ProgressReporter(("det", "seg"))
        with self.assertLogs("cvtrain.progress", level="INFO"):
            reporter.preflight(1.0)
            reporter.train_epoch(0, 4, 5, "det")
        held = reporter.last_percent
        self.assertGreater(held, 2)
        with self.assertLogs("cvtrain.progress", level="WARNING") as captured:
            reporter.preflight(0.0)
        self.assertEqual(reporter.last_percent, held)
        self.assertTrue(any("held at" in line for line in captured.output))
        self.assertTrue(all("pct=" not in line for line in captured.output))

    def test_a_one_based_epoch_counter_degrades_gracefully(self) -> None:
        # The backend forwards the framework counter verbatim (0-based); a
        # 1-based counter must read as a finished bar, never as ``6/5``.
        reporter = ProgressReporter(("det",))
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            reporter.train_epoch(0, 5, 5, "det")
        self.assertIn("epoch=5/5", captured.output[-1])
        self.assertEqual(reporter.last_percent, compute_percent(0, 1, "train", 1.0))

    def test_the_dataset_band_reports_how_far_the_fingerprint_got(self) -> None:
        # Reading the whole dataset is minutes of work on a large one; the
        # operator must see the bar move during it.
        reporter = ProgressReporter(("det",))
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            reporter.preflight(1.0)
            reporter.dataset(1, 4)
            reporter.dataset(4, 4)
        dataset_lines = [line for line in captured.output if "stage=dataset" in line]
        self.assertEqual(len(dataset_lines), 2)
        self.assertIn("pct=4", dataset_lines[0])
        self.assertIn("pct=8", dataset_lines[-1])
        self.assertEqual(reporter.last_percent, compute_percent(0, 1, "dataset", 1.0))

    def test_the_dataset_band_does_not_flood_the_log(self) -> None:
        # The callback fires once per file; a dataset with a hundred thousand
        # files must not write a hundred thousand identical lines into the log
        # the Qt dock reads line by line. Only a change is published.
        reporter = ProgressReporter(("det",))
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            reporter.preflight(1.0)
            for done in range(1, 1001):
                reporter.dataset(done, 1000)
        dataset_lines = [line for line in captured.output if "stage=dataset" in line]
        self.assertLessEqual(len(dataset_lines), 8)
        self.assertTrue(dataset_lines)
        self.assertIn("pct=8", dataset_lines[-1])

    def test_an_empty_dataset_is_a_finished_band_not_a_stuck_one(self) -> None:
        reporter = ProgressReporter(("det",))
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            reporter.preflight(1.0)
            reporter.dataset(0, 0)
        self.assertIn("stage=dataset", captured.output[-1])
        self.assertEqual(reporter.last_percent, compute_percent(0, 1, "dataset", 1.0))


if __name__ == "__main__":
    unittest.main()
