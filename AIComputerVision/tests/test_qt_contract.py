"""Cross-language contract tests: the Qt dock widget <-> ``TrainModel.py``.

``AITrainDockWidget`` builds the command line by hand, labels its checkboxes by
hand, and parses stdout with a hand-written regular expression. None of that is
visible to the Python test suite unless it is mirrored here — so a rename on
either side becomes a red test instead of a dialog that silently trains the
wrong models, or a progress bar that never moves.
"""

from __future__ import annotations

import logging
import re
import sys
import unittest
from pathlib import Path

from cvtrain.cli import build_parser, config_from_args
from cvtrain.config import ALL_MODEL_KEYS, MODEL_SPECS
from cvtrain.logging_setup import PROGRESS_PREFIX, ProgressReporter, setup_logging

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCK = REPO_ROOT / "src" / "modules" / "AIProcessorPlugin" / "AITrainDockWidget.cpp"
HEADER = REPO_ROOT / "src" / "app" / "AppConstants.h"
ENTRY_POINT = Path(__file__).resolve().parents[1] / "TrainModel.py"

_missing_dock = f"AITrainDockWidget.cpp not found at {DOCK}"


@unittest.skipUnless(DOCK.exists(), _missing_dock)
class LaunchContractTests(unittest.TestCase):
    """``startTraining`` assembles argv; the CLI must accept exactly that."""

    text: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = DOCK.read_text(encoding="utf-8")

    def _app_flags(self) -> list[str]:
        """Every ``--flag`` literal the desktop app appends to its argv."""

        flags: list[str] = []
        for line in self.text.splitlines():
            if "args <<" in line or "args<<" in line:
                flags.extend(re.findall(r'"(--[a-z][a-z-]*)"', line))
        return flags

    def test_the_app_actually_passes_flags(self) -> None:
        found = self._app_flags()
        self.assertTrue(found, 'no `args << "--flag"` lines found in AITrainDockWidget.cpp')

    def test_every_flag_the_app_passes_is_accepted_by_the_cli(self) -> None:
        for flag in self._app_flags():
            with self.subTest(flag=flag):
                # argparse exits with 2 on an unknown option, which the desktop
                # app would surface as a bogus "Mã lỗi: 2" dialog.
                try:
                    build_parser().parse_args([flag])
                except SystemExit:
                    self.fail(f"the Qt dialog passes {flag}, but the CLI no longer accepts it")

    def test_the_app_always_passes_yes_so_it_never_blocks_on_a_prompt(self) -> None:
        self.assertIn("--yes", self._app_flags())

    def test_the_full_selection_resolves_to_every_model(self) -> None:
        cfg = config_from_args(build_parser().parse_args(self._app_flags()))
        self.assertTrue(cfg.auto_confirm, "without --yes the app would hang on our Y/N prompt")
        self.assertEqual(set(cfg.models), set(ALL_MODEL_KEYS))

    def test_train_script_constant_points_at_the_existing_entry_point(self) -> None:
        if not HEADER.exists():  # pragma: no cover - checkout without C++ sources
            self.skipTest("AppConstants.h not found")
        match = re.search(r'trainScript\(\)\s*\{[^"]*"([^"]+)"', HEADER.read_text(encoding="utf-8"))
        self.assertIsNotNone(match, "trainScript() not found in AppConstants.h")
        assert match is not None
        self.assertEqual(match.group(1), ENTRY_POINT.name)
        self.assertTrue(ENTRY_POINT.exists(), f"{ENTRY_POINT} must exist, the app launches it by name")


@unittest.skipUnless(DOCK.exists(), _missing_dock)
class DialogContractTests(unittest.TestCase):
    """Checkbox labels in the dialog are the labels the CLI prints in --help."""

    def test_dialog_checkbox_labels_match_the_model_specs_in_order(self) -> None:
        text = DOCK.read_text(encoding="utf-8")
        labels = re.findall(r'new QCheckBox\("([^"]+)"', text)
        expected = [MODEL_SPECS[key].label for key in ALL_MODEL_KEYS]
        self.assertEqual(labels, expected)


@unittest.skipUnless(DOCK.exists(), _missing_dock)
class ProgressProtocolContractTests(unittest.TestCase):
    """The dock parses ``[PROGRESS]`` + ``pct=(\\d{1,3})`` out of stdout."""

    text: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = DOCK.read_text(encoding="utf-8")

    def _qt_percent_regex(self) -> re.Pattern[str]:
        """Rebuild the C++ ``pctRe`` pattern from the source it is written in."""

        match = re.search(r'QStringLiteral\("pct=\((.+?)\)"\)', self.text)
        self.assertIsNotNone(match, r"pct=(\d{1,3}) regex literal not found in AITrainDockWidget.cpp")
        assert match is not None
        # The file holds the C++-escaped form ``\\d{1,3}``; unescape once so the
        # very same pattern is applied here as the dock applies at runtime.
        return re.compile(r"pct=(" + match.group(1).replace("\\\\", "\\") + r")")

    def test_emitted_markers_are_parseable_by_the_qt_regex(self) -> None:
        percent_re = self._qt_percent_regex()
        reporter = ProgressReporter(("det", "seg"))
        with self.assertLogs("cvtrain.progress", level="INFO") as captured:
            reporter.preflight(1.0)
            reporter.train_epoch(0, 2, 5, "det")
            reporter.exporting(1, "seg")
            reporter.manifest(1.0)
            reporter.done()
            reporter.error()

        self.assertEqual(len(captured.output), 6)
        for line in captured.output:
            with self.subTest(line=line):
                self.assertIn(PROGRESS_PREFIX, line, "the dock skips any line without the marker")
                match = percent_re.search(line)
                self.assertIsNotNone(match, f"Qt pct regex does not match: {line}")
                assert match is not None
                # qBound(0, value, 100) cannot rescue a value the regex missed.
                self.assertGreaterEqual(int(match.group(1)), 0)
                self.assertLessEqual(int(match.group(1)), 100)

    def test_progress_is_written_to_stdout_which_is_all_qt_reads(self) -> None:
        logger = logging.getLogger("cvtrain")
        saved_handlers = list(logger.handlers)
        saved_propagate = logger.propagate
        for handler in saved_handlers:
            logger.removeHandler(handler)
        try:
            attached = setup_logging()
            handler = attached.handlers[0]
            if not isinstance(handler, logging.StreamHandler):
                self.fail(f"unexpected handler type: {type(handler).__name__}")
            self.assertIs(handler.stream, sys.stdout, "the dock only reads stdout")
        finally:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
            for handler in saved_handlers:
                logger.addHandler(handler)
            logger.propagate = saved_propagate


if __name__ == "__main__":
    unittest.main()
